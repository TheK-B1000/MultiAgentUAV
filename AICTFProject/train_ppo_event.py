# ==========================================================
# train_ppo_event.py (UPDATED: EPISODE-STRICT + PENDING ACCUMULATOR FIXED
#                     + OP3 STABILIZATION: SCORE-DELTA + OUTCOME TERMINAL
#                     + TERMINAL-SAFE UPDATES + OP3 ENTROPY RESET)
# ==========================================================
import os
import random
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import torch_directml
    HAS_TDML = True
except ImportError:
    torch_directml = None
    HAS_TDML = False

from game_field import GameField, CNN_COLS, CNN_ROWS, NUM_CNN_CHANNELS
from game_manager import GameManager
from macro_actions import MacroAction
from rl_policy import ActorCriticNet
from policies import OP1RedPolicy, OP2RedPolicy, OP3RedPolicy


# ==========================================================
# Device
# ==========================================================
def get_device() -> torch.device:
    if HAS_TDML:
        return torch_directml.device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


# ==========================================================
# Reproducibility
# ==========================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================================
# Hyperparams
# ==========================================================
GRID_ROWS = CNN_ROWS
GRID_COLS = CNN_COLS

TOTAL_STEPS: int = 1_000_000
UPDATE_EVERY: int = 4096          # transitions (not windows)
PPO_EPOCHS: int = 10
MINIBATCH_SIZE: int = 256

LR: float = 3e-4
CLIP_EPS: float = 0.2
VALUE_COEF: float = 2.0
MAX_GRAD_NORM: float = 0.5

GAMMA: float = 0.995
GAE_LAMBDA: float = 0.97

DECISION_WINDOW: float = 1.0
SIM_DT: float = 0.1

CHECKPOINT_DIR: str = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

USED_MACROS: List[MacroAction] = [
    MacroAction.GO_TO,
    MacroAction.GRAB_MINE,
    MacroAction.GET_FLAG,
    MacroAction.PLACE_MINE,
    MacroAction.GO_HOME,
]
N_MACROS: int = len(USED_MACROS)

PHASE_SEQUENCE: List[str] = ["OP1", "OP2", "OP3"]
MIN_PHASE_EPISODES: Dict[str, int] = {"OP1": 300, "OP2": 700, "OP3": 1500}
TARGET_PHASE_WINRATE: Dict[str, float] = {"OP1": 0.74, "OP2": 0.70, "OP3": 0.65}
PHASE_WINRATE_WINDOW: int = 50

ENT_COEF_BY_PHASE: Dict[str, float] = {"OP1": 0.06, "OP2": 0.05, "OP3": 0.04}
PHASE_CONFIG: Dict[str, Dict[str, float]] = {
    "OP1": dict(score_limit=3, max_time=200.0, max_macro_steps=500),
    "OP2": dict(score_limit=3, max_time=200.0, max_macro_steps=500),
    "OP3": dict(score_limit=3, max_time=200.0, max_macro_steps=500),
}

TIME_PENALTY_PER_AGENT_PER_MACRO = 0.001

# ==========================================================
# OP3 Stabilization: Score-delta + Outcome terminal reward
# ==========================================================
WIN_BONUS: float = 25.0
LOSS_PENALTY: float = -25.0
DRAW_PENALTY: float = -5.0

BLUE_SCORED_BONUS: float = 15.0     # per point blue gains
RED_SCORED_PENALTY: float = -15.0   # per point red gains (penalty to blue)

# Entropy bump on OP3 entry
OP3_ENT_RESET_VALUE: float = 0.08
OP3_ENT_RESET_EPISODES: int = 100   # first N OP3 episodes get higher entropy then decay


# ==========================================================
# Env helpers
# ==========================================================
def set_red_policy_for_phase(env: GameField, phase: str) -> None:
    if phase == "OP1":
        env.policies["red"] = OP1RedPolicy("red")
    elif phase == "OP2":
        env.policies["red"] = OP2RedPolicy("red")
    elif phase == "OP3":
        env.policies["red"] = OP3RedPolicy("red")
    else:
        raise ValueError(f"Unknown phase: {phase}")


def make_env() -> GameField:
    grid = [[0] * GRID_COLS for _ in range(GRID_ROWS)]
    env = GameField(grid)
    env.use_internal_policies = True

    if hasattr(env, "set_external_control"):
        env.set_external_control("blue", True)
        env.set_external_control("red", False)

    if hasattr(env, "external_missing_action_mode"):
        env.external_missing_action_mode = "idle"

    return env


def agent_uid(agent: Any) -> str:
    """
    Stable agent id for reward routing/logging.
    In your current Agent class, this is typically like 'blue_0_1344' (unique per spawn instance).
    """
    uid = getattr(agent, "unique_id", None) or getattr(agent, "slot_id", None)
    if uid is None:
        side = getattr(agent, "side", "blue")
        aid = getattr(agent, "agent_id", 0)
        uid = f"{side}_{aid}"
    return str(uid)


def external_key_for_agent(env: GameField, agent: Any) -> str:
    """
    Minimal, deterministic external key for submit_external_actions.
    Prefer env._external_key_for_agent if present, else agent_uid.
    """
    if hasattr(env, "_external_key_for_agent"):
        k = str(env._external_key_for_agent(agent))
        if k.strip():
            return k
    return agent_uid(agent)


def apply_blue_actions(env: GameField, actions_by_agent: Dict[str, Tuple[int, int]]) -> None:
    """
    Clean path: env.submit_external_actions(actions_dict) is expected.
    """
    fn = getattr(env, "submit_external_actions", None)
    if fn is None or (not callable(fn)):
        raise RuntimeError("GameField must implement submit_external_actions(actions_by_agent).")
    fn(actions_by_agent)


def compute_dt_for_window(window_s: float = DECISION_WINDOW) -> float:
    return float(window_s)


def sim_decision_window(env: GameField, gm: GameManager, window_s: float, sim_dt: float) -> None:
    if window_s <= 0.0:
        return
    if sim_dt <= 0.0:
        raise ValueError("SIM_DT must be > 0")

    n_full = int(window_s // sim_dt)
    rem = float(window_s - n_full * sim_dt)

    for _ in range(n_full):
        if gm.game_over:
            return
        env.update(sim_dt)

    if rem > 1e-9 and (not gm.game_over):
        env.update(rem)


# ==========================================================
# Macro mask (USED_MACROS index-space)
# ==========================================================
def _get_agent_xy(agent: Any) -> Tuple[float, float]:
    fp = getattr(agent, "float_pos", None)
    if isinstance(fp, (tuple, list)) and len(fp) >= 2:
        return float(fp[0]), float(fp[1])
    return float(getattr(agent, "x", 0.0)), float(getattr(agent, "y", 0.0))


def compute_used_macro_mask(env: GameField, agent: Any) -> np.ndarray:
    """
    Deterministic, research-safe masking aligned to USED_MACROS.
    If env.get_macro_mask(agent) exists, we use it (must match N_MACROS).
    """
    if hasattr(env, "get_macro_mask") and callable(getattr(env, "get_macro_mask")):
        mm = env.get_macro_mask(agent)
        mm = np.array(mm, dtype=np.bool_)
        if mm.shape == (N_MACROS,):
            if not mm.any():
                mm[:] = True
            return mm
        raise RuntimeError(f"env.get_macro_mask returned shape {mm.shape}, expected {(N_MACROS,)}")

    mask = np.ones((N_MACROS,), dtype=np.bool_)

    carrying = False
    if hasattr(agent, "isCarryingFlag") and callable(getattr(agent, "isCarryingFlag")):
        carrying = bool(agent.isCarryingFlag())
    else:
        carrying = bool(getattr(agent, "is_carrying_flag", False))

    # PLACE_MINE needs charges
    if getattr(agent, "mine_charges", 0) <= 0 and MacroAction.PLACE_MINE in USED_MACROS:
        mask[USED_MACROS.index(MacroAction.PLACE_MINE)] = False

    # GRAB_MINE requires friendly pickup nearby (soft gate)
    has_pickup_near = False
    ax, ay = _get_agent_xy(agent)
    for p in getattr(env, "mine_pickups", []):
        if getattr(p, "owner_side", None) == getattr(agent, "side", None):
            dx = float(getattr(p, "x", 0.0)) - ax
            dy = float(getattr(p, "y", 0.0)) - ay
            if (dx * dx + dy * dy) ** 0.5 < 3.0:
                has_pickup_near = True
                break
    if (not has_pickup_near) and MacroAction.GRAB_MINE in USED_MACROS:
        mask[USED_MACROS.index(MacroAction.GRAB_MINE)] = False

    # GET_FLAG invalid while carrying
    if carrying and MacroAction.GET_FLAG in USED_MACROS:
        mask[USED_MACROS.index(MacroAction.GET_FLAG)] = False

    if not mask.any():
        mask[:] = True

    return mask


# ==========================================================
# Rollout Buffer (fixed mask shape, no empty masks)
# ==========================================================
class RolloutBuffer:
    def __init__(self, n_macros: int) -> None:
        self.n_macros = int(n_macros)
        self.obs: List[np.ndarray] = []
        self.macro_actions: List[int] = []
        self.target_actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.next_values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.dts: List[float] = []
        self.macro_masks: List[np.ndarray] = []

    def add(
        self,
        obs: np.ndarray,
        macro_action: int,
        target_action: int,
        log_prob: float,
        value: float,
        next_value: float,
        reward: float,
        done: bool,
        dt: float,
        macro_mask: np.ndarray,
    ) -> None:
        mm = np.array(macro_mask, dtype=np.bool_)
        if mm.shape != (self.n_macros,):
            raise ValueError(f"macro_mask must be shape {(self.n_macros,)}, got {mm.shape}")

        self.obs.append(np.array(obs, dtype=np.float32))
        self.macro_actions.append(int(macro_action))
        self.target_actions.append(int(target_action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.next_values.append(float(next_value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.dts.append(float(dt))
        self.macro_masks.append(mm)

    def size(self) -> int:
        return len(self.obs)

    def clear(self) -> None:
        self.__init__(self.n_macros)

    def to_tensors(self, device: torch.device) -> Tuple[torch.Tensor, ...]:
        obs = torch.tensor(np.stack(self.obs), dtype=torch.float32, device=device)                 # [T,C,H,W]
        macro_actions = torch.tensor(self.macro_actions, dtype=torch.long, device=device)          # [T]
        target_actions = torch.tensor(self.target_actions, dtype=torch.long, device=device)        # [T]
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)               # [T]
        values = torch.tensor(self.values, dtype=torch.float32, device=device)                      # [T]
        next_values = torch.tensor(self.next_values, dtype=torch.float32, device=device)           # [T]
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)                    # [T]
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)                        # [T]
        dts = torch.tensor(self.dts, dtype=torch.float32, device=device)                            # [T]
        macro_masks = torch.tensor(np.stack(self.macro_masks), dtype=torch.bool, device=device)     # [T,A]
        return obs, macro_actions, target_actions, log_probs, values, next_values, rewards, dones, dts, macro_masks


# ==========================================================
# Event-driven GAE
# ==========================================================
def compute_gae_event(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    dts: torch.Tensor,
    gamma: float = GAMMA,
    lam: float = GAE_LAMBDA,
) -> Tuple[torch.Tensor, torch.Tensor]:
    T = rewards.size(0)
    advantages = torch.zeros(T, device=rewards.device)
    next_adv = 0.0

    for t in reversed(range(T)):
        gamma_dt = gamma ** dts[t]
        lam_gamma_dt = (gamma * lam) ** dts[t]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma_dt * next_values[t] * mask - values[t]
        advantages[t] = delta + lam_gamma_dt * next_adv * mask
        next_adv = advantages[t]

    returns = advantages + values
    return advantages, returns


def normalize_advantages(adv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = adv.mean()
    var = (adv - mean).pow(2).mean()
    return (adv - mean) / torch.sqrt(var + eps)


# ==========================================================
# PPO Update
# ==========================================================
def ppo_update(
    policy: ActorCriticNet,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    device: torch.device,
    ent_coef: float,
) -> None:
    (
        obs,
        macro_actions,
        target_actions,
        old_log_probs,
        values,
        next_values,
        rewards,
        dones,
        dts,
        macro_masks,
    ) = buffer.to_tensors(device)

    T = obs.size(0)
    if T == 0:
        buffer.clear()
        return

    advantages, returns = compute_gae_event(rewards, values, next_values, dones, dts)
    advantages = normalize_advantages(advantages)

    policy.train()
    for _ in range(PPO_EPOCHS):
        perm = torch.randperm(T, device=device)

        for start in range(0, T, MINIBATCH_SIZE):
            end = min(start + MINIBATCH_SIZE, T)
            mb_idx = perm[start:end]

            mb_obs      = obs.index_select(0, mb_idx)
            mb_macro    = macro_actions.index_select(0, mb_idx)
            mb_target   = target_actions.index_select(0, mb_idx)
            mb_old_logp = old_log_probs.index_select(0, mb_idx)
            mb_adv      = advantages.index_select(0, mb_idx)
            mb_ret      = returns.index_select(0, mb_idx)
            mb_mask     = macro_masks.index_select(0, mb_idx)

            new_logp, entropy, new_values = policy.evaluate_actions(
                mb_obs,
                mb_macro,
                mb_target,
                macro_mask_batch=mb_mask,
            )

            ratio = torch.exp(new_logp - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (mb_ret - new_values).pow(2).mean()
            loss = policy_loss + VALUE_COEF * value_loss - ent_coef * entropy.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

    buffer.clear()


# ==========================================================
# Reward collection (EPISODE-STRICT + PENDING ACCUMULATOR)
# ==========================================================
def pop_reward_events_strict(gm: GameManager) -> List[Tuple[float, str, float]]:
    fn = getattr(gm, "pop_reward_events", None)
    if fn is None or (not callable(fn)):
        raise RuntimeError("GameManager must implement pop_reward_events() returning (t, agent_id:str, r).")
    return fn()


def clear_reward_events_best_effort(gm: GameManager) -> None:
    """
    Defensive: clear any stale events on episode reset.
    We do not validate ids here, we just drain.
    """
    try:
        _ = pop_reward_events_strict(gm)
    except Exception:
        return


def init_episode_reward_routing(env: GameField) -> Tuple[set, Dict[str, float], Dict[str, int]]:
    """
    Returns:
      episode_uid_set: all blue agent uids that can legally receive reward this episode
      pending:         per-uid reward accumulator
      last_buf_idx:    per-uid last transition index stored in buffer (for flush)
    """
    uids: List[str] = []
    for a in getattr(env, "blue_agents", []):
        if a is None:
            continue
        uids.append(agent_uid(a))
    episode_uid_set = set(uids)
    pending = {uid: 0.0 for uid in episode_uid_set}
    last_buf_idx: Dict[str, int] = {}
    return episode_uid_set, pending, last_buf_idx


def accumulate_rewards_for_uid_set(
    gm: GameManager,
    allowed_uids: set,
    pending: Dict[str, float],
) -> None:
    """
    Pop ALL reward events, but only accumulate those that belong to allowed_uids.
    Everything else (red team, old episode ids, etc.) is ignored.

    Still STRICT about invariants:
      - agent_id must exist and be a non-empty string
    """
    events = pop_reward_events_strict(gm)
    for _t, agent_id, r in events:
        if agent_id is None:
            raise RuntimeError("Found reward event with agent_id=None (must always be per-agent).")
        key = str(agent_id).strip()
        if not key:
            raise RuntimeError("Found reward event with empty agent_id.")
        if key in allowed_uids:
            pending[key] = pending.get(key, 0.0) + float(r)
        # else: ignore


def consume_pending_reward_for_uid(
    uid: str,
    pending: Dict[str, float],
    time_penalty: float = 0.0,
) -> float:
    """
    Consume and zero-out pending reward for this uid, then apply optional time penalty.
    """
    r = float(pending.get(uid, 0.0))
    pending[uid] = 0.0
    return r - float(time_penalty)


def flush_pending_rewards_into_buffer(
    pending: Dict[str, float],
    last_buf_idx: Dict[str, int],
    buffer: RolloutBuffer,
) -> None:
    """
    Attach any leftover pending rewards to each agent’s LAST stored transition in the buffer.
    This prevents terminal rewards from being dropped when an agent did not act in the final window.
    """
    for uid, r in list(pending.items()):
        if abs(r) <= 1e-12:
            continue
        idx = last_buf_idx.get(uid, None)
        if idx is None:
            pending[uid] = 0.0
            continue
        buffer.rewards[idx] = float(buffer.rewards[idx]) + float(r)
        pending[uid] = 0.0


# ==========================================================
# OP3 Stabilization helpers
# ==========================================================
def outcome_bonus_from_scores(blue_score: int, red_score: int) -> float:
    if blue_score > red_score:
        return float(WIN_BONUS)
    if red_score > blue_score:
        return float(LOSS_PENALTY)
    return float(DRAW_PENALTY)


def add_terminal_reward_to_all_uids(
    pending_reward: Dict[str, float],
    episode_uid_set: set,
    terminal_r: float,
) -> float:
    """
    Add terminal reward to each uid's pending bucket.
    Returns the TEAM total injected (terminal_r * num_uids).
    """
    if abs(float(terminal_r)) <= 1e-12:
        return 0.0
    n = max(1, len(episode_uid_set))
    for uid in episode_uid_set:
        pending_reward[uid] = pending_reward.get(uid, 0.0) + float(terminal_r)
    return float(terminal_r) * float(n)


def apply_score_delta_shaping(
    gm: GameManager,
    episode_uid_set: set,
    pending_reward: Dict[str, float],
    prev_blue_score: int,
    prev_red_score: int,
) -> Tuple[int, int, float]:
    """
    Adds immediate rewards/penalties when the SCORE changes, so PPO cares about the objective.

    Returns updated (prev_blue_score, prev_red_score, team_total_added)
    """
    cur_b = int(getattr(gm, "blue_score", 0))
    cur_r = int(getattr(gm, "red_score", 0))

    db = cur_b - int(prev_blue_score)
    dr = cur_r - int(prev_red_score)

    team_added = 0.0
    n = max(1, len(episode_uid_set))

    if db != 0:
        per_uid = float(BLUE_SCORED_BONUS) * float(db)
        for uid in episode_uid_set:
            pending_reward[uid] = pending_reward.get(uid, 0.0) + per_uid
        team_added += per_uid * float(n)

    if dr != 0:
        per_uid = float(RED_SCORED_PENALTY) * float(dr)
        for uid in episode_uid_set:
            pending_reward[uid] = pending_reward.get(uid, 0.0) + per_uid
        team_added += per_uid * float(n)

    return cur_b, cur_r, team_added


# ==========================================================
# Entropy schedule (with OP3 reset bump)
# ==========================================================
def get_entropy_coef(cur_phase: str, phase_episode_count: int) -> float:
    base = float(ENT_COEF_BY_PHASE[cur_phase])

    if cur_phase == "OP1":
        start_ent, horizon = 0.07, 300.0
        frac = min(1.0, float(phase_episode_count) / horizon)
        return float(start_ent - (start_ent - base) * frac)

    if cur_phase == "OP2":
        start_ent, horizon = 0.06, 500.0
        frac = min(1.0, float(phase_episode_count) / horizon)
        return float(start_ent - (start_ent - base) * frac)

    # OP3: temporary bump at entry, then decay to base
    if phase_episode_count <= OP3_ENT_RESET_EPISODES:
        start_ent = float(OP3_ENT_RESET_VALUE)
        frac = min(1.0, float(phase_episode_count) / float(OP3_ENT_RESET_EPISODES))
        return float(start_ent - (start_ent - base) * frac)

    return base


# ==========================================================
# Action sampling (mask always present)
# ==========================================================
@torch.no_grad()
def sample_blue_action(
    policy: ActorCriticNet,
    obs_tensor: torch.Tensor,  # [1,C,H,W]
    env: GameField,
    agent: Any,
    deterministic: bool = False,
) -> Tuple[int, int, float, float, np.ndarray]:
    out = policy.act(
        obs_tensor,
        agent=agent,
        game_field=env,
        deterministic=deterministic,
        return_old_log_prob_key=False,
    )

    macro_idx = int(out["macro_action"].reshape(-1)[0].item())
    target_idx = int(out["target_action"].reshape(-1)[0].item())
    logp = float(out["log_prob"].reshape(-1)[0].item())
    val = float(out["value"].reshape(-1)[0].item())

    mm = out.get("macro_mask", None)
    if mm is None:
        mask_np = compute_used_macro_mask(env, agent)
    elif torch.is_tensor(mm):
        mask_np = mm.detach().cpu().numpy().astype(np.bool_)
        if mask_np.shape != (N_MACROS,):
            mask_np = mask_np.reshape(-1).astype(np.bool_)
    else:
        mask_np = np.array(mm, dtype=np.bool_).reshape(-1)

    if mask_np.shape != (N_MACROS,):
        raise RuntimeError(f"policy.act returned macro_mask shape {mask_np.shape}, expected {(N_MACROS,)}")

    return macro_idx, target_idx, logp, val, mask_np


# ==========================================================
# Training loop
# ==========================================================
def train_ppo_event(total_steps: int = TOTAL_STEPS) -> None:
    set_seed(42)

    env = make_env()
    gm = env.getGameManager()

    if hasattr(gm, "set_shaping_gamma"):
        gm.set_shaping_gamma(GAMMA)

    # Probe obs shape
    env.reset_default()
    if not env.blue_agents:
        raise RuntimeError("No blue agents after env.reset_default().")

    dummy_obs = env.build_observation(env.blue_agents[0])
    c = len(dummy_obs)
    h = len(dummy_obs[0])
    w = len(dummy_obs[0][0])
    print(f"[train_ppo_event] Sample obs shape: C={c}, H={h}, W={w}")

    policy = ActorCriticNet(
        n_macros=N_MACROS,
        n_targets=env.num_macro_targets,
        n_agents=getattr(env, "agents_per_team", 2),
        in_channels=NUM_CNN_CHANNELS,
        height=CNN_ROWS,
        width=CNN_COLS,
    ).to(DEVICE)

    print(f"[DEVICE] Using: {DEVICE}")

    optimizer = optim.Adam(policy.parameters(), lr=LR, foreach=False)
    buffer = RolloutBuffer(n_macros=N_MACROS)

    global_step = 0
    episode_idx = 0
    blue_wins = red_wins = draws = 0

    phase_idx = 0
    phase_episode_count = 0
    phase_recent: List[int] = []

    while global_step < total_steps:
        episode_idx += 1
        phase_episode_count += 1

        cur_phase = PHASE_SEQUENCE[phase_idx]
        phase_cfg = PHASE_CONFIG[cur_phase]

        gm.score_limit = int(phase_cfg["score_limit"])
        gm.max_time = float(phase_cfg["max_time"])
        max_steps = int(phase_cfg["max_macro_steps"])

        if hasattr(gm, "set_phase"):
            gm.set_phase(cur_phase)

        set_red_policy_for_phase(env, cur_phase)
        opponent_tag = cur_phase

        # Reset env for this episode (uids can change here)
        env.reset_default()

        # Drain any stale reward events after reset (defensive)
        clear_reward_events_best_effort(gm)

        # MUST be per-episode (prevents unknown-agent crashes after reset)
        episode_uid_set, pending_reward, last_buf_idx = init_episode_reward_routing(env)

        red_pol = env.policies.get("red")
        if hasattr(red_pol, "reset"):
            red_pol.reset()

        done = False
        steps = 0

        # Logging totals
        step_return_total = 0.0  # sum of consumed per-step rewards (team-total per window)
        term_return_total = 0.0  # terminal injections + leftover flush totals (team-total)

        # Score-delta shaping trackers
        prev_blue_score = int(getattr(gm, "blue_score", 0))
        prev_red_score  = int(getattr(gm, "red_score", 0))

        while (not done) and steps < max_steps and global_step < total_steps:
            blue_agents = [a for a in env.blue_agents if a.isEnabled()]
            if not blue_agents:
                done = True
                break

            decisions: List[Tuple[Any, np.ndarray, int, int, float, float, np.ndarray, str]] = []
            submit_actions: Dict[str, Tuple[int, int]] = {}

            # --- Decide actions for enabled agents ---
            for agent in blue_agents:
                obs = np.array(env.build_observation(agent), dtype=np.float32)
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                macro_idx, target_idx, logp, val, mask_np = sample_blue_action(
                    policy, obs_tensor, env, agent, deterministic=False
                )

                macro_enum = USED_MACROS[macro_idx]
                macro_val = int(getattr(macro_enum, "value", int(macro_enum)))

                k = external_key_for_agent(env, agent)
                submit_actions[k] = (macro_val, int(target_idx))

                uid = agent_uid(agent)
                decisions.append((agent, obs, macro_idx, target_idx, logp, val, mask_np, uid))

            # --- Submit actions ---
            apply_blue_actions(env, submit_actions)

            # --- Simulate one decision window ---
            sim_decision_window(env, gm, DECISION_WINDOW, SIM_DT)

            done = bool(gm.game_over)
            dt = compute_dt_for_window(DECISION_WINDOW)

            # 1) collect event-driven rewards into pending (episode-filtered)
            accumulate_rewards_for_uid_set(gm, episode_uid_set, pending_reward)

            # 2) add objective-aligned score-delta shaping into pending
            prev_blue_score, prev_red_score, _team_added = apply_score_delta_shaping(
                gm=gm,
                episode_uid_set=episode_uid_set,
                pending_reward=pending_reward,
                prev_blue_score=prev_blue_score,
                prev_red_score=prev_red_score,
            )

            # --- Next values for bootstrap ---
            done_for_rollout_global = bool(done or ((steps + 1) >= max_steps))
            decision_uids = [uid for *_rest, uid in decisions]
            next_vals_by_uid: Dict[str, float] = {uid: 0.0 for uid in decision_uids}

            live_obs: List[np.ndarray] = []
            live_uids: List[str] = []
            for (agent, _obs, _mi, _ti, _lp, _v, _mm, uid) in decisions:
                agent_dead = (not agent.isEnabled())
                if (not done_for_rollout_global) and (not agent_dead):
                    live_obs.append(np.array(env.build_observation(agent), dtype=np.float32))
                    live_uids.append(uid)

            if live_obs:
                next_obs_tensor = torch.tensor(np.stack(live_obs), dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    next_v = policy.forward_local_critic(next_obs_tensor).detach().cpu().numpy()
                for i, uid in enumerate(live_uids):
                    next_vals_by_uid[uid] = float(next_v[i])

            # --- Store transitions (consume pending per acting uid) ---
            step_reward_sum = 0.0
            for (agent, obs, macro_idx, target_idx, logp, val, mask_np, uid) in decisions:
                r = consume_pending_reward_for_uid(uid, pending_reward, TIME_PENALTY_PER_AGENT_PER_MACRO)
                step_reward_sum += r

                agent_dead = (not agent.isEnabled())
                agent_done = bool(done_for_rollout_global or agent_dead)
                nv = 0.0 if agent_done else float(next_vals_by_uid[uid])

                buffer.add(
                    obs=obs,
                    macro_action=macro_idx,
                    target_action=target_idx,
                    log_prob=logp,
                    value=val,
                    next_value=nv,
                    reward=r,
                    done=agent_done,
                    dt=dt,
                    macro_mask=mask_np,
                )
                global_step += 1
                last_buf_idx[uid] = buffer.size() - 1

            step_return_total += float(step_reward_sum)
            steps += 1

            # --- Update (terminal-safe): do not update on a terminal step ---
            if buffer.size() >= UPDATE_EVERY and (not done_for_rollout_global):
                flush_pending_rewards_into_buffer(pending_reward, last_buf_idx, buffer)

                current_ent_coef = get_entropy_coef(cur_phase, phase_episode_count)
                print(
                    f"[UPDATE] step={global_step} episode={episode_idx} "
                    f"phase={cur_phase} ENT={current_ent_coef:.4f} Opp={opponent_tag}"
                )

                ppo_update(policy, optimizer, buffer, DEVICE, current_ent_coef)

                # buffer cleared inside ppo_update
                last_buf_idx.clear()

        # ==========================================================
        # Episode end: outcome terminal + flush leftovers into buffer
        # ==========================================================
        outcome_r = outcome_bonus_from_scores(gm.blue_score, gm.red_score)
        team_outcome_added = add_terminal_reward_to_all_uids(pending_reward, episode_uid_set, outcome_r)
        term_return_total += float(team_outcome_added)

        # leftover pending is per-uid; include it in TermR logs and learning
        leftover_before = float(sum(pending_reward.values()))
        term_return_total += float(leftover_before)

        # attach leftovers to last transitions
        flush_pending_rewards_into_buffer(pending_reward, last_buf_idx, buffer)

        # If we skipped an update because episode ended, catch up now
        if buffer.size() >= UPDATE_EVERY:
            current_ent_coef = get_entropy_coef(cur_phase, phase_episode_count)
            print(
                f"[UPDATE@EP_END] step={global_step} episode={episode_idx} "
                f"phase={cur_phase} ENT={current_ent_coef:.4f} Opp={opponent_tag}"
            )
            ppo_update(policy, optimizer, buffer, DEVICE, current_ent_coef)
            last_buf_idx.clear()

        # --- Episode result ---
        if gm.blue_score > gm.red_score:
            result = "BLUE WIN"
            blue_wins += 1
            phase_recent.append(1)
        elif gm.red_score > gm.blue_score:
            result = "RED WIN"
            red_wins += 1
            phase_recent.append(0)
        else:
            result = "DRAW"
            draws += 1
            phase_recent.append(0)

        if len(phase_recent) > PHASE_WINRATE_WINDOW:
            phase_recent = phase_recent[-PHASE_WINRATE_WINDOW:]

        phase_wr = sum(phase_recent) / max(1, len(phase_recent))
        avg_step_r = step_return_total / max(1, steps)

        print(
            f"[{episode_idx:5d}] {result:8} | "
            f"StepR {avg_step_r:+.3f} "
            f"TermR {term_return_total:+.1f} | "
            f"Score {gm.blue_score}:{gm.red_score} | "
            f"BlueWins: {blue_wins} RedWins: {red_wins} Draws: {draws} | "
            f"PhaseWin {phase_wr * 100:.1f}% | Phase={cur_phase} Opp={opponent_tag}"
        )

        # --- Curriculum advance ---
        if cur_phase != PHASE_SEQUENCE[-1]:
            min_eps = MIN_PHASE_EPISODES[cur_phase]
            target_wr = TARGET_PHASE_WINRATE[cur_phase]
            if phase_episode_count >= min_eps and len(phase_recent) >= PHASE_WINRATE_WINDOW and phase_wr >= target_wr:
                print(f"[CURRICULUM] Advancing from {cur_phase} -> next phase.")
                phase_idx += 1
                phase_episode_count = 0
                phase_recent.clear()

    final_path = os.path.join(CHECKPOINT_DIR, "research_model1.pth")
    torch.save(policy.state_dict(), final_path)
    print(f"\nTraining complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    train_ppo_event()
