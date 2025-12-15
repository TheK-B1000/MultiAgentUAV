import os
import random
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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


# ---------------- Device ----------------
def get_device() -> torch.device:
    if HAS_TDML:
        return torch_directml.device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


# ---------------- Reproducibility ----------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN flags only matter on CUDA, harmless otherwise
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------- Hyperparams ----------------
GRID_ROWS = CNN_ROWS  # 40
GRID_COLS = CNN_COLS  # 30

TOTAL_STEPS: int = 5_000_000
UPDATE_EVERY: int = 4096
PPO_EPOCHS: int = 10
MINIBATCH_SIZE: int = 256

LR: float = 3e-4
CLIP_EPS: float = 0.2
VALUE_COEF: float = 1.0
MAX_GRAD_NORM: float = 0.5

GAMMA: float = 0.995
GAE_LAMBDA: float = 0.97

DECISION_WINDOW: float = 1.0
SIM_DT: float = 0.1

CHECKPOINT_DIR: str = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# NOTE: Policy outputs macro indices 0..len(USED_MACROS)-1.
USED_MACROS: List[MacroAction] = [
    MacroAction.GO_TO,
    MacroAction.GRAB_MINE,
    MacroAction.GET_FLAG,
    MacroAction.PLACE_MINE,
    MacroAction.GO_HOME,
]
NUM_ACTIONS: int = len(USED_MACROS)

PHASE_SEQUENCE: List[str] = ["OP1", "OP2", "OP3"]
MIN_PHASE_EPISODES: Dict[str, int] = {"OP1": 300, "OP2": 700, "OP3": 1500}
TARGET_PHASE_WINRATE: Dict[str, float] = {"OP1": 0.95, "OP2": 0.85, "OP3": 0.75}
PHASE_WINRATE_WINDOW: int = 50

ENT_COEF_BY_PHASE: Dict[str, float] = {"OP1": 0.07, "OP2": 0.04, "OP3": 0.025}
PHASE_CONFIG: Dict[str, Dict[str, float]] = {
    "OP1": dict(score_limit=3, max_time=200.0, max_macro_steps=500),
    "OP2": dict(score_limit=3, max_time=200.0, max_macro_steps=450),
    "OP3": dict(score_limit=3, max_time=200.0, max_macro_steps=550),
}

TIME_PENALTY_PER_AGENT_PER_MACRO = 0.001


# ---------------- Env helpers ----------------
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

    # BLUE is externally controlled by trainer
    if hasattr(env, "set_external_control"):
        env.set_external_control("blue", True)
        env.set_external_control("red", False)

    if hasattr(env, "external_missing_action_mode"):
        env.external_missing_action_mode = "idle"

    return env


def apply_blue_actions_compat(env: GameField, actions_by_agent: Dict[str, Tuple[int, int]]) -> None:
    if hasattr(env, "submit_external_actions"):
        if hasattr(env, "pending_external_actions"):
            try:
                env.pending_external_actions.clear()
            except Exception:
                pass
        env.submit_external_actions(actions_by_agent)
        return


# ---------------- Rollout Buffer ----------------
class RolloutBuffer:
    def __init__(self) -> None:
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
        macro_mask: Optional[np.ndarray] = None,
    ) -> None:
        self.obs.append(np.array(obs, dtype=np.float32))
        self.macro_actions.append(int(macro_action))
        self.target_actions.append(int(target_action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.next_values.append(float(next_value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.dts.append(float(dt))

        if macro_mask is None:
            self.macro_masks.append(np.array([], dtype=np.bool_))
        else:
            self.macro_masks.append(np.array(macro_mask, dtype=np.bool_))

    def size(self) -> int:
        return len(self.obs)

    def clear(self) -> None:
        self.__init__()

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

        if len(self.macro_masks) > 0 and self.macro_masks[0].size > 0:
            macro_masks = torch.tensor(np.stack(self.macro_masks), dtype=torch.bool, device=device) # [T,n_macros]
        else:
            macro_masks = torch.empty((obs.size(0), 0), dtype=torch.bool, device=device)

        return obs, macro_actions, target_actions, log_probs, values, next_values, rewards, dones, dts, macro_masks


# ---------------- Event-driven GAE ----------------
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


# ---------------- PPO Update ----------------
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

    advantages, returns = compute_gae_event(rewards, values, next_values, dones, dts)
    advantages = normalize_advantages(advantages)

    T = obs.size(0)
    if T == 0:
        buffer.clear()
        return

    for _ in range(PPO_EPOCHS):
        # Torch permutation (avoid numpy->advanced indexing on DML)
        perm = torch.randperm(T, device=device)

        for start in range(0, T, MINIBATCH_SIZE):
            end = min(start + MINIBATCH_SIZE, T)
            mb_idx = perm[start:end]  # 1D torch tensor on device

            # DML-safe: index_select (single-dimension gather)
            mb_obs      = obs.index_select(0, mb_idx)
            mb_macro    = macro_actions.index_select(0, mb_idx)
            mb_target   = target_actions.index_select(0, mb_idx)
            mb_old_logp = old_log_probs.index_select(0, mb_idx)
            mb_adv      = advantages.index_select(0, mb_idx)
            mb_ret      = returns.index_select(0, mb_idx)

            mb_mask = None
            if macro_masks.numel() > 0 and macro_masks.size(1) > 0:
                mb_mask = macro_masks.index_select(0, mb_idx)

            # Forward
            macro_logits, target_logits, _ = policy.forward_actor(mb_obs)
            new_values = policy.forward_local_critic(mb_obs)

            # DML-safe logp/entropy (NO gather/scatter backward)
            macro_logp, macro_ent = masked_categorical_logp_entropy_no_scatter(
                macro_logits, mb_macro, mask=mb_mask
            )
            target_logp, target_ent = masked_categorical_logp_entropy_no_scatter(
                target_logits, mb_target, mask=None
            )

            new_logp = macro_logp + target_logp  # [B]
            entropy = macro_ent + target_ent  # [B]

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

# ---------------- Reward collection (NO extra discount) ----------------
def agent_uid(agent: Any) -> str:
    if hasattr(agent, "unique_id"):
        return str(agent.unique_id)
    if hasattr(agent, "slot_id"):
        return str(agent.slot_id)
    side = getattr(agent, "side", "blue")
    aid = getattr(agent, "agent_id", 0)
    return f"{side}_{aid}"


def gm_pop_reward_events_safe(gm: GameManager):
    if hasattr(gm, "pop_reward_events") and callable(getattr(gm, "pop_reward_events")):
        return gm.pop_reward_events()
    ev = getattr(gm, "reward_events", [])
    try:
        out = list(ev)
        if isinstance(ev, list):
            ev.clear()
        return out
    except Exception:
        return []


def collect_blue_rewards_for_step(
    gm: GameManager,
    blue_agents: List[Any],
) -> Dict[str, float]:
    """
    Collect all reward events since last pop.
    IMPORTANT: no discounting here. Discounting is handled in dt-aware GAE.
    """
    raw_events = gm_pop_reward_events_safe(gm)
    uids = [agent_uid(a) for a in blue_agents]
    rewards_sum_by_id: Dict[str, float] = {uid: 0.0 for uid in uids}
    team_r_total = 0.0

    for _t_event, agent_id, r in raw_events:
        if agent_id is None:
            team_r_total += float(r)
        else:
            aid = str(agent_id)
            if aid in rewards_sum_by_id:
                rewards_sum_by_id[aid] += float(r)

    if team_r_total != 0.0 and len(rewards_sum_by_id) > 0:
        share = team_r_total / len(rewards_sum_by_id)
        for aid in rewards_sum_by_id:
            rewards_sum_by_id[aid] += share

    return rewards_sum_by_id


# ---------------- Time helpers ----------------
def gm_get_time(gm: GameManager) -> float:
    if hasattr(gm, "get_sim_time") and callable(getattr(gm, "get_sim_time")):
        return float(gm.get_sim_time())
    if hasattr(gm, "sim_time"):
        return float(getattr(gm, "sim_time"))
    if hasattr(gm, "max_time") and hasattr(gm, "current_time"):
        try:
            return float(getattr(gm, "max_time")) - float(getattr(gm, "current_time"))
        except Exception:
            return 0.0
    return 0.0


def get_entropy_coef(cur_phase: str, phase_episode_count: int) -> float:
    base = ENT_COEF_BY_PHASE[cur_phase]
    if cur_phase == "OP1":
        start_ent, horizon = 0.07, 300.0
    elif cur_phase == "OP2":
        start_ent, horizon = 0.04, 500.0
    else:
        start_ent, horizon = 0.025, 800.0
    frac = min(1.0, phase_episode_count / horizon)
    return float(start_ent - (start_ent - base) * frac)


# ---------------- Masking in USED_MACROS index-space ----------------
def _get_agent_xy(agent: Any) -> Tuple[float, float]:
    fp = getattr(agent, "float_pos", None)
    if isinstance(fp, (tuple, list)) and len(fp) >= 2:
        return float(fp[0]), float(fp[1])
    return float(getattr(agent, "x", 0.0)), float(getattr(agent, "y", 0.0))

def masked_categorical_logp_entropy_no_scatter(
    logits: torch.Tensor,              # [B, A]
    actions: torch.Tensor,             # [B] int64
    mask: Optional[torch.Tensor] = None # [B, A] bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      logp:    [B]
      entropy: [B]
    DML-safe: avoids gather/scatter in backward by using CPU one-hot as constant.
    """
    if mask is not None and mask.numel() > 0:
        logits = logits.masked_fill(~mask, -1e10)

    logp_all = F.log_softmax(logits, dim=-1)         # [B,A]
    p_all = torch.exp(logp_all)                      # [B,A]

    # Build one-hot on CPU so any scatter happens on CPU and is NOT in the backward graph.
    oh = F.one_hot(actions.to("cpu"), num_classes=logits.size(-1)).to(
        device=logits.device, dtype=logp_all.dtype
    )                                                # [B,A] constant wrt logits

    logp = (oh * logp_all).sum(dim=-1)               # [B] (dense ops, no gather)
    entropy = -(p_all * logp_all).sum(dim=-1)        # [B]
    return logp, entropy

def compute_used_macro_mask(env: GameField, agent: Any) -> np.ndarray:
    """
    Returns bool mask of shape [len(USED_MACROS)] aligned with macro index space.
    """
    mask = np.ones((len(USED_MACROS),), dtype=np.bool_)

    # Helper: carrying flag?
    carrying = False
    if hasattr(agent, "isCarryingFlag") and callable(getattr(agent, "isCarryingFlag")):
        carrying = bool(agent.isCarryingFlag())
    else:
        carrying = bool(getattr(agent, "carrying_flag", False))

    # PLACE_MINE needs charges
    if getattr(agent, "mine_charges", 0) <= 0:
        if MacroAction.PLACE_MINE in USED_MACROS:
            mask[USED_MACROS.index(MacroAction.PLACE_MINE)] = False

    # GRAB_MINE requires at least one friendly pickup nearby (soft gate)
    has_pickup_near = False
    ax, ay = _get_agent_xy(agent)
    for p in getattr(env, "mine_pickups", []):
        if getattr(p, "owner_side", None) == getattr(agent, "side", None):
            dx = float(getattr(p, "x", 0.0)) - ax
            dy = float(getattr(p, "y", 0.0)) - ay
            if (dx * dx + dy * dy) ** 0.5 < 3.0:
                has_pickup_near = True
                break
    if not has_pickup_near and MacroAction.GRAB_MINE in USED_MACROS:
        mask[USED_MACROS.index(MacroAction.GRAB_MINE)] = False

    # GET_FLAG is pointless (and sometimes harmful) while carrying
    if carrying and MacroAction.GET_FLAG in USED_MACROS:
        mask[USED_MACROS.index(MacroAction.GET_FLAG)] = False

    # Guard: never all-false
    if not mask.any():
        mask[:] = True

    return mask


@torch.no_grad()
def sample_blue_action(
    policy: ActorCriticNet,
    obs_tensor: torch.Tensor,  # [1,C,H,W]
    env: GameField,
    agent: Any,
    deterministic: bool = False,
) -> Tuple[int, int, float, float, np.ndarray]:
    """
    Samples (macro_idx, target_idx) with mask aligned to USED_MACROS index-space.
    Also returns logp, value, and mask (np.bool_).
    """
    device = next(policy.parameters()).device
    obs_tensor = obs_tensor.to(device, non_blocking=True)

    macro_logits, target_logits, latent = policy.forward_actor(obs_tensor)
    value = policy.local_value_head(latent).squeeze(-1)  # [1]

    mask_np = compute_used_macro_mask(env, agent)
    mask_t = torch.tensor(mask_np, dtype=torch.bool, device=device).unsqueeze(0)  # [1,n_macros]
    macro_logits = macro_logits.masked_fill(~mask_t, -1e10)

    macro_dist = torch.distributions.Categorical(logits=macro_logits)
    target_dist = torch.distributions.Categorical(logits=target_logits)

    if deterministic:
        macro_action = macro_logits.argmax(dim=-1)
        target_action = target_logits.argmax(dim=-1)
    else:
        macro_action = macro_dist.sample()
        target_action = target_dist.sample()

    logp = macro_dist.log_prob(macro_action) + target_dist.log_prob(target_action)

    macro_idx = int(macro_action.item())
    target_idx = int(target_action.item())
    return macro_idx, target_idx, float(logp.item()), float(value.item()), mask_np


# ---------------- Sim stepping ----------------
def sim_decision_window(env: GameField, gm: GameManager, window_s: float, sim_dt: float) -> None:
    """
    Steps env for ~window_s seconds using sim_dt, with remainder handling.
    Avoids drift due to float compare quirks.
    """
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


# ---------------- Training loop ----------------
def train_ppo_event(total_steps: int = TOTAL_STEPS) -> None:
    set_seed(42)

    env = make_env()
    gm = env.getGameManager()

    env.reset_default()
    if env.blue_agents:
        dummy_obs = env.build_observation(env.blue_agents[0])
        c = len(dummy_obs)
        h = len(dummy_obs[0])
        w = len(dummy_obs[0][0])
        print(f"[train_ppo_event] Sample obs shape: C={c}, H={h}, W={w}")
    else:
        print("[train_ppo_event] WARNING: No blue agents in env.reset_default().")

    policy = ActorCriticNet(
        n_macros=len(USED_MACROS),
        n_targets=env.num_macro_targets,
        n_agents=getattr(env, "agents_per_team", 2),
        in_channels=NUM_CNN_CHANNELS,
        height=CNN_ROWS,
        width=CNN_COLS,
    ).to(DEVICE)

    print(f"[DEVICE] Using: {DEVICE}")

    optimizer = optim.Adam(policy.parameters(), lr=LR, foreach=False)
    buffer = RolloutBuffer()

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

        gm.score_limit = phase_cfg["score_limit"]
        gm.max_time = phase_cfg["max_time"]
        max_steps = int(phase_cfg["max_macro_steps"])

        gm.set_phase(cur_phase)
        set_red_policy_for_phase(env, cur_phase)
        opponent_tag = cur_phase

        # One reset is enough; do NOT double-reset gm.
        env.reset_default()

        red_pol = env.policies.get("red")
        if hasattr(red_pol, "reset"):
            red_pol.reset()

        done = False
        ep_return = 0.0
        steps = 0

        decision_time_start = gm_get_time(gm)

        while (not done) and steps < max_steps and global_step < total_steps:
            # agents that can act at the start of decision
            blue_agents = [a for a in env.blue_agents if a.isEnabled()]
            decisions: List[Tuple[Any, np.ndarray, int, int, float, float, np.ndarray, str]] = []
            submit_actions: Dict[str, Tuple[int, int]] = {}

            # Decide for each enabled agent
            for agent in blue_agents:
                obs = np.array(env.build_observation(agent), dtype=np.float32)
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                macro_idx, target_idx, logp, val, mask_np = sample_blue_action(
                    policy, obs_tensor, env, agent, deterministic=False
                )

                # Map macro index -> actual MacroAction enum -> macro_val for env
                macro_enum = USED_MACROS[macro_idx]
                macro_val = int(getattr(macro_enum, "value", int(macro_enum)))

                slot_id = getattr(agent, "slot_id", f"{agent.side}_{getattr(agent, 'agent_id', 0)}")
                submit_actions[str(slot_id)] = (macro_val, int(target_idx))

                decisions.append((agent, obs, macro_idx, target_idx, logp, val, mask_np, agent_uid(agent)))

            # Submit external actions for this boundary
            apply_blue_actions_compat(env, submit_actions)

            # Simulate exactly one decision window
            sim_decision_window(env, gm, DECISION_WINDOW, SIM_DT)

            done = bool(gm.game_over)
            decision_time_end = gm_get_time(gm)
            dt = max(0.0, float(decision_time_end) - float(decision_time_start))

            # Collect rewards for this window (no extra discount)
            rewards_by_uid = collect_blue_rewards_for_step(gm, blue_agents)
            for agent in blue_agents:
                uid = agent_uid(agent)
                rewards_by_uid[uid] = rewards_by_uid.get(uid, 0.0) - TIME_PENALTY_PER_AGENT_PER_MACRO

            # Compute per-agent next values ONLY for agents that are alive and not terminal
            done_for_rollout_global = bool(done or ((steps + 1) >= max_steps))
            next_vals_by_uid: Dict[str, float] = {}

            enabled_obs = []
            enabled_uids = []
            for (agent, _obs, _macro_idx, _target_idx, _logp, _val, _mask_np, uid) in decisions:
                next_vals_by_uid[uid] = 0.0
                agent_dead = (not agent.isEnabled())
                if (not done_for_rollout_global) and (not agent_dead):
                    enabled_obs.append(np.array(env.build_observation(agent), dtype=np.float32))
                    enabled_uids.append(uid)

            if enabled_obs:
                next_obs_tensor = torch.tensor(np.stack(enabled_obs), dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    next_v = policy.forward_local_critic(next_obs_tensor).detach().cpu().numpy()
                for i, uid in enumerate(enabled_uids):
                    next_vals_by_uid[uid] = float(next_v[i])

            # Store transitions (with per-agent done if agent died)
            step_reward_sum = 0.0
            for (agent, obs, macro_idx, target_idx, logp, val, mask_np, uid) in decisions:
                r = float(rewards_by_uid.get(uid, 0.0))
                step_reward_sum += r

                agent_dead = (not agent.isEnabled())
                agent_done = bool(done_for_rollout_global or agent_dead)
                nv = 0.0 if agent_done else float(next_vals_by_uid.get(uid, 0.0))

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

            ep_return += step_reward_sum
            steps += 1
            decision_time_start = decision_time_end

            # Update
            if buffer.size() >= UPDATE_EVERY:
                current_ent_coef = get_entropy_coef(cur_phase, phase_episode_count)
                print(
                    f"[UPDATE] step={global_step} episode={episode_idx} "
                    f"phase={cur_phase} ENT={current_ent_coef:.4f} Opp={opponent_tag}"
                )
                ppo_update(policy, optimizer, buffer, DEVICE, current_ent_coef)

        # Episode result
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
        avg_step_r = ep_return / max(1, steps)

        print(
            f"[{episode_idx:5d}] {result:8} | "
            f"StepR {avg_step_r:+.3f} "
            f"TermR {ep_return:+.1f} | "
            f"Score {gm.blue_score}:{gm.red_score} | "
            f"BlueWins: {blue_wins} RedWins: {red_wins} Draws: {draws} | "
            f"PhaseWin {phase_wr * 100:.1f}% | Phase={cur_phase} Opp={opponent_tag}"
        )

        # Curriculum advance
        if cur_phase != PHASE_SEQUENCE[-1]:
            min_eps = MIN_PHASE_EPISODES[cur_phase]
            target_wr = TARGET_PHASE_WINRATE[cur_phase]
            if phase_episode_count >= min_eps and len(phase_recent) >= PHASE_WINRATE_WINDOW and phase_wr >= target_wr:
                print(f"[CURRICULUM] Advancing from {cur_phase} → next phase.")
                phase_idx += 1
                phase_episode_count = 0
                phase_recent.clear()

    final_path = os.path.join(CHECKPOINT_DIR, "research_model1.pth")
    torch.save(policy.state_dict(), final_path)
    print(f"\nTraining complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    train_ppo_event()
