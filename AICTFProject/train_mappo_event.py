"""
train_mappo_event.py (MAPPO / CTDE, DirectML-safe by design) — UPDATED TO FIX “unknown agent_id” CRASH

Key design:
  - policy_act runs on DirectML/CUDA for action sampling only (forward only).
  - policy_train runs on CPU (or CUDA) for MAPPO updates (backward only).
  - After each update, policy_train weights are synced to policy_act.

Core fix vs your crash:
  ✅ Rewards are now collected EPISODE-STRICT (must belong to this episode’s blue agents),
     but NOT “acted-only strict”.
     We use a per-uid PENDING accumulator and attach leftover rewards to the agent’s last
     stored transition (flush at update boundaries + episode end).

This prevents:
  RuntimeError: Reward event for unknown agent_id='blue_1_925'. Expected one of: ['blue_0_924']
which happens when GameManager emits reward for an agent that didn’t act that window.
"""

import os
import random
import copy
from collections import deque
from typing import Dict, List, Tuple, Any, Deque, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================================
# DEVICE SELECTION (DirectML → CUDA → CPU)
# ==========================================================
try:
    import torch_directml
    HAS_TDML = True
except ImportError:
    torch_directml = None
    HAS_TDML = False


def prefer_device() -> torch.device:
    if HAS_TDML:
        return torch_directml.device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ==========================================================
# IMPORTS
# ==========================================================
from game_field import GameField, CNN_ROWS, CNN_COLS, NUM_CNN_CHANNELS
from game_manager import GameManager
from macro_actions import MacroAction
from rl_policy import ActorCriticNet
from policies import OP1RedPolicy, OP2RedPolicy, OP3RedPolicy


# ==========================================================
# CONFIG
# ==========================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


GRID_ROWS = CNN_ROWS
GRID_COLS = CNN_COLS

TOTAL_STEPS = 5_000_000
UPDATE_EVERY = 2_048          # samples (per-agent) in buffer
PPO_EPOCHS = 10
MINIBATCH_SIZE = 64

LR = 3e-4
CLIP_EPS = 0.2
VALUE_COEF = 2.0
MAX_GRAD_NORM = 0.5

GAMMA = 0.995
GAE_LAMBDA = 0.99

DECISION_WINDOW = 0.7
SIM_DT = 0.1

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

PHASE_SEQUENCE = ["OP1", "OP2", "OP3"]
MIN_PHASE_EPISODES = {"OP1": 500, "OP2": 1000, "OP3": 2000}
TARGET_PHASE_WINRATE = {"OP1": 0.90, "OP2": 0.86, "OP3": 0.80}
PHASE_WINRATE_WINDOW = 50

ENT_COEF_BY_PHASE = {"OP1": 0.06, "OP2": 0.05, "OP3": 0.04}
OP2_ENTROPY_FLOOR = 0.008

TIME_PENALTY_PER_AGENT_PER_MACRO = 0.01

PHASE_CONFIG = {
    "OP1": dict(score_limit=3, max_time=200.0, max_macro_steps=500),
    "OP2": dict(score_limit=3, max_time=200.0, max_macro_steps=500),
    "OP3": dict(score_limit=3, max_time=200.0, max_macro_steps=500),
}

USED_MACROS = [
    MacroAction.GO_TO,
    MacroAction.GRAB_MINE,
    MacroAction.GET_FLAG,
    MacroAction.PLACE_MINE,
    MacroAction.GO_HOME,
]
N_MACROS = len(USED_MACROS)

MACRO_STATS_PRINT_EVERY = 10
MACRO_STATS_PRINT_ON_WIN = True


# ==========================================================
# ENV HELPERS
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
        try:
            env.set_external_control("blue", True)
            env.set_external_control("red", False)
        except Exception:
            pass

    if hasattr(env, "external_missing_action_mode"):
        try:
            env.external_missing_action_mode = "idle"
        except Exception:
            pass

    return env


def external_keys_for_agent(env: GameField, agent: Any) -> List[str]:
    keys: List[str] = []
    if hasattr(env, "_external_key_for_agent"):
        try:
            keys.append(str(env._external_key_for_agent(agent)))
        except Exception:
            pass
    if hasattr(agent, "slot_id"):
        keys.append(str(getattr(agent, "slot_id")))
    if hasattr(agent, "unique_id"):
        keys.append(str(getattr(agent, "unique_id")))
    keys.append(f"{getattr(agent, 'side', 'blue')}_{getattr(agent, 'agent_id', 0)}")

    out, seen = [], set()
    for k in keys:
        if k and k not in seen:
            out.append(k)
            seen.add(k)
    return out


def agent_uid(agent: Any) -> str:
    """
    Reward-routing uid.
    MUST match GameManager reward event agent_id strings.
    In your logs, rewards are emitted as 'blue_0_924', 'blue_1_925', etc => use unique_id first.
    """
    uid = getattr(agent, "unique_id", None)
    if uid is None or str(uid).strip() == "":
        uid = getattr(agent, "slot_id", None)
    if uid is None or str(uid).strip() == "":
        uid = f"{getattr(agent,'side','blue')}_{getattr(agent,'agent_id',0)}"
    return str(uid)


def agent_slot(agent: Any) -> str:
    """Stable printing id (does not change every episode)."""
    sid = getattr(agent, "slot_id", None)
    if sid is None or str(sid).strip() == "":
        sid = f"{getattr(agent,'side','blue')}_{getattr(agent,'agent_id',0)}"
    return str(sid)


def gm_get_sim_time_safe(gm: GameManager) -> float:
    if hasattr(gm, "get_sim_time") and callable(getattr(gm, "get_sim_time")):
        return float(gm.get_sim_time())
    if hasattr(gm, "sim_time"):
        return float(getattr(gm, "sim_time"))
    return 0.0


def sim_decision_window(env: GameField, gm: GameManager, window_s: float, sim_dt: float) -> None:
    """
    Steps env for ~window_s seconds using sim_dt, with remainder handling.
    Stops early if gm.game_over becomes True.
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


def submit_external_actions_robust(env: GameField, action_dict: Dict[str, Tuple[int, int]]) -> None:
    # clear any stale “pending” dict if you keep one around
    if hasattr(env, "pending_external_actions"):
        try:
            env.pending_external_actions.clear()
        except Exception:
            pass

    if hasattr(env, "submit_external_actions") and callable(getattr(env, "submit_external_actions")):
        try:
            env.submit_external_actions(action_dict)
            return
        except Exception:
            pass

    if hasattr(env, "pending_external_actions"):
        try:
            for k, v in action_dict.items():
                env.pending_external_actions[k] = v
            return
        except Exception:
            pass

    if hasattr(env, "external_actions"):
        try:
            env.external_actions = action_dict
            return
        except Exception:
            pass

    raise RuntimeError("No supported external action submission path found in GameField.")


def zero_obs_like() -> np.ndarray:
    return np.zeros((NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32)


def get_blue_team_obs_in_id_order(env: GameField) -> List[np.ndarray]:
    """
    Central obs uses fixed agent_id ordering.
    Disabled agent slots get a zero obs (keeps tensor shapes stable).
    """
    blue_all = list(env.blue_agents)
    blue_all.sort(key=lambda a: getattr(a, "agent_id", 0))

    out: List[np.ndarray] = []
    for a in blue_all:
        if a is None or (not a.isEnabled()):
            out.append(zero_obs_like())
        else:
            out.append(np.array(env.build_observation(a), dtype=np.float32))
    return out

# ==========================================================
# REWARDS (FILTERED BY ALLOWED UID SET + PENDING ACCUMULATOR) ✅ FIX
# ==========================================================
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


def clear_reward_events_best_effort(gm: GameManager) -> None:
    try:
        _ = gm_pop_reward_events_safe(gm)
    except Exception:
        pass


def init_episode_reward_routing(env: GameField) -> Tuple[Set[str], Dict[str, float], Dict[str, int]]:
    """
    allowed_uids = the uids we train on for THIS episode (blue agents at reset).
    pending = per-uid reward accumulator (rewards can arrive even when the uid didn't act this window).
    last_buf_idx = uid -> last transition index in the buffer (for flushing pending at boundaries).
    """
    uids: List[str] = []
    for a in getattr(env, "blue_agents", []):
        if a is None:
            continue
        uids.append(agent_uid(a))

    allowed_uids = set(uids)
    pending = {uid: 0.0 for uid in allowed_uids}
    last_buf_idx: Dict[str, int] = {}
    return allowed_uids, pending, last_buf_idx


def accumulate_rewards_for_uid_set(
    gm: GameManager,
    allowed_uids: Set[str],
    pending: Dict[str, float],
) -> None:
    """
    Pop ALL reward events, but only accumulate those whose agent_id is in allowed_uids.
    Everything else is ignored (red team, old episode uids, etc.).

    Still strict about invariants:
      - agent_id must exist and be a non-empty string
    """
    events = gm_pop_reward_events_safe(gm)
    for _t, agent_id, r in events:
        if agent_id is None:
            raise RuntimeError("Found reward event with agent_id=None (must always be per-agent).")
        key = str(agent_id).strip()
        if not key:
            raise RuntimeError("Found reward event with empty agent_id.")

        if key in allowed_uids:
            pending[key] = pending.get(key, 0.0) + float(r)
        # else: ignore


def consume_pending_reward_for_uid(uid: str, pending: Dict[str, float], time_penalty: float = 0.0) -> float:
    r = float(pending.get(uid, 0.0))
    pending[uid] = 0.0
    return r - float(time_penalty)


def flush_pending_rewards_into_buffer(
    pending: Dict[str, float],
    last_buf_idx: Dict[str, int],
    buffer: "MAPPORolloutBuffer",
) -> None:
    """
    At update boundaries + episode end, attach leftover pending rewards to each uid’s LAST transition.
    Prevents reward loss when a uid didn’t act in the last window.
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
# ENTROPY SCHEDULE
# ==========================================================
def get_entropy_coef(cur_phase: str, phase_episode_count: int) -> float:
    base = ENT_COEF_BY_PHASE[cur_phase]
    if cur_phase == "OP1":
        start_ent, horizon = 0.07, 1000.0
    elif cur_phase == "OP2":
        start_ent, horizon = 0.06, 1200.0
    else:
        start_ent, horizon = 0.05, 2000.0

    frac = min(1.0, phase_episode_count / horizon)
    coef = float(start_ent - (start_ent - base) * frac)
    if cur_phase == "OP2":
        coef = max(coef, OP2_ENTROPY_FLOOR)
    return coef


# ==========================================================
# ACTION SAMPLING (policy_act forward only)
# ==========================================================
@torch.no_grad()
def sample_mappo_action_via_act(
    policy_act: ActorCriticNet,
    actor_obs_tensor: torch.Tensor,      # [1,C,H,W]
    central_obs_tensor: torch.Tensor,    # [1,N,C,H,W]
    agent: Any,
    env: GameField,
    deterministic: bool = False,
) -> Dict[str, Any]:
    """
    Uses policy_act.act(...) for macro/target/logp.
    Value is taken from CTDE central critic explicitly.
    """
    device = next(policy_act.parameters()).device
    actor_obs_tensor = actor_obs_tensor.to(device).float()
    central_obs_tensor = central_obs_tensor.to(device).float()

    out = policy_act.act(
        actor_obs_tensor,
        agent=agent,
        game_field=env,
        deterministic=deterministic,
    )

    def _grab_1d(key: str, alt: Optional[List[str]] = None) -> torch.Tensor:
        if alt is None:
            alt = []
        v = out.get(key, None)
        if v is None:
            for k in alt:
                v = out.get(k, None)
                if v is not None:
                    break
        if v is None:
            raise KeyError(f"policy.act() missing '{key}' (also tried {alt})")
        if not torch.is_tensor(v):
            v = torch.tensor(v, device=device)
        return v.reshape(-1)

    macro_action = _grab_1d("macro_action", ["macro"]).long()
    target_action = _grab_1d("target_action", ["target"]).long()
    logp = _grab_1d("log_prob", ["old_log_prob", "logp"]).float()

    central_value = policy_act.forward_central_critic(central_obs_tensor).reshape(-1).float()

    mm = out.get("macro_mask", None)
    mm_np = None
    if mm is not None:
        if torch.is_tensor(mm):
            mm_np = mm.detach().cpu().numpy().astype(np.bool_).reshape(-1)
        else:
            mm_np = np.array(mm, dtype=np.bool_).reshape(-1)
        if mm_np.shape != (N_MACROS,):
            mm_np = mm_np.reshape(-1)
            if mm_np.shape != (N_MACROS,):
                raise RuntimeError(f"policy.act macro_mask shape {mm_np.shape}, expected {(N_MACROS,)}")
        if not mm_np.any():
            mm_np[:] = True

    return {
        "macro_action": macro_action,     # [1]
        "target_action": target_action,   # [1]
        "old_log_prob": logp,             # [1]
        "value": central_value,           # [1] CTDE value
        "macro_mask": mm_np,              # np.bool_ [A] or None
    }


# ==========================================================
# ROLLOUT BUFFER
# ==========================================================
class MAPPORolloutBuffer:
    def __init__(self) -> None:
        self.actor_obs: List[np.ndarray] = []
        self.central_obs: List[np.ndarray] = []
        self.macro_actions: List[int] = []
        self.target_actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.next_values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.dts: List[float] = []
        self.traj_ids: List[int] = []
        self.macro_masks: List[np.ndarray] = []

    def add(
        self,
        actor_obs: np.ndarray,
        central_obs: np.ndarray,
        macro_action: int,
        target_action: int,
        log_prob: float,
        value: float,
        next_value: float,
        reward: float,
        done: bool,
        dt: float,
        traj_id: int,
        macro_mask: Optional[np.ndarray] = None,
    ) -> None:
        self.actor_obs.append(np.array(actor_obs, dtype=np.float32))
        self.central_obs.append(np.array(central_obs, dtype=np.float32))
        self.macro_actions.append(int(macro_action))
        self.target_actions.append(int(target_action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.next_values.append(float(next_value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.dts.append(float(dt))
        self.traj_ids.append(int(traj_id))

        if macro_mask is None:
            self.macro_masks.append(np.zeros((N_MACROS,), dtype=np.bool_))  # safe default (no mask)
            self.macro_masks[-1][:] = True
        else:
            mm = np.array(macro_mask, dtype=np.bool_).reshape(-1)
            if mm.shape != (N_MACROS,):
                raise ValueError(f"macro_mask must be shape {(N_MACROS,)}, got {mm.shape}")
            if not mm.any():
                mm[:] = True
            self.macro_masks.append(mm)

    def size(self) -> int:
        return len(self.actor_obs)

    def clear(self) -> None:
        self.__init__()

    def to_tensors(self, device: torch.device):
        actor_obs = torch.tensor(np.stack(self.actor_obs), dtype=torch.float32, device=device)          # [T,C,H,W]
        central_obs = torch.tensor(np.stack(self.central_obs), dtype=torch.float32, device=device)     # [T,N,C,H,W]
        macro_actions = torch.tensor(self.macro_actions, dtype=torch.long, device=device)              # [T]
        target_actions = torch.tensor(self.target_actions, dtype=torch.long, device=device)            # [T]
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)               # [T]
        values = torch.tensor(self.values, dtype=torch.float32, device=device)                          # [T]
        next_values = torch.tensor(self.next_values, dtype=torch.float32, device=device)               # [T]
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)                        # [T]
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)                            # [T]
        dts = torch.tensor(self.dts, dtype=torch.float32, device=device)                                # [T]
        traj_ids = torch.tensor(self.traj_ids, dtype=torch.long, device=device)                         # [T]
        macro_masks = torch.tensor(np.stack(self.macro_masks), dtype=torch.bool, device=device)         # [T,A]

        return (
            actor_obs, central_obs, macro_actions, target_actions,
            old_log_probs, values, next_values, rewards, dones, dts, traj_ids,
            macro_masks
        )


# ==========================================================
# CPU GAE (grouped by traj_id)
# ==========================================================
def normalize_advantages(adv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = adv.mean()
    var = (adv - mean).pow(2).mean()
    return (adv - mean) / torch.sqrt(var + eps)


def compute_gae_event_grouped_nextvalues_cpu(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    dts: np.ndarray,
    traj_ids: np.ndarray,
    gamma: float = GAMMA,
    lam: float = GAE_LAMBDA,
) -> Tuple[np.ndarray, np.ndarray]:
    T = rewards.shape[0]
    advantages = np.zeros((T,), dtype=np.float32)

    traj_to_idxs: Dict[int, List[int]] = {}
    for i in range(T):
        tid = int(traj_ids[i])
        traj_to_idxs.setdefault(tid, []).append(i)

    for _, idxs in traj_to_idxs.items():
        next_adv = 0.0
        for i in reversed(idxs):
            dt = float(dts[i])
            gamma_dt = gamma ** dt
            lam_gamma_dt = (gamma * lam) ** dt
            mask = 1.0 - float(dones[i])
            delta = float(rewards[i]) + gamma_dt * float(next_values[i]) * mask - float(values[i])
            advantages[i] = float(delta + lam_gamma_dt * next_adv * mask)
            next_adv = float(advantages[i])

    returns = advantages + values.astype(np.float32)
    return advantages.astype(np.float32), returns.astype(np.float32)


# ==========================================================
# MAPPO UPDATE (runs on train_device, usually CPU)
# ==========================================================
def _fix_all_false_rows(mask: torch.Tensor) -> torch.Tensor:
    if mask.numel() == 0:
        return mask
    row_sum = mask.sum(dim=1)
    bad = row_sum == 0
    if bad.any():
        mask = mask.clone()
        mask[bad, :] = True
    return mask


def mappo_update(
    policy_train: ActorCriticNet,
    optimizer: optim.Optimizer,
    buffer: MAPPORolloutBuffer,
    device: torch.device,
    ent_coef: float,
) -> None:
    (
        actor_obs, central_obs, macro_actions, target_actions,
        old_log_probs, values, next_values, rewards, dones, dts, traj_ids,
        macro_masks
    ) = buffer.to_tensors(device)

    T = actor_obs.size(0)
    if T == 0:
        buffer.clear()
        return

    adv_np, ret_np = compute_gae_event_grouped_nextvalues_cpu(
        rewards.detach().cpu().numpy(),
        values.detach().cpu().numpy(),
        next_values.detach().cpu().numpy(),
        dones.detach().cpu().numpy(),
        dts.detach().cpu().numpy(),
        traj_ids.detach().cpu().numpy(),
        gamma=GAMMA,
        lam=GAE_LAMBDA,
    )
    advantages = torch.tensor(adv_np, dtype=torch.float32, device=device)
    returns = torch.tensor(ret_np, dtype=torch.float32, device=device)
    advantages = normalize_advantages(advantages)

    policy_train.train()

    for _ in range(PPO_EPOCHS):
        perm_np = np.random.permutation(T).astype(np.int64)

        for start in range(0, T, MINIBATCH_SIZE):
            end = min(start + MINIBATCH_SIZE, T)
            idx_np = perm_np[start:end]
            mb_idx = torch.tensor(idx_np, dtype=torch.long, device=device)

            mb_actor    = actor_obs.index_select(0, mb_idx)
            mb_central  = central_obs.index_select(0, mb_idx)
            mb_macro    = macro_actions.index_select(0, mb_idx)
            mb_target   = target_actions.index_select(0, mb_idx)
            mb_old_logp = old_log_probs.index_select(0, mb_idx)
            mb_adv      = advantages.index_select(0, mb_idx)
            mb_ret      = returns.index_select(0, mb_idx)
            mb_mask     = _fix_all_false_rows(macro_masks.index_select(0, mb_idx))

            new_values = policy_train.forward_central_critic(mb_central).reshape(-1)

            new_logp, entropy, _ = policy_train.evaluate_actions(
                mb_actor, mb_macro, mb_target, macro_mask_batch=mb_mask
            )
            new_logp = new_logp.reshape(-1)
            entropy = entropy.reshape(-1)

            ratio = torch.exp(new_logp - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (mb_ret - new_values).pow(2).mean()
            loss = policy_loss + VALUE_COEF * value_loss - ent_coef * entropy.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy_train.parameters(), MAX_GRAD_NORM)
            optimizer.step()

    buffer.clear()


# ==========================================================
# MACRO USAGE TRACKER (stable per-slot printing)
# ==========================================================
def _macro_name(m: Any) -> str:
    try:
        return m.name
    except Exception:
        return str(m)


class MacroUsageTracker:
    def __init__(self) -> None:
        self.total: Dict[str, int] = {}
        self.by_agent: Dict[str, Dict[str, int]] = {}

    def reset(self) -> None:
        self.total.clear()
        self.by_agent.clear()

    def add(self, agent: Any, macro_enum: Any) -> None:
        sid = agent_slot(agent)
        mn = _macro_name(macro_enum)
        self.total[mn] = self.total.get(mn, 0) + 1
        self.by_agent.setdefault(sid, {})
        self.by_agent[sid][mn] = self.by_agent[sid].get(mn, 0) + 1

    def summary_lines(self) -> List[str]:
        lines: List[str] = []
        total_n = sum(self.total.values())
        if total_n <= 0:
            return ["(no macros recorded)"]

        items = sorted(self.total.items(), key=lambda kv: kv[1], reverse=True)
        parts = [f"{k}:{v} ({100.0*v/total_n:.1f}%)" for k, v in items]
        lines.append("Total: " + " | ".join(parts))

        for sid, dist in sorted(self.by_agent.items(), key=lambda kv: kv[0]):
            n = sum(dist.values())
            it = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
            parts = [f"{k}:{v} ({100.0*v/n:.1f}%)" for k, v in it]
            lines.append(f"  {sid}: " + " | ".join(parts))
        return lines


# ==========================================================
# MAIN LOOP
# ==========================================================
def train_mappo_event(total_steps: int = TOTAL_STEPS) -> None:
    set_seed(42)

    env = make_env()
    gm = env.getGameManager()

    if hasattr(gm, "set_shaping_gamma"):
        try:
            gm.set_shaping_gamma(GAMMA)
        except Exception:
            pass

    env.reset_default()
    if env.blue_agents:
        sample = env.build_observation(env.blue_agents[0])
        print(f"[train_mappo_event] Sample obs shape: C={len(sample)}, H={len(sample[0])}, W={len(sample[0][0])}")

    n_agents = len(getattr(env, "blue_agents", [])) or getattr(env, "agents_per_team", 2)

    def _make_policy():
        return ActorCriticNet(
            n_macros=N_MACROS,
            n_targets=env.num_macro_targets,
            n_agents=n_agents,
            in_channels=NUM_CNN_CHANNELS,
            height=CNN_ROWS,
            width=CNN_COLS,
        )

    act_device = prefer_device()
    # DirectML backward tends to be fragile; keep training on CPU when using privateuseone
    train_device = torch.device("cpu") if (HAS_TDML and "privateuseone" in str(act_device).lower()) else act_device

    print(f"[train_mappo_event] act_device:   {act_device}")
    print(f"[train_mappo_event] train_device: {train_device}")

    policy_train = _make_policy().to(train_device)
    policy_act = _make_policy().to(act_device)
    policy_act.load_state_dict(policy_train.state_dict())
    policy_act.eval()

    optimizer = optim.Adam(policy_train.parameters(), lr=LR, foreach=False)
    buffer = MAPPORolloutBuffer()

    global_step = 0
    episode_idx = 0

    blue_wins = red_wins = draws = 0
    phase_idx = 0
    phase_episode_count = 0
    phase_recent: List[int] = []

    ENABLE_SELFPLAY = False
    POLICY_SAMPLE_CHANCE = 0.15
    old_policies_buffer: Deque[Dict[str, Any]] = deque(maxlen=20)

    traj_id_counter = 0
    traj_id_map: Dict[Tuple[int, str], int] = {}

    macro_tracker = MacroUsageTracker()

    while global_step < total_steps:
        episode_idx += 1
        phase_episode_count += 1
        macro_tracker.reset()

        cur_phase = PHASE_SEQUENCE[phase_idx]
        phase_cfg = PHASE_CONFIG[cur_phase]
        gm.score_limit = int(phase_cfg["score_limit"])
        gm.max_time = float(phase_cfg["max_time"])
        max_steps = int(phase_cfg["max_macro_steps"])

        if hasattr(gm, "set_phase"):
            try:
                gm.set_phase(cur_phase)
            except Exception:
                pass

        opponent_tag = cur_phase
        if ENABLE_SELFPLAY and cur_phase == "OP3" and len(old_policies_buffer) > 0 and hasattr(env, "set_red_policy_neural"):
            if random.random() < POLICY_SAMPLE_CHANCE:
                red_net = _make_policy().to(torch.device("cpu"))
                red_net.load_state_dict(random.choice(old_policies_buffer))
                red_net.eval()
                env.set_red_policy_neural(red_net)
                opponent_tag = "SELFPLAY"
            else:
                set_red_policy_for_phase(env, cur_phase)
        else:
            set_red_policy_for_phase(env, cur_phase)

        # ---- Episode reset ----
        env.reset_default()
        clear_reward_events_best_effort(gm)

        # ✅ NEW: per-episode reward routing (this is the critical fix)
        episode_uid_set, pending_reward, last_buf_idx = init_episode_reward_routing(env)

        red_pol = env.policies.get("red")
        if hasattr(red_pol, "reset"):
            try:
                red_pol.reset()
            except Exception:
                pass

        done = False
        ep_return = 0.0
        steps = 0

        sim_time_prev = gm_get_sim_time_safe(gm)
        traj_id_map.clear()

        while (not done) and steps < max_steps and global_step < total_steps:
            blue_joint_obs = get_blue_team_obs_in_id_order(env)
            central_obs_np = np.stack(blue_joint_obs, axis=0)
            central_obs_tensor = torch.tensor(central_obs_np, dtype=torch.float32, device=act_device).unsqueeze(0)

            blue_agents_enabled = [a for a in env.blue_agents if a is not None and a.isEnabled()]
            decisions = []
            submit_actions: Dict[str, Tuple[int, int]] = {}

            for agent in blue_agents_enabled:
                actor_obs_np = np.array(env.build_observation(agent), dtype=np.float32)
                actor_obs_tensor = torch.tensor(actor_obs_np, dtype=torch.float32, device=act_device).unsqueeze(0)

                out = sample_mappo_action_via_act(
                    policy_act,
                    actor_obs_tensor,
                    central_obs_tensor,
                    agent=agent,
                    env=env,
                    deterministic=False,
                )

                macro_idx = int(out["macro_action"][0].item())
                target_idx = int(out["target_action"][0].item())
                logp = float(out["old_log_prob"][0].item())
                val = float(out["value"][0].item())
                mm_np = out.get("macro_mask", None)

                macro_enum = USED_MACROS[macro_idx]
                macro_tracker.add(agent, macro_enum)
                macro_val = int(getattr(macro_enum, "value", int(macro_enum)))

                # traj id per agent-slot per episode
                key = (episode_idx, agent_slot(agent))
                tid = traj_id_map.get(key)
                if tid is None:
                    traj_id_map[key] = traj_id_counter
                    tid = traj_id_counter
                    traj_id_counter += 1

                uid = agent_uid(agent)
                decisions.append((agent, uid, actor_obs_np, central_obs_np, macro_idx, target_idx, logp, val, tid, mm_np))

                for k in external_keys_for_agent(env, agent):
                    submit_actions[k] = (macro_val, target_idx)

            submit_external_actions_robust(env, submit_actions)

            sim_decision_window(env, gm, DECISION_WINDOW, SIM_DT)
            done = bool(gm.game_over)

            sim_time_now = gm_get_sim_time_safe(gm)
            dt = max(0.0, float(sim_time_now - sim_time_prev))
            sim_time_prev = sim_time_now

            rollout_done = bool(done or ((steps + 1) >= max_steps))

            accumulate_rewards_for_uid_set(gm, episode_uid_set, pending_reward)

            # Bootstrap V(s_{t+1}) from central critic (forward only)
            with torch.no_grad():
                next_joint_obs = get_blue_team_obs_in_id_order(env)
                next_central_np = np.stack(next_joint_obs, axis=0)
                next_central_tensor = torch.tensor(next_central_np, dtype=torch.float32, device=act_device).unsqueeze(0)
                bootstrap_value = float(
                    policy_act.forward_central_critic(next_central_tensor).detach().reshape(-1)[0].item()
                )
            if rollout_done:
                bootstrap_value = 0.0

            # Store transitions (consume pending ONLY for uids that acted)
            step_reward_sum = 0.0
            for agent, uid, actor_obs_np, central_obs_np, macro_idx, target_idx, logp, val, tid, mm_np in decisions:
                r = consume_pending_reward_for_uid(uid, pending_reward, TIME_PENALTY_PER_AGENT_PER_MACRO)
                step_reward_sum += r

                agent_dead = (not agent.isEnabled())
                agent_done = bool(rollout_done or agent_dead)
                nv = 0.0 if agent_done else bootstrap_value

                buffer.add(
                    actor_obs=actor_obs_np,
                    central_obs=central_obs_np,
                    macro_action=macro_idx,
                    target_action=target_idx,
                    log_prob=logp,
                    value=val,
                    next_value=nv,
                    reward=r,
                    done=agent_done,
                    dt=dt,
                    traj_id=tid,
                    macro_mask=mm_np,
                )
                global_step += 1
                last_buf_idx[uid] = buffer.size() - 1

            ep_return += step_reward_sum
            steps += 1

            # Update (train_device), then sync → act model
            if buffer.size() >= UPDATE_EVERY:
                # ✅ NEW: flush pending into buffer BEFORE update clears buffer
                flush_pending_rewards_into_buffer(pending_reward, last_buf_idx, buffer)

                ent = get_entropy_coef(cur_phase, phase_episode_count)
                print(f"[MAPPO UPDATE] step={global_step} episode={episode_idx} phase={cur_phase} ENT={ent:.4f} Opp={opponent_tag}")

                mappo_update(policy_train, optimizer, buffer, train_device, ent)

                policy_act.load_state_dict(policy_train.state_dict())
                policy_act.eval()

                if ENABLE_SELFPLAY:
                    old_policies_buffer.append(copy.deepcopy(policy_train.state_dict()))

                # buffer cleared inside update => indices invalid now
                last_buf_idx.clear()

        # Episode end flush (if any rewards still pending, attach to last transitions)
        flush_pending_rewards_into_buffer(pending_reward, last_buf_idx, buffer)

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
            f"[{episode_idx:5d}] {result:8} | StepR {avg_step_r:+.3f} TermR {ep_return:+.1f} | "
            f"Score {gm.blue_score}:{gm.red_score} | BlueWins: {blue_wins} RedWins: {red_wins} Draws: {draws} | "
            f"PhaseWin {phase_wr*100:.1f}% | Phase={cur_phase} Opp={opponent_tag}"
        )

        if (episode_idx % MACRO_STATS_PRINT_EVERY == 0) or (MACRO_STATS_PRINT_ON_WIN and result == "BLUE WIN"):
            print("[MACROS] " + f"episode={episode_idx} phase={cur_phase} result={result}")
            for line in macro_tracker.summary_lines():
                print("  " + line)

        # Curriculum advance
        if cur_phase != PHASE_SEQUENCE[-1]:
            min_eps = MIN_PHASE_EPISODES[cur_phase]
            target_wr = TARGET_PHASE_WINRATE[cur_phase]
            if phase_episode_count >= min_eps and len(phase_recent) >= PHASE_WINRATE_WINDOW and phase_wr >= target_wr:
                print(f"[CURRICULUM] Advancing from {cur_phase} -> next phase (MAPPO).")
                phase_idx += 1
                phase_episode_count = 0
                phase_recent.clear()

        # Periodic checkpoint
        if episode_idx % 50 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"mappo_ckpt_ep{episode_idx}.pth")
            torch.save(policy_train.state_dict(), ckpt_path)
            print(f"[CKPT] Saved: {ckpt_path}")

    final_path = os.path.join(CHECKPOINT_DIR, "research_mappo_model1.pth")
    torch.save(policy_train.state_dict(), final_path)
    print(f"\n[MAPPO] Training complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    train_mappo_event()
