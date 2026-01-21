"""
MAPPO / CTDE training with self-play opponent pool (PFSP) and curriculum logging.
"""

import os
import random
import copy
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Deque, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# DEVICE SELECTION (DirectML → CUDA → CPU)
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


# IMPORTS (your project)
from game_field import GameField, CNN_ROWS, CNN_COLS, NUM_CNN_CHANNELS
from game_manager import GameManager
from macro_actions import MacroAction
from rl_policy import ActorCriticNet
from policies import OP1RedPolicy, OP2RedPolicy, OP3RedPolicy
from map_registry import make_game_field
from behavior_logging import BehaviorLogger, append_records_csv
from config import (
    MAP_NAME as DEFAULT_MAP_NAME,
    MAP_PATH as DEFAULT_MAP_PATH,
    LOG_DIR as DEFAULT_LOG_DIR,
    PPO_PHASE_MIN_EPISODES,
    PPO_PHASE_ELO_MARGIN,
    PPO_PHASE_REQUIRED_WIN_BY,
    PPO_SWITCH_TO_ELO_AFTER_OP3,
)

# CONFIG
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


def elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(r_a - r_b) / 400.0))


GRID_ROWS = CNN_ROWS
GRID_COLS = CNN_COLS

# Map selection
MAP_NAME = DEFAULT_MAP_NAME
MAP_PATH = DEFAULT_MAP_PATH

TOTAL_STEPS = 1_000_000
UPDATE_EVERY = 2_048  # transitions (per-agent) in buffer
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

# Elo (match PPO league style)
ELO_K_FACTOR = 32.0
ELO_MATCH_TAU = 250.0
LEAGUE_SCRIPTED_MIX_FLOOR = 0.10

PHASE_SEQUENCE = ["OP1", "OP2", "OP3"]
MIN_PHASE_EPISODES = {"OP1": 500, "OP2": 1000, "OP3": 2000}
TARGET_PHASE_WINRATE = {"OP1": 0.90, "OP2": 0.86, "OP3": 0.80}
PHASE_WINRATE_WINDOW = 50

PHASE_CONFIG = {
    "OP1": dict(score_limit=3, max_time=200.0, max_macro_steps=500),
    "OP2": dict(score_limit=3, max_time=200.0, max_macro_steps=500),
    "OP3": dict(score_limit=3, max_time=200.0, max_macro_steps=500),
}

USED_MACROS = []  # synced from env.macro_order inside train_mappo_event()
N_MACROS = 0

MACRO_STATS_PRINT_EVERY = 10
MACRO_STATS_PRINT_ON_WIN = True
DEBUG_PRINT_UIDS_ONCE = True
REWARD_DEBUG_FIRST_EPISODES = 3

# OP2/OP3 DRAW-BREAKER SHAPING (phase-specific)
ENT_SCHEDULE = {
    "OP1": (0.07, 0.05, 800.0),
    "OP2": (0.04, 0.018, 1200.0),
    "OP3": (0.03, 0.012, 2000.0),
}

TIME_PENALTY_BY_PHASE = {"OP1": 0.0010, "OP2": 0.0004, "OP3": 0.0004}
SCORE_DELTA_REWARD_BY_PHASE = {"OP1": 25.0, "OP2": 18.0, "OP3": 22.0}

TERMINAL_WIN_BONUS_BY_PHASE = {"OP1": 25.0, "OP2": 25.0, "OP3": 25.0}
TERMINAL_LOSS_PENALTY_BY_PHASE = {"OP1": -25.0, "OP2": -25.0, "OP3": -25.0}
TERMINAL_DRAW_PENALTY_BY_PHASE = {"OP1": -12.0, "OP2": -14.0, "OP3": -10.0}

NO_SCORE_PENALTY_PER_WINDOW_BY_PHASE = {
    "OP1": 0.030,
    "OP2": 0.015,
    "OP3": 0.010,
}
NO_SCORE_GRACE_WINDOWS_BY_PHASE = {
    "OP1": 6,
    "OP2": 12,
    "OP3": 16,
}

GET_FLAG_BONUS_BY_PHASE = {
    "OP1": 1.0,
    "OP2": 0.0,
    "OP3": 0.0,
}


def get_entropy_coef(cur_phase: str, phase_episode_count: int) -> float:
    start, end, horizon = ENT_SCHEDULE[cur_phase]
    frac = min(1.0, float(phase_episode_count) / float(horizon))
    return float(start + (end - start) * frac)


def get_phase_time_penalty(cur_phase: str) -> float:
    return float(TIME_PENALTY_BY_PHASE.get(cur_phase, 0.001))


def get_phase_score_delta_reward(cur_phase: str) -> float:
    return float(SCORE_DELTA_REWARD_BY_PHASE.get(cur_phase, 15.0))


def get_terminal_team_reward(cur_phase: str, blue_score: int, red_score: int) -> float:
    if blue_score > red_score:
        return float(TERMINAL_WIN_BONUS_BY_PHASE[cur_phase])
    if blue_score < red_score:
        return float(TERMINAL_LOSS_PENALTY_BY_PHASE[cur_phase])
    return float(TERMINAL_DRAW_PENALTY_BY_PHASE[cur_phase])


def get_no_score_penalty(cur_phase: str, windows_since_score: int) -> float:
    grace = int(NO_SCORE_GRACE_WINDOWS_BY_PHASE.get(cur_phase, 999999))
    per = float(NO_SCORE_PENALTY_PER_WINDOW_BY_PHASE.get(cur_phase, 0.0))
    if per <= 0.0:
        return 0.0
    if windows_since_score <= grace:
        return 0.0
    ramp = min(6, max(1, windows_since_score - grace))
    return float(per * ramp)


def get_get_flag_bonus(cur_phase: str) -> float:
    return float(GET_FLAG_BONUS_BY_PHASE.get(cur_phase, 0.0))


# SELF-PLAY CONFIG (Modern-ish MARL defaults)
ENABLE_SELFPLAY = True

# Mix scripted vs neural opponents per phase (start more scripted, end more neural)
SCRIPTED_MIX_PROB_BY_PHASE = {
    "OP1": 1.00,  # pure scripted early
    "OP2": 0.55,  # mix
    "OP3": 0.25,  # mostly self-play
}

# When neural opponent is chosen: PFSP samples from pool
PFSP_WINDOW = 60           # per-opponent rolling results window
PFSP_ALPHA = 2.0           # higher => stronger preference for ~50% opponents
POOL_MAX = 40              # max historical snapshots kept
SNAPSHOT_EVERY_UPDATES = 1 # snapshot cadence on each MAPPO update
WARMUP_UPDATES = 2         # require at least this many updates before neural opponents are allowed

# Opponent hold to reduce non-stationarity (same opponent for K episodes)
OPPONENT_HOLD_EPISODES = 4

# Probability to use "latest" snapshot (hardens vs recent versions)
LATEST_BIAS_PROB = 0.25

# Opponent stochasticity
RED_DETERMINISTIC = False  # typical: stochastic opponents reduce overfitting

# ENV HELPERS
def make_env() -> GameField:
    env = make_game_field(
        map_name=MAP_NAME or None,
        map_path=MAP_PATH or None,
        rows=GRID_ROWS,
        cols=GRID_COLS,
    )
    if hasattr(env, "set_seed"):
        try:
            env.set_seed(42)
        except Exception:
            pass
    env.use_internal_policies = True

    # We drive BOTH sides via external actions, so env won't internally decide.
    if hasattr(env, "set_external_control"):
        try:
            env.set_external_control("blue", True)
            env.set_external_control("red", True)
        except Exception:
            pass

    if hasattr(env, "external_missing_action_mode"):
        try:
            env.external_missing_action_mode = "idle"
        except Exception:
            pass
    return env


@dataclass
class RedCurriculumScheduler:
    scripted_episodes: int = 300
    pfsp_episodes: int = 900
    mode: str = "scripted"  # scripted -> pfsp -> latest

    def update_mode(self, episode_idx: int) -> str:
        if episode_idx <= self.scripted_episodes:
            self.mode = "scripted"
        elif episode_idx <= (self.scripted_episodes + self.pfsp_episodes):
            self.mode = "pfsp"
        else:
            self.mode = "latest"
        return self.mode


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
        k = str(k).strip()
        if k and k not in seen:
            out.append(k)
            seen.add(k)
    return out


def agent_uid(agent: Any) -> str:
    uid = getattr(agent, "unique_id", None)
    if uid is None or str(uid).strip() == "":
        uid = getattr(agent, "slot_id", None)
    if uid is None or str(uid).strip() == "":
        uid = f"{getattr(agent,'side','blue')}_{getattr(agent,'agent_id',0)}"
    return str(uid)


def agent_slot(agent: Any) -> str:
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


def submit_external_actions_robust(env: GameField, action_dict: Dict[str, Tuple[int, Any]]) -> None:
    # clear old
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


def get_team_obs_in_id_order(env: GameField, side: str) -> List[np.ndarray]:
    agents = list(env.blue_agents if side == "blue" else env.red_agents)
    agents.sort(key=lambda a: getattr(a, "agent_id", 0))
    out: List[np.ndarray] = []
    for a in agents:
        if a is None or (not a.isEnabled()):
            out.append(zero_obs_like())
        else:
            out.append(np.array(env.build_observation(a), dtype=np.float32))
    return out


def get_blue_team_obs_in_id_order(env: GameField) -> List[np.ndarray]:
    return get_team_obs_in_id_order(env, "blue")


def macro_enum_from_any(x: Any) -> Optional[MacroAction]:
    if isinstance(x, MacroAction):
        return x
    try:
        xi = int(getattr(x, "value", x))
        return MacroAction(xi)
    except Exception:
        return None


def macro_idx_from_enum(m: MacroAction) -> int:
    # macro index in USED_MACROS order
    for i, mm in enumerate(USED_MACROS):
        if mm == m:
            return i
    return 0


def nearest_target_idx(env: GameField, cell: Tuple[int, int]) -> int:
    # find nearest macro target (stable list in env)
    try:
        targets = env.get_all_macro_targets() if hasattr(env, "get_all_macro_targets") else [env.get_macro_target(i) for i in range(env.num_macro_targets)]
    except Exception:
        targets = [env.get_macro_target(i) for i in range(env.num_macro_targets)]
    cx, cy = int(cell[0]), int(cell[1])
    best_i, best_d2 = 0, 10**9
    for i, (tx, ty) in enumerate(targets):
        dx, dy = int(tx) - cx, int(ty) - cy
        d2 = dx*dx + dy*dy
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return int(best_i)


# REWARDS (EPISODE-STRICT UID SET + PENDING ACCUMULATOR)
def gm_pop_reward_events_safe(gm: GameManager):
    events = []
    if hasattr(gm, "pop_reward_events") and callable(getattr(gm, "pop_reward_events")):
        try:
            events = gm.pop_reward_events()
        except Exception:
            events = []
    else:
        ev = getattr(gm, "reward_events", [])
        try:
            events = list(ev)
            if isinstance(ev, list):
                ev.clear()
        except Exception:
            events = []

    parsed = []
    for e in (events or []):
        if isinstance(e, dict):
            aid = e.get("agent_id", None)
            r = e.get("reward", e.get("r", None))
            t = e.get("t", 0.0)
            parsed.append((t, aid, r))
            continue
        if isinstance(e, (tuple, list)):
            if len(e) == 3:
                t, aid, r = e
                parsed.append((t, aid, r))
                continue
            if len(e) == 2:
                aid, r = e
                parsed.append((0.0, aid, r))
                continue
        parsed.append((0.0, None, None))
    return parsed


def clear_reward_events_best_effort(gm: GameManager) -> None:
    try:
        _ = gm_pop_reward_events_safe(gm)
    except Exception:
        pass


def init_episode_reward_routing(env: GameField) -> Tuple[Set[str], Dict[str, float], Dict[str, int]]:
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
    debug_print: bool = False,
) -> None:
    events = gm_pop_reward_events_safe(gm)

    for _t, agent_id, r in events:
        if agent_id is None:
            continue
        key = str(agent_id).strip()
        if not key or r is None:
            continue
        if key in allowed_uids:
            pending[key] = pending.get(key, 0.0) + float(r)


def add_team_reward_to_pending(allowed_uids: Set[str], pending: Dict[str, float], r_team: float) -> None:
    if abs(r_team) <= 1e-12:
        return
    for uid in allowed_uids:
        pending[uid] = pending.get(uid, 0.0) + float(r_team)


def consume_pending_reward_for_uid(uid: str, pending: Dict[str, float], time_penalty: float = 0.0) -> float:
    r = float(pending.get(uid, 0.0))
    pending[uid] = 0.0
    return r - float(time_penalty)


def flush_pending_rewards_into_buffer(
    pending: Dict[str, float],
    last_buf_idx: Dict[str, int],
    buffer: "MAPPORolloutBuffer",
) -> None:
    for uid, r in list(pending.items()):
        if abs(r) <= 1e-12:
            continue
        idx = last_buf_idx.get(uid, None)
        if idx is None:
            pending[uid] = 0.0
            continue
        buffer.rewards[idx] = float(buffer.rewards[idx]) + float(r)
        pending[uid] = 0.0


# ACTION SAMPLING (policy_act forward only)
@torch.no_grad()
def sample_mappo_action_via_act(
    policy_act: ActorCriticNet,
    actor_obs_tensor: torch.Tensor,   # [1,C,H,W]
    central_obs_tensor: torch.Tensor, # [1,N,C,H,W]
    agent: Any,
    env: GameField,
    deterministic: bool = False,
) -> Dict[str, Any]:
    device = next(policy_act.parameters()).device
    actor_obs_tensor = actor_obs_tensor.to(device).float()
    central_obs_tensor = central_obs_tensor.to(device).float()

    out = policy_act.act(
        actor_obs_tensor,
        agent=agent,
        game_field=env,
        deterministic=deterministic,
    )

    def _grab_1d(key: str, alts: Optional[List[str]] = None) -> torch.Tensor:
        if alts is None:
            alts = []
        v = out.get(key, None)
        if v is None:
            for k in alts:
                v = out.get(k, None)
                if v is not None:
                    break
        if v is None:
            raise KeyError(f"policy.act() missing '{key}' (also tried {alts})")
        if not torch.is_tensor(v):
            v = torch.tensor(v, device=device)
        return v.reshape(-1)

    macro_action = _grab_1d("macro_action", ["macro"]).long()
    target_action = _grab_1d("target_action", ["target"]).long()
    logp = _grab_1d("log_prob", ["old_log_prob", "logp"]).float()

    v_all = policy_act.forward_central_critic(central_obs_tensor).detach().reshape(-1).float()
    if v_all.numel() == 1:
        v_pick = v_all
    else:
        idx = int(getattr(agent, "agent_id", 0))
        idx = max(0, min(idx, int(v_all.numel() - 1)))
        v_pick = v_all[idx:idx + 1]

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
        "macro_action": macro_action,      # [1]
        "target_action": target_action,    # [1]
        "old_log_prob": logp,              # [1]
        "value": v_pick,                   # [1]
        "macro_mask": mm_np,               # np.bool_ [A] or None
    }


@torch.no_grad()
def sample_opponent_action(
    opp_policy_act: ActorCriticNet,
    obs_np: np.ndarray,   # [C,H,W]
    agent: Any,
    env: GameField,
    deterministic: bool,
) -> Tuple[int, int]:
    device = next(opp_policy_act.parameters()).device
    obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
    out = opp_policy_act.act(obs_tensor, agent=agent, game_field=env, deterministic=deterministic)

    def _to_int(v: Any) -> int:
        if torch.is_tensor(v):
            return int(v.reshape(-1)[0].item())
        return int(v)

    if isinstance(out, dict):
        mi = _to_int(out.get("macro_action", 0))
        ti = _to_int(out.get("target_action", 0))
        return int(mi), int(ti)
    if isinstance(out, (tuple, list)):
        mi = int(out[0]) if len(out) > 0 else 0
        ti = int(out[1]) if len(out) > 1 else 0
        return int(mi), int(ti)
    return int(out), 0


def scripted_red_action(
    red_policy: Any,
    agent: Any,
    env: GameField,
) -> Tuple[int, Any]:
    """
    Returns (macro_val, target_param) for env external action dictionary.
    We support either:
      - policy.select_action(obs, agent, env) -> (macro_any, param_any)
    where param_any may be None, (x,y), or already a target index.

    We convert:
      - macro_any -> MacroAction -> macro_val (enum .value)
      - param_any:
          None -> 0
          (x,y) -> nearest target idx
          int -> used as target idx
    """
    obs = env.build_observation(agent)
    if hasattr(red_policy, "select_action"):
        macro_any, param_any = red_policy.select_action(obs, agent, env)
    else:
        # fallback callable: red_policy(obs, agent, env)
        out = red_policy(obs, agent, env)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            macro_any, param_any = out[0], out[1]
        else:
            macro_any, param_any = out, None

    m = macro_enum_from_any(macro_any)
    if m is None:
        m = MacroAction.GO_TO

    macro_val = int(getattr(m, "value", int(m)))

    if param_any is None:
        target_param = 0
    elif isinstance(param_any, (tuple, list)) and len(param_any) == 2:
        target_param = nearest_target_idx(env, (int(param_any[0]), int(param_any[1])))
    else:
        try:
            target_param = int(param_any)
        except Exception:
            target_param = 0

    return macro_val, int(target_param)


# ROLLOUT BUFFER
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
            self.macro_masks.append(np.ones((N_MACROS,), dtype=np.bool_))
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
        actor_obs = torch.tensor(np.stack(self.actor_obs), dtype=torch.float32, device=device)   # [T,C,H,W]
        central_obs = torch.tensor(np.stack(self.central_obs), dtype=torch.float32, device=device) # [T,N,C,H,W]
        macro_actions = torch.tensor(self.macro_actions, dtype=torch.long, device=device)       # [T]
        target_actions = torch.tensor(self.target_actions, dtype=torch.long, device=device)     # [T]
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)        # [T]
        values = torch.tensor(self.values, dtype=torch.float32, device=device)                  # [T]
        next_values = torch.tensor(self.next_values, dtype=torch.float32, device=device)        # [T]
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)                # [T]
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)                    # [T]
        dts = torch.tensor(self.dts, dtype=torch.float32, device=device)                        # [T]
        traj_ids = torch.tensor(self.traj_ids, dtype=torch.long, device=device)                 # [T]
        macro_masks = torch.tensor(np.stack(self.macro_masks), dtype=torch.bool, device=device) # [T,A]
        return (
            actor_obs, central_obs, macro_actions, target_actions, old_log_probs,
            values, next_values, rewards, dones, dts, traj_ids, macro_masks
        )


# GAE (grouped by traj_id, event-driven dt)
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


# MAPPO UPDATE (train_device, usually CPU)
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
        actor_obs, central_obs, macro_actions, target_actions, old_log_probs,
        values, next_values, rewards, dones, dts, traj_ids, macro_masks
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

            mb_actor = actor_obs.index_select(0, mb_idx)
            mb_central = central_obs.index_select(0, mb_idx)
            mb_macro = macro_actions.index_select(0, mb_idx)
            mb_target = target_actions.index_select(0, mb_idx)
            mb_old_logp = old_log_probs.index_select(0, mb_idx)
            mb_adv = advantages.index_select(0, mb_idx)
            mb_ret = returns.index_select(0, mb_idx)
            mb_mask = _fix_all_false_rows(macro_masks.index_select(0, mb_idx))

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


# MACRO USAGE TRACKER (stable per-slot printing)
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
            lines.append(f"{sid}: " + " | ".join(parts))
        return lines


# SELF-PLAY OPPONENT POOL (PFSP)
@dataclass
class OpponentEntry:
    oid: str
    state_dict: Optional[Dict[str, Any]] = None  # for neural opponents
    kind: str = "neural"                         # "neural" or "scripted"
    scripted_tag: Optional[str] = None           # "OP1"/"OP2"/"OP3"
    results: Deque[int] = None                   # 1=blue win, 0=otherwise
    created_update: int = 0
    rating: float = 1200.0

    def __post_init__(self):
        if self.results is None:
            self.results = deque(maxlen=PFSP_WINDOW)

    def winrate(self) -> float:
        if not self.results:
            return 0.5
        return float(sum(self.results)) / float(len(self.results))


class OpponentPool:
    def __init__(self, max_size: int = POOL_MAX):
        self.max_size = int(max_size)
        self.entries: Dict[str, OpponentEntry] = {}
        self.neural_ids: List[str] = []
        self.latest_neural_id: Optional[str] = None
        self.learner_rating: float = 1200.0
        self.tau = float(ELO_MATCH_TAU)

    def add_scripted_defaults(self):
        # keep stable scripted ids
        for tag in ("OP1", "OP2", "OP3"):
            oid = f"scripted:{tag}"
            if oid not in self.entries:
                self.entries[oid] = OpponentEntry(oid=oid, kind="scripted", scripted_tag=tag, rating=1200.0)

    def add_snapshot(self, state_dict: Dict[str, Any], update_idx: int, suffix: str = "") -> str:
        oid = f"neural:u{update_idx}{suffix}"
        self.entries[oid] = OpponentEntry(
            oid=oid,
            kind="neural",
            state_dict=copy.deepcopy(state_dict),
            created_update=int(update_idx),
            rating=float(self.learner_rating),
        )
        self.neural_ids.append(oid)
        self.latest_neural_id = oid

        # trim pool (keep latest always)
        self._trim_neural()
        return oid

    def _trim_neural(self):
        if len(self.neural_ids) <= self.max_size:
            return
        latest = self.latest_neural_id
        # drop oldest first, but never drop latest
        keep: List[str] = []
        dropped: List[str] = []
        for oid in self.neural_ids:
            if oid == latest:
                keep.append(oid)
                continue
            if len(self.neural_ids) - len(dropped) > self.max_size:
                dropped.append(oid)
            else:
                keep.append(oid)
        for oid in dropped:
            self.entries.pop(oid, None)
        self.neural_ids = keep

    def record_result(self, opponent_id: str, blue_win: bool):
        ent = self.entries.get(opponent_id, None)
        if ent is None:
            return
        ent.results.append(1 if blue_win else 0)

    def get_rating(self, opponent_id: str) -> float:
        ent = self.entries.get(opponent_id)
        return float(ent.rating) if ent is not None else 1200.0

    def update_elo(self, opponent_id: str, actual: float) -> None:
        ent = self.entries.get(opponent_id)
        if ent is None:
            return
        lr = float(self.learner_rating)
        opp_r = float(ent.rating)
        exp = elo_expected(lr, opp_r)
        learner_new = lr + float(ELO_K_FACTOR) * (float(actual) - exp)
        opp_new = opp_r + float(ELO_K_FACTOR) * ((1.0 - float(actual)) - (1.0 - exp))
        self.learner_rating = float(learner_new)
        ent.rating = float(opp_new)

    def sample_by_elo(self) -> OpponentEntry:
        if not self.neural_ids:
            return self.entries["scripted:OP3"]
        lr = float(self.learner_rating)
        weights = []
        for oid in self.neural_ids:
            r = float(self.entries[oid].rating)
            dist = abs(r - lr)
            w =   math.exp(-dist / max(1e-6, self.tau)) + 1e-3
            weights.append(w)
        total = sum(weights)
        if total <= 0:
            return self.entries[self.neural_ids[-1]]
        pick = random.random() * total
        acc = 0.0
        for oid, w in zip(self.neural_ids, weights):
            acc += w
            if acc >= pick:
                return self.entries[oid]
        return self.entries[self.neural_ids[-1]]

    def sample_league(self, scripted_mix_floor: float = LEAGUE_SCRIPTED_MIX_FLOOR) -> OpponentEntry:
        if random.random() < float(scripted_mix_floor):
            return self.entries["scripted:OP3"]
        return self.sample_by_elo()

    def can_use_neural(self, updates_done: int) -> bool:
        return (updates_done >= WARMUP_UPDATES) and (len(self.neural_ids) > 0)

    def sample_opponent(self, phase: str, updates_done: int) -> OpponentEntry:
        phase = str(phase).upper()

        # Scripted mix by phase
        p_scripted = float(SCRIPTED_MIX_PROB_BY_PHASE.get(phase, 0.5))
        if (not self.can_use_neural(updates_done)) or (random.random() < p_scripted):
            oid = f"scripted:{phase if phase in ('OP1','OP2','OP3') else 'OP3'}"
            return self.entries[oid]

        # Neural: optionally bias toward latest
        if self.latest_neural_id is not None and random.random() < LATEST_BIAS_PROB:
            return self.entries[self.latest_neural_id]

        # PFSP weights: prefer opponents with winrate near 0.5
        ids = list(self.neural_ids)
        if not ids:
            return self.entries[f"scripted:{phase}"]

        wrs = np.array([self.entries[oid].winrate() for oid in ids], dtype=np.float32)
        # score = 1 - |wr-0.5|*2  => 1 at 0.5, 0 at 0 or 1
        scores = 1.0 - np.minimum(1.0, np.abs(wrs - 0.5) * 2.0)
        scores = np.maximum(scores, 1e-3)
        weights = scores ** float(PFSP_ALPHA)
        weights = weights / np.sum(weights)

        pick = np.random.choice(np.arange(len(ids)), p=weights)
        return self.entries[ids[int(pick)]]


# MAIN LOOP
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

    # Sync macros from env to avoid order mismatch
    global USED_MACROS, N_MACROS
    USED_MACROS = list(getattr(env, "macro_order", []))
    if not USED_MACROS:
        raise RuntimeError("env.macro_order missing or empty (cannot align MAPPO macros).")
    N_MACROS = int(len(USED_MACROS))

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
    train_device = torch.device("cpu") if (HAS_TDML and "privateuseone" in str(act_device).lower()) else act_device

    print(f"[train_mappo_event] act_device:   {act_device}")
    print(f"[train_mappo_event] train_device: {train_device}")

    policy_train = _make_policy().to(train_device)
    policy_act = _make_policy().to(act_device)
    policy_act.load_state_dict(policy_train.state_dict())
    policy_act.eval()

    # Dedicated opponent network (forward-only)
    opp_policy_act = _make_policy().to(act_device)
    opp_policy_act.eval()

    optimizer = optim.Adam(policy_train.parameters(), lr=LR, foreach=False)
    buffer = MAPPORolloutBuffer()

    # Self-play pool
    pool = OpponentPool(max_size=POOL_MAX)
    pool.add_scripted_defaults()

    # Behavior logging
    behavior_logger = BehaviorLogger()
    behavior_csv_path = os.path.join(DEFAULT_LOG_DIR, "behavior_mappo.csv")
    last_written_idx = 0

    # Curriculum/league mode (PPO-style)
    league_mode = False

    global_step = 0
    episode_idx = 0
    blue_wins = red_wins = draws = 0

    phase_idx = 0
    phase_episode_count = 0
    phase_recent: List[int] = []

    traj_id_counter = 0
    traj_id_map: Dict[Tuple[int, str], int] = {}

    macro_tracker = MacroUsageTracker()
    debug_uids = DEBUG_PRINT_UIDS_ONCE

    # hold opponent for K episodes
    hold_left = 0
    held_opp: Optional[OpponentEntry] = None

    # update counter (for snapshot ids)
    updates_done = 0
    snapshot_tick = 0

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

        # --- pick opponent (PPO-style curriculum then Elo league) ---
        cur_mode = "league_elo" if league_mode else "curriculum_scripted"
        if (not ENABLE_SELFPLAY) or hold_left <= 0 or held_opp is None:
            if not league_mode:
                held_opp = pool.entries[f"scripted:{cur_phase}"]
            else:
                held_opp = pool.sample_league()
            hold_left = int(OPPONENT_HOLD_EPISODES)
        hold_left -= 1

        opponent_tag = held_opp.oid

        # ---- Episode reset ----
        behavior_logger.start_episode()
        env.reset_default()
        clear_reward_events_best_effort(gm)

        # per-episode reward routing (BLUE ONLY)
        episode_uid_set, pending_reward, last_buf_idx = init_episode_reward_routing(env)
        if debug_uids:
            debug_uids = False
            print(f"[UIDS] episode_uid_set={sorted(list(episode_uid_set))}")

        # scripted policy instance for red, if needed
        red_scripted = None
        if held_opp.kind == "scripted":
            if held_opp.scripted_tag == "OP1":
                red_scripted = OP1RedPolicy("red")
            elif held_opp.scripted_tag == "OP2":
                red_scripted = OP2RedPolicy("red")
            else:
                red_scripted = OP3RedPolicy("red")
            if hasattr(red_scripted, "reset"):
                try:
                    red_scripted.reset()
                except Exception:
                    pass
        else:
            # neural: load snapshot into opp_policy_act
            assert held_opp.state_dict is not None
            opp_policy_act.load_state_dict(held_opp.state_dict)
            opp_policy_act.eval()

        done = False
        ep_return = 0.0
        ep_terminal_only = 0.0
        steps = 0

        sim_time_prev = gm_get_sim_time_safe(gm)
        traj_id_map.clear()

        prev_blue_score = int(getattr(gm, "blue_score", 0))
        prev_red_score = int(getattr(gm, "red_score", 0))
        windows_since_score = 0

        while (not done) and steps < max_steps and global_step < total_steps:
            # Central obs for BLUE team (fixed agent_id order)
            blue_joint_obs = get_blue_team_obs_in_id_order(env)
            central_obs_np = np.stack(blue_joint_obs, axis=0)  # [N,C,H,W]
            central_obs_tensor = torch.tensor(central_obs_np, dtype=torch.float32, device=act_device).unsqueeze(0)

            blue_agents_enabled = [a for a in env.blue_agents if a is not None and a.isEnabled()]
            red_agents_enabled = [a for a in env.red_agents if a is not None and a.isEnabled()]

            decisions = []
            submit_actions: Dict[str, Tuple[int, Any]] = {}

            # -------- BLUE actions (learner) --------
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
                behavior_logger.log_decision(agent, macro_val)

                key = (episode_idx, agent_slot(agent))
                tid = traj_id_map.get(key)
                if tid is None:
                    traj_id_map[key] = traj_id_counter
                    tid = traj_id_counter
                    traj_id_counter += 1

                uid = agent_uid(agent)
                if macro_enum == MacroAction.GET_FLAG:
                    bonus = get_get_flag_bonus(cur_phase)
                    if bonus > 0.0 and uid in pending_reward:
                        pending_reward[uid] = pending_reward.get(uid, 0.0) + float(bonus)
                decisions.append((agent, uid, actor_obs_np, central_obs_np, macro_idx, target_idx, logp, val, tid, mm_np))

                for k in external_keys_for_agent(env, agent):
                    submit_actions[k] = (macro_val, target_idx)

            # -------- RED actions (opponent) --------
            if red_agents_enabled:
                if held_opp.kind == "scripted":
                    # drive scripted red externally (so control is consistent)
                    for agent in red_agents_enabled:
                        macro_val, target_param = scripted_red_action(red_scripted, agent, env)
                        # (optional) track macros for debugging
                        m = macro_enum_from_any(macro_val) or MacroAction.GO_TO
                        macro_tracker.add(agent, m)
                        behavior_logger.log_decision(agent, int(getattr(m, "value", int(m))))
                        for k in external_keys_for_agent(env, agent):
                            submit_actions[k] = (macro_val, target_param)
                else:
                    # neural opponent snapshot
                    for agent in red_agents_enabled:
                        obs_np = np.array(env.build_observation(agent), dtype=np.float32)
                        mi, ti = sample_opponent_action(
                            opp_policy_act,
                            obs_np,
                            agent=agent,
                            env=env,
                            deterministic=bool(RED_DETERMINISTIC),
                        )
                        mi = int(mi) % int(N_MACROS)
                        ti = int(ti) % int(env.num_macro_targets)
                        m = USED_MACROS[mi]
                        macro_val = int(getattr(m, "value", int(m)))
                        macro_tracker.add(agent, m)
                        behavior_logger.log_decision(agent, macro_val)
                        for k in external_keys_for_agent(env, agent):
                            submit_actions[k] = (macro_val, ti)

            # Submit + simulate window
            submit_external_actions_robust(env, submit_actions)
            sim_decision_window(env, gm, DECISION_WINDOW, SIM_DT)

            done = bool(getattr(gm, "game_over", False))
            sim_time_now = gm_get_sim_time_safe(gm)
            dt = max(0.0, float(sim_time_now - sim_time_prev))
            sim_time_prev = sim_time_now

            rollout_done = bool(done or ((steps + 1) >= max_steps))

            # 1) accumulate native events (BLUE only)
            dbg_rewards = (episode_idx <= REWARD_DEBUG_FIRST_EPISODES and steps <= 1)
            accumulate_rewards_for_uid_set(gm, episode_uid_set, pending_reward, debug_print=dbg_rewards)

            # 2) score-delta shaping (team-level, phase-specific) for BLUE
            blue_score_now = int(getattr(gm, "blue_score", 0))
            red_score_now = int(getattr(gm, "red_score", 0))
            d_blue = blue_score_now - prev_blue_score
            d_red = red_score_now - prev_red_score
            prev_blue_score, prev_red_score = blue_score_now, red_score_now

            if d_blue != 0 or d_red != 0:
                windows_since_score = 0
                scale = get_phase_score_delta_reward(cur_phase)
                team_r = float(scale * d_blue - scale * d_red)
                add_team_reward_to_pending(episode_uid_set, pending_reward, team_r)
            else:
                windows_since_score += 1
                stall_pen = get_no_score_penalty(cur_phase, windows_since_score)
                if stall_pen != 0.0:
                    add_team_reward_to_pending(episode_uid_set, pending_reward, -stall_pen)

            # Bootstrap V(s_{t+1}) from central critic (forward only)
            with torch.no_grad():
                next_joint_obs = get_blue_team_obs_in_id_order(env)
                next_central_np = np.stack(next_joint_obs, axis=0)
                next_central_tensor = torch.tensor(next_central_np, dtype=torch.float32, device=act_device).unsqueeze(0)
                v_all = policy_act.forward_central_critic(next_central_tensor).detach().reshape(-1).float()

            bootstrap_map: Dict[str, float] = {}
            for a in blue_agents_enabled:
                uid = agent_uid(a)
                if v_all.numel() == 1:
                    bootstrap_map[uid] = float(v_all[0].item())
                else:
                    idx = int(getattr(a, "agent_id", 0))
                    idx = max(0, min(idx, int(v_all.numel() - 1)))
                    bootstrap_map[uid] = float(v_all[idx].item())

            if rollout_done:
                for k in list(bootstrap_map.keys()):
                    bootstrap_map[k] = 0.0

            # Store BLUE transitions (consume pending only for uids that acted)
            step_reward_sum = 0.0
            time_penalty = get_phase_time_penalty(cur_phase)

            for agent, uid, actor_obs_np, central_obs_np, macro_idx, target_idx, logp, val, tid, mm_np in decisions:
                r = consume_pending_reward_for_uid(uid, pending_reward, time_penalty)
                step_reward_sum += r

                agent_dead = (not agent.isEnabled())
                agent_done = bool(rollout_done or agent_dead)
                nv = 0.0 if agent_done else float(bootstrap_map.get(uid, 0.0))

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
                flush_pending_rewards_into_buffer(pending_reward, last_buf_idx, buffer)
                ent = get_entropy_coef(cur_phase, phase_episode_count)

                updates_done += 1
                phase_display = "ELO" if league_mode else cur_phase
                print(f"[MAPPO UPDATE] step={global_step} episode={episode_idx} phase={phase_display} ENT={ent:.4f} Opp={opponent_tag} updates={updates_done}")

                mappo_update(policy_train, optimizer, buffer, train_device, ent)
                policy_act.load_state_dict(policy_train.state_dict())
                policy_act.eval()

                # snapshot into pool
                if ENABLE_SELFPLAY:
                    snapshot_tick += 1
                    if snapshot_tick >= int(SNAPSHOT_EVERY_UPDATES):
                        snapshot_tick = 0
                        pool.add_snapshot(policy_train.state_dict(), update_idx=updates_done)

                last_buf_idx.clear()

        # ---- Episode end: terminal team reward (BLUE only) ----
        term_team = get_terminal_team_reward(
            cur_phase,
            int(getattr(gm, "blue_score", 0)),
            int(getattr(gm, "red_score", 0)),
        )
        add_team_reward_to_pending(episode_uid_set, pending_reward, term_team)
        ep_terminal_only += float(term_team) * float(max(1, len(episode_uid_set)))
        flush_pending_rewards_into_buffer(pending_reward, last_buf_idx, buffer)

        # Episode result
        if gm.blue_score > gm.red_score:
            result = "BLUE WIN"
            blue_wins += 1
            phase_recent.append(1)
            blue_win_bool = True
        elif gm.red_score > gm.blue_score:
            result = "RED WIN"
            red_wins += 1
            phase_recent.append(0)
            blue_win_bool = False
        else:
            result = "DRAW"
            draws += 1
            phase_recent.append(0)
            blue_win_bool = False

        # record result to PFSP stats + Elo
        if ENABLE_SELFPLAY and held_opp is not None:
            pool.record_result(held_opp.oid, blue_win_bool)
            actual_score = 1.0 if blue_win_bool else (0.5 if result == "DRAW" else 0.0)
            pool.update_elo(held_opp.oid, actual_score)

        if len(phase_recent) > PHASE_WINRATE_WINDOW:
            phase_recent = phase_recent[-PHASE_WINRATE_WINDOW:]
        phase_wr = sum(phase_recent) / max(1, len(phase_recent))

        avg_step_r = ep_return / max(1, steps)

        # Behavior logging flush (publication-ready metrics)
        behavior_logger.finalize_episode(
            gm,
            meta={
                "episode": episode_idx,
                "phase": cur_phase,
                "opponent_id": opponent_tag,
                "opponent_kind": held_opp.kind if held_opp is not None else "unknown",
                "curriculum_mode": cur_mode,
                "map_name": MAP_NAME or "",
                "map_path": MAP_PATH or "",
                "result": result,
            },
        )
        new_eps = behavior_logger.episodes[last_written_idx:]
        if new_eps:
            records = []
            for ep in new_eps:
                records.extend(ep.to_flat_records())
            append_records_csv(behavior_csv_path, records)
            last_written_idx = len(behavior_logger.episodes)
        phase_display = "ELO" if league_mode else cur_phase
        pick_phase = "OP3" if league_mode else cur_phase
        pick_source = "scripted" if (held_opp is not None and held_opp.kind == "scripted") else "snapshot"
        elo_str = ""
        if league_mode:
            elo_str = f" | Elo L={pool.learner_rating:.1f} O={pool.get_rating(opponent_tag):.1f}"
        print(
            f"[{episode_idx:5d}] {result:8} | "
            f"StepR {avg_step_r:+.3f} TermR {ep_terminal_only:+.1f} | "
            f"Score {gm.blue_score}:{gm.red_score} | "
            f"BlueWins: {blue_wins} RedWins: {red_wins} Draws: {draws} | "
            f"PhaseWin {phase_wr*100:.1f}% | Phase={phase_display} Opp={opponent_tag} "
            f"pick={pick_source} pick_phase={pick_phase} Curric={cur_mode}{elo_str}"
        )

        if (episode_idx % MACRO_STATS_PRINT_EVERY == 0) or (MACRO_STATS_PRINT_ON_WIN and result == "BLUE WIN"):
            print("[MACROS] " + f"episode={episode_idx} phase={phase_display} result={result}")
            for line in macro_tracker.summary_lines():
                print(" " + line)

        # Curriculum advance (PPO-style Elo gate + win-by score gate)
        if (not league_mode) and cur_phase != PHASE_SEQUENCE[-1]:
            min_eps = int(PPO_PHASE_MIN_EPISODES.get(cur_phase, 0))
            target_key = f"scripted:{cur_phase}"
            opp_r = float(pool.get_rating(target_key))
            required_win_by = int(PPO_PHASE_REQUIRED_WIN_BY.get(cur_phase, 0))
            meets_score_gate = True
            if required_win_by > 0:
                meets_score_gate = False
                if (
                    held_opp is not None
                    and held_opp.oid == target_key
                    and blue_win_bool
                    and (int(gm.blue_score) - int(gm.red_score)) >= required_win_by
                ):
                    meets_score_gate = True
            if (
                phase_episode_count >= min_eps
                and pool.learner_rating >= (opp_r + float(PPO_PHASE_ELO_MARGIN))
                and meets_score_gate
            ):
                print(f"[CURRICULUM] Advancing from {cur_phase} -> next phase (MAPPO).")
                phase_idx += 1
                phase_episode_count = 0
                phase_recent.clear()
                hold_left = 0
                held_opp = None
                if PPO_SWITCH_TO_ELO_AFTER_OP3 and phase_idx == (len(PHASE_SEQUENCE) - 1):
                    league_mode = True

        # Periodic checkpoint
        if episode_idx % 50 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"mappo_ckpt_ep{episode_idx}.pth")
            torch.save(policy_train.state_dict(), ckpt_path)
            print(f"[CKPT] Saved: {ckpt_path}")

            # quick pool status line
            if ENABLE_SELFPLAY:
                latest = pool.latest_neural_id
                if latest is not None:
                    print(f"[POOL] neural={len(pool.neural_ids)} latest={latest} latest_wr={pool.entries[latest].winrate():.2f}")

    final_path = os.path.join(CHECKPOINT_DIR, "research_mappo_model1.pth")
    torch.save(policy_train.state_dict(), final_path)
    print(f"\n[MAPPO] Training complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    train_mappo_event()
