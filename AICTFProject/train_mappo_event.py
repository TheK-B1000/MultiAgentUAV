"""
train_mappo_event.py (MAPPO / CTDE, DirectML-safe by design)

Key design:
  - policy_act runs on DirectML/CUDA for action sampling only (forward only).
  - policy_train runs on CPU for PPO/MAPPO updates (backward only).
  - After each update, policy_train weights are synced to policy_act.

This avoids DirectML backward crashes like:
  "The parameter is incorrect."
"""

import os
import random
import copy
from collections import deque
from typing import Dict, List, Tuple, Any, Deque, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# IMPORTS (GameField constants)
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

POLICY_SAVE_INTERVAL = UPDATE_EVERY  # ghost snapshots by sample count
POLICY_SAMPLE_CHANCE = 0.15

PHASE_SEQUENCE = ["OP1", "OP2", "OP3"]
MIN_PHASE_EPISODES = {"OP1": 500, "OP2": 1000, "OP3": 2000}
TARGET_PHASE_WINRATE = {"OP1": 0.90, "OP2": 0.86, "OP3": 0.80}
PHASE_WINRATE_WINDOW = 50

ENT_COEF_BY_PHASE = {"OP1": 0.06, "OP2": 0.05, "OP3": 0.04}
OP2_ENTROPY_FLOOR = 0.008

OP2_DRAW_PENALTY = -0.8
OP2_SCORE_BONUS = 0.5
OP1_DRAW_PENALTY = -0.2
OP1_SCORE_BONUS = 3.0

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

MACRO_STATS_PRINT_EVERY = 10
MACRO_STATS_PRINT_ON_WIN = True
PRINT_ROLLOUT_DIAG_EVERY_UPDATE = True

# Team reward stabilizer (recommended for CTDE critic)
USE_TEAM_REWARD_FOR_MAPPO = True


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
    side = getattr(agent, "side", "blue")
    aid = getattr(agent, "agent_id", 0)
    return f"{side}_{aid}"


def gm_get_sim_time_safe(gm: GameManager) -> float:
    if hasattr(gm, "get_sim_time") and callable(getattr(gm, "get_sim_time")):
        return float(gm.get_sim_time())
    if hasattr(gm, "sim_time"):
        return float(getattr(gm, "sim_time"))
    return 0.0


def zero_obs_like() -> np.ndarray:
    return np.zeros((NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32)


def get_blue_team_obs_in_id_order(env: GameField) -> List[np.ndarray]:
    blue_all = list(env.blue_agents)
    blue_all.sort(key=lambda a: getattr(a, "agent_id", 0))

    out: List[np.ndarray] = []
    for a in blue_all:
        if not a.isEnabled():
            out.append(zero_obs_like())
        else:
            out.append(np.array(env.build_observation(a), dtype=np.float32))
    return out


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


def submit_external_actions_robust(env: GameField, action_dict: Dict[str, Tuple[int, int]]) -> None:
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


# ==========================================================
# REWARDS (robust agent_id mapping)
# ==========================================================
def _build_event_id_to_uid_map(blue_agents: List[Any]) -> Dict[Any, str]:
    m: Dict[Any, str] = {}
    for a in blue_agents:
        uid = agent_uid(a)
        aid = getattr(a, "agent_id", None)
        side = getattr(a, "side", "blue")
        uniq = getattr(a, "unique_id", None)
        slot = getattr(a, "slot_id", None)

        candidates: List[Any] = [
            uid, str(uid),
            f"{side}_{aid}", str(f"{side}_{aid}"),
            aid, str(aid) if aid is not None else None,
            uniq, str(uniq) if uniq is not None else None,
            slot, str(slot) if slot is not None else None,
        ]
        for c in candidates:
            if c is None:
                continue
            m[c] = uid
            m[str(c)] = uid
    return m


def collect_blue_rewards_for_step(
    gm: GameManager,
    blue_agents: List[Any],
    cur_phase: str,
) -> Dict[str, float]:
    raw_events = gm_pop_reward_events_safe(gm)

    uids = [agent_uid(a) for a in blue_agents]
    rewards_sum_by_uid: Dict[str, float] = {uid: 0.0 for uid in uids}
    event_map = _build_event_id_to_uid_map(blue_agents)

    team_r_total = 0.0
    for _t_event, agent_id, r in raw_events:
        rr = float(r)
        if agent_id is None:
            team_r_total += rr
            continue

        uid = event_map.get(agent_id, event_map.get(str(agent_id), None))
        if uid is not None and uid in rewards_sum_by_uid:
            rewards_sum_by_uid[uid] += rr

    if team_r_total != 0.0 and len(rewards_sum_by_uid) > 0:
        share = team_r_total / len(rewards_sum_by_uid)
        for uid in rewards_sum_by_uid:
            rewards_sum_by_uid[uid] += share

    if gm.game_over and gm.blue_score == gm.red_score and len(rewards_sum_by_uid) > 0:
        if cur_phase == "OP2":
            per = OP2_DRAW_PENALTY / len(rewards_sum_by_uid)
            for uid in rewards_sum_by_uid:
                rewards_sum_by_uid[uid] += per
        elif cur_phase == "OP1":
            per = OP1_DRAW_PENALTY / len(rewards_sum_by_uid)
            for uid in rewards_sum_by_uid:
                rewards_sum_by_uid[uid] += per

    return rewards_sum_by_uid


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
# DML-safe logp/entropy (CPU one-hot, no gather/scatter backward)
# ==========================================================
def masked_categorical_logp_entropy_no_scatter(
    logits: torch.Tensor,                 # [B,A]
    actions: torch.Tensor,                # [B]
    mask_float: Optional[torch.Tensor] = None,  # [B,A] float {0,1}
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mask_float is not None and mask_float.numel() > 0:
        logits = logits + (mask_float - 1.0) * 1e10

    logp_all = F.log_softmax(logits, dim=-1)
    p_all = torch.exp(logp_all)

    oh = F.one_hot(actions.to("cpu"), num_classes=logits.size(-1)).to(
        device=logits.device, dtype=logp_all.dtype
    )
    logp = (oh * logp_all).sum(dim=-1)
    entropy = -(p_all * logp_all).sum(dim=-1)
    return logp, entropy


def _fix_all_false_rows_np(mask_np: np.ndarray) -> np.ndarray:
    if mask_np.size == 0:
        return mask_np
    row_sum = mask_np.sum(axis=1)
    bad = row_sum == 0
    if np.any(bad):
        mask_np = mask_np.copy()
        mask_np[bad, :] = True
    return mask_np


def _get_macro_mask_np(policy_act: ActorCriticNet, agent: Any, env: GameField) -> Optional[np.ndarray]:
    mm = None
    if hasattr(policy_act, "get_action_mask") and callable(getattr(policy_act, "get_action_mask")):
        try:
            mm = policy_act.get_action_mask(agent, env)
        except Exception:
            mm = None
    if mm is None and hasattr(env, "get_macro_mask") and callable(getattr(env, "get_macro_mask")):
        try:
            mm = env.get_macro_mask(agent)
        except Exception:
            mm = None

    if mm is None:
        return None
    if isinstance(mm, np.ndarray):
        return mm.astype(np.bool_)
    if torch.is_tensor(mm):
        return mm.detach().cpu().numpy().astype(np.bool_)
    return np.array(list(mm), dtype=np.bool_)


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
    MAPPO sampling using policy_act.act() for (macro,target,logp,mask) so behavior and training can't drift.

    CTDE value: we compute central critic value explicitly from central_obs_tensor to keep MAPPO correct,
    because act() often returns local critic value (PPO-style) in many codebases.
    """
    device = next(policy_act.parameters()).device
    actor_obs_tensor = actor_obs_tensor.to(device).float()
    central_obs_tensor = central_obs_tensor.to(device).float()

    # --- Use the network's act() path (masking + logp consistency) ---
    try:
        out = policy_act.act(
            actor_obs_tensor,
            agent=agent,
            game_field=env,
            deterministic=deterministic,
        )
    except TypeError:
        # Some implementations have extra kwargs
        out = policy_act.act(
            actor_obs_tensor,
            agent=agent,
            game_field=env,
            deterministic=deterministic,
            return_old_log_prob_key=False,
        )

    # Robust extraction (handles [1] tensors and scalars)
    def _grab(key: str, alt_keys: List[str] = None) -> torch.Tensor:
        if alt_keys is None:
            alt_keys = []
        if key in out:
            v = out[key]
        else:
            v = None
            for k in alt_keys:
                if k in out:
                    v = out[k]
                    break
            if v is None:
                raise KeyError(f"policy.act() output missing '{key}' (also tried {alt_keys})")
        if not torch.is_tensor(v):
            v = torch.tensor(v, device=device)
        return v.reshape(-1)

    macro_action = _grab("macro_action", ["macro"]).long()          # [1]
    target_action = _grab("target_action", ["target"]).long()       # [1]

    # log prob key might be "log_prob" or "old_log_prob"
    logp = _grab("log_prob", ["old_log_prob", "logp"]).float()      # [1]

    # --- Central critic value for CTDE ---
    central_value = policy_act.forward_central_critic(central_obs_tensor).reshape(-1).float()  # [1]

    # --- Macro mask (optional) ---
    mm = out.get("macro_mask", None)
    mm_np = None
    if mm is not None:
        if torch.is_tensor(mm):
            mm_np = mm.detach().cpu().numpy().astype(np.bool_).reshape(-1)
        else:
            mm_np = np.array(mm, dtype=np.bool_).reshape(-1)

    return {
        "macro_action": macro_action,          # [1]
        "target_action": target_action,        # [1]
        "old_log_prob": logp,                  # [1]
        "value": central_value,                # [1] central critic value
        "macro_mask": mm_np,                   # np.bool_ [A] or None
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
            self.macro_masks.append(np.array([], dtype=np.bool_))
        else:
            self.macro_masks.append(np.array(macro_mask, dtype=np.bool_).reshape(-1))

    def size(self) -> int:
        return len(self.actor_obs)

    def clear(self) -> None:
        self.__init__()

    def to_tensors(self, device: torch.device):
        actor_obs = torch.tensor(np.stack(self.actor_obs), dtype=torch.float32, device=device)
        central_obs = torch.tensor(np.stack(self.central_obs), dtype=torch.float32, device=device)
        macro_actions = torch.tensor(self.macro_actions, dtype=torch.long, device=device)
        target_actions = torch.tensor(self.target_actions, dtype=torch.long, device=device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        values = torch.tensor(self.values, dtype=torch.float32, device=device)
        next_values = torch.tensor(self.next_values, dtype=torch.float32, device=device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)
        dts = torch.tensor(self.dts, dtype=torch.float32, device=device)
        traj_ids = torch.tensor(self.traj_ids, dtype=torch.long, device=device)

        if len(self.macro_masks) > 0 and self.macro_masks[0].size > 0:
            macro_masks_np = np.stack(self.macro_masks).astype(np.bool_)
        else:
            macro_masks_np = np.zeros((len(self.actor_obs), 0), dtype=np.bool_)

        return (
            actor_obs, central_obs, macro_actions, target_actions,
            old_log_probs, values, next_values, rewards, dones, dts, traj_ids,
            macro_masks_np
        )


# ==========================================================
# CPU GAE (robust for DirectML setups)
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
        macro_masks_np
    ) = buffer.to_tensors(device)

    # --- GAE on CPU (your existing grouped-by-traj setup) ---
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

    T = actor_obs.size(0)
    if T == 0:
        buffer.clear()
        return

    policy_train.train()

    for _ in range(PPO_EPOCHS):
        perm_np = np.random.permutation(T).astype(np.int64)

        for start in range(0, T, MINIBATCH_SIZE):
            end = min(start + MINIBATCH_SIZE, T)
            idx_np = perm_np[start:end]
            mb_idx = torch.tensor(idx_np, dtype=torch.long, device=device)

            mb_actor    = actor_obs.index_select(0, mb_idx)      # [B,C,H,W]
            mb_central  = central_obs.index_select(0, mb_idx)    # [B,N,C,H,W]
            mb_macro    = macro_actions.index_select(0, mb_idx)  # [B]
            mb_target   = target_actions.index_select(0, mb_idx) # [B]
            mb_old_logp = old_log_probs.index_select(0, mb_idx)  # [B]
            mb_adv      = advantages.index_select(0, mb_idx)     # [B]
            mb_ret      = returns.index_select(0, mb_idx)        # [B]

            # macro mask batch (optional)
            mb_mask = None
            if macro_masks_np is not None and getattr(macro_masks_np, "shape", (0, 0))[1] > 0:
                mb_mask_np = macro_masks_np[idx_np, :]
                mb_mask_np = _fix_all_false_rows_np(mb_mask_np)
                mb_mask = torch.tensor(mb_mask_np, dtype=torch.bool, device=device)  # [B,A]

            # --- IMPORTANT: CTDE critic path ---
            new_values = policy_train.forward_central_critic(mb_central).reshape(-1)  # [B]

            # --- IMPORTANT: actor/logp path uses evaluate_actions() to avoid drift ---
            # We only need (new_logp, entropy). Some implementations also return values; we ignore them.
            try:
                new_logp, entropy, _ = policy_train.evaluate_actions(
                    mb_actor, mb_macro, mb_target, macro_mask_batch=mb_mask
                )
            except TypeError:
                # Fallback param name variations
                try:
                    new_logp, entropy, _ = policy_train.evaluate_actions(
                        mb_actor, mb_macro, mb_target, macro_mask=mb_mask
                    )
                except TypeError:
                    # Last resort: no mask argument supported
                    new_logp, entropy, _ = policy_train.evaluate_actions(
                        mb_actor, mb_macro, mb_target
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
# MACRO USAGE TRACKER
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
        uid = agent_uid(agent)
        mn = _macro_name(macro_enum)
        self.total[mn] = self.total.get(mn, 0) + 1
        self.by_agent.setdefault(uid, {})
        self.by_agent[uid][mn] = self.by_agent[uid].get(mn, 0) + 1

    def summary_lines(self) -> List[str]:
        lines: List[str] = []
        total_n = sum(self.total.values())
        if total_n <= 0:
            return ["(no macros recorded)"]

        items = sorted(self.total.items(), key=lambda kv: kv[1], reverse=True)
        parts = [f"{k}:{v} ({100.0*v/total_n:.1f}%)" for k, v in items]
        lines.append("Total: " + " | ".join(parts))

        for uid, dist in self.by_agent.items():
            n = sum(dist.values())
            it = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
            parts = [f"{k}:{v} ({100.0*v/n:.1f}%)" for k, v in it]
            lines.append(f"  {uid}: " + " | ".join(parts))
        return lines


# ==========================================================
# MAIN LOOP
# ==========================================================
def train_mappo_event(total_steps: int = TOTAL_STEPS) -> None:
    set_seed(42)

    env = make_env()
    gm = env.getGameManager()

    env.reset_default()
    if hasattr(gm, "reset_game"):
        try:
            gm.reset_game(reset_scores=True)
        except Exception:
            pass

    if env.blue_agents:
        sample = env.build_observation(env.blue_agents[0])
        print(f"[train_mappo_event] Sample obs shape: C={len(sample)}, H={len(sample[0])}, W={len(sample[0][0])}")

    n_agents = len(getattr(env, "blue_agents", [])) or getattr(env, "agents_per_team", 2)

    def _make_policy():
        return ActorCriticNet(
            n_macros=len(USED_MACROS),
            n_targets=env.num_macro_targets,
            n_agents=n_agents,
            in_channels=NUM_CNN_CHANNELS,
            height=CNN_ROWS,
            width=CNN_COLS,
        )

    # Dual-device setup:
    act_device = prefer_device()
    train_device = torch.device("cpu") if (HAS_TDML and "privateuseone" in str(act_device).lower()) else act_device

    print(f"[train_mappo_event] Using act_device: {act_device}")
    print(f"[train_mappo_event] Using train_device: {train_device}")

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
    phase_wr = 0.0

    old_policies_buffer: Deque[Dict[str, Any]] = deque(maxlen=20)

    traj_id_counter = 0
    traj_id_map: Dict[Tuple[int, str], int] = {}

    macro_tracker = MacroUsageTracker()
    samples_since_save = 0

    while global_step < total_steps:
        episode_idx += 1
        phase_episode_count += 1
        macro_tracker.reset()

        cur_phase = PHASE_SEQUENCE[phase_idx]
        phase_cfg = PHASE_CONFIG[cur_phase]
        gm.score_limit = phase_cfg["score_limit"]
        gm.max_time = phase_cfg["max_time"]
        max_steps = int(phase_cfg["max_macro_steps"])

        if hasattr(gm, "set_phase"):
            try:
                gm.set_phase(cur_phase)
            except Exception:
                pass

        # Self-play eligibility (optional)
        ENABLE_SELFPLAY_PHASE = "OP3"
        MIN_EPISODES_FOR_SELFPLAY = 500
        MIN_WR_FOR_SELFPLAY = 0.70
        can_selfplay = (
            cur_phase == ENABLE_SELFPLAY_PHASE
            and phase_episode_count >= MIN_EPISODES_FOR_SELFPLAY
            and phase_wr >= MIN_WR_FOR_SELFPLAY
            and len(old_policies_buffer) > 0
            and hasattr(env, "set_red_policy_neural")
        )

        opponent_tag = cur_phase
        if can_selfplay and random.random() < POLICY_SAMPLE_CHANCE:
            # keep red_net on CPU unless your env explicitly runs neural on GPU
            red_net = _make_policy().to(torch.device("cpu"))
            red_net.load_state_dict(random.choice(old_policies_buffer))
            red_net.eval()
            env.set_red_policy_neural(red_net)
            opponent_tag = "SELFPLAY"
        else:
            set_red_policy_for_phase(env, cur_phase)

        # Reset episode
        env.reset_default()
        if hasattr(gm, "reset_game"):
            try:
                gm.reset_game(reset_scores=True)
            except Exception:
                pass

        red_pol = env.policies.get("red")
        if hasattr(red_pol, "reset"):
            try:
                red_pol.reset()
            except Exception:
                pass

        done = False
        ep_return = 0.0
        steps = 0
        prev_blue_score = gm.blue_score

        sim_time_prev = gm_get_sim_time_safe(gm)
        traj_id_map.clear()

        while not done and steps < max_steps and global_step < total_steps:
            if samples_since_save >= POLICY_SAVE_INTERVAL:
                old_policies_buffer.append(copy.deepcopy(policy_train.state_dict()))
                samples_since_save = 0

            # Central obs at decision boundary (sample with policy_act)
            blue_joint_obs = get_blue_team_obs_in_id_order(env)
            central_obs_np = np.stack(blue_joint_obs, axis=0)
            central_obs_tensor = torch.tensor(central_obs_np, dtype=torch.float32, device=act_device).unsqueeze(0)

            blue_agents_enabled = [a for a in env.blue_agents if a.isEnabled()]
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
                mm_np = out.get("macro_mask", None)  # already np.bool_ [A] or None

                macro_enum = USED_MACROS[macro_idx]
                macro_tracker.add(agent, macro_enum)
                macro_val = int(getattr(macro_enum, "value", int(macro_enum)))

                key = (episode_idx, str(agent_uid(agent)))
                tid = traj_id_map.get(key)
                if tid is None:
                    traj_id_map[key] = traj_id_counter
                    tid = traj_id_counter
                    traj_id_counter += 1

                decisions.append((agent, actor_obs_np, central_obs_np, macro_idx, target_idx, logp, val, tid, mm_np))

                for k in external_keys_for_agent(env, agent):
                    submit_actions[k] = (macro_val, target_idx)

            submit_external_actions_robust(env, submit_actions)

            # Sim window
            sim_t = 0.0
            while sim_t < (DECISION_WINDOW - 1e-9) and not gm.game_over:
                env.update(SIM_DT)
                sim_t += SIM_DT

            done = bool(gm.game_over)

            sim_time_now = gm_get_sim_time_safe(gm)
            dt = max(0.0, float(sim_time_now - sim_time_prev))
            sim_time_prev = sim_time_now

            # Rewards
            rewards_all = collect_blue_rewards_for_step(gm, env.blue_agents, cur_phase)
            enabled_uids = [agent_uid(a) for a in blue_agents_enabled]
            rewards = {uid: float(rewards_all.get(uid, 0.0)) for uid in enabled_uids}

            for uid in enabled_uids:
                rewards[uid] = rewards.get(uid, 0.0) - TIME_PENALTY_PER_AGENT_PER_MACRO

            if USE_TEAM_REWARD_FOR_MAPPO and len(enabled_uids) > 0:
                team_total = sum(float(rewards.get(uid, 0.0)) for uid in enabled_uids)
                team_mean = team_total / max(1, len(enabled_uids))
                for uid in enabled_uids:
                    rewards[uid] = team_mean

            blue_score_delta = gm.blue_score - prev_blue_score
            if blue_score_delta > 0 and len(enabled_uids) > 0:
                if cur_phase == "OP2":
                    per = (OP2_SCORE_BONUS * blue_score_delta) / len(enabled_uids)
                else:
                    per = (OP1_SCORE_BONUS * blue_score_delta) / len(enabled_uids)
                if cur_phase in ("OP1", "OP2"):
                    for uid in enabled_uids:
                        rewards[uid] += per
            prev_blue_score = gm.blue_score

            # Bootstrap next V(s_{t+1}) using policy_act forward only
            with torch.no_grad():
                next_joint_obs = get_blue_team_obs_in_id_order(env)
                next_central_np = np.stack(next_joint_obs, axis=0)
                next_central_tensor = torch.tensor(next_central_np, dtype=torch.float32, device=act_device).unsqueeze(0)
                bootstrap_value = float(policy_act.forward_central_critic(next_central_tensor).detach().reshape(-1)[0].item())

            rollout_done = bool(done or ((steps + 1) >= max_steps))
            if rollout_done:
                bootstrap_value = 0.0

            step_reward_sum = 0.0
            for agent, actor_obs_np, central_obs_np, macro_idx, target_idx, logp, val, tid, mm_np in decisions:
                uid = agent_uid(agent)
                r = float(rewards.get(uid, 0.0))
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
                samples_since_save += 1

            ep_return += step_reward_sum
            steps += 1

            # Update on train_device only, then sync weights to policy_act
            if buffer.size() >= UPDATE_EVERY:
                ent = get_entropy_coef(cur_phase, phase_episode_count)
                print(f"[MAPPO UPDATE] step={global_step} episode={episode_idx} phase={cur_phase} ENT={ent:.4f} Opp={opponent_tag}")

                mappo_update(policy_train, optimizer, buffer, train_device, ent)

                # sync updated weights -> sampling model
                policy_act.load_state_dict(policy_train.state_dict())
                policy_act.eval()

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
                old_policies_buffer.clear()

        # Periodic checkpoint (save training weights)
        if episode_idx % 50 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"mappo_ckpt_ep{episode_idx}.pth")
            torch.save(policy_train.state_dict(), ckpt_path)
            print(f"[CKPT] Saved: {ckpt_path}")

    final_path = os.path.join(CHECKPOINT_DIR, "research_mappo_model1.pth")
    torch.save(policy_train.state_dict(), final_path)
    print(f"\n[MAPPO] Training complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    train_mappo_event()
