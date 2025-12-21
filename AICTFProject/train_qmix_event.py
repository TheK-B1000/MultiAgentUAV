# ==========================================================
# train_qmix_event.py (UPDATED: GLOBAL "GOD VIEW" STATE + SELF-PLAY)
#   - Uses env.get_global_state() for mixer state input (flat 3200)
#   - state_dim = env.get_global_state_dim()
#   - Event-driven dt-aware discounting (gamma ** dt)
#   - Masked epsilon-greedy for valid macro/target combos
#   - CTDE: decentralized agent obs, centralized mixer state
#   - AUTO-SYNC macros from env.macro_order
#
# Self-play (modern-ish MARL defaults for QMIX in competitive games):
#   ✅ Opponent pool of frozen snapshots (neural) + scripted baselines (OP1/2/3)
#   ✅ PFSP sampling (prefer opponents with ~50% winrate vs you)
#   ✅ Opponent hold for K episodes (reduces non-stationarity)
#   ✅ Phase-based scripted-to-neural mix schedule
#   ✅ Latest-bias (sometimes fight your newest self)
#
# Key note:
#   - Blue is the learner and is the ONLY side written to replay.
#   - Red is an opponent (scripted or neural snapshot) driven via external actions.
# ==========================================================

import os
import random
import copy
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Deque

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
from policies import OP1RedPolicy, OP2RedPolicy, OP3RedPolicy

from obs_encoder import ObsEncoder


# ==========================================================
# Device
# ==========================================================
def get_act_device() -> torch.device:
    if HAS_TDML:
        return torch_directml.device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


ACT_DEVICE = get_act_device()
TRAIN_DEVICE = torch.device("cpu")  # DirectML-safe


# ==========================================================
# Reproducibility
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


# ==========================================================
# Hyperparams
# ==========================================================
TOTAL_STEPS: int = 1_000_000
UPDATE_EVERY: int = 2048
BATCH_SIZE: int = 256
REPLAY_CAPACITY: int = 200_000
WARMUP_STEPS: int = 10_000

LR: float = 3e-4
MAX_GRAD_NORM: float = 10.0

GAMMA: float = 0.995
DECISION_WINDOW: float = 1.0
SIM_DT: float = 0.1

MIX_EMBED_DIM: int = 32

EPS_START: float = 1.0
EPS_END: float = 0.05
EPS_DECAY_STEPS: int = 250_000

TARGET_UPDATE_EVERY: int = 2000
TAU_SOFT: float = 1.0

CHECKPOINT_DIR: str = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

PHASE_SEQUENCE: List[str] = ["OP1", "OP2", "OP3"]
MIN_PHASE_EPISODES = {"OP1": 300, "OP2": 700, "OP3": 1500}
TARGET_PHASE_WINRATE = {"OP1": 0.74, "OP2": 0.70, "OP3": 0.65}
PHASE_WINRATE_WINDOW: int = 50
PHASE_CONFIG = {
    "OP1": dict(score_limit=3, max_time=200.0, max_macro_steps=500),
    "OP2": dict(score_limit=3, max_time=200.0, max_macro_steps=500),
    "OP3": dict(score_limit=3, max_time=200.0, max_macro_steps=500),
}

WIN_BONUS: float = 25.0
LOSS_PENALTY: float = -25.0
DRAW_PENALTY: float = -5.0

BLUE_SCORED_BONUS: float = 15.0
RED_SCORED_PENALTY: float = -15.0

TIME_PENALTY_PER_AGENT_PER_MACRO: float = 0.001


# ==========================================================
# Self-play config (QMIX)
# ==========================================================
ENABLE_SELFPLAY = True

# scripted vs neural mix by phase
SCRIPTED_MIX_PROB_BY_PHASE = {"OP1": 1.00, "OP2": 0.55, "OP3": 0.25}

# PFSP (Prioritized Fictitious Self-Play)
PFSP_WINDOW = 60
PFSP_ALPHA = 2.0

POOL_MAX = 40
SNAPSHOT_EVERY_OPT_STEPS = 1   # snapshot every opt step by default
WARMUP_OPT_STEPS_FOR_NEURAL = 2

OPPONENT_HOLD_EPISODES = 4
LATEST_BIAS_PROB = 0.25

# opponent action noise (helps prevent overfitting)
RED_DETERMINISTIC = False
RED_EPS = 0.05  # small epsilon for opponent sampling when neural (only used if not deterministic)


# ==========================================================
# Env helpers
# ==========================================================
def make_env() -> GameField:
    grid = [[0] * CNN_COLS for _ in range(CNN_ROWS)]
    env = GameField(grid)
    env.use_internal_policies = True

    # Self-play needs both sides externally driven
    env.set_external_control("blue", True)
    env.set_external_control("red", True)

    env.external_missing_action_mode = "idle"
    return env


def external_key_for_agent(agent: Any) -> str:
    return f"{getattr(agent, 'side', 'blue')}_{int(getattr(agent, 'agent_id', 0))}"


def submit_external_actions_robust(env: GameField, actions_by_agent: Dict[str, Tuple[int, int]]) -> None:
    # clear old if present
    if hasattr(env, "pending_external_actions"):
        try:
            env.pending_external_actions.clear()
        except Exception:
            pass

    fn = getattr(env, "submit_external_actions", None)
    if fn is None or (not callable(fn)):
        raise RuntimeError("GameField must implement submit_external_actions(actions_by_agent).")

    fn(actions_by_agent)


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


def compute_dt_for_window(window_s: float = DECISION_WINDOW) -> float:
    return float(window_s)


def nearest_target_idx(env: GameField, cell_xy: Tuple[int, int]) -> int:
    cx, cy = int(cell_xy[0]), int(cell_xy[1])
    try:
        if hasattr(env, "get_all_macro_targets"):
            targets = env.get_all_macro_targets()
        else:
            targets = [env.get_macro_target(i) for i in range(int(env.num_macro_targets))]
    except Exception:
        targets = [env.get_macro_target(i) for i in range(int(getattr(env, "num_macro_targets", 1)))]

    best_i, best_d2 = 0, 10**9
    for i, (tx, ty) in enumerate(targets):
        dx, dy = int(tx) - cx, int(ty) - cy
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return int(best_i)


def macro_index_from_enum(used_macros: List[MacroAction], macro_any: Any) -> int:
    # Convert MacroAction or int-like to index in env.macro_order list
    m = None
    if isinstance(macro_any, MacroAction):
        m = macro_any
    else:
        try:
            mi = int(getattr(macro_any, "value", macro_any))
            m = MacroAction(mi)
        except Exception:
            m = None

    if m is None:
        return 0

    for i, mm in enumerate(used_macros):
        if mm == m:
            return int(i)
    return 0


def scripted_red_action(
    red_policy: Any,
    agent: Any,
    env: GameField,
    used_macros: List[MacroAction],
    n_targets: int,
) -> Tuple[int, int]:
    """
    Returns (macro_idx, target_idx) aligned with env.macro_order + env.num_macro_targets.
    """
    obs = env.build_observation(agent)

    if hasattr(red_policy, "select_action"):
        macro_any, param_any = red_policy.select_action(obs, agent, env)
    else:
        out = red_policy(obs, agent, env)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            macro_any, param_any = out[0], out[1]
        else:
            macro_any, param_any = out, None

    macro_idx = macro_index_from_enum(used_macros, macro_any)

    if param_any is None:
        targ_idx = 0
    elif isinstance(param_any, (tuple, list)) and len(param_any) == 2:
        targ_idx = nearest_target_idx(env, (int(param_any[0]), int(param_any[1])))
    else:
        try:
            targ_idx = int(param_any)
        except Exception:
            targ_idx = 0

    targ_idx = int(max(0, min(int(n_targets - 1), int(targ_idx))))
    macro_idx = int(max(0, min(int(len(used_macros) - 1), int(macro_idx))))
    return macro_idx, targ_idx


# ==========================================================
# Reward routing (episode-filtered, team reward)
# ==========================================================
def pop_reward_events_strict(gm: GameManager) -> List[Tuple[float, str, float]]:
    fn = getattr(gm, "pop_reward_events", None)
    if fn is None or (not callable(fn)):
        raise RuntimeError("GameManager must implement pop_reward_events() returning (t, agent_id:str, r).")
    return fn()


def clear_reward_events_best_effort(gm: GameManager) -> None:
    try:
        _ = pop_reward_events_strict(gm)
    except Exception:
        return


def init_episode_uid_set(env: GameField) -> set:
    uids = []
    for a in getattr(env, "blue_agents", []):
        if a is None:
            continue
        uids.append(str(getattr(a, "unique_id", f"{a.side}_{a.agent_id}")))
    return set(uids)


def accumulate_team_reward(gm: GameManager, allowed_uids: set, pending: Dict[str, float]) -> None:
    events = pop_reward_events_strict(gm)
    for _t, agent_id, r in events:
        if agent_id is None:
            continue
        key = str(agent_id).strip()
        if key in allowed_uids:
            pending[key] = pending.get(key, 0.0) + float(r)


def consume_team_reward(pending: Dict[str, float], allowed_uids: set, time_penalty_per_agent: float) -> float:
    team_r = 0.0
    for uid in allowed_uids:
        team_r += float(pending.get(uid, 0.0))
        pending[uid] = 0.0
    team_r -= float(time_penalty_per_agent) * float(max(1, len(allowed_uids)))
    return float(team_r)


def outcome_bonus_from_scores(blue_score: int, red_score: int) -> float:
    if blue_score > red_score:
        return float(WIN_BONUS)
    if red_score > blue_score:
        return float(LOSS_PENALTY)
    return float(DRAW_PENALTY)


def apply_score_delta_shaping(
    gm: GameManager,
    allowed_uids: set,
    pending: Dict[str, float],
    prev_blue_score: int,
    prev_red_score: int,
) -> Tuple[int, int]:
    cur_b = int(getattr(gm, "blue_score", 0))
    cur_r = int(getattr(gm, "red_score", 0))

    db = cur_b - int(prev_blue_score)
    dr = cur_r - int(prev_red_score)

    if db != 0:
        per_uid = float(BLUE_SCORED_BONUS) * float(db)
        for uid in allowed_uids:
            pending[uid] = pending.get(uid, 0.0) + per_uid

    if dr != 0:
        per_uid = float(RED_SCORED_PENALTY) * float(dr)
        for uid in allowed_uids:
            pending[uid] = pending.get(uid, 0.0) + per_uid

    return cur_b, cur_r


# ==========================================================
# Masking: macro-mask -> flat action mask
# ==========================================================
def compute_used_macro_mask(env: GameField, agent: Any, n_macros: int) -> np.ndarray:
    mm = env.get_macro_mask(agent)
    mm = np.asarray(mm, dtype=np.bool_).reshape(-1)
    if mm.shape != (n_macros,):
        raise RuntimeError(f"env.get_macro_mask returned {mm.shape}, expected {(n_macros,)}")
    if not mm.any():
        mm[:] = True
    return mm


def macro_mask_to_flat_action_mask(macro_mask: np.ndarray, n_targets: int) -> np.ndarray:
    return np.repeat(np.asarray(macro_mask, dtype=np.bool_), int(n_targets))


# ==========================================================
# Networks: AgentQNet + QMixer
# ==========================================================
class AgentQNet(nn.Module):
    def __init__(
        self,
        n_actions: int,
        in_channels: int = NUM_CNN_CHANNELS,
        height: int = CNN_ROWS,
        width: int = CNN_COLS,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = ObsEncoder(
            in_channels=int(in_channels),
            height=int(height),
            width=int(width),
            latent_dim=int(latent_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(int(latent_dim), 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, int(n_actions)),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs_bchw: torch.Tensor) -> torch.Tensor:
        z = self.encoder(obs_bchw.contiguous())
        return self.head(z)


class QMixer(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 32) -> None:
        super().__init__()
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        self.embed_dim = int(embed_dim)

        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, self.n_agents * self.embed_dim),
        )
        self.hyper_b1 = nn.Linear(self.state_dim, self.embed_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, self.embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        B = agent_qs.size(0)
        N = agent_qs.size(1)
        if N != self.n_agents:
            raise ValueError(f"Expected agent_qs N={self.n_agents}, got {N}")

        w1 = self.hyper_w1(states).view(B, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(B, 1, self.embed_dim)
        w1 = torch.abs(w1)

        hidden = torch.bmm(agent_qs.view(B, 1, self.n_agents), w1) + b1
        hidden = torch.relu(hidden)

        w2 = self.hyper_w2(states).view(B, self.embed_dim, 1)
        w2 = torch.abs(w2)
        b2 = self.hyper_b2(states).view(B, 1, 1)

        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(B, 1)


# ==========================================================
# Replay Buffer
# ==========================================================
@dataclass
class Transition:
    obs_agents: np.ndarray
    state: np.ndarray
    actions: np.ndarray
    avail_actions: np.ndarray
    reward: float
    done: bool
    next_obs_agents: np.ndarray
    next_state: np.ndarray
    next_avail_actions: np.ndarray
    dt: float


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.data: List[Transition] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.data)

    def add(self, tr: Transition) -> None:
        if len(self.data) < self.capacity:
            self.data.append(tr)
        else:
            self.data[self.pos] = tr
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.data, k=int(batch_size))


# ==========================================================
# Epsilon schedule
# ==========================================================
def epsilon_by_step(step: int) -> float:
    if step <= 0:
        return float(EPS_START)
    frac = min(1.0, float(step) / float(max(1, EPS_DECAY_STEPS)))
    return float(EPS_START + frac * (EPS_END - EPS_START))


# ==========================================================
# Action selection (masked epsilon-greedy)
# ==========================================================
@torch.no_grad()
def select_actions_qmix(
    agent_net_act: AgentQNet,
    obs_agents: np.ndarray,
    avail_actions: np.ndarray,
    eps: float,
    device: torch.device,
) -> np.ndarray:
    N = obs_agents.shape[0]
    A = avail_actions.shape[1]
    actions = np.zeros((N,), dtype=np.int64)

    obs_t = torch.tensor(obs_agents, dtype=torch.float32, device=device)
    q = agent_net_act(obs_t).detach().cpu().numpy()

    for i in range(N):
        mask = avail_actions[i].astype(np.bool_)
        if not mask.any():
            actions[i] = 0
            continue

        if random.random() < float(eps):
            valid_idx = np.flatnonzero(mask)
            actions[i] = int(valid_idx[random.randrange(len(valid_idx))])
        else:
            q_i = q[i].copy()
            q_i[~mask] = -1e9
            actions[i] = int(np.argmax(q_i))
    return actions


# ==========================================================
# Training update (Double-QMIX)
# ==========================================================
def soft_update_(tgt: nn.Module, src: nn.Module, tau: float) -> None:
    tau = float(tau)
    with torch.no_grad():
        for p_t, p in zip(tgt.parameters(), src.parameters()):
            p_t.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


def qmix_update(
    agent_net: AgentQNet,
    mixer: QMixer,
    agent_net_tgt: AgentQNet,
    mixer_tgt: QMixer,
    optimizer: optim.Optimizer,
    batch: List[Transition],
    n_agents: int,
    n_actions: int,
) -> float:
    B = len(batch)

    obs_agents = np.stack([tr.obs_agents for tr in batch], axis=0)
    next_obs_agents = np.stack([tr.next_obs_agents for tr in batch], axis=0)
    states = np.stack([tr.state for tr in batch], axis=0)
    next_states = np.stack([tr.next_state for tr in batch], axis=0)
    actions = np.stack([tr.actions for tr in batch], axis=0)
    next_avail = np.stack([tr.next_avail_actions for tr in batch], axis=0).astype(np.bool_)
    rewards = np.array([tr.reward for tr in batch], dtype=np.float32)
    dones = np.array([tr.done for tr in batch], dtype=np.float32)
    dts = np.array([tr.dt for tr in batch], dtype=np.float32)

    obs_t = torch.tensor(obs_agents, dtype=torch.float32, device=TRAIN_DEVICE)
    next_obs_t = torch.tensor(next_obs_agents, dtype=torch.float32, device=TRAIN_DEVICE)
    s_t = torch.tensor(states, dtype=torch.float32, device=TRAIN_DEVICE)
    s2_t = torch.tensor(next_states, dtype=torch.float32, device=TRAIN_DEVICE)
    a_t = torch.tensor(actions, dtype=torch.long, device=TRAIN_DEVICE)
    r_t = torch.tensor(rewards, dtype=torch.float32, device=TRAIN_DEVICE).view(B, 1)
    d_t = torch.tensor(dones, dtype=torch.float32, device=TRAIN_DEVICE).view(B, 1)
    dt_t = torch.tensor(dts, dtype=torch.float32, device=TRAIN_DEVICE).view(B, 1)

    obs_flat = obs_t.view(B * n_agents, NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS)
    q_all = agent_net(obs_flat).view(B, n_agents, n_actions)

    q_chosen = torch.gather(q_all, dim=2, index=a_t.unsqueeze(-1)).squeeze(-1)
    q_tot = mixer(q_chosen, s_t)

    with torch.no_grad():
        next_obs_flat = next_obs_t.view(B * n_agents, NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS)

        q_next_online = agent_net(next_obs_flat).view(B, n_agents, n_actions)
        mask_next = torch.tensor(next_avail, dtype=torch.bool, device=TRAIN_DEVICE)
        q_next_online = q_next_online.masked_fill(~mask_next, -1e9)
        a_star = q_next_online.argmax(dim=2)

        q_next_tgt = agent_net_tgt(next_obs_flat).view(B, n_agents, n_actions)
        q_next_tgt = q_next_tgt.masked_fill(~mask_next, -1e9)
        q_next_sel = torch.gather(q_next_tgt, dim=2, index=a_star.unsqueeze(-1)).squeeze(-1)

        q_tot_next = mixer_tgt(q_next_sel, s2_t)

        gamma_dt = (float(GAMMA) ** dt_t)
        y = r_t + gamma_dt * (1.0 - d_t) * q_tot_next

    loss = (q_tot - y).pow(2).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(list(agent_net.parameters()) + list(mixer.parameters()), MAX_GRAD_NORM)
    optimizer.step()

    return float(loss.item())


# ==========================================================
# Self-play opponent pool (PFSP)
# ==========================================================
@dataclass
class OpponentEntry:
    oid: str
    kind: str  # "scripted" or "neural"
    scripted_tag: Optional[str] = None
    agent_state: Optional[Dict[str, Any]] = None
    results: Deque[int] = None  # 1=blue win, 0=otherwise
    created_opt_step: int = 0

    def __post_init__(self):
        if self.results is None:
            self.results = deque(maxlen=int(PFSP_WINDOW))

    def winrate(self) -> float:
        if not self.results:
            return 0.5
        return float(sum(self.results)) / float(len(self.results))


class OpponentPool:
    def __init__(self, max_size: int = POOL_MAX) -> None:
        self.max_size = int(max_size)
        self.entries: Dict[str, OpponentEntry] = {}
        self.neural_ids: List[str] = []
        self.latest_neural_id: Optional[str] = None

    def add_scripted_defaults(self) -> None:
        for tag in ("OP1", "OP2", "OP3"):
            oid = f"scripted:{tag}"
            if oid not in self.entries:
                self.entries[oid] = OpponentEntry(oid=oid, kind="scripted", scripted_tag=tag)

    def add_snapshot(self, agent_state: Dict[str, Any], opt_step: int) -> str:
        oid = f"neural:o{int(opt_step)}"
        self.entries[oid] = OpponentEntry(
            oid=oid,
            kind="neural",
            agent_state=copy.deepcopy(agent_state),
            created_opt_step=int(opt_step),
        )
        self.neural_ids.append(oid)
        self.latest_neural_id = oid
        self._trim()
        return oid

    def _trim(self) -> None:
        if len(self.neural_ids) <= self.max_size:
            return
        latest = self.latest_neural_id
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

    def record_result(self, opponent_id: str, blue_win: bool) -> None:
        ent = self.entries.get(opponent_id, None)
        if ent is None:
            return
        ent.results.append(1 if blue_win else 0)

    def can_use_neural(self, opt_steps: int) -> bool:
        return (opt_steps >= int(WARMUP_OPT_STEPS_FOR_NEURAL)) and (len(self.neural_ids) > 0)

    def sample(self, phase: str, opt_steps: int) -> OpponentEntry:
        phase = str(phase).upper()
        p_scripted = float(SCRIPTED_MIX_PROB_BY_PHASE.get(phase, 0.5))

        if (not self.can_use_neural(opt_steps)) or (random.random() < p_scripted):
            oid = f"scripted:{phase if phase in ('OP1','OP2','OP3') else 'OP3'}"
            return self.entries[oid]

        # neural
        if self.latest_neural_id is not None and random.random() < float(LATEST_BIAS_PROB):
            return self.entries[self.latest_neural_id]

        ids = list(self.neural_ids)
        if not ids:
            return self.entries[f"scripted:{phase}"]

        wrs = np.array([self.entries[oid].winrate() for oid in ids], dtype=np.float32)
        scores = 1.0 - np.minimum(1.0, np.abs(wrs - 0.5) * 2.0)  # 1 at 0.5, 0 at {0,1}
        scores = np.maximum(scores, 1e-3)
        weights = scores ** float(PFSP_ALPHA)
        weights = weights / np.sum(weights)

        pick = int(np.random.choice(np.arange(len(ids)), p=weights))
        return self.entries[ids[pick]]


# ==========================================================
# MAIN TRAIN LOOP
# ==========================================================
def train_qmix_event(total_steps: int = TOTAL_STEPS) -> None:
    set_seed(42)

    env = make_env()
    gm = env.getGameManager()

    if hasattr(gm, "set_shaping_gamma"):
        gm.set_shaping_gamma(GAMMA)

    env.reset_default()
    if not env.blue_agents:
        raise RuntimeError("No blue agents after env.reset_default().")

    # AUTO-SYNC macros from env.macro_order
    USED_MACROS: List[MacroAction] = list(getattr(env, "macro_order", []))
    if not USED_MACROS:
        raise RuntimeError("env.macro_order missing or empty.")
    N_MACROS = int(len(USED_MACROS))
    if N_MACROS != int(getattr(env, "n_macros", N_MACROS)):
        raise RuntimeError("Macro count mismatch between env.macro_order and env.n_macros.")

    dummy_obs = np.array(env.build_observation(env.blue_agents[0]), dtype=np.float32)
    C, H, W = int(dummy_obs.shape[0]), int(dummy_obs.shape[1]), int(dummy_obs.shape[2])
    print(f"[train_qmix_event] Sample obs shape: C={C}, H={H}, W={W}")

    n_targets = int(getattr(env, "num_macro_targets", 8))
    n_actions = int(N_MACROS * n_targets)
    n_agents = int(getattr(env, "agents_per_team", 2))

    state_dim = int(env.get_global_state_dim())
    st0 = np.asarray(env.get_global_state(), dtype=np.float32).reshape(-1)
    if int(st0.size) != state_dim:
        raise RuntimeError(f"env.get_global_state() size={st0.size} but env.get_global_state_dim()={state_dim}")

    print(f"[train_qmix_event] n_agents={n_agents} n_targets={n_targets} n_actions={n_actions} state_dim={state_dim}")
    print(f"[train_qmix_event] ACT_DEVICE={ACT_DEVICE} TRAIN_DEVICE={TRAIN_DEVICE}")

    # Learner networks (blue)
    agent_net = AgentQNet(n_actions=n_actions, in_channels=NUM_CNN_CHANNELS, height=CNN_ROWS, width=CNN_COLS).to(TRAIN_DEVICE)
    mixer = QMixer(n_agents=n_agents, state_dim=state_dim, embed_dim=MIX_EMBED_DIM).to(TRAIN_DEVICE)

    agent_net_tgt = AgentQNet(n_actions=n_actions, in_channels=NUM_CNN_CHANNELS, height=CNN_ROWS, width=CNN_COLS).to(TRAIN_DEVICE)
    mixer_tgt = QMixer(n_agents=n_agents, state_dim=state_dim, embed_dim=MIX_EMBED_DIM).to(TRAIN_DEVICE)
    agent_net_tgt.load_state_dict(agent_net.state_dict())
    mixer_tgt.load_state_dict(mixer.state_dict())

    agent_net_act = AgentQNet(n_actions=n_actions, in_channels=NUM_CNN_CHANNELS, height=CNN_ROWS, width=CNN_COLS).to(ACT_DEVICE)
    agent_net_act.load_state_dict(agent_net.state_dict())
    agent_net_act.eval()

    # Opponent neural net (red) for snapshot play
    opp_agent_net_act = AgentQNet(n_actions=n_actions, in_channels=NUM_CNN_CHANNELS, height=CNN_ROWS, width=CNN_COLS).to(ACT_DEVICE)
    opp_agent_net_act.eval()

    optimizer = optim.Adam(list(agent_net.parameters()) + list(mixer.parameters()), lr=LR, foreach=False)
    replay = ReplayBuffer(REPLAY_CAPACITY)

    # self-play pool
    pool = OpponentPool(max_size=POOL_MAX)
    pool.add_scripted_defaults()

    global_step = 0
    episode_idx = 0
    opt_steps = 0

    blue_wins = red_wins = draws = 0

    phase_idx = 0
    phase_episode_count = 0
    phase_recent: List[int] = []

    # opponent hold state
    hold_left = 0
    held_opp: Optional[OpponentEntry] = None

    def make_scripted_policy(tag: str):
        if tag == "OP1":
            return OP1RedPolicy("red")
        if tag == "OP2":
            return OP2RedPolicy("red")
        return OP3RedPolicy("red")

    def collect_team_obs_and_avail(side: str) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
        """
        Returns:
          obs_agents: [N,C,H,W] ordered by agent_id
          avail_actions: [N, A] bool
          agents_sorted: list of agent objects in that order (None/disabled may exist)
        """
        if side == "blue":
            agents = list(env.blue_agents)
        else:
            agents = list(env.red_agents)

        try:
            agents.sort(key=lambda a: int(getattr(a, "agent_id", 0)))
        except Exception:
            pass

        obs_list: List[np.ndarray] = []
        avail_list: List[np.ndarray] = []

        for i in range(n_agents):
            if i < len(agents) and agents[i] is not None and agents[i].isEnabled():
                a = agents[i]
                obs = np.asarray(env.build_observation(a), dtype=np.float32)
                mm = compute_used_macro_mask(env, a, n_macros=N_MACROS)
                flat_mask = macro_mask_to_flat_action_mask(mm, n_targets)
                obs_list.append(obs)
                avail_list.append(flat_mask)
            else:
                obs_list.append(np.zeros((NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32))
                avail_list.append(np.zeros((n_actions,), dtype=np.bool_))

        return np.stack(obs_list, axis=0), np.stack(avail_list, axis=0), agents

    while global_step < total_steps:
        episode_idx += 1
        phase_episode_count += 1

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

        # pick opponent (with hold)
        if (not ENABLE_SELFPLAY) or hold_left <= 0 or held_opp is None:
            held_opp = pool.sample(cur_phase, opt_steps) if ENABLE_SELFPLAY else pool.entries[f"scripted:{cur_phase}"]
            hold_left = int(OPPONENT_HOLD_EPISODES)
        hold_left -= 1

        opponent_tag = held_opp.oid

        # configure opponent
        red_scripted = None
        if held_opp.kind == "scripted":
            red_scripted = make_scripted_policy(str(held_opp.scripted_tag))
            if hasattr(red_scripted, "reset"):
                try:
                    red_scripted.reset()
                except Exception:
                    pass
        else:
            assert held_opp.agent_state is not None
            opp_agent_net_act.load_state_dict(held_opp.agent_state)
            opp_agent_net_act.eval()

        env.reset_default()
        clear_reward_events_best_effort(gm)

        allowed_uids = init_episode_uid_set(env)
        pending: Dict[str, float] = {uid: 0.0 for uid in allowed_uids}

        prev_blue_score = int(getattr(gm, "blue_score", 0))
        prev_red_score = int(getattr(gm, "red_score", 0))

        done = False
        steps = 0

        step_return_total = 0.0
        losses: List[float] = []

        # initial blue obs/state for replay
        blue_obs_agents, blue_avail_actions, blue_agents_sorted = collect_team_obs_and_avail("blue")
        state = np.asarray(env.get_global_state(), dtype=np.float32).reshape(-1)

        while (not done) and steps < max_steps and global_step < total_steps:
            if not any(a is not None and a.isEnabled() for a in env.blue_agents):
                done = True
                break

            # --- BLUE selects actions (learner) ---
            eps = epsilon_by_step(global_step)
            blue_actions_flat = select_actions_qmix(
                agent_net_act=agent_net_act,
                obs_agents=blue_obs_agents,
                avail_actions=blue_avail_actions,
                eps=eps,
                device=ACT_DEVICE,
            )

            # --- RED selects actions (opponent) ---
            red_obs_agents, red_avail_actions, red_agents_sorted = collect_team_obs_and_avail("red")

            if held_opp.kind == "scripted":
                red_actions_flat = np.zeros((n_agents,), dtype=np.int64)
                for i in range(n_agents):
                    if i < len(red_agents_sorted) and red_agents_sorted[i] is not None and red_agents_sorted[i].isEnabled():
                        ra = red_agents_sorted[i]
                        macro_idx, targ_idx = scripted_red_action(
                            red_policy=red_scripted,
                            agent=ra,
                            env=env,
                            used_macros=USED_MACROS,
                            n_targets=n_targets,
                        )
                        red_actions_flat[i] = int(macro_idx * n_targets + targ_idx)
                    else:
                        red_actions_flat[i] = 0
            else:
                red_eps = 0.0 if RED_DETERMINISTIC else float(RED_EPS)
                red_actions_flat = select_actions_qmix(
                    agent_net_act=opp_agent_net_act,
                    obs_agents=red_obs_agents,
                    avail_actions=red_avail_actions,
                    eps=red_eps,
                    device=ACT_DEVICE,
                )

            # Submit BOTH sides as external actions
            submit_actions: Dict[str, Tuple[int, int]] = {}

            # blue
            for i in range(n_agents):
                if i < len(blue_agents_sorted) and blue_agents_sorted[i] is not None and blue_agents_sorted[i].isEnabled():
                    a = blue_agents_sorted[i]
                    a_flat = int(blue_actions_flat[i])
                    macro_idx = int(a_flat // n_targets)
                    targ_idx = int(a_flat % n_targets)
                    submit_actions[external_key_for_agent(a)] = (macro_idx, targ_idx)

            # red
            for i in range(n_agents):
                if i < len(red_agents_sorted) and red_agents_sorted[i] is not None and red_agents_sorted[i].isEnabled():
                    a = red_agents_sorted[i]
                    a_flat = int(red_actions_flat[i])
                    macro_idx = int(a_flat // n_targets)
                    targ_idx = int(a_flat % n_targets)
                    submit_actions[external_key_for_agent(a)] = (macro_idx, targ_idx)

            submit_external_actions_robust(env, submit_actions)

            # Step env
            sim_decision_window(env, gm, DECISION_WINDOW, SIM_DT)
            done = bool(gm.game_over)
            dt = compute_dt_for_window(DECISION_WINDOW)

            # Reward (BLUE only)
            accumulate_team_reward(gm, allowed_uids, pending)
            prev_blue_score, prev_red_score = apply_score_delta_shaping(
                gm, allowed_uids, pending, prev_blue_score, prev_red_score
            )
            reward = consume_team_reward(pending, allowed_uids, TIME_PENALTY_PER_AGENT_PER_MACRO)

            # Next obs/state
            next_blue_obs_agents, next_blue_avail_actions, _ = collect_team_obs_and_avail("blue")
            next_state = np.asarray(env.get_global_state(), dtype=np.float32).reshape(-1)

            if state.size != state_dim or next_state.size != state_dim:
                raise RuntimeError(f"State dim mismatch: got {state.size}->{next_state.size}, expected {state_dim}")

            # Store transition for BLUE only
            replay.add(
                Transition(
                    obs_agents=blue_obs_agents,
                    state=state,
                    actions=np.asarray(blue_actions_flat, dtype=np.int64),
                    avail_actions=blue_avail_actions.astype(np.uint8),
                    reward=float(reward),
                    done=bool(done),
                    next_obs_agents=next_blue_obs_agents,
                    next_state=next_state,
                    next_avail_actions=next_blue_avail_actions.astype(np.uint8),
                    dt=float(dt),
                )
            )

            blue_obs_agents = next_blue_obs_agents
            blue_avail_actions = next_blue_avail_actions
            state = next_state

            global_step += 1
            steps += 1
            step_return_total += float(reward)

            # Update
            if len(replay) >= WARMUP_STEPS and (global_step % UPDATE_EVERY == 0):
                batch = replay.sample(BATCH_SIZE)
                loss = qmix_update(
                    agent_net=agent_net,
                    mixer=mixer,
                    agent_net_tgt=agent_net_tgt,
                    mixer_tgt=mixer_tgt,
                    optimizer=optimizer,
                    batch=batch,
                    n_agents=n_agents,
                    n_actions=n_actions,
                )
                opt_steps += 1
                losses.append(loss)

                # sync act net
                agent_net_act.load_state_dict(agent_net.state_dict())
                agent_net_act.eval()

                # target update
                if opt_steps % TARGET_UPDATE_EVERY == 0:
                    if TAU_SOFT >= 1.0:
                        agent_net_tgt.load_state_dict(agent_net.state_dict())
                        mixer_tgt.load_state_dict(mixer.state_dict())
                    else:
                        soft_update_(agent_net_tgt, agent_net, TAU_SOFT)
                        soft_update_(mixer_tgt, mixer, TAU_SOFT)

                # snapshot into pool (agent only, for decentralized opponent play)
                if ENABLE_SELFPLAY and (opt_steps % int(SNAPSHOT_EVERY_OPT_STEPS) == 0):
                    pool.add_snapshot(agent_net.state_dict(), opt_step=opt_steps)

                avg_loss = float(np.mean(losses[-10:])) if losses else 0.0
                print(
                    f"[UPDATE] step={global_step} ep={episode_idx} phase={cur_phase} "
                    f"eps={epsilon_by_step(global_step):.3f} loss~{avg_loss:.4f} Opp={opponent_tag} opt={opt_steps}"
                )

        # Terminal bonus (BLUE)
        outcome_r = outcome_bonus_from_scores(gm.blue_score, gm.red_score)
        for uid in allowed_uids:
            pending[uid] = pending.get(uid, 0.0) + float(outcome_r)
        final_team_bonus = consume_team_reward(pending, allowed_uids, 0.0)
        step_return_total += float(final_team_bonus)

        # Result
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

        # PFSP stats update
        if ENABLE_SELFPLAY and held_opp is not None:
            pool.record_result(held_opp.oid, blue_win_bool)

        if len(phase_recent) > PHASE_WINRATE_WINDOW:
            phase_recent = phase_recent[-PHASE_WINRATE_WINDOW:]

        phase_wr = sum(phase_recent) / max(1, len(phase_recent))
        avg_step_r = step_return_total / max(1, steps)

        print(
            f"[{episode_idx:5d}] {result:8} | "
            f"StepR {avg_step_r:+.3f} | "
            f"Score {gm.blue_score}:{gm.red_score} | "
            f"BlueWins: {blue_wins} RedWins: {red_wins} Draws: {draws} | "
            f"PhaseWin {phase_wr * 100:.1f}% | Phase={cur_phase} Opp={opponent_tag}"
        )

        # Curriculum advance
        if cur_phase != PHASE_SEQUENCE[-1]:
            min_eps = MIN_PHASE_EPISODES[cur_phase]
            target_wr = TARGET_PHASE_WINRATE[cur_phase]
            if phase_episode_count >= min_eps and len(phase_recent) >= PHASE_WINRATE_WINDOW and phase_wr >= target_wr:
                print(f"[CURRICULUM] Advancing from {cur_phase} -> next phase.")
                phase_idx += 1
                phase_episode_count = 0
                phase_recent.clear()
                hold_left = 0
                held_opp = None

        # Checkpoint
        if episode_idx % 200 == 0:
            ckpt = os.path.join(CHECKPOINT_DIR, f"qmix_step{global_step}.pth")
            torch.save(
                {
                    "agent_net": agent_net.state_dict(),
                    "mixer": mixer.state_dict(),
                    "agent_net_tgt": agent_net_tgt.state_dict(),
                    "mixer_tgt": mixer_tgt.state_dict(),
                    "global_step": global_step,
                    "episode_idx": episode_idx,
                    "opt_steps": opt_steps,
                    "selfplay_pool_neural": len(pool.neural_ids),
                    "selfplay_latest": pool.latest_neural_id,
                },
                ckpt,
            )
            print(f"[CKPT] Saved: {ckpt}")
            if ENABLE_SELFPLAY and pool.latest_neural_id is not None:
                latest = pool.latest_neural_id
                print(f"[POOL] neural={len(pool.neural_ids)} latest={latest} latest_wr={pool.entries[latest].winrate():.2f}")

    final_path = os.path.join(CHECKPOINT_DIR, "qmix_final.pth")
    torch.save(
        {
            "agent_net": agent_net.state_dict(),
            "mixer": mixer.state_dict(),
            "agent_net_tgt": agent_net_tgt.state_dict(),
            "mixer_tgt": mixer_tgt.state_dict(),
            "global_step": global_step,
            "episode_idx": episode_idx,
            "opt_steps": opt_steps,
        },
        final_path,
    )
    print(f"\nTraining complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    train_qmix_event()
