# ==========================================================
# train_qmix_event.py (UPDATED: GLOBAL "GOD VIEW" STATE)
#   - Uses env.get_global_state() for mixer state input
#   - state_dim now based on 8 * CNN_ROWS * CNN_COLS (default 3200)
#   - Event-driven dt-aware discounting (gamma ** dt)
#   - Masked epsilon-greedy for valid macro/target combos
#   - CTDE: decentralized agent obs, centralized mixer state
# ==========================================================

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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

# If you have this in your codebase already, use it (recommended).
# Otherwise, replace AgentQNet.encoder with your own CNN.
from obs_encoder import ObsEncoder


# ==========================================================
# Device
# ==========================================================
def get_act_device() -> torch.device:
    # For action selection forward-only.
    if HAS_TDML:
        return torch_directml.device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


ACT_DEVICE = get_act_device()
TRAIN_DEVICE = torch.device("cpu")  # safest for backward if TDML is flaky


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
TOTAL_STEPS: int = 1_000_000
UPDATE_EVERY: int = 2048         # transitions
BATCH_SIZE: int = 256
REPLAY_CAPACITY: int = 200_000
WARMUP_STEPS: int = 10_000

LR: float = 3e-4
MAX_GRAD_NORM: float = 10.0

GAMMA: float = 0.995
DECISION_WINDOW: float = 1.0
SIM_DT: float = 0.1

# QMIX
MIX_EMBED_DIM: int = 32

# Epsilon schedule
EPS_START: float = 1.0
EPS_END: float = 0.05
EPS_DECAY_STEPS: int = 250_000  # linear decay horizon

# Target network updates
TARGET_UPDATE_EVERY: int = 2000  # optimizer steps
TAU_SOFT: float = 1.0            # set <1.0 for soft updates; 1.0 == hard copy

CHECKPOINT_DIR: str = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Macros (must match GameField.macro_order)
USED_MACROS: List[MacroAction] = [
    MacroAction.GO_TO,
    MacroAction.GRAB_MINE,
    MacroAction.GET_FLAG,
    MacroAction.PLACE_MINE,
    MacroAction.GO_HOME,
]
N_MACROS: int = len(USED_MACROS)

# Curriculum
PHASE_SEQUENCE: List[str] = ["OP1", "OP2", "OP3"]
MIN_PHASE_EPISODES: Dict[str, int] = {"OP1": 300, "OP2": 700, "OP3": 1500}
TARGET_PHASE_WINRATE: Dict[str, float] = {"OP1": 0.74, "OP2": 0.70, "OP3": 0.65}
PHASE_WINRATE_WINDOW: int = 50
PHASE_CONFIG: Dict[str, Dict[str, float]] = {
    "OP1": dict(score_limit=3, max_time=200.0, max_macro_steps=500),
    "OP2": dict(score_limit=3, max_time=200.0, max_macro_steps=500),
    "OP3": dict(score_limit=3, max_time=200.0, max_macro_steps=500),
}

# Objective shaping (optional but usually helps)
WIN_BONUS: float = 25.0
LOSS_PENALTY: float = -25.0
DRAW_PENALTY: float = -5.0

BLUE_SCORED_BONUS: float = 15.0
RED_SCORED_PENALTY: float = -15.0

TIME_PENALTY_PER_AGENT_PER_MACRO: float = 0.001


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
    grid = [[0] * CNN_COLS for _ in range(CNN_ROWS)]
    env = GameField(grid)
    env.use_internal_policies = True

    if hasattr(env, "set_external_control"):
        env.set_external_control("blue", True)
        env.set_external_control("red", False)

    if hasattr(env, "external_missing_action_mode"):
        env.external_missing_action_mode = "idle"

    return env


def agent_uid(agent: Any) -> str:
    uid = getattr(agent, "unique_id", None) or getattr(agent, "slot_id", None)
    if uid is None:
        side = getattr(agent, "side", "blue")
        aid = getattr(agent, "agent_id", 0)
        uid = f"{side}_{aid}"
    return str(uid)


def external_key_for_agent(agent: Any) -> str:
    # GameField already accepts side_agentId keys robustly.
    return f"{getattr(agent, 'side', 'blue')}_{int(getattr(agent, 'agent_id', 0))}"


def apply_blue_actions(env: GameField, actions_by_agent: Dict[str, Tuple[int, int]]) -> None:
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
        uids.append(agent_uid(a))
    return set(uids)


def accumulate_team_reward(
    gm: GameManager,
    allowed_uids: set,
    pending: Dict[str, float],
) -> None:
    events = pop_reward_events_strict(gm)
    for _t, agent_id, r in events:
        if agent_id is None:
            continue
        key = str(agent_id).strip()
        if (key in allowed_uids):
            pending[key] = pending.get(key, 0.0) + float(r)


def consume_team_reward(
    pending: Dict[str, float],
    allowed_uids: set,
    time_penalty_per_agent: float,
) -> float:
    team_r = 0.0
    for uid in allowed_uids:
        team_r += float(pending.get(uid, 0.0))
        pending[uid] = 0.0
    # per-agent per-decision penalty (keeps pressure to finish)
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
def compute_used_macro_mask(env: GameField, agent: Any) -> np.ndarray:
    mm = env.get_macro_mask(agent)
    mm = np.asarray(mm, dtype=np.bool_).reshape(-1)
    if mm.shape != (N_MACROS,):
        raise RuntimeError(f"env.get_macro_mask returned {mm.shape}, expected {(N_MACROS,)}")
    if not mm.any():
        mm[:] = True
    return mm


def macro_mask_to_flat_action_mask(macro_mask: np.ndarray, n_targets: int) -> np.ndarray:
    # flat action = macro_idx * n_targets + target_idx
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

        # Small init helps stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs_bchw: torch.Tensor) -> torch.Tensor:
        z = self.encoder(obs_bchw.contiguous())
        return self.head(z)


class QMixer(nn.Module):
    """
    Standard QMIX mixer with hypernetworks:
      q_tot = w(s) @ q_agents + b(s)
    Constraints: w >= 0 enforced via abs()
    """
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
        """
        agent_qs: [B, N]
        states:   [B, state_dim]
        returns:  [B, 1]
        """
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
    obs_agents: np.ndarray          # [N,C,H,W]
    state: np.ndarray               # [S]
    actions: np.ndarray             # [N] (flat)
    avail_actions: np.ndarray       # [N,A] bool/uint8
    reward: float
    done: bool
    next_obs_agents: np.ndarray     # [N,C,H,W]
    next_state: np.ndarray          # [S]
    next_avail_actions: np.ndarray  # [N,A] bool/uint8
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
# Action selection
# ==========================================================
@torch.no_grad()
def select_actions_qmix(
    agent_net_act: AgentQNet,
    obs_agents: np.ndarray,              # [N,C,H,W]
    avail_actions: np.ndarray,           # [N,A] bool
    eps: float,
) -> np.ndarray:
    """
    Returns flat actions [N].
    Masked epsilon-greedy.
    """
    N = obs_agents.shape[0]
    A = avail_actions.shape[1]
    actions = np.zeros((N,), dtype=np.int64)

    obs_t = torch.tensor(obs_agents, dtype=torch.float32, device=ACT_DEVICE)  # [N,C,H,W]
    q = agent_net_act(obs_t).detach().cpu().numpy()                           # [N,A]

    for i in range(N):
        mask = avail_actions[i].astype(np.bool_)
        if not mask.any():
            actions[i] = 0
            continue

        if random.random() < eps:
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

    obs_agents = np.stack([tr.obs_agents for tr in batch], axis=0)              # [B,N,C,H,W]
    next_obs_agents = np.stack([tr.next_obs_agents for tr in batch], axis=0)    # [B,N,C,H,W]
    states = np.stack([tr.state for tr in batch], axis=0)                       # [B,S]
    next_states = np.stack([tr.next_state for tr in batch], axis=0)             # [B,S]
    actions = np.stack([tr.actions for tr in batch], axis=0)                    # [B,N]
    avail = np.stack([tr.avail_actions for tr in batch], axis=0).astype(np.bool_)         # [B,N,A]
    next_avail = np.stack([tr.next_avail_actions for tr in batch], axis=0).astype(np.bool_) # [B,N,A]
    rewards = np.array([tr.reward for tr in batch], dtype=np.float32)           # [B]
    dones = np.array([tr.done for tr in batch], dtype=np.float32)               # [B]
    dts = np.array([tr.dt for tr in batch], dtype=np.float32)                   # [B]

    # tensors
    obs_t = torch.tensor(obs_agents, dtype=torch.float32, device=TRAIN_DEVICE)          # [B,N,C,H,W]
    next_obs_t = torch.tensor(next_obs_agents, dtype=torch.float32, device=TRAIN_DEVICE)
    s_t = torch.tensor(states, dtype=torch.float32, device=TRAIN_DEVICE)                # [B,S]
    s2_t = torch.tensor(next_states, dtype=torch.float32, device=TRAIN_DEVICE)
    a_t = torch.tensor(actions, dtype=torch.long, device=TRAIN_DEVICE)                  # [B,N]
    r_t = torch.tensor(rewards, dtype=torch.float32, device=TRAIN_DEVICE).view(B, 1)    # [B,1]
    d_t = torch.tensor(dones, dtype=torch.float32, device=TRAIN_DEVICE).view(B, 1)      # [B,1]
    dt_t = torch.tensor(dts, dtype=torch.float32, device=TRAIN_DEVICE).view(B, 1)       # [B,1]

    # --- current Q_tot ---
    obs_flat = obs_t.view(B * n_agents, NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS)
    q_all = agent_net(obs_flat).view(B, n_agents, n_actions)                            # [B,N,A]

    q_chosen = torch.gather(q_all, dim=2, index=a_t.unsqueeze(-1)).squeeze(-1)           # [B,N]
    q_tot = mixer(q_chosen, s_t)                                                        # [B,1]

    # --- target Q_tot (Double-Q) ---
    with torch.no_grad():
        next_obs_flat = next_obs_t.view(B * n_agents, NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS)

        q_next_online = agent_net(next_obs_flat).view(B, n_agents, n_actions)           # [B,N,A]
        # mask invalid actions
        mask_next = torch.tensor(next_avail, dtype=torch.bool, device=TRAIN_DEVICE)     # [B,N,A]
        q_next_online = q_next_online.masked_fill(~mask_next, -1e9)
        a_star = q_next_online.argmax(dim=2)                                            # [B,N]

        q_next_tgt = agent_net_tgt(next_obs_flat).view(B, n_agents, n_actions)
        q_next_tgt = q_next_tgt.masked_fill(~mask_next, -1e9)
        q_next_sel = torch.gather(q_next_tgt, dim=2, index=a_star.unsqueeze(-1)).squeeze(-1)  # [B,N]

        q_tot_next = mixer_tgt(q_next_sel, s2_t)                                        # [B,1]

        gamma_dt = (float(GAMMA) ** dt_t)                                               # [B,1]
        y = r_t + gamma_dt * (1.0 - d_t) * q_tot_next

    loss = (q_tot - y).pow(2).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(list(agent_net.parameters()) + list(mixer.parameters()), MAX_GRAD_NORM)
    optimizer.step()

    return float(loss.item())


# ==========================================================
# MAIN TRAIN LOOP (UPDATED: env.get_global_state())
# ==========================================================
def train_qmix_event(total_steps: int = TOTAL_STEPS) -> None:
    set_seed(42)

    env = make_env()
    gm = env.getGameManager()

    # shaping gamma (if supported)
    if hasattr(gm, "set_shaping_gamma"):
        gm.set_shaping_gamma(GAMMA)

    # Reset env once to probe dims
    env.reset_default()
    if not env.blue_agents:
        raise RuntimeError("No blue agents after env.reset_default().")

    # Basic obs shape
    dummy_obs = np.array(env.build_observation(env.blue_agents[0]), dtype=np.float32)  # [C,H,W]
    C, H, W = int(dummy_obs.shape[0]), int(dummy_obs.shape[1]), int(dummy_obs.shape[2])
    print(f"[train_qmix_event] Sample obs shape: C={C}, H={H}, W={W}")

    # Targets (action space)
    n_targets = int(getattr(env, "num_macro_targets", 8))
    n_actions = int(N_MACROS * n_targets)
    n_agents = int(getattr(env, "agents_per_team", 2))

    # ----------------------------------------------------------
    # UPDATED: Global state dimension for mixer
    # Per your request:
    #   state_dim = 8 * CNN_ROWS * CNN_COLS  # 3200
    #
    # We still sanity-check against env.get_global_state() shape.
    # ----------------------------------------------------------
    state_dim_requested = int(8 * CNN_ROWS * CNN_COLS)

    st0 = env.get_global_state()
    st0 = np.asarray(st0, dtype=np.float32)
    if st0.ndim == 3:
        st0 = st0.reshape(-1)
    state_dim_env = int(st0.size)

    state_dim = int(state_dim_requested)
    if state_dim_env != state_dim_requested:
        # Don’t crash: trust the environment output dimension.
        print(f"[WARN] env.get_global_state() dim={state_dim_env} != requested {state_dim_requested}. Using {state_dim_env}.")
        state_dim = state_dim_env

    print(f"[train_qmix_event] n_agents={n_agents} n_targets={n_targets} n_actions={n_actions} state_dim={state_dim}")

    # Networks
    agent_net = AgentQNet(n_actions=n_actions, in_channels=NUM_CNN_CHANNELS, height=CNN_ROWS, width=CNN_COLS).to(TRAIN_DEVICE)
    mixer = QMixer(n_agents=n_agents, state_dim=state_dim, embed_dim=MIX_EMBED_DIM).to(TRAIN_DEVICE)

    agent_net_tgt = AgentQNet(n_actions=n_actions, in_channels=NUM_CNN_CHANNELS, height=CNN_ROWS, width=CNN_COLS).to(TRAIN_DEVICE)
    mixer_tgt = QMixer(n_agents=n_agents, state_dim=state_dim, embed_dim=MIX_EMBED_DIM).to(TRAIN_DEVICE)

    agent_net_tgt.load_state_dict(agent_net.state_dict())
    mixer_tgt.load_state_dict(mixer.state_dict())

    # Separate "act" net on ACT_DEVICE (forward-only)
    agent_net_act = AgentQNet(n_actions=n_actions, in_channels=NUM_CNN_CHANNELS, height=CNN_ROWS, width=CNN_COLS).to(ACT_DEVICE)
    agent_net_act.load_state_dict(agent_net.state_dict())
    agent_net_act.eval()

    optimizer = optim.Adam(list(agent_net.parameters()) + list(mixer.parameters()), lr=LR, foreach=False)

    replay = ReplayBuffer(REPLAY_CAPACITY)

    global_step = 0
    episode_idx = 0
    opt_steps = 0

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

        env.reset_default()
        clear_reward_events_best_effort(gm)

        allowed_uids = init_episode_uid_set(env)
        pending: Dict[str, float] = {uid: 0.0 for uid in allowed_uids}

        # score shaping trackers
        prev_blue_score = int(getattr(gm, "blue_score", 0))
        prev_red_score = int(getattr(gm, "red_score", 0))

        done = False
        steps = 0

        # Logging
        step_return_total = 0.0
        losses: List[float] = []

        # Build initial per-agent obs
        def collect_obs_agents() -> Tuple[np.ndarray, np.ndarray]:
            # return obs_agents [N,C,H,W], avail_actions [N,A]
            blue_agents = [a for a in env.blue_agents if a.isEnabled()]
            if len(blue_agents) < n_agents:
                # pad missing agents with zeros obs and "no actions"
                pass

            obs_list: List[np.ndarray] = []
            avail_list: List[np.ndarray] = []

            # Ensure consistent ordering by agent_id
            agents_sorted = list(env.blue_agents)
            try:
                agents_sorted.sort(key=lambda a: int(getattr(a, "agent_id", 0)))
            except Exception:
                pass

            for i in range(n_agents):
                if i < len(agents_sorted) and agents_sorted[i] is not None and agents_sorted[i].isEnabled():
                    a = agents_sorted[i]
                    obs = np.asarray(env.build_observation(a), dtype=np.float32)  # [C,H,W]
                    mm = compute_used_macro_mask(env, a)                         # [N_MACROS]
                    flat_mask = macro_mask_to_flat_action_mask(mm, n_targets)    # [A]
                    obs_list.append(obs)
                    avail_list.append(flat_mask)
                else:
                    obs_list.append(np.zeros((NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32))
                    avail_list.append(np.zeros((n_actions,), dtype=np.bool_))

            obs_agents = np.stack(obs_list, axis=0)            # [N,C,H,W]
            avail_actions = np.stack(avail_list, axis=0)       # [N,A]
            return obs_agents, avail_actions

        obs_agents, avail_actions = collect_obs_agents()

        # UPDATED: Global state from environment (truth)
        state = np.asarray(env.get_global_state(), dtype=np.float32)
        if state.ndim == 3:
            state = state.reshape(-1)

        while (not done) and steps < max_steps and global_step < total_steps:
            # If all disabled, end episode
            if not any(a.isEnabled() for a in env.blue_agents):
                done = True
                break

            eps = epsilon_by_step(global_step)

            # Choose actions (flat)
            actions_flat = select_actions_qmix(agent_net_act, obs_agents, avail_actions, eps=eps)  # [N]

            # Submit actions to env (macro_idx, target_idx)
            submit_actions: Dict[str, Tuple[int, int]] = {}
            # Use same ordering as collect_obs_agents
            agents_sorted = list(env.blue_agents)
            try:
                agents_sorted.sort(key=lambda a: int(getattr(a, "agent_id", 0)))
            except Exception:
                pass

            for i in range(n_agents):
                if i < len(agents_sorted) and agents_sorted[i] is not None and agents_sorted[i].isEnabled():
                    a = agents_sorted[i]
                    a_flat = int(actions_flat[i])
                    macro_idx = int(a_flat // n_targets)
                    targ_idx = int(a_flat % n_targets)

                    key = external_key_for_agent(a)
                    submit_actions[key] = (macro_idx, targ_idx)

            apply_blue_actions(env, submit_actions)

            # Simulate one decision window
            sim_decision_window(env, gm, DECISION_WINDOW, SIM_DT)
            done = bool(gm.game_over)
            dt = compute_dt_for_window(DECISION_WINDOW)

            # Rewards (event-driven)
            accumulate_team_reward(gm, allowed_uids, pending)
            prev_blue_score, prev_red_score = apply_score_delta_shaping(
                gm, allowed_uids, pending, prev_blue_score, prev_red_score
            )
            reward = consume_team_reward(pending, allowed_uids, TIME_PENALTY_PER_AGENT_PER_MACRO)

            # Next obs/state
            next_obs_agents, next_avail_actions = collect_obs_agents()

            # UPDATED: Global next-state from environment
            next_state = np.asarray(env.get_global_state(), dtype=np.float32)
            if next_state.ndim == 3:
                next_state = next_state.reshape(-1)

            # If state dims differ (shouldn't), fail loudly
            if state.size != state_dim or next_state.size != state_dim:
                raise RuntimeError(
                    f"State dim mismatch: got {state.size}->{next_state.size}, expected {state_dim}"
                )

            # Store transition
            replay.add(
                Transition(
                    obs_agents=obs_agents,
                    state=state,
                    actions=np.asarray(actions_flat, dtype=np.int64),
                    avail_actions=avail_actions.astype(np.uint8),
                    reward=float(reward),
                    done=bool(done),
                    next_obs_agents=next_obs_agents,
                    next_state=next_state,
                    next_avail_actions=next_avail_actions.astype(np.uint8),
                    dt=float(dt),
                )
            )

            # advance
            obs_agents = next_obs_agents
            avail_actions = next_avail_actions
            state = next_state

            global_step += 1
            steps += 1
            step_return_total += float(reward)

            # Train
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

                # Sync act net (cheap hard copy)
                agent_net_act.load_state_dict(agent_net.state_dict())
                agent_net_act.eval()

                # Target update
                if opt_steps % TARGET_UPDATE_EVERY == 0:
                    if TAU_SOFT >= 1.0:
                        agent_net_tgt.load_state_dict(agent_net.state_dict())
                        mixer_tgt.load_state_dict(mixer.state_dict())
                    else:
                        soft_update_(agent_net_tgt, agent_net, TAU_SOFT)
                        soft_update_(mixer_tgt, mixer, TAU_SOFT)

                avg_loss = float(np.mean(losses[-10:])) if losses else 0.0
                print(
                    f"[UPDATE] step={global_step} ep={episode_idx} phase={cur_phase} "
                    f"eps={epsilon_by_step(global_step):.3f} loss~{avg_loss:.4f} Opp={opponent_tag}"
                )

        # Episode end: add outcome bonus into pending and consume once (team)
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

        # Occasional checkpoint
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
                },
                ckpt,
            )
            print(f"[CKPT] Saved: {ckpt}")

    final_path = os.path.join(CHECKPOINT_DIR, "qmix_final.pth")
    torch.save(
        {
            "agent_net": agent_net.state_dict(),
            "mixer": mixer.state_dict(),
            "agent_net_tgt": agent_net_tgt.state_dict(),
            "mixer_tgt": mixer_tgt.state_dict(),
            "global_step": global_step,
            "episode_idx": episode_idx,
        },
        final_path,
    )
    print(f"\nTraining complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    train_qmix_event()
