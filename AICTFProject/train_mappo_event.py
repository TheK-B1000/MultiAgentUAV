"""
train_mappo_event.py

Multi-agent PPO (MAPPO-style) baseline for the 2-vs-2 UAV Capture-the-Flag (CTF) environment.

This script trains the BLUE team using an event-driven variant of MAPPO with:
  - Event-based reward aggregation and continuous-time GAE (matching the IHMC UAV CTF paper).
  - A macro-action interface (GO_TO, GET_FLAG, PLACE_MINE, etc.) over a CNN-based observation.
  - Curriculum over scripted RED opponents (OP1 → OP2 → OP3).
  - Optional self-play via "ghost" neural opponents sampled from past BLUE policies.
  - A cooperation HUD to monitor emergent role specialization and mine usage.

Dense potential-based proximity shaping is handled inside GameField.apply_macro_action /
GameManager (e.g., reward_potential_shaping). This trainer only consumes event rewards
from GameManager.reward_events.
"""

import os
import math
import random
import copy
from collections import deque
from typing import Dict, List, Tuple, Any, Deque, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================================
# DEVICE SELECTION (DirectML → CUDA → CPU)
# ==========================================================
#
# With an AMD Radeon (e.g., 5700 XT) on Windows, we use torch-directml
# to get GPU acceleration. On NVIDIA, we fall back to CUDA. Otherwise, CPU.
try:
    import torch_directml
    HAS_TDML = True
except ImportError:
    torch_directml = None
    HAS_TDML = False


def get_device() -> torch.device:
    """
    Prefer DirectML (AMD / any DX12 GPU), then CUDA, then CPU.
    """
    if HAS_TDML:
        # DirectML uses its own device type; this still behaves like a torch device.
        return torch_directml.device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


from game_field import GameField
from game_manager import GameManager
from macro_actions import MacroAction
from rl_policy import ActorCriticNet
from policies import OP1RedPolicy, OP2RedPolicy, OP3RedPolicy


# ======================================================================
# GLOBAL CONFIG & HYPERPARAMETERS
# ======================================================================

def set_seed(seed: int = 42) -> None:
    """
    Set all relevant RNG seeds for full determinism (as far as PyTorch allows).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Environment grid dimensions (paper-style 30x40 arena)
GRID_ROWS: int = 30
GRID_COLS: int = 40

# Use DirectML / CUDA / CPU in that order
DEVICE = get_device()

# PPO / MAPPO hyperparameters (paper-style)
TOTAL_STEPS: int = 3_000_000
UPDATE_EVERY: int = 2_048
PPO_EPOCHS: int = 10
MINIBATCH_SIZE: int = 256

LR: float = 3e-4
CLIP_EPS: float = 0.2
VALUE_COEF: float = 1.0
MAX_GRAD_NORM: float = 0.5

# Event-driven RL discounting (continuous in time, see paper)
GAMMA: float = 0.995
GAE_LAMBDA: float = 0.99

# Simulation timing (macro-actions last multiple physics steps)
DECISION_WINDOW: float = 0.7    # seconds between macro-decisions
SIM_DT: float = 0.1             # physics step size

CHECKPOINT_DIR: str = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Self-play / policy snapshotting
POLICY_SAVE_INTERVAL: int = UPDATE_EVERY
POLICY_SAMPLE_CHANCE: float = 0.15

# Cooperation HUD / diagnostics
COOP_HUD_EVERY: int = 50
COOP_WINDOW: int = 50

# Macro-action set for BLUE (and neural RED if used)
USED_MACROS: List[MacroAction] = [
    MacroAction.GO_TO,
    MacroAction.GRAB_MINE,
    MacroAction.GET_FLAG,
    MacroAction.PLACE_MINE,
    MacroAction.GO_HOME,
]
NUM_ACTIONS: int = len(USED_MACROS)
ACTION_NAMES: List[str] = [ma.name for ma in USED_MACROS]

# Curriculum phases (scripted opponents OP1 → OP2 → OP3)
PHASE_SEQUENCE: List[str] = ["OP1", "OP2", "OP3"]

# Minimum episodes per phase before considering an advance
MIN_PHASE_EPISODES: Dict[str, int] = {
    "OP1": 500,
    "OP2": 1000,
    "OP3": 2000,
}

# Target phase win-rates for curriculum advancement
TARGET_PHASE_WINRATE: Dict[str, float] = {
    "OP1": 0.90,
    "OP2": 0.86,
    "OP3": 0.80,
}

PHASE_WINRATE_WINDOW: int = 50

# Phase-specific (fixed) entropy coefficients (baseline values)
ENT_COEF_BY_PHASE: Dict[str, float] = {
    "OP1": 0.04,
    "OP2": 0.035,
    "OP3": 0.02,
}

# OP2-specific exploration / shaping tweaks
OP2_ENTROPY_FLOOR: float = 0.003     # don't let ent_coef collapse below this in OP2
OP2_DRAW_PENALTY: float = -0.4       # extra team penalty for 0–0 draws vs OP2
OP2_SCORE_BONUS: float = 0.5         # extra team bonus per blue score vs OP2

# Episode timing / max macro-steps per phase
PHASE_CONFIG: Dict[str, Dict[str, float]] = {
    "OP1": dict(score_limit=3, max_time=200.0, max_macro_steps=500),
    "OP2": dict(score_limit=3, max_time=200.0, max_macro_steps=450),
    "OP3": dict(score_limit=3, max_time=200.0, max_macro_steps=550),
}


# ======================================================================
# ENVIRONMENT SETUP & OPPONENT SELECTION
# ======================================================================

def set_red_policy_for_phase(env: GameField, phase: str) -> None:
    """
    Attach a scripted RED opponent corresponding to the current curriculum phase.
    """
    if phase == "OP1":
        env.policies["red"] = OP1RedPolicy("red")
    elif phase == "OP2":
        env.policies["red"] = OP2RedPolicy("red")
    elif phase == "OP3":
        env.policies["red"] = OP3RedPolicy("red")
    else:
        raise ValueError(f"Unknown phase: {phase}")


def make_env() -> GameField:
    """
    Create the 30x40 grid CTF environment.

    BLUE is controlled externally by this script via MAPPO.
    RED is controlled by internal policies (scripted OPx or a neural policy).
    """
    grid = [[0] * GRID_COLS for _ in range(GRID_ROWS)]
    env = GameField(grid)

    # Use env-internal policy dispatch for RED (scripted or neural)
    env.use_internal_policies = True

    # BLUE controlled externally; RED controlled internally
    env.set_external_control("blue", True)
    env.set_external_control("red", False)
    return env


# ======================================================================
# ROLLOUT BUFFER (CNN OBSERVATIONS, MULTI-AGENT)
# ======================================================================

class RolloutBuffer:
    """
    On-policy rollout buffer for MAPPO with event-driven time steps.

    Each stored time step corresponds to one macro-decision per BLUE agent:
      - obs: CNN observation [C,H,W] (local, ego-centric)
      - macro_action: macro-action index
      - target_action: discrete target index (e.g., which macro-target)
      - log_prob: log π(a|s)
      - value: V(s) (possibly centralized if the network uses more context)
      - reward: event-aggregated reward for this macro step
      - done: episode termination flag
      - dt: time until the next macro-decision (Δt_j)

    We simply treat all agent-time pairs as samples for PPO-style updates
    with a shared policy, i.e. MAPPO via parameter sharing.
    """

    def __init__(self) -> None:
        self.obs: List[np.ndarray] = []
        self.macro_actions: List[int] = []
        self.target_actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.dts: List[float] = []

    def add(
        self,
        obs: np.ndarray,
        macro_action: int,
        target_action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        dt: float,
    ) -> None:
        self.obs.append(np.array(obs, dtype=np.float32))
        self.macro_actions.append(int(macro_action))
        self.target_actions.append(int(target_action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.dts.append(float(dt))

    def size(self) -> int:
        return len(self.obs)

    def clear(self) -> None:
        self.__init__()

    def to_tensors(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        obs = torch.tensor(np.stack(self.obs), dtype=torch.float32, device=device)
        macro_actions = torch.tensor(self.macro_actions, dtype=torch.long, device=device)
        target_actions = torch.tensor(self.target_actions, dtype=torch.long, device=device)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        values = torch.tensor(self.values, dtype=torch.float32, device=device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)
        dts = torch.tensor(self.dts, dtype=torch.float32, device=device)
        return obs, macro_actions, target_actions, log_probs, values, rewards, dones, dts


# ======================================================================
# EVENT-DRIVEN GAE (CONTINUOUS TIME)
# ======================================================================

def compute_gae_event(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    dts: torch.Tensor,
    gamma: float = GAMMA,
    lam: float = GAE_LAMBDA,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Continuous-time GAE, matching the event-driven formulation:

      TD(t_j) = sum_{i in I_j} gamma^{t_i - t_j} r_i
                + gamma^{Δt_j} v(t_{j+1}) - v(t_j)

      A(t_j) = TD(t_j) + (lambda * gamma)^{Δt_j} A(t_{j+1}}

    Here:
      - rewards[j] == sum_{i in I_j} gamma^{t_i - t_j} r_i  (pre-discounted per decision)
      - dts[j] == Δt_j = t_{j+1} - t_j
    """
    T = rewards.size(0)
    advantages = torch.zeros(T, device=rewards.device)
    next_adv = 0.0
    next_value = 0.0

    for t in reversed(range(T)):
        gamma_dt = gamma ** dts[t]
        lam_gamma_dt = (gamma * lam) ** dts[t]
        mask = 1.0 - dones[t]

        # TD(t_j) using aggregated (already back-discounted) rewards:
        delta = rewards[t] + gamma_dt * next_value * mask - values[t]
        advantages[t] = delta + lam_gamma_dt * next_adv * mask

        next_adv = advantages[t]
        next_value = values[t]

    returns = advantages + values
    return advantages, returns


def mappo_update(
    policy: ActorCriticNet,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    device: torch.device,
    ent_coef: float,
) -> None:
    """
    Run multiple MAPPO (shared-parameter PPO) epochs over the collected rollout buffer.
    """
    (
        obs,
        macro_actions,
        target_actions,
        old_log_probs,
        values,
        rewards,
        dones,
        dts,
    ) = buffer.to_tensors(device)

    advantages, returns = compute_gae_event(rewards, values, dones, dts)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    idxs = np.arange(obs.size(0))
    for _ in range(PPO_EPOCHS):
        np.random.shuffle(idxs)
        for start in range(0, obs.size(0), MINIBATCH_SIZE):
            mb_idx = idxs[start:start + MINIBATCH_SIZE]
            mb_obs = obs[mb_idx]
            mb_macro = macro_actions[mb_idx]
            mb_target = target_actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_returns = returns[mb_idx]

            new_log_probs, entropy, new_values = policy.evaluate_actions(
                mb_obs,
                mb_macro,
                mb_target,
            )

            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (mb_returns - new_values).pow(2).mean()
            entropy_loss = entropy.mean()

            loss = policy_loss + VALUE_COEF * value_loss - ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

    buffer.clear()


# ======================================================================
# REWARD COLLECTION (BLUE, MULTI-AGENT)
# ======================================================================

def collect_blue_rewards_for_step(
    gm: GameManager,
    blue_agents: List[Any],
    decision_time_start: float,
    decision_time_end: float,
    cur_phase: str = "OP1",   # kept for future phase-specific shaping if needed
) -> Dict[str, float]:
    """
    Collect per-agent BLUE rewards from GameManager using timestamped events.

    For each macro-decision j, we compute:
      R_j = sum_{i in I_j} gamma^{t_i - t_j} r_i

    where I_j is the set of events that occurred since the last macro-decision.
    Dense proximity shaping, flag pickups, kills, etc. are all injected into
    GameManager.reward_events by the environment and GameManager.
    """

    # Each event: (timestamp, agent_id, reward_value)
    raw_events = list(gm.reward_events)
    gm.reward_events.clear()

    rewards_sum_by_id: Dict[str, float] = {a.unique_id: 0.0 for a in blue_agents}
    team_r_total = 0.0  # events with agent_id=None

    for t_event, agent_id, r in raw_events:
        # Discount reward back to the macro-decision time t_j
        time_since_action = t_event - decision_time_start
        if time_since_action < 0:
            time_since_action = 0.0

        discounted_r = r * (GAMMA ** time_since_action)

        if agent_id is None:
            team_r_total += discounted_r
        elif agent_id in rewards_sum_by_id:
            rewards_sum_by_id[agent_id] += discounted_r

    # Share team reward equally among active blue agents
    if abs(team_r_total) > 0.0 and len(rewards_sum_by_id) > 0:
        share = team_r_total / len(rewards_sum_by_id)
        for aid in rewards_sum_by_id:
            rewards_sum_by_id[aid] += share

    # ------------------------------------------------------
    # Extra phase-specific shaping:
    #   In OP2, punish "safe" 0–0 draws so the agents learn
    #   that stalling forever is worse than taking risks.
    # ------------------------------------------------------
    if (
        cur_phase == "OP2"
        and gm.game_over
        and gm.blue_score == gm.red_score
        and len(rewards_sum_by_id) > 0
    ):
        per_agent_pen = OP2_DRAW_PENALTY / len(rewards_sum_by_id)
        for aid in rewards_sum_by_id:
            rewards_sum_by_id[aid] += per_agent_pen

    return rewards_sum_by_id


def get_entropy_coef(
    cur_phase: str,
    phase_episode_count: int,
    phase_wr: float,
) -> float:
    """
    Simple entropy annealing schedule per phase.

    Starts above the baseline ENT_COEF_BY_PHASE value, then decays
    toward it over a horizon (in episodes). This keeps exploration
    higher early in each curriculum phase.

    In OP2, we also apply a small floor so exploration never collapses
    completely while the agents are learning to beat the defensive bot.
    """
    base = ENT_COEF_BY_PHASE[cur_phase]

    if cur_phase == "OP1":
        start_ent, horizon = 0.05, 1000.0
    elif cur_phase == "OP2":
        start_ent, horizon = 0.04, 1500.0
    else:
        start_ent, horizon = 0.035, 2000.0

    frac = min(1.0, phase_episode_count / horizon)
    coef = float(start_ent - (start_ent - base) * frac)

    if cur_phase == "OP2":
        # Don't let entropy become *too* small vs OP2.
        coef = max(coef, OP2_ENTROPY_FLOOR)

    return coef


# ======================================================================
# MAIN TRAINING LOOP (MAPPO)
# ======================================================================

def train_mappo_event(total_steps: int = TOTAL_STEPS) -> None:
    """
    Main training loop for the MAPPO-style baseline.

    BLUE is trained with shared-parameter PPO over both BLUE agents;
    RED is either scripted (OP1/OP2/OP3) or a neural "ghost" opponent.
    """
    set_seed(42)

    print(f"[train_mappo_event] Using device: {DEVICE}")

    env = make_env()
    gm = env.getGameManager()

    # Sanity-check CNN observation shape
    env.reset_default()
    if env.blue_agents:
        dummy_obs = env.build_observation(env.blue_agents[0])
        c = len(dummy_obs)
        h = len(dummy_obs[0])
        w = len(dummy_obs[0][0])
        print(f"[train_mappo_event] Sample obs shape: C={c}, H={h}, W={w}")
    else:
        print("[train_mappo_event] WARNING: No blue agents in env.reset_default().")

    # Shared policy & optimizer
    policy = ActorCriticNet(
        n_macros=len(USED_MACROS),
        n_targets=env.num_macro_targets,
    ).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    buffer = RolloutBuffer()

    global_step = 0
    episode_idx = 0

    blue_wins = red_wins = draws = 0

    # Curriculum tracking
    phase_idx = 0
    phase_episode_count = 0
    phase_recent: List[int] = []  # rolling [1,0,...] for phase win-rate
    phase_wr = 0.0

    # Rolling window for cooperation diagnostics
    coop_window: Deque[Dict[str, Any]] = deque(maxlen=COOP_WINDOW)

    # Buffer of old policies (for ghost self-play opponents)
    old_policies_buffer: Deque[Dict[str, Any]] = deque(maxlen=20)

    while global_step < total_steps:
        episode_idx += 1
        phase_episode_count += 1

        cur_phase = PHASE_SEQUENCE[phase_idx]
        phase_cfg = PHASE_CONFIG[cur_phase]
        gm.score_limit = phase_cfg["score_limit"]
        gm.max_time = phase_cfg["max_time"]
        max_steps = int(phase_cfg["max_macro_steps"])
        gm.set_phase(cur_phase)

        # --------------------------------------------------
        # OPPONENT SELECTION (SCRIPTED vs SELF-PLAY)
        # --------------------------------------------------
        use_selfplay = False

        ENABLE_SELFPLAY_PHASE = "OP3"
        MIN_EPISODES_FOR_SELFPLAY = 500
        MIN_WR_FOR_SELFPLAY = 0.70

        can_selfplay = (
            cur_phase == ENABLE_SELFPLAY_PHASE
            and phase_episode_count >= MIN_EPISODES_FOR_SELFPLAY
            and phase_wr >= MIN_WR_FOR_SELFPLAY
            and len(old_policies_buffer) > 0
        )

        if can_selfplay and random.random() < POLICY_SAMPLE_CHANCE:
            red_net = ActorCriticNet(
                n_macros=len(USED_MACROS),
                n_targets=env.num_macro_targets,
            ).to(DEVICE)
            state_dict = random.choice(old_policies_buffer)
            red_net.load_state_dict(state_dict)
            red_net.eval()
            env.set_red_policy_neural(red_net)
            opponent_tag = "SELFPLAY"
            use_selfplay = True
        else:
            set_red_policy_for_phase(env, cur_phase)
            opponent_tag = cur_phase

        # Reset environment & scores
        env.reset_default()
        gm.reset_game(reset_scores=True)

        # Reset OP3 per-episode state if present (for scripted opponents)
        red_pol = env.policies.get("red")
        if hasattr(red_pol, "reset"):
            red_pol.reset()

        # Clear per-episode mine stats
        gm.blue_mine_kills_this_episode = 0
        gm.red_mine_kills_this_episode = 0
        gm.mines_triggered_by_red_this_episode = 0

        done = False
        ep_return = 0.0
        steps = 0

        # Episode diagnostics
        ep_macro_counts: Dict[int, List[int]] = {0: [0] * NUM_ACTIONS, 1: [0] * NUM_ACTIONS}
        ep_mine_attempts: Dict[int, int] = {0: 0, 1: 0}
        ep_combat_events = 0
        ep_score_events = 0
        prev_blue_score = gm.blue_score
        ep_mines_placed_by_uid: Dict[str, int] = {}

        # Starting time for this macro-decision window
        decision_time_start = gm.get_sim_time()

        while not done and steps < max_steps and global_step < total_steps:
            # Save current BLUE policy periodically for ghost self-play
            if global_step > 0 and global_step % POLICY_SAVE_INTERVAL == 0:
                old_policies_buffer.append(copy.deepcopy(policy.state_dict()))

            blue_agents = [a for a in env.blue_agents if a.isEnabled()]
            decisions: List[Tuple[Any, Any, int, int, float, float, float]] = []

            # ==================================================
            # BLUE MACRO DECISIONS (MULTI-AGENT)
            # ==================================================
            for agent in blue_agents:
                obs = env.build_observation(agent)
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

                out = policy.act(
                    obs_tensor,
                    agent=agent,
                    game_field=env,
                    deterministic=False,
                )

                macro_idx = int(out["macro_action"][0].item())
                target_idx = int(out["target_action"][0].item())
                logp = float(out["log_prob"][0].item())
                val = float(out["value"][0].item())

                macro_enum = USED_MACROS[macro_idx]

                # Track macro-usage statistics
                local_id = getattr(agent, "agent_id", 0)
                if local_id in ep_macro_counts and 0 <= macro_idx < NUM_ACTIONS:
                    ep_macro_counts[local_id][macro_idx] += 1
                    if macro_enum == MacroAction.PLACE_MINE:
                        ep_mine_attempts[local_id] += 1

                if agent.unique_id not in ep_mines_placed_by_uid:
                    ep_mines_placed_by_uid[agent.unique_id] = 0

                # Distance to enemy flag BEFORE the macro (for shaping)
                side = agent.getSide()
                ex, ey = gm.get_enemy_flag_position(side)
                prev_flag_dist = math.dist([agent.x, agent.y], [ex, ey])

                # Apply macro-action (steers lower-level motion controller)
                env.apply_macro_action(agent, macro_enum, target_idx)

                decisions.append(
                    (agent, obs, macro_idx, target_idx, logp, val, prev_flag_dist)
                )

            # ==================================================
            # SIMULATE PHYSICS + INTERNAL POLICIES (RED, ETC.)
            # ==================================================
            sim_t = 0.0
            while sim_t < DECISION_WINDOW and not gm.game_over:
                env.update(SIM_DT)
                sim_t += SIM_DT

            dt = sim_t
            decision_time_end = gm.get_sim_time()
            done = gm.game_over

            # Potential-based flag proximity shaping
            for agent, _, _, _, _, _, prev_flag_dist in decisions:
                gm.reward_flag_proximity(agent, prev_flag_dist)

            # --------------------------------------------------
            # COLLECT EVENT-BASED REWARDS FOR THIS MACRO STEP
            # --------------------------------------------------
            rewards = collect_blue_rewards_for_step(
                gm,
                blue_agents,
                decision_time_start,
                decision_time_end,
                cur_phase=cur_phase,
            )

            # HUD: track mine stats per agent
            for agent, _, macro_idx, _, _, _, _ in decisions:
                uid = agent.unique_id
                macro_enum = USED_MACROS[macro_idx]
                if macro_enum == MacroAction.PLACE_MINE:
                    ep_mines_placed_by_uid[uid] = ep_mines_placed_by_uid.get(uid, 0) + 1

            # HUD: track scoring & combat events
            blue_score_delta = gm.blue_score - prev_blue_score
            if blue_score_delta > 0:
                ep_score_events += blue_score_delta
            prev_blue_score = gm.blue_score

            total_team_reward = sum(rewards.values())
            if total_team_reward > 0.0 and blue_score_delta == 0:
                ep_combat_events += 1

            # --------------------------------------------------
            # Extra OP2 shaping: reward actual scoring
            # --------------------------------------------------
            if cur_phase == "OP2" and blue_score_delta > 0:
                # Small positive team bonus for scoring vs a defensive opponent
                n_agents = len(rewards)
                if n_agents > 0:
                    per_agent_bonus = (OP2_SCORE_BONUS * blue_score_delta) / n_agents
                    for aid in rewards:
                        rewards[aid] = rewards.get(aid, 0.0) + per_agent_bonus

            # --------------------------------------------------
            # ADD TO MAPPO BUFFER (ONE SAMPLE PER AGENT)
            # --------------------------------------------------
            step_reward_sum = 0.0
            for agent, obs, macro_idx, target_idx, logp, val, _ in decisions:
                r = rewards.get(agent.unique_id, 0.0)
                step_reward_sum += r
                buffer.add(obs, macro_idx, target_idx, logp, val, r, done, dt)
                global_step += 1

            ep_return += step_reward_sum
            steps += 1

            # Next macro-decision window starts here
            decision_time_start = decision_time_end

            # MAPPO update when buffer is full
            if buffer.size() >= UPDATE_EVERY:
                current_ent_coef = get_entropy_coef(cur_phase, phase_episode_count, phase_wr)

                print(
                    f"[MAPPO UPDATE] step={global_step} episode={episode_idx} "
                    f"phase={cur_phase} ENT={current_ent_coef:.4f} Opp={opponent_tag}"
                )

                mappo_update(policy, optimizer, buffer, DEVICE, current_ent_coef)

        # ==================================================
        # EPISODE END: RESULT + STATS
        # ==================================================
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

        # ==================================================
        # COOPERATION HUD (ROLE SPECIALIZATION / MINES)
        # ==================================================
        miner0_runner1 = (
            ep_score_events > 0 and ep_mine_attempts[0] > 0 and ep_mine_attempts[1] == 0
        )
        miner1_runner0 = (
            ep_score_events > 0 and ep_mine_attempts[1] > 0 and ep_mine_attempts[0] == 0
        )
        both_mine_and_score = (
            ep_score_events > 0 and ep_mine_attempts[0] > 0 and ep_mine_attempts[1] > 0
        )

        coop_window.append(
            {
                "macro_counts": ep_macro_counts,
                "mine_attempts": ep_mine_attempts,
                "combat_events": ep_combat_events,
                "score_events": ep_score_events,
                "miner0_runner1": 1 if miner0_runner1 else 0,
                "miner1_runner0": 1 if miner1_runner0 else 0,
                "both_mine_and_score": 1 if both_mine_and_score else 0,
                "blue_mine_kills": gm.blue_mine_kills_this_episode,
                "red_mine_kills": gm.red_mine_kills_this_episode,
                "mines_placed_in_enemy_half": gm.mines_placed_in_enemy_half_this_episode,
                "mines_triggered_by_red": gm.mines_triggered_by_red_this_episode,
            }
        )

        if episode_idx % COOP_HUD_EVERY == 0 and coop_window:
            agg_mines = {0: 0, 1: 0}
            agg_scores = 0
            agg_combat = 0
            agg_macros = {0: [0] * NUM_ACTIONS, 1: [0] * NUM_ACTIONS}
            agg_miner0_runner1 = 0
            agg_miner1_runner0 = 0
            agg_both_mine_score = 0

            for ep in coop_window:
                for pid in (0, 1):
                    agg_mines[pid] += ep["mine_attempts"].get(pid, 0)
                    mc = ep["macro_counts"].get(pid, [0] * NUM_ACTIONS)
                    for i in range(NUM_ACTIONS):
                        agg_macros[pid][i] += mc[i]
                agg_scores += ep["score_events"]
                agg_combat += ep["combat_events"]
                agg_miner0_runner1 += ep["miner0_runner1"]
                agg_miner1_runner0 += ep["miner1_runner0"]
                agg_both_mine_score += ep["both_mine_and_score"]

            window_len = float(len(coop_window))
            avg_mines_0 = agg_mines[0] / window_len
            avg_mines_1 = agg_mines[1] / window_len
            avg_scores = agg_scores / window_len
            avg_combat = agg_combat / window_len

            total_dec_0 = max(1, sum(agg_macros[0]))
            total_dec_1 = max(1, sum(agg_macros[1]))
            pct_0 = [100.0 * c / total_dec_0 for c in agg_macros[0]]
            pct_1 = [100.0 * c / total_dec_1 for c in agg_macros[1]]

            blue_kills = sum(ep["blue_mine_kills"] for ep in coop_window) / window_len
            red_kills = sum(ep["red_mine_kills"] for ep in coop_window) / window_len
            enemy_half = sum(ep["mines_placed_in_enemy_half"] for ep in coop_window) / window_len
            triggered = sum(ep["mines_triggered_by_red"] for ep in coop_window) / window_len
            total_mines = avg_mines_0 + avg_mines_1

            print("   ================== COOP HUD (MAPPO) ==================")
            print(f"   Window: last {len(coop_window)} episodes")
            print(f"   Avg mines/ep          : Blue0={avg_mines_0:.2f}, Blue1={avg_mines_1:.2f}")
            print(f"   Avg scores/ep         : {avg_scores:.2f}")
            print(f"   Avg combat-events/ep  : {avg_combat:.2f}")

            print(f"   {'MINE EFFECTIVENESS':_^58}")
            print(f"   Blue mine kills/ep    : {blue_kills:5.2f}")
            print(f"   Red mine kills/ep     : {red_kills:5.2f}")
            print(f"   Mines in enemy half/ep: {enemy_half:5.2f}")
            print(f"   Mines triggered by Red: {triggered:5.2f}")
            if total_mines > 0.1:
                kill_efficiency = blue_kills / total_mines * 100
                line = f"   Kill/Mine Efficiency  : {kill_efficiency:5.1f}%  "
                if kill_efficiency > 40:
                    line += "INSANE"
                elif kill_efficiency > 25:
                    line += "GREAT"
                elif kill_efficiency > 15:
                    line += "GOOD"
                else:
                    line += "WASTED"
                print(line)
            else:
                print("   Kill/Mine Efficiency  :  N/A (no mines)")

            print("   Role breakdown (macro-action usage %) over window:")
            for i, name in enumerate(ACTION_NAMES):
                p0 = pct_0[i] if i < len(pct_0) else 0.0
                p1 = pct_1[i] if i < len(pct_1) else 0.0
                print(f"      {name:20s} | Blue0 {p0:5.1f}% | Blue1 {p1:5.1f}%")
            print("   =====================================================")

        # ==================================================
        # CURRICULUM ADVANCEMENT
        # ==================================================
        if cur_phase != PHASE_SEQUENCE[-1]:
            min_eps = MIN_PHASE_EPISODES[cur_phase]
            target_wr = TARGET_PHASE_WINRATE[cur_phase]
            if (
                phase_episode_count >= min_eps
                and len(phase_recent) >= PHASE_WINRATE_WINDOW
                and phase_wr >= target_wr
            ):
                print(f"[CURRICULUM] Advancing from {cur_phase} → next phase (MAPPO).")
                phase_idx += 1
                phase_episode_count = 0
                phase_recent.clear()
                old_policies_buffer.clear()

    # Save final trained BLUE MAPPO policy
    final_path = os.path.join(CHECKPOINT_DIR, "research_mappo_model1.pth")
    torch.save(policy.state_dict(), final_path)
    print(f"\n[MAPPO] Training complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    train_mappo_event()
