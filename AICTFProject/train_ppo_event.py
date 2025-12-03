import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game_field import GameField
from game_manager import GameManager
from macro_actions import MacroAction
from rl_policy import ActorCriticNet
from policies import HeuristicPolicy, OP1RedPolicy, OP2RedPolicy, OP3RedPolicy


# =========================
# BASIC CONFIG (PAPER-STYLE)
# =========================

GRID_ROWS = 30
GRID_COLS = 40

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO hyperparameters
TOTAL_STEPS = 2_500_000       # safety cap
UPDATE_EVERY = 2_048          # steps per PPO update
PPO_EPOCHS = 10
MINIBATCH_SIZE = 256

LR = 3e-4                     # PPO learning rate
CLIP_EPS = 0.2                # PPO clipping epsilon
VALUE_COEF = 1.0              # c1
MAX_GRAD_NORM = 0.5

# Event-driven RL discounting
GAMMA = 0.995                 # γ
GAE_LAMBDA = 0.99             # λ

# Event-driven timing:
DECISION_WINDOW = 0.7         # max simulated seconds between macro decisions
SIM_DT = 0.1                  # inner sim dt

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =========================
# CURRICULUM: OP1 -> OP2 -> OP3
# =========================

PHASE_SEQUENCE = ["OP1", "OP2", "OP3"]  # no SELF

MIN_PHASE_EPISODES = {
    "OP1": 300,
    "OP2": 300,
    "OP3": 500,   # stay longer vs OP3
}

TARGET_PHASE_WINRATE = {
    "OP1": 0.30,  # loosened for debugging; tighten later
    "OP2": 0.50,
    "OP3": 0.70,
}

PHASE_WINRATE_WINDOW = 50

ENT_COEF_BY_PHASE = {
    "OP1": 0.03,
    "OP2": 0.02,
    "OP3": 0.015,
}

# Per-phase episode config (to avoid OP2 farming forever)
PHASE_CONFIG = {
    "OP1": dict(score_limit=3, max_time=200.0, max_macro_steps=500),
    "OP2": dict(score_limit=2, max_time=150.0, max_macro_steps=350),  # shorter & harsher
    "OP3": dict(score_limit=3, max_time=200.0, max_macro_steps=500),
}


def set_red_policy_for_phase(env: GameField, phase: str) -> None:
    """Swap the RED policy according to the current curriculum phase."""
    if phase == "OP1":
        env.policies["red"] = OP1RedPolicy("red")
    elif phase == "OP2":
        env.policies["red"] = OP2RedPolicy("red")
    elif phase == "OP3":
        env.policies["red"] = OP3RedPolicy("red")
    else:
        raise ValueError(f"Unknown phase: {phase}")


# =========================
# ENV FACTORY
# =========================

def make_env() -> GameField:
    grid = [[0] * GRID_COLS for _ in range(GRID_ROWS)]
    env = GameField(grid)

    env.use_internal_policies = True        # allow RED scripted policies
    env.set_external_control("blue", True)  # RL for blue
    env.set_external_control("red", False)  # scripted for red

    return env


# =========================
# ROLLOUT STORAGE (Δt)
# =========================

class RolloutBuffer:
    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.dts: List[float] = []        # Δt between macro decisions

    def add(self, obs, action, log_prob, value, reward, done, dt):
        self.obs.append(np.array(obs, dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.dts.append(float(dt))

    def size(self) -> int:
        return len(self.obs)

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.dts.clear()

    def to_tensors(self, device: torch.device):
        obs = torch.tensor(np.stack(self.obs, axis=0), dtype=torch.float32, device=device)
        actions = torch.tensor(self.actions, dtype=torch.long, device=device)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        values = torch.tensor(self.values, dtype=torch.float32, device=device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)
        dts = torch.tensor(self.dts, dtype=torch.float32, device=device)
        return obs, actions, log_probs, values, rewards, dones, dts


# =========================
# EVENT-DRIVEN GAE
# =========================

def compute_gae_event(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    dts: torch.Tensor,
    gamma: float = GAMMA,
    lam: float = GAE_LAMBDA,
):
    T = rewards.size(0)
    advantages = torch.zeros(T, device=rewards.device)
    next_advantage = 0.0
    next_value = 0.0

    for t in reversed(range(T)):
        dt = dts[t]
        gamma_dt = gamma ** dt
        lam_gamma_dt = (lam * gamma) ** dt

        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma_dt * next_value * mask - values[t]
        advantages[t] = delta + lam_gamma_dt * next_advantage * mask

        next_advantage = advantages[t]
        next_value = values[t]

    returns = advantages + values
    return advantages, returns


# =========================
# PPO UPDATE
# =========================

def ppo_update(
    policy: ActorCriticNet,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    device: torch.device,
    ent_coef: float,
):
    obs, actions, old_log_probs, values, rewards, dones, dts = buffer.to_tensors(device)
    advantages, returns = compute_gae_event(rewards, values, dones, dts, GAMMA, GAE_LAMBDA)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset_size = obs.size(0)
    idxs = np.arange(dataset_size)

    for _ in range(PPO_EPOCHS):
        np.random.shuffle(idxs)

        for start in range(0, dataset_size, MINIBATCH_SIZE):
            end = start + MINIBATCH_SIZE
            mb_idx = idxs[start:end]

            mb_obs = obs[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_returns = returns[mb_idx]

            new_log_probs, entropy, new_values = policy.evaluate_actions(mb_obs, mb_actions)

            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (mb_returns - new_values).pow(2).mean()
            entropy_loss = entropy.mean()

            loss = policy_loss + VALUE_COEF * value_loss - ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

    buffer.clear()


# =========================
# BLUE REWARD COLLECTION
# =========================

def collect_blue_rewards_for_step(
    gm: GameManager,
    blue_agents,
) -> Dict[str, float]:
    raw = gm.get_step_rewards()  # clears internal buffer

    rewards_by_id = {a.unique_id: 0.0 for a in blue_agents}
    for k, v in raw.items():
        if k is None:
            # safety; shouldn't normally happen with current GameManager
            for aid in rewards_by_id:
                rewards_by_id[aid] += v
        else:
            for aid in rewards_by_id:
                if k == aid or k.startswith(aid.split("_")[0] + "_"):
                    rewards_by_id[aid] += v

    return rewards_by_id


# =========================
# TRAINING LOOP: OP1 -> OP2 -> OP3
# =========================

def train_ppo_event(
    total_steps: int = TOTAL_STEPS,
    checkpoint_every: int = 10_000,
):
    env = make_env()
    gm = env.getGameManager()

    policy = ActorCriticNet().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    buffer = RolloutBuffer()

    global_step = 0
    episode_idx = 0

    blue_wins = 0
    red_wins = 0
    draws = 0
    running_avg_return = 0.0
    start_time = time.time()

    # Curriculum state
    phase_idx = 0
    phase_episode_count = 0
    phase_recent_results: List[int] = []  # 1 for blue win, 0 otherwise

    # Initial episode-step cap (will be updated per-phase inside the loop)
    current_max_episode_steps = PHASE_CONFIG[PHASE_SEQUENCE[0]]["max_macro_steps"]

    while global_step < total_steps:
        # ---- PHASE CONFIG FOR THIS EPISODE ----
        cur_phase = PHASE_SEQUENCE[phase_idx]
        phase_cfg = PHASE_CONFIG[cur_phase]

        gm.score_limit = phase_cfg["score_limit"]
        gm.max_time = phase_cfg["max_time"]
        current_max_episode_steps = phase_cfg["max_macro_steps"]
        gm.set_phase(cur_phase)

        episode_idx += 1
        phase_episode_count += 1

        # Set opponent for this phase
        set_red_policy_for_phase(env, cur_phase)

        env.reset_default()
        gm.reset_game(reset_scores=True)

        done_episode = False
        episode_steps = 0
        episode_return = 0.0
        last_step_r = 0.0

        # Choose entropy coefficient for this phase
        ENT_COEF = ENT_COEF_BY_PHASE.get(cur_phase, 0.02)

        while (
            not done_episode
            and episode_steps < current_max_episode_steps
            and global_step < total_steps
        ):
            blue_agents = [a for a in env.blue_agents if a.isEnabled()]

            if not blue_agents:
                # No blue agents alive: just sim forward until terminal or window elapsed
                sim_t = 0.0
                while sim_t < DECISION_WINDOW and not gm.game_over:
                    env.update(SIM_DT)
                    sim_t += SIM_DT
                if gm.game_over:
                    done_episode = True
                continue

            decisions: List[Tuple[str, List[float], int, float, float]] = []

            # ====== BLUE MACRO DECISIONS ======
            for agent in blue_agents:
                obs_vec = env.build_observation(agent)
                obs_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=DEVICE)

                out = policy.act(obs_tensor, deterministic=False)
                action_tensor = out["action"][0]
                log_prob = out["log_prob"][0].item()
                value = out["value"][0].item()

                action_id = int(action_tensor.item())
                macro = MacroAction(action_id)
                env.apply_macro_action(agent, macro)

                decisions.append((agent.unique_id, obs_vec, action_id, log_prob, value))

            # ====== SIMULATE FOR DECISION_WINDOW ======
            sim_t = 0.0
            while sim_t < DECISION_WINDOW and not gm.game_over:
                env.update(SIM_DT)
                sim_t += SIM_DT
            dt_decision = sim_t

            # ====== COLLECT BLUE REWARDS ======
            rewards_by_id = collect_blue_rewards_for_step(gm, blue_agents)
            step_done = gm.game_over

            step_reward_sum = 0.0
            n_decisions = len(decisions)

            for aid, obs_vec, act_id, log_p, val in decisions:
                r = rewards_by_id.get(aid, 0.0)

                step_reward_sum += r
                buffer.add(
                    obs=obs_vec,
                    action=act_id,
                    log_prob=log_p,
                    value=val,
                    reward=r,
                    done=step_done,
                    dt=dt_decision,
                )
                global_step += 1

            episode_return += step_reward_sum
            if n_decisions > 0:
                last_step_r = step_reward_sum / n_decisions

            episode_steps += 1
            done_episode = step_done

            # ====== PPO UPDATE WHEN BUFFER FULL ======
            if buffer.size() >= UPDATE_EVERY:
                print(
                    f"[UPDATE] step={global_step} episode={episode_idx} "
                    f"buffer={buffer.size()} phase={cur_phase} ent={ENT_COEF:.4f}"
                )
                ppo_update(policy, optimizer, buffer, DEVICE, ENT_COEF)

                ckpt_path = os.path.join(CHECKPOINT_DIR, f"ctf_ppo_step{global_step}.pth")
                torch.save(policy.state_dict(), ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

        # ====== EPISODE DONE — LOG RESULT ======
        if gm.blue_score > gm.red_score:
            result = "BLUE WIN"
            blue_wins += 1
            phase_recent_results.append(1)
        elif gm.red_score > gm.blue_score:
            result = "RED WIN"
            red_wins += 1
            phase_recent_results.append(0)
        else:
            result = "DRAW"
            draws += 1
            phase_recent_results.append(0)

        if len(phase_recent_results) > PHASE_WINRATE_WINDOW:
            phase_recent_results = phase_recent_results[-PHASE_WINRATE_WINDOW:]

        running_avg_return = 0.99 * running_avg_return + 0.01 * episode_return
        avg_r = running_avg_return

        total_games = blue_wins + red_wins + draws
        winrate = 100.0 * blue_wins / max(1, total_games)
        phase_winrate = float(sum(phase_recent_results)) / max(1, len(phase_recent_results))

        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        term_r = episode_return

        print(
            f"[{episode_idx:5d}] {result:8} | "
            f"StepR {last_step_r:+6.3f} TermR {term_r:+5.2f} AvgR {avg_r:+6.3f} | "
            f"Win% {winrate:5.1f}% | "
            f"PhaseWin {phase_winrate*100:5.1f}% | "
            f"Score {gm.blue_score}:{gm.red_score} | "
            f"B {blue_wins} R {red_wins} D {draws} | "
            f"{elapsed} | {cur_phase}"
        )

        # ---- CURRICULUM ADVANCEMENT ----
        # Only advance if *not* already in the final phase (OP3)
        if cur_phase != PHASE_SEQUENCE[-1]:
            min_eps = MIN_PHASE_EPISODES[cur_phase]
            target_wr = TARGET_PHASE_WINRATE[cur_phase]

            if (
                phase_episode_count >= min_eps
                and len(phase_recent_results) >= PHASE_WINRATE_WINDOW
                and phase_winrate >= target_wr
            ):
                print(
                    f"[CURRICULUM] Phase {cur_phase} complete: "
                    f"episodes={phase_episode_count}, "
                    f"PhaseWin={phase_winrate*100:.1f}% "
                    f"(target={target_wr*100:.1f}%) → advancing."
                )
                phase_idx += 1
                phase_episode_count = 0
                phase_recent_results.clear()

        # Episode-based checkpoints
        if (episode_idx % 100) == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"ctf_ppo_ep{episode_idx}.pth")
            torch.save(policy.state_dict(), ckpt_path)
            print(f"[EP-CHECKPOINT] Saved model at episode {episode_idx} -> {ckpt_path}")

    final_path = os.path.join(CHECKPOINT_DIR, "ctf_ppo_final.pth")
    torch.save(policy.state_dict(), final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    train_ppo_event()
