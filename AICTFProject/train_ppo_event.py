# train_continuous_ctf.py
import os
import time
import math
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game_field import GameField
from macro_actions import MacroAction
from rl_policy import ActorCriticNet
from policies import OP1RedPolicy, OP2RedPolicy, OP3RedPolicy

# ========================= CONFIG (PAPER-ALIGNED) =========================
GRID_ROWS = 30
GRID_COLS = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO Hyperparameters (Table 3)
TOTAL_TIMESTEPS = 5_000_000
UPDATE_EVERY = 2048          # collect 2048 macro-steps before update
PPO_EPOCHS = 10
MINIBATCH_SIZE = 256
LR = 3e-4
CLIP_EPS = 0.2
VALUE_COEF = 1.0
ENT_COEF_BASE = 0.01
MAX_GRAD_NORM = 0.5

# Event-driven RL
GAMMA = 0.995
GAE_LAMBDA = 0.99
DECISION_INTERVAL = 0.7      # seconds between macro decisions
SIM_DT = 0.1                 # simulation step
MAX_MACRO_STEPS = 600        # safety cap per episode

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ========================= CURRICULUM =========================
PHASE_SEQUENCE = ["OP1", "OP2", "OP3", "SELF"]

MIN_EPISODES_PER_PHASE = {"OP1": 300, "OP2": 300, "OP3": 500, "SELF": 1000}
TARGET_WINRATE = {"OP1": 0.90, "OP2": 0.85, "OP3": 0.80}  # SELF has no gate
WINRATE_WINDOW = 50

ENT_COEF_BY_PHASE = {"OP1": 0.02, "OP2": 0.015, "OP3": 0.010, "SELF": 0.0075}

def set_red_opponent(env: GameField, phase: str) -> None:
    if phase == "OP1":
        env.set_red_opponent("OP1")
    elif phase == "OP2":
        env.set_red_opponent("OP2")
    elif phase == "OP3" or phase == "SELF":
        env.set_red_opponent("OP3")


# ========================= ENVIRONMENT =========================
def make_env() -> GameField:
    grid = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    env = GameField(grid)
    env.use_internal_policies = True
    env.set_external_control("blue", True)   # RL controls blue
    env.set_external_control("red", False)   # scripted red
    return env


# ========================= ROLLOUT BUFFER (EVENT-DRIVEN) =========================
class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.dts = []  # Δt between decisions

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

    def tensors(self, device):
        return (
            torch.tensor(np.stack(self.obs), dtype=torch.float32, device=device),
            torch.tensor(self.actions, dtype=torch.long, device=device),
            torch.tensor(self.log_probs, dtype=torch.float32, device=device),
            torch.tensor(self.values, dtype=torch.float32, device=device),
            torch.tensor(self.rewards, dtype=torch.float32, device=device),
            torch.tensor(self.dones, dtype=torch.float32, device=device),
            torch.tensor(self.dts, dtype=torch.float32, device=device),
        )


# ========================= EVENT-DRIVEN GAE =========================
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    dts: torch.Tensor,
    gamma: float = GAMMA,
    lam: float = GAE_LAMBDA,
):
    T = rewards.size(0)
    advantages = torch.zeros(T, device=rewards.device)
    returns = torch.zeros(T, device=rewards.device)
    next_adv = 0.0
    next_val = 0.0

    for t in reversed(range(T)):
        dt = dts[t].item()
        gamma_dt = gamma ** dt
        lam_gamma_dt = (lam * gamma) ** dt
        mask = 1.0 - dones[t]

        delta = rewards[t] + gamma_dt * next_val * mask - values[t]
        advantages[t] = delta + lam_gamma_dt * next_adv * mask
        returns[t] = advantages[t] + values[t]

        next_adv = advantages[t]
        next_val = values[t]

    return advantages, returns


# ========================= PPO UPDATE =========================
def ppo_update(policy, optimizer, buffer: RolloutBuffer, ent_coef: float):
    obs, actions, old_log_probs, old_values, rewards, dones, dts = buffer.tensors(DEVICE)
    advantages, returns = compute_gae(rewards, old_values, dones, dts)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset_size = obs.size(0)
    indices = np.arange(dataset_size)

    for _ in range(PPO_EPOCHS):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, MINIBATCH_SIZE):
            end = start + MINIBATCH_SIZE
            idx = indices[start:end]

            new_logits, new_values = policy(obs[idx])
            dist = torch.distributions.Categorical(logits=new_logits)
            new_log_probs = dist.log_prob(actions[idx])
            entropy = dist.entropy()

            ratio = torch.exp(new_log_probs - old_log_probs[idx])
            surr1 = ratio * advantages[idx]
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(new_values.squeeze(-1), returns[idx])
            entropy_loss = entropy.mean()

            loss = policy_loss + VALUE_COEF * value_loss - ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

    buffer.clear()


# ========================= REWARD COLLECTION =========================
def collect_blue_rewards(gm, blue_agents) -> Dict[str, float]:
    raw = gm.get_step_rewards()
    rewards = {a.unique_id: 0.0 for a in blue_agents}
    for agent_id, r in raw.items():
        if agent_id is None:
            # Global rewards (win/loss) split equally
            per_agent = r / len(blue_agents) if blue_agents else 0.0
            for aid in rewards:
                rewards[aid] += per_agent
        else:
            rewards[agent_id] = rewards.get(agent_id, 0.0) + r
    return rewards


# ========================= MAIN TRAINING LOOP =========================
def train():
    env = make_env()
    gm = env.getGameManager()

    policy = ActorCriticNet().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    buffer = RolloutBuffer()

    timestep = 0
    episode = 0

    phase_idx = 0
    phase_episodes = 0
    recent_wins = []  # for winrate tracking

    total_blue_wins = total_red_wins = total_draws = 0
    running_return = 0.0

    start_time = time.time()

    while timestep < TOTAL_TIMESTEPS and phase_idx < len(PHASE_SEQUENCE):
        episode += 1
        phase_episodes += 1
        cur_phase = PHASE_SEQUENCE[phase_idx]
        set_red_opponent(env, cur_phase)

        env.reset_default()
        gm.reset_game(reset_scores=True)

        done = False
        macro_steps = 0
        episode_return = 0.0
        last_step_reward = 0.0

        while not done and macro_steps < MAX_MACRO_STEPS and timestep < TOTAL_TIMESTEPS:
            blue_agents = [a for a in env.blue_agents if a.isEnabled()]
            if not blue_agents:
                env.update(SIM_DT)
                continue

            # === 1. Action selection ===
            decisions = []
            for agent in blue_agents:
                obs = np.array(env.build_observation(agent), dtype=np.float32)
                obs_tensor = torch.tensor(obs, device=DEVICE).unsqueeze(0)

                with torch.no_grad():
                    logits, value = policy(obs_tensor)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                action_id = action.item()
                env.apply_macro_action(agent, MacroAction(action_id))

                decisions.append((agent.unique_id, obs, action_id, log_prob.item(), value.item()))

            # === 2. Simulate until next decision ===
            sim_time = 0.0
            while sim_time < DECISION_INTERVAL and not gm.game_over:
                env.update(SIM_DT)
                sim_time += SIM_DT

            dt = sim_time

            # === 3. Collect rewards ===
            rewards = collect_blue_rewards(gm, blue_agents)
            step_reward = sum(rewards.values())
            episode_return += step_reward
            last_step_reward = step_reward / len(blue_agents) if blue_agents else 0.0

            # === 4. Store transitions ===
            for uid, obs, act, lp, val in decisions:
                r = rewards.get(uid, 0.0)
                buffer.add(obs, act, lp, val, r, gm.game_over, dt)
                timestep += 1

            macro_steps += 1
            done = gm.game_over

            # === 5. PPO Update ===
            if buffer.size() >= UPDATE_EVERY:
                ent_coef = ENT_COEF_BY_PHASE.get(cur_phase, ENT_COEF_BASE)
                ppo_update(policy, optimizer, buffer, ent_coef)

                if timestep % 50_000 < UPDATE_EVERY:
                    path = os.path.join(CHECKPOINT_DIR, f"ctf_ppo_{timestep}.pth")
                    torch.save(policy.state_dict(), path)
                    print(f"[CHECKPOINT] Saved {path}")

        # === Episode finished ===
        running_return = running_return * 0.99 + episode_return * 0.01

        if gm.blue_score > gm.red_score:
            result = "BLUE WIN"
            total_blue_wins += 1
            recent_wins.append(1)
        elif gm.red_score > gm.blue_score:
            result = "RED WIN"
            total_red_wins += 1
            recent_wins.append(0)
        else:
            result = "DRAW"
            total_draws += 1
            recent_wins.append(0)

        if len(recent_wins) > WINRATE_WINDOW:
            recent_wins.pop(0)

        winrate = sum(recent_wins) / len(recent_wins) if recent_wins else 0.0
        total_games = total_blue_wins + total_red_wins + total_draws
        global_winrate = total_blue_wins / total_games if total_games > 0 else 0.0

        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))

        print(
            f"[{episode:5d}] {result:9} | "
            f"StepR {last_step_reward:+6.3f} EpR {episode_return:+7.2f} AvgR {running_return:+7.3f} | "
            f"Win {global_winrate:5.1%} PhaseWin {winrate:5.1%} | "
            f"B{total_blue_wins} R{total_red_wins} D{total_draws} | "
            f"{elapsed} | {cur_phase} ({phase_episodes})"
        )

        # === Curriculum advancement ===
        if cur_phase != "SELF":
            min_eps = MIN_EPISODES_PER_PHASE[cur_phase]
            target = TARGET_WINRATE[cur_phase]
            if (phase_episodes >= min_eps and
                len(recent_wins) >= WINRATE_WINDOW and
                winrate >= target):
                print(f"\n[CURRICULUM] {cur_phase} → NEXT PHASE (winrate {winrate:.1%} ≥ {target:.1%})\n")
                phase_idx += 1
                phase_episodes = 0
                recent_wins.clear()

    # Final save
    final_path = os.path.join(CHECKPOINT_DIR, "ctf_ppo_final.pth")
    torch.save(policy.state_dict(), final_path)
    print(f"\nTraining complete! Final model: {final_path}")


if __name__ == "__main__":
    train()