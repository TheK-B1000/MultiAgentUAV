import os
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np

from game_field import GameField, MacroAction
from game_manager import GameManager
from rl_policy import ActorCriticNet


# =========================
# BASIC CONFIG
# =========================

GRID_ROWS = 30
GRID_COLS = 40

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO hyperparameters (tweak as needed)
TOTAL_STEPS = 50_000          # total *macro-decision* steps
UPDATE_EVERY = 2_048          # collect this many steps before each PPO update
PPO_EPOCHS = 10
MINIBATCH_SIZE = 256

LR = 3e-4
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENT_COEF = 0.01
MAX_GRAD_NORM = 0.5

GAMMA = 0.99
GAE_LAMBDA = 0.95

DECISION_WINDOW = 0.7         # seconds of simulation between macro decisions
SIM_DT = 0.1                  # simulation dt inside a decision window

MAX_EPISODE_STEPS = 500       # number of macro decisions per episode (safety cap)

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def make_env() -> GameField:
    grid = [[0] * GRID_COLS for _ in range(GRID_ROWS)]
    env = GameField(grid)

    # Blue will be controlled externally (RL); Red stays internal (OP3)
    env.use_internal_policies = False
    env.set_external_control("blue", True)
    env.set_external_control("red", False)
    return env


# =========================
# ROLLOUT STORAGE
# =========================

class RolloutBuffer:
    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []

    def add(self, obs, action, log_prob, value, reward, done):
        self.obs.append(np.array(obs, dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def size(self) -> int:
        return len(self.obs)

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def to_tensors(self, device: torch.device):
        obs = torch.tensor(np.stack(self.obs, axis=0), dtype=torch.float32, device=device)
        actions = torch.tensor(self.actions, dtype=torch.long, device=device)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        values = torch.tensor(self.values, dtype=torch.float32, device=device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)
        return obs, actions, log_probs, values, rewards, dones


# =========================
# ADVANTAGE / RETURN
# =========================

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = GAMMA,
    lam: float = GAE_LAMBDA,
):
    """
    Standard discrete-time GAE.
    For full event-driven GAE, you'd incorporate variable dt here (γ^Δt, (λγ)^Δt).
    """
    T = rewards.size(0)
    advantages = torch.zeros(T, device=rewards.device)
    next_advantage = 0.0
    next_value = 0.0

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        advantages[t] = delta + gamma * lam * next_advantage * mask
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
    device: torch.device = DEVICE,
):
    obs, actions, old_log_probs, values, rewards, dones = buffer.to_tensors(device)
    advantages, returns = compute_gae(rewards, values, dones, GAMMA, GAE_LAMBDA)

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

            loss = policy_loss + VALUE_COEF * value_loss - ENT_COEF * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

    buffer.clear()


# =========================
# TRAINING LOOP
# =========================

def collect_blue_rewards_for_step(
    gm: GameManager,
    blue_agents,
) -> Dict[str, float]:
    """
    Get per-agent rewards for BLUE from GameManager.get_step_rewards().

    GameManager stores events keyed by agent_id (Agent.unique_id).
    We'll sum up any keys that match a given BLUE agent.
    """
    raw = gm.get_step_rewards()  # clears internal buffer

    rewards_by_id = {a.unique_id: 0.0 for a in blue_agents}
    for k, v in raw.items():
        if k is None:
            # safety; shouldn't normally happen with current GameManager
            for aid in rewards_by_id:
                rewards_by_id[aid] += v
        else:
            # exact match or "starts with" if you ever fall back to short ids
            for aid in rewards_by_id:
                if k == aid or k.startswith(aid.split("_")[0] + "_"):
                    rewards_by_id[aid] += v

    return rewards_by_id


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

    while global_step < total_steps:
        # ====== NEW EPISODE ======
        episode_idx += 1
        env.reset_default()
        gm.reset_game(reset_scores=True)

        done_episode = False
        episode_steps = 0

        while not done_episode and episode_steps < MAX_EPISODE_STEPS and global_step < total_steps:
            blue_agents = [a for a in env.blue_agents if a.isEnabled()]

            if not blue_agents:
                # If both blue agents disabled, just tick forward until respawn
                sim_t = 0.0
                while sim_t < DECISION_WINDOW and not gm.game_over:
                    env.update(SIM_DT)
                    sim_t += SIM_DT
                # no obs/action for this step; just continue to next decision
                if gm.game_over:
                    done_episode = True
                continue

            # ==== 1) ACTION SELECTION FOR ALL BLUE AGENTS ====
            decisions: List[Tuple[str, List[float], int, float, float]] = []
            # (agent_id, obs, action_id, log_prob, value)

            for agent in blue_agents:
                obs_vec = env.build_observation(agent)
                obs_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=DEVICE)

                out = policy.act(obs_tensor, deterministic=False)
                action = out["action"][0]
                log_prob = out["log_prob"][0].item()
                value = out["value"][0].item()

                macro = MacroAction(int(action.item()))
                env.apply_macro_action(agent, macro)

                decisions.append((agent.unique_id, obs_vec, int(action.item()), log_prob, value))

            # ==== 2) SIMULATE FOR A SHORT WINDOW ====
            sim_t = 0.0
            while sim_t < DECISION_WINDOW and not gm.game_over:
                env.update(SIM_DT)
                sim_t += SIM_DT

            # ==== 3) REWARDS FOR THIS DECISION STEP ====
            rewards_by_id = collect_blue_rewards_for_step(gm, blue_agents)

            step_done = gm.game_over

            # Record one transition per decision (per agent)
            for aid, obs_vec, act_id, log_p, val in decisions:
                r = rewards_by_id.get(aid, 0.0)
                buffer.add(
                    obs=obs_vec,
                    action=act_id,
                    log_prob=log_p,
                    value=val,
                    reward=r,
                    done=step_done,
                )
                global_step += 1

            episode_steps += 1
            done_episode = step_done

            # ==== PPO UPDATE IF BUFFER FULL ====
            if buffer.size() >= UPDATE_EVERY:
                print(f"[UPDATE] step={global_step} episode={episode_idx} buffer={buffer.size()}")
                ppo_update(policy, optimizer, buffer, DEVICE)

                # Optional: save checkpoint
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"ctf_ppo_step{global_step}.pth")
                torch.save(policy.state_dict(), ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

        print(f"[EPISODE {episode_idx}] steps={episode_steps} game_over={gm.game_over} global_step={global_step}")

    # Final model save
    final_path = os.path.join(CHECKPOINT_DIR, "ctf_ppo_final.pth")
    torch.save(policy.state_dict(), final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    train_ppo_event()
