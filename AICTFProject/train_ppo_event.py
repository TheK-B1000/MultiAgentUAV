# train_ppo_event.py
import os
import time
from collections import deque
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game_field import GameField, MacroAction
from game_manager import GameManager
from rl_policy import ActorCriticNet
from policies import OP1RedPolicy, OP2RedPolicy, OP3RedPolicy, HeuristicPolicy

# =========================
# BASIC CONFIG
# =========================

GRID_ROWS = 30
GRID_COLS = 40

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO hyperparameters
TOTAL_STEPS = 300_000         # safety cap only (curriculum controls length)
UPDATE_EVERY = 2_048          # collect this many decisions before PPO update
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
SIM_DT = 0.1                  # physics dt inside a decision window

MAX_EPISODE_STEPS = 500       # max macro decisions per episode

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =========================
# CURRICULUM HELPERS
# =========================

def get_phase_for_episode(ep: int) -> str:
    """
    Curriculum by EPISODE index (1-based):

      1–100    -> OP1
      101–350  -> OP2  (100 + 250)
      351–850  -> OP3  (350 + 500)
      851–1600 -> SELF (850 + 750)
      >1600    -> DONE
    """
    if ep <= 100:
        return "OP1"
    elif ep <= 350:
        return "OP2"
    elif ep <= 850:
        return "OP3"
    elif ep <= 1600:
        return "SELF"
    else:
        return "DONE"


def set_red_policy_for_phase(env: GameField, phase: str) -> None:
    """
    Swap the red opponent behavior according to the current curriculum phase.
    Blue is always RL-controlled; red is fixed OPx/heuristic here.

    NOTE: For true self-play in "SELF", you'd plug a learned policy on red.
          For now we use OP3RedPolicy as the strongest scripted opponent.
    """
    if phase == "OP1":
        env.policies["red"] = OP1RedPolicy(side="red")
    elif phase == "OP2":
        env.policies["red"] = OP2RedPolicy(side="red")
    elif phase == "OP3":
        env.policies["red"] = OP3RedPolicy(side="red")
    elif phase == "SELF":
        # Placeholder: strongest scripted opponent.
        # TODO: replace with learned policy for real self-play.
        env.policies["red"] = OP3RedPolicy(side="red")
    else:
        # Fallback
        env.policies["red"] = HeuristicPolicy()


def make_env() -> GameField:
    grid = [[0] * GRID_COLS for _ in range(GRID_ROWS)]
    env = GameField(grid)

    # Internal policies on for RED, but not for BLUE.
    env.use_internal_policies = True
    env.set_external_control("blue", True)   # RL controls blue
    env.set_external_control("red", False)   # internal OPx controls red

    return env


# =========================
# ROLLOUT BUFFER
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
    (For full event-driven GAE, you'd use γ^Δt and (λγ)^Δt with variable dt.)
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
    if buffer.size() == 0:
        return

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
# REWARD COLLECTION
# =========================

def collect_blue_rewards_for_step(
    gm: GameManager,
    blue_agents,
) -> Dict[str, float]:
    """
    Get per-agent rewards for BLUE from GameManager.get_step_rewards().

    GameManager stores events keyed by agent_id (Agent.unique_id).
    We sum up any keys that match a given BLUE agent.
    """
    raw = gm.get_step_rewards()  # clears internal buffer

    rewards_by_id = {a.unique_id: 0.0 for a in blue_agents}
    for k, v in raw.items():
        if k is None:
            # Team-wide reward → split among blue agents
            for aid in rewards_by_id:
                rewards_by_id[aid] += v
        else:
            # exact match or "starts with"
            for aid in rewards_by_id:
                if k == aid or k.startswith(aid.split("_")[0] + "_"):
                    rewards_by_id[aid] += v

    return rewards_by_id


# =========================
# TRAINING LOOP
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

    # Stats for printing
    blue_wins = 0
    red_wins = 0
    draws = 0

    last_step_r = 0.0
    term_r = 0.0
    recent_returns = deque(maxlen=100)

    start_time = time.time()

    while True:
        episode_idx += 1
        cur_phase = get_phase_for_episode(episode_idx)

        if cur_phase == "DONE":
            # Curriculum complete: 100 + 250 + 500 + 750 episodes
            break

        set_red_policy_for_phase(env, cur_phase)

        env.reset_default()
        gm.reset_game(reset_scores=True)

        done_episode = False
        episode_steps = 0
        episode_return = 0.0
        term_r = 0.0

        while (
            not done_episode
            and episode_steps < MAX_EPISODE_STEPS
            and global_step < total_steps
        ):
            blue_agents = [a for a in env.blue_agents if a.isEnabled()]

            # If blue is temporarily dead, just roll sim forward until respawn
            if not blue_agents:
                sim_t = 0.0
                while sim_t < DECISION_WINDOW and not gm.game_over:
                    env.update(SIM_DT)
                    sim_t += SIM_DT

                if gm.game_over:
                    done_episode = True
                continue

            # ==== 1) ACTION SELECTION FOR ALL BLUE AGENTS ====
            decisions = []  # (agent_id, obs_vec, act_id, log_prob, value)

            for agent in blue_agents:
                obs_vec = env.build_observation(agent)
                obs_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=DEVICE)

                out = policy.act(obs_tensor, deterministic=False)
                action_tensor = out["action"]
                log_prob_tensor = out["log_prob"]
                value_tensor = out["value"]

                # Shapes are [1]
                act_id = int(action_tensor[0].item())
                log_prob = float(log_prob_tensor[0].item())
                value = float(value_tensor[0].item())

                macro = MacroAction(act_id)
                env.apply_macro_action(agent, macro)

                decisions.append((agent.unique_id, obs_vec, act_id, log_prob, value))

            # ==== 2) SIMULATE FOR DECISION WINDOW ====
            sim_t = 0.0
            while sim_t < DECISION_WINDOW and not gm.game_over:
                env.update(SIM_DT)
                sim_t += SIM_DT

            # ==== 3) REWARDS FOR THIS DECISION STEP ====
            rewards_by_id = collect_blue_rewards_for_step(gm, blue_agents)

            step_done = gm.game_over
            # total reward this macro step for blue team
            step_r = sum(rewards_by_id.values())
            last_step_r = step_r
            episode_return += step_r
            if step_done:
                term_r = step_r

            # Record transitions
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
                print(f"[UPDATE] step={global_step} episode={episode_idx} "
                      f"buffer={buffer.size()} phase={cur_phase}")
                ppo_update(policy, optimizer, buffer, DEVICE)

                # Optional: save checkpoint
                ckpt_path = os.path.join(
                    CHECKPOINT_DIR,
                    f"ctf_ppo_step{global_step}_ep{episode_idx}_{cur_phase}.pth"
                )
                torch.save({"model": policy.state_dict(),
                            "global_step": global_step,
                            "episode": episode_idx,
                            "phase": cur_phase},
                           ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

            # Safety stop on steps
            if global_step >= total_steps:
                done_episode = True
                break

        # ===== EPISODE DONE – UPDATE STATS & PRINT =====
        # Who won?
        if gm.blue_score > gm.red_score:
            blue_wins += 1
            result = "BLUE_WIN"
        elif gm.red_score > gm.blue_score:
            red_wins += 1
            result = "RED_WIN"
        else:
            draws += 1
            result = "DRAW"

        recent_returns.append(episode_return)
        avg_r = sum(recent_returns) / max(1, len(recent_returns))

        total_games = blue_wins + red_wins + draws
        winrate = 100.0 * blue_wins / max(1, total_games)

        elapsed = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - start_time)
        )

        # EXACT print format you requested:
        print(
            f"[{episode_idx:5d}] {result:11} | "
            f"StepR {last_step_r:+6.3f} TermR {term_r:+5.2f} AvgR {avg_r:+6.3f} | "
            f"Win% {winrate:5.1f}% | "
            f"B {blue_wins} R {red_wins} D {draws} | "
            f"{elapsed} | {cur_phase}"
        )

        # If safety cap hit, end training even if curriculum not finished
        if global_step >= total_steps:
            print("[STOP] Reached TOTAL_STEPS safety cap.")
            break

    # Final model save
    final_path = os.path.join(CHECKPOINT_DIR, "ctf_ppo_final.pth")
    torch.save({"model": policy.state_dict(),
                "global_step": global_step,
                "last_episode": episode_idx},
               final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    train_ppo_event()
