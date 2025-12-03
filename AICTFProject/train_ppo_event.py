# train_ppo_event.py
# ===========================================================
#  2D DIGITAL CTF — PPO EVENT-DRIVEN (JACOB ET AL.-INSPIRED)
#  - 14 macro-actions (categorical targets + tactics)
#  - Event-based rewards via GameManager
#  - Curriculum: OP1 → OP2 → OP3
#  - Distance-to-flag shaping + time penalty
#  - Emperor evaluation vs OP1/OP2/OP3 (crowning ceremony)
# ===========================================================

import os
import time
import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game_field import GameField
from game_manager import GameManager
from macro_actions import MacroAction
from rl_policy import ActorCriticNet
from policies import OP1RedPolicy, OP2RedPolicy, OP3RedPolicy


# =========================
# BASIC CONFIG
# =========================

GRID_ROWS = 30
GRID_COLS = 40

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PPO / RL hyperparameters ---
TOTAL_STEPS = 3_000_000          # hard safety cap on steps
ROLLOUT_STEPS = 2048             # how many transitions before a PPO update

PPO_EPOCHS = 10
MINIBATCH_SIZE = 256

LR_BASE = 2.5e-4                 # base LR (train2-style)
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENT_COEF_BASE = 0.015
MAX_GRAD_NORM = 0.5

GAMMA = 0.995
LAMBDA = 0.95

DT = 0.1                         # env.update(0.1) per macro decision
MAX_EPISODE_STEPS = 500

# Terminal rewards (team-level, injected at episode end)
WIN_REWARD = +1.0
LOSE_REWARD = -1.0
DRAW_REWARD = -0.3

# Small shaping for carrying the flag each decision
FLAG_CARRY_BONUS = 0.001

# Checkpoints
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

BEST_WINRATE = 0.0
BEST_STATE_DICT = None

# If you want to resume, set this to a checkpoint path or None
RESUME_FROM: Optional[str] = None  # e.g. "checkpoints/ctf_ppo_step1798427.pth"


# =========================
# CURRICULUM (PHASE ORDER + GATING)
# =========================

# No SELF phase now; just grind through OP1 → OP2 → OP3
PHASE_SEQUENCE = ["OP1", "OP2", "OP3"]

MIN_PHASE_EPISODES = {
    "OP1": 500,    # spend more time vs OP1 to overfit & exploit
    "OP2": 800,
    "OP3": 1200,
}

# Softer thresholds so we actually reach OP3
TARGET_PHASE_WINRATE = {
    "OP1": 0.80,
    "OP2": 0.75,
    "OP3": 0.70,
}

PHASE_WINRATE_WINDOW = 100

# Entropy by phase (more exploration early)
ENT_COEF_BY_PHASE = {
    "OP1": 0.02,
    "OP2": 0.0175,
    "OP3": 0.0125,
}

# Optional LR boost in later phases to “shake loose” better strategies
LR_MULT_BY_PHASE = {
    "OP1": 1.0,
    "OP2": 1.25,
    "OP3": 1.5,
}


# =========================
# PHASE / OPPONENT WIRING
# =========================

def set_red_policy_for_phase(env: GameField, phase: str) -> None:
    """
    Swap the RED policy according to the current curriculum phase.
    Blue is always RL; Red is scripted OPX.
    """
    if phase == "OP1":
        env.policies["red"] = OP1RedPolicy("red")
    elif phase == "OP2":
        env.policies["red"] = OP2RedPolicy("red")
    else:
        # Default / strongest baseline
        env.policies["red"] = OP3RedPolicy("red")


# =========================
# ENV FACTORY
# =========================

def make_env() -> GameField:
    grid = [[0] * GRID_COLS for _ in range(GRID_ROWS)]
    env = GameField(grid)

    # Allow RED to use OPX scripted policies internally.
    env.use_internal_policies = True

    # Blue = external RL; Red = internal scripted opponent.
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
# GAE
# =========================

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = GAMMA,
    lam: float = LAMBDA,
):
    T = rewards.size(0)
    advantages = torch.zeros(T, device=rewards.device)
    gae = 0.0
    next_value = 0.0

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
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
    if buffer.size() == 0:
        return

    obs, actions, old_log_probs, values, rewards, dones = buffer.to_tensors(device)
    advantages, returns = compute_gae(rewards, values, dones, GAMMA, LAMBDA)

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
    """
    - Get raw events from GameManager (agent_id or None => global).
    - Split global rewards (None) equally across active blue agents.
    """
    raw = gm.get_step_rewards()  # dict[agent_id or None] -> reward

    rewards_by_id = {a.unique_id: 0.0 for a in blue_agents}

    # Per-agent events
    for aid in rewards_by_id.keys():
        if aid in raw:
            rewards_by_id[aid] += raw[aid]

    # Global team reward (key=None)
    if None in raw and len(rewards_by_id) > 0:
        per_agent = raw[None] / float(len(rewards_by_id))
        for aid in rewards_by_id:
            rewards_by_id[aid] += per_agent

    return rewards_by_id


# =========================
# EMPEROR EVALUATION
# =========================

def evaluate_emperor_crowning(
    policy: ActorCriticNet,
    n_games_per_opponent: int = 20,
) -> float:
    """
    Run clean evaluation vs OP1, OP2, and OP3 with deterministic actions.
    Returns a combined winrate over all evaluation games.
    """
    opponents: List[Tuple[str, type]] = [
        ("OP1", OP1RedPolicy),
        ("OP2", OP2RedPolicy),
        ("OP3", OP3RedPolicy),
    ]

    policy.eval()
    total_eval_games = 0
    total_blue_wins = 0

    print("   [CROWNING CEREMONY] The trial by combat begins...")

    with torch.no_grad():
        for name, OppPolicy in opponents:
            blue_wins_vs_this = 0
            red_wins_vs_this = 0
            draws_vs_this = 0

            for g in range(n_games_per_opponent):
                env = make_env()
                env.policies["red"] = OppPolicy("red")
                gm = env.getGameManager()

                env.reset_default()

                done_episode = False
                steps = 0

                while not done_episode and steps < MAX_EPISODE_STEPS:
                    blue_agents = [a for a in env.blue_agents if a.isEnabled()]

                    if not blue_agents:
                        env.update(DT)
                        _ = gm.get_step_rewards()
                        if gm.game_over:
                            done_episode = True
                        steps += 1
                        continue

                    # Deterministic macro-actions from current policy
                    for agent in blue_agents:
                        obs_vec = env.build_observation(agent)
                        obs_tensor = torch.tensor(
                            obs_vec, dtype=torch.float32, device=DEVICE
                        )
                        out = policy.act(obs_tensor, deterministic=True)
                        action_id = int(out["action"][0].item())
                        macro = MacroAction(action_id)
                        env.apply_macro_action(agent, macro)

                    env.update(DT)
                    _ = gm.get_step_rewards()  # ignore rewards during eval
                    steps += 1
                    done_episode = gm.game_over

                if gm.blue_score > gm.red_score:
                    blue_wins_vs_this += 1
                elif gm.red_score > gm.blue_score:
                    red_wins_vs_this += 1
                else:
                    draws_vs_this += 1

            total_games_this = n_games_per_opponent
            total_eval_games += total_games_this
            total_blue_wins += blue_wins_vs_this

            wr = 100.0 * blue_wins_vs_this / max(1, total_games_this)
            print(
                f"   [EVAL] vs {name}: "
                f"B={blue_wins_vs_this} R={red_wins_vs_this} D={draws_vs_this} "
                f"Win%={wr:5.1f}%"
            )

    combined_score = total_blue_wins / max(1, total_eval_games)
    policy.train()
    print(f"   [VERDICT] Combined score: {combined_score*100:.2f}%")
    return combined_score


# =========================
# MAIN TRAINING LOOP
# =========================

def train_ppo_event(
    total_steps: int = TOTAL_STEPS,
    resume_from: Optional[str] = RESUME_FROM,
):
    global BEST_WINRATE, BEST_STATE_DICT

    env = make_env()
    gm = env.getGameManager()

    policy = ActorCriticNet().to(DEVICE)

    # Optional: resume from checkpoint
    if resume_from is not None and os.path.isfile(resume_from):
        state_dict = torch.load(resume_from, map_location=DEVICE)
        policy.load_state_dict(state_dict)
        print(f"[RESUME] Loaded policy from {resume_from}")

    # Optimizer with base LR (we’ll scale per phase)
    optimizer = optim.Adam(policy.parameters(), lr=LR_BASE)

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
    phase_recent_results: List[int] = []  # 1 = blue win, 0 = not win
    best_eval_ever = 0.0  # for crowning ceremony

    print("\n" + "=" * 110)
    print("  2D DIGITAL CTF — PPO EVENT-DRIVEN")
    print("  14 macro-actions • Event rewards • Curriculum vs OP1/OP2/OP3")
    print("=" * 110 + "\n")

    while True:
        if phase_idx >= len(PHASE_SEQUENCE):
            print("[STOP] Curriculum complete (finished OP3 phase).")
            break
        if global_step >= total_steps:
            print(f"[STOP] Reached global_step safety cap ({global_step} >= {total_steps})")
            break

        cur_phase = PHASE_SEQUENCE[phase_idx]
        episode_idx += 1
        phase_episode_count += 1

        # Adjust opponent & LR for this phase
        set_red_policy_for_phase(env, cur_phase)
        lr_mult = LR_MULT_BY_PHASE.get(cur_phase, 1.0)
        for g in optimizer.param_groups:
            g["lr"] = LR_BASE * lr_mult

        # Reset episode — only env.reset_default (no double gm.reset_game)
        env.reset_default()

        done_episode = False
        episode_steps = 0
        episode_return = 0.0
        last_step_r = 0.0

        # Index in buffer where this episode starts (for terminal reward injection)
        ep_start_idx = buffer.size()

        while (
            not done_episode
            and episode_steps < MAX_EPISODE_STEPS
            and global_step < total_steps
        ):
            blue_agents = [a for a in env.blue_agents if a.isEnabled()]

            if not blue_agents:
                # No active blue agents – just advance time until respawn or game over
                env.update(DT)
                _ = gm.get_step_rewards()  # flush any events
                if gm.game_over:
                    done_episode = True
                episode_steps += 1
                continue

            # 1) Select actions for all blue agents
            decisions = []  # (agent_id, obs_vec, action_id, log_prob, value)

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

            # 2) Advance environment one decision step
            env.update(DT)

            # 3) Collect event-based rewards (GameManager + shaping)
            rewards_by_id = collect_blue_rewards_for_step(gm, blue_agents)

            # Small per-step bonus for carrying the flag
            for agent in blue_agents:
                if getattr(agent, "isCarryingFlag", None) and agent.isCarryingFlag():
                    rewards_by_id[agent.unique_id] = (
                        rewards_by_id.get(agent.unique_id, 0.0) + FLAG_CARRY_BONUS
                    )

            step_done = gm.game_over
            step_reward_sum = 0.0
            n_decisions = len(decisions)

            # 4) Store transitions
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
                )
                global_step += 1

            episode_return += step_reward_sum
            if n_decisions > 0:
                last_step_r = step_reward_sum / n_decisions

            episode_steps += 1
            done_episode = step_done

        # ==============================
        # EPISODE TERMINAL REWARD
        # ==============================
        if gm.blue_score > gm.red_score:
            result = "BLUE WIN"
            blue_wins += 1
            phase_recent_results.append(1)
            term_r = WIN_REWARD
        elif gm.red_score > gm.blue_score:
            result = "RED WIN"
            red_wins += 1
            phase_recent_results.append(0)
            term_r = LOSE_REWARD
        else:
            result = "DRAW"
            draws += 1
            phase_recent_results.append(0)
            term_r = DRAW_REWARD

        if len(phase_recent_results) > PHASE_WINRATE_WINDOW:
            phase_recent_results = phase_recent_results[-PHASE_WINRATE_WINDOW:]

        # Inject terminal bonus only into this episode's transitions
        ep_end_idx = buffer.size()
        ep_len = ep_end_idx - ep_start_idx
        if ep_len > 0:
            bonus = term_r / ep_len
            for i in range(ep_start_idx, ep_end_idx):
                buffer.rewards[i] += bonus

        # Running EMA of returns
        running_avg_return = 0.99 * running_avg_return + 0.01 * (episode_return + term_r)
        avg_r = running_avg_return

        # Global winrate
        total_games = blue_wins + red_wins + draws
        winrate = 100.0 * blue_wins / max(1, total_games)

        # Phase-local winrate
        phase_winrate = float(sum(phase_recent_results)) / max(
            1, len(phase_recent_results)
        )

        # Track best model (training winrate-based)
        if winrate > BEST_WINRATE and total_games >= 50:
            BEST_WINRATE = winrate
            BEST_STATE_DICT = policy.state_dict()
            torch.save(BEST_STATE_DICT, "marl_policy.pth")
            print(f"[BEST] New best winrate: {BEST_WINRATE:.2f}% → marl_policy.pth")

        # Elapsed wall-clock time
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))

        # Print episode summary
        print(
            f"[{episode_idx:5d}] {result:11} | "
            f"StepR {last_step_r:+6.3f} TermR {term_r:+5.2f} AvgR {avg_r:+6.3f} | "
            f"Win% {winrate:5.1f}% | "
            f"PhaseWin {phase_winrate*100:5.1f}% | "
            f"B {blue_wins} R {red_wins} D {draws} | "
            f"{elapsed} | {cur_phase}"
        )

        # ==============================
        # PPO UPDATE (when rollout is full)
        # ==============================
        if buffer.size() >= ROLLOUT_STEPS:
            current_ent_coef = ENT_COEF_BY_PHASE.get(cur_phase, ENT_COEF_BASE)
            print(
                f"[UPDATE] step={global_step} episode={episode_idx} "
                f"buffer={buffer.size()} phase={cur_phase} ent={current_ent_coef:.4f}"
            )
            ppo_update(policy, optimizer, buffer, DEVICE, current_ent_coef)

            ckpt_path = os.path.join(
                CHECKPOINT_DIR, f"ctf_ppo_step{global_step}.pth"
            )
            torch.save(policy.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        # ==============================
        # CURRICULUM ADVANCEMENT
        # ==============================
        if cur_phase in TARGET_PHASE_WINRATE:
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

        # ==============================
        # EMPEROR CROWNING CEREMONY (OP3 ONLY)
        # ==============================
        if cur_phase == "OP3" and (episode_idx % 200) == 0:
            crowning_score = evaluate_emperor_crowning(
                policy, n_games_per_opponent=20
            )
            if crowning_score > best_eval_ever:
                best_eval_ever = crowning_score
                BEST_WINRATE = crowning_score * 100.0
                state = {
                    "model": copy.deepcopy(policy.state_dict()),
                    "episode": episode_idx,
                }
                torch.save(state, "models/CTF_TRUE_EMPEROR.pth")
                print("   " + "=" * 70)
                print("   >>> THE TRUE EMPEROR HAS ASCENDED <<<")
                print("   >>> LET THE GOLDEN AGE BEGIN <<<")
                print("   " + "=" * 70)

        # Extra episode-based checkpoint
        if (episode_idx % 250) == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"ctf_ppo_ep{episode_idx}.pth")
            torch.save(policy.state_dict(), ckpt_path)
            print(f"[EP-CHECKPOINT] Saved model at episode {episode_idx} -> {ckpt_path}")

    # ===== FINAL SAVE =====
    final_path = os.path.join(CHECKPOINT_DIR, "ctf_ppo_final.pth")
    torch.save(policy.state_dict(), final_path)
    print(f"Training complete. Final model saved to {final_path}")

    if BEST_STATE_DICT is not None:
        print(f"BEST AGENT MODEL (train winrate): marl_policy.pth  (Win% = {BEST_WINRATE:.2f}%)")
    else:
        print("No BEST_STATE_DICT recorded yet (maybe no wins or too few games).")


if __name__ == "__main__":
    train_ppo_event()
