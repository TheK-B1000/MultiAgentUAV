import os
import time
import random
from collections import deque
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
# FULLY DETERMINISTIC SEEDING
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# BASIC CONFIG (PAPER-STYLE)
# =========================
GRID_ROWS = 30
GRID_COLS = 40

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO hyperparameters
TOTAL_STEPS = 4_000_000
UPDATE_EVERY = 2_048
PPO_EPOCHS = 10
MINIBATCH_SIZE = 256

LR = 3e-4
CLIP_EPS = 0.2
VALUE_COEF = 1.0
MAX_GRAD_NORM = 0.5

# Event-driven RL discounting
GAMMA = 0.995
GAE_LAMBDA = 0.99

# Simulation timing
DECISION_WINDOW = 0.7
SIM_DT = 0.1

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =========================
# COOP HUD CONFIG
# =========================
COOP_HUD_EVERY = 50
COOP_WINDOW = 50
NUM_ACTIONS = len(MacroAction)
ACTION_NAMES = [ma.name for ma in sorted(MacroAction, key=lambda m: m.value)]


# =========================
# CURRICULUM (OP1 → OP2 → OP3)
# =========================
PHASE_SEQUENCE = ["OP1", "OP2", "OP3"]

MIN_PHASE_EPISODES = {
    "OP1": 300,
    "OP2": 300,
    "OP3": 500,
}

TARGET_PHASE_WINRATE = {
    "OP1": 0.30,
    "OP2": 0.50,
    "OP3": 0.70,
}

PHASE_WINRATE_WINDOW = 50

ENT_COEF_BY_PHASE = {
    "OP1": 0.03,
    "OP2": 0.02,
    "OP3": 0.015,
}

PHASE_CONFIG = {
    "OP1": dict(score_limit=3, max_time=200.0, max_macro_steps=500),
    "OP2": dict(score_limit=2, max_time=150.0, max_macro_steps=350),
    "OP3": dict(score_limit=3, max_time=200.0, max_macro_steps=550),
}

# Target OP3 winrate for emperor save
EMPEROR_TARGET_WR = 0.80  # 80%


def set_red_policy_for_phase(env: GameField, phase: str) -> None:
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
    env.use_internal_policies = True
    env.set_external_control("blue", True)
    env.set_external_control("red", False)
    return env


# =========================
# ROLLOUT BUFFER
# =========================
class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.dts = []

    def add(self, obs, action, log_prob, value, reward, done, dt):
        self.obs.append(np.array(obs, dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.dts.append(float(dt))

    def size(self):
        return len(self.obs)

    def clear(self):
        self.__init__()

    def to_tensors(self, device):
        obs = torch.tensor(np.stack(self.obs), dtype=torch.float32, device=device)
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
def compute_gae_event(rewards, values, dones, dts, gamma=GAMMA, lam=GAE_LAMBDA):
    T = rewards.size(0)
    advantages = torch.zeros(T, device=rewards.device)
    next_adv = 0.0
    next_value = 0.0

    for t in reversed(range(T)):
        gamma_dt = gamma ** dts[t]
        lam_gamma_dt = (gamma * lam) ** dts[t]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma_dt * next_value * mask - values[t]
        advantages[t] = delta + lam_gamma_dt * next_adv * mask
        next_adv = advantages[t]
        next_value = values[t]

    returns = advantages + values
    return advantages, returns


# =========================
# PPO UPDATE
# =========================
def ppo_update(policy, optimizer, buffer, device, ent_coef):
    obs, actions, old_log_probs, values, rewards, dones, dts = buffer.to_tensors(device)
    advantages, returns = compute_gae_event(rewards, values, dones, dts)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    idxs = np.arange(obs.size(0))
    for _ in range(PPO_EPOCHS):
        np.random.shuffle(idxs)
        for start in range(0, obs.size(0), MINIBATCH_SIZE):
            mb_idx = idxs[start:start + MINIBATCH_SIZE]
            mb_obs = obs[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_returns = returns[mb_idx]

            new_log_probs, entropy, new_values = policy.evaluate_actions(mb_obs, mb_actions)

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


# =========================
# BLUE REWARD + PHASE-AWARE FREELOADER PENALTY
# =========================
def collect_blue_rewards_for_step(
    gm: GameManager,
    blue_agents,
    mix_alpha: float = 0.0,
    cur_phase: str = "OP1",
):
    """
    mix_alpha = 0.0 → pure team reward (everyone gets team mean).
    mix_alpha = 1.0 → pure individual reward.
    We stay at mix_alpha=0.0 for now (pure team reward).
    """
    raw = gm.get_step_rewards()

    indiv_rewards_by_id = {a.unique_id: 0.0 for a in blue_agents}
    for k, v in raw.items():
        if k is None:
            for aid in indiv_rewards_by_id:
                indiv_rewards_by_id[aid] += v
        else:
            for aid in indiv_rewards_by_id:
                if k == aid or k.startswith(aid.split("_")[0] + "_"):
                    indiv_rewards_by_id[aid] += v

    team_mean = sum(indiv_rewards_by_id.values()) / max(1, len(indiv_rewards_by_id))

    # Phase-dependent freeloader penalty (light touch)
    penalty = 0.05 if cur_phase == "OP3" else 0.02

    rewards_by_id = {}
    for aid, r_indiv in indiv_rewards_by_id.items():
        r_final = mix_alpha * r_indiv + (1.0 - mix_alpha) * team_mean
        # If team is doing well but this agent has almost no reward, nudge them.
        if team_mean > 0.3 and r_indiv < 0.05:
            r_final -= penalty
        rewards_by_id[aid] = r_final

    return rewards_by_id


# =========================
# ENTROPY + OPPONENT MIXING
# =========================
def get_entropy_coef(cur_phase: str, phase_episode_count: int, phase_wr: float) -> float:
    """
    Base schedule by phase + extra boost when losing (especially OP3).
    """
    base = ENT_COEF_BY_PHASE[cur_phase]
    if cur_phase == "OP1":
        horizon, start_ent = 500.0, max(base, 0.04)
    elif cur_phase == "OP2":
        horizon, start_ent = 800.0, max(base, 0.03)
    else:
        horizon, start_ent = 2000.0, max(base, 0.05)

    frac = min(1.0, phase_episode_count / horizon)
    ent = start_ent - (start_ent - base) * frac

    # Extra exploration when losing, mainly for OP3
    if cur_phase == "OP3":
        if phase_wr < 0.15:
            ent = max(ent, 0.06)  # strong boost
        elif phase_wr < 0.30:
            ent = max(ent, 0.05)  # mild boost

    return float(ent)


def set_mixed_opponent_for_op3(env: GameField, phase_episode_count: int, phase_wr: float) -> str:
    """
    Always use mixing in OP3 (no hard lock to pure OP3).
    We just shift the distribution over OP1/OP2/OP3 as training progresses.
    """
    HARD_START = 900
    COLLAPSE_WR = 0.20

    if phase_episode_count < 300:
        # Early OP3: lots of practice vs weaker opponents
        p1, p2, p3 = 0.34, 0.33, 0.33
    elif phase_episode_count < HARD_START:
        # Middle OP3: more OP3, but still variety
        p1, p2, p3 = 0.25, 0.25, 0.50
    else:
        # Late OP3: OP3-heavy, but *still* mixed to avoid overfitting
        if phase_wr < COLLAPSE_WR:
            # Collapsed: ease back to more OP2
            p1, p2, p3 = 0.10, 0.30, 0.60
        elif phase_wr < 0.50:
            # Learning but not dominant
            p1, p2, p3 = 0.10, 0.20, 0.70
        else:
            # Strong policy: mostly OP3, still a bit of structure
            p1, p2, p3 = 0.05, 0.15, 0.80

    r = random.random()
    if r < p1:
        env.policies["red"] = OP1RedPolicy("red")
        return "OP1-mix"
    elif r < p1 + p2:
        env.policies["red"] = OP2RedPolicy("red")
        return "OP2-mix"
    else:
        env.policies["red"] = OP3RedPolicy("red")
        return "OP3"


# =========================
# TRUE EMPEROR EVALUATION
# =========================
def evaluate_emperor_crowning(policy: ActorCriticNet, device: torch.device, n_games: int = 20) -> float:
    policy.eval()
    env = make_env()
    gm = env.getGameManager()
    cfg = PHASE_CONFIG["OP3"]
    gm.score_limit = cfg["score_limit"]
    gm.max_time = cfg["max_time"]
    gm.set_phase("OP3")

    wins = 0
    with torch.no_grad():
        for _ in range(n_games):
            env.reset_default()
            gm.reset_game(reset_scores=True)
            gm.set_phase("OP3")
            env.policies["red"] = OP3RedPolicy("red")
            done = False
            steps = 0
            max_steps = cfg["max_macro_steps"]

            while not done and steps < max_steps:
                blue_agents = [a for a in env.blue_agents if a.isEnabled()]
                for agent in blue_agents:
                    obs = env.build_observation(agent)
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                    out = policy.act(obs_tensor)
                    a = int(out["action"][0].item())
                    env.apply_macro_action(agent, MacroAction(a))

                sim_t = 0.0
                while sim_t < DECISION_WINDOW and not gm.game_over:
                    env.update(SIM_DT)
                    sim_t += SIM_DT
                done = gm.game_over
                steps += 1

            if gm.blue_score > gm.red_score:
                wins += 1

    policy.train()
    return wins / max(1, n_games)


# =========================
# MAIN TRAINING LOOP
# =========================
def train_ppo_event(total_steps=TOTAL_STEPS):
    set_seed(42)

    env = make_env()
    gm = env.getGameManager()

    policy = ActorCriticNet().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    buffer = RolloutBuffer()

    global_step = 0
    episode_idx = 0

    blue_wins = red_wins = draws = 0

    # Curriculum tracking
    phase_idx = 0
    phase_episode_count = 0
    phase_recent = []
    phase_wr = 0.0

    # Rolling window for cooperation HUD
    coop_window = deque(maxlen=COOP_WINDOW)

    # Best OP3 champion tracking
    best_op3_wr = 0.0
    BEST_OP3_EMPEROR_PATH = os.path.join(CHECKPOINT_DIR, "ctf_true_emperor_op3.pth")
    BEST_OP3_SO_FAR_PATH = os.path.join(CHECKPOINT_DIR, "ctf_best_so_far_op3.pth")

    while global_step < total_steps:
        episode_idx += 1
        phase_episode_count += 1

        cur_phase = PHASE_SEQUENCE[phase_idx]
        phase_cfg = PHASE_CONFIG[cur_phase]
        gm.score_limit = phase_cfg["score_limit"]
        gm.max_time = phase_cfg["max_time"]
        max_steps = phase_cfg["max_macro_steps"]
        gm.set_phase(cur_phase)

        if cur_phase == "OP3":
            opponent_tag = set_mixed_opponent_for_op3(env, phase_episode_count, phase_wr)
        else:
            set_red_policy_for_phase(env, cur_phase)
            opponent_tag = cur_phase

        env.reset_default()
        gm.reset_game(reset_scores=True)

        done = False
        ep_return = 0.0
        last_step_r = 0.0

        ENT_COEF = get_entropy_coef(cur_phase, phase_episode_count, phase_wr)

        steps = 0
        ep_macro_counts = {0: [0] * NUM_ACTIONS, 1: [0] * NUM_ACTIONS}
        ep_mine_attempts = {0: 0, 1: 0}
        ep_combat_events = 0
        ep_score_events = 0
        prev_blue_score = gm.blue_score

        while not done and steps < max_steps and global_step < total_steps:
            blue_agents = [a for a in env.blue_agents if a.isEnabled()]
            decisions = []

            for agent in blue_agents:
                obs = env.build_observation(agent)
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
                out = policy.act(obs_tensor)
                a = int(out["action"][0].item())
                logp = float(out["log_prob"][0].item())
                val = float(out["value"][0].item())

                local_id = getattr(agent, "agent_id", 0)
                if local_id in ep_macro_counts and 0 <= a < NUM_ACTIONS:
                    ep_macro_counts[local_id][a] += 1
                    if MacroAction(a) == MacroAction.PLACE_MINE:
                        ep_mine_attempts[local_id] += 1

                env.apply_macro_action(agent, MacroAction(a))
                decisions.append((agent.unique_id, obs, a, logp, val))

            sim_t = 0.0
            while sim_t < DECISION_WINDOW and not gm.game_over:
                env.update(SIM_DT)
                sim_t += SIM_DT

            dt = sim_t
            rewards = collect_blue_rewards_for_step(
                gm, blue_agents, mix_alpha=0.0, cur_phase=cur_phase
            )
            done = gm.game_over

            blue_score_delta = gm.blue_score - prev_blue_score
            if blue_score_delta > 0:
                ep_score_events += blue_score_delta
            prev_blue_score = gm.blue_score

            total_team_reward = sum(rewards.values())
            if total_team_reward > 0.0 and blue_score_delta == 0:
                ep_combat_events += 1

            step_reward_sum = 0.0
            for aid, obs, act, logp, val in decisions:
                r = rewards.get(aid, 0.0)
                step_reward_sum += r
                buffer.add(obs, act, logp, val, r, done, dt)
                global_step += 1

            ep_return += step_reward_sum
            if decisions:
                last_step_r = step_reward_sum / len(decisions)
            steps += 1

            if buffer.size() >= UPDATE_EVERY:
                print(
                    f"[UPDATE] step={global_step} episode={episode_idx} "
                    f"phase={cur_phase} ENT={ENT_COEF:.4f} Opp={opponent_tag}"
                )
                ppo_update(policy, optimizer, buffer, DEVICE, ENT_COEF)

        # Episode end
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

        print(
            f"[{episode_idx:5d}] {result:8} | "
            f"StepR {last_step_r:+.3f} "
            f"TermR {ep_return:+.1f} | "
            f"Score {gm.blue_score}:{gm.red_score} | "
            f"BlueWins: {blue_wins} RedWins: {red_wins} Draws: {draws} | "
            f"PhaseWin {phase_wr * 100:.1f}% | Phase={cur_phase} Opp={opponent_tag}"
        )

        # ---------- OP3 "CROWNING CEREMONY" EVAL ----------
        if cur_phase == "OP3" and episode_idx % 200 == 0:
            print("   [CROWNING CEREMONY] The trial by combat begins...")
            crowning_score = evaluate_emperor_crowning(policy, device=DEVICE, n_games=20)
            print(f"   [VERDICT] OP3 winrate: {crowning_score * 100:.2f}%")

            if crowning_score > best_op3_wr:
                prev = best_op3_wr
                best_op3_wr = crowning_score

                # Always save "best so far" checkpoint
                torch.save(
                    {
                        "model": policy.state_dict(),
                        "episode": episode_idx,
                        "winrate": crowning_score,
                    },
                    BEST_OP3_SO_FAR_PATH,
                )
                print(
                    f"   [PROGRESS] New best OP3 winrate so far: "
                    f"{best_op3_wr * 100:.2f}% (prev {prev * 100:.2f}%)"
                )
                print(f"   [PROGRESS] Best-so-far model saved → {BEST_OP3_SO_FAR_PATH}")

                # Emperor promotion if it meets the threshold
                if crowning_score >= EMPEROR_TARGET_WR:
                    torch.save(
                        {
                            "model": policy.state_dict(),
                            "episode": episode_idx,
                            "winrate": crowning_score,
                        },
                        BEST_OP3_EMPEROR_PATH,
                    )
                    print("   " + "=" * 70)
                    print("   >>> THE TRUE EMPEROR HAS ASCENDED (OP3 CHAMPION SAVED) <<<")
                    print(
                        f"   >>> Winrate: {crowning_score * 100:.2f}% "
                        f"@ episode {episode_idx} <<<"
                    )
                    print(f"   >>> Emperor model saved → {BEST_OP3_EMPEROR_PATH} <<<")
                    print("   " + "=" * 70)

        # Cooperation HUD
        miner0_runner1 = ep_score_events > 0 and ep_mine_attempts[0] > 0 and ep_mine_attempts[1] == 0
        miner1_runner0 = ep_score_events > 0 and ep_mine_attempts[1] > 0 and ep_mine_attempts[0] == 0
        both_mine_and_score = ep_score_events > 0 and ep_mine_attempts[0] > 0 and ep_mine_attempts[1] > 0

        coop_window.append(
            {
                "macro_counts": ep_macro_counts,
                "mine_attempts": ep_mine_attempts,
                "combat_events": ep_combat_events,
                "score_events": ep_score_events,
                "miner0_runner1": 1 if miner0_runner1 else 0,
                "miner1_runner0": 1 if miner1_runner0 else 0,
                "both_mine_and_score": 1 if both_mine_and_score else 0,
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

            p_m0_r1 = 100.0 * agg_miner0_runner1 / window_len
            p_m1_r0 = 100.0 * agg_miner1_runner0 / window_len
            p_both = 100.0 * agg_both_mine_score / window_len

            print("   ================== COOPERATION HUD ==================")
            print(f"   Window: last {len(coop_window)} episodes")
            print(f"   Avg mines/ep    : Blue0={avg_mines_0:.2f}, Blue1={avg_mines_1:.2f}")
            print(f"   Avg scores/ep   : {avg_scores:.2f}")
            print(
                f"   Avg combat-events/ep (mine/suppress/flag-ish): "
                f"{avg_combat:.2f}"
            )
            print("   Role breakdown (macro-action usage %) over window:")
            for i, name in enumerate(ACTION_NAMES):
                p0 = pct_0[i] if i < len(pct_0) else 0.0
                p1 = pct_1[i] if i < len(pct_1) else 0.0
                print(f"      {name:20s} | Blue0 {p0:5.1f}% | Blue1 {p1:5.1f}%")
            print("   Cooperation patterns (episodes with ≥1 score):")
            print(
                f"      both_mine_and_score              : {p_both:5.1f}% of window"
            )
            print("   =====================================================")

        # Curriculum advance
        if cur_phase != PHASE_SEQUENCE[-1]:
            min_eps = MIN_PHASE_EPISODES[cur_phase]
            target_wr = TARGET_PHASE_WINRATE[cur_phase]
            if (
                phase_episode_count >= min_eps
                and len(phase_recent) >= PHASE_WINRATE_WINDOW
                and phase_wr >= target_wr
            ):
                print(f"[CURRICULUM] Advancing from {cur_phase} → next phase.")
                phase_idx += 1
                phase_episode_count = 0
                phase_recent.clear()

    final_path = os.path.join(CHECKPOINT_DIR, "ctf_final_model.pth")
    torch.save(policy.state_dict(), final_path)
    print(f"\nTraining complete. Final model: {final_path}")
    print(f"Best OP3 eval winrate achieved: {best_op3_wr * 100:.2f}%")
    print(f"Best-so-far OP3 model (if any): {BEST_OP3_SO_FAR_PATH}")
    print(f"Emperor OP3 model (if threshold reached): {BEST_OP3_EMPEROR_PATH}")


if __name__ == "__main__":
    train_ppo_event()
