import os
import time
import math
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
from policies import OP1RedPolicy, OP2RedPolicy, OP3RedPolicy

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# CONFIG
GRID_ROWS = 30
GRID_COLS = 40

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO hyperparameters
TOTAL_STEPS = 3_000_000
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

COOP_HUD_EVERY = 50
COOP_WINDOW = 50
NUM_ACTIONS = len(MacroAction)
ACTION_NAMES = [ma.name for ma in sorted(MacroAction, key=lambda m: m.value)]


PHASE_SEQUENCE = ["OP1", "OP2", "OP3"]

MIN_PHASE_EPISODES = {
    "OP1": 500,
    "OP2": 1000,
    "OP3": 2000,
}

TARGET_PHASE_WINRATE = {
    "OP1": 0.99,
    "OP2": 0.90,
    "OP3": 0.80,
}

PHASE_WINRATE_WINDOW = 50

ENT_COEF_BY_PHASE = {
    "OP1": 0.03,
    "OP2": 0.025,
    "OP3": 0.02,
}

PHASE_CONFIG = {
    "OP1": dict(score_limit=3, max_time=200.0, max_macro_steps=500),
    "OP2": dict(score_limit=3, max_time=200.0, max_macro_steps=450),
    "OP3": dict(score_limit=3, max_time=200.0, max_macro_steps=550),
}


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
    # BLUE is controlled externally by PPO (we call apply_macro_action)
    env.set_external_control("blue", True)
    # RED is controlled by scripted OPx policies
    env.set_external_control("red", False)
    return env


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

def collect_blue_rewards_for_step(gm: GameManager, blue_agents, mix_alpha: float = 0.0, cur_phase: str = "OP1", ):
    raw = gm.get_step_rewards()

    indiv_rewards_by_id = {a.unique_id: 0.0 for a in blue_agents}
    if not indiv_rewards_by_id:
        return indiv_rewards_by_id

    # team reward under key None (if your GM uses that)
    team_r = raw.get(None, 0.0)
    if abs(team_r) > 0.0:
        share = team_r / len(indiv_rewards_by_id)
        for aid in indiv_rewards_by_id:
            indiv_rewards_by_id[aid] += share

    # agent-specific rewards
    for aid, r in raw.items():
        if aid is None:
            continue
        if aid in indiv_rewards_by_id:
            indiv_rewards_by_id[aid] += r

    return indiv_rewards_by_id

def get_entropy_coef(cur_phase: str, phase_episode_count: int, phase_wr: float) -> float:
    base = ENT_COEF_BY_PHASE[cur_phase]

    if cur_phase == "OP1":
        start_ent, horizon = 0.05, 300.0
    elif cur_phase == "OP2":
        start_ent, horizon = 0.03, 500.0
    else:  # OP3
        start_ent, horizon = 0.03, 800.0

    frac = min(1.0, phase_episode_count / horizon)
    return float(start_ent - (start_ent - base) * frac)

def train_ppo_event(total_steps=TOTAL_STEPS):
    set_seed(42)

    env = make_env()
    gm = env.getGameManager()

    env.reset_default()
    if env.blue_agents:
        dummy_obs = env.build_observation(env.blue_agents[0])
    else:
        dummy_obs = [0.0] * 42

    obs_dim = len(dummy_obs)
    n_actions = len(MacroAction)

    policy = ActorCriticNet(input_dim=obs_dim, n_actions=n_actions).to(DEVICE)
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

    while global_step < total_steps:
        episode_idx += 1
        phase_episode_count += 1

        cur_phase = PHASE_SEQUENCE[phase_idx]
        phase_cfg = PHASE_CONFIG[cur_phase]
        gm.score_limit = phase_cfg["score_limit"]
        gm.max_time = phase_cfg["max_time"]
        max_steps = phase_cfg["max_macro_steps"]
        gm.set_phase(cur_phase)

        set_red_policy_for_phase(env, cur_phase)
        opponent_tag = cur_phase

        env.reset_default()
        gm.reset_game(reset_scores=True)

        # Reset OP3 per-episode state
        red_pol = env.policies.get("red")
        if hasattr(red_pol, "reset"):
            red_pol.reset()

        # make sure per-episode stats are clean
        gm.blue_mine_kills_this_episode = 0
        gm.red_mine_kills_this_episode = 0
        gm.mines_triggered_by_red_this_episode = 0

        done = False
        ep_return = 0.0
        steps = 0

        ENT_COEF = get_entropy_coef(cur_phase, phase_episode_count, phase_wr)

        ep_macro_counts = {0: [0] * NUM_ACTIONS, 1: [0] * NUM_ACTIONS}
        ep_mine_attempts = {0: 0, 1: 0}
        ep_combat_events = 0
        ep_score_events = 0
        prev_blue_score = gm.blue_score
        ep_mines_placed_by_uid = {}

        while not done and steps < max_steps and global_step < total_steps:
            blue_agents = [a for a in env.blue_agents if a.isEnabled()]
            decisions = []

            # === Blue macro decisions ===
            for agent in blue_agents:
                obs = env.build_observation(agent)
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
                out = policy.act(
                    obs_tensor,
                    agent=agent,
                    game_field=env,
                    deterministic=False,
                )
                a = int(out["action"][0].item())
                logp = float(out["log_prob"][0].item())
                val = float(out["value"][0].item())

                local_id = getattr(agent, "agent_id", 0)
                if local_id in ep_macro_counts and 0 <= a < NUM_ACTIONS:
                    ep_macro_counts[local_id][a] += 1
                    if MacroAction(a) == MacroAction.PLACE_MINE:
                        ep_mine_attempts[local_id] += 1

                if agent.unique_id not in ep_mines_placed_by_uid:
                    ep_mines_placed_by_uid[agent.unique_id] = 0

                # Enemy flag distance before moving (for optional shaping / debug)
                side = agent.getSide()
                if hasattr(gm, "get_enemy_flag_position"):
                    ex, ey = gm.get_enemy_flag_position(side)
                else:
                    if side == "blue":
                        ex, ey = gm.red_flag_position
                    else:
                        ex, ey = gm.blue_flag_position
                prev_flag_dist = math.dist([agent.x, agent.y], [ex, ey])

                env.apply_macro_action(agent, MacroAction(a))

                decisions.append(
                    (agent, obs, a, logp, val, prev_flag_dist, ex, ey)
                )

            # simulate until next decision window
            sim_t = 0.0
            while sim_t < DECISION_WINDOW and not gm.game_over:
                env.update(SIM_DT)
                sim_t += SIM_DT

            dt = sim_t
            done = gm.game_over

            # collect rewards from environment
            rewards = collect_blue_rewards_for_step(
                gm, blue_agents, mix_alpha=0.0, cur_phase=cur_phase
            )

            # === SMALL TIME PENALTY ===
            for agent in env.blue_agents:
                uid = agent.unique_id
                rewards[uid] = rewards.get(uid, 0.0) - 0.001

            # No extra shaping
            for agent, _, act, _, _, prev_flag_dist, ex, ey in decisions:
                uid = agent.unique_id
                if MacroAction(act) == MacroAction.PLACE_MINE:
                    ep_mines_placed_by_uid[uid] = ep_mines_placed_by_uid.get(uid, 0) + 1

            # HUD stats (scores & combat)
            blue_score_delta = gm.blue_score - prev_blue_score
            if blue_score_delta > 0:
                ep_score_events += blue_score_delta
            prev_blue_score = gm.blue_score

            total_team_reward = sum(rewards.values())
            if total_team_reward > 0.0 and blue_score_delta == 0:
                ep_combat_events += 1

            # add to buffer
            step_reward_sum = 0.0
            for agent, obs, act, logp, val, _, _, _ in decisions:
                r = rewards.get(agent.unique_id, 0.0)
                step_reward_sum += r
                buffer.add(obs, act, logp, val, r, done, dt)
                global_step += 1

            ep_return += step_reward_sum
            steps += 1

            # PPO update
            if buffer.size() >= UPDATE_EVERY:
                print(
                    f"[UPDATE] step={global_step} episode={episode_idx} "
                    f"phase={cur_phase} ENT={ENT_COEF:.4f} Opp={opponent_tag}"
                )
                ppo_update(policy, optimizer, buffer, DEVICE, ENT_COEF)

        # ------------- Episode end: result + stats -------------
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

        # Cooperation HUD
        miner0_runner1 = ep_score_events > 0 and ep_mine_attempts[0] > 0 and ep_mine_attempts[1] == 0
        miner1_runner0 = ep_score_events > 0 and ep_mine_attempts[1] > 0 and ep_mine_attempts[0] == 0
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

            p_m0_r1 = 100.0 * agg_miner0_runner1 / window_len
            p_m1_r0 = 100.0 * agg_miner1_runner0 / window_len
            p_both = 100.0 * agg_both_mine_score / window_len

            blue_kills = sum(ep["blue_mine_kills"] for ep in coop_window) / window_len
            red_kills = sum(ep["red_mine_kills"] for ep in coop_window) / window_len
            enemy_half = sum(ep["mines_placed_in_enemy_half"] for ep in coop_window) / window_len
            triggered = sum(ep["mines_triggered_by_red"] for ep in coop_window) / window_len
            total_mines = avg_mines_0 + avg_mines_1

            print("   ================== COOPERATION HUD ==================")
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

        # ---------------------------
        # Curriculum advance
        # ---------------------------
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

    # BEST TRAINED BLUE AGENT
    final_path = os.path.join(CHECKPOINT_DIR, "ctf_fixed_blue_op3.pth")
    torch.save(policy.state_dict(), final_path)
    print(f"\nTraining complete. Final model saved to: {final_path}")

if __name__ == "__main__":
    train_ppo_event()
