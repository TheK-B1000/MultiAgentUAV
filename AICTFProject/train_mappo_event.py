"""
train_mappo_event.py (TRUE MAPPO / CTDE, DirectML-safe, GameField CNN-correct)

- Central obs matches rl_policy.py contract: [B, N, C, H, W]
- GRID_ROWS = CNN_ROWS, GRID_COLS = CNN_COLS
- Uses submit_external_actions() (your current GameField API)
- Update uses log_softmax + gather (no torch.distributions in backward)
"""

import os
import random
import copy
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Any, Deque, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ==========================================================
# DEVICE SELECTION (DirectML → CUDA → CPU)
# ==========================================================
try:
    import torch_directml
    HAS_TDML = True
except ImportError:
    torch_directml = None
    HAS_TDML = False


def prefer_device() -> torch.device:
    if HAS_TDML:
        return torch_directml.device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ==========================================================
# IMPORTS (GameField CNN constants)
# ==========================================================
from game_field import GameField, CNN_ROWS, CNN_COLS, NUM_CNN_CHANNELS
from game_manager import GameManager
from macro_actions import MacroAction
from rl_policy import ActorCriticNet, MAPPOBuffer
from policies import OP1RedPolicy, OP2RedPolicy, OP3RedPolicy


# ==========================================================
# CONFIG
# ==========================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Inspired by your request
GRID_ROWS = CNN_ROWS  # 40
GRID_COLS = CNN_COLS  # 30

TOTAL_STEPS = 5_000_000
UPDATE_EVERY = 2_048
PPO_EPOCHS = 10
MINIBATCH_SIZE = 256

LR = 3e-4
CLIP_EPS = 0.2
VALUE_COEF = 1.0
MAX_GRAD_NORM = 0.5

GAMMA = 0.995
GAE_LAMBDA = 0.99

DECISION_WINDOW = 0.7
SIM_DT = 0.1

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

POLICY_SAVE_INTERVAL = UPDATE_EVERY
POLICY_SAMPLE_CHANCE = 0.15

PHASE_SEQUENCE = ["OP1", "OP2", "OP3"]
MIN_PHASE_EPISODES = {"OP1": 500, "OP2": 1000, "OP3": 2000}
TARGET_PHASE_WINRATE = {"OP1": 0.90, "OP2": 0.86, "OP3": 0.80}
PHASE_WINRATE_WINDOW = 50

ENT_COEF_BY_PHASE = {"OP1": 0.04, "OP2": 0.035, "OP3": 0.02}
OP2_ENTROPY_FLOOR = 0.008
OP2_DRAW_PENALTY = -0.8
OP2_SCORE_BONUS = 0.5

TIME_PENALTY_PER_AGENT_PER_MACRO = 0.001

PHASE_CONFIG = {
    "OP1": dict(score_limit=3, max_time=200.0, max_macro_steps=500),
    "OP2": dict(score_limit=3, max_time=200.0, max_macro_steps=450),
    "OP3": dict(score_limit=3, max_time=200.0, max_macro_steps=550),
}

USED_MACROS = [
    MacroAction.GO_TO,
    MacroAction.GRAB_MINE,
    MacroAction.GET_FLAG,
    MacroAction.PLACE_MINE,
    MacroAction.GO_HOME,
]
NUM_ACTIONS = len(USED_MACROS)


# ==========================================================
# ENV (your current GameField only)
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
    grid = [[0] * GRID_COLS for _ in range(GRID_ROWS)]
    env = GameField(grid)

    env.use_internal_policies = True
    env.set_external_control("blue", True)   # trainer controls BLUE
    env.set_external_control("red", False)   # scripted RED acts inside env.update()
    env.external_missing_action_mode = "idle"
    return env


def external_keys_for_agent(env: GameField, agent: Any) -> List[str]:
    keys: List[str] = []
    if hasattr(env, "_external_key_for_agent"):
        try:
            keys.append(str(env._external_key_for_agent(agent)))
        except Exception:
            pass
    if hasattr(agent, "slot_id"):
        keys.append(str(getattr(agent, "slot_id")))
    if hasattr(agent, "unique_id"):
        keys.append(str(getattr(agent, "unique_id")))
    keys.append(f"{agent.side}_{getattr(agent, 'agent_id', 0)}")

    seen = set()
    out = []
    for k in keys:
        if k and k not in seen:
            out.append(k)
            seen.add(k)
    return out


def gm_get_time(gm: GameManager) -> float:
    if hasattr(gm, "get_sim_time") and callable(getattr(gm, "get_sim_time")):
        return float(gm.get_sim_time())
    if hasattr(gm, "sim_time"):
        return float(getattr(gm, "sim_time"))
    # fallback: time elapsed estimate
    if hasattr(gm, "max_time") and hasattr(gm, "current_time"):
        try:
            return float(getattr(gm, "max_time")) - float(getattr(gm, "current_time"))
        except Exception:
            return 0.0
    return 0.0


def zero_obs_like() -> np.ndarray:
    return np.zeros((NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32)


def get_blue_team_obs_in_id_order(env: GameField) -> List[np.ndarray]:
    """
    Returns list length N in agent_id order.
    If disabled: returns zeros for that slot (stabilizes central critic).
    """
    blue_all = list(env.blue_agents)
    blue_all.sort(key=lambda a: getattr(a, "agent_id", 0))

    out: List[np.ndarray] = []
    for a in blue_all:
        if not a.isEnabled():
            out.append(zero_obs_like())
        else:
            out.append(np.array(env.build_observation(a), dtype=np.float32))
    return out


# ==========================================================
# REWARDS
# ==========================================================
def collect_blue_rewards_for_step(
    gm: GameManager,
    blue_agents: List[Any],
    decision_time_start: float,
    cur_phase: str,
) -> Dict[str, float]:
    raw_events = gm.pop_reward_events()  # your GM API

    rewards_sum_by_id: Dict[str, float] = {a.unique_id: 0.0 for a in blue_agents}
    team_r_total = 0.0

    for t_event, agent_id, r in raw_events:
        dt = max(0.0, float(t_event) - float(decision_time_start))
        discounted_r = float(r) * (GAMMA ** dt)

        if agent_id is None:
            team_r_total += discounted_r
        elif agent_id in rewards_sum_by_id:
            rewards_sum_by_id[agent_id] += discounted_r

    # split team rewards evenly
    if abs(team_r_total) > 0.0 and len(rewards_sum_by_id) > 0:
        share = team_r_total / len(rewards_sum_by_id)
        for aid in rewards_sum_by_id:
            rewards_sum_by_id[aid] += share

    # OP2 draw penalty
    if (
        cur_phase == "OP2"
        and gm.game_over
        and gm.blue_score == gm.red_score
        and len(rewards_sum_by_id) > 0
    ):
        per = OP2_DRAW_PENALTY / len(rewards_sum_by_id)
        for aid in rewards_sum_by_id:
            rewards_sum_by_id[aid] += per

    return rewards_sum_by_id


def get_entropy_coef(cur_phase: str, phase_episode_count: int) -> float:
    base = ENT_COEF_BY_PHASE[cur_phase]
    if cur_phase == "OP1":
        start_ent, horizon = 0.05, 1000.0
    elif cur_phase == "OP2":
        start_ent, horizon = 0.05, 1200.0
    else:
        start_ent, horizon = 0.04, 2000.0

    frac = min(1.0, phase_episode_count / horizon)
    coef = float(start_ent - (start_ent - base) * frac)
    if cur_phase == "OP2":
        coef = max(coef, OP2_ENTROPY_FLOOR)
    return coef


# ==========================================================
# ADV / GAE (grouped by traj, bootstrap-correct)
# ==========================================================
def normalize_advantages(adv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = adv.mean()
    var = (adv - mean).pow(2).mean()
    return (adv - mean) / torch.sqrt(var + eps)


def compute_gae_event_grouped_bootstrap(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    dts: torch.Tensor,
    traj_ids: torch.Tensor,
    bootstrap_value: torch.Tensor,
    gamma: float = GAMMA,
    lam: float = GAE_LAMBDA,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-traj GAE. Uses bootstrap_value for the last transition of a traj if not terminal.
    """
    T = rewards.size(0)
    advantages = torch.zeros(T, device=rewards.device)

    traj_to_indices: Dict[int, List[int]] = defaultdict(list)
    for i, tid in enumerate(traj_ids.detach().cpu().tolist()):
        traj_to_indices[int(tid)].append(i)

    bootstrap_value = bootstrap_value.to(rewards.device).reshape(())

    for _, idxs in traj_to_indices.items():
        last_i = idxs[-1]
        next_value = torch.tensor(0.0, device=rewards.device)
        if dones[last_i] < 0.5:
            next_value = bootstrap_value

        next_adv = torch.tensor(0.0, device=rewards.device)

        for i in reversed(idxs):
            gamma_dt = gamma ** dts[i]
            lam_gamma_dt = (gamma * lam) ** dts[i]
            mask = 1.0 - dones[i]

            delta = rewards[i] + gamma_dt * next_value * mask - values[i]
            advantages[i] = delta + lam_gamma_dt * next_adv * mask

            next_adv = advantages[i]
            next_value = values[i]

    returns = advantages + values
    return advantages, returns


def _fix_all_false_rows(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.numel() == 0:
        return mask
    row_sum = mask.sum(dim=-1)
    bad = row_sum == 0
    if bad.any():
        mask = mask.clone()
        mask[bad] = True
    return mask


# ==========================================================
# MAPPO UPDATE (DirectML-safe: log_softmax + gather)
# ==========================================================
def mappo_update(
    policy: ActorCriticNet,
    optimizer: optim.Optimizer,
    buffer: MAPPOBuffer,
    device: torch.device,
    ent_coef: float,
    bootstrap_value: torch.Tensor,
) -> None:
    packed = buffer.to_tensors(device)

    actor_obs = packed[0]       # [T,C,H,W]
    central_obs = packed[1]     # [T,N,C,H,W]
    macro_actions = packed[2]   # [T]
    target_actions = packed[3]  # [T]
    old_log_probs = packed[4]   # [T]
    values = packed[5]          # [T]
    rewards = packed[6]         # [T]
    dones = packed[7]           # [T]
    dts = packed[8]             # [T]
    traj_ids = packed[9]        # [T]
    macro_masks = packed[10]    # [T,n_macros] or [T,0]

    advantages, returns = compute_gae_event_grouped_bootstrap(
        rewards, values, dones, dts, traj_ids, bootstrap_value
    )
    advantages = normalize_advantages(advantages)

    idxs = np.arange(actor_obs.size(0))
    for _ in range(PPO_EPOCHS):
        np.random.shuffle(idxs)
        for start in range(0, actor_obs.size(0), MINIBATCH_SIZE):
            mb = idxs[start:start + MINIBATCH_SIZE]

            mb_actor = actor_obs[mb]
            mb_central = central_obs[mb]
            mb_macro = macro_actions[mb]
            mb_target = target_actions[mb]
            mb_old_logp = old_log_probs[mb]
            mb_adv = advantages[mb]
            mb_ret = returns[mb]

            mb_mask = None
            if macro_masks.numel() > 0 and macro_masks.size(1) > 0:
                mb_mask = _fix_all_false_rows(macro_masks[mb])

            # forward
            new_values = policy.forward_central_critic(mb_central)  # expects [B,N,C,H,W]
            macro_logits, target_logits, _ = policy.forward_actor(mb_actor)

            if mb_mask is not None:
                macro_logits = macro_logits.masked_fill(~mb_mask, -1e10)

            logp_macro_all = F.log_softmax(macro_logits, dim=-1)
            logp_targ_all = F.log_softmax(target_logits, dim=-1)

            new_logp_macro = logp_macro_all.gather(1, mb_macro.unsqueeze(1)).squeeze(1)
            new_logp_targ = logp_targ_all.gather(1, mb_target.unsqueeze(1)).squeeze(1)
            new_logp = new_logp_macro + new_logp_targ

            # entropy (categorical)
            p_macro = torch.exp(logp_macro_all)
            p_targ = torch.exp(logp_targ_all)
            entropy = -(p_macro * logp_macro_all).sum(dim=-1) + -(p_targ * logp_targ_all).sum(dim=-1)

            ratio = torch.exp(new_logp - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (mb_ret - new_values).pow(2).mean()
            loss = policy_loss + VALUE_COEF * value_loss - ent_coef * entropy.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

    buffer.clear()


# ==========================================================
# DEVICE SMOKE TEST (must match rl_policy central shape)
# ==========================================================
def smoke_test_device(policy_ctor, device: torch.device, n_agents: int) -> bool:
    try:
        net = policy_ctor().to(device)
        net.train()

        B = 2
        actor = torch.randn(B, NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS, device=device)
        central = torch.randn(B, n_agents, NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS, device=device)

        macro_logits, target_logits, _ = net.forward_actor(actor)
        v = net.forward_central_critic(central)

        loss = macro_logits.mean() + target_logits.mean() + v.mean()
        loss.backward()
        return True
    except Exception as e:
        print(f"[DEVICE] Smoke test failed on {device}: {type(e).__name__}: {e}")
        return False


# ==========================================================
# MAIN LOOP
# ==========================================================
def train_mappo_event(total_steps: int = TOTAL_STEPS) -> None:
    set_seed(42)

    env = make_env()
    gm = env.getGameManager()

    env.reset_default()
    gm.reset_game(reset_scores=True)

    # sanity prints like your log
    if env.blue_agents:
        sample = env.build_observation(env.blue_agents[0])
        print(f"[train_mappo_event] Sample obs shape: C={len(sample)}, H={len(sample[0])}, W={len(sample[0][0])}")
    print(f"[train_mappo_event] Using CNN constants: C={NUM_CNN_CHANNELS}, H={CNN_ROWS}, W={CNN_COLS}")
    print(f"[train_mappo_event] Using GRID dims: rows={GRID_ROWS}, cols={GRID_COLS}")

    n_agents = getattr(env, "agents_per_team", 2)

    def _make_policy():
        return ActorCriticNet(
            n_macros=len(USED_MACROS),
            n_targets=env.num_macro_targets,
            n_agents=n_agents,
            in_channels=NUM_CNN_CHANNELS,
            height=CNN_ROWS,
            width=CNN_COLS,
        )

    device = prefer_device()
    if HAS_TDML and "privateuseone" in str(device).lower():
        if not smoke_test_device(_make_policy, device, n_agents=n_agents):
            device = torch.device("cpu")

    DEVICE = device
    print(f"[train_mappo_event] Using device: {DEVICE}")

    policy = _make_policy().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    buffer = MAPPOBuffer()

    global_step = 0
    episode_idx = 0

    blue_wins = red_wins = draws = 0
    phase_idx = 0
    phase_episode_count = 0
    phase_recent: List[int] = []
    phase_wr = 0.0

    old_policies_buffer: Deque[Dict[str, Any]] = deque(maxlen=20)

    traj_id_counter = 0
    traj_id_map: Dict[Tuple[int, str], int] = {}

    while global_step < total_steps:
        episode_idx += 1
        phase_episode_count += 1

        cur_phase = PHASE_SEQUENCE[phase_idx]
        phase_cfg = PHASE_CONFIG[cur_phase]
        gm.score_limit = phase_cfg["score_limit"]
        gm.max_time = phase_cfg["max_time"]
        max_steps = int(phase_cfg["max_macro_steps"])
        if hasattr(gm, "set_phase"):
            gm.set_phase(cur_phase)

        # optional self-play (kept)
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
            red_net = _make_policy().to(DEVICE)
            red_net.load_state_dict(random.choice(old_policies_buffer))
            red_net.eval()
            env.set_red_policy_neural(red_net)
            opponent_tag = "SELFPLAY"
        else:
            set_red_policy_for_phase(env, cur_phase)
            opponent_tag = cur_phase

        env.reset_default()
        gm.reset_game(reset_scores=True)

        red_pol = env.policies.get("red")
        if hasattr(red_pol, "reset"):
            red_pol.reset()

        done = False
        ep_return = 0.0
        steps = 0
        prev_blue_score = gm.blue_score

        decision_time_start = gm_get_time(gm)
        traj_id_map.clear()

        while not done and steps < max_steps and global_step < total_steps:
            if global_step > 0 and global_step % POLICY_SAVE_INTERVAL == 0:
                old_policies_buffer.append(copy.deepcopy(policy.state_dict()))

            # CENTRAL OBS at decision boundary: [N,C,H,W] then batch -> [1,N,C,H,W]
            blue_joint_obs = get_blue_team_obs_in_id_order(env)
            central_obs_np = np.stack(blue_joint_obs, axis=0)  # [N,C,H,W]
            central_obs_tensor = torch.tensor(central_obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            blue_agents_enabled = [a for a in env.blue_agents if a.isEnabled()]
            decisions = []
            submit_actions: Dict[str, Tuple[int, int]] = {}

            for agent in blue_agents_enabled:
                actor_obs_np = np.array(env.build_observation(agent), dtype=np.float32)
                actor_obs_tensor = torch.tensor(actor_obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                out = policy.act_mappo(
                    actor_obs_tensor,
                    central_obs_tensor,
                    agent=agent,
                    game_field=env,
                    deterministic=False,
                )

                macro_idx = int(out["macro_action"][0].item())
                target_idx = int(out["target_action"][0].item())
                logp = float(out["log_prob"][0].item())
                val = float(out["value"][0].item())  # CENTRAL value

                mm = out.get("macro_mask", None)
                mm_np = mm.detach().cpu().numpy() if mm is not None else None

                macro_enum = USED_MACROS[macro_idx]
                macro_val = int(getattr(macro_enum, "value", int(macro_enum)))

                key = (episode_idx, agent.unique_id)
                if key not in traj_id_map:
                    traj_id_map[key] = traj_id_counter
                    traj_id_counter += 1
                tid = traj_id_map[key]

                decisions.append(
                    (agent, actor_obs_np, central_obs_np, macro_idx, target_idx, logp, val, tid, mm_np)
                )

                for k in external_keys_for_agent(env, agent):
                    submit_actions[k] = (macro_val, target_idx)

            # Submit external actions (your GameField consumes them on decision boundary)
            env.pending_external_actions.clear()
            env.submit_external_actions(submit_actions)

            # Sim window
            sim_t = 0.0
            while sim_t < (DECISION_WINDOW - 1e-9) and not gm.game_over:
                env.update(SIM_DT)
                sim_t += SIM_DT

            done = gm.game_over
            decision_time_end = gm_get_time(gm)
            dt = max(0.0, float(decision_time_end) - float(decision_time_start))

            rewards = collect_blue_rewards_for_step(gm, env.blue_agents, decision_time_start, cur_phase)

            # time pressure (all blue agents)
            for a in env.blue_agents:
                rewards[a.unique_id] = rewards.get(a.unique_id, 0.0) - TIME_PENALTY_PER_AGENT_PER_MACRO

            # OP2 score bonus
            blue_score_delta = gm.blue_score - prev_blue_score
            if blue_score_delta > 0 and cur_phase == "OP2" and len(rewards) > 0:
                per_agent_bonus = (OP2_SCORE_BONUS * blue_score_delta) / len(rewards)
                for aid in rewards:
                    rewards[aid] += per_agent_bonus
            prev_blue_score = gm.blue_score

            # Bootstrap value at next decision boundary (post-sim)
            with torch.no_grad():
                next_joint_obs = get_blue_team_obs_in_id_order(env)
                next_central_np = np.stack(next_joint_obs, axis=0)
                next_central_tensor = torch.tensor(next_central_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                bootstrap_value = policy.forward_central_critic(next_central_tensor).detach().reshape(())
                if done:
                    bootstrap_value = torch.tensor(0.0, device=DEVICE)

            # Add to buffer (one entry per enabled agent)
            step_reward_sum = 0.0
            for agent, actor_obs_np, central_obs_np, macro_idx, target_idx, logp, val, tid, mm_np in decisions:
                r = rewards.get(agent.unique_id, 0.0)
                step_reward_sum += r

                buffer.add(
                    actor_obs=actor_obs_np,
                    central_obs=central_obs_np,  # [N,C,H,W]
                    macro_action=macro_idx,
                    target_action=target_idx,
                    log_prob=logp,
                    value=val,
                    reward=r,
                    done=done,
                    dt=dt,
                    traj_id=tid,
                    macro_mask=mm_np,
                )
                global_step += 1

            ep_return += step_reward_sum
            steps += 1
            decision_time_start = decision_time_end

            # Update
            if buffer.size() >= UPDATE_EVERY:
                ent = get_entropy_coef(cur_phase, phase_episode_count)
                print(
                    f"[MAPPO UPDATE] step={global_step} episode={episode_idx} "
                    f"phase={cur_phase} ENT={ent:.4f} Opp={opponent_tag}"
                )

                try:
                    mappo_update(policy, optimizer, buffer, DEVICE, ent, bootstrap_value)
                except RuntimeError as e:
                    msg = str(e).lower()
                    if HAS_TDML and "privateuseone" in str(DEVICE).lower():
                        print(f"[DEVICE] MAPPO update failed on {DEVICE}. Falling back to CPU.\n  Error: {e}")
                        DEVICE = torch.device("cpu")
                        policy = _make_policy().to(DEVICE)
                        optimizer = optim.Adam(policy.parameters(), lr=LR)
                        buffer.clear()
                    else:
                        raise

        # Episode result
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
            f"[{episode_idx:5d}] {result:8} | StepR {avg_step_r:+.3f} TermR {ep_return:+.1f} | "
            f"Score {gm.blue_score}:{gm.red_score} | BlueWins: {blue_wins} RedWins: {red_wins} Draws: {draws} | "
            f"PhaseWin {phase_wr*100:.1f}% | Phase={cur_phase} Opp={opponent_tag}"
        )

        # Curriculum advance
        if cur_phase != PHASE_SEQUENCE[-1]:
            min_eps = MIN_PHASE_EPISODES[cur_phase]
            target_wr = TARGET_PHASE_WINRATE[cur_phase]
            if phase_episode_count >= min_eps and len(phase_recent) >= PHASE_WINRATE_WINDOW and phase_wr >= target_wr:
                print(f"[CURRICULUM] Advancing from {cur_phase} → next phase (MAPPO).")
                phase_idx += 1
                phase_episode_count = 0
                phase_recent.clear()
                old_policies_buffer.clear()

        # Periodic checkpoint
        if episode_idx % 50 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"mappo_ckpt_ep{episode_idx}.pth")
            torch.save(policy.state_dict(), ckpt_path)
            print(f"[CKPT] Saved: {ckpt_path}")

    final_path = os.path.join(CHECKPOINT_DIR, "research_mappo_model1.pth")
    torch.save(policy.state_dict(), final_path)
    print(f"\n[MAPPO] Training complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    train_mappo_event()
