"""
MAPPO (Multi-Agent PPO) with central critic: curriculum, league, fixed opponent, self-play.

Same training modes as IPPO: CURRICULUM_LEAGUE, CURRICULUM_NO_LEAGUE, FIXED_OPPONENT, SELF_PLAY.
Uses GameField + decision windows and rl_policy.ActorCriticNet. Saves .pth state_dict.

Usage:
  python -m rl.train_mappo [--mode CURRICULUM_LEAGUE] [--total_steps 500000] ...
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game_field import GameField, CNN_ROWS, CNN_COLS, NUM_CNN_CHANNELS
from game_field import make_game_field
from macro_actions import MacroAction
from rl_policy import ActorCriticNet
from rl.common import batch_by_agent_id, collect_team_uids, pop_reward_events_best_effort, set_global_seed, simulate_decision_window
from rl.curriculum import CurriculumConfig, CurriculumController, CurriculumControllerConfig, CurriculumState, phase_from_tag
from rl.league import EloLeague, OpponentSpec
from config import MAP_NAME, MAP_PATH


class TrainMode(str, Enum):
    CURRICULUM_LEAGUE = "CURRICULUM_LEAGUE"
    CURRICULUM_NO_LEAGUE = "CURRICULUM_NO_LEAGUE"
    FIXED_OPPONENT = "FIXED_OPPONENT"
    SELF_PLAY = "SELF_PLAY"


@dataclass
class MAPPOConfig:
    seed: int = 42
    total_steps: int = 1_000_000
    update_every: int = 2048
    ppo_epochs: int = 10
    minibatch_size: int = 128
    lr: float = 3e-4
    clip_eps: float = 0.2
    clip_range_vf: Optional[float] = 0.2  # value function clipping (SB3-style); prevents value collapse from draw-heavy data
    value_coef: float = 1.0
    max_grad_norm: float = 0.5
    gamma: float = 0.995
    gae_lambda: float = 0.99
    decision_window: float = 0.7
    sim_dt: float = 0.1
    max_macro_steps: int = 600
    terminal_bonus_scale: float = 2.0  # scale win/draw/loss so policy strongly prefers winning over draws
    draw_penalty_scale: float = 1.5   # extra scale on draw penalty (makes draw worse than current baseline)
    entropy_coef: float = 0.03  # encourage exploration, reduce collapse to passive (draw) behavior

    checkpoint_dir: str = "checkpoints_mappo"
    run_tag: str = "mappo_league_curriculum"
    save_every_steps: int = 50_000
    log_every_steps: int = 2_000

    mode: str = TrainMode.CURRICULUM_LEAGUE.value
    fixed_opponent_tag: str = "OP3"
    op3_gate_tag: str = "OP3_HARD"
    snapshot_every_episodes: int = 100
    league_max_snapshots: int = 25
    league_scripted_fallback_tag: str = "OP3"
    league_easy_scripted_elo_threshold: float = 1200.0
    self_play_snapshot_every_episodes: int = 25
    self_play_max_snapshots: int = 25

    curriculum_min_episodes: Dict[str, int] = None
    curriculum_min_winrate: Dict[str, float] = None
    curriculum_winrate_window: int = 50
    curriculum_required_win_by: Dict[str, int] = None

    def __post_init__(self) -> None:
        if self.curriculum_min_episodes is None:
            self.curriculum_min_episodes = {"OP1": 200, "OP2": 200, "OP3": 250}
        if self.curriculum_min_winrate is None:
            self.curriculum_min_winrate = {"OP1": 0.50, "OP2": 0.50, "OP3": 0.55}
        if self.curriculum_required_win_by is None:
            self.curriculum_required_win_by = {"OP1": 0, "OP2": 0, "OP3": 0}


class MAPPORolloutBuffer:
    def __init__(self, n_macros: int) -> None:
        self.n_macros = int(n_macros)
        self.actor_obs: List[np.ndarray] = []
        self.central_obs: List[np.ndarray] = []
        self.macro_actions: List[int] = []
        self.target_actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.next_values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.dts: List[float] = []
        self.traj_ids: List[int] = []
        self.macro_masks: List[np.ndarray] = []
        self.agent_idxs: List[int] = []

    def add(
        self,
        *,
        actor_obs: np.ndarray,
        central_obs: np.ndarray,
        macro_action: int,
        target_action: int,
        log_prob: float,
        value: float,
        next_value: float,
        reward: float,
        done: bool,
        dt: float,
        traj_id: int,
        agent_idx: int,
        macro_mask: Optional[np.ndarray] = None,
    ) -> None:
        self.actor_obs.append(np.array(actor_obs, dtype=np.float32))
        self.central_obs.append(np.array(central_obs, dtype=np.float32))
        self.macro_actions.append(int(macro_action))
        self.target_actions.append(int(target_action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.next_values.append(float(next_value))
        self.agent_idxs.append(int(agent_idx))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.dts.append(float(dt))
        self.traj_ids.append(int(traj_id))
        if macro_mask is None:
            self.macro_masks.append(np.ones((self.n_macros,), dtype=np.bool_))
        else:
            mm = np.array(macro_mask, dtype=np.bool_).reshape(-1)
            if mm.size != self.n_macros:
                mm = np.ones((self.n_macros,), dtype=np.bool_)
            if not mm.any():
                mm[:] = True
            self.macro_masks.append(mm)

    def clear(self) -> None:
        self.__init__(self.n_macros)

    def size(self) -> int:
        return len(self.actor_obs)

    def to_tensors(self, device: torch.device):
        actor_obs = torch.tensor(np.stack(self.actor_obs), dtype=torch.float32, device=device)
        central_obs = torch.tensor(np.stack(self.central_obs), dtype=torch.float32, device=device)
        macro_actions = torch.tensor(self.macro_actions, dtype=torch.long, device=device)
        target_actions = torch.tensor(self.target_actions, dtype=torch.long, device=device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        values = torch.tensor(self.values, dtype=torch.float32, device=device)
        next_values = torch.tensor(self.next_values, dtype=torch.float32, device=device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)
        dts = torch.tensor(self.dts, dtype=torch.float32, device=device)
        traj_ids = torch.tensor(self.traj_ids, dtype=torch.long, device=device)
        agent_idxs = torch.tensor(self.agent_idxs, dtype=torch.long, device=device)
        macro_masks = torch.tensor(np.stack(self.macro_masks), dtype=torch.bool, device=device)
        return (
            actor_obs,
            central_obs,
            macro_actions,
            target_actions,
            old_log_probs,
            values,
            next_values,
            rewards,
            dones,
            dts,
            traj_ids,
            agent_idxs,
            macro_masks,
        )


def compute_gae_event_grouped(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    dts: np.ndarray,
    traj_ids: np.ndarray,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    T = rewards.shape[0]
    advantages = np.zeros((T,), dtype=np.float32)
    traj_to_idxs: Dict[int, List[int]] = {}
    for i in range(T):
        traj_to_idxs.setdefault(int(traj_ids[i]), []).append(i)
    for _, idxs in traj_to_idxs.items():
        next_adv = 0.0
        for i in reversed(idxs):
            dt = float(dts[i])
            gamma_dt = gamma ** dt
            lam_gamma_dt = (gamma * lam) ** dt
            mask = 1.0 - float(dones[i])
            delta = float(rewards[i]) + gamma_dt * float(next_values[i]) * mask - float(values[i])
            advantages[i] = float(delta + lam_gamma_dt * next_adv * mask)
            next_adv = float(advantages[i])
    returns = advantages + values.astype(np.float32)
    return advantages.astype(np.float32), returns.astype(np.float32)


def normalize_advantages(adv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = adv.mean()
    var = (adv - mean).pow(2).mean()
    return (adv - mean) / torch.sqrt(var + eps)


def mappo_update(
    policy_train: ActorCriticNet,
    optimizer: optim.Optimizer,
    buffer: MAPPORolloutBuffer,
    device: torch.device,
    cfg: MAPPOConfig,
) -> None:
    (
        actor_obs,
        central_obs,
        macro_actions,
        target_actions,
        old_log_probs,
        values,
        next_values,
        rewards,
        dones,
        dts,
        traj_ids,
        agent_idxs,
        macro_masks,
    ) = buffer.to_tensors(device)

    if actor_obs.size(0) == 0:
        buffer.clear()
        return

    adv_np, ret_np = compute_gae_event_grouped(
        rewards.detach().cpu().numpy(),
        values.detach().cpu().numpy(),
        next_values.detach().cpu().numpy(),
        dones.detach().cpu().numpy(),
        dts.detach().cpu().numpy(),
        traj_ids.detach().cpu().numpy(),
        gamma=cfg.gamma,
        lam=cfg.gae_lambda,
    )
    advantages = torch.tensor(adv_np, dtype=torch.float32, device=device)
    returns = torch.tensor(ret_np, dtype=torch.float32, device=device)
    advantages = normalize_advantages(advantages)

    policy_train.train()
    T = actor_obs.size(0)
    for _ in range(int(cfg.ppo_epochs)):
        perm = np.random.permutation(T).astype(np.int64)
        for start in range(0, T, int(cfg.minibatch_size)):
            end = min(start + int(cfg.minibatch_size), T)
            idx = torch.tensor(perm[start:end], dtype=torch.long, device=device)
            mb_actor = actor_obs.index_select(0, idx)
            mb_central = central_obs.index_select(0, idx)
            mb_macro = macro_actions.index_select(0, idx)
            mb_target = target_actions.index_select(0, idx)
            mb_old_logp = old_log_probs.index_select(0, idx)
            mb_adv = advantages.index_select(0, idx)
            mb_ret = returns.index_select(0, idx)
            mb_old_v = values.index_select(0, idx)
            mb_agent_idx = agent_idxs.index_select(0, idx)
            mb_mask = macro_masks.index_select(0, idx)

            new_values_all = policy_train.forward_central_critic(mb_central)
            new_values = new_values_all.gather(1, mb_agent_idx[:, None]).squeeze(1)
            new_logp, entropy, _ = policy_train.evaluate_actions(
                mb_actor, mb_macro, mb_target, macro_mask_batch=mb_mask
            )
            new_logp = new_logp.reshape(-1)
            entropy = entropy.reshape(-1)

            ratio = torch.exp(new_logp - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function clipping (SB3 / OpenAI-style): clip delta from old value to prevent value collapse.
            # Without this, a run of draws (negative returns) lets the value head chase down and the policy
            # shifts toward "safe" (draw) behavior. clip_range_vf limits change per update (e.g. 0.2).
            clip_vf = getattr(cfg, "clip_range_vf", None)
            if clip_vf is not None and float(clip_vf) > 0:
                values_pred = mb_old_v + (new_values - mb_old_v).clamp(-float(clip_vf), float(clip_vf))
            else:
                values_pred = new_values
            value_loss = (mb_ret - values_pred).pow(2).mean()

            loss = policy_loss + cfg.value_coef * value_loss - float(cfg.entropy_coef) * entropy.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy_train.parameters(), cfg.max_grad_norm)
            optimizer.step()

    buffer.clear()


def _enforce_league_snapshot_limit(league: EloLeague, max_snapshots: int) -> None:
    while len(league.snapshots) > max(1, int(max_snapshots)):
        league.snapshots.pop(0)
    try:
        league.snapshots = league.snapshots[-int(max_snapshots):]
    except Exception:
        pass


def train_mappo(cfg: Optional[MAPPOConfig] = None) -> None:
    cfg = cfg or MAPPOConfig()
    set_global_seed(cfg.seed)
    mode = str(cfg.mode or TrainMode.CURRICULUM_LEAGUE.value).strip().upper()

    if mode == TrainMode.FIXED_OPPONENT.value:
        red_tag = str(cfg.fixed_opponent_tag).upper()
        phase_name = red_tag
    elif mode == TrainMode.SELF_PLAY.value:
        red_tag = "OP3"
        phase_name = "SELF_PLAY"
    else:
        red_tag = "OP1"
        phase_name = "OP1"

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    env = make_game_field(
        map_name=MAP_NAME or None,
        map_path=MAP_PATH or None,
        rows=CNN_ROWS,
        cols=CNN_COLS,
    )
    env.set_external_control("blue", True)
    env.set_external_control("red", False)
    env.use_internal_policies = True
    env.reset_default()
    gm = env.getGameManager()

    n_agents = len(getattr(env, "blue_agents", [])) or getattr(env, "agents_per_team", 2)
    n_macros = int(env.n_macros)
    n_targets = int(env.num_macro_targets)

    policy_train = ActorCriticNet(
        n_macros=n_macros,
        n_targets=n_targets,
        n_agents=n_agents,
        in_channels=NUM_CNN_CHANNELS,
        height=CNN_ROWS,
        width=CNN_COLS,
    )
    policy_act = ActorCriticNet(
        n_macros=n_macros,
        n_targets=n_targets,
        n_agents=n_agents,
        in_channels=NUM_CNN_CHANNELS,
        height=CNN_ROWS,
        width=CNN_COLS,
    )
    policy_act.load_state_dict(policy_train.state_dict())
    policy_act.eval()

    optimizer = optim.Adam(policy_train.parameters(), lr=cfg.lr, foreach=False)
    buffer = MAPPORolloutBuffer(n_macros=n_macros)

    league: Optional[EloLeague] = None
    curriculum: Optional[CurriculumState] = None
    controller: Optional[CurriculumController] = None
    league_mode = False
    current_phase = phase_name

    if mode == TrainMode.CURRICULUM_LEAGUE.value:
        league = EloLeague(
            seed=cfg.seed,
            k_factor=32.0,
            matchmaking_tau=250.0,
            scripted_floor=0.50,
            species_prob=0.20,
            snapshot_prob=0.30,
        )
        curriculum = CurriculumState(
            CurriculumConfig(
                phases=["OP1", "OP2", "OP3"],
                min_episodes=cfg.curriculum_min_episodes,
                min_winrate=cfg.curriculum_min_winrate,
                winrate_window=cfg.curriculum_winrate_window,
                required_win_by={"OP1": 0, "OP2": 1, "OP3": 1},
                elo_margin=80.0,
                switch_to_league_after_op3_win=False,
            )
        )
        controller = CurriculumController(
            CurriculumControllerConfig(
                seed=cfg.seed,
                enable_snapshots=False,
                enable_species=True,
            ),
            league=league,
        )
        current_phase = curriculum.phase
        red_tag = current_phase
    elif mode == TrainMode.CURRICULUM_NO_LEAGUE.value:
        curriculum = CurriculumState(
            CurriculumConfig(
                phases=["OP1", "OP2", "OP3"],
                min_episodes=cfg.curriculum_min_episodes,
                min_winrate=cfg.curriculum_min_winrate,
                winrate_window=cfg.curriculum_winrate_window,
                required_win_by=cfg.curriculum_required_win_by,
                elo_margin=-1000.0,
                switch_to_league_after_op3_win=False,
            )
        )
        current_phase = curriculum.phase
        red_tag = current_phase
    elif mode == TrainMode.SELF_PLAY.value:
        league = EloLeague(
            seed=cfg.seed,
            k_factor=32.0,
            matchmaking_tau=250.0,
            scripted_floor=0.50,
            species_prob=0.20,
            snapshot_prob=0.30,
        )
        init_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_selfplay_init.pth")
        try:
            torch.save(policy_train.state_dict(), init_path)
            league.add_snapshot(init_path)
        except Exception as exc:
            print(f"[WARN] MAPPO self-play init save failed: {exc}")

    global_step = 0
    episode_idx = 0
    traj_id_counter = 0
    traj_id_map: Dict[Tuple[int, str], int] = {}
    win_count = 0
    loss_count = 0
    draw_count = 0

    def build_central_obs(agents_sorted: List[Any]) -> np.ndarray:
        obs_list = []
        for i in range(n_agents):
            if i < len(agents_sorted) and agents_sorted[i] is not None and agents_sorted[i].isEnabled():
                obs_list.append(np.asarray(env.build_observation(agents_sorted[i]), dtype=np.float32))
            else:
                obs_list.append(np.zeros((NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32))
        return np.stack(obs_list, axis=0)

    def set_red_for_episode(tag: str) -> None:
        gm.set_phase(phase_from_tag(tag))
        env.set_red_opponent(tag)

    # Set initial red opponent
    set_red_for_episode(red_tag)

    while global_step < int(cfg.total_steps):
        episode_idx += 1
        if curriculum is not None:
            curriculum.phase_episode_count += 1

        set_red_for_episode(red_tag)
        env.reset_default()
        done = False
        steps = 0
        traj_id_map.clear()

        while (not done) and steps < int(cfg.max_macro_steps) and global_step < int(cfg.total_steps):
            blue_agents = batch_by_agent_id(env.blue_agents)
            blue_agents_enabled = [a for a in blue_agents if a is not None and a.isEnabled()]

            central_obs_np = build_central_obs(blue_agents)
            central_obs_tensor = torch.tensor(central_obs_np, dtype=torch.float32).unsqueeze(0)

            submit_actions: Dict[str, Tuple[int, Any]] = {}
            decisions = []

            for agent in blue_agents_enabled:
                obs_np = np.asarray(env.build_observation(agent), dtype=np.float32)
                obs_tensor = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
                out = policy_act.act(obs_tensor, agent=agent, game_field=env, deterministic=False)

                macro_idx = int(out["macro_action"][0].item())
                target_idx = int(out["target_action"][0].item())
                logp = float(out["log_prob"][0].item())
                val = float(out["value"][0].item())
                macro_mask = out.get("macro_mask", None)

                uid = str(getattr(agent, "unique_id", f"{agent.side}_{agent.agent_id}"))
                key = (episode_idx, uid)
                tid = traj_id_map.get(key)
                if tid is None:
                    traj_id_map[key] = traj_id_counter
                    tid = traj_id_counter
                    traj_id_counter += 1

                decisions.append((agent, uid, obs_np, central_obs_np, macro_idx, target_idx, logp, val, tid, macro_mask))
                submit_actions[uid] = (macro_idx, target_idx)

            env.submit_external_actions(submit_actions)
            simulate_decision_window(env, gm, cfg.decision_window, cfg.sim_dt)

            done = bool(getattr(gm, "game_over", False))
            dt = float(cfg.decision_window)

            reward_events = pop_reward_events_best_effort(gm)
            per_agent_reward: Dict[str, float] = {uid: 0.0 for uid in collect_team_uids(blue_agents_enabled)}
            for _t, aid, r in reward_events:
                if aid is None:
                    continue
                key = str(aid)
                if key in per_agent_reward:
                    per_agent_reward[key] += float(r)
            # Credit terminal outcome to last step; explicit negative for draws so they are punished
            if done:
                bs, rs = int(gm.blue_score), int(gm.red_score)
                scale = float(getattr(cfg, "terminal_bonus_scale", 2.0))
                draw_scale = float(getattr(cfg, "draw_penalty_scale", 1.5))
                if bs > rs:
                    bonus = 1.0 * scale
                elif bs < rs:
                    bonus = -1.0 * scale
                else:
                    bonus = -1.0 * scale * draw_scale
                n_blue = max(1, len(per_agent_reward))
                for uid in per_agent_reward:
                    per_agent_reward[uid] += float(bonus) / n_blue

            with torch.no_grad():
                central_tensor_cur = torch.tensor(central_obs_np, dtype=torch.float32).unsqueeze(0)
                v_all_cur = policy_act.forward_central_critic(central_tensor_cur).squeeze(0).cpu().numpy()
            next_central = build_central_obs(blue_agents)
            with torch.no_grad():
                next_central_tensor = torch.tensor(next_central, dtype=torch.float32).unsqueeze(0)
                v_all_next = policy_act.forward_central_critic(next_central_tensor).squeeze(0).cpu().numpy()

            for i, (agent, uid, obs_np, central_obs_np, macro_idx, target_idx, logp, _val, tid, macro_mask) in enumerate(decisions):
                agent_done = done or (not agent.isEnabled())
                idx = int(getattr(agent, "agent_id", i))
                idx = max(0, min(idx, len(v_all_cur) - 1, len(v_all_next) - 1))
                old_v = float(v_all_cur[idx])
                nv = 0.0 if agent_done else float(v_all_next[idx])
                reward = float(per_agent_reward.get(uid, 0.0))
                buffer.add(
                    actor_obs=obs_np,
                    central_obs=central_obs_np,
                    macro_action=macro_idx,
                    target_action=target_idx,
                    log_prob=logp,
                    value=old_v,
                    next_value=nv,
                    reward=reward,
                    done=agent_done,
                    dt=dt,
                    traj_id=tid,
                    agent_idx=idx,
                    macro_mask=macro_mask.detach().cpu().numpy() if torch.is_tensor(macro_mask) else macro_mask,
                )
                global_step += 1

            steps += 1

            if buffer.size() >= int(cfg.update_every):
                mappo_update(policy_train, optimizer, buffer, torch.device("cpu"), cfg)
                policy_act.load_state_dict(policy_train.state_dict())
                policy_act.eval()

        blue_score = int(gm.blue_score)
        red_score = int(gm.red_score)
        win_by = blue_score - red_score
        if blue_score > red_score:
            result = "WIN"
            win_count += 1
            win_val = 1.0
        elif blue_score < red_score:
            result = "LOSS"
            loss_count += 1
            win_val = 0.0
        else:
            result = "DRAW"
            draw_count += 1
            win_val = 0.5
        opp_key = f"SCRIPTED:{red_tag}"

        if mode == TrainMode.FIXED_OPPONENT.value:
            print(
                f"[MAPPO|FIXED] ep={episode_idx} result={result} score={blue_score}:{red_score} "
                f"opp={opp_key} W={win_count} | L={loss_count} | D={draw_count}"
            )
        elif mode == TrainMode.CURRICULUM_NO_LEAGUE.value and curriculum is not None:
            curriculum.record_result(red_tag, win_val)
            curriculum.advance_if_ready(1200.0, 1200.0, win_by)
            current_phase = curriculum.phase
            red_tag = current_phase
            print(
                f"[MAPPO|CURR] ep={episode_idx} result={result} score={blue_score}:{red_score} "
                f"phase={current_phase} opp={opp_key} W={win_count} | L={loss_count} | D={draw_count}"
            )
        elif mode == TrainMode.CURRICULUM_LEAGUE.value and curriculum is not None and controller is not None and league is not None:
            league.update_elo(opp_key, win_val)
            controller.record_result(opp_key, win_val)
            is_scripted = opp_key.startswith("SCRIPTED:")
            if is_scripted:
                curriculum.record_result(red_tag, win_val)
            else:
                curriculum.record_result(curriculum.phase, win_val)
            phase = curriculum.phase
            if is_scripted:
                opp_rating = league.get_rating(opp_key)
                if curriculum.advance_if_ready(
                    learner_rating=league.learner_rating,
                    opponent_rating=opp_rating,
                    win_by=win_by,
                ):
                    phase = curriculum.phase
            if phase == "OP3" and is_scripted:
                min_eps = int(curriculum.config.min_episodes.get("OP3", 0))
                min_wr = float(curriculum.config.min_winrate.get("OP3", 0.0))
                req_win_by = int(curriculum.config.required_win_by.get("OP3", 0))
                meets_eps = curriculum.phase_episode_count >= min_eps
                meets_wr = curriculum.phase_winrate("OP3") >= min_wr
                meets_score = (req_win_by <= 0) or (win_by >= req_win_by)
                gate_tag = str(cfg.op3_gate_tag).upper()
                gate_ok = opp_key.endswith(f":{gate_tag}")
                if meets_eps and meets_wr and meets_score and gate_ok:
                    league_mode = True
            if (episode_idx % int(cfg.snapshot_every_episodes)) == 0:
                _enforce_league_snapshot_limit(league, cfg.league_max_snapshots)
                path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_league_snapshot_ep{episode_idx:06d}.pth")
                try:
                    torch.save(policy_train.state_dict(), path)
                    league.add_snapshot(path)
                except Exception as exc:
                    print(f"[WARN] MAPPO league snapshot save failed: {exc}")
                _enforce_league_snapshot_limit(league, cfg.league_max_snapshots)
            next_spec = controller.select_opponent(curriculum.phase, league_mode=league_mode)
            if next_spec.kind == "SNAPSHOT":
                tag = str(getattr(cfg, "league_scripted_fallback_tag", "OP3")).upper()
                next_spec = OpponentSpec(kind="SCRIPTED", key=tag, rating=league.get_rating(f"SCRIPTED:{tag}"))
            elif (next_spec.kind == "SCRIPTED" and next_spec.key == "OP3_HARD" and
                  league.learner_rating < float(getattr(cfg, "league_easy_scripted_elo_threshold", 1200.0))):
                next_spec = OpponentSpec(kind="SCRIPTED", key="OP3", rating=league.get_rating("SCRIPTED:OP3"))
            red_tag = next_spec.key
            mode_str = "LEAGUE" if league_mode else "CURR"
            base = (
                f"[MAPPO|{mode_str}] ep={episode_idx} result={result} score={blue_score}:{red_score} "
                f"phase={phase} opp={opp_key} W={win_count} | L={loss_count} | D={draw_count}"
            )
            if league_mode:
                base = f"{base} elo={league.learner_rating:.1f}"
            print(base)
        elif mode == TrainMode.SELF_PLAY.value and league is not None:
            if (episode_idx % int(cfg.self_play_snapshot_every_episodes)) == 0:
                _enforce_league_snapshot_limit(league, cfg.self_play_max_snapshots)
                max_s = max(1, cfg.self_play_max_snapshots)
                slot = (episode_idx // int(cfg.self_play_snapshot_every_episodes)) % max_s + 1
                path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_selfplay_snapshot_slot{slot:03d}.pth")
                try:
                    torch.save(policy_train.state_dict(), path)
                    league.add_snapshot(path)
                except Exception as exc:
                    print(f"[WARN] MAPPO self-play snapshot save failed: {exc}")
                _enforce_league_snapshot_limit(league, cfg.self_play_max_snapshots)
            print(
                f"[MAPPO|SELF] ep={episode_idx} result={result} score={blue_score}:{red_score} "
                f"snapshots={len(league.snapshots)} W={win_count} | L={loss_count} | D={draw_count}"
            )

        if cfg.save_every_steps and global_step > 0 and global_step % int(cfg.save_every_steps) < int(cfg.update_every) * 2:
            path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_step{global_step}.pth")
            try:
                torch.save(policy_train.state_dict(), path)
            except Exception as exc:
                print(f"[WARN] MAPPO step save failed: {exc}")

    final_path = os.path.join(cfg.checkpoint_dir, f"final_{cfg.run_tag}.pth")
    try:
        torch.save(policy_train.state_dict(), final_path)
        print(f"[MAPPO] Training complete. Final model saved to: {final_path}")
    except Exception as exc:
        print(f"[WARN] MAPPO final save failed: {exc}")
    print(f"[MAPPO] Done. total_steps={global_step} episodes={episode_idx}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="MAPPO training: curriculum, league, fixed, self-play")
    p.add_argument("--mode", type=str, default=TrainMode.CURRICULUM_LEAGUE.value,
                   choices=[TrainMode.CURRICULUM_LEAGUE.value, TrainMode.CURRICULUM_NO_LEAGUE.value,
                            TrainMode.FIXED_OPPONENT.value, TrainMode.SELF_PLAY.value],
                   help="CURRICULUM_LEAGUE | CURRICULUM_NO_LEAGUE | FIXED_OPPONENT | SELF_PLAY")
    p.add_argument("--total_steps", type=int, default=1_000_000)
    p.add_argument("--update_every", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_tag", type=str, default="mappo_league_curriculum")
    p.add_argument("--log_every_steps", type=int, default=2000)
    p.add_argument("--fixed_opponent_tag", type=str, default="OP3", help="For FIXED_OPPONENT mode")
    args = p.parse_args()
    cfg = MAPPOConfig(
        mode=args.mode,
        total_steps=args.total_steps,
        update_every=args.update_every,
        lr=args.lr,
        seed=args.seed,
        run_tag=args.run_tag,
        log_every_steps=args.log_every_steps,
        fixed_opponent_tag=args.fixed_opponent_tag,
    )
    train_mappo(cfg)
