from __future__ import annotations

import os
from dataclasses import dataclass
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
from rl.curriculum import CurriculumConfig, CurriculumState
from config import MAP_NAME, MAP_PATH


@dataclass
class MAPPOConfig:
    seed: int = 42
    total_steps: int = 1_000_000
    update_every: int = 2048
    ppo_epochs: int = 10
    minibatch_size: int = 128
    lr: float = 3e-4
    clip_eps: float = 0.2
    value_coef: float = 1.0
    max_grad_norm: float = 0.5
    gamma: float = 0.995
    gae_lambda: float = 0.99
    decision_window: float = 0.7
    sim_dt: float = 0.1
    max_macro_steps: int = 600
    checkpoint_dir: str = "checkpoints"


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
        macro_mask: Optional[np.ndarray] = None,
    ) -> None:
        self.actor_obs.append(np.array(actor_obs, dtype=np.float32))
        self.central_obs.append(np.array(central_obs, dtype=np.float32))
        self.macro_actions.append(int(macro_action))
        self.target_actions.append(int(target_action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.next_values.append(float(next_value))
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
            mb_mask = macro_masks.index_select(0, idx)

            new_values = policy_train.forward_central_critic(mb_central).reshape(-1)
            new_logp, entropy, _ = policy_train.evaluate_actions(
                mb_actor, mb_macro, mb_target, macro_mask_batch=mb_mask
            )
            new_logp = new_logp.reshape(-1)
            entropy = entropy.reshape(-1)

            ratio = torch.exp(new_logp - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (mb_ret - new_values).pow(2).mean()
            loss = policy_loss + cfg.value_coef * value_loss - 0.01 * entropy.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy_train.parameters(), cfg.max_grad_norm)
            optimizer.step()

    buffer.clear()


def train_mappo(cfg: Optional[MAPPOConfig] = None) -> None:
    cfg = cfg or MAPPOConfig()
    set_global_seed(cfg.seed)

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

    curriculum = CurriculumState(
        CurriculumConfig(
            phases=["OP1", "OP2", "OP3"],
            min_episodes={"OP1": 150, "OP2": 150, "OP3": 0},
            min_winrate={"OP1": 0.55, "OP2": 0.55, "OP3": 0.50},
            winrate_window=50,
            required_win_by={"OP1": 1, "OP2": 1, "OP3": 0},
            elo_margin=0.0,
            switch_to_league_after_op3_win=False,
        )
    )

    global_step = 0
    episode_idx = 0
    traj_id_counter = 0
    traj_id_map: Dict[Tuple[int, str], int] = {}

    def build_central_obs(agents_sorted: List[Any]) -> np.ndarray:
        obs_list = []
        for i in range(n_agents):
            if i < len(agents_sorted) and agents_sorted[i] is not None and agents_sorted[i].isEnabled():
                obs_list.append(np.asarray(env.build_observation(agents_sorted[i]), dtype=np.float32))
            else:
                obs_list.append(np.zeros((NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32))
        return np.stack(obs_list, axis=0)

    while global_step < int(cfg.total_steps):
        episode_idx += 1
        curriculum.phase_episode_count += 1
        phase = curriculum.phase

        gm.set_phase(phase)
        env.set_red_opponent(phase)

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

            next_central = build_central_obs(blue_agents)
            with torch.no_grad():
                next_central_tensor = torch.tensor(next_central, dtype=torch.float32).unsqueeze(0)
                v_all = policy_act.forward_central_critic(next_central_tensor).reshape(-1).cpu().numpy()

            for i, (agent, uid, obs_np, central_obs_np, macro_idx, target_idx, logp, val, tid, macro_mask) in enumerate(decisions):
                agent_done = done or (not agent.isEnabled())
                idx = int(getattr(agent, "agent_id", i))
                idx = max(0, min(idx, len(v_all) - 1))
                nv = 0.0 if agent_done else float(v_all[idx])
                reward = float(per_agent_reward.get(uid, 0.0))
                buffer.add(
                    actor_obs=obs_np,
                    central_obs=central_obs_np,
                    macro_action=macro_idx,
                    target_action=target_idx,
                    log_prob=logp,
                    value=val,
                    next_value=nv,
                    reward=reward,
                    done=agent_done,
                    dt=dt,
                    traj_id=tid,
                    macro_mask=macro_mask.detach().cpu().numpy() if torch.is_tensor(macro_mask) else macro_mask,
                )
                global_step += 1

            steps += 1

            if buffer.size() >= int(cfg.update_every):
                mappo_update(policy_train, optimizer, buffer, torch.device("cpu"), cfg)
                policy_act.load_state_dict(policy_train.state_dict())
                policy_act.eval()

        # terminal outcome bonus (team)
        bonus = gm.terminal_outcome_bonus(int(gm.blue_score), int(gm.red_score))
        blue_agents = batch_by_agent_id(env.blue_agents)
        for agent in blue_agents:
            if agent is None:
                continue
            uid = str(getattr(agent, "unique_id", f"{agent.side}_{agent.agent_id}"))
            buffer.add(
                actor_obs=np.asarray(env.build_observation(agent), dtype=np.float32),
                central_obs=build_central_obs(blue_agents),
                macro_action=0,
                target_action=0,
                log_prob=0.0,
                value=0.0,
                next_value=0.0,
                reward=float(bonus),
                done=True,
                dt=float(cfg.decision_window),
                traj_id=traj_id_counter,
                macro_mask=None,
            )
            traj_id_counter += 1

        if int(gm.blue_score) > int(gm.red_score):
            actual = 1.0
        elif int(gm.blue_score) < int(gm.red_score):
            actual = 0.0
        else:
            actual = 0.5
        curriculum.record_result(phase, actual)

        if curriculum.advance_if_ready(learner_rating=0.0, opponent_rating=0.0, win_by=int(gm.blue_score - gm.red_score)):
            print(f"[MAPPO] curriculum advance -> {curriculum.phase}")

        if episode_idx % 50 == 0:
            ckpt = os.path.join(cfg.checkpoint_dir, f"mappo_ckpt_ep{episode_idx}.pth")
            torch.save(policy_train.state_dict(), ckpt)
            print(f"[MAPPO] saved {ckpt}")

    final_path = os.path.join(cfg.checkpoint_dir, "mappo_final.pth")
    torch.save(policy_train.state_dict(), final_path)
    print(f"[MAPPO] Training complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    train_mappo()
