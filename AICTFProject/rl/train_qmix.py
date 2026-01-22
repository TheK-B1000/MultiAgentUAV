from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game_field import GameField, CNN_ROWS, CNN_COLS, NUM_CNN_CHANNELS
from game_field import make_game_field
from macro_actions import MacroAction
from obs_encoder import ObsEncoder
from rl.common import batch_by_agent_id, collect_team_uids, pop_reward_events_best_effort, set_global_seed, simulate_decision_window
from rl.curriculum import CurriculumConfig, CurriculumState
from config import MAP_NAME, MAP_PATH


@dataclass
class QMIXConfig:
    seed: int = 42
    total_steps: int = 1_000_000
    update_every: int = 2048
    batch_size: int = 256
    replay_capacity: int = 200_000
    warmup_steps: int = 10_000
    lr: float = 3e-4
    max_grad_norm: float = 10.0
    gamma: float = 0.995
    decision_window: float = 0.7
    sim_dt: float = 0.1
    max_macro_steps: int = 400
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    target_update_every: int = 2000
    checkpoint_dir: str = "checkpoints"


class AgentQNet(nn.Module):
    def __init__(self, n_actions: int, latent_dim: int = 128) -> None:
        super().__init__()
        self.encoder = ObsEncoder(
            in_channels=NUM_CNN_CHANNELS,
            height=CNN_ROWS,
            width=CNN_COLS,
            latent_dim=int(latent_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(int(latent_dim), 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, int(n_actions)),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs_bchw: torch.Tensor) -> torch.Tensor:
        z = self.encoder(obs_bchw.contiguous())
        return self.head(z)


class QMixer(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 32) -> None:
        super().__init__()
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        self.embed_dim = int(embed_dim)

        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, self.n_agents * self.embed_dim),
        )
        self.hyper_b1 = nn.Linear(self.state_dim, self.embed_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, self.embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        B = agent_qs.size(0)
        N = agent_qs.size(1)
        if N != self.n_agents:
            raise ValueError(f"Expected agent_qs N={self.n_agents}, got {N}")

        w1 = self.hyper_w1(states).view(B, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(B, 1, self.embed_dim)
        w1 = torch.abs(w1)
        hidden = torch.bmm(agent_qs.view(B, 1, self.n_agents), w1) + b1
        hidden = torch.relu(hidden)

        w2 = self.hyper_w2(states).view(B, self.embed_dim, 1)
        w2 = torch.abs(w2)
        b2 = self.hyper_b2(states).view(B, 1, 1)

        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(B, 1)


@dataclass
class Transition:
    obs_agents: np.ndarray
    state: np.ndarray
    actions: np.ndarray
    avail_actions: np.ndarray
    reward: float
    done: bool
    next_obs_agents: np.ndarray
    next_state: np.ndarray
    next_avail_actions: np.ndarray
    dt: float


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.data: List[Transition] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.data)

    def add(self, tr: Transition) -> None:
        if len(self.data) < self.capacity:
            self.data.append(tr)
        else:
            self.data[self.pos] = tr
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.data, k=int(batch_size))


def epsilon_by_step(cfg: QMIXConfig, step: int) -> float:
    if step <= 0:
        return float(cfg.epsilon_start)
    frac = min(1.0, float(step) / float(max(1, cfg.epsilon_decay_steps)))
    return float(cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start))


@torch.no_grad()
def select_actions_qmix(
    agent_net: AgentQNet,
    obs_agents: np.ndarray,
    avail_actions: np.ndarray,
    eps: float,
    device: torch.device,
) -> np.ndarray:
    n_agents = obs_agents.shape[0]
    actions = np.zeros((n_agents,), dtype=np.int64)
    obs_t = torch.tensor(obs_agents, dtype=torch.float32, device=device)
    q = agent_net(obs_t).detach().cpu().numpy()
    for i in range(n_agents):
        mask = avail_actions[i].astype(np.bool_)
        if not mask.any():
            actions[i] = 0
            continue
        if random.random() < float(eps):
            valid = np.flatnonzero(mask)
            actions[i] = int(valid[random.randrange(len(valid))])
        else:
            q_i = q[i].copy()
            q_i[~mask] = -1e9
            actions[i] = int(np.argmax(q_i))
    return actions


def qmix_update(
    agent_net: AgentQNet,
    mixer: QMixer,
    agent_net_tgt: AgentQNet,
    mixer_tgt: QMixer,
    optimizer: optim.Optimizer,
    batch: List[Transition],
    n_agents: int,
    n_actions: int,
    gamma: float,
) -> float:
    B = len(batch)
    obs_agents = np.stack([tr.obs_agents for tr in batch], axis=0)
    next_obs_agents = np.stack([tr.next_obs_agents for tr in batch], axis=0)
    states = np.stack([tr.state for tr in batch], axis=0)
    next_states = np.stack([tr.next_state for tr in batch], axis=0)
    actions = np.stack([tr.actions for tr in batch], axis=0)
    next_avail = np.stack([tr.next_avail_actions for tr in batch], axis=0).astype(np.bool_)
    rewards = np.array([tr.reward for tr in batch], dtype=np.float32)
    dones = np.array([tr.done for tr in batch], dtype=np.float32)

    obs_t = torch.tensor(obs_agents, dtype=torch.float32)
    next_obs_t = torch.tensor(next_obs_agents, dtype=torch.float32)
    s_t = torch.tensor(states, dtype=torch.float32)
    s2_t = torch.tensor(next_states, dtype=torch.float32)
    a_t = torch.tensor(actions, dtype=torch.long)
    r_t = torch.tensor(rewards, dtype=torch.float32).view(B, 1)
    d_t = torch.tensor(dones, dtype=torch.float32).view(B, 1)

    obs_flat = obs_t.view(B * n_agents, NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS)
    q_all = agent_net(obs_flat).view(B, n_agents, n_actions)
    q_chosen = torch.gather(q_all, dim=2, index=a_t.unsqueeze(-1)).squeeze(-1)
    q_tot = mixer(q_chosen, s_t)

    with torch.no_grad():
        next_obs_flat = next_obs_t.view(B * n_agents, NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS)
        q_next_online = agent_net(next_obs_flat).view(B, n_agents, n_actions)
        mask_next = torch.tensor(next_avail, dtype=torch.bool)
        q_next_online = q_next_online.masked_fill(~mask_next, -1e9)
        a_star = q_next_online.argmax(dim=2)

        q_next_tgt = agent_net_tgt(next_obs_flat).view(B, n_agents, n_actions)
        q_next_tgt = q_next_tgt.masked_fill(~mask_next, -1e9)
        q_next_sel = torch.gather(q_next_tgt, dim=2, index=a_star.unsqueeze(-1)).squeeze(-1)

        q_tot_next = mixer_tgt(q_next_sel, s2_t)
        y = r_t + (gamma * (1.0 - d_t) * q_tot_next)

    loss = (q_tot - y).pow(2).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(list(agent_net.parameters()) + list(mixer.parameters()), 10.0)
    optimizer.step()
    return float(loss.item())


def train_qmix(cfg: Optional[QMIXConfig] = None) -> None:
    cfg = cfg or QMIXConfig()
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

    used_macros: List[MacroAction] = list(getattr(env, "macro_order", []))
    if not used_macros:
        raise RuntimeError("env.macro_order missing or empty.")
    n_macros = int(len(used_macros))
    n_targets = int(getattr(env, "num_macro_targets", 8))
    n_actions = int(n_macros * n_targets)
    n_agents = int(getattr(env, "agents_per_team", 2))

    state_dim = int(env.get_global_state_dim())

    agent_net = AgentQNet(n_actions=n_actions)
    mixer = QMixer(n_agents=n_agents, state_dim=state_dim)
    agent_net_tgt = AgentQNet(n_actions=n_actions)
    mixer_tgt = QMixer(n_agents=n_agents, state_dim=state_dim)
    agent_net_tgt.load_state_dict(agent_net.state_dict())
    mixer_tgt.load_state_dict(mixer.state_dict())

    optimizer = optim.Adam(list(agent_net.parameters()) + list(mixer.parameters()), lr=cfg.lr, foreach=False)
    replay = ReplayBuffer(cfg.replay_capacity)

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
    opt_steps = 0

    def collect_team_obs_and_avail(side: str) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
        agents = batch_by_agent_id(env.blue_agents if side == "blue" else env.red_agents)
        obs_list: List[np.ndarray] = []
        avail_list: List[np.ndarray] = []
        for i in range(n_agents):
            if i < len(agents) and agents[i] is not None and agents[i].isEnabled():
                a = agents[i]
                obs = np.asarray(env.build_observation(a), dtype=np.float32)
                mm = np.asarray(env.get_macro_mask(a), dtype=np.bool_).reshape(-1)
                if mm.shape != (n_macros,) or (not mm.any()):
                    mm = np.ones((n_macros,), dtype=np.bool_)
                flat_mask = np.repeat(mm, int(n_targets))
                obs_list.append(obs)
                avail_list.append(flat_mask)
            else:
                obs_list.append(np.zeros((NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32))
                avail_list.append(np.zeros((n_actions,), dtype=np.bool_))
        return np.stack(obs_list, axis=0), np.stack(avail_list, axis=0), agents

    while global_step < int(cfg.total_steps):
        episode_idx += 1
        curriculum.phase_episode_count += 1
        phase = curriculum.phase
        gm.set_phase(phase)
        env.set_red_opponent(phase)

        env.reset_default()
        done = False
        steps = 0
        prev_state = np.asarray(env.get_global_state(), dtype=np.float32).reshape(-1)
        blue_obs, blue_avail, blue_agents_sorted = collect_team_obs_and_avail("blue")

        while (not done) and steps < int(cfg.max_macro_steps) and global_step < int(cfg.total_steps):
            eps = epsilon_by_step(cfg, global_step)
            blue_actions_flat = select_actions_qmix(agent_net, blue_obs, blue_avail, eps, device=torch.device("cpu"))

            submit_actions: Dict[str, Tuple[int, int]] = {}
            for i in range(n_agents):
                if i < len(blue_agents_sorted) and blue_agents_sorted[i] is not None and blue_agents_sorted[i].isEnabled():
                    a = blue_agents_sorted[i]
                    a_flat = int(blue_actions_flat[i])
                    macro_idx = int(a_flat // n_targets)
                    targ_idx = int(a_flat % n_targets)
                    submit_actions[str(getattr(a, "unique_id", f"{a.side}_{a.agent_id}"))] = (macro_idx, targ_idx)

            env.submit_external_actions(submit_actions)
            simulate_decision_window(env, gm, cfg.decision_window, cfg.sim_dt)
            done = bool(getattr(gm, "game_over", False))

            reward = 0.0
            allowed = collect_team_uids([a for a in blue_agents_sorted if a is not None])
            for _t, aid, r in pop_reward_events_best_effort(gm):
                if aid is None:
                    continue
                if str(aid) in allowed:
                    reward += float(r)
            if done:
                reward += float(gm.terminal_outcome_bonus(int(gm.blue_score), int(gm.red_score)))

            next_blue_obs, next_blue_avail, _ = collect_team_obs_and_avail("blue")
            next_state = np.asarray(env.get_global_state(), dtype=np.float32).reshape(-1)

            replay.add(
                Transition(
                    obs_agents=blue_obs,
                    state=prev_state,
                    actions=np.asarray(blue_actions_flat, dtype=np.int64),
                    avail_actions=blue_avail.astype(np.uint8),
                    reward=float(reward),
                    done=bool(done),
                    next_obs_agents=next_blue_obs,
                    next_state=next_state,
                    next_avail_actions=next_blue_avail.astype(np.uint8),
                    dt=float(cfg.decision_window),
                )
            )

            blue_obs = next_blue_obs
            blue_avail = next_blue_avail
            prev_state = next_state
            steps += 1
            global_step += 1

            if len(replay) >= int(cfg.warmup_steps) and (global_step % int(cfg.update_every) == 0):
                batch = replay.sample(cfg.batch_size)
                loss = qmix_update(
                    agent_net=agent_net,
                    mixer=mixer,
                    agent_net_tgt=agent_net_tgt,
                    mixer_tgt=mixer_tgt,
                    optimizer=optimizer,
                    batch=batch,
                    n_agents=n_agents,
                    n_actions=n_actions,
                    gamma=cfg.gamma,
                )
                opt_steps += 1
                if opt_steps % int(cfg.target_update_every) == 0:
                    agent_net_tgt.load_state_dict(agent_net.state_dict())
                    mixer_tgt.load_state_dict(mixer.state_dict())
                if opt_steps % 50 == 0:
                    print(f"[QMIX] step={global_step} loss={loss:.4f} eps={eps:.3f} phase={phase}")

        win = int(gm.blue_score) > int(gm.red_score)
        curriculum.record_result(phase, win)
        if curriculum.advance_if_ready(learner_rating=0.0, opponent_rating=0.0, win_by=int(gm.blue_score - gm.red_score)):
            print(f"[QMIX] curriculum advance -> {curriculum.phase}")

        if episode_idx % 200 == 0:
            ckpt = os.path.join(cfg.checkpoint_dir, f"qmix_step{global_step}.pth")
            torch.save(
                {
                    "agent_net": agent_net.state_dict(),
                    "mixer": mixer.state_dict(),
                    "agent_net_tgt": agent_net_tgt.state_dict(),
                    "mixer_tgt": mixer_tgt.state_dict(),
                    "global_step": global_step,
                    "episode_idx": episode_idx,
                },
                ckpt,
            )
            print(f"[QMIX] saved {ckpt}")

    final_path = os.path.join(cfg.checkpoint_dir, "qmix_final.pth")
    torch.save(
        {
            "agent_net": agent_net.state_dict(),
            "mixer": mixer.state_dict(),
            "agent_net_tgt": agent_net_tgt.state_dict(),
            "mixer_tgt": mixer_tgt.state_dict(),
            "global_step": global_step,
            "episode_idx": episode_idx,
        },
        final_path,
    )
    print(f"[QMIX] Training complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    train_qmix()
