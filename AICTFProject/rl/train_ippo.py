"""
IPPO (Independent PPO) baseline: parameter-shared, per-agent rewards and rollouts.

Each agent is a PPO learner using its own trajectory; one shared policy with agent_id
in the observation vector. Rollouts from all agents are concatenated into one PPO update.

Usage:
  python -m rl.ippo_train [--total_timesteps 500000] [--n_steps 2048] ...
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_field import NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS
from game_field import make_game_field
from ctf_sb3_env import CTFGameFieldSB3Env
from rl.common import env_seed, set_global_seed
from rl.marl_env import MARLEnvWrapper, _agent_keys, _actions_dict_to_flat
from config import MAP_NAME, MAP_PATH


# ---------------------------------------------------------------------------
# Config & env
# ---------------------------------------------------------------------------

@dataclass
class IPPOConfig:
    seed: int = 42
    total_timesteps: int = 500_000
    n_steps: int = 2048
    batch_size: int = 512
    n_epochs: int = 10
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    vf_coef: float = 0.5
    device: str = "cpu"

    max_decision_steps: int = 900
    max_blue_agents: int = 2
    add_agent_id_to_vec: bool = True

    checkpoint_dir: str = "checkpoints_ippo"
    run_tag: str = "ippo_baseline"
    save_every_steps: int = 50_000
    log_every_steps: int = 2_000

    default_opponent_kind: str = "SCRIPTED"
    default_opponent_key: str = "OP3"


def make_marl_env(cfg: IPPOConfig, rank: int = 0) -> MARLEnvWrapper:
    """Build CTF env and wrap with MARL wrapper (per-agent obs/rew/done/info)."""
    s = env_seed(cfg.seed, rank)
    np.random.seed(s)
    torch.manual_seed(s)
    inner = CTFGameFieldSB3Env(
        make_game_field_fn=lambda: make_game_field(
            map_name=MAP_NAME or None,
            map_path=MAP_PATH or None,
        ),
        max_decision_steps=cfg.max_decision_steps,
        enforce_masks=True,
        seed=s,
        include_mask_in_obs=True,
        default_opponent_kind=cfg.default_opponent_kind,
        default_opponent_key=cfg.default_opponent_key,
        ppo_gamma=cfg.gamma,
        max_blue_agents=cfg.max_blue_agents,
        use_obs_builder=True,
    )
    env = MARLEnvWrapper(inner, add_agent_id_to_vec=cfg.add_agent_id_to_vec)
    env.reset(seed=s)
    return env


# ---------------------------------------------------------------------------
# Single-agent observation space (for shared policy)
# ---------------------------------------------------------------------------

def get_single_agent_obs_space(
    vec_dim: int,
    n_macros: int = 5,
    n_targets: int = 8,
) -> Dict[str, Any]:
    """Observation space for one agent: grid (1,C,H,W), vec (1,V), mask (n_macros+n_targets)."""
    from gymnasium import spaces
    C, H, W = NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS
    return spaces.Dict({
        "grid": spaces.Box(low=0.0, high=1.0, shape=(1, C, H, W), dtype=np.float32),
        "vec": spaces.Box(low=-2.0, high=2.0, shape=(1, vec_dim), dtype=np.float32),
        "mask": spaces.Box(
            low=0.0, high=1.0,
            shape=(n_macros + n_targets,),
            dtype=np.float32,
        ),
    })


def get_single_agent_action_space(n_macros: int = 5, n_targets: int = 8) -> Any:
    """MultiDiscrete [n_macros, n_targets] for one agent."""
    from gymnasium import spaces
    return spaces.MultiDiscrete([n_macros, n_targets])


# ---------------------------------------------------------------------------
# Policy: single-agent feature extractor + actor-critic
# ---------------------------------------------------------------------------

class SingleAgentExtractor(nn.Module):
    """CNN for grid (1,C,H,W) + vec (1,V) -> features. Tuned for 20x20 grid (CNN_ROWS/COLS)."""

    def __init__(self, cnn_in_channels: int, cnn_hw: int, vec_dim: int, cnn_output_dim: int = 256):
        super().__init__()
        self._vec_dim = vec_dim
        # 20x20 -> stride 2 -> 7x7 -> stride 2 -> 2x2 -> stride 1 -> 1x1
        self.conv = nn.Sequential(
            nn.Conv2d(cnn_in_channels, 32, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            x = torch.zeros(1, cnn_in_channels, cnn_hw, cnn_hw)
            cnn_flatten_size = self.conv(x).shape[1]
        self.fc_cnn = nn.Linear(cnn_flatten_size, cnn_output_dim)
        self.features_dim = cnn_output_dim + vec_dim

    def forward(self, grid: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        # grid (B, 1, C, H, W) -> (B, C, H, W)
        if grid.dim() == 5:
            grid = grid.squeeze(1)
        cnn_out = self.conv(grid)
        cnn_out = F.relu(self.fc_cnn(cnn_out))
        if vec.dim() == 3:
            vec = vec.squeeze(1)
        return torch.cat([cnn_out, vec], dim=1)


class IPPOPolicy(nn.Module):
    """Shared policy: single-agent obs -> action logits (masked) + value."""

    def __init__(
        self,
        obs_space: Dict[str, Any],
        action_nvec: List[int],
        hidden_dim: int = 256,
        cnn_output_dim: int = 256,
    ):
        super().__init__()
        self.action_nvec = action_nvec
        n_macros, n_targets = action_nvec[0], action_nvec[1]
        grid_shape = obs_space["grid"].shape
        vec_shape = obs_space["vec"].shape
        C, H, W = grid_shape[1], grid_shape[2], grid_shape[3]
        V = vec_shape[-1]
        self.extractor = SingleAgentExtractor(C, H, V, cnn_output_dim)
        feat_dim = self.extractor.features_dim
        self.mlp_pi = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mlp_vf = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_net = nn.Linear(hidden_dim, n_macros + n_targets)
        self.value_net = nn.Linear(hidden_dim, 1)
        self._n_macros = n_macros
        self._n_targets = n_targets

    def forward_features(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        grid = obs["grid"]
        vec = obs["vec"]
        return self.extractor(grid, vec)

    def get_value(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = self.forward_features(obs)
        return self.value_net(self.mlp_vf(feats)).squeeze(-1)

    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.forward_features(obs)
        latent_pi = self.mlp_pi(feats)
        logits = self.action_net(latent_pi)
        if "mask" in obs and obs["mask"] is not None:
            logits = self._apply_mask(logits, obs["mask"])
        dist = self._get_distribution(logits)
        if action is None:
            action = dist.sample() if not deterministic else dist.mode()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value_net(self.mlp_vf(feats)).squeeze(-1)
        return action, log_prob, entropy, value

    def _apply_mask(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        mask = mask.float()
        n_macros, n_targets = self._n_macros, self._n_targets
        # logits: (B, n_macros + n_targets)
        if mask.shape[1] >= n_macros + n_targets:
            m = mask[:, : n_macros + n_targets]
        else:
            m = F.pad(mask, (0, n_macros + n_targets - mask.shape[1]), value=1.0)
        return logits.masked_fill(m <= 0.0, -1e8)

    def _get_distribution(self, logits: torch.Tensor) -> "MultiCategorical":
        return MultiCategorical(logits, self.action_nvec)


class MultiCategorical:
    """MultiDiscrete distribution: logits split into [macro_logits, target_logits]."""

    def __init__(self, logits: torch.Tensor, nvec: List[int]):
        self.nvec = nvec
        n_macros, n_targets = nvec[0], nvec[1]
        self.logits_macro = logits[:, :n_macros]
        self.logits_target = logits[:, n_macros : n_macros + n_targets]
        self.dist_macro = torch.distributions.Categorical(logits=self.logits_macro)
        self.dist_target = torch.distributions.Categorical(logits=self.logits_target)

    def sample(self) -> torch.Tensor:
        a0 = self.dist_macro.sample()
        a1 = self.dist_target.sample()
        return torch.stack([a0, a1], dim=1)

    def mode(self) -> torch.Tensor:
        a0 = self.logits_macro.argmax(dim=1)
        a1 = self.logits_target.argmax(dim=1)
        return torch.stack([a0, a1], dim=1)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.dist_macro.log_prob(action[:, 0]) + self.dist_target.log_prob(action[:, 1])

    def entropy(self) -> torch.Tensor:
        return self.dist_macro.entropy() + self.dist_target.entropy()


# ---------------------------------------------------------------------------
# Rollout buffer (per-agent transitions, flattened for one update)
# ---------------------------------------------------------------------------

def dict_obs_to_torch(obs_dict: Dict[str, Dict[str, np.ndarray]], agent_keys: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
    """Stack per-agent obs into batch tensors; add leading dim for (1,C,H,W) and (1,V)."""
    batch: Dict[str, List[np.ndarray]] = {}
    for k in agent_keys:
        o = obs_dict[k]
        for key, arr in o.items():
            if key not in batch:
                batch[key] = []
            batch[key].append(arr)
    out = {}
    for key, arrs in batch.items():
        stacked = np.stack(arrs, axis=0)
        if key == "grid" and stacked.ndim == 3:
            stacked = np.expand_dims(stacked, axis=1)
        if key == "vec" and stacked.ndim == 1:
            stacked = np.expand_dims(stacked, axis=0)
        if key == "vec" and stacked.ndim == 2:
            stacked = np.expand_dims(stacked, axis=1)
        out[key] = torch.tensor(stacked, dtype=torch.float32, device=device)
    return out


def append_rollout(
    buffers: Dict[str, List],
    obs_dict: Dict[str, Dict[str, np.ndarray]],
    actions_list: List[np.ndarray],
    rewards_list: List[float],
    dones: bool,
    log_probs: torch.Tensor,
    values: torch.Tensor,
    agent_keys: List[str],
) -> None:
    """Append one step of per-agent data to buffers (lists of arrays/tensors)."""
    for i, k in enumerate(agent_keys):
        o = obs_dict[k]
        for key, arr in o.items():
            key_list = f"obs_{key}"
            if key_list not in buffers:
                buffers[key_list] = []
            buf = np.asarray(arr, dtype=np.float32)
            if buf.ndim == 0:
                buf = buf.reshape(1)
            buffers[key_list].append(buf)
        buffers["actions"].append(actions_list[i])
        buffers["rewards"].append(rewards_list[i])
        buffers["dones"].append(dones)
        buffers["log_probs"].append(log_probs[i].item() if log_probs.dim() > 0 else log_probs.item())
        buffers["values"].append(values[i].item() if values.dim() > 0 else values.item())


def rollout_buffers_to_tensors(
    buffers: Dict[str, List],
    agent_keys: List[str],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert list buffers to batched tensors: (n_steps * n_agents, ...)."""
    obs_batch: Dict[str, torch.Tensor] = {}
    for key in list(buffers.keys()):
        if not key.startswith("obs_"):
            continue
        arrs = buffers[key]
        stacked = np.stack(arrs, axis=0)
        if "grid" in key and stacked.ndim == 4:
            stacked = np.expand_dims(stacked, axis=1)
        if "vec" in key and stacked.ndim == 2:
            stacked = np.expand_dims(stacked, axis=1)
        obs_batch[key.replace("obs_", "")] = torch.tensor(stacked, dtype=torch.float32, device=device)
    actions = np.array(buffers["actions"], dtype=np.int64)
    actions = torch.tensor(actions, device=device)
    rewards = torch.tensor(buffers["rewards"], dtype=torch.float32, device=device)
    dones = torch.tensor(buffers["dones"], dtype=torch.float32, device=device)
    log_probs_old = torch.tensor(buffers["log_probs"], dtype=torch.float32, device=device)
    values_old = torch.tensor(buffers["values"], dtype=torch.float32, device=device)
    return obs_batch, actions, rewards, dones, log_probs_old, values_old


# ---------------------------------------------------------------------------
# GAE and PPO update
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    n_steps: int,
    n_agents: int,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
    last_values: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute advantages and returns. rewards/values/dones shape (n_steps * n_agents,). last_values (n_agents,) for bootstrap."""
    N = n_steps * n_agents
    advantages = torch.zeros(N, device=device)
    last_gae = 0.0
    for t in reversed(range(n_steps)):
        start = t * n_agents
        end = start + n_agents
        r = rewards[start:end]
        v = values[start:end]
        d = dones[start:end]
        if t == n_steps - 1:
            next_v = last_values if last_values is not None else torch.zeros_like(v)
            if next_v.dim() == 0:
                next_v = next_v.unsqueeze(0).expand(n_agents)
        else:
            next_v = values[end : end + n_agents]
        delta = r + gamma * next_v * (1 - d) - v
        last_gae = delta + gamma * gae_lambda * (1 - d) * last_gae
        advantages[start:end] = last_gae
    returns = advantages + values
    return advantages, returns


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


def ppo_update(
    policy: IPPOPolicy,
    obs_batch: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    log_probs_old: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    cfg: IPPOConfig,
    optimizer: torch.optim.Optimizer,
    n_steps: int,
    n_agents: int,
) -> Dict[str, float]:
    N = n_steps * n_agents
    indices = np.random.permutation(N)
    total_pi_loss = 0.0
    total_vf_loss = 0.0
    total_ent = 0.0
    n_batches = 0
    for start in range(0, N, cfg.batch_size):
        end = min(start + cfg.batch_size, N)
        mb_indices = indices[start:end]
        mb_obs = {k: v[mb_indices] for k, v in obs_batch.items()}
        mb_actions = actions[mb_indices]
        mb_log_old = log_probs_old[mb_indices]
        mb_returns = returns[mb_indices]
        mb_adv = advantages[mb_indices]
        mb_adv = normalize_advantages(mb_adv)

        _, log_prob_new, entropy, value = policy.get_action_and_value(mb_obs, mb_actions)
        ratio = torch.exp(log_prob_new - mb_log_old)
        surr1 = ratio * mb_adv
        surr2 = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * mb_adv
        pi_loss = -torch.min(surr1, surr2).mean()
        vf_loss = F.mse_loss(value, mb_returns)
        ent_loss = -entropy.mean()
        loss = pi_loss + cfg.vf_coef * vf_loss + cfg.ent_coef * ent_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
        optimizer.step()
        total_pi_loss += pi_loss.item()
        total_vf_loss += vf_loss.item()
        total_ent += entropy.mean().item()
        n_batches += 1
    return {
        "pi_loss": total_pi_loss / max(1, n_batches),
        "vf_loss": total_vf_loss / max(1, n_batches),
        "entropy": total_ent / max(1, n_batches),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_ippo(cfg: Optional[IPPOConfig] = None) -> None:
    cfg = cfg or IPPOConfig()
    set_global_seed(cfg.seed)
    device = torch.device(cfg.device)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    env = make_marl_env(cfg)
    agent_keys = env.agent_keys
    n_agents = len(agent_keys)
    vec_dim = env._vec_per_agent + (1 if cfg.add_agent_id_to_vec else 0)
    n_macros = env._n_macros
    n_targets = env._n_targets

    obs_space = get_single_agent_obs_space(vec_dim, n_macros, n_targets)
    action_nvec = [n_macros, n_targets]
    policy = IPPOPolicy(obs_space, action_nvec).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)

    global_step = 0
    episode_count = 0
    per_agent_ep_rewards: Dict[str, List[float]] = {k: [] for k in agent_keys}

    while global_step < cfg.total_timesteps:
        obs_dict, _ = env.reset()
        buffers: Dict[str, List] = { "actions": [], "rewards": [], "dones": [], "log_probs": [], "values": [] }
        for key in list(obs_dict[agent_keys[0]].keys()):
            buffers[f"obs_{key}"] = []

        for step in range(cfg.n_steps):
            obs_torch = dict_obs_to_torch(obs_dict, agent_keys, device)
            with torch.no_grad():
                actions_t, log_prob, _, values_t = policy.get_action_and_value(obs_torch, deterministic=False)
            actions_np = actions_t.cpu().numpy()
            actions_list = [actions_np[i] for i in range(n_agents)]
            action_flat = _actions_dict_to_flat(
                {k: (int(actions_list[i][0]), int(actions_list[i][1])) for i, k in enumerate(agent_keys)},
                agent_keys,
            )
            next_obs_dict, rews_dict, dones_dict, infos_dict, info = env.step(
                {k: (int(actions_list[i][0]), int(actions_list[i][1])) for i, k in enumerate(agent_keys)}
            )
            done = dones_dict[agent_keys[0]]
            rews_list = [rews_dict[k] for k in agent_keys]
            append_rollout(buffers, obs_dict, actions_list, rews_list, done, log_prob, values_t, agent_keys)
            obs_dict = next_obs_dict
            global_step += n_agents

            if done:
                episode_count += 1
                for k in agent_keys:
                    per_agent_ep_rewards[k].append(infos_dict[k].get("reward", 0.0))
                obs_dict, _ = env.reset()

        obs_batch, actions, rewards, dones, log_probs_old, values_old = rollout_buffers_to_tensors(
            buffers, agent_keys, device
        )
        with torch.no_grad():
            last_values = policy.get_value(dict_obs_to_torch(obs_dict, agent_keys, device))
        advantages, returns = compute_gae(
            rewards, values_old, dones, cfg.n_steps, n_agents, cfg.gamma, cfg.gae_lambda, device, last_values=last_values
        )
        update_stats = ppo_update(
            policy, obs_batch, actions, log_probs_old, returns, advantages, cfg, optimizer, cfg.n_steps, n_agents
        )

        if global_step % cfg.log_every_steps < cfg.n_steps * n_agents or global_step >= cfg.total_timesteps:
            mean_rew = rewards.cpu().numpy().mean()
            log_str = (
                f"step={global_step} ep={episode_count} "
                f"pi_loss={update_stats['pi_loss']:.4f} vf_loss={update_stats['vf_loss']:.4f} ent={update_stats['entropy']:.4f} "
                f"mean_rew={mean_rew:.4f}"
            )
            for k in agent_keys:
                if per_agent_ep_rewards[k]:
                    r_mean = np.mean(per_agent_ep_rewards[k][-100:])
                    log_str += f" {k}_rew={r_mean:.3f}"
            print(log_str)

        if cfg.save_every_steps and global_step > 0 and global_step % cfg.save_every_steps < cfg.n_steps * n_agents:
            path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_step{global_step}.pt")
            torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict()}, path)

    env.close()
    print(f"[IPPO] Done. total_steps={global_step} episodes={episode_count}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--total_timesteps", type=int, default=500_000)
    p.add_argument("--n_steps", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--run_tag", type=str, default="ippo_baseline")
    p.add_argument("--log_every_steps", type=int, default=2000)
    args = p.parse_args()
    cfg = IPPOConfig(
        total_timesteps=args.total_timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        run_tag=args.run_tag,
        log_every_steps=args.log_every_steps,
    )
    run_ippo(cfg)
