"""Local PPO/MAPPO-style trainer used as the default audit baseline."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl.global_state import GLOBAL_STATE_DIM
from rl.networks import CNNEncoder, CentralizedCritic, orthogonal_init
from rl.ppo_core import TensorDictRolloutBuffer, ppo_policy_loss, ppo_value_loss


def _torch_load_checkpoint(path: str, *, map_location: str | torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def read_custom_ppo_metadata(path: str) -> dict[str, Any]:
    """Read lightweight metadata from a local PPO checkpoint."""
    payload = _torch_load_checkpoint(path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Not a custom PPO checkpoint.")
    cfg = payload.get("cfg") or {}
    meta: dict[str, Any] = {
        "format": payload.get("format", "custom_ppo_v1"),
        "model_path": path,
        "cfg": cfg,
    }
    if isinstance(cfg, dict):
        if "max_blue_agents" in cfg:
            meta["n_blue"] = int(cfg["max_blue_agents"])
        elif "n_agents_per_team" in cfg:
            meta["n_blue"] = int(cfg["n_agents_per_team"])
    return meta


class CustomPPOInferencePolicy:
    """Small inference wrapper with a ``predict`` method for viewer/eval code."""

    def __init__(self, model: SharedActorCentralizedCritic, *, device: str | torch.device = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def _tensor_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {
            "grid": torch.as_tensor(obs["grid"], dtype=torch.float32, device=self.device),
            "vec": torch.as_tensor(obs["vec"], dtype=torch.float32, device=self.device),
            "agent_mask": torch.as_tensor(obs["agent_mask"], dtype=torch.float32, device=self.device),
            "mask": torch.as_tensor(obs["mask"], dtype=torch.float32, device=self.device),
        }

    def _batched_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        batched: Dict[str, np.ndarray] = {}
        for key, value in obs.items():
            arr = np.asarray(value, dtype=np.float32)
            if key == "grid" and arr.ndim == 4:
                arr = arr[None, ...]
            elif key == "vec" and arr.ndim == 2:
                arr = arr[None, ...]
            elif key in {"agent_mask", "mask"} and arr.ndim == 1:
                arr = arr[None, ...]
            batched[key] = arr
        return batched

    def predict(
        self,
        obs: Dict[str, np.ndarray],
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        """Return flattened MultiDiscrete actions for each batch row."""
        obs_t = self._tensor_obs(self._batched_obs(obs))
        with torch.no_grad():
            logits = self.model._mask_logits(self.model.policy_logits(obs_t), obs_t.get("mask"))
            actions = []
            for dist in self.model._categoricals(logits):
                actions.append(torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample())
            action_tensor = torch.stack(actions, dim=1)
        actions_np = action_tensor.detach().cpu().numpy().astype(np.int64)
        if actions_np.shape[0] == 1:
            return actions_np[0], None
        return actions_np, None

    def entropy(self, obs: Dict[str, np.ndarray]) -> float:
        """Mean summed action-head entropy for a batch of observations."""
        obs_t = self._tensor_obs(self._batched_obs(obs))
        with torch.no_grad():
            logits = self.model._mask_logits(self.model.policy_logits(obs_t), obs_t.get("mask"))
            entropy = torch.stack([dist.entropy() for dist in self.model._categoricals(logits)], dim=0).sum(dim=0)
        return float(entropy.mean().detach().cpu().item())


def load_custom_ppo_policy(
    path: str,
    observation_space,
    action_space,
    *,
    device: str | torch.device = "cpu",
) -> CustomPPOInferencePolicy:
    """Load a policy checkpoint produced by :class:`CustomPPOTrainer` for inference."""
    device_t = torch.device(device)
    payload = _torch_load_checkpoint(path, map_location=device_t)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Not a custom PPO checkpoint.")
    model = SharedActorCentralizedCritic(observation_space, action_space).to(device_t)
    model.load_state_dict(payload["model_state_dict"])
    return CustomPPOInferencePolicy(model, device=device_t)


class SharedActorCentralizedCritic(nn.Module):
    """Shared decentralized actor with a centralized global-state critic."""

    def __init__(
        self,
        observation_space,
        action_space,
        *,
        actor_feature_dim: int = 256,
        actor_hidden_dim: int = 256,
        critic_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        grid_shape = tuple(int(v) for v in observation_space.spaces["grid"].shape)
        vec_shape = tuple(int(v) for v in observation_space.spaces["vec"].shape)
        if len(grid_shape) != 4:
            raise ValueError(f"Expected tokenized grid shape (N, C, H, W), got {grid_shape!r}")
        if len(vec_shape) != 2:
            raise ValueError(f"Expected tokenized vec shape (N, V), got {vec_shape!r}")

        self.n_agents = int(grid_shape[0])
        self.vec_dim = int(vec_shape[1])
        self.action_dims = tuple(int(v) for v in getattr(action_space, "nvec", []))
        if len(self.action_dims) % self.n_agents != 0:
            raise ValueError("MultiDiscrete action heads must divide evenly across agents.")
        self.heads_per_agent = len(self.action_dims) // self.n_agents
        self.per_agent_action_dims = self.action_dims[: self.heads_per_agent]
        for idx in range(self.n_agents):
            start = idx * self.heads_per_agent
            end = start + self.heads_per_agent
            if self.action_dims[start:end] != self.per_agent_action_dims:
                raise ValueError("All agents must share the same macro/target action heads.")
        self.per_agent_logits = int(sum(self.per_agent_action_dims))

        self.cnn = CNNEncoder(grid_shape[1:], feature_dim=int(actor_feature_dim))
        self.actor_body = nn.Sequential(
            nn.Linear(int(actor_feature_dim) + self.vec_dim, int(actor_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(actor_hidden_dim), int(actor_hidden_dim)),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(int(actor_hidden_dim), self.per_agent_logits)
        self.critic = CentralizedCritic(global_state_dim=GLOBAL_STATE_DIM, hidden_dim=int(critic_hidden_dim))

        self.actor_body.apply(orthogonal_init)
        orthogonal_init(self.actor_head, gain=0.01)

    def policy_logits(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return flattened MultiDiscrete logits with shape ``(B, sum(action_dims))``."""
        grid = obs["grid"].float()
        vec = obs["vec"].float()
        if grid.dim() != 5:
            raise ValueError(f"grid must have shape (B, N, C, H, W), got {tuple(grid.shape)}")
        if vec.dim() != 3:
            raise ValueError(f"vec must have shape (B, N, V), got {tuple(vec.shape)}")
        batch = int(grid.shape[0])
        grid_flat = grid.reshape(batch * self.n_agents, *grid.shape[2:])
        grid_features = self.cnn(grid_flat).reshape(batch, self.n_agents, -1)

        agent_mask = obs.get("agent_mask")
        if agent_mask is not None:
            if agent_mask.dim() == 1:
                agent_mask = agent_mask.unsqueeze(0)
            mask = agent_mask.float().unsqueeze(-1)
            grid_features = grid_features * mask
            vec = vec * mask

        actor_in = torch.cat([grid_features, vec], dim=-1)
        hidden = self.actor_body(actor_in.reshape(batch * self.n_agents, -1))
        per_agent_logits = self.actor_head(hidden).reshape(batch, self.n_agents, self.per_agent_logits)
        return per_agent_logits.reshape(batch, self.n_agents * self.per_agent_logits)

    def values(self, global_state: torch.Tensor) -> torch.Tensor:
        """Return centralized value estimates with shape ``(B,)``."""
        return self.critic(global_state.float()).squeeze(-1)

    def _mask_logits(self, logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return logits
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        mask = mask.float()
        masked_chunks = []
        offset = 0
        for dim in self.action_dims:
            chunk = logits[:, offset : offset + dim]
            mask_chunk = mask[:, offset : offset + dim]
            if mask_chunk.shape[1] < dim:
                pad = torch.ones((mask.shape[0], dim - mask_chunk.shape[1]), device=mask.device)
                mask_chunk = torch.cat([mask_chunk, pad], dim=1)
            all_invalid = mask_chunk.sum(dim=1, keepdim=True) <= 0.0
            safe_mask = torch.where(all_invalid, torch.ones_like(mask_chunk), mask_chunk)
            masked_chunks.append(chunk.masked_fill(safe_mask <= 0.0, -1e8))
            offset += dim
        return torch.cat(masked_chunks, dim=1)

    def _categoricals(self, logits: torch.Tensor) -> Iterable[Categorical]:
        offset = 0
        for dim in self.action_dims:
            yield Categorical(logits=logits[:, offset : offset + dim])
            offset += dim

    def _log_prob_entropy(self, logits: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions = actions.long()
        log_probs = []
        entropies = []
        for col, dist in enumerate(self._categoricals(logits)):
            action = actions[:, col].clamp(min=0, max=dist.logits.shape[1] - 1)
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
        return torch.stack(log_probs, dim=0).sum(dim=0), torch.stack(entropies, dim=0).sum(dim=0)

    def act(
        self,
        obs: Dict[str, torch.Tensor],
        global_state: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or greedily select actions and return values/log-probs/entropy."""
        logits = self._mask_logits(self.policy_logits(obs), obs.get("mask"))
        actions = []
        for dist in self._categoricals(logits):
            actions.append(torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample())
        action_tensor = torch.stack(actions, dim=1)
        log_prob, entropy = self._log_prob_entropy(logits, action_tensor)
        values = self.values(global_state)
        return action_tensor, values, log_prob, entropy

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        global_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate fixed actions under the current policy."""
        logits = self._mask_logits(self.policy_logits(obs), obs.get("mask"))
        log_prob, entropy = self._log_prob_entropy(logits, actions)
        values = self.values(global_state)
        return values, log_prob, entropy


class CustomPPOTrainer:
    """Small PPO trainer that owns rollout, GAE, and update math locally."""

    def __init__(
        self,
        env,
        cfg,
        *,
        learning_rate: float,
        clip_range: float,
        ent_coef: float,
        n_epochs: int,
        batch_size: int,
        value_clip_range: Optional[float] = None,
    ) -> None:
        self.env = env
        self.cfg = cfg
        self.device = torch.device(str(cfg.device))
        self.model = SharedActorCentralizedCritic(env.observation_space, env.action_space).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(learning_rate), eps=1e-5)
        self.base_learning_rate = float(learning_rate)
        self.clip_range = float(clip_range)
        self.ent_coef = float(ent_coef)
        self.vf_coef = 1.0
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.value_clip_range = float(value_clip_range if value_clip_range is not None else clip_range)
        self.global_step = 0
        self.last_stats: dict[str, float] = {}
        self._ep_wins = 0
        self._ep_losses = 0
        self._ep_draws = 0
        self._episodes_completed = 0

    def _opponent_legend(self, info: dict[str, Any]) -> str:
        """Compact opponent string for logging (scripted:OP3, snapshot:name, ...)."""
        er = info.get("episode_result") or {}
        kind = str(er.get("opponent_kind", info.get("opponent_kind", "scripted")) or "scripted").lower()
        if kind == "scripted":
            tag = str(er.get("scripted_tag") or info.get("opponent_key", getattr(self.cfg, "fixed_opponent_tag", "OP3")))
            return f"SCRIPTED:{str(tag).upper()}"
        if kind == "snapshot":
            snap = str(er.get("opponent_snapshot", "") or info.get("opponent_key", ""))
            return f"SNAPSHOT:{snap}" if snap else "SNAPSHOT:unknown"
        return f"{kind.upper()}:?"

    def _phase_legend(self, info: dict[str, Any]) -> str:
        """Curriculum / scripted-difficulty label (e.g. OP3), not the train-mode name."""
        er = info.get("episode_result") or {}
        return str(
            er.get("phase_name")
            or info.get("phase")
            or getattr(self.cfg, "fixed_opponent_tag", "OP3")
        ).upper()

    def _on_episode_done(self, info: dict[str, Any]) -> None:
        er = info.get("episode_result")
        if isinstance(er, dict):
            bs = int(er.get("blue_score", 0))
            rs = int(er.get("red_score", 0))
        else:
            bs = int(info.get("blue_score", 0))
            rs = int(info.get("red_score", 0))
        if bs > rs:
            self._ep_wins += 1
        elif bs < rs:
            self._ep_losses += 1
        else:
            self._ep_draws += 1
        self._episodes_completed += 1
        every = int(getattr(self.cfg, "episode_log_every", 0) or 0)
        if every > 0 and self._episodes_completed % every == 0:
            self._print_episode_progress(info)

    def _print_episode_progress(self, info: dict[str, Any]) -> None:
        n = self._episodes_completed
        w, l, d = self._ep_wins, self._ep_losses, self._ep_draws
        wr = 100.0 * float(w) / float(max(1, w + l + d))
        mode = str(getattr(self.cfg, "mode", "FIXED_OPPONENT"))
        phase = self._phase_legend(info)
        opp = self._opponent_legend(info)
        print(
            f"[PPO] ep={n} mode={mode} phase={phase} opp={opp} "
            f"W={w} L={l} D={d} WR={wr:.1f}%"
        )

    def _tensor_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {
            "grid": torch.as_tensor(obs["grid"], dtype=torch.float32, device=self.device),
            "vec": torch.as_tensor(obs["vec"], dtype=torch.float32, device=self.device),
            "agent_mask": torch.as_tensor(obs["agent_mask"], dtype=torch.float32, device=self.device),
            "mask": torch.as_tensor(obs["mask"], dtype=torch.float32, device=self.device),
        }

    def _make_buffer(self, obs: Dict[str, np.ndarray]) -> TensorDictRolloutBuffer:
        n_steps = int(self.cfg.n_steps)
        n_envs = int(self.env.num_envs)
        buffer = TensorDictRolloutBuffer(n_steps, n_envs, device=self.device)
        buffer.register_field("obs_grid", tuple(obs["grid"].shape[1:]))
        buffer.register_field("obs_vec", tuple(obs["vec"].shape[1:]))
        buffer.register_field("obs_agent_mask", tuple(obs["agent_mask"].shape[1:]))
        buffer.register_field("obs_mask", tuple(obs["mask"].shape[1:]))
        buffer.register_field("global_state", (GLOBAL_STATE_DIM,))
        buffer.register_field("actions", (len(getattr(self.env.action_space, "nvec", [])),), dtype=torch.long)
        buffer.register_field("log_probs")
        buffer.register_field("values")
        buffer.register_field("next_values")
        buffer.register_field("rewards")
        buffer.register_field("terminated", dtype=torch.bool)
        buffer.register_field("truncated", dtype=torch.bool)
        return buffer

    def _next_values(self, infos: list[dict], next_global_state: np.ndarray) -> torch.Tensor:
        rows = []
        for env_i, info in enumerate(infos):
            if bool(info.get("terminated", False)):
                rows.append(np.zeros((GLOBAL_STATE_DIM,), dtype=np.float32))
            elif bool(info.get("truncated", False)):
                terminal_obs = info.get("terminal_observation") or {}
                rows.append(np.asarray(terminal_obs.get("global_state", next_global_state[env_i]), dtype=np.float32))
            else:
                rows.append(np.asarray(next_global_state[env_i], dtype=np.float32))
        gs = torch.as_tensor(np.stack(rows, axis=0), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.model.values(gs)

    def collect_rollout(self) -> TensorDictRolloutBuffer:
        """Collect one rollout and compute advantages/returns."""
        obs = self.env.reset()
        global_state = self.env.state().astype(np.float32)
        buffer = self._make_buffer(obs)
        for _ in range(int(self.cfg.n_steps)):
            obs_t = self._tensor_obs(obs)
            gs_t = torch.as_tensor(global_state, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                actions_t, values_t, log_probs_t, _ = self.model.act(obs_t, gs_t)
            actions_np = actions_t.detach().cpu().numpy().astype(np.int64)
            self.env.step_async(actions_np)
            next_obs, rewards, dones, infos = self.env.step_wait()
            for done_i, info in zip(dones, infos):
                if bool(done_i):
                    self._on_episode_done(dict(info))
            next_global_state = self.env.state().astype(np.float32)
            next_values_t = self._next_values(infos, next_global_state)
            terminated = np.asarray([bool(info.get("terminated", bool(done))) for info, done in zip(infos, dones)])
            truncated = np.asarray([bool(info.get("truncated", False)) for info in infos])

            buffer.add(
                obs_grid=torch.as_tensor(obs["grid"], dtype=torch.float32, device=self.device),
                obs_vec=torch.as_tensor(obs["vec"], dtype=torch.float32, device=self.device),
                obs_agent_mask=torch.as_tensor(obs["agent_mask"], dtype=torch.float32, device=self.device),
                obs_mask=torch.as_tensor(obs["mask"], dtype=torch.float32, device=self.device),
                global_state=gs_t,
                actions=actions_t,
                log_probs=log_probs_t,
                values=values_t,
                next_values=next_values_t,
                rewards=torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
                terminated=torch.as_tensor(terminated, dtype=torch.bool, device=self.device),
                truncated=torch.as_tensor(truncated, dtype=torch.bool, device=self.device),
            )
            obs = next_obs
            global_state = next_global_state
            self.global_step += int(self.env.num_envs)

        buffer.compute_returns_and_advantages(
            gamma=float(self.cfg.gamma),
            gae_lambda=float(self.cfg.gae_lambda),
        )
        return buffer

    def update(self, buffer: TensorDictRolloutBuffer, *, total_timesteps: int) -> dict[str, float]:
        """Run PPO epochs over one rollout."""
        progress_remaining = max(0.0, 1.0 - float(self.global_step) / max(1.0, float(total_timesteps)))
        lr = self.base_learning_rate * progress_remaining
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        stats: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "grad_norm": [],
        }
        for _ in range(self.n_epochs):
            for batch in buffer.iter_minibatches(self.batch_size, shuffle=True):
                obs_batch = {
                    "grid": batch["obs_grid"],
                    "vec": batch["obs_vec"],
                    "agent_mask": batch["obs_agent_mask"],
                    "mask": batch["obs_mask"],
                }
                values, log_prob, entropy = self.model.evaluate_actions(
                    obs_batch,
                    batch["global_state"],
                    batch["actions"],
                )
                advantages = batch["advantages"]
                if advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
                policy_loss, ppo_stats = ppo_policy_loss(
                    log_prob,
                    batch["log_probs"],
                    advantages,
                    self.clip_range,
                )
                value_loss = ppo_value_loss(values, batch["values"], batch["returns"], self.value_clip_range)
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.max_grad_norm))
                self.optimizer.step()

                stats["policy_loss"].append(float(policy_loss.detach().cpu().item()))
                stats["value_loss"].append(float(value_loss.detach().cpu().item()))
                stats["entropy"].append(float(entropy.mean().detach().cpu().item()))
                stats["approx_kl"].append(float(ppo_stats["approx_kl"].detach().cpu().item()))
                stats["clip_fraction"].append(float(ppo_stats["clip_fraction"].detach().cpu().item()))
                stats["grad_norm"].append(float(torch.as_tensor(grad_norm).detach().cpu().item()))

        self.last_stats = {name: float(np.mean(values)) if values else 0.0 for name, values in stats.items()}
        self.last_stats["learning_rate"] = float(lr)
        return self.last_stats

    def learn(self, total_timesteps: int) -> dict[str, float]:
        """Train until at least ``total_timesteps`` environment transitions have been collected."""
        total = int(total_timesteps)
        while self.global_step < total:
            rollout = self.collect_rollout()
            stats = self.update(rollout, total_timesteps=total)
            if bool(getattr(self.cfg, "verbose_training", False)):
                print(
                    "[PPO|custom] "
                    f"steps={self.global_step} policy_loss={stats['policy_loss']:.4f} "
                    f"value_loss={stats['value_loss']:.4f} approx_kl={stats['approx_kl']:.5f}"
                )
        return self.last_stats

    def save(self, path: str) -> None:
        """Save a torch checkpoint. The project keeps the historical ``.zip`` suffix."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "cfg": asdict(self.cfg),
                "last_stats": self.last_stats,
                "format": "custom_ppo_v1",
            },
            path,
        )

    def load(self, path: str) -> None:
        """Restore a checkpoint produced by :meth:`save`."""
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.global_step = int(payload.get("global_step", 0))
        self.last_stats = dict(payload.get("last_stats", {}))
