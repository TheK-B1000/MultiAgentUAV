"""Local PPO/MAPPO-style trainer with optional latent team strategy."""

from __future__ import annotations

import csv
import os
from dataclasses import asdict
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from rl.global_state import GLOBAL_STATE_DIM
from rl.latent_marl import StrategyEncoder, expected_strategy_switch_penalty
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
        meta["use_latent_strategy"] = bool(cfg.get("use_latent_strategy", False))
        if "latent_k" in cfg:
            meta["latent_k"] = int(cfg["latent_k"])
    return meta


class CustomPPOInferencePolicy:
    """Small inference wrapper with a ``predict`` method for viewer/eval code."""

    def __init__(
        self,
        model: SharedActorCentralizedCritic,
        *,
        device: str | torch.device = "cpu",
        cfg: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self._prev_z: Optional[torch.Tensor] = None
        cfg = cfg or {}
        self.strategy_interval = max(0, int(cfg.get("latent_resample_every_n", 0) or 0))
        self._strategy_age = 0
        self._last_strategy_z: Optional[torch.Tensor] = None
        self._last_strategy_probs: Optional[torch.Tensor] = None
        self._last_strategy_entropy: Optional[torch.Tensor] = None
        self._last_strategy_resampled = False

    def reset_strategy(self) -> None:
        """Forget the persisted inference strategy, typically at episode reset."""
        self._prev_z = None
        self._strategy_age = 0
        self._last_strategy_z = None
        self._last_strategy_probs = None
        self._last_strategy_entropy = None
        self._last_strategy_resampled = False

    def _tensor_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {
            "grid": torch.as_tensor(obs["grid"], dtype=torch.float32, device=self.device),
            "vec": torch.as_tensor(obs["vec"], dtype=torch.float32, device=self.device),
            "agent_mask": torch.as_tensor(obs["agent_mask"], dtype=torch.float32, device=self.device),
            "mask": torch.as_tensor(obs["mask"], dtype=torch.float32, device=self.device),
        }

    def _global_state_tensor(self, obs: Dict[str, np.ndarray], batch: int) -> torch.Tensor:
        raw = obs.get("global_state")
        if raw is None:
            return torch.zeros((batch, GLOBAL_STATE_DIM), dtype=torch.float32, device=self.device)
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, ...]
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

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
        batched = self._batched_obs(obs)
        obs_t = self._tensor_obs(batched)
        with torch.no_grad():
            if self.model.uses_latent_strategy:
                batch = int(obs_t["grid"].shape[0])
                global_state = self._global_state_tensor(batched, batch)
                z_logits = self.model.strategy_logits(global_state)
                z_dist = Categorical(logits=z_logits)
                needs_strategy = (
                    self._prev_z is None
                    or int(self._prev_z.numel()) != batch
                    or (self.strategy_interval > 0 and self._strategy_age >= self.strategy_interval)
                )
                if needs_strategy:
                    z_idx = torch.argmax(z_logits, dim=-1) if deterministic else z_dist.sample()
                    self._prev_z = z_idx.detach()
                    self._strategy_age = 0
                else:
                    z_idx = self._prev_z.to(self.device)
                self._last_strategy_z = z_idx.detach().cpu()
                self._last_strategy_probs = torch.softmax(z_logits, dim=-1).detach().cpu()
                self._last_strategy_entropy = z_dist.entropy().detach().cpu()
                self._last_strategy_resampled = bool(needs_strategy)
                action_tensor, _, _, _ = self.model.act(
                    obs_t,
                    global_state,
                    deterministic=deterministic,
                    z_idx=z_idx,
                )
                self._strategy_age += 1
            else:
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
        batched = self._batched_obs(obs)
        obs_t = self._tensor_obs(batched)
        with torch.no_grad():
            z_idx = None
            z_entropy = torch.zeros((obs_t["grid"].shape[0],), device=self.device)
            if self.model.uses_latent_strategy:
                global_state = self._global_state_tensor(batched, int(obs_t["grid"].shape[0]))
                z_idx, _, z_entropy, _ = self.model.sample_strategy(global_state, deterministic=True)
            logits = self.model._mask_logits(self.model.policy_logits(obs_t, z_idx=z_idx), obs_t.get("mask"))
            entropy = torch.stack([dist.entropy() for dist in self.model._categoricals(logits)], dim=0).sum(dim=0)
        return float((entropy + z_entropy).mean().detach().cpu().item())

    def strategy_info(self) -> dict[str, Any]:
        """Return the most recent latent strategy diagnostics for single-env evaluation."""
        if not self.model.uses_latent_strategy or self._last_strategy_z is None:
            return {}
        z = self._last_strategy_z.reshape(-1)
        probs = self._last_strategy_probs
        entropy = self._last_strategy_entropy
        out: dict[str, Any] = {
            "strategy": int(z[0].item()),
            "strategy_batch": [int(v) for v in z.tolist()],
            "strategy_resampled": bool(self._last_strategy_resampled),
        }
        if probs is not None and probs.numel() > 0:
            p0 = probs.reshape(probs.shape[0], -1)[0]
            out["strategy_k"] = int(p0.numel())
            for idx, prob in enumerate(p0.tolist()):
                out[f"strategy_prob_{idx}"] = float(prob)
        if entropy is not None and entropy.numel() > 0:
            out["strategy_entropy"] = float(entropy.reshape(-1)[0].item())
        return out


def _model_kwargs_from_cfg(cfg: Any) -> dict[str, int]:
    if not isinstance(cfg, dict) or not bool(cfg.get("use_latent_strategy", False)):
        return {}
    return {
        "latent_k": int(cfg.get("latent_k", 4)),
        "z_embed_dim": int(cfg.get("latent_z_embed_dim", 16)),
        "strategy_hidden_dim": int(cfg.get("latent_strategy_hidden", 128)),
        "critic_hidden_dim": int(cfg.get("latent_vf_hidden", 128)),
    }


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
    model = SharedActorCentralizedCritic(
        observation_space,
        action_space,
        **_model_kwargs_from_cfg(payload.get("cfg") or {}),
    ).to(device_t)
    model.load_state_dict(payload["model_state_dict"])
    return CustomPPOInferencePolicy(model, device=device_t, cfg=payload.get("cfg") or {})


class SharedActorCentralizedCritic(nn.Module):
    """Shared decentralized actor with an optional latent team strategy."""

    def __init__(
        self,
        observation_space,
        action_space,
        *,
        actor_feature_dim: int = 256,
        actor_hidden_dim: int = 256,
        critic_hidden_dim: int = 128,
        latent_k: int = 0,
        z_embed_dim: int = 16,
        strategy_hidden_dim: int = 128,
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
        self.joint_action_onehot_dim = int(sum(self.action_dims))
        self.latent_k = max(0, int(latent_k))
        self.uses_latent_strategy = self.latent_k > 0
        self.z_embed_dim = int(z_embed_dim) if self.uses_latent_strategy else 0

        self.cnn = CNNEncoder(grid_shape[1:], feature_dim=int(actor_feature_dim))
        if self.uses_latent_strategy:
            self.strategy_encoder = StrategyEncoder(
                state_dim=GLOBAL_STATE_DIM,
                latent_k=self.latent_k,
                hidden=int(strategy_hidden_dim),
            )
            self.strategy_embedding = nn.Embedding(self.latent_k, self.z_embed_dim)
            nn.init.orthogonal_(self.strategy_embedding.weight, gain=1.0)
        else:
            self.strategy_encoder = None
            self.strategy_embedding = None

        self.actor_body = nn.Sequential(
            nn.Linear(int(actor_feature_dim) + self.vec_dim + self.z_embed_dim, int(actor_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(actor_hidden_dim), int(actor_hidden_dim)),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(int(actor_hidden_dim), self.per_agent_logits)
        critic_extra_dim = self.joint_action_onehot_dim + self.latent_k if self.uses_latent_strategy else 0
        self.critic = CentralizedCritic(
            global_state_dim=GLOBAL_STATE_DIM,
            hidden_dim=int(critic_hidden_dim),
            extra_dim=critic_extra_dim,
        )

        self.actor_body.apply(orthogonal_init)
        orthogonal_init(self.actor_head, gain=0.01)

    def strategy_logits(self, global_state: torch.Tensor) -> torch.Tensor:
        """Return ``q_phi(z | s)`` logits for latent strategy mode."""
        if not self.uses_latent_strategy or self.strategy_encoder is None:
            raise RuntimeError("strategy_logits is only available when latent strategy is enabled.")
        return self.strategy_encoder(global_state.float())

    def sample_strategy(
        self,
        global_state: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or greedily choose team strategy indices from ``q_phi(z | s)``."""
        logits = self.strategy_logits(global_state)
        dist = Categorical(logits=logits)
        z_idx = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return z_idx.long(), dist.log_prob(z_idx), dist.entropy(), logits

    def policy_logits(self, obs: Dict[str, torch.Tensor], z_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
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

        actor_inputs = [grid_features, vec]
        if self.uses_latent_strategy:
            if z_idx is None:
                raise ValueError("z_idx is required when latent strategy is enabled.")
            z = z_idx.long().reshape(-1).clamp(min=0, max=self.latent_k - 1)
            if z.shape[0] != batch:
                raise ValueError(f"z_idx must have shape ({batch},), got {tuple(z_idx.shape)}")
            assert self.strategy_embedding is not None
            z_emb = self.strategy_embedding(z).unsqueeze(1).expand(batch, self.n_agents, self.z_embed_dim)
            actor_inputs.append(z_emb)

        actor_in = torch.cat(actor_inputs, dim=-1)
        hidden = self.actor_body(actor_in.reshape(batch * self.n_agents, -1))
        per_agent_logits = self.actor_head(hidden).reshape(batch, self.n_agents, self.per_agent_logits)
        return per_agent_logits.reshape(batch, self.n_agents * self.per_agent_logits)

    def _joint_action_one_hot(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.long()
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        chunks = []
        for col, dim in enumerate(self.action_dims):
            action = actions[:, col].clamp(min=0, max=dim - 1)
            chunks.append(F.one_hot(action, num_classes=dim).float())
        return torch.cat(chunks, dim=-1)

    def _critic_extra(self, actions: Optional[torch.Tensor], z_idx: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.uses_latent_strategy:
            return None
        if actions is None or z_idx is None:
            raise ValueError("actions and z_idx are required by the latent action-conditioned critic.")
        z = z_idx.long().reshape(-1).clamp(min=0, max=self.latent_k - 1)
        z_one_hot = F.one_hot(z, num_classes=self.latent_k).float()
        return torch.cat([self._joint_action_one_hot(actions).to(z_one_hot.device), z_one_hot], dim=-1)

    def values(
        self,
        global_state: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        z_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return centralized value estimates with shape ``(B,)``."""
        return self.critic(global_state.float(), extra=self._critic_extra(actions, z_idx)).squeeze(-1)

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
        z_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or greedily select actions and return values/log-probs/entropy."""
        if self.uses_latent_strategy and z_idx is None:
            z_idx, _, _, _ = self.sample_strategy(global_state, deterministic=deterministic)
        logits = self._mask_logits(self.policy_logits(obs, z_idx=z_idx), obs.get("mask"))
        actions = []
        for dist in self._categoricals(logits):
            actions.append(torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample())
        action_tensor = torch.stack(actions, dim=1)
        log_prob, entropy = self._log_prob_entropy(logits, action_tensor)
        values = self.values(global_state, actions=action_tensor, z_idx=z_idx)
        return action_tensor, values, log_prob, entropy

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        global_state: torch.Tensor,
        actions: torch.Tensor,
        *,
        z_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate fixed actions under the current policy."""
        logits = self._mask_logits(self.policy_logits(obs, z_idx=z_idx), obs.get("mask"))
        log_prob, entropy = self._log_prob_entropy(logits, actions)
        values = self.values(global_state, actions=actions, z_idx=z_idx)
        aux: dict[str, torch.Tensor] = {}
        if self.uses_latent_strategy:
            if z_idx is None:
                raise ValueError("z_idx is required when latent strategy is enabled.")
            z_logits = self.strategy_logits(global_state)
            z_dist = Categorical(logits=z_logits)
            z = z_idx.long().reshape(-1).clamp(min=0, max=self.latent_k - 1)
            aux["strategy_logits"] = z_logits
            aux["strategy_log_prob"] = z_dist.log_prob(z)
            aux["strategy_entropy"] = z_dist.entropy()
        return values, log_prob, entropy, aux


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
        self.use_latent_strategy = bool(getattr(cfg, "use_latent_strategy", False))
        self.latent_k = int(getattr(cfg, "latent_k", 4)) if self.use_latent_strategy else 0
        self.latent_resample_every_n = max(0, int(getattr(cfg, "latent_resample_every_n", 0) or 0))
        model_kwargs: dict[str, int] = {}
        if self.use_latent_strategy:
            model_kwargs = {
                "latent_k": self.latent_k,
                "z_embed_dim": int(getattr(cfg, "latent_z_embed_dim", 16)),
                "strategy_hidden_dim": int(getattr(cfg, "latent_strategy_hidden", 128)),
                "critic_hidden_dim": int(getattr(cfg, "latent_vf_hidden", 128)),
            }
        self.model = SharedActorCentralizedCritic(env.observation_space, env.action_space, **model_kwargs).to(self.device)
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
        self._updates_completed = 0
        self._ep_wins = 0
        self._ep_losses = 0
        self._ep_draws = 0
        self._episodes_completed = 0
        self.metrics_csv_path = str(getattr(cfg, "metrics_csv_path", "") or "")
        self.episode_csv_path = str(getattr(cfg, "episode_csv_path", "") or "")
        self._last_obs: Optional[Dict[str, np.ndarray]] = None
        self._last_global_state: Optional[np.ndarray] = None
        self._current_z: Optional[torch.Tensor] = None
        self._strategy_age = torch.zeros((int(env.num_envs),), dtype=torch.long, device=self.device)
        self._needs_strategy_sample = torch.ones((int(env.num_envs),), dtype=torch.bool, device=self.device)

    def _write_csv_row(self, path: str, fieldnames: list[str], row: dict[str, Any]) -> None:
        """Append one row with a stable header; used for long-run audit telemetry."""
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        exists = os.path.isfile(path) and os.path.getsize(path) > 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    def _episode_fieldnames(self) -> list[str]:
        return [
            "episode_id",
            "timesteps",
            "mode",
            "map_set",
            "phase_name",
            "opponent",
            "success",
            "blue_score",
            "red_score",
            "win_margin",
            "decision_steps",
            "zone_coverage",
            "collision_free_episode",
            "time_to_first_score",
            "mean_inter_robot_dist",
        ]

    def _update_fieldnames(self) -> list[str]:
        fields = [
            "update",
            "timesteps",
            "episodes_completed",
            "wins",
            "losses",
            "draws",
            "win_rate",
            "rollout_reward_mean",
            "rollout_reward_std",
            "rollout_return_mean",
            "rollout_return_std",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "grad_norm",
            "learning_rate",
            "strategy_entropy",
            "strategy_persist_loss",
            "strategy_resample_fraction",
            "strategy_unique_count",
            "strategy_dominant",
            "strategy_switch_count",
            "strategy_switch_fraction",
            "strategy_resample_fraction_rollout",
        ]
        if self.use_latent_strategy:
            fields.extend(f"strategy_occupancy_{idx}" for idx in range(self.latent_k))
        return fields

    def _write_episode_metrics(self, info: dict[str, Any], *, blue_score: int, red_score: int, timestep: int) -> None:
        if not self.episode_csv_path:
            return
        er = info.get("episode_result") if isinstance(info.get("episode_result"), dict) else {}
        row = {
            "episode_id": self._episodes_completed,
            "timesteps": int(timestep),
            "mode": str(getattr(self.cfg, "mode", "FIXED_OPPONENT")),
            "map_set": str(info.get("map_set", getattr(self.cfg, "map_set", "train"))).lower(),
            "phase_name": self._phase_legend(info),
            "opponent": self._opponent_legend(info),
            "success": 1 if blue_score > red_score else 0,
            "blue_score": int(blue_score),
            "red_score": int(red_score),
            "win_margin": int(blue_score) - int(red_score),
            "decision_steps": int(er.get("decision_steps", info.get("decision_steps", 0)) or 0),
            "zone_coverage": float(er.get("zone_coverage", 0.0) or 0.0),
            "collision_free_episode": int(er.get("collision_free_episode", 1) or 0),
            "time_to_first_score": er.get("time_to_first_score", ""),
            "mean_inter_robot_dist": er.get("mean_inter_robot_dist", ""),
        }
        self._write_csv_row(self.episode_csv_path, self._episode_fieldnames(), row)

    def _write_update_metrics(self, stats: dict[str, float], buffer: TensorDictRolloutBuffer) -> None:
        if not self.metrics_csv_path:
            return
        rewards = buffer.fields["rewards"][: int(buffer.pos)].detach().float().reshape(-1)
        returns = buffer.fields["returns"][: int(buffer.pos)].detach().float().reshape(-1)
        games = self._ep_wins + self._ep_losses + self._ep_draws
        row: dict[str, Any] = {
            "update": self._updates_completed,
            "timesteps": int(self.global_step),
            "episodes_completed": int(self._episodes_completed),
            "wins": int(self._ep_wins),
            "losses": int(self._ep_losses),
            "draws": int(self._ep_draws),
            "win_rate": float(self._ep_wins) / float(max(1, games)),
            "rollout_reward_mean": float(rewards.mean().detach().cpu().item()) if rewards.numel() > 0 else 0.0,
            "rollout_reward_std": float(rewards.std(unbiased=False).detach().cpu().item()) if rewards.numel() > 1 else 0.0,
            "rollout_return_mean": float(returns.mean().detach().cpu().item()) if returns.numel() > 0 else 0.0,
            "rollout_return_std": float(returns.std(unbiased=False).detach().cpu().item()) if returns.numel() > 1 else 0.0,
        }
        row.update(stats)
        self._write_csv_row(self.metrics_csv_path, self._update_fieldnames(), row)

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

    def _on_episode_done(self, info: dict[str, Any], *, timestep: Optional[int] = None) -> None:
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
        self._write_episode_metrics(info, blue_score=bs, red_score=rs, timestep=int(timestep or self.global_step))
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

    def _reset_strategy_state(self) -> None:
        if not self.use_latent_strategy:
            return
        n_envs = int(self.env.num_envs)
        self._current_z = torch.zeros((n_envs,), dtype=torch.long, device=self.device)
        self._strategy_age = torch.zeros((n_envs,), dtype=torch.long, device=self.device)
        self._needs_strategy_sample = torch.ones((n_envs,), dtype=torch.bool, device=self.device)

    def _strategy_for_step(
        self,
        global_state: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], dict[str, torch.Tensor]]:
        """Return current sparse strategy and sampling metadata for one rollout step."""
        if not self.use_latent_strategy:
            return None, None, {}
        if self._current_z is None:
            self._reset_strategy_state()
        assert self._current_z is not None

        resample_mask = self._needs_strategy_sample.clone()
        if self.latent_resample_every_n > 0:
            resample_mask |= self._strategy_age >= self.latent_resample_every_n

        prev_z = self._current_z.clone()
        z_idx = self._current_z.clone()
        persist_mask = resample_mask & (~self._needs_strategy_sample)

        if bool(resample_mask.any().item()):
            idx = torch.where(resample_mask)[0]
            sampled_z, _, _, _ = self.model.sample_strategy(
                global_state.index_select(0, idx),
                deterministic=False,
            )
            z_idx[idx] = sampled_z
            self._current_z = z_idx.clone()
            self._strategy_age[idx] = 0
            self._needs_strategy_sample[idx] = False

        z_logits = self.model.strategy_logits(global_state)
        z_dist = Categorical(logits=z_logits)
        z_log_prob = z_dist.log_prob(z_idx)
        z_entropy = z_dist.entropy()

        aux = {
            "z": z_idx,
            "prev_z": prev_z,
            "z_log_prob": z_log_prob,
            "z_entropy": z_entropy,
            "z_logits": z_logits,
            "z_resampled": resample_mask,
            "z_persist_mask": persist_mask,
        }
        return z_idx, prev_z, aux

    def _mark_strategy_step_done(self, dones: np.ndarray) -> None:
        if not self.use_latent_strategy:
            return
        done_t = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        self._strategy_age += 1
        if bool(done_t.any().item()):
            self._strategy_age[done_t] = 0
            self._needs_strategy_sample[done_t] = True

    def _obs_rows_from_next(
        self,
        next_obs: Dict[str, np.ndarray],
        infos: list[dict],
    ) -> Dict[str, np.ndarray]:
        rows: dict[str, list[np.ndarray]] = {key: [] for key in ("grid", "vec", "agent_mask", "mask")}
        for env_i, info in enumerate(infos):
            use_terminal = bool(info.get("truncated", False)) and isinstance(info.get("terminal_observation"), dict)
            terminal_obs = info.get("terminal_observation") if use_terminal else {}
            for key in rows:
                source = terminal_obs.get(key, next_obs[key][env_i]) if isinstance(terminal_obs, dict) else next_obs[key][env_i]
                rows[key].append(np.asarray(source, dtype=np.float32))
        return {key: np.stack(values, axis=0) for key, values in rows.items()}

    def _latent_rollout_stats(self, buffer: TensorDictRolloutBuffer) -> dict[str, float]:
        """Summarize strategy occupancy and switching for the latest rollout."""
        if not self.use_latent_strategy or "z" not in buffer.fields:
            return {}
        length = int(buffer.pos)
        z = buffer.fields["z"][:length].reshape(-1).long()
        prev_z = buffer.fields["prev_z"][:length].reshape(-1).long()
        if z.numel() == 0:
            return {}
        counts = torch.bincount(z.clamp(min=0, max=self.latent_k - 1), minlength=self.latent_k).float()
        occupancy = counts / counts.sum().clamp_min(1.0)
        persist_mask = buffer.fields["z_persist_mask"][:length].reshape(-1).bool()
        resampled = buffer.fields["z_resampled"][:length].reshape(-1).bool()
        switched = persist_mask & (z != prev_z)
        out = {
            "strategy_unique_count": float((counts > 0).sum().detach().cpu().item()),
            "strategy_dominant": float(torch.argmax(counts).detach().cpu().item()),
            "strategy_switch_count": float(switched.sum().detach().cpu().item()),
            "strategy_switch_fraction": float(
                (switched.float().sum() / persist_mask.float().sum().clamp_min(1.0)).detach().cpu().item()
            ),
            "strategy_resample_fraction_rollout": float(resampled.float().mean().detach().cpu().item()),
        }
        for idx, value in enumerate(occupancy.detach().cpu().tolist()):
            out[f"strategy_occupancy_{idx}"] = float(value)
        return out

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
        if self.use_latent_strategy:
            buffer.register_field("z", dtype=torch.long)
            buffer.register_field("prev_z", dtype=torch.long)
            buffer.register_field("z_log_probs")
            buffer.register_field("z_logits", (self.latent_k,))
            buffer.register_field("z_resampled", dtype=torch.bool)
            buffer.register_field("z_persist_mask", dtype=torch.bool)
        return buffer

    def _next_values(
        self,
        infos: list[dict],
        next_global_state: np.ndarray,
        next_obs: Optional[Dict[str, np.ndarray]] = None,
        prev_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
            if not self.use_latent_strategy:
                return self.model.values(gs)
            if next_obs is None or prev_z is None:
                raise ValueError("latent next value bootstrap requires next_obs and prev_z.")
            obs_rows = self._obs_rows_from_next(next_obs, infos)
            next_obs_t = self._tensor_obs(obs_rows)
            next_z = prev_z.long().reshape(-1)
            _, next_values, _, _ = self.model.act(
                next_obs_t,
                gs,
                deterministic=True,
                z_idx=next_z,
            )
            terminated = torch.as_tensor(
                [bool(info.get("terminated", False)) for info in infos],
                dtype=torch.bool,
                device=self.device,
            )
            return torch.where(terminated, torch.zeros_like(next_values), next_values)

    def collect_rollout(self) -> TensorDictRolloutBuffer:
        """Collect one rollout and compute advantages/returns."""
        if self._last_obs is None or self._last_global_state is None:
            obs = self.env.reset()
            global_state = self.env.state().astype(np.float32)
            self._reset_strategy_state()
        else:
            obs = self._last_obs
            global_state = self._last_global_state
        buffer = self._make_buffer(obs)
        for _ in range(int(self.cfg.n_steps)):
            obs_t = self._tensor_obs(obs)
            gs_t = torch.as_tensor(global_state, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                z_t, prev_z_t, strategy_aux = self._strategy_for_step(gs_t)
                actions_t, values_t, action_log_probs_t, _ = self.model.act(obs_t, gs_t, z_idx=z_t)
                if self.use_latent_strategy:
                    log_probs_t = action_log_probs_t + strategy_aux["z_log_prob"]
                else:
                    log_probs_t = action_log_probs_t
            actions_np = actions_t.detach().cpu().numpy().astype(np.int64)
            self.env.step_async(actions_np)
            next_obs, rewards, dones, infos = self.env.step_wait()
            step_after = self.global_step + int(self.env.num_envs)
            for done_i, info in zip(dones, infos):
                if bool(done_i):
                    self._on_episode_done(dict(info), timestep=step_after)
            next_global_state = self.env.state().astype(np.float32)
            next_values_t = self._next_values(infos, next_global_state, next_obs=next_obs, prev_z=z_t)
            terminated = np.asarray([bool(info.get("terminated", bool(done))) for info, done in zip(infos, dones)])
            truncated = np.asarray([bool(info.get("truncated", False)) for info in infos])

            add_items: dict[str, torch.Tensor] = dict(
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
            if self.use_latent_strategy:
                add_items.update(
                    z=strategy_aux["z"],
                    prev_z=strategy_aux["prev_z"],
                    z_log_probs=strategy_aux["z_log_prob"],
                    z_logits=strategy_aux["z_logits"],
                    z_resampled=strategy_aux["z_resampled"],
                    z_persist_mask=strategy_aux["z_persist_mask"],
                )
            buffer.add(**add_items)
            obs = next_obs
            global_state = next_global_state
            self.global_step += int(self.env.num_envs)
            self._mark_strategy_step_done(dones)

        buffer.compute_returns_and_advantages(
            gamma=float(self.cfg.gamma),
            gae_lambda=float(self.cfg.gae_lambda),
        )
        self._last_obs = obs
        self._last_global_state = global_state
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
            "strategy_entropy": [],
            "strategy_persist_loss": [],
            "strategy_resample_fraction": [],
        }
        for _ in range(self.n_epochs):
            for batch in buffer.iter_minibatches(self.batch_size, shuffle=True):
                obs_batch = {
                    "grid": batch["obs_grid"],
                    "vec": batch["obs_vec"],
                    "agent_mask": batch["obs_agent_mask"],
                    "mask": batch["obs_mask"],
                }
                z_idx = batch["z"] if self.use_latent_strategy else None
                values, action_log_prob, entropy, aux = self.model.evaluate_actions(
                    obs_batch,
                    batch["global_state"],
                    batch["actions"],
                    z_idx=z_idx,
                )
                if self.use_latent_strategy:
                    resample = batch["z_resampled"].bool()
                    persist_mask = batch["z_persist_mask"].bool()
                    strategy_log_prob = aux["strategy_log_prob"]
                    log_prob = action_log_prob + strategy_log_prob
                    strategy_entropy = aux["strategy_entropy"]
                    strategy_entropy_loss = -float(getattr(self.cfg, "latent_lam_h", 0.0)) * strategy_entropy.mean()
                    switch_penalty = expected_strategy_switch_penalty(aux["strategy_logits"], batch["prev_z"])
                    if bool(persist_mask.any().item()):
                        persist_loss = switch_penalty[persist_mask].mean()
                    else:
                        persist_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                    latent_loss = float(getattr(self.cfg, "latent_lam_p", 0.0)) * persist_loss + strategy_entropy_loss
                else:
                    log_prob = action_log_prob
                    strategy_entropy = torch.zeros_like(entropy)
                    persist_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                    latent_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                    resample = torch.zeros_like(entropy, dtype=torch.bool)

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
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss + latent_loss

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
                stats["strategy_entropy"].append(float(strategy_entropy.mean().detach().cpu().item()))
                stats["strategy_persist_loss"].append(float(persist_loss.detach().cpu().item()))
                stats["strategy_resample_fraction"].append(float(resample.float().mean().detach().cpu().item()))

        self.last_stats = {name: float(np.mean(values)) if values else 0.0 for name, values in stats.items()}
        self.last_stats["learning_rate"] = float(lr)
        self.last_stats.update(self._latent_rollout_stats(buffer))
        return self.last_stats

    def learn(self, total_timesteps: int) -> dict[str, float]:
        """Train until at least ``total_timesteps`` environment transitions have been collected."""
        total = int(total_timesteps)
        while self.global_step < total:
            rollout = self.collect_rollout()
            stats = self.update(rollout, total_timesteps=total)
            self._updates_completed += 1
            self._write_update_metrics(stats, rollout)
            if bool(getattr(self.cfg, "verbose_training", False)):
                latent_bits = ""
                if self.use_latent_strategy:
                    latent_bits = (
                        f" z_entropy={stats.get('strategy_entropy', 0.0):.4f} "
                        f"z_persist={stats.get('strategy_persist_loss', 0.0):.4f}"
                    )
                print(
                    "[PPO|custom] "
                    f"steps={self.global_step} policy_loss={stats['policy_loss']:.4f} "
                    f"value_loss={stats['value_loss']:.4f} approx_kl={stats['approx_kl']:.5f}"
                    f"{latent_bits}"
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
                "updates_completed": self._updates_completed,
                "cfg": asdict(self.cfg),
                "last_stats": self.last_stats,
                "format": "custom_ppo_latent_v1" if self.use_latent_strategy else "custom_ppo_v1",
            },
            path,
        )

    def load(self, path: str) -> None:
        """Restore a checkpoint produced by :meth:`save`."""
        payload = _torch_load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.global_step = int(payload.get("global_step", 0))
        self._updates_completed = int(payload.get("updates_completed", 0))
        self.last_stats = dict(payload.get("last_stats", {}))
        self._last_obs = None
        self._last_global_state = None
        self._current_z = None
