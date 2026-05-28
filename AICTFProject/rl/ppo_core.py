"""PPO math and named rollout storage for the CTF MARL trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Optional

import torch
import torch.nn.functional as F


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: Optional[torch.Tensor] = None,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    latent_z: Optional[torch.Tensor] = None,
    reset_gae_on_z_change: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE while bootstrapping time-limit truncations.

    Args:
        rewards: Tensor with shape ``(T, B)``.
        values: Value estimates for ``s_t`` with shape ``(T, B)``.
        next_values: Value estimates for transition next states with shape
            ``(T, B)``. For truncated auto-reset envs this must be the value of
            the terminal observation, not the reset observation.
        terminated: Game-rule terminal flags with shape ``(T, B)``.
        truncated: Time-limit/reset flags with shape ``(T, B)``.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        latent_z: Optional strategy index per transition ``(T, B)`` (integral). When
            ``reset_gae_on_z_change`` is True, GAE carry from step ``t+1`` back into ``t``
            is zeroed if ``z[t] != z[t+1]``, so a discontinuous :math:`V(s,z)` across
            mid-episode strategy switches does not smear credit across the boundary.
        reset_gae_on_z_change: Only used when ``latent_z`` is provided.

    Returns:
        ``(advantages, returns)`` tensors with shape ``(T, B)``.
    """
    if rewards.shape != values.shape or rewards.shape != next_values.shape:
        raise ValueError("rewards, values, and next_values must have matching (T, B) shapes.")
    if rewards.shape != terminated.shape:
        raise ValueError("terminated must match rewards shape.")
    if truncated is None:
        truncated = torch.zeros_like(terminated, dtype=torch.bool)
    if truncated.shape != rewards.shape:
        raise ValueError("truncated must match rewards shape.")

    rewards = rewards.float()
    values = values.float()
    next_values = next_values.float()
    terminated = terminated.bool()
    truncated = truncated.bool()

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[0])
    gamma_f = float(gamma)
    lambda_f = float(gae_lambda)

    z_use = latent_z
    if z_use is not None and z_use.shape != rewards.shape:
        raise ValueError("latent_z must match rewards shape when provided.")
    for step in reversed(range(rewards.shape[0])):
        bootstrap_non_terminal = (~terminated[step]).float()
        same_episode_next = (~(terminated[step] | truncated[step])).float()
        if bool(reset_gae_on_z_change) and z_use is not None and step + 1 < int(rewards.shape[0]):
            same_z = (z_use[step].long() == z_use[step + 1].long()).float()
        else:
            same_z = torch.ones_like(same_episode_next)
        same_episode_next = same_episode_next * same_z
        delta = rewards[step] + gamma_f * next_values[step] * bootstrap_non_terminal - values[step]
        last_gae = delta + gamma_f * lambda_f * same_episode_next * last_gae
        advantages[step] = last_gae
    return advantages, advantages + values


def align_next_values_to_rollout_actions(
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Use collected ``V/Q`` rows as next values for in-rollout continuations.

    The trainer's centralized critic can be action-conditioned. For transition
    ``t`` inside the same rollout and episode, the actual next action/value is
    collected at row ``t + 1``; using that row avoids bootstrapping from a
    synthetic deterministic action. Final rollout rows and episode boundaries
    keep their already-computed bootstrap values.
    """
    if values.shape != next_values.shape:
        raise ValueError("values and next_values must have matching (T, B) shapes.")
    if values.shape != terminated.shape:
        raise ValueError("terminated must match values shape.")
    if truncated is None:
        truncated = torch.zeros_like(terminated, dtype=torch.bool)
    if truncated.shape != values.shape:
        raise ValueError("truncated must match values shape.")

    aligned = next_values.float().clone()
    if int(values.shape[0]) <= 1:
        return aligned
    continuation = ~(terminated[:-1].bool() | truncated[:-1].bool())
    aligned[:-1] = torch.where(continuation, values[1:].float(), aligned[:-1])
    return aligned


def ppo_policy_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return clipped PPO policy loss and diagnostic tensors."""
    log_ratio = new_log_prob - old_log_prob
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_range), 1.0 + float(clip_range))
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    loss = -surrogate.mean()
    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = (torch.abs(ratio - 1.0) > float(clip_range)).float().mean()
    return loss, {"ratio": ratio, "approx_kl": approx_kl, "clip_fraction": clip_fraction}


def ppo_value_loss(
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip_range_vf: Optional[float],
) -> torch.Tensor:
    """Return PPO value loss with optional clipped value updates."""
    new_values = new_values.float()
    old_values = old_values.float()
    returns = returns.float()
    if clip_range_vf is None:
        return F.mse_loss(new_values, returns)
    clipped = old_values + torch.clamp(new_values - old_values, -float(clip_range_vf), float(clip_range_vf))
    unclipped_loss = (new_values - returns) ** 2
    clipped_loss = (clipped - returns) ** 2
    return torch.maximum(unclipped_loss, clipped_loss).mean()


@dataclass(frozen=True)
class RolloutField:
    """Metadata for one registered rollout tensor."""

    shape: tuple[int, ...]
    dtype: torch.dtype


class TensorDictRolloutBuffer:
    """Named-field rollout buffer with one-line extensibility for new fields."""

    def __init__(self, buffer_size: int, n_envs: int, *, device: torch.device | str = "cpu") -> None:
        self.buffer_size = int(buffer_size)
        self.n_envs = int(n_envs)
        self.device = torch.device(device)
        self.registry: dict[str, RolloutField] = {}
        self.fields: dict[str, torch.Tensor] = {}
        self.pos = 0
        self.full = False

    def register_field(
        self,
        name: str,
        shape: tuple[int, ...] = (),
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Register a field tensor with shape ``(T, B, *shape)``."""
        if name in self.fields:
            raise ValueError(f"Rollout field {name!r} is already registered.")
        shape_t = tuple(int(v) for v in shape)
        self.registry[name] = RolloutField(shape=shape_t, dtype=dtype)
        self.fields[name] = torch.zeros(
            (self.buffer_size, self.n_envs, *shape_t),
            dtype=dtype,
            device=self.device,
        )

    def add(self, **items: torch.Tensor) -> None:
        """Append one timestep of registered field values."""
        if self.pos >= self.buffer_size:
            raise RuntimeError("Rollout buffer is full; call reset() before adding more data.")
        missing = set(self.fields).difference(items)
        if missing:
            raise KeyError(f"Missing rollout fields: {sorted(missing)}")
        extra = set(items).difference(self.fields)
        if extra:
            raise KeyError(f"Unregistered rollout fields: {sorted(extra)}")
        for name, value in items.items():
            target = self.fields[name][self.pos]
            tensor = torch.as_tensor(value, device=self.device, dtype=target.dtype)
            if tuple(tensor.shape) != tuple(target.shape):
                raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {tuple(target.shape)}")
            target.copy_(tensor)
        self.pos += 1
        self.full = self.pos == self.buffer_size

    def reset(self) -> None:
        """Clear write position while keeping allocated tensors and registry."""
        self.pos = 0
        self.full = False

    def compute_returns_and_advantages(
        self,
        *,
        gamma: float,
        gae_lambda: float,
        reward_field: str = "rewards",
        value_field: str = "values",
        next_value_field: str = "next_values",
        terminated_field: str = "terminated",
        truncated_field: str = "truncated",
        latent_z_field: Optional[str] = None,
        reset_gae_on_z_change: bool = False,
    ) -> None:
        """Populate ``advantages`` and ``returns`` from registered transition fields."""
        length = self.pos
        if length != self.buffer_size:
            raise RuntimeError(f"Rollout buffer must be full before GAE; got {length}/{self.buffer_size}.")
        latent_z_tensor: Optional[torch.Tensor] = None
        if latent_z_field:
            latent_z_tensor = self.fields[str(latent_z_field)].long()
        advantages, returns = compute_gae(
            self.fields[reward_field],
            self.fields[value_field],
            self.fields[next_value_field],
            self.fields[terminated_field].bool(),
            self.fields[truncated_field].bool(),
            gamma=gamma,
            gae_lambda=gae_lambda,
            latent_z=latent_z_tensor,
            reset_gae_on_z_change=bool(reset_gae_on_z_change),
        )
        for name, tensor in (("advantages", advantages), ("returns", returns)):
            if name not in self.fields:
                self.register_field(name)
            self.fields[name].copy_(tensor)

    def iter_minibatches(
        self,
        batch_size: int,
        *,
        shuffle: bool = True,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield flattened minibatches for all registered fields."""
        length = self.pos
        total = length * self.n_envs
        if total <= 0:
            return
        indices = torch.randperm(total, device=self.device) if shuffle else torch.arange(total, device=self.device)
        flat: dict[str, torch.Tensor] = {
            name: value[:length].reshape(total, *value.shape[2:])
            for name, value in self.fields.items()
        }
        bs = max(1, int(batch_size))
        for start in range(0, total, bs):
            idx = indices[start : start + bs] 
            yield {name: value.index_select(0, idx) for name, value in flat.items()}

    def as_mapping(self) -> Mapping[str, torch.Tensor]:
        """Return the underlying field tensors."""
        return self.fields
