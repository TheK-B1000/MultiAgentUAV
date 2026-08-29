"""Neural network building blocks for CTF PPO and MAPPO policies."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn


def orthogonal_init(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    """Apply orthogonal initialization to linear and convolutional layers."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class CNNEncoder(nn.Module):
    """Encode a channels-first local game field into a fixed feature vector."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        feature_dim: int = 512,
        conv_channels: Sequence[int] = (32, 64, 64),
    ) -> None:
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"CNNEncoder input_shape must be (C, H, W), got {input_shape!r}")
        if len(conv_channels) != 3:
            raise ValueError("CNNEncoder requires exactly three convolutional channel sizes.")

        channels, height, width = (int(v) for v in input_shape)
        if channels <= 0 or height <= 0 or width <= 0:
            raise ValueError(f"CNNEncoder input dimensions must be positive, got {input_shape!r}")

        blocks: list[nn.Module] = []
        in_channels = channels
        for out_channels in conv_channels:
            blocks.extend(
                [
                    nn.Conv2d(in_channels, int(out_channels), kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                ]
            )
            in_channels = int(out_channels)

        self.conv = nn.Sequential(*blocks)
        self.flatten = nn.Flatten(start_dim=1)
        self.proj = nn.Linear(in_channels * height * width, int(feature_dim))
        self.feature_dim = int(feature_dim)
        self.input_shape = (channels, height, width)
        self.apply(orthogonal_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return encoded features with shape ``(B, feature_dim)``."""
        if obs.dim() != 4:
            raise ValueError(f"CNNEncoder expects (B, C, H, W), got {tuple(obs.shape)}")
        return self.proj(self.flatten(self.conv(obs.float())))


class PPOPolicy(nn.Module):
    """Visual actor that can accept dormant extra conditioning features."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        action_dim: int,
        feature_dim: int = 512,
        extra_dim: int = 0,
    ) -> None:
        super().__init__()
        self.cnn = CNNEncoder(obs_shape, feature_dim=feature_dim)
        self.extra_dim = int(extra_dim)
        self.actor_head = nn.Linear(int(feature_dim) + self.extra_dim, int(action_dim))
        orthogonal_init(self.actor_head, gain=0.01)

    def _combine(self, features: torch.Tensor, extra: torch.Tensor | None) -> torch.Tensor:
        if extra is None:
            if self.extra_dim == 0:
                return features
            extra = torch.zeros(
                (features.shape[0], self.extra_dim),
                dtype=features.dtype,
                device=features.device,
            )
        if extra.dim() != 2 or extra.shape[0] != features.shape[0] or extra.shape[1] != self.extra_dim:
            raise ValueError(
                f"extra must have shape ({features.shape[0]}, {self.extra_dim}), "
                f"got {tuple(extra.shape)}"
            )
        return torch.cat([features, extra.to(features.dtype)], dim=-1)

    def forward(self, obs: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        """Return action logits from local visual observations."""
        features = self.cnn(obs)
        return self.actor_head(self._combine(features, extra))


class CentralizedCritic(nn.Module):
    """
    Centralized **scalar state-value function** :math:`V_\\phi(s, z)` for clipped PPO / GAE.

    Inputs are global summary :math:`s` and, when latent strategy is enabled, a one-hot encoding
    of the discrete team strategy :math:`z`. The critic is **not** conditioned on the realized
    joint action :math:`\\mathbf{a}`; that conditioning would turn the PPO baseline into an
    action-value :math:`Q(s, \\mathbf{a}, z)` and bias the policy-gradient estimate.
    """

    def __init__(
        self,
        global_state_dim: int = 19,
        hidden_dim: int = 128,
        extra_dim: int = 0,
        private_z_heads: bool = False,
    ) -> None:
        super().__init__()
        self.global_state_dim = int(global_state_dim)
        self.extra_dim = int(extra_dim)
        self.private_z_heads = bool(private_z_heads)
        if self.private_z_heads and self.extra_dim != 2:
            raise ValueError("private_z_heads requires extra_dim=2 for z0/z1 routing")
        input_dim = self.global_state_dim + self.extra_dim
        self.input_dim = int(input_dim)
        # 128–128 MLP to a scalar; Word spec names ``critic_input``; no custom init in the spec.
        if not self.private_z_heads:
            self.net = nn.Sequential(
                nn.Linear(input_dim, int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
            )
            # Construct the ordinary shared projection first. It is deliberately
            # unregistered and exists only until copy_shared_head_into_private()
            # establishes exact R1/R2 initial-function equivalence.
            object.__setattr__(self, "_shared_head_init", nn.Linear(int(hidden_dim), 1))
            self.head_V0 = nn.Linear(int(hidden_dim), 1)
            self.head_V1 = nn.Linear(int(hidden_dim), 1)

    @property
    def trunk(self) -> nn.Module:
        """The shared two-layer critic trunk."""
        return self.net

    def copy_shared_head_into_private(self) -> None:
        """Copy the ordinary scalar-head initialization into both z heads."""
        if not self.private_z_heads:
            return
        source = getattr(self, "_shared_head_init", None)
        if source is None:
            raise RuntimeError("private critic shared initialization has already been consumed")
        with torch.no_grad():
            for head in (self.head_V0, self.head_V1):
                head.weight.copy_(source.weight)
                head.bias.copy_(source.bias)
        object.__setattr__(self, "_shared_head_init", None)

    def _combine(self, global_state: torch.Tensor, extra: torch.Tensor | None) -> torch.Tensor:
        if global_state.dim() != 2 or global_state.shape[1] != self.global_state_dim:
            raise ValueError(
                f"global_state must have shape (B, {self.global_state_dim}), "
                f"got {tuple(global_state.shape)}"
            )
        if extra is None:
            if self.extra_dim == 0:
                return global_state.float()
            extra = torch.zeros(
                (global_state.shape[0], self.extra_dim),
                dtype=global_state.dtype,
                device=global_state.device,
            )
        if extra.dim() != 2 or extra.shape[0] != global_state.shape[0] or extra.shape[1] != self.extra_dim:
            raise ValueError(
                f"extra must have shape ({global_state.shape[0]}, {self.extra_dim}), "
                f"got {tuple(extra.shape)}"
            )
        return torch.cat([global_state.float(), extra.to(global_state.dtype)], dim=-1)

    def forward(self, global_state: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        """Return scalar :math:`V(s,\\mathbf{a},z)` with shape ``(B, 1)``."""
        combined = self._combine(global_state, extra)
        if not self.private_z_heads:
            return self.net(combined)
        if extra is None:
            raise ValueError("private_z_heads requires an explicit z one-hot extra")
        z = extra.argmax(dim=-1)
        if bool(((z < 0) | (z > 1)).any().item()):
            raise ValueError("private critic supports only z0 and z1")
        h = self.trunk(combined)
        out = h.new_empty((h.shape[0], 1))
        rows0 = torch.where(z == 0)[0]
        rows1 = torch.where(z == 1)[0]
        if int(rows0.numel()):
            out[rows0] = self.head_V0(h.index_select(0, rows0))
        if int(rows1.numel()):
            out[rows1] = self.head_V1(h.index_select(0, rows1))
        return out


__all__ = ["CNNEncoder", "CentralizedCritic", "PPOPolicy", "orthogonal_init"]
