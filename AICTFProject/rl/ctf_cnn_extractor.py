"""Shared CNN feature extractor for CTF tokenized observations (used by PPO and latent MARL)."""

from __future__ import annotations

import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class _PerAgentGridCNN(torch.nn.Module):
    """Light CNN for small spatial maps (e.g. 20×20). SB3 ``NatureCNN`` is sized for Atari (~84×84)."""

    def __init__(self, n_channels: int, features_dim: int):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(start_dim=1),
        )
        self.proj = torch.nn.Linear(64, features_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.trunk(x))


class TokenizedCombinedExtractor(BaseFeaturesExtractor):
    """
    Per-agent CNN on local grids, concatenated with per-agent vectors.
    grid (B, M, C, H, W), vec (B, M, V) -> (B, M * cnn_dim + M * V).
    """

    def __init__(self, observation_space, cnn_output_dim: int = 256, normalized_image: bool = True):
        import gymnasium as gym
        from gymnasium import spaces

        assert isinstance(observation_space, gym.Space) and hasattr(observation_space, "spaces")
        spaces_dict = observation_space.spaces
        grid_space = spaces_dict.get("grid")
        vec_space = spaces_dict.get("vec")
        assert grid_space is not None and vec_space is not None
        grid_shape = getattr(grid_space, "shape", None)
        vec_shape = getattr(vec_space, "shape", None)
        assert len(grid_shape) == 4, f"tokenized grid must be (M, C, H, W), got {grid_shape}"
        assert len(vec_shape) == 2, f"tokenized vec must be (M, V), got {vec_shape}"
        M, C, H, W = grid_shape
        V = vec_shape[1]

        assert H >= 3 and W >= 3, f"CNN requires H,W>=3; got {(H, W)}"
        cnn_latent_dim = int(cnn_output_dim)
        features_dim = int(M) * cnn_latent_dim + int(M) * int(V)
        context_space = spaces_dict.get("context")
        self._context_dim = 0
        if context_space is not None and hasattr(context_space, "shape"):
            self._context_dim = int(np.prod(context_space.shape))
            features_dim += self._context_dim
        super().__init__(observation_space, features_dim)
        self._M = int(M)
        self._V = int(V)
        self.vec_dim = int(V)
        self.cnn = _PerAgentGridCNN(int(C), cnn_latent_dim)

    def forward(self, observations):
        grid = observations["grid"]
        vec = observations["vec"]
        B, M = grid.shape[0], self._M

        grid_flat = grid.reshape(B * M, *grid.shape[2:])
        cnn_out = self.cnn(grid_flat)
        D = cnn_out.shape[1]
        cnn_out = cnn_out.reshape(B, M, D)

        agent_mask = observations.get("agent_mask", None)
        if agent_mask is not None:
            if agent_mask.dim() == 1:
                agent_mask = agent_mask.unsqueeze(0)
            agent_mask = agent_mask.float().unsqueeze(-1)
            cnn_out = cnn_out * agent_mask
            vec = vec * agent_mask

        cnn_out = cnn_out.reshape(B, M * D)
        vec_flat = vec.reshape(B, M * self._V)
        out = torch.cat([cnn_out, vec_flat], dim=1)
        if self._context_dim > 0 and "context" in observations:
            ctx = observations["context"]
            if ctx.dim() == 1:
                ctx = ctx.unsqueeze(0)
            ctx = ctx.float()
            if ctx.shape[-1] != self._context_dim:
                ctx = ctx.reshape(ctx.shape[0], -1)[:, : self._context_dim]
            out = torch.cat([out, ctx], dim=1)
        return out
