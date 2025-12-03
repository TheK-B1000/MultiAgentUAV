# obs_encoder.py
# ==========================================================
# Observation encoder for 2D CTF MARL
# - Expects: 15 scalar features + 5x5 occupancy (25) = 40 dims
# - Splits into scalar + spatial streams, then fuses to latent
# ==========================================================

from typing import Optional

import torch
import torch.nn as nn


class ObsEncoder(nn.Module):
    """
    Encodes a flat observation vector into a latent feature.

    Expected layout (total_dim = n_scalar + spatial_side*spatial_side):

        obs = [
            scalars (n_scalar floats),
            occupancy_flat (spatial_side * spatial_side floats)
        ]

    In your current GameField.build_observation:

        n_scalar = 15
        spatial_side = 5  -> 25 occupancy cells
        total_dim = 40
    """

    def __init__(
        self,
        n_scalar: int = 15,
        spatial_side: int = 5,
        hidden_dim: int = 128,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.n_scalar = n_scalar
        self.spatial_side = spatial_side
        self.spatial_dim = spatial_side * spatial_side
        self.total_dim = n_scalar + self.spatial_dim

        # Simple 2-layer MLP for scalar channel
        self.scalar_mlp = nn.Sequential(
            nn.Linear(self.n_scalar, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Simple 2-layer MLP for spatial (flattened occupancy) channel
        self.spatial_mlp = nn.Sequential(
            nn.Linear(self.spatial_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Fusion MLP to produce the final latent feature
        self.fuse_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, latent_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [B, total_dim] or [total_dim]
        returns: latent [B, latent_dim]
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # [1, total_dim]

        # Sanity check on last dimension (optional but helpful)
        if obs.size(-1) != self.total_dim:
            raise ValueError(
                f"ObsEncoder expected last dim = {self.total_dim} "
                f"(n_scalar={self.n_scalar}, spatial_dim={self.spatial_dim}), "
                f"got {obs.size(-1)}"
            )

        # Split into scalar and spatial parts
        scalar = obs[..., : self.n_scalar]                  # [B, n_scalar]
        spatial_flat = obs[..., self.n_scalar :]            # [B, spatial_dim]

        # Pass each through its own MLP
        scalar_feat = self.scalar_mlp(scalar)               # [B, hidden_dim]
        spatial_feat = self.spatial_mlp(spatial_flat)       # [B, hidden_dim]

        # Fuse and project to latent
        fused = torch.cat([scalar_feat, spatial_feat], dim=-1)  # [B, 2*hidden_dim]
        latent = self.fuse_mlp(fused)                           # [B, latent_dim]
        return latent
