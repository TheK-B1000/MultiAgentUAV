# obs_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ObsEncoder(nn.Module):
    """
    Advanced encoder for the 37-D observation vector from the continuous CTF environment.

    Observation breakdown (37 dims):
        0-11 : Scalar features (12 total)
               [dx_enemy_flag, dy_enemy_flag,
                dx_own_flag,    dy_own_flag,
                is_carrying_flag, own_flag_taken, enemy_flag_taken, side_blue,
                ammo_norm, is_miner, dx_mine_pickup, dy_mine_pickup]

        12-36: 5x5 local occupancy grid (25 values)
               0=empty, 1=wall/out-of-bounds, 2=friendly, 3=enemy, 4=mine, 5=own_pickup
    """

    def __init__(
        self,
        n_scalar: int = 12,
        spatial_size: int = 5,
        embed_dim: int = 128,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.n_scalar = n_scalar
        self.spatial_size = spatial_size
        self.spatial_channels = 8  # Learned spatial feature maps

        # --- Scalar branch ---
        self.scalar_net = nn.Sequential(
            nn.Linear(n_scalar, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        # --- Spatial branch: treat 5x5 as 1-channel image ---
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, self.spatial_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Flatten spatial features
        self.spatial_flatten_size = self.spatial_channels * spatial_size * spatial_size

        # --- Fusion ---
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + self.spatial_flatten_size, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [B, 37] or [37]

        Returns:
            latent: [B, latent_dim]
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # [1, 37]
        B = obs.shape[0]

        # Split
        scalars = obs[:, : self.n_scalar]                    # [B, 12]
        spatial_flat = obs[:, self.n_scalar :]               # [B, 25]

        # Process scalars
        s_feat = self.scalar_net(scalars)                    # [B, embed_dim]

        # Process spatial: reshape to [B, 1, 5, 5]
        spatial = spatial_flat.view(B, 1, self.spatial_size, self.spatial_size)
        p_feat = self.spatial_conv(spatial)                  # [B, C, 5, 5]
        p_feat = p_feat.view(B, -1)                          # [B, C*5*5]

        # Fuse
        combined = torch.cat([s_feat, p_feat], dim=-1)
        latent = self.fusion(combined)
        return latent