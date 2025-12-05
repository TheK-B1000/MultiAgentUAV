import torch
import torch.nn as nn


class ObsEncoder(nn.Module):
    """
    ObsEncoder for the 42-D observation (scalar + 5×5 occupancy).

    Layout (must match GameField.build_observation):

      • 17 scalar features:
          0-11 : original 12 (dx/dy flag, carrying, taken flags, side, ammo, etc.)
          12-13: agent_id one-hot [1,0] or [0,1]
          14   : teammate_mines_norm
          15   : teammate_has_flag
          16   : teammate_dist

      • 25 spatial (5×5 local occupancy grid)
      → total_dim = 17 + 25 = 42

    We process:
      - Scalars via an MLP with residual.
      - Spatial part via small ConvNet over the 5×5 grid.
      - Then fuse both into a latent representation for actor/critic.
    """

    def __init__(
        self,
        n_scalar: int = 17,
        spatial_side: int = 5,
        hidden_dim: int = 256,
        latent_dim: int = 256,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.n_scalar = n_scalar
        self.spatial_side = spatial_side
        self.spatial_dim = spatial_side * spatial_side  # 25

        # ── Scalar branch (richer + residual) ─────────────────────────────
        self.scalar_net = nn.Sequential(
            nn.Linear(n_scalar, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.scalar_residual = nn.Linear(n_scalar, hidden_dim)  # residual connection

        # ── Spatial branch (Conv2d over 5×5 neighborhood) ─────────────────
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),  # → [B, hidden_dim, 1, 1]
            nn.Flatten(),             # → [B, hidden_dim]
        )

        # ── Final fusion → latent ─────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Orthogonal init for stability
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [B, 42] or [42]
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        expected_dim = self.n_scalar + self.spatial_dim  # 17 + 25 = 42
        assert obs.size(1) == expected_dim, \
            f"ObsEncoder expected {expected_dim} dims, got {obs.size(1)}"

        # Split
        scalars = obs[:, :self.n_scalar]                    # [B, 17]
        spatial_flat = obs[:, self.n_scalar:]               # [B, 25]
        spatial = spatial_flat.view(-1, 1, self.spatial_side, self.spatial_side)  # [B, 1, 5, 5]

        # Scalar processing + residual
        s = self.scalar_net(scalars) + self.scalar_residual(scalars)
        s = nn.functional.silu(s)

        # Spatial processing via CNN
        p = self.spatial_conv(spatial)                      # [B, hidden_dim]

        # Fuse
        fused = torch.cat([s, p], dim=-1)                   # [B, hidden_dim * 2]
        latent = self.fusion(fused)

        return latent
