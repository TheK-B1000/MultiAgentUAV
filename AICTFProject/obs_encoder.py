import torch
import torch.nn as nn


class ObsEncoder(nn.Module):
    """
    Encode the 37-D observation from GameField.build_observation(agent)
    into a latent feature vector.

    Layout (current GameField):
        - 12 scalars:
            [dx_enemy_flag, dy_enemy_flag, dx_own_flag, dy_own_flag,
             is_carrying_flag, own_flag_taken, enemy_flag_taken, side_blue,
             ammo_norm, is_miner, dx_mine, dy_mine]
        - 25 local occupancy values (5x5)
    """

    def __init__(
        self,
        n_scalar: int = 12,
        spatial_side: int = 5,
        hidden_dim: int = 128,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.n_scalar = n_scalar
        self.spatial_side = spatial_side
        self.spatial_dim = spatial_side * spatial_side

        # Simple MLP branches for scalar + spatial parts
        self.scalar_mlp = nn.Sequential(
            nn.Linear(n_scalar, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.spatial_mlp = nn.Sequential(
            nn.Linear(self.spatial_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, latent_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [B, 37] (or [37] -> will be unsqueezed to [1, 37])
        returns: [B, latent_dim]
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        assert (
            obs.size(1) == self.n_scalar + self.spatial_dim
        ), f"Expected {self.n_scalar + self.spatial_dim}-D obs, got {obs.size(1)}"

        scalars = obs[:, : self.n_scalar]
        spatial = obs[:, self.n_scalar :]

        s_feat = self.scalar_mlp(scalars)
        p_feat = self.spatial_mlp(spatial)

        fused = torch.cat([s_feat, p_feat], dim=-1)
        latent = self.fuse(fused)
        return latent
