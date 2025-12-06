# obs_encoder.py
import torch
import torch.nn as nn


class ObsEncoder(nn.Module):
    """
    Paper-style ObsEncoder: simple MLP over the flat observation vector.

    Observation layout (from GameField.build_observation):

      • 19 scalar features:
          0  : dx_enemy_flag
          1  : dy_enemy_flag
          2  : dx_own_flag
          3  : dy_own_flag
          4  : is_carrying_flag
          5  : own_flag_taken
          6  : enemy_flag_taken
          7  : side_blue
          8  : ammo_norm
          9  : is_miner
          10 : dx_mine
          11 : dy_mine
          12 : agent_id_onehot[0]
          13 : agent_id_onehot[1]
          14 : teammate_mines_norm
          15 : teammate_has_flag
          16 : teammate_dist
          17 : time_norm
          18 : decision_norm

      • 25 spatial features: 5×5 local occupancy grid (flattened, normalized to [0,1])

      → total_dim = 19 + 25 = 44
    """

    def __init__(
        self,
        input_dim: int = 44,
        hidden_dim: int = 128,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            gain = nn.init.calculate_gain("tanh")
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [B, input_dim] or [input_dim]
        returns:
            latent: [B, latent_dim]
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        assert obs.size(1) == self.input_dim, \
            f"ObsEncoder expected {self.input_dim} dims, got {obs.size(1)}"

        latent = self.net(obs)
        return latent
