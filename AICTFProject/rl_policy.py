import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Dict, Any

from obs_encoder import ObsEncoder
from macro_actions import MacroAction   # ← FIXED: import from macro_actions


class ActorCriticNet(nn.Module):
    """
    Actor-Critic network for the new observation space.

    Observation layout (from GameField.build_observation):

      • 17 scalar features:
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

      • 25 spatial features: 5×5 local occupancy grid (flattened)

      → total_dim = 17 + 25 = 42
    """

    def __init__(
        self,
        n_scalar: int = 17,           # scalar part (see above)
        spatial_side: int = 5,        # 5×5 occupancy
        n_actions: int = len(MacroAction),
        hidden_dim: int = 256,
        latent_dim: int = 256,
    ):
        super().__init__()

        # Encoder that handles scalar + spatial split
        self.encoder = ObsEncoder(
            n_scalar=n_scalar,
            spatial_side=spatial_side,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )

        self.actor = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # Optional: orthogonal init for stability
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.8)
                nn.init.constant_(layer.bias, 0)

        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0)

    def forward(self, obs: torch.Tensor):
        """
        obs: [B, 42] or [42]
        returns:
            logits: [B, n_actions]
            value:  [B]  (1D, squeezed)
        """
        latent = self.encoder(obs)
        logits = self.actor(latent)
        value = self.critic(latent).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """
        Used during rollout collection.

        Returns:
            {
              "action":   LongTensor [B],
              "log_prob": FloatTensor [B],
              "value":    FloatTensor [B],
            }
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return {
            "action": action,
            "log_prob": log_prob,
            "value": value,
        }

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Used in PPO update:

        Inputs:
            obs:     [B, 42]
            actions: [B] (LongTensor, MacroAction indices)

        Outputs:
            log_probs: [B]
            entropy:   [B]
            values:    [B]
        """
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values
