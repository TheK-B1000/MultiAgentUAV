# rl_policy.py
# ==========================================================
# Actor-Critic network for 2D CTF MARL
# - Uses ObsEncoder to process 40-dim observations
# - Outputs logits over MacroAction and a scalar value
# ==========================================================

from typing import Dict, Any

import torch
import torch.nn as nn
from torch.distributions import Categorical

from obs_encoder import ObsEncoder
from game_field import MacroAction


class ActorCriticNet(nn.Module):
    """
    Shared Actor-Critic network:
      - ObsEncoder -> latent feature
      - Actor head -> logits over MacroAction
      - Critic head -> scalar state-value V(s)

    Obs layout (from GameField.build_observation):
      - 15 scalars
      - 25 occupancy cells (5x5)
      => 40-dim vector per agent
    """

    def __init__(
        self,
        obs_dim: int = 40,                  # kept for reference / debugging
        n_actions: int = len(MacroAction),  # should be 14 with your expanded enum
        latent_dim: int = 128,
    ):
        super().__init__()

        # Encoder knows how to split [15 scalars + 25 occupancy] = 40 dims
        self.encoder = ObsEncoder(
            n_scalar=15,
            spatial_side=5,
            hidden_dim=128,
            latent_dim=latent_dim,
        )

        self.obs_dim = obs_dim
        self.n_actions = n_actions

        # Policy and value heads
        self.actor = nn.Linear(latent_dim, n_actions)
        self.critic = nn.Linear(latent_dim, 1)

    def forward(self, obs: torch.Tensor):
        """
        obs: [B, 40] or [40]
        returns:
            - logits: [B, n_actions]
            - value : [B]
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # [1, obs_dim]

        if obs.size(-1) != self.encoder.total_dim:
            raise ValueError(
                f"ActorCriticNet expected obs last dim = {self.encoder.total_dim}, "
                f"got {obs.size(-1)}"
            )

        latent = self.encoder(obs)               # [B, latent_dim]
        logits = self.actor(latent)              # [B, n_actions]
        value = self.critic(latent).squeeze(-1)  # [B]
        return logits, value

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """
        Sample (or take argmax) action for PPO.

        obs: [B, 40] or [40]
        returns dict with:
            - "action"   (Tensor [B])
            - "log_prob" (Tensor [B])
            - "value"    (Tensor [B])
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
            "action": action,       # [B]
            "log_prob": log_prob,   # [B]
            "value": value,         # [B]
        }

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Used by PPO for minibatch updates.

        obs:     [B, 40]
        actions: [B] (long)
        returns:
            - log_probs: [B]
            - entropy:   [B]
            - values:    [B]
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values
