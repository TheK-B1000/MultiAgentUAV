# rl_policy.py

from typing import Dict, Any

import torch
import torch.nn as nn
from torch.distributions import Categorical

from obs_encoder import ObsEncoder
from game_field import MacroAction


class ActorCriticNet(nn.Module):
    """
    Shared Actor-Critic network:

      obs (12 scalars + 25 occupancy = 37) ──> ObsEncoder
                                               ↓ latent (latent_dim)
                                            shared MLP trunk
                                               ↓
                                  ┌──────── actor head  → logits over MacroAction
                                  └──────── critic head → scalar value
    """

    def __init__(
        self,
        obs_dim: int = 37,                 # kept for compatibility; encoder handles layout
        n_actions: int = len(MacroAction), # automatically 14 if MacroAction has 14 entries
        latent_dim: int = 128,
        trunk_hidden: int = 128,
    ):
        super().__init__()

        # ObsEncoder expects:
        #   - n_scalar: number of scalar features at the front of obs
        #   - spatial_side: side length of square occupancy grid (5x5 = 25)
        self.encoder = ObsEncoder(
            n_scalar=12,
            spatial_side=5,
            hidden_dim=latent_dim,
            latent_dim=latent_dim,
        )

        # Shared trunk after encoder — gives more capacity for 14 macro-actions
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, trunk_hidden),
            nn.ReLU(),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.ReLU(),
        )

        # Heads
        self.actor = nn.Linear(trunk_hidden, n_actions)
        self.critic = nn.Linear(trunk_hidden, 1)

    def forward(self, obs: torch.Tensor):
        """
        obs: [B, 37] or [37]
        returns:
            logits: [B, n_actions]
            value : [B]
        """
        # Ensure batch dimension
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        latent = self.encoder(obs)          # [B, latent_dim]
        trunk_latent = self.trunk(latent)   # [B, trunk_hidden]

        logits = self.actor(trunk_latent)   # [B, n_actions]
        value = self.critic(trunk_latent).squeeze(-1)  # [B]
        return logits, value

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """
        Sample (or take argmax) action for PPO.

        obs: [B, 37] or [37]
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
        Used by PPO for minibatch updates.

        obs: [B, 37]
        actions: [B] (long)
        returns:
            - log_probs: [B]
            - entropy  : [B]
            - values   : [B]
        """
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values
