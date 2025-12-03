import torch
import torch.nn as nn
from torch.distributions import Categorical

from typing import Dict, Any

from obs_encoder import ObsEncoder
from game_field import MacroAction


class ActorCriticNet(nn.Module):
    """
    Shared Actor-Critic network:
      - ObsEncoder -> latent feature
      - Actor head  -> logits over MacroAction
      - Critic head -> scalar value
    """

    def __init__(
        self,
        obs_dim: int = 37,
        n_actions: int = len(MacroAction),
        latent_dim: int = 128,
    ):
        super().__init__()

        # Hard-coded split for now: 12 scalars + 25 occupancy = 37
        self.encoder = ObsEncoder(
            n_scalar=12,
            spatial_side=5,
            hidden_dim=128,
            latent_dim=latent_dim,
        )

        self.actor = nn.Linear(latent_dim, n_actions)
        self.critic = nn.Linear(latent_dim, 1)

    def forward(self, obs: torch.Tensor):
        """
        obs: [B, 37] or [37]
        returns: logits [B, n_actions], value [B]
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
            - log_probs, entropy, values
        """
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values
