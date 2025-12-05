# rl_policy.py
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Dict, Any

from obs_encoder import ObsEncoder
from macro_actions import MacroAction


class ActorCriticNet(nn.Module):
    """
    Actor-Critic network for the 42-D macro-action observation.

    Architecture (paper-aligned, fully-connected):

      obs (42) → shared MLP trunk (2 layers, Tanh)
                → latent (128)

      From latent:
        - actor head:  Linear(latent_dim → n_actions)
        - critic head: Linear(latent_dim → 1)

    This is closer to a classic PPO / paper-style setup:
    simple, linear layers with Tanh, no convolutions or extra bells.
    """

    def __init__(
        self,
        input_dim: int = 42,          # full observation dimension
        n_actions: int = len(MacroAction),
        hidden_dim: int = 128,
        latent_dim: int = 128,
    ):
        super().__init__()

        # Shared encoder MLP
        self.encoder = ObsEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )

        # Simple linear heads from shared latent
        self.actor = nn.Linear(latent_dim, n_actions)
        self.critic = nn.Linear(latent_dim, 1)

        self._init_heads()

    def _init_heads(self):
        # Orthogonal init for actor/critic with appropriate gains
        nn.init.orthogonal_(self.actor.weight, gain=0.01)  # small logits at start
        nn.init.constant_(self.actor.bias, 0.0)

        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        """
        obs: [B, input_dim] or [input_dim]
        returns:
            logits: [B, n_actions]
            value:  [B]  (1D, squeezed)
        """
        latent = self.encoder(obs)           # [B, latent_dim]
        logits = self.actor(latent)          # [B, n_actions]
        value = self.critic(latent).squeeze(-1)  # [B]
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
            obs:     [B, input_dim]
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
