# rl_policy.py
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Dict, Any, Optional

from obs_encoder import ObsEncoder
from macro_actions import MacroAction


class ActorCriticNet(nn.Module):
    """
    High-performance Actor-Critic network with shared ObsEncoder.
    Designed for continuous CTF with 37-D observations.
    """

    def __init__(
        self,
        obs_dim: int = 37,
        n_actions: int = len(MacroAction),
        latent_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_actions = n_actions

        self.encoder = ObsEncoder(
            n_scalar=12,
            spatial_size=5,
            embed_dim=128,
            latent_dim=latent_dim,
        )

        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, n_actions),
        )

        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, 1),
        )

    def forward(self, obs: torch.Tensor):
        latent = self.encoder(obs)
        logits = self.actor_head(latent)
        value = self.critic_head(latent).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """
        Used during inference (e.g. in driver.py)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return {
            "action": action,           # [B]
            "log_prob": log_prob,       # [B]
            "value": value,             # [B]
        }

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Used by PPO during training.
        """
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, values