import math
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.distributions import Categorical

from obs_encoder import ObsEncoder
from macro_actions import MacroAction


class ActorCriticNet(nn.Module):
    """
    Actor-Critic network for the 44-D macro-action observation.
    """
    def __init__(
        self,
        input_dim: int = 44, # full observation dimension
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
        latent = self.encoder(obs)           # [B, latent_dim]
        logits = self.actor(latent)          # [B, n_actions]
        value = self.critic(latent).squeeze(-1)  # [B]
        return logits, value

    @torch.no_grad()
    def get_action_mask(self, agent, game_field) -> torch.Tensor:
        n = len(MacroAction)
        device = next(self.parameters()).device
        mask = torch.ones(n, dtype=torch.bool, device=device)

        # Can't place mine without charges
        if getattr(agent, "mine_charges", 0) <= 0:
            mask[MacroAction.PLACE_MINE.value] = False

        # Can't grab mine if no friendly pickups within ~3 cells
        has_friendly_pickup = any(
            p.owner_side == agent.side and
            math.hypot(p.x - agent._float_x, p.y - agent._float_y) < 3.0
            for p in game_field.mine_pickups
        )
        if not has_friendly_pickup:
            mask[MacroAction.GRAB_MINE.value] = False

        # Optional: never DEFEND_ZONE when carrying the flag
        if agent.isCarryingFlag():
            mask[MacroAction.DEFEND_ZONE.value] = False

        # Block INTERCEPT / SUPPRESS if no enemy carrier
        enemy_has_flag = (
            (agent.side == "blue" and game_field.manager.red_flag_taken) or
            (agent.side == "red"  and game_field.manager.blue_flag_taken)
        )
        if not enemy_has_flag:
            mask[MacroAction.INTERCEPT_CARRIER.value] = False
            mask[MacroAction.SUPPRESS_CARRIER.value] = False

        return mask

    @torch.no_grad()
    def act(self, obs: torch.Tensor, agent=None, game_field=None, deterministic: bool = False, ) -> Dict[str, Any]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        logits, value = self.forward(obs)

        mask = None
        if agent is not None and game_field is not None:
            # ACTION MASKING (invalid → -inf)
            mask = self.get_action_mask(agent, game_field)  # [n_actions] bool
            logits = logits.masked_fill(~mask, -1e10)

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
            "mask": mask,
        }

    def evaluate_actions( self, obs: torch.Tensor, actions: torch.Tensor,):
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values
