import math
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.distributions import Categorical

from obs_encoder import ObsEncoder
from macro_actions import MacroAction


class ActorCriticNet(nn.Module):
    """
    Actor-Critic network for the 7-channel 20x20 CNN observation.

    The policy is *architecturally* like the paper:
      - One head over macro-actions (5 discrete macros).
      - One head over 50 discrete spatial targets (macro_targets index).

    The joint action is (macro, target_idx).
    """

    def __init__(
        self,
        # legacy, unused
        input_dim: int = 44,
        n_macros: int = len(MacroAction),
        n_targets: int = 50,
        hidden_dim: int = 128,  # unused (we go straight from encoder to heads)
        latent_dim: int = 128,
    ):
        super().__init__()

        self.n_macros = n_macros
        self.n_targets = n_targets

        # Shared encoder: CNN over 7x20x20 → latent_dim
        self.encoder = ObsEncoder(
            in_channels=7,
            height=20,
            width=20,
            latent_dim=latent_dim,
        )

        # Two actor heads + one critic head
        self.actor_macro = nn.Linear(latent_dim, n_macros)
        self.actor_target = nn.Linear(latent_dim, n_targets)
        self.critic = nn.Linear(latent_dim, 1)

        self._init_heads()

    def _init_heads(self):
        # Orthogonal init for actor/critic with appropriate gains
        nn.init.orthogonal_(self.actor_macro.weight, gain=0.01)
        nn.init.constant_(self.actor_macro.bias, 0.0)

        nn.init.orthogonal_(self.actor_target.weight, gain=0.01)
        nn.init.constant_(self.actor_target.bias, 0.0)

        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        """
        obs: [B, 7, 20, 20] or [7, 20, 20]

        returns:
            macro_logits : [B, n_macros]
            target_logits: [B, n_targets]
            value        : [B]
        """
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)  # [1, 7, 20, 20]

        latent = self.encoder(obs)                   # [B, latent_dim]
        macro_logits = self.actor_macro(latent)      # [B, n_macros]
        target_logits = self.actor_target(latent)    # [B, n_targets]
        value = self.critic(latent).squeeze(-1)      # [B]

        return macro_logits, target_logits, value

    @torch.no_grad()
    def get_action_mask(self, agent, game_field) -> torch.Tensor:
        """
        Mask is ONLY over macros (not targets).

        True  = allowed
        False = invalid
        """
        n_macros = len(MacroAction)
        device = next(self.parameters()).device
        mask = torch.ones(n_macros, dtype=torch.bool, device=device)

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

        return mask

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        agent=None,
        game_field=None,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """
        obs: [7, 20, 20] or [B, 7, 20, 20]

        Returns dict:
            {
                "macro_action":  [B] long,
                "target_action": [B] long,
                "log_prob":      [B] float (joint log prob),
                "value":         [B],
                "macro_mask":    [n_macros] bool or None,
            }
        """
        # Forward pass
        macro_logits, target_logits, value = self.forward(obs)

        macro_mask = None
        if agent is not None and game_field is not None:
            macro_mask = self.get_action_mask(agent, game_field)  # [n_macros] bool
            # Mask invalid macros
            macro_logits = macro_logits.masked_fill(~macro_mask, -1e10)

        macro_dist = Categorical(logits=macro_logits)
        target_dist = Categorical(logits=target_logits)

        if deterministic:
            macro_action = torch.argmax(macro_logits, dim=-1)
            target_action = torch.argmax(target_logits, dim=-1)
        else:
            macro_action = macro_dist.sample()
            target_action = target_dist.sample()

        log_prob_macro = macro_dist.log_prob(macro_action)
        log_prob_target = target_dist.log_prob(target_action)
        joint_log_prob = log_prob_macro + log_prob_target

        return {
            "macro_action": macro_action,      # [B]
            "target_action": target_action,    # [B]
            "log_prob": joint_log_prob,        # [B]
            "value": value,                    # [B]
            "macro_mask": macro_mask,
        }

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        macro_actions: torch.Tensor,
        target_actions: torch.Tensor,
    ):
        """
        obs:           [B, 7, 20, 20]
        macro_actions: [B]   (indices in MacroAction)
        target_actions:[B]   (indices in macro_targets 0..49)
        """
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)

        macro_logits, target_logits, values = self.forward(obs)

        macro_dist = Categorical(logits=macro_logits)
        target_dist = Categorical(logits=target_logits)

        log_probs_macro = macro_dist.log_prob(macro_actions)
        log_probs_target = target_dist.log_prob(target_actions)
        log_probs = log_probs_macro + log_probs_target

        # Sum entropies: joint entropy of independent factors ~ H(m) + H(t)
        entropy = macro_dist.entropy() + target_dist.entropy()

        return log_probs, entropy, values
