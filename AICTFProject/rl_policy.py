import math
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.distributions import Categorical

from obs_encoder import ObsEncoder
from macro_actions import MacroAction


class ActorCriticNet(nn.Module):
    """
    CNN-based Actor-Critic network for macro-action PPO in the UAV CTF domain.

    Design:
      - Observation: per-agent egocentric CNN map, shape [7, 40, 30].
      - Actor: decentralized policy π(a_macro, a_target | s_i).
      - Critic: value function V(s_i) over the same per-agent latent.

    Notes:
      • This implementation is single-team PPO (BLUE only), not full MAPPO yet.
      • It is written to be easily extended to CTDE / MAPPO by swapping the
        value head to consume a centralized state (e.g., concatenated latents).
    """

    def __init__(
        self,
        n_macros: int = len(MacroAction),
        n_targets: int = 50,
        latent_dim: int = 128,
        in_channels: int = 7,
        height: int = 40,   # CNN_ROWS
        width: int = 30,    # CNN_COLS
    ):
        super().__init__()

        self.n_macros = n_macros
        self.n_targets = n_targets
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.height = height
        self.width = width

        # ------------------------------------------------------------------
        # 1. Shared encoder: [B, C, H, W] -> [B, latent_dim]
        # ------------------------------------------------------------------
        self.encoder = ObsEncoder(
            in_channels=in_channels,
            height=height,
            width=width,
            latent_dim=latent_dim,
        )

        # ------------------------------------------------------------------
        # 2. Actor heads over latent
        # ------------------------------------------------------------------
        self.actor_macro = nn.Linear(latent_dim, n_macros)
        self.actor_target = nn.Linear(latent_dim, n_targets)

        # ------------------------------------------------------------------
        # 3. Critic head over latent (per-agent V(s_i))
        # ------------------------------------------------------------------
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        self._init_heads()

    # ----------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------
    def _init_heads(self) -> None:
        # Orthogonal init for actor outputs (small gain for stable logits)
        nn.init.orthogonal_(self.actor_macro.weight, gain=0.01)
        nn.init.constant_(self.actor_macro.bias, 0.0)

        nn.init.orthogonal_(self.actor_target.weight, gain=0.01)
        nn.init.constant_(self.actor_target.bias, 0.0)

        # Critic last layer: gain 1.0
        last = self.value_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.orthogonal_(last.weight, gain=1.0)
            nn.init.constant_(last.bias, 0.0)

    # ----------------------------------------------------------------------
    # Core forwards
    # ----------------------------------------------------------------------
    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [C, H, W] or [B, C, H, W]
        returns: latent [B, latent_dim]
        """
        return self.encoder(obs)

    def forward_actor(self, obs: torch.Tensor):
        """
        Actor forward pass for a single agent's observation s_i.

        Args:
            obs: [C, 40, 30] or [B, C, 40, 30]

        Returns:
            macro_logits:  [B, n_macros]
            target_logits: [B, n_targets]
            latent:        [B, latent_dim]
        """
        latent = self._encode(obs)  # [B, latent_dim]
        macro_logits = self.actor_macro(latent)
        target_logits = self.actor_target(latent)
        return macro_logits, target_logits, latent

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Critic forward pass: per-agent value V(s_i).

        Args:
            obs: [B, C, 40, 30] per-agent observation batch.

        Returns:
            values: [B]
        """
        latent = self._encode(obs)
        value = self.value_head(latent).squeeze(-1)
        return value

    # ----------------------------------------------------------------------
    # Action masking (same semantics as before)
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def get_action_mask(self, agent, game_field) -> torch.Tensor:
        """
        Returns a boolean mask over macro-actions indicating which actions
        are currently valid for this agent in the given game state.

        True  => action allowed
        False => action masked out (logits -> -inf)
        """
        n_macros = len(MacroAction)
        device = next(self.parameters()).device
        mask = torch.ones(n_macros, dtype=torch.bool, device=device)

        # Can't place mine without charges
        if getattr(agent, "mine_charges", 0) <= 0:
            mask[MacroAction.PLACE_MINE.value] = False

        # Can't grab mine if no friendly pickups within ~3 cells
        has_friendly_pickup = any(
            p.owner_side == agent.side
            and math.hypot(p.x - agent._float_x, p.y - agent._float_y) < 3.0
            for p in game_field.mine_pickups
        )
        if not has_friendly_pickup:
            mask[MacroAction.GRAB_MINE.value] = False

        return mask

    # ----------------------------------------------------------------------
    # Act: used during rollouts / viewer
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        agent=None,
        game_field=None,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """
        Decentralized actor inference for a single agent.

        Args:
            obs: [C, 40, 30] or [B, C, 40, 30] (per-agent observation s_i).
            agent: Agent instance (for action masking).
            game_field: GameField instance (for action masking).
            deterministic: If True, take argmax; otherwise sample.

        Returns:
            dict with:
              - "macro_action": LongTensor [B]
              - "target_action": LongTensor [B]
              - "log_prob": FloatTensor [B]
              - "value": FloatTensor [B]    (V(s_i))
              - "macro_mask": optional BoolTensor [n_macros]
        """
        # Actor forward
        macro_logits, target_logits, latent = self.forward_actor(obs)
        value = self.value_head(latent).squeeze(-1)

        macro_mask = None
        if agent is not None and game_field is not None:
            macro_mask = self.get_action_mask(agent, game_field)
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
            "macro_action": macro_action,
            "target_action": target_action,
            "log_prob": joint_log_prob,
            "value": value,
            "macro_mask": macro_mask,
        }

    # ----------------------------------------------------------------------
    # evaluate_actions: used by PPO update
    # ----------------------------------------------------------------------
    def evaluate_actions(
        self,
        obs_batch: torch.Tensor,
        macro_actions: torch.Tensor,
        target_actions: torch.Tensor,
    ):
        """
        Recompute log π(a|s) and V(s) for PPO on a batch.

        Args:
            obs_batch:      [B, C, 40, 30]   (per-agent observations)
            macro_actions:  [B]              (taken macro actions)
            target_actions: [B]              (taken target indices)

        Returns:
            log_probs: [B]
            entropy:  [B]
            values:   [B]
        """
        # Critic: V(s_i)
        values = self.forward_critic(obs_batch)

        # Actor: π(a_macro, a_target | s_i)
        macro_logits, target_logits, _ = self.forward_actor(obs_batch)

        macro_dist = Categorical(logits=macro_logits)
        target_dist = Categorical(logits=target_logits)

        log_probs_macro = macro_dist.log_prob(macro_actions)
        log_probs_target = target_dist.log_prob(target_actions)
        log_probs = log_probs_macro + log_probs_target

        entropy = macro_dist.entropy() + target_dist.entropy()

        return log_probs, entropy, values
