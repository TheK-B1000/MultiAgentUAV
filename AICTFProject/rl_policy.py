import math
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.distributions import Categorical

from obs_encoder import ObsEncoder
from macro_actions import MacroAction


class ActorCriticNet(nn.Module):
    """
    CNN-based Actor-Critic network for the UAV CTF domain.

    Modes:
      • Single-team PPO (current setup):
          - Actor: π(a_macro, a_target | s_i)
          - Critic: V(s_i) (local value)
          - Used by train_ppo_event.py via `evaluate_actions(obs, macro, target)`.

      • MAPPO / CTDE (optional):
          - Actor: same decentralized policy π(a_i | s_i)
          - Critic: centralized V(S), where S is the joint state of all N agents.
          - Exposed via `evaluate_actions_central(central_obs, actor_obs, ...)`.

    Obs layout:
      s_i: per-agent CNN observation [7, 40, 30].
    """

    def __init__(
        self,
        n_macros: int = len(MacroAction),
        n_targets: int = 50,
        latent_dim: int = 128,
        in_channels: int = 7,
        height: int = 40,
        width: int = 30,
        n_agents: int = 2,
        use_central_critic: bool = False,
    ):
        super().__init__()

        self.n_macros = n_macros
        self.n_targets = n_targets
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.n_agents = n_agents
        self.use_central_critic = use_central_critic

        # ------------------------------------------------------------------
        # Shared CNN encoder: [B, C, H, W] -> [B, latent_dim]
        # ------------------------------------------------------------------
        self.encoder = ObsEncoder(
            in_channels=in_channels,
            height=height,
            width=width,
            latent_dim=latent_dim,
        )

        # ------------------------------------------------------------------
        # Actor heads
        # ------------------------------------------------------------------
        self.actor_macro = nn.Linear(latent_dim, n_macros)
        self.actor_target = nn.Linear(latent_dim, n_targets)

        # ------------------------------------------------------------------
        # Local critic: V(s_i)
        # ------------------------------------------------------------------
        self.local_value_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        # ------------------------------------------------------------------
        # Centralized critic: V(S), where S = concat(latent_i, ..., latent_N)
        # Only used if use_central_critic=True
        # ------------------------------------------------------------------
        critic_input_dim = n_agents * latent_dim
        self.central_value_head = nn.Sequential(
            nn.Linear(critic_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self._init_heads()

    # ----------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------
    def _init_heads(self) -> None:
        nn.init.orthogonal_(self.actor_macro.weight, gain=0.01)
        nn.init.constant_(self.actor_macro.bias, 0.0)

        nn.init.orthogonal_(self.actor_target.weight, gain=0.01)
        nn.init.constant_(self.actor_target.bias, 0.0)

        last_local = self.local_value_head[-1]
        if isinstance(last_local, nn.Linear):
            nn.init.orthogonal_(last_local.weight, gain=1.0)
            nn.init.constant_(last_local.bias, 0.0)

        last_central = self.central_value_head[-1]
        if isinstance(last_central, nn.Linear):
            nn.init.orthogonal_(last_central.weight, gain=1.0)
            nn.init.constant_(last_central.bias, 0.0)

    # ----------------------------------------------------------------------
    # Encoders / forwards
    # ----------------------------------------------------------------------
    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [C, H, W] or [B, C, H, W]
        returns latent: [B, latent_dim]
        """
        return self.encoder(obs)

    def forward_actor(self, obs: torch.Tensor):
        """
        obs: [C, 40, 30] or [B, C, 40, 30]
        returns:
          macro_logits:  [B, n_macros]
          target_logits: [B, n_targets]
          latent:        [B, latent_dim]
        """
        latent = self._encode(obs)
        macro_logits = self.actor_macro(latent)
        target_logits = self.actor_target(latent)
        return macro_logits, target_logits, latent

    def forward_local_critic(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Local critic V(s_i).

        obs: [B, C, 40, 30]
        returns values: [B]
        """
        latent = self._encode(obs)
        value = self.local_value_head(latent).squeeze(-1)
        return value

    def forward_central_critic(self, central_obs: torch.Tensor) -> torch.Tensor:
        """
        Centralized critic V(S) for MAPPO.

        Args:
            central_obs: [B, N_agents, C, 40, 30]
                         or [B * N_agents, C, 40, 30] if pre-flattened.

        Returns:
            values: [B]
        """
        if central_obs.dim() == 5:
            # [B, N, C, H, W] -> [B*N, C, H, W]
            B, N, C, H, W = central_obs.shape
            assert N == self.n_agents, f"Expected n_agents={self.n_agents}, got {N}"
            flat = central_obs.view(B * N, C, H, W)
        else:
            # Assume already [B*N, C, H, W]
            flat = central_obs
            B = flat.size(0) // self.n_agents

        # Encode all agents
        latent_flat = self._encode(flat)                      # [B*N, latent_dim]
        latent_joint = latent_flat.view(B, self.n_agents * self.latent_dim)
        value = self.central_value_head(latent_joint).squeeze(-1)  # [B]
        return value

    # ----------------------------------------------------------------------
    # Action masking
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def get_action_mask(self, agent, game_field) -> torch.Tensor:
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
    # act(): used in rollouts / viewer (unchanged semantics)
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
        Decentralized actor inference for one agent.

        obs: [C, 40, 30] or [B, C, 40, 30]
        """
        macro_logits, target_logits, latent = self.forward_actor(obs)
        local_value = self.local_value_head(latent).squeeze(-1)

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
            "value": local_value,      # V(s_i)
            "macro_mask": macro_mask,
        }

    # ----------------------------------------------------------------------
    # BACKWARDS-COMPATIBLE evaluate_actions (local critic)
    # ----------------------------------------------------------------------
    def evaluate_actions(
        self,
        obs_batch: torch.Tensor,
        macro_actions: torch.Tensor,
        target_actions: torch.Tensor,
    ):
        """
        Backwards-compatible API for train_ppo_event.py.

        Args:
            obs_batch:     [B, C, 40, 30]  (local s_i)
            macro_actions: [B]
            target_actions:[B]

        Returns:
            log_probs: [B]
            entropy:   [B]
            values:    [B] = V(s_i)
        """
        values = self.forward_local_critic(obs_batch)

        macro_logits, target_logits, _ = self.forward_actor(obs_batch)
        macro_dist = Categorical(logits=macro_logits)
        target_dist = Categorical(logits=target_logits)

        log_probs_macro = macro_dist.log_prob(macro_actions)
        log_probs_target = target_dist.log_prob(target_actions)
        log_probs = log_probs_macro + log_probs_target

        entropy = macro_dist.entropy() + target_dist.entropy()

        return log_probs, entropy, values

    # ----------------------------------------------------------------------
    # NEW: MAPPO-style evaluate_actions with CENTRAL critic V(S)
    # ----------------------------------------------------------------------
    def evaluate_actions_central(
        self,
        central_obs_batch: torch.Tensor,
        actor_obs_batch: torch.Tensor,
        macro_actions: torch.Tensor,
        target_actions: torch.Tensor,
    ):
        """
        MAPPO / CTDE API.

        Args:
            central_obs_batch: [B, N_agents, C, 40, 30]
            actor_obs_batch:   [B, C, 40, 30]  (this agent's s_i)
            macro_actions:     [B]
            target_actions:    [B]

        Returns:
            log_probs: [B]
            entropy:   [B]
            values:    [B] = V(S)
        """
        # 1. Centralized critic V(S)
        values = self.forward_central_critic(central_obs_batch)

        # 2. Decentralized actor π(a_i | s_i)
        macro_logits, target_logits, _ = self.forward_actor(actor_obs_batch)
        macro_dist = Categorical(logits=macro_logits)
        target_dist = Categorical(logits=target_logits)

        log_probs_macro = macro_dist.log_prob(macro_actions)
        log_probs_target = target_dist.log_prob(target_actions)
        log_probs = log_probs_macro + log_probs_target

        entropy = macro_dist.entropy() + target_dist.entropy()

        return log_probs, entropy, values
