import math
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.distributions import Categorical

from obs_encoder import ObsEncoder
from macro_actions import MacroAction


class ActorCriticNet(nn.Module):
    """
    MAPPO (CTDE) Actor-Critic network.

    - Actor (Policy) is decentralized: input s_i (individual observation).
    - Critic (Value) is centralized: input S (concatenated observations).
    """

    def __init__(
            self,
            n_agents: int = 2,
            n_macros: int = len(MacroAction),
            n_targets: int = 50,
            latent_dim: int = 128,
    ):
        super().__init__()

        self.n_agents = n_agents  # Number of controllable agents (e.g., 2 for blue team)
        self.n_macros = n_macros
        self.n_targets = n_targets
        self.latent_dim = latent_dim

        # --- 1. Shared Decentralized Encoder (s_i -> latent) ---
        # This CNN takes one agent's observation [7, 20, 20] and outputs [latent_dim]
        self.actor_encoder = ObsEncoder(
            in_channels=7,
            height=20,
            width=20,
            latent_dim=latent_dim,
        )

        # --- 2. Decentralized Actor Heads (uses latent from actor_encoder) ---
        self.actor_macro = nn.Linear(latent_dim, n_macros)
        self.actor_target = nn.Linear(latent_dim, n_targets)

        # --- 3. Centralized Critic Head (V(S)) ---
        # The centralized state S is the concatenation of all N_agents * latent_dim
        critic_input_dim = n_agents * latent_dim
        self.critic_head = nn.Sequential(
            nn.Linear(critic_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self._init_heads()

    def _init_heads(self):
        # Orthogonal init for actor/critic with appropriate gains
        nn.init.orthogonal_(self.actor_macro.weight, gain=0.01)
        nn.init.constant_(self.actor_macro.bias, 0.0)

        nn.init.orthogonal_(self.actor_target.weight, gain=0.01)
        nn.init.constant_(self.actor_target.bias, 0.0)

        # Use 1.0 gain for the final critic layer
        if isinstance(self.critic_head[-1], nn.Linear):
            nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)
            nn.init.constant_(self.critic_head[-1].bias, 0.0)

    def forward_actor(self, obs_i: torch.Tensor):
        """
        Actor forward pass for a single agent's observation s_i.
        obs_i: [B, 7, 20, 20] (or [7, 20, 20])
        """
        latent = self.actor_encoder(obs_i)  # [B, latent_dim]
        macro_logits = self.actor_macro(latent)
        target_logits = self.actor_target(latent)
        return macro_logits, target_logits

    def forward_critic(self, obs_batch: torch.Tensor):
        """
        Critic forward pass for the centralized state S.
        obs_batch: [B, N_agents * 7, 20, 20]

        The CTDE implementation requires processing a batch where
        each row is the concatenated observation of all N_agents.
        """
        B, C_total, H, W = obs_batch.shape

        # Reshape the batch for independent processing by the actor_encoder
        # [B, N*C_i, H, W] -> [B*N, C_i, H, W]
        obs_i_flat = obs_batch.view(B * self.n_agents, 7, H, W)

        # Pass all individual observations through the *same* encoder
        latent_flat = self.actor_encoder(obs_i_flat)  # [B*N, latent_dim]

        # Reshape back to the centralized format: concatenate latents
        # [B*N, latent_dim] -> [B, N, latent_dim] -> [B, N*latent_dim]
        latent_concat = latent_flat.view(B, self.n_agents * self.latent_dim)

        # Critic predicts joint value V(S)
        value = self.critic_head(latent_concat).squeeze(-1)  # [B]

        return value

    # --- Utility methods (act, get_action_mask, evaluate_actions) ---
    # These remain similar but MUST use `forward_actor`

    @torch.no_grad()
    def get_action_mask(self, agent, game_field) -> torch.Tensor:
        # NOTE: This implementation is identical to the original
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
        Decentralized Actor inference: computes action and value for one agent.

        obs: [7, 20, 20] or [B, 7, 20, 20] (s_i)

        Note: The value function (V) returned here is V(s_i), as the full
              centralized state S is not available during decentralized execution.
              This V(s_i) is often used as a *proxy* during execution only.
        """
        # Actor forward pass (Decentralized)
        macro_logits, target_logits = self.forward_actor(obs)

        # NOTE: Fallback to decentralized V(s_i) for execution-time monitoring
        latent = self.actor_encoder(obs)
        value = self.critic_head(latent.repeat(1, self.n_agents)).squeeze(-1)

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

    def evaluate_actions(
            self,
            central_obs_batch: torch.Tensor,
            macro_actions: torch.Tensor,
            target_actions: torch.Tensor,
            actor_obs: torch.Tensor,
    ):
        """
        Computes joint log_prob (for Actor update) and joint value (for Critic update).

        central_obs_batch: [B, N_agents * 7, 20, 20] (Centralized State S)
        macro_actions:     [B] (Single agent's macro action)
        target_actions:    [B] (Single agent's target action)
        actor_obs:         [B, 7, 20, 20] (Single agent's observation s_i)
        """
        # --- 1. Critic Update: V(S) ---
        values = self.forward_critic(central_obs_batch)

        # --- 2. Actor Update: pi(a_i | s_i) ---
        # The actor uses the decentralized observation (s_i)
        macro_logits, target_logits = self.forward_actor(actor_obs)

        # Calculate joint log_prob
        macro_dist = Categorical(logits=macro_logits)
        target_dist = Categorical(logits=target_logits)

        log_probs_macro = macro_dist.log_prob(macro_actions)
        log_probs_target = target_dist.log_prob(target_actions)
        log_probs = log_probs_macro + log_probs_target

        # Sum entropies: H(m) + H(t)
        entropy = macro_dist.entropy() + target_dist.entropy()

        return log_probs, entropy, values