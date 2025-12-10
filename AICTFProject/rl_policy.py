"""
rl_policy.py

CNN-based Actor-Critic network for the 2-vs-2 UAV CTF domain.

Features:
  - Shared ObsEncoder over 7×40×30 spatial observations.
  - Decentralized actor: π(a_i | s_i) over macro-actions and macro-targets.
  - Local critic: V(s_i) for single-agent PPO baselines.
  - Central critic: V(S) for MAPPO (CTDE), where S stacks all N_agents' obs.
  - MAPPOBuffer for event-driven MAPPO training (central_obs + actor_obs).

This file is used by:
  - train_ppo_event.py     (via ActorCriticNet.evaluate_actions for PPO)
  - train_mappo_event.py   (via MAPPOBuffer + evaluate_actions_central for MAPPO)
  - ctfviewer.py           (via ActorCriticNet.act for runtime control/viewing)
"""

from typing import Dict, Any, List, Tuple, Optional

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from obs_encoder import ObsEncoder
from macro_actions import MacroAction


# ======================================================================
# MAPPO ROLLOUT BUFFER (CENTRALIZED CRITIC, DECENTRALIZED ACTOR)
# ======================================================================


class MAPPOBuffer:
    """
    Rollout buffer for MAPPO with centralized critic (CTDE).

    Each entry corresponds to ONE agent at ONE macro-decision time j:
      - actor_obs_j:   s_i(t_j)      [C, H, W]       (this agent's local obs)
      - central_obs_j: S(t_j)        [N, C, H, W]    (joint obs for all N agents)
      - macro_action:  macro-action index
      - target_action: macro-target index
      - log_prob:      log π(a_i | s_i)
      - value:         V(S)          (central critic value for this step)
      - reward:        event-aggregated reward for this agent at step j
      - done:          episode termination flag
      - dt:            Δt_j in continuous-time GAE
    """

    def __init__(self) -> None:
        self.actor_obs: List[np.ndarray] = []
        self.central_obs: List[np.ndarray] = []
        self.macro_actions: List[int] = []
        self.target_actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.dts: List[float] = []

    def add(
        self,
        actor_obs: np.ndarray,      # [C, H, W] (individual s_i)
        central_obs: np.ndarray,    # [N_agents, C, H, W] (joint S)
        macro_action: int,
        target_action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        dt: float,
    ) -> None:
        self.actor_obs.append(actor_obs.astype(np.float32))
        self.central_obs.append(central_obs.astype(np.float32))
        self.macro_actions.append(int(macro_action))
        self.target_actions.append(int(target_action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.dts.append(float(dt))

    def size(self) -> int:
        return len(self.actor_obs)

    def clear(self) -> None:
        self.__init__()

    def to_tensors(self, device: torch.device) -> Tuple[torch.Tensor, ...]:
        """
        Returns tensors on the given device:

          actor_obs   : [T, C, H, W]
          central_obs : [T, N_agents, C, H, W]
          macro_actions, target_actions, log_probs, values, rewards, dones, dts: [T]
        """
        actor_obs = torch.tensor(
            np.stack(self.actor_obs), dtype=torch.float32, device=device
        )  # [T, C, H, W]
        central_obs = torch.tensor(
            np.stack(self.central_obs), dtype=torch.float32, device=device
        )  # [T, N, C, H, W]

        macro_actions = torch.tensor(
            self.macro_actions, dtype=torch.long, device=device
        )
        target_actions = torch.tensor(
            self.target_actions, dtype=torch.long, device=device
        )
        log_probs = torch.tensor(
            self.log_probs, dtype=torch.float32, device=device
        )
        values = torch.tensor(
            self.values, dtype=torch.float32, device=device
        )
        rewards = torch.tensor(
            np.array(self.rewards), dtype=torch.float32, device=device
        )
        dones = torch.tensor(
            np.array(self.dones), dtype=torch.float32, device=device
        )
        dts = torch.tensor(
            np.array(self.dts), dtype=torch.float32, device=device
        )

        return (
            actor_obs,
            central_obs,
            macro_actions,
            target_actions,
            log_probs,
            values,
            rewards,
            dones,
            dts,
        )


# ======================================================================
# CNN ACTOR-CRITIC NETWORK (PPO + MAPPO), DIRECTML-FRIENDLY
# ======================================================================


class ActorCriticNet(nn.Module):
    """
    CNN-based Actor-Critic network for the UAV CTF domain.

    Supports:
      - Single-agent PPO (local critic V(s_i)) via evaluate_actions().
      - MAPPO / CTDE (central critic V(S)) via evaluate_actions_central().

    Implementation details (DirectML-friendly):
      - Avoids torch.distributions.Categorical in the training path.
      - Computes log-probs and entropy manually from logits via log_softmax.
      - Uses mask-based selection instead of gather/scatter in backward.
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
    ) -> None:
        super().__init__()

        self.n_macros = n_macros
        self.n_targets = n_targets
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.n_agents = n_agents
        self.use_central_critic = use_central_critic

        # --------------------------------------------------------------
        # Shared CNN encoder: [B, C, H, W] -> [B, latent_dim]
        # --------------------------------------------------------------
        self.encoder = ObsEncoder(
            in_channels=in_channels,
            height=height,
            width=width,
            latent_dim=latent_dim,
        )

        # --------------------------------------------------------------
        # Actor heads: π(a_i | s_i)
        # --------------------------------------------------------------
        self.actor_macro = nn.Linear(latent_dim, n_macros)
        self.actor_target = nn.Linear(latent_dim, n_targets)

        # --------------------------------------------------------------
        # Local critic: V(s_i)  (used by PPO and by act() at runtime)
        # --------------------------------------------------------------
        self.local_value_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        # --------------------------------------------------------------
        # Centralized critic: V(S) (MAPPO / CTDE)
        # S is concatenation of N_agents latent vectors.
        # --------------------------------------------------------------
        critic_input_dim = n_agents * latent_dim
        self.central_value_head = nn.Sequential(
            nn.Linear(critic_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self._init_heads()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _init_heads(self) -> None:
        # Actor: small init for stable logits
        nn.init.orthogonal_(self.actor_macro.weight, gain=0.01)
        nn.init.constant_(self.actor_macro.bias, 0.0)

        nn.init.orthogonal_(self.actor_target.weight, gain=0.01)
        nn.init.constant_(self.actor_target.bias, 0.0)

        # Local critic final layer
        last_local = self.local_value_head[-1]
        if isinstance(last_local, nn.Linear):
            nn.init.orthogonal_(last_local.weight, gain=1.0)
            nn.init.constant_(last_local.bias, 0.0)

        # Central critic final layer
        last_central = self.central_value_head[-1]
        if isinstance(last_central, nn.Linear):
            nn.init.orthogonal_(last_central.weight, gain=1.0)
            nn.init.constant_(last_central.bias, 0.0)

    # ------------------------------------------------------------------
    # Encoders / forwards
    # ------------------------------------------------------------------
    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [C, H, W] or [B, C, H, W]
        returns latent: [B, latent_dim]
        """
        return self.encoder(obs)

    def forward_actor(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decentralized forward pass for the actor.

        Args:
            obs: [C, 40, 30] or [B, C, 40, 30]

        Returns:
            macro_logits: [B, n_macros]
            target_logits: [B, n_targets]
            latent: [B, latent_dim]
        """
        latent = self._encode(obs)
        macro_logits = self.actor_macro(latent)
        target_logits = self.actor_target(latent)
        return macro_logits, target_logits, latent

    def forward_local_critic(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Local critic V(s_i) for PPO / runtime diagnostics.

        Args:
            obs: [C, H, W] or [B, C, H, W]
        Returns:
            value: [B]
        """
        latent = self._encode(obs)
        value = self.local_value_head(latent).squeeze(-1)
        return value

    def forward_central_critic(self, central_obs: torch.Tensor) -> torch.Tensor:
        """
        Centralized critic V(S) for MAPPO.

        Args:
            central_obs:
                - Preferred: [B, N_agents, C, H, W]
                - Also allowed: [B*N_agents, C, H, W] (flattened)

        Returns:
            value: [B]
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

        latent_flat = self._encode(flat)  # [B*N, latent_dim]
        latent_joint = latent_flat.view(B, self.n_agents * self.latent_dim)
        value = self.central_value_head(latent_joint).squeeze(-1)  # [B]
        return value

    # ------------------------------------------------------------------
    # Action masking (macro-level feasibility)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_action_mask(self, agent, game_field) -> torch.Tensor:
        """
        Returns a boolean mask over macro-actions indicating which ones are legal.

        Current constraints:
          - Can't PLACE_MINE if agent.mine_charges <= 0
          - Can't GRAB_MINE if no friendly pickup within ~3 cells
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

    # ------------------------------------------------------------------
    # act(): used during rollouts / viewer (Decentralized Execution)
    # ------------------------------------------------------------------
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

        Args:
            obs: [C, H, W] or [B, C, H, W]
            agent, game_field: optional, for action masking.
            deterministic: if True, picks argmax instead of sampling.

        Returns:
            {
                "macro_action":  [B] LongTensor,
                "target_action": [B] LongTensor,
                "log_prob":      [B] float tensor of joint log-prob,
                "value":         [B] local V(s_i),
                "macro_mask":    [n_macros] bool tensor or None,
            }
        """
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)  # [1,C,H,W]

        macro_logits, target_logits, latent = self.forward_actor(obs)
        local_value = self.local_value_head(latent).squeeze(-1)  # [B]
        B = macro_logits.size(0)
        device = macro_logits.device

        # Optional action mask for macro-actions
        macro_mask = None
        if agent is not None and game_field is not None:
            macro_mask = self.get_action_mask(agent, game_field)
            if macro_logits.dim() == 2:
                macro_logits = macro_logits.masked_fill(
                    ~macro_mask.unsqueeze(0), -1e10
                )
            else:
                macro_logits = macro_logits.masked_fill(~macro_mask, -1e10)

        # Log-softmax for numeric stability
        macro_log_probs_all = F.log_softmax(macro_logits, dim=-1)   # [B,n_macros]
        target_log_probs_all = F.log_softmax(target_logits, dim=-1) # [B,n_targets]

        macro_probs = macro_log_probs_all.exp()    # [B,n_macros]
        target_probs = target_log_probs_all.exp()  # [B,n_targets]

        if deterministic:
            macro_action = macro_logits.argmax(dim=-1)   # [B]
            target_action = target_logits.argmax(dim=-1) # [B]
        else:
            # Sample on CPU to avoid any exotic DML ops.
            macro_action_cpu = torch.multinomial(macro_probs.cpu(), 1)   # [B,1]
            target_action_cpu = torch.multinomial(target_probs.cpu(), 1) # [B,1]

            macro_action = macro_action_cpu.to(device).view(B)   # [B]
            target_action = target_action_cpu.to(device).view(B) # [B]

        # Compute log_prob of the joint action via masks (no gather)
        macro_actions_view = macro_action.view(B, 1)
        target_actions_view = target_action.view(B, 1)

        macro_mask_sel = (
            torch.arange(self.n_macros, device=device)
            .view(1, self.n_macros)
            .expand(B, self.n_macros)
            == macro_actions_view
        ).to(macro_logits.dtype)

        target_mask_sel = (
            torch.arange(self.n_targets, device=device)
            .view(1, self.n_targets)
            .expand(B, self.n_targets)
            == target_actions_view
        ).to(target_logits.dtype)

        macro_log_prob = (macro_log_probs_all * macro_mask_sel).sum(dim=-1)    # [B]
        target_log_prob = (target_log_probs_all * target_mask_sel).sum(dim=-1) # [B]
        log_prob = macro_log_prob + target_log_prob                            # [B]

        return {
            "macro_action": macro_action,
            "target_action": target_action,
            "log_prob": log_prob,
            "value": local_value,
            "macro_mask": macro_mask,
        }

    # ------------------------------------------------------------------
    # BACKWARDS-COMPATIBLE evaluate_actions (PPO training, local critic)
    # ------------------------------------------------------------------
    def evaluate_actions(
        self,
        obs_batch: torch.Tensor,
        macro_actions: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        PPO-style evaluation with a local critic V(s_i).

        Args:
            obs_batch:     [B, C, H, W]
            macro_actions: [B]
            target_actions:[B]

        Returns:
            log_probs: [B]
            entropy:  [B]
            values:   [B]  (local V(s_i))
        """
        if obs_batch.dim() == 3:
            obs_batch = obs_batch.unsqueeze(0)

        B = obs_batch.size(0)
        device = obs_batch.device

        # Critic
        values = self.forward_local_critic(obs_batch)  # [B]

        # Actor
        macro_logits, target_logits, _ = self.forward_actor(obs_batch)
        macro_log_probs_all = F.log_softmax(macro_logits, dim=-1)   # [B,n_macros]
        target_log_probs_all = F.log_softmax(target_logits, dim=-1) # [B,n_targets]

        macro_probs = macro_log_probs_all.exp()    # [B,n_macros]
        target_probs = target_log_probs_all.exp()  # [B,n_targets]

        macro_actions = macro_actions.view(B, 1)
        target_actions = target_actions.view(B, 1)

        macro_mask_sel = (
            torch.arange(self.n_macros, device=device)
            .view(1, self.n_macros)
            .expand(B, self.n_macros)
            == macro_actions
        ).to(macro_logits.dtype)

        target_mask_sel = (
            torch.arange(self.n_targets, device=device)
            .view(1, self.n_targets)
            .expand(B, self.n_targets)
            == target_actions
        ).to(target_logits.dtype)

        macro_log_prob = (macro_log_probs_all * macro_mask_sel).sum(dim=-1)    # [B]
        target_log_prob = (target_log_probs_all * target_mask_sel).sum(dim=-1) # [B]
        log_probs = macro_log_prob + target_log_prob                           # [B]

        # Entropy H(p) = -Σ p * log p
        macro_entropy = -(macro_probs * macro_log_probs_all).sum(dim=-1)       # [B]
        target_entropy = -(target_probs * target_log_probs_all).sum(dim=-1)    # [B]
        entropy = macro_entropy + target_entropy                               # [B]

        return log_probs, entropy, values

    # ------------------------------------------------------------------
    # MAPPO evaluate_actions_central (CTDE training, central critic)
    # ------------------------------------------------------------------
    def evaluate_actions_central(
        self,
        central_obs_batch: torch.Tensor,
        actor_obs_batch: torch.Tensor,
        macro_actions: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MAPPO / CTDE training API.

        Critic: V(S) where S is the joint state (central_obs_batch).
        Actor:  π(a_i | s_i) where s_i is this agent's local observation.

        Args:
            central_obs_batch: [B, N_agents, C, H, W]
            actor_obs_batch:   [B, C, H, W]
            macro_actions:     [B]
            target_actions:    [B]

        Returns:
            log_probs: [B]
            entropy:  [B]
            values:   [B]  (central V(S))
        """
        if actor_obs_batch.dim() == 3:
            actor_obs_batch = actor_obs_batch.unsqueeze(0)

        B = actor_obs_batch.size(0)
        device = actor_obs_batch.device

        # 1. Central critic V(S)
        values = self.forward_central_critic(central_obs_batch)  # [B]

        # 2. Decentralized actor π(a_i | s_i)
        macro_logits, target_logits, _ = self.forward_actor(actor_obs_batch)
        macro_log_probs_all = F.log_softmax(macro_logits, dim=-1)   # [B,n_macros]
        target_log_probs_all = F.log_softmax(target_logits, dim=-1) # [B,n_targets]

        macro_probs = macro_log_probs_all.exp()
        target_probs = target_log_probs_all.exp()

        macro_actions = macro_actions.view(B, 1)
        target_actions = target_actions.view(B, 1)

        macro_mask_sel = (
            torch.arange(self.n_macros, device=device)
            .view(1, self.n_macros)
            .expand(B, self.n_macros)
            == macro_actions
        ).to(macro_logits.dtype)

        target_mask_sel = (
            torch.arange(self.n_targets, device=device)
            .view(1, self.n_targets)
            .expand(B, self.n_targets)
            == target_actions
        ).to(target_logits.dtype)

        macro_log_prob = (macro_log_probs_all * macro_mask_sel).sum(dim=-1)    # [B]
        target_log_prob = (target_log_probs_all * target_mask_sel).sum(dim=-1) # [B]
        log_probs = macro_log_prob + target_log_prob                           # [B]

        macro_entropy = -(macro_probs * macro_log_probs_all).sum(dim=-1)
        target_entropy = -(target_probs * target_log_probs_all).sum(dim=-1)
        entropy = macro_entropy + target_entropy

        return log_probs, entropy, values
