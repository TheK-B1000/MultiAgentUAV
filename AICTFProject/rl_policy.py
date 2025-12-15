# rl_policy.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from obs_encoder import ObsEncoder
from macro_actions import MacroAction


# ======================================================================
# MAPPO BUFFER (CENTRAL OBS + ACTOR OBS)
# ======================================================================

class MAPPOBuffer:
    """
    Each entry corresponds to ONE agent at ONE macro decision:
      actor_obs    : [C,H,W]
      central_obs  : [N,C,H,W] (joint obs, stacked in fixed order)
      macro_action, target_action
      log_prob
      value        : central critic value V(S)
      reward, done, dt
      traj_id      : for correct per-agent GAE when samples are interleaved

    Optional but strongly recommended for PPO correctness when masking:
      macro_mask   : [n_macros] bool mask used when sampling the macro action
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
        self.traj_ids: List[int] = []
        self.macro_masks: List[np.ndarray] = []

    def add(
        self,
        actor_obs: np.ndarray,
        central_obs: np.ndarray,
        macro_action: int,
        target_action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        dt: float,
        traj_id: int,
        macro_mask: Optional[np.ndarray] = None,
    ) -> None:
        self.actor_obs.append(np.array(actor_obs, dtype=np.float32))
        self.central_obs.append(np.array(central_obs, dtype=np.float32))
        self.macro_actions.append(int(macro_action))
        self.target_actions.append(int(target_action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.dts.append(float(dt))
        self.traj_ids.append(int(traj_id))

        if macro_mask is None:
            self.macro_masks.append(np.array([], dtype=np.bool_))
        else:
            self.macro_masks.append(np.array(macro_mask, dtype=np.bool_))

    def size(self) -> int:
        return len(self.actor_obs)

    def clear(self) -> None:
        self.__init__()

    def to_tensors(self, device: torch.device) -> Tuple[torch.Tensor, ...]:
        actor_obs = torch.tensor(np.stack(self.actor_obs), dtype=torch.float32, device=device)     # [T,C,H,W]
        central_obs = torch.tensor(np.stack(self.central_obs), dtype=torch.float32, device=device) # [T,N,C,H,W]
        macro_actions = torch.tensor(self.macro_actions, dtype=torch.long, device=device)          # [T]
        target_actions = torch.tensor(self.target_actions, dtype=torch.long, device=device)        # [T]
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)               # [T]
        values = torch.tensor(self.values, dtype=torch.float32, device=device)                     # [T]
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)                   # [T]
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)                       # [T]
        dts = torch.tensor(self.dts, dtype=torch.float32, device=device)                           # [T]
        traj_ids = torch.tensor(self.traj_ids, dtype=torch.long, device=device)                    # [T]

        if len(self.macro_masks) > 0 and self.macro_masks[0].size > 0:
            macro_masks = torch.tensor(np.stack(self.macro_masks), dtype=torch.bool, device=device) # [T,n_macros]
        else:
            macro_masks = torch.empty((actor_obs.size(0), 0), dtype=torch.bool, device=device)      # [T,0]

        return (
            actor_obs, central_obs, macro_actions, target_actions,
            log_probs, values, rewards, dones, dts, traj_ids, macro_masks
        )


# ======================================================================
# ACTOR-CRITIC NET (PPO local + MAPPO central)
# ======================================================================

class ActorCriticNet(nn.Module):
    """
    Shared encoder. Decentralized actor heads.
    Local critic (PPO) + Central critic (MAPPO/CTDE).

    Observations:
      [B, C, H, W] = [B, 7, 40, 30]
    """

    def __init__(
        self,
        n_macros: int = 5,
        n_targets: int = 50,
        latent_dim: int = 128,
        in_channels: int = 7,
        height: int = 40,
        width: int = 30,
        n_agents: int = 2,
        used_macros: Optional[List[MacroAction]] = None,
    ) -> None:
        super().__init__()
        self.n_macros = int(n_macros)
        self.n_targets = int(n_targets)
        self.latent_dim = int(latent_dim)
        self.in_channels = int(in_channels)
        self.height = int(height)
        self.width = int(width)
        self.n_agents = int(n_agents)

        # IMPORTANT: defines the meaning of macro index 0..n_macros-1
        if used_macros is None:
            # default: assume your training uses these 5 in this order
            used_macros = [
                MacroAction.GO_TO,
                MacroAction.GRAB_MINE,
                MacroAction.GET_FLAG,
                MacroAction.PLACE_MINE,
                MacroAction.GO_HOME,
            ]
        self.used_macros: List[MacroAction] = list(used_macros)
        if len(self.used_macros) != self.n_macros:
            raise ValueError(f"used_macros length {len(self.used_macros)} != n_macros {self.n_macros}")

        # map MacroAction -> actor index
        self._macro_to_index: Dict[MacroAction, int] = {m: i for i, m in enumerate(self.used_macros)}

        self.encoder = ObsEncoder(
            in_channels=self.in_channels,
            height=self.height,
            width=self.width,
            latent_dim=self.latent_dim,
        )

        self.actor_macro = nn.Linear(self.latent_dim, self.n_macros)
        self.actor_target = nn.Linear(self.latent_dim, self.n_targets)

        self.local_value_head = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        critic_in = self.n_agents * self.latent_dim
        self.central_value_head = nn.Sequential(
            nn.Linear(critic_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self._init_heads()

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

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def forward_actor(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self._encode(obs)
        macro_logits = self.actor_macro(latent)
        target_logits = self.actor_target(latent)
        return macro_logits, target_logits, latent

    def forward_local_critic(self, obs: torch.Tensor) -> torch.Tensor:
        latent = self._encode(obs)
        return self.local_value_head(latent).squeeze(-1)

    def forward_central_critic(self, central_obs: torch.Tensor) -> torch.Tensor:
        if central_obs.dim() != 5:
            raise ValueError(f"Expected [B,N,C,H,W], got {tuple(central_obs.shape)}")
        B, N, C, H, W = central_obs.shape
        if N != self.n_agents:
            raise ValueError(f"Expected N={self.n_agents}, got {N}")
        flat = central_obs.reshape(B * N, C, H, W)
        latent_flat = self._encode(flat)
        latent_joint = latent_flat.reshape(B, N * self.latent_dim)
        return self.central_value_head(latent_joint).squeeze(-1)

    def _idx(self, macro: MacroAction) -> Optional[int]:
        return self._macro_to_index.get(macro, None)

    @torch.no_grad()
    def get_action_mask(self, agent, game_field) -> torch.Tensor:
        """
        Macro feasibility mask. Returns: [n_macros] bool on policy device.

        Priority:
          1) If env provides get_macro_mask(agent), use it (single source of truth).
          2) Otherwise apply lightweight, safe gating here.
        """
        device = self._device()

        # 1) env-defined mask (best)
        if game_field is not None and hasattr(game_field, "get_macro_mask"):
            try:
                m = game_field.get_macro_mask(agent)
                m = torch.as_tensor(m, dtype=torch.bool, device=device)
                if m.numel() == self.n_macros:
                    if not m.any():
                        m[:] = True
                    return m
            except Exception:
                pass

        mask = torch.ones(self.n_macros, dtype=torch.bool, device=device)

        # Carry logic: if carrying, GET_FLAG is nonsense
        carrying = bool(getattr(agent, "isCarryingFlag", lambda: False)())
        if carrying:
            idx = self._idx(MacroAction.GET_FLAG)
            if idx is not None:
                mask[idx] = False

        # PLACE_MINE requires charges
        if int(getattr(agent, "mine_charges", 0)) <= 0:
            idx = self._idx(MacroAction.PLACE_MINE)
            if idx is not None:
                mask[idx] = False

        # GRAB_MINE only useless if (a) no friendly pickups exist OR (b) already full
        max_charges = int(getattr(game_field, "max_mine_charges_per_agent", 2)) if game_field is not None else 2
        if int(getattr(agent, "mine_charges", 0)) >= max_charges:
            idx = self._idx(MacroAction.GRAB_MINE)
            if idx is not None:
                mask[idx] = False
        else:
            any_friendly_pickup = False
            for p in getattr(game_field, "mine_pickups", []) if game_field is not None else []:
                if getattr(p, "owner_side", None) == getattr(agent, "side", None):
                    any_friendly_pickup = True
                    break
            if not any_friendly_pickup:
                idx = self._idx(MacroAction.GRAB_MINE)
                if idx is not None:
                    mask[idx] = False

        if not mask.any():
            mask[:] = True
        return mask

    @torch.no_grad()
    def act(self, obs: torch.Tensor, agent=None, game_field=None, deterministic: bool = False) -> Dict[str, Any]:
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)

        device = self._device()
        obs = obs.to(device).float()

        macro_logits, target_logits, latent = self.forward_actor(obs)
        local_value = self.local_value_head(latent).squeeze(-1)

        macro_mask = None
        if agent is not None and game_field is not None:
            macro_mask = self.get_action_mask(agent, game_field)  # [n_macros]
            macro_logits = macro_logits.masked_fill(~macro_mask.unsqueeze(0), -1e10)

        macro_dist = torch.distributions.Categorical(logits=macro_logits)
        target_dist = torch.distributions.Categorical(logits=target_logits)

        if deterministic:
            macro_action = macro_logits.argmax(dim=-1)
            target_action = target_logits.argmax(dim=-1)
        else:
            macro_action = macro_dist.sample()
            target_action = target_dist.sample()

        log_prob = macro_dist.log_prob(macro_action) + target_dist.log_prob(target_action)

        return {
            "macro_action": macro_action,
            "target_action": target_action,
            "log_prob": log_prob,
            "value": local_value,
            "macro_mask": macro_mask,
        }

    @torch.no_grad()
    def act_mappo(
        self,
        actor_obs: torch.Tensor,
        central_obs: torch.Tensor,
        agent=None,
        game_field=None,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        if actor_obs.dim() == 3:
            actor_obs = actor_obs.unsqueeze(0)
        if central_obs.dim() == 4:
            central_obs = central_obs.unsqueeze(0)

        device = self._device()
        actor_obs = actor_obs.to(device).float()
        central_obs = central_obs.to(device).float()

        macro_logits, target_logits, _ = self.forward_actor(actor_obs)
        central_value = self.forward_central_critic(central_obs)

        macro_mask = None
        if agent is not None and game_field is not None:
            macro_mask = self.get_action_mask(agent, game_field)
            macro_logits = macro_logits.masked_fill(~macro_mask.unsqueeze(0), -1e10)

        macro_dist = torch.distributions.Categorical(logits=macro_logits)
        target_dist = torch.distributions.Categorical(logits=target_logits)

        if deterministic:
            macro_action = macro_logits.argmax(dim=-1)
            target_action = target_logits.argmax(dim=-1)
        else:
            macro_action = macro_dist.sample()
            target_action = target_dist.sample()

        log_prob = macro_dist.log_prob(macro_action) + target_dist.log_prob(target_action)

        return {
            "macro_action": macro_action,
            "target_action": target_action,
            "log_prob": log_prob,
            "value": central_value,
            "macro_mask": macro_mask,
        }

    def evaluate_actions(
        self,
        obs_batch: torch.Tensor,
        macro_actions: torch.Tensor,
        target_actions: torch.Tensor,
        macro_mask_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs_batch.dim() == 3:
            obs_batch = obs_batch.unsqueeze(0)

        device = self._device()
        obs_batch = obs_batch.to(device).float()
        macro_actions = macro_actions.to(device)
        target_actions = target_actions.to(device)

        if macro_mask_batch is not None:
            macro_mask_batch = macro_mask_batch.to(device)

        values = self.forward_local_critic(obs_batch)
        macro_logits, target_logits, _ = self.forward_actor(obs_batch)

        if macro_mask_batch is not None and macro_mask_batch.numel() > 0:
            macro_logits = macro_logits.masked_fill(~macro_mask_batch, -1e10)

        macro_dist = torch.distributions.Categorical(logits=macro_logits)
        target_dist = torch.distributions.Categorical(logits=target_logits)

        log_probs = macro_dist.log_prob(macro_actions) + target_dist.log_prob(target_actions)
        entropy = macro_dist.entropy() + target_dist.entropy()
        return log_probs, entropy, values

    def evaluate_actions_central(
        self,
        central_obs_batch: torch.Tensor,
        actor_obs_batch: torch.Tensor,
        macro_actions: torch.Tensor,
        target_actions: torch.Tensor,
        macro_mask_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if actor_obs_batch.dim() == 3:
            actor_obs_batch = actor_obs_batch.unsqueeze(0)

        device = self._device()
        central_obs_batch = central_obs_batch.to(device).float()
        actor_obs_batch = actor_obs_batch.to(device).float()
        macro_actions = macro_actions.to(device)
        target_actions = target_actions.to(device)

        if macro_mask_batch is not None:
            macro_mask_batch = macro_mask_batch.to(device)

        values = self.forward_central_critic(central_obs_batch)
        macro_logits, target_logits, _ = self.forward_actor(actor_obs_batch)

        if macro_mask_batch is not None and macro_mask_batch.numel() > 0:
            macro_logits = macro_logits.masked_fill(~macro_mask_batch, -1e10)

        macro_dist = torch.distributions.Categorical(logits=macro_logits)
        target_dist = torch.distributions.Categorical(logits=target_logits)

        log_probs = macro_dist.log_prob(macro_actions) + target_dist.log_prob(target_actions)
        entropy = macro_dist.entropy() + target_dist.entropy()
        return log_probs, entropy, values
