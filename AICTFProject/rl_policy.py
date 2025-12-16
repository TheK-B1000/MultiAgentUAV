from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from obs_encoder import ObsEncoder
from macro_actions import MacroAction


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
        height: int = 20,
        width: int = 20,
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

        if used_macros is None:
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

        self._macro_to_index: Dict[MacroAction, int] = {m: i for i, m in enumerate(self.used_macros)}

        self.encoder = ObsEncoder(
            in_channels=self.in_channels,
            height=self.height,
            width=self.width,
            latent_dim=self.latent_dim,
        )

        self.actor_macro = nn.Linear(self.latent_dim, self.n_macros)
        self.actor_target = nn.Linear(self.latent_dim, self.n_targets)

        # IMPORTANT: no inplace ReLU for DML stability
        self.local_value_head = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 1),
        )

        critic_in = self.n_agents * self.latent_dim
        self.central_value_head = nn.Sequential(
            nn.Linear(critic_in, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 1),
        )

        self._init_heads()

    # -----------------------------
    # init / helpers
    # -----------------------------
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
        # DML: keep contiguous through convs
        return self.encoder(obs.contiguous())

    def _idx(self, macro: MacroAction) -> Optional[int]:
        return self._macro_to_index.get(macro, None)

    @staticmethod
    def _fix_all_false_rows(mask_bool: torch.Tensor) -> torch.Tensor:
        """
        mask_bool: [B,A] bool
        Ensures no row is all-false (would make logits all -inf).
        """
        if mask_bool.numel() == 0:
            return mask_bool
        row_sum = mask_bool.sum(dim=-1)
        bad = row_sum == 0
        if bad.any():
            mask_bool = mask_bool.clone()
            mask_bool[bad] = True
        return mask_bool

    @staticmethod
    def _apply_mask_float_penalty(
        logits: torch.Tensor,
        mask_bool: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        logits: [B,A]
        mask_bool: [B,A] bool (True=valid)
        Applies huge negative penalty to invalid entries using float math (DML-friendlier).
        """
        if mask_bool is None or mask_bool.numel() == 0:
            return logits
        mask_bool = ActorCriticNet._fix_all_false_rows(mask_bool)
        mf = mask_bool.to(dtype=logits.dtype)
        return logits + (mf - 1.0) * 1e10

    @staticmethod
    def _masked_logp_entropy_no_scatter(
        logits: torch.Tensor,                 # [B,A]
        actions: torch.Tensor,                # [B]
        mask_bool: Optional[torch.Tensor] = None,  # [B,A] bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DML-safe: no gather/scatter in backward.
        Uses CPU one-hot and multiply-sum to pick logp(action).
        """
        logits = ActorCriticNet._apply_mask_float_penalty(logits, mask_bool)

        logp_all = F.log_softmax(logits, dim=-1)   # [B,A]
        p_all = torch.exp(logp_all)                # [B,A]

        oh = F.one_hot(actions.to("cpu"), num_classes=logits.size(-1)).to(
            device=logits.device, dtype=logp_all.dtype
        )
        logp = (oh * logp_all).sum(dim=-1)         # [B]
        entropy = -(p_all * logp_all).sum(dim=-1)  # [B]
        return logp, entropy

    # -----------------------------
    # forward passes
    # -----------------------------
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

        central_obs = central_obs.contiguous()
        flat = central_obs.view(B * N, C, H, W).contiguous()

        latent_flat = self._encode(flat).contiguous()
        latent_joint = latent_flat.view(B, N * self.latent_dim).contiguous()

        return self.central_value_head(latent_joint).squeeze(-1)

    # -----------------------------
    # masks
    # -----------------------------
    @torch.no_grad()
    def get_action_mask(self, agent, game_field) -> torch.Tensor:
        """
        Returns [A] bool mask where True means macro is valid.
        Tries env.get_macro_mask(agent) first.
        """
        device = self._device()

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

        carrying = bool(getattr(agent, "isCarryingFlag", lambda: False)())
        if carrying:
            idx = self._idx(MacroAction.GET_FLAG)
            if idx is not None:
                mask[idx] = False

        if int(getattr(agent, "mine_charges", 0)) <= 0:
            idx = self._idx(MacroAction.PLACE_MINE)
            if idx is not None:
                mask[idx] = False

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

    # -----------------------------
    # action sampling
    # -----------------------------
    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        agent=None,
        game_field=None,
        deterministic: bool = False,
        return_old_log_prob_key: bool = False,
    ) -> Dict[str, Any]:
        """
        PPO-style act (local critic). Returns:
          macro_action, target_action, log_prob, value, macro_mask
        If return_old_log_prob_key=True, also returns "old_log_prob".
        """
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)

        device = self._device()
        obs = obs.to(device).float().contiguous()

        macro_logits, target_logits, latent = self.forward_actor(obs)
        local_value = self.local_value_head(latent).squeeze(-1)  # [B]

        macro_mask = None
        if agent is not None and game_field is not None:
            macro_mask = self.get_action_mask(agent, game_field)  # [A]
            macro_logits = self._apply_mask_float_penalty(macro_logits, macro_mask.unsqueeze(0))

        if deterministic:
            macro_action = macro_logits.argmax(dim=-1)      # [B]
            target_action = target_logits.argmax(dim=-1)    # [B]
        else:
            macro_action = torch.multinomial(torch.softmax(macro_logits, dim=-1), 1).squeeze(1)
            target_action = torch.multinomial(torch.softmax(target_logits, dim=-1), 1).squeeze(1)

        macro_logp, _ = self._masked_logp_entropy_no_scatter(macro_logits, macro_action, mask_bool=None)
        targ_logp, _ = self._masked_logp_entropy_no_scatter(target_logits, target_action, mask_bool=None)
        log_prob = macro_logp + targ_logp  # [B]

        out = {
            "macro_action": macro_action,
            "target_action": target_action,
            "log_prob": log_prob,
            "value": local_value,
            "macro_mask": macro_mask,
        }
        if return_old_log_prob_key:
            out["old_log_prob"] = log_prob
        return out

    @torch.no_grad()
    def act_mappo(
        self,
        actor_obs: torch.Tensor,
        central_obs: torch.Tensor,
        agent=None,
        game_field=None,
        deterministic: bool = False,
        return_old_log_prob_key: bool = False,
    ) -> Dict[str, Any]:
        """
        MAPPO-style act (central critic). Returns:
          macro_action, target_action, log_prob, value(central), macro_mask
        """
        if actor_obs.dim() == 3:
            actor_obs = actor_obs.unsqueeze(0)
        if central_obs.dim() == 4:
            central_obs = central_obs.unsqueeze(0)

        device = self._device()
        actor_obs = actor_obs.to(device).float().contiguous()
        central_obs = central_obs.to(device).float().contiguous()

        macro_logits, target_logits, _ = self.forward_actor(actor_obs)
        central_value = self.forward_central_critic(central_obs).reshape(-1)  # [B]

        macro_mask = None
        if agent is not None and game_field is not None:
            macro_mask = self.get_action_mask(agent, game_field)  # [A]
            macro_logits = self._apply_mask_float_penalty(macro_logits, macro_mask.unsqueeze(0))

        if deterministic:
            macro_action = macro_logits.argmax(dim=-1)
            target_action = target_logits.argmax(dim=-1)
        else:
            macro_action = torch.multinomial(torch.softmax(macro_logits, dim=-1), 1).squeeze(1)
            target_action = torch.multinomial(torch.softmax(target_logits, dim=-1), 1).squeeze(1)

        macro_logp, _ = self._masked_logp_entropy_no_scatter(macro_logits, macro_action, mask_bool=None)
        targ_logp, _ = self._masked_logp_entropy_no_scatter(target_logits, target_action, mask_bool=None)
        log_prob = macro_logp + targ_logp

        out = {
            "macro_action": macro_action,
            "target_action": target_action,
            "log_prob": log_prob,
            "value": central_value,
            "macro_mask": macro_mask,
        }
        if return_old_log_prob_key:
            out["old_log_prob"] = log_prob
        return out

    # -----------------------------
    # PPO evaluation
    # -----------------------------
    def evaluate_actions(
        self,
        obs_batch: torch.Tensor,                 # [B,C,H,W]
        macro_actions: torch.Tensor,             # [B]
        target_actions: torch.Tensor,            # [B]
        macro_mask_batch: Optional[Union[torch.Tensor, np.ndarray]] = None,  # [B,A]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        PPO evaluation with local critic.
        Returns: log_probs [B], entropy [B], values [B]
        """
        if obs_batch.dim() == 3:
            obs_batch = obs_batch.unsqueeze(0)

        device = self._device()
        obs_batch = obs_batch.to(device).float().contiguous()
        macro_actions = macro_actions.to(device)
        target_actions = target_actions.to(device)

        mask_bool = None
        if macro_mask_batch is not None:
            if isinstance(macro_mask_batch, np.ndarray):
                macro_mask_batch = torch.from_numpy(macro_mask_batch)
            macro_mask_batch = macro_mask_batch.to(device)
            # allow float {0,1} or bool
            mask_bool = macro_mask_batch.bool()

        values = self.forward_local_critic(obs_batch)  # [B]
        macro_logits, target_logits, _ = self.forward_actor(obs_batch)

        macro_logp, macro_ent = self._masked_logp_entropy_no_scatter(macro_logits, macro_actions, mask_bool=mask_bool)
        targ_logp, targ_ent = self._masked_logp_entropy_no_scatter(target_logits, target_actions, mask_bool=None)

        log_probs = macro_logp + targ_logp
        entropy = macro_ent + targ_ent
        return log_probs, entropy, values

    def evaluate_actions_central(
        self,
        central_obs_batch: torch.Tensor,         # [B,N,C,H,W]
        actor_obs_batch: torch.Tensor,           # [B,C,H,W]
        macro_actions: torch.Tensor,             # [B]
        target_actions: torch.Tensor,            # [B]
        macro_mask_batch: Optional[Union[torch.Tensor, np.ndarray]] = None,  # [B,A]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MAPPO evaluation with central critic.
        Returns: log_probs [B], entropy [B], values [B]
        """
        if actor_obs_batch.dim() == 3:
            actor_obs_batch = actor_obs_batch.unsqueeze(0)

        device = self._device()
        central_obs_batch = central_obs_batch.to(device).float().contiguous()
        actor_obs_batch = actor_obs_batch.to(device).float().contiguous()
        macro_actions = macro_actions.to(device)
        target_actions = target_actions.to(device)

        mask_bool = None
        if macro_mask_batch is not None:
            if isinstance(macro_mask_batch, np.ndarray):
                macro_mask_batch = torch.from_numpy(macro_mask_batch)
            macro_mask_batch = macro_mask_batch.to(device)
            mask_bool = macro_mask_batch.bool()

        values = self.forward_central_critic(central_obs_batch).reshape(-1)  # [B]
        macro_logits, target_logits, _ = self.forward_actor(actor_obs_batch)

        macro_logp, macro_ent = self._masked_logp_entropy_no_scatter(macro_logits, macro_actions, mask_bool=mask_bool)
        targ_logp, targ_ent = self._masked_logp_entropy_no_scatter(target_logits, target_actions, mask_bool=None)

        log_probs = macro_logp + targ_logp
        entropy = macro_ent + targ_ent
        return log_probs, entropy, values
