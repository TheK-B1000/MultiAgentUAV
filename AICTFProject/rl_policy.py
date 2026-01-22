from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from obs_encoder import ObsEncoder
from macro_actions import MacroAction


@dataclass(frozen=True)
class ActorCriticSpec:
    n_macros: int = 5
    n_targets: int = 50
    latent_dim: int = 128
    in_channels: int = 7
    height: int = 20
    width: int = 20
    n_agents: int = 2  # for central critic input [B,N,C,H,W]


class ActorCriticNet(nn.Module):
    """
    Shared CNN encoder.
      - Decentralized actor heads: macro + target
      - Local critic (PPO)
      - Central critic (MAPPO / CTDE)

    Shapes:
      obs:         [B,C,H,W] or [C,H,W]
      central_obs: [B,N,C,H,W] or [N,C,H,W] (will be batched)

    Notes:
      - Action masking is applied with float math (no -inf) for DirectML friendliness.
      - Log-prob selection avoids gather/scatter in backward by using CPU one-hot.
      - Target head is only “active” for parameterized macros (GO_TO, PLACE_MINE by default).
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
        self.spec = ActorCriticSpec(
            n_macros=int(n_macros),
            n_targets=int(n_targets),
            latent_dim=int(latent_dim),
            in_channels=int(in_channels),
            height=int(height),
            width=int(width),
            n_agents=int(n_agents),
        )

        if used_macros is None:
            used_macros = [
                MacroAction.GO_TO,
                MacroAction.GRAB_MINE,
                MacroAction.GET_FLAG,
                MacroAction.PLACE_MINE,
                MacroAction.GO_HOME,
            ]
        self.used_macros: List[MacroAction] = list(used_macros)
        if len(self.used_macros) != self.spec.n_macros:
            raise ValueError(
                f"used_macros length {len(self.used_macros)} != n_macros {self.spec.n_macros}"
            )

        self._macro_to_index: Dict[MacroAction, int] = {m: i for i, m in enumerate(self.used_macros)}

        self.encoder = ObsEncoder(
            in_channels=self.spec.in_channels,
            height=self.spec.height,
            width=self.spec.width,
            latent_dim=self.spec.latent_dim,
        )

        # Actor heads
        self.actor_macro = nn.Linear(self.spec.latent_dim, self.spec.n_macros)
        self.actor_target = nn.Linear(self.spec.latent_dim, self.spec.n_targets)

        # Critics (no inplace ReLU for DML stability)
        self.local_value_head = nn.Sequential(
            nn.Linear(self.spec.latent_dim, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 1),
        )

        critic_in = self.spec.n_agents * self.spec.latent_dim
        self.central_value_head = nn.Sequential(
            nn.Linear(critic_in, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 1),
        )

        self._init_heads()

    # -----------------------------
    # Initialization
    # -----------------------------
    def _init_heads(self) -> None:
        nn.init.orthogonal_(self.actor_macro.weight, gain=0.01)
        nn.init.constant_(self.actor_macro.bias, 0.0)

        nn.init.orthogonal_(self.actor_target.weight, gain=0.01)
        nn.init.constant_(self.actor_target.bias, 0.0)

        for head in (self.local_value_head, self.central_value_head):
            last = head[-1]
            if isinstance(last, nn.Linear):
                nn.init.orthogonal_(last.weight, gain=1.0)
                nn.init.constant_(last.bias, 0.0)

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _encode(self, obs_bchw: torch.Tensor) -> torch.Tensor:
        # Keep contiguous through convs for DML safety/perf
        return self.encoder(obs_bchw.contiguous())

    def _idx(self, macro: MacroAction) -> Optional[int]:
        return self._macro_to_index.get(macro, None)

    # -----------------------------
    # Target gating
    # -----------------------------
    def _needs_target(self, macro_actions: torch.Tensor) -> torch.Tensor:
        """
        macro_actions: [B] (long/int)
        returns: [B] float in {0,1}
        """
        idx_go_to = self._idx(MacroAction.GO_TO)
        idx_place = self._idx(MacroAction.PLACE_MINE)

        needs = torch.zeros_like(macro_actions, dtype=torch.float32, device=macro_actions.device)
        if idx_go_to is not None:
            needs = needs + (macro_actions == int(idx_go_to)).float()
        if idx_place is not None:
            needs = needs + (macro_actions == int(idx_place)).float()

        return torch.clamp(needs, 0.0, 1.0)

    # -----------------------------
    # Masking (float penalty)
    # -----------------------------
    @staticmethod
    def _fix_all_false_rows(mask_bool: torch.Tensor) -> torch.Tensor:
        """
        mask_bool: [B,A] bool. Ensure no row is all-false (would make distribution invalid).
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
        logits: torch.Tensor,                 # [B,A]
        mask_bool: Optional[torch.Tensor],     # [B,A] bool
        penalty: float = 1e10,
    ) -> torch.Tensor:
        if mask_bool is None or mask_bool.numel() == 0:
            return logits
        mask_bool = ActorCriticNet._fix_all_false_rows(mask_bool)
        mf = mask_bool.to(dtype=logits.dtype)
        # valid -> +0, invalid -> -penalty
        return logits + (mf - 1.0) * float(penalty)

    # -----------------------------
    # DML-safe logp/entropy (no gather/scatter backward)
    # -----------------------------
    @staticmethod
    def _one_hot_cpu(actions: torch.Tensor, num_classes: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        oh = F.one_hot(actions.to("cpu").long(), num_classes=num_classes)
        return oh.to(device=device, dtype=dtype)

    @staticmethod
    def _masked_logp_entropy_no_scatter(
        logits: torch.Tensor,                      # [B,A]
        actions: torch.Tensor,                     # [B]
        mask_bool: Optional[torch.Tensor] = None,  # [B,A] bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = ActorCriticNet._apply_mask_float_penalty(logits, mask_bool)

        logp_all = F.log_softmax(logits, dim=-1)          # [B,A]
        p_all = torch.exp(logp_all)                       # [B,A]

        oh = ActorCriticNet._one_hot_cpu(actions, logits.size(-1), logits.device, logp_all.dtype)
        logp = (oh * logp_all).sum(dim=-1)                # [B]
        entropy = -(p_all * logp_all).sum(dim=-1)         # [B]
        return logp, entropy

    @staticmethod
    def _masked_logp_entropy(
        logits: torch.Tensor,                      # [B,A]
        actions: torch.Tensor,                     # [B]
        mask_bool: Optional[torch.Tensor] = None,  # [B,A] bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fast path (gather) on standard backends; DML-safe path otherwise.
        """
        if logits.device.type in ("privateuseone",):
            return ActorCriticNet._masked_logp_entropy_no_scatter(logits, actions, mask_bool=mask_bool)

        logits = ActorCriticNet._apply_mask_float_penalty(logits, mask_bool)
        logp_all = F.log_softmax(logits, dim=-1)          # [B,A]
        p_all = torch.exp(logp_all)                       # [B,A]
        logp = logp_all.gather(dim=-1, index=actions.long().unsqueeze(-1)).squeeze(-1)  # [B]
        entropy = -(p_all * logp_all).sum(dim=-1)         # [B]
        return logp, entropy

    # -----------------------------
    # Forward: actor / critics
    # -----------------------------
    def forward_actor(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self._encode(obs)
        macro_logits = self.actor_macro(latent)
        target_logits = self.actor_target(latent)
        return macro_logits, target_logits, latent

    def forward_local_critic(self, obs: torch.Tensor) -> torch.Tensor:
        latent = self._encode(obs)
        return self.local_value_head(latent).squeeze(-1)

    def _local_value_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.local_value_head(latent).squeeze(-1)

    def forward_central_critic(self, central_obs: torch.Tensor) -> torch.Tensor:
        if central_obs.dim() != 5:
            raise ValueError(f"Expected [B,N,C,H,W], got {tuple(central_obs.shape)}")

        B, N, C, H, W = central_obs.shape
        if N != self.spec.n_agents:
            raise ValueError(f"Expected N={self.spec.n_agents}, got N={N}")

        flat = central_obs.contiguous().view(B * N, C, H, W).contiguous()
        latent_flat = self._encode(flat).contiguous()                   # [B*N, latent]
        latent_joint = latent_flat.view(B, N * self.spec.latent_dim)    # [B, N*latent]
        return self.central_value_head(latent_joint).squeeze(-1)

    # -----------------------------
    # Masks (env-first)
    # -----------------------------
    @torch.no_grad()
    def get_action_mask(self, agent: Any, game_field: Any) -> torch.Tensor:
        """
        Returns [A] bool mask where True means macro is valid.
        Prefers env.get_macro_mask(agent) which should match network macro indices.
        """
        device = self._device()

        if game_field is not None and hasattr(game_field, "get_macro_mask"):
            try:
                m = game_field.get_macro_mask(agent)
                m = torch.as_tensor(m, dtype=torch.bool, device=device)
                if m.numel() == self.spec.n_macros:
                    if not m.any():
                        m[:] = True
                    return m
            except Exception:
                pass

        # Conservative fallback (should rarely be used)
        mask = torch.ones(self.spec.n_macros, dtype=torch.bool, device=device)

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
    # Action sampling (act / act_mappo)
    # -----------------------------
    @staticmethod
    def _ensure_bchw(obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 3:
            return obs.unsqueeze(0)
        if obs.dim() == 4:
            return obs
        raise ValueError(f"Expected [C,H,W] or [B,C,H,W], got {tuple(obs.shape)}")

    @staticmethod
    def _ensure_bnchw(central_obs: torch.Tensor) -> torch.Tensor:
        if central_obs.dim() == 4:
            return central_obs.unsqueeze(0)
        if central_obs.dim() == 5:
            return central_obs
        raise ValueError(f"Expected [N,C,H,W] or [B,N,C,H,W], got {tuple(central_obs.shape)}")

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        agent: Any = None,
        game_field: Any = None,
        deterministic: bool = False,
        return_old_log_prob_key: bool = False,
    ) -> Dict[str, Any]:
        """
        PPO-style act (local critic). Returns dict with:
          macro_action [B], target_action [B], log_prob [B], value [B], macro_mask [A] or None
        """
        device = self._device()
        obs = self._ensure_bchw(obs).to(device).float().contiguous()

        macro_logits, target_logits, latent = self.forward_actor(obs)
        value = self.local_value_head(latent).squeeze(-1)  # [B]

        macro_mask: Optional[torch.Tensor] = None
        mask_batch: Optional[torch.Tensor] = None
        if agent is not None and game_field is not None:
            macro_mask = self.get_action_mask(agent, game_field)  # [A]
            mask_batch = macro_mask.unsqueeze(0)                  # [1,A]
            macro_logits = self._apply_mask_float_penalty(macro_logits, mask_batch)

        if deterministic:
            macro_action = macro_logits.argmax(dim=-1)
            target_action = target_logits.argmax(dim=-1)
        else:
            macro_action = torch.multinomial(torch.softmax(macro_logits, dim=-1), 1).squeeze(1)
            target_action = torch.multinomial(torch.softmax(target_logits, dim=-1), 1).squeeze(1)

        macro_logp, _ = self._masked_logp_entropy(macro_logits, macro_action, mask_bool=None)
        targ_logp, _ = self._masked_logp_entropy(target_logits, target_action, mask_bool=None)

        needs_t = self._needs_target(macro_action)
        log_prob = macro_logp + needs_t * targ_logp

        out: Dict[str, Any] = {
            "macro_action": macro_action,
            "target_action": target_action,
            "log_prob": log_prob,
            "value": value,
            "macro_mask": macro_mask,  # [A] bool or None
        }
        if return_old_log_prob_key:
            out["old_log_prob"] = log_prob
        return out

    @torch.no_grad()
    def act_mappo(
        self,
        actor_obs: torch.Tensor,
        central_obs: torch.Tensor,
        agent: Any = None,
        game_field: Any = None,
        deterministic: bool = False,
        return_old_log_prob_key: bool = False,
    ) -> Dict[str, Any]:
        """
        MAPPO-style act (central critic). Returns dict with:
          macro_action [B], target_action [B], log_prob [B], value [B], macro_mask [A] or None
        """
        device = self._device()
        actor_obs = self._ensure_bchw(actor_obs).to(device).float().contiguous()
        central_obs = self._ensure_bnchw(central_obs).to(device).float().contiguous()

        macro_logits, target_logits, _ = self.forward_actor(actor_obs)
        value = self.forward_central_critic(central_obs).reshape(-1)  # [B]

        macro_mask: Optional[torch.Tensor] = None
        mask_batch: Optional[torch.Tensor] = None
        if agent is not None and game_field is not None:
            macro_mask = self.get_action_mask(agent, game_field)  # [A]
            mask_batch = macro_mask.unsqueeze(0)
            macro_logits = self._apply_mask_float_penalty(macro_logits, mask_batch)

        if deterministic:
            macro_action = macro_logits.argmax(dim=-1)
            target_action = target_logits.argmax(dim=-1)
        else:
            macro_action = torch.multinomial(torch.softmax(macro_logits, dim=-1), 1).squeeze(1)
            target_action = torch.multinomial(torch.softmax(target_logits, dim=-1), 1).squeeze(1)

        macro_logp, _ = self._masked_logp_entropy(macro_logits, macro_action, mask_bool=None)
        targ_logp, _ = self._masked_logp_entropy(target_logits, target_action, mask_bool=None)

        needs_t = self._needs_target(macro_action)
        log_prob = macro_logp + needs_t * targ_logp

        out: Dict[str, Any] = {
            "macro_action": macro_action,
            "target_action": target_action,
            "log_prob": log_prob,
            "value": value,
            "macro_mask": macro_mask,
        }
        if return_old_log_prob_key:
            out["old_log_prob"] = log_prob
        return out

    # -----------------------------
    # PPO / MAPPO evaluation
    # -----------------------------
    def _coerce_mask_batch(
        self,
        macro_mask_batch: Optional[Union[torch.Tensor, np.ndarray]],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if macro_mask_batch is None:
            return None
        if isinstance(macro_mask_batch, np.ndarray):
            macro_mask_batch = torch.from_numpy(macro_mask_batch)
        macro_mask_batch = macro_mask_batch.to(device)
        return macro_mask_batch.bool()

    def evaluate_actions(
        self,
        obs_batch: torch.Tensor,                                  # [B,C,H,W] (or [C,H,W])
        macro_actions: torch.Tensor,                              # [B]
        target_actions: torch.Tensor,                             # [B]
        macro_mask_batch: Optional[Union[torch.Tensor, np.ndarray]] = None,  # [B,A]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        PPO evaluation with local critic.
        Returns: log_probs [B], entropy [B], values [B]
        """
        device = self._device()

        obs_batch = self._ensure_bchw(obs_batch).to(device).float().contiguous()
        macro_actions = macro_actions.to(device).long()
        target_actions = target_actions.to(device).long()

        mask_bool = self._coerce_mask_batch(macro_mask_batch, device=device)

        macro_logits, target_logits, latent = self.forward_actor(obs_batch)
        values = self._local_value_from_latent(latent)            # [B]

        macro_logp, macro_ent = self._masked_logp_entropy(macro_logits, macro_actions, mask_bool=mask_bool)
        targ_logp, targ_ent = self._masked_logp_entropy(target_logits, target_actions, mask_bool=None)

        needs_t = self._needs_target(macro_actions)
        log_probs = macro_logp + needs_t * targ_logp
        entropy = macro_ent + needs_t * targ_ent
        return log_probs, entropy, values

    def evaluate_actions_central(
        self,
        central_obs_batch: torch.Tensor,                          # [B,N,C,H,W]
        actor_obs_batch: torch.Tensor,                            # [B,C,H,W] (or [C,H,W])
        macro_actions: torch.Tensor,                              # [B]
        target_actions: torch.Tensor,                             # [B]
        macro_mask_batch: Optional[Union[torch.Tensor, np.ndarray]] = None,  # [B,A]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MAPPO evaluation with central critic.
        Returns: log_probs [B], entropy [B], values [B]
        """
        device = self._device()

        central_obs_batch = self._ensure_bnchw(central_obs_batch).to(device).float().contiguous()
        actor_obs_batch = self._ensure_bchw(actor_obs_batch).to(device).float().contiguous()
        macro_actions = macro_actions.to(device).long()
        target_actions = target_actions.to(device).long()

        mask_bool = self._coerce_mask_batch(macro_mask_batch, device=device)

        values = self.forward_central_critic(central_obs_batch).reshape(-1)  # [B]
        macro_logits, target_logits, _ = self.forward_actor(actor_obs_batch)

        macro_logp, macro_ent = self._masked_logp_entropy(macro_logits, macro_actions, mask_bool=mask_bool)
        targ_logp, targ_ent = self._masked_logp_entropy(target_logits, target_actions, mask_bool=None)

        needs_t = self._needs_target(macro_actions)
        log_probs = macro_logp + needs_t * targ_logp
        entropy = macro_ent + needs_t * targ_ent
        return log_probs, entropy, values


__all__ = ["ActorCriticNet", "ActorCriticSpec"]
