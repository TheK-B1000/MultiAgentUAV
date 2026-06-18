"""Latent strategy coordinator — composes router, credit, and intervention owners."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

from rl.custom_ppo.latent.checkpoint import (
    SCHEMA_VERSION,
    latent_checkpoint_payload,
    restore_latent_checkpoint_payload,
)
from rl.custom_ppo.latent.credit.arc_credit import ArcCreditManager
from rl.custom_ppo.latent.credit.episode_credit import EpisodeCreditManager
from rl.custom_ppo.latent.credit.macro_credit import MacroCreditManager
from rl.custom_ppo.latent.credit.refresh_credit import RefreshCreditManager
from rl.custom_ppo.latent.intervention_state import InterventionEMAController
from rl.custom_ppo.latent.router_sampling import RouterSamplingState
from rl.custom_ppo.latent_strategy_state import LatentStrategyStateCore

if TYPE_CHECKING:
    from rl.custom_ppo.trainer import CustomPPOTrainer

__all__ = ["LatentStrategyState"]


class LatentStrategyState(LatentStrategyStateCore):
    """Thin coordinator delegating to modular algorithm owners under ``latent/``."""

    def _wire_managers(self) -> None:
        if getattr(self, "_managers_wired", False):
            return
        self.router = RouterSamplingState(self)
        self.macro_credit = MacroCreditManager(self)
        self.arc_credit = ArcCreditManager(self)
        self.episode_credit = EpisodeCreditManager(self)
        self.refresh_credit = RefreshCreditManager(self)
        self.intervention_ema = InterventionEMAController(self)
        self._managers_wired = True

    def __init__(self, trainer: "CustomPPOTrainer") -> None:
        super().__init__(trainer)
        self._wire_managers()

    # Router sampling
    def strategy_for_step(
        self, global_state: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], dict[str, torch.Tensor]]:
        return self.router.strategy_for_step(global_state)

    def mark_strategy_step_done(self, dones: np.ndarray) -> None:
        self.router.mark_strategy_step_done(dones)

    def record_tactical_context_step(self, global_state: torch.Tensor) -> None:
        self.router.record_tactical_context_step(global_state)

    def representative_tactical_bucket(self, env_index: int) -> int:
        return self.router.representative_tactical_bucket(env_index)

    # Macro credit
    def reset_macro_rollout_state(self) -> None:
        self.macro_credit.reset_macro_rollout_state()

    def macro_accumulate_step(self, rewards: torch.Tensor) -> None:
        self.macro_credit.macro_accumulate_step(rewards)

    def macro_finalize(self, finalize_mask: torch.Tensor, *, reason: str = "boundary") -> int:
        return self.macro_credit.macro_finalize(finalize_mask, reason=reason)

    def macro_open(
        self,
        open_mask: torch.Tensor,
        *,
        global_state: torch.Tensor,
        z_idx: torch.Tensor,
        z_log_prob: torch.Tensor,
        selector_hidden: torch.Tensor | None = None,
    ) -> int:
        return self.macro_credit.macro_open(
            open_mask,
            global_state=global_state,
            z_idx=z_idx,
            z_log_prob=z_log_prob,
            selector_hidden=selector_hidden,
        )

    def apply_macro_strategy_ppo(self) -> dict[str, float]:
        return self.macro_credit.apply_macro_strategy_ppo()

    @staticmethod
    def empty_macro_strategy_stats() -> dict[str, float]:
        return MacroCreditManager.empty_macro_strategy_stats()

    # Arc credit
    def reset_arc_credit_rollout_state(self) -> None:
        self.arc_credit.reset_arc_credit_rollout_state()

    def arc_accumulate_step(self, rewards: torch.Tensor) -> None:
        self.arc_credit.arc_accumulate_step(rewards)

    def arc_finalize(
        self,
        finalize_mask: torch.Tensor,
        *,
        opponent_ids: torch.Tensor | None = None,
        reason: str = "z_change",
    ) -> int:
        return self.arc_credit.arc_finalize(
            finalize_mask, opponent_ids=opponent_ids, reason=reason
        )

    def arc_open(
        self,
        open_mask: torch.Tensor,
        *,
        global_state: torch.Tensor,
        z_idx: torch.Tensor,
        z_log_prob: torch.Tensor,
        opponent_ids: torch.Tensor | None = None,
        selector_hidden: torch.Tensor | None = None,
    ) -> int:
        return self.arc_credit.arc_open(
            open_mask,
            global_state=global_state,
            z_idx=z_idx,
            z_log_prob=z_log_prob,
            opponent_ids=opponent_ids,
            selector_hidden=selector_hidden,
        )

    def apply_arc_strategy_ppo(self) -> dict[str, float]:
        return self.arc_credit.apply_arc_strategy_ppo()

    @staticmethod
    def empty_arc_strategy_stats() -> dict[str, float]:
        return ArcCreditManager.empty_arc_strategy_stats()

    # Episode credit
    def store_episode_strategy_start(
        self,
        *,
        start_mask: torch.Tensor,
        global_state: torch.Tensor,
        z_idx: torch.Tensor,
        z_log_prob: torch.Tensor,
        z_logits: torch.Tensor,
        selector_hidden: torch.Tensor | None = None,
    ) -> None:
        self.episode_credit.store_episode_strategy_start(
            start_mask=start_mask,
            global_state=global_state,
            z_idx=z_idx,
            z_log_prob=z_log_prob,
            z_logits=z_logits,
            selector_hidden=selector_hidden,
        )

    def apply_episode_strategy_ppo(self, *, latent_lam_h: float) -> dict[str, float]:
        return self.episode_credit.apply_episode_strategy_ppo(latent_lam_h=latent_lam_h)

    @staticmethod
    def empty_episode_strategy_stats(latent_k: int = 4) -> dict[str, float]:
        return EpisodeCreditManager.empty_episode_strategy_stats(latent_k)

    def episode_strategy_training_batch(self) -> Optional[dict[str, torch.Tensor]]:
        return self.episode_credit.episode_strategy_training_batch()

    def record_episode_strategy_outcome(
        self,
        env_index: int,
        info: dict[str, Any],
        *,
        episode_return: float,
    ) -> None:
        self.episode_credit.record_episode_strategy_outcome(
            env_index, info, episode_return=episode_return
        )

    # Refresh credit
    def finalize_v3i3_refresh_records(
        self,
        env_index: int,
        info: dict[str, Any],
        *,
        episode_return: float,
    ) -> None:
        self.refresh_credit.finalize_v3i3_refresh_records(
            env_index, info, episode_return=episode_return
        )

    # Intervention EMA
    def update_cf_pair_jsd_ema(self, pair_values: list[float], timestep: int) -> bool:
        self._wire_managers()
        return self.intervention_ema.update_cf_pair_jsd_ema(pair_values, timestep)

    def update_macro_pair_jsd_ema(self, pair_values: list[float], timestep: int) -> bool:
        self._wire_managers()
        return self.intervention_ema.update_macro_pair_jsd_ema(pair_values, timestep)

    def update_intervention_gate_from_profile(self, profile_stats: dict[str, float]) -> bool:
        self._wire_managers()
        return self.intervention_ema.update_intervention_gate_from_profile(profile_stats)

    def update_actor_intervention_gate_from_cf_pairs(self, pair_vals: list[float]) -> bool:
        self._wire_managers()
        return self.intervention_ema.update_actor_intervention_gate_from_cf_pairs(pair_vals)

    def update_macro_pair_jsd_ema_from_profile(self, profile_stats: dict[str, float]) -> bool:
        self._wire_managers()
        return self.intervention_ema.update_macro_pair_jsd_ema_from_profile(profile_stats)

    # Checkpoint schema v2
    def state_dict(self) -> dict[str, Any]:
        return latent_checkpoint_payload(self)

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        restore_latent_checkpoint_payload(self, payload)
