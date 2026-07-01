"""Per-step buffer-fill helper for :class:`RolloutCollector` (PR-9).

This module owns the small but real complexity of *what fields go into the
rollout buffer at every step*, including the latent-vs-no-latent branching
that previously lived inline as a ~60-line block inside
:meth:`rl.custom_ppo.rollout_collector.RolloutCollector.collect`.

Why a dedicated helper
----------------------
The buffer field schema is the contract between the rollout collector and the
PPO update / GAE / option-return code paths, and it grows whenever the trainer
adds a new diagnostic (telemetry buckets, KL-prev fields, phase / outcome ids,
...). Centralising the *writes* here means anyone touching the schema only
needs to look at this file plus
:meth:`rl.custom_ppo.rollout_collector.RolloutCollector.make_buffer`,
not the per-step stepping loop.

Design
------
The recorder is intentionally a thin object: it holds a ``trainer`` reference
(for ``device``, ``use_latent_strategy``, ``latent_kl_consecutive`` and the
mutable ``latent_state`` it reads when filling the optional KL-prev fields)
and exposes a single public method, :meth:`RolloutStepRecorder.record`. All
per-step inputs are bundled into a :class:`StepFrame` dataclass so the
collector's call site reads as ``recorder.record(buffer, frame)`` instead of a
20-keyword call.

The recorder does **not** mutate trainer state. End-of-step latent housekeeping
(``prev_z_logits`` / ``z_kl_first_in_ep`` updates, ``mark_strategy_step_done``)
is owned by the collector — the recorder only *reads* what it needs from
``trainer.latent_state`` to populate the optional KL-prev fields for the row
being written.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch

from rl.latent_phase_labels import (
    outcome_id_from_global_state,
    team_phase_id_from_global_state,
)
from rl.ppo_core import TensorDictRolloutBuffer
from rl.custom_ppo.rollout.models import StepFrame

if TYPE_CHECKING:
    from rl.custom_ppo.trainer import CustomPPOTrainer

class RolloutStepRecorder:
    """Builds the per-step ``add_items`` dict and writes it to a rollout buffer.

    Owns the latent-vs-no-latent branching that previously lived inline
    inside :meth:`RolloutCollector.collect`. Constructed once per
    :class:`~rl.custom_ppo.rollout_collector.RolloutCollector` and reused
    across every step of every rollout.

    The recorder reads:

    * ``trainer.device`` — placement of newly built tensors
    * ``trainer.use_latent_strategy`` — toggles latent extras
    * ``trainer.latent_kl_consecutive`` and ``trainer.latent_state``
      (``prev_z_logits``, ``z_kl_first_in_ep``) — only when populating the
      optional KL-prev fields
    """

    def __init__(self, trainer: "CustomPPOTrainer") -> None:
        self.trainer = trainer

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    def record(self, buffer: TensorDictRolloutBuffer, frame: StepFrame) -> None:
        """Build the per-step ``add_items`` dict and append a buffer row."""
        items = self._base_items(frame)
        if self.trainer.use_latent_strategy:
            items.update(self._latent_items(frame))
            kl_extras = self._latent_kl_extras(frame)
            if kl_extras is not None:
                items.update(kl_extras)
        if bool(getattr(self.trainer.model, "communication_enabled", False)):
            items.update(self._communication_items(frame))
        buffer.add(**items)

    # ------------------------------------------------------------------
    # Private helpers.
    # ------------------------------------------------------------------

    def _base_items(self, frame: StepFrame) -> Dict[str, torch.Tensor]:
        """Always-present buffer fields (obs, action, value, reward, flags)."""
        device = self.trainer.device
        obs = frame.obs
        rc = frame.reward_component
        return dict(
            obs_grid=torch.as_tensor(obs["grid"], dtype=torch.float32, device=device),
            obs_vec=torch.as_tensor(obs["vec"], dtype=torch.float32, device=device),
            obs_agent_mask=torch.as_tensor(obs["agent_mask"], dtype=torch.float32, device=device),
            obs_mask=torch.as_tensor(obs["mask"], dtype=torch.float32, device=device),
            global_state=frame.context_state,
            actions=frame.actions_t,
            log_probs=frame.log_probs_t,
            values=frame.values_t,
            values_norm=frame.values_norm_t,
            next_values=frame.next_values_t,
            rewards=rc["reward_total"],
            reward_terminal=rc["reward_terminal"],
            reward_offense=rc["reward_offense"],
            reward_pbrs=rc["reward_pbrs"],
            reward_team=rc["reward_team"],
            reward_sparse=rc["reward_sparse"],
            reward_sparse_points=rc["reward_sparse_points"],
            reward_failure=rc["reward_failure"],
            reward_behavior_contrast=rc["reward_behavior_contrast"],
            reward_csia=rc["reward_csia"],
            reward_total=rc["reward_total"],
            terminated=torch.as_tensor(frame.terminated, dtype=torch.bool, device=device),
            truncated=torch.as_tensor(frame.truncated, dtype=torch.bool, device=device),
            opponent_id=frame.opp_row,
            **({} if "router_reward" not in rc else {"router_reward": rc["router_reward"]}),
        )

    def _latent_items(self, frame: StepFrame) -> Dict[str, torch.Tensor]:
        """Latent-strategy extras (z, phase / outcome, telemetry, buckets)."""
        if frame.strategy_aux is None:
            raise ValueError(
                "Latent rollout step requires StepFrame.strategy_aux to be populated."
            )
        if (
            frame.behavior_telemetry is None
            or frame.spread_bucket is None
            or frame.role_bucket is None
            or frame.pressure_bucket is None
            or frame.attack_defense_ratio_bucket is None
            or frame.blue_ahead is None
        ):
            raise ValueError(
                "Latent rollout step requires behavior telemetry / bucket / "
                "blue_ahead tensors on StepFrame."
            )
        phase_t, outcome_t = self._phase_outcome_ids(frame)
        sa = frame.strategy_aux
        items = dict(
            z=sa["z"],
            prev_z=sa["prev_z"],
            z_log_probs=sa["z_log_prob"],
            z_logits=sa["z_logits"],
            z_resampled=sa["z_resampled"],
            z_resampled_actual=sa.get("z_resampled_actual", sa["z_resampled"]),
            z_forced=sa["z_forced"],
            z_persist_mask=sa["z_persist_mask"],
            phase_id=phase_t,
            outcome_id=outcome_t,
            behavior_telemetry=frame.behavior_telemetry,
            spread_bucket_id=frame.spread_bucket,
            role_bucket_id=frame.role_bucket,
            pressure_bucket_id=frame.pressure_bucket,
            attack_defense_ratio_bucket_id=frame.attack_defense_ratio_bucket,
            blue_ahead=frame.blue_ahead,
        )
        if "selector_hidden" in sa:
            items["selector_hidden"] = sa["selector_hidden"]
        for key in (
            "router_context",
            "prev_router_context",
            "persistence_valid",
            "episode_id",
            "opportunity_index",
            "env_id",
        ):
            if key in sa:
                items[key] = sa[key]
        return items

    def _communication_items(self, frame: StepFrame) -> Dict[str, torch.Tensor]:
        if frame.message_aux is None:
            raise ValueError("Communication rollout step requires StepFrame.message_aux.")
        aux = frame.message_aux
        return dict(
            message_symbols=aux["message_symbols"],
            message_log_probs=aux["message_log_probs"],
            message_entropy=aux["message_entropy"],
            message_boundary_mask=aux["message_boundary_mask"],
        )

    def _latent_kl_extras(
        self, frame: StepFrame
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Optional KL-prev fields used by ``latent_kl_consecutive``."""
        trainer = self.trainer
        if trainer.latent_kl_consecutive <= 0.0:
            return None
        if trainer.latent_state.z_kl_first_in_ep is None:
            return None
        if frame.strategy_aux is None:
            return None
        z_logits_cur = frame.strategy_aux["z_logits"]
        zlp = trainer.latent_state.prev_z_logits
        if zlp is None:
            zlp = torch.zeros_like(z_logits_cur)
        return {
            "z_logits_prev": zlp,
            "z_kl_prev_valid": (~trainer.latent_state.z_kl_first_in_ep).to(
                dtype=torch.float32
            ),
        }

    def _phase_outcome_ids(
        self, frame: StepFrame
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute ``phase_id`` and ``outcome_id`` rows from the global state."""
        device = self.trainer.device
        gs_np = frame.decision_global_state_np
        n_e = int(gs_np.shape[0])
        infos = frame.infos
        phase_list: list[int] = []
        outcome_list: list[int] = []
        for e in range(n_e):
            info_e = dict(infos[e]) if e < len(infos) else {}
            sf = float(info_e.get("stalemate_frac", 0.0) or 0.0)
            phase_list.append(
                int(team_phase_id_from_global_state(gs_np[e], stalemate_frac=sf))
            )
            outcome_list.append(int(outcome_id_from_global_state(gs_np[e])))
        return (
            torch.as_tensor(phase_list, dtype=torch.long, device=device),
            torch.as_tensor(outcome_list, dtype=torch.long, device=device),
        )


__all__ = ["RolloutStepRecorder", "StepFrame"]
