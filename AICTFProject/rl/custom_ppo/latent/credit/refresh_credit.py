"""v3i3 event refresh record finalization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rl.custom_ppo.csv_writers import _opponent_id_int_from_info as opponent_id_int_from_info

if TYPE_CHECKING:
    from rl.custom_ppo.latent.state import LatentStrategyState


class RefreshCreditManager:
    def __init__(self, host: LatentStrategyState) -> None:
        self.host = host

    def finalize_v3i3_refresh_records(
        self,
        env_index: int,
        info: dict[str, Any],
        *,
        episode_return: float,
    ) -> None:
        """Finalize all pending v3i3 refresh records for an env on episode-done.

        Each pending record gets ``opponent_id`` (read from completion info)
        and ``future_return = episode_return - return_at_refresh`` (the post-
        refresh credit signal the v3i3 teacher distills into a target z
        distribution). Finalized records flow into two sinks:

        * ``rollout_refresh_records`` -- drained per rollout. Consumed by the
          v3i3 KL loss (provides per-refresh training queries) and by the
          per-refresh CSV log writer.
        * ``refresh_preference_buffer`` -- cumulative across rollouts (capped
          by ``latent_v3i3_event_preference_buffer_size``). The teacher's
          evidence library, keyed by ``(opp, event_type, flag_state)`` with
          hierarchical fallback at lookup time.

        Always-safe to call (no-op when v3i3 is disabled and no pending
        records). Independent of ``latent_episode_strategy_ppo`` so the
        per-refresh log can be enabled even without the episode-credit path.
        """
        trainer = self.host.trainer
        env_i = int(env_index)
        v3i3_enabled = bool(
            getattr(trainer, "latent_v3i3_event_preference_enabled", False)
            or getattr(trainer, "latent_v3i3_refresh_log_enabled", False)
        )
        if not v3i3_enabled:
            return
        pending = self.host.pending_refresh_records.get(env_i, [])
        if not pending:
            return
        try:
            opponent_id = int(opponent_id_int_from_info(trainer.cfg, info))
        except Exception:
            opponent_id = -1
        ep_return = float(episode_return)
        pref_buffer_on = bool(
            getattr(trainer, "latent_v3i3_event_preference_enabled", False)
        )
        for rec in pending:
            future_return = ep_return - float(rec["return_at_refresh"])
            finalized = dict(rec)
            finalized["opponent_id"] = opponent_id
            finalized["future_return"] = future_return
            finalized["return_from_now_to_end"] = future_return
            self.host.rollout_refresh_records.append(finalized)
            if pref_buffer_on:
                self.host.refresh_preference_buffer.append(
                    {
                        "opponent_id": opponent_id,
                        "event_type": int(rec["reason_id"]),
                        "flag_state_bucket": int(rec["flag_state_bucket"]),
                        "carrier_progress_bucket": int(rec.get("carrier_progress_bucket", -1)),
                        "z": int(rec["next_z"]),
                        "future_return": future_return,
                    }
                )
        self.host.pending_refresh_records[env_i] = []


