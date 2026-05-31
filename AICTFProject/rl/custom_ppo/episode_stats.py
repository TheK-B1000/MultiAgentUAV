"""Episode-scoped counters and record buffer for :class:`CustomPPOTrainer`.

This sub-component owns the small handful of episode-level mutable state
that used to live on the trainer as underscore-prefixed private attributes
(``_ep_wins`` / ``_ep_losses`` / ``_ep_draws`` / ``_episodes_completed`` /
``_rollout_episode_records`` / ``_recent_episode_successes``). Those were
written from ``rollout_collector.on_episode_done`` and read from
``training_telemetry``, ``curriculum_runtime``, and ``latent_strategy_state``
— a textbook cross-module private-attribute smell.

Held by the trainer as :attr:`CustomPPOTrainer.episode_stats`. Pure Python
state; no torch, no device.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Optional


class EpisodeStats:
    """Win / loss / draw counters plus the per-rollout episode record buffer.

    Records are produced once per episode in ``rollout_collector.on_episode_done``
    and drained at the start of each new rollout via :meth:`reset_rollout`.
    ``recent_successes`` is a fixed-size deque (default 200) used by
    ``training_telemetry.rolling_win_rate``.
    """

    def __init__(self, *, success_window: int = 200) -> None:
        self.wins: int = 0
        self.losses: int = 0
        self.draws: int = 0
        self.episodes_completed: int = 0
        self.rollout_records: list[dict[str, Any]] = []
        self.recent_successes: deque[int] = deque(maxlen=int(success_window))

    # ------------------------------------------------------------------
    # Mutation: one call per finished episode.
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        blue_score: int,
        red_score: int,
        latent_z: Optional[int],
        opponent_id: int,
    ) -> int:
        """Fold one finished episode into the counters and return success (1/0)."""
        bs, rs = int(blue_score), int(red_score)
        success = 1 if bs > rs else 0
        if bs > rs:
            self.wins += 1
        elif bs < rs:
            self.losses += 1
        else:
            self.draws += 1
        self.episodes_completed += 1
        self.rollout_records.append(
            {
                "blue_score": bs,
                "red_score": rs,
                "win_margin": bs - rs,
                "success": success,
                "latent_z": latent_z,
                "opponent_id": int(opponent_id),
            }
        )
        self.recent_successes.append(success)
        return success

    def reset_rollout(self) -> None:
        """Drop the per-rollout records list; counters persist across rollouts."""
        self.rollout_records = []

    # ------------------------------------------------------------------
    # Read-only views.
    # ------------------------------------------------------------------

    @property
    def total_games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def cumulative_win_rate(self) -> float:
        return float(self.wins) / float(max(1, self.total_games))

    def rolling_win_rate(self, window: int) -> float:
        n = max(1, int(window))
        recent = list(self.recent_successes)[-n:]
        if not recent:
            return 0.0
        return float(sum(recent)) / float(len(recent))


__all__ = ["EpisodeStats"]
