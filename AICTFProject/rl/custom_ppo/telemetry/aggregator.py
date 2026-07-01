"""Event aggregation for telemetry snapshots."""

from __future__ import annotations

from dataclasses import replace

from rl.custom_ppo.telemetry.events import (
    CheckpointSaved,
    EpisodeCompleted,
    OptimizationCompleted,
    RolloutCompleted,
    TelemetryEvent,
)
from rl.custom_ppo.telemetry.models import TrainingTelemetrySnapshot
from rl.custom_ppo.telemetry.validation import EventOrderValidator


class TrainingTelemetryAggregator:
    def __init__(self) -> None:
        self._validator = EventOrderValidator()
        self._snapshot = TrainingTelemetrySnapshot()

    def consume(self, event: TelemetryEvent) -> None:
        self._validator.consume(event)
        global_step = int(getattr(event, "global_step", self._snapshot.global_step))
        snapshot = replace(self._snapshot, global_step=global_step)
        if isinstance(event, RolloutCompleted):
            snapshot = replace(
                snapshot,
                rollout_count=snapshot.rollout_count + 1,
                environment_steps=snapshot.environment_steps + int(event.environment_transitions),
                last_reward_summary=event.reward_summary,
                last_latent_summary=event.latent_summary,
            )
        elif isinstance(event, OptimizationCompleted):
            snapshot = replace(
                snapshot,
                optimization_count=snapshot.optimization_count + 1,
                last_explained_variance=event.explained_variance,
            )
        elif isinstance(event, EpisodeCompleted):
            snapshot = replace(snapshot, episode_count=snapshot.episode_count + 1)
        elif isinstance(event, CheckpointSaved):
            snapshot = replace(snapshot, checkpoint_count=snapshot.checkpoint_count + 1)
        self._snapshot = snapshot

    def snapshot(self) -> TrainingTelemetrySnapshot:
        return self._snapshot

    def reset_window(self) -> None:
        self._snapshot = replace(
            self._snapshot,
            rollout_count=0,
            optimization_count=0,
            episode_count=0,
            checkpoint_count=0,
            environment_steps=0,
        )


__all__ = ["TrainingTelemetryAggregator"]
