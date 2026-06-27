"""Telemetry validation helpers."""

from __future__ import annotations

import math
from typing import Optional

from rl.custom_ppo.telemetry.errors import TelemetryValidationError
from rl.custom_ppo.telemetry.events import (
    CheckpointLoaded,
    CheckpointSaved,
    EpisodeCompleted,
    EpisodesCompleted,
    OptimizationCompleted,
    PerformanceSample,
    RolloutCompleted,
    TelemetryEvent,
    TrainingCompleted,
    TrainingFailed,
    TrainingInterrupted,
    TrainingStarted,
)


def validate_finite_optional(name: str, value: Optional[float]) -> None:
    if value is None:
        return
    if not math.isfinite(float(value)):
        raise TelemetryValidationError(f"{name} must be finite or None")


def validate_nonnegative(name: str, value: float | int) -> None:
    if value is None:
        return
    if float(value) < 0:
        raise TelemetryValidationError(f"{name} must be nonnegative")


def validate_event(event: TelemetryEvent) -> None:
    if isinstance(event, TrainingStarted):
        validate_nonnegative("global_step", event.global_step)
        validate_nonnegative("requested_total_steps", event.requested_total_steps)
    elif isinstance(event, TrainingCompleted):
        validate_nonnegative("final_global_step", event.final_global_step)
        validate_nonnegative("duration_seconds", event.duration_seconds)
    elif isinstance(event, TrainingFailed):
        validate_nonnegative("final_global_step", event.final_global_step)
        validate_nonnegative("duration_seconds", event.duration_seconds)
    elif isinstance(event, TrainingInterrupted):
        validate_nonnegative("final_global_step", event.final_global_step)
        validate_nonnegative("duration_seconds", event.duration_seconds)
    elif isinstance(event, RolloutCompleted):
        validate_nonnegative("global_step", event.global_step)
        validate_nonnegative("rollout_index", event.rollout_index)
        validate_nonnegative("vector_environment_count", event.vector_environment_count)
        validate_nonnegative("vector_steps", event.vector_steps)
        validate_nonnegative("environment_transitions", event.environment_transitions)
        validate_nonnegative("duration_seconds", event.duration_seconds)
        validate_finite_optional("environment_step_duration_seconds", event.environment_step_duration_seconds)
        validate_finite_optional("policy_inference_duration_seconds", event.policy_inference_duration_seconds)
        validate_finite_optional("transition_build_duration_seconds", event.transition_build_duration_seconds)
        validate_finite_optional("buffer_write_duration_seconds", event.buffer_write_duration_seconds)
        validate_finite_optional("episode_bookkeeping_duration_seconds", event.episode_bookkeeping_duration_seconds)
        validate_finite_optional("environment_transitions_per_second", event.environment_transitions_per_second)
        validate_finite_optional("rollout_transitions_per_second", event.rollout_transitions_per_second)
        validate_nonnegative("completed_episode_count", event.completed_episode_count)
        validate_finite_optional("episode_return_mean", event.episode_return_mean)
        validate_finite_optional("episode_length_mean", event.episode_length_mean)
        validate_nonnegative("gpu_memory_allocated_peak_bytes", event.gpu_memory_allocated_peak_bytes)
        validate_nonnegative("gpu_memory_reserved_peak_bytes", event.gpu_memory_reserved_peak_bytes)
    elif isinstance(event, OptimizationCompleted):
        validate_nonnegative("global_step", event.global_step)
        validate_nonnegative("optimization_index", event.optimization_index)
        validate_nonnegative("duration_seconds", event.duration_seconds)
        validate_nonnegative("samples_processed", event.samples_processed)
        validate_nonnegative("minibatches_processed", event.minibatches_processed)
        validate_nonnegative("optimizer_updates", event.optimizer_updates)
        validate_finite_optional("optimization_samples_per_second", event.optimization_samples_per_second)
        for name in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
            validate_finite_optional(name, float(getattr(event, name)))
        validate_finite_optional("explained_variance", event.explained_variance)
        validate_nonnegative("gpu_memory_allocated_peak_bytes", event.gpu_memory_allocated_peak_bytes)
        validate_nonnegative("gpu_memory_reserved_peak_bytes", event.gpu_memory_reserved_peak_bytes)
    elif isinstance(event, EpisodeCompleted):
        validate_nonnegative("global_step", event.global_step)
        validate_nonnegative("environment_index", event.environment_index)
        validate_nonnegative("episode_length", event.episode_length)
        validate_finite_optional("episode_return", event.episode_return)
    elif isinstance(event, EpisodesCompleted):
        validate_nonnegative("global_step", event.global_step)
        for ep in event.episodes:
            validate_event(ep)
    elif isinstance(event, CheckpointSaved):
        validate_nonnegative("global_step", event.global_step)
        validate_nonnegative("save_duration_seconds", event.save_duration_seconds)
        validate_nonnegative("checkpoint_size_bytes", event.checkpoint_size_bytes)
    elif isinstance(event, CheckpointLoaded):
        validate_nonnegative("global_step", event.global_step)
        validate_nonnegative("load_duration_seconds", event.load_duration_seconds)
    elif isinstance(event, PerformanceSample):
        validate_nonnegative("timestamp_seconds", event.timestamp_seconds)
        validate_nonnegative("global_step", event.global_step)
        for name in (
            "environment_steps_per_second",
            "rollout_steps_per_second",
            "optimization_samples_per_second",
            "gpu_utilization_percent",
        ):
            validate_finite_optional(name, getattr(event, name))
        for name in ("gpu_memory_allocated_bytes", "gpu_memory_reserved_bytes"):
            value = getattr(event, name)
            if value is not None:
                validate_nonnegative(name, value)


class EventOrderValidator:
    def __init__(self) -> None:
        self._last_global_step = 0

    def consume(self, event: TelemetryEvent) -> None:
        validate_event(event)
        global_step = int(getattr(event, "global_step", self._last_global_step))
        # Handle final step check: final_global_step for completed/failed/interrupted
        if hasattr(event, "final_global_step"):
            global_step = int(event.final_global_step)
        if global_step < self._last_global_step:
            raise TelemetryValidationError(
                f"global_step regressed from {self._last_global_step} to {global_step}"
            )
        self._last_global_step = global_step


__all__ = [
    "EventOrderValidator",
    "validate_event",
    "validate_finite_optional",
    "validate_nonnegative",
]
