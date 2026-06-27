"""Typed telemetry infrastructure for custom PPO training."""

from rl.custom_ppo.telemetry.aggregator import TrainingTelemetryAggregator
from rl.custom_ppo.telemetry.emitter import (
    BufferedTelemetrySink,
    CompositeTelemetrySink,
    NullTelemetrySink,
    SafeTelemetrySink,
    TelemetrySink,
)
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
from rl.custom_ppo.telemetry.models import (
    LatentSummary,
    PerformanceSummary,
    RewardSummary,
    TrainingTelemetrySnapshot,
)
from rl.custom_ppo.telemetry.schemas import TrainingTelemetryMode

__all__ = [
    "BufferedTelemetrySink",
    "CheckpointLoaded",
    "CheckpointSaved",
    "CompositeTelemetrySink",
    "EpisodeCompleted",
    "EpisodesCompleted",
    "LatentSummary",
    "NullTelemetrySink",
    "OptimizationCompleted",
    "PerformanceSample",
    "PerformanceSummary",
    "RewardSummary",
    "RolloutCompleted",
    "SafeTelemetrySink",
    "TelemetryEvent",
    "TelemetrySink",
    "TrainingTelemetryAggregator",
    "TrainingCompleted",
    "TrainingFailed",
    "TrainingInterrupted",
    "TrainingStarted",
    "TrainingTelemetryMode",
    "TrainingTelemetrySnapshot",
]
