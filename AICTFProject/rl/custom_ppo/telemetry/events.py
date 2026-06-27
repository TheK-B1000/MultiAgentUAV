"""Immutable telemetry event types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Tuple


@dataclass(frozen=True)
class TrainingStarted:
    run_id: str
    timestamp_seconds: float
    global_step: int
    requested_total_steps: int
    device: str
    preset_name: Optional[str]
    preset_hash: Optional[str]
    checkpoint_path: Optional[str]
    checkpoint_hash: Optional[str]
    telemetry_mode: str


@dataclass(frozen=True)
class TrainingCompleted:
    run_id: str
    timestamp_seconds: float
    final_global_step: int
    duration_seconds: float
    checkpoint_path: Optional[str]
    status: str


@dataclass(frozen=True)
class TrainingFailed:
    run_id: str
    timestamp_seconds: float
    final_global_step: int
    duration_seconds: float
    checkpoint_path: Optional[str]
    exception_type: str
    exception_message: str
    phase: str


@dataclass(frozen=True)
class TrainingInterrupted:
    run_id: str
    timestamp_seconds: float
    final_global_step: int
    duration_seconds: float
    checkpoint_path: Optional[str]
    reason: str


from rl.custom_ppo.telemetry.models import RewardSummary, LatentSummary


@dataclass(frozen=True)
class RolloutCompleted:
    run_id: str
    global_step: int
    rollout_index: int

    vector_environment_count: int
    vector_steps: int
    environment_transitions: int
    agent_transitions: Optional[int]

    duration_seconds: float
    environment_step_duration_seconds: Optional[float]
    policy_inference_duration_seconds: Optional[float]
    transition_build_duration_seconds: Optional[float]
    buffer_write_duration_seconds: Optional[float]
    episode_bookkeeping_duration_seconds: Optional[float]

    environment_transitions_per_second: float
    rollout_transitions_per_second: float

    completed_episode_count: int
    episode_return_mean: Optional[float]
    episode_length_mean: Optional[float]

    gpu_memory_allocated_peak_bytes: Optional[int]
    gpu_memory_reserved_peak_bytes: Optional[int]

    reward_summary: RewardSummary
    latent_summary: Optional[LatentSummary] = None


@dataclass(frozen=True)
class OptimizationCompleted:
    run_id: str
    global_step: int
    optimization_index: int

    duration_seconds: float
    samples_processed: int
    minibatches_processed: int
    optimizer_updates: int

    optimization_samples_per_second: float
    minibatches_per_second: Optional[float]
    optimizer_updates_per_second: Optional[float]

    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: Optional[float]

    gpu_memory_allocated_peak_bytes: Optional[int]
    gpu_memory_reserved_peak_bytes: Optional[int]


@dataclass(frozen=True)
class EpisodeCompleted:
    run_id: str
    global_step: int
    environment_index: int
    episode_return: float
    episode_length: int
    score_for: Optional[int]
    score_against: Optional[int]
    won: Optional[bool]
    opponent_name: Optional[str]
    map_name: Optional[str]
    terminal_reason: Optional[str]


@dataclass(frozen=True)
class EpisodesCompleted:
    run_id: str
    global_step: int
    episodes: Tuple[EpisodeCompleted, ...]


@dataclass(frozen=True)
class CheckpointSaved:
    run_id: str
    timestamp_seconds: float
    global_step: int
    checkpoint_path: str
    checkpoint_hash: Optional[str]
    save_duration_seconds: float
    checkpoint_size_bytes: Optional[int]
    parent_checkpoint_hash: Optional[str]
    preset_hash: Optional[str]
    checkpoint_write_duration_seconds: Optional[float] = None
    checkpoint_hash_duration_seconds: Optional[float] = None
    checkpoint_total_duration_seconds: Optional[float] = None


@dataclass(frozen=True)
class CheckpointLoaded:
    run_id: str
    timestamp_seconds: float
    global_step: int
    checkpoint_path: str
    checkpoint_hash: Optional[str]
    load_duration_seconds: float
    source_observation_channels: Optional[int]
    target_observation_channels: Optional[int]
    migration_ids: Tuple[str, ...]
    behavioral_equivalence_result: Optional[str]
    device: str
    archive_read_duration: Optional[float] = None
    model_construction_duration: Optional[float] = None
    state_load_duration: Optional[float] = None
    migration_duration: Optional[float] = None
    behavioral_equivalence_duration: Optional[float] = None
    hash_duration: Optional[float] = None
    total_duration: Optional[float] = None


@dataclass(frozen=True)
class PerformanceSample:
    timestamp_seconds: float
    global_step: int
    phase: str
    environment_steps_per_second: Optional[float]
    rollout_steps_per_second: Optional[float]
    optimization_samples_per_second: Optional[float]
    gpu_utilization_percent: Optional[float]
    gpu_memory_allocated_bytes: Optional[int]
    gpu_memory_reserved_bytes: Optional[int]


TelemetryEvent = Union[
    TrainingStarted,
    TrainingCompleted,
    TrainingFailed,
    TrainingInterrupted,
    RolloutCompleted,
    OptimizationCompleted,
    EpisodeCompleted,
    EpisodesCompleted,
    CheckpointSaved,
    CheckpointLoaded,
    PerformanceSample,
]


@dataclass(frozen=True)
class TelemetryEnvelope:
    schema_version: int
    event_type: str
    run_id: str
    sequence: int
    timestamp_seconds: float
    payload: TelemetryEvent


__all__ = [
    "CheckpointLoaded",
    "CheckpointSaved",
    "EpisodeCompleted",
    "EpisodesCompleted",
    "OptimizationCompleted",
    "PerformanceSample",
    "RolloutCompleted",
    "TelemetryEvent",
    "TelemetryEnvelope",
    "TrainingCompleted",
    "TrainingFailed",
    "TrainingInterrupted",
    "TrainingStarted",
]
