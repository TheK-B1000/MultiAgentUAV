"""Typed telemetry summaries and metric records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Any


@dataclass(frozen=True)
class RewardSummary:
    actor_reward_mean: float
    router_reward_mean: Optional[float]
    sparse_reward_mean: float
    shaping_reward_mean: float
    component_means: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LatentSummary:
    latent_occupancy: Mapping[int, float] = field(default_factory=dict)
    strategy_entropy: Optional[float] = None
    effective_latent_count: Optional[float] = None
    switching_rate: Optional[float] = None
    persistence_rate: Optional[float] = None
    router_entropy: Optional[float] = None
    router_kl: Optional[float] = None


@dataclass(frozen=True)
class TrainingTelemetrySnapshot:
    global_step: int = 0
    rollout_count: int = 0
    optimization_count: int = 0
    episode_count: int = 0
    checkpoint_count: int = 0
    environment_steps: int = 0
    last_reward_summary: Optional[RewardSummary] = None
    last_latent_summary: Optional[LatentSummary] = None
    last_explained_variance: Optional[float] = None


@dataclass(frozen=True)
class PerformanceSummary:
    schema_version: int
    run_id: str
    git_commit: Optional[str]
    preset_name: Optional[str]
    preset_hash: Optional[str]
    checkpoint_hash: Optional[str]
    device: str
    gpu_model: Optional[str]
    pytorch_version: str
    cuda_version: Optional[str]
    telemetry_mode: str
    timing_method: str
    environment_count: int
    rollout_length: int
    total_training_duration: float
    environment_transitions_per_second: Optional[float]
    median_rollout_transitions_per_second: Optional[float]
    median_optimization_samples_per_second: Optional[float]
    checkpoint_load_duration: Optional[float]
    mean_checkpoint_save_duration: Optional[float]
    peak_allocated_cuda_memory: Optional[int]
    peak_reserved_cuda_memory: Optional[int]
    gpu_utilization_summary: Optional[dict[str, Any]] = None


__all__ = [
    "LatentSummary",
    "PerformanceSummary",
    "RewardSummary",
    "TrainingTelemetrySnapshot",
]
