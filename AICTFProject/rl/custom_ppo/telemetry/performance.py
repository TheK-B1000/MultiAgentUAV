"""Performance metric calculations and recording."""

from __future__ import annotations

import time
from typing import Optional, Any
import numpy as np

from rl.custom_ppo.telemetry.models import PerformanceSummary
from rl.custom_ppo.telemetry.schemas import PERFORMANCE_METRICS_SCHEMA_VERSION


def rate_per_second(count: int | float, duration_seconds: float) -> Optional[float]:
    if duration_seconds <= 0:
        return None
    return float(count) / float(duration_seconds)


def environment_transitions_per_second(
    transitions: int,
    duration_seconds: float,
) -> Optional[float]:
    return rate_per_second(transitions, duration_seconds)


def rollout_steps_per_second(
    transitions_collected: int,
    rollout_duration_seconds: float,
) -> Optional[float]:
    return rate_per_second(transitions_collected, rollout_duration_seconds)


def optimization_samples_per_second(
    samples_processed: int,
    optimization_duration_seconds: float,
) -> Optional[float]:
    return rate_per_second(samples_processed, optimization_duration_seconds)


class PerformanceRecorder:
    """Records performance events and aggregates them into a PerformanceSummary."""

    def __init__(
        self,
        run_id: str,
        telemetry_mode: str,
        timing_method: str = "wall_clock",
    ) -> None:
        self.run_id = str(run_id)
        self.telemetry_mode = str(telemetry_mode)
        self.timing_method = str(timing_method)

        self.rollout_durations: list[float] = []
        self.rollout_steps: list[int] = []
        
        self.optimization_durations: list[float] = []
        self.optimization_samples: list[int] = []
        
        self.checkpoint_load_durations: list[float] = []
        self.checkpoint_save_durations: list[float] = []
        
        self.cuda_allocated_peaks: list[int] = []
        self.cuda_reserved_peaks: list[int] = []

    def measure_rollout(
        self,
        *,
        duration_seconds: float,
        steps: int,
        gpu_allocated_peak: Optional[int] = None,
        gpu_reserved_peak: Optional[int] = None,
    ) -> None:
        if duration_seconds > 0:
            self.rollout_durations.append(float(duration_seconds))
            self.rollout_steps.append(int(steps))
        if gpu_allocated_peak is not None:
            self.cuda_allocated_peaks.append(int(gpu_allocated_peak))
        if gpu_reserved_peak is not None:
            self.cuda_reserved_peaks.append(int(gpu_reserved_peak))

    def measure_optimization(
        self,
        *,
        duration_seconds: float,
        samples: int,
        gpu_allocated_peak: Optional[int] = None,
        gpu_reserved_peak: Optional[int] = None,
    ) -> None:
        if duration_seconds > 0:
            self.optimization_durations.append(float(duration_seconds))
            self.optimization_samples.append(int(samples))
        if gpu_allocated_peak is not None:
            self.cuda_allocated_peaks.append(int(gpu_allocated_peak))
        if gpu_reserved_peak is not None:
            self.cuda_reserved_peaks.append(int(gpu_reserved_peak))

    def measure_checkpoint_load(self, *, duration_seconds: float) -> None:
        if duration_seconds > 0:
            self.checkpoint_load_durations.append(float(duration_seconds))

    def measure_checkpoint_save(self, *, duration_seconds: float) -> None:
        if duration_seconds > 0:
            self.checkpoint_save_durations.append(float(duration_seconds))

    def summary(
        self,
        *,
        git_commit: Optional[str] = None,
        preset_name: Optional[str] = None,
        preset_hash: Optional[str] = None,
        checkpoint_hash: Optional[str] = None,
        device: str = "cpu",
        gpu_model: Optional[str] = None,
        pytorch_version: str = "",
        cuda_version: Optional[str] = None,
        environment_count: int = 1,
        rollout_length: int = 1,
        total_training_duration: float = 0.0,
        gpu_utilization_summary: Optional[dict[str, Any]] = None,
        total_transitions_collected: Optional[int] = None,
    ) -> PerformanceSummary:
        
        # Calculate mean env transitions per second
        total_steps = sum(self.rollout_steps) if self.rollout_steps else (total_transitions_collected or 0)
        total_rollout_dur = sum(self.rollout_durations) if self.rollout_durations else total_training_duration
        mean_env_tps = (total_steps / total_rollout_dur) if total_rollout_dur > 0 else None
        
        # Calculate median rollout transitions per second
        rollout_tps_list = [
            steps / dur
            for steps, dur in zip(self.rollout_steps, self.rollout_durations)
            if dur > 0
        ]
        median_rollout_tps = float(np.median(rollout_tps_list)) if rollout_tps_list else None
        
        # Calculate median optimization samples per second
        opt_sps_list = [
            samples / dur
            for samples, dur in zip(self.optimization_samples, self.optimization_durations)
            if dur > 0
        ]
        median_opt_sps = float(np.median(opt_sps_list)) if opt_sps_list else None
        
        checkpoint_load = self.checkpoint_load_durations[0] if self.checkpoint_load_durations else None
        mean_checkpoint_save = float(np.mean(self.checkpoint_save_durations)) if self.checkpoint_save_durations else None
        
        peak_allocated = int(max(self.cuda_allocated_peaks)) if self.cuda_allocated_peaks else None
        peak_reserved = int(max(self.cuda_reserved_peaks)) if self.cuda_reserved_peaks else None

        return PerformanceSummary(
            schema_version=PERFORMANCE_METRICS_SCHEMA_VERSION,
            run_id=self.run_id,
            git_commit=git_commit,
            preset_name=preset_name,
            preset_hash=preset_hash,
            checkpoint_hash=checkpoint_hash,
            device=str(device),
            gpu_model=gpu_model,
            pytorch_version=str(pytorch_version),
            cuda_version=cuda_version,
            telemetry_mode=self.telemetry_mode,
            timing_method=self.timing_method,
            environment_count=int(environment_count),
            rollout_length=int(rollout_length),
            total_training_duration=float(total_training_duration),
            environment_transitions_per_second=mean_env_tps,
            median_rollout_transitions_per_second=median_rollout_tps,
            median_optimization_samples_per_second=median_opt_sps,
            checkpoint_load_duration=checkpoint_load,
            mean_checkpoint_save_duration=mean_checkpoint_save,
            peak_allocated_cuda_memory=peak_allocated,
            peak_reserved_cuda_memory=peak_reserved,
            gpu_utilization_summary=gpu_utilization_summary,
        )


__all__ = [
    "PerformanceRecorder",
    "environment_transitions_per_second",
    "optimization_samples_per_second",
    "rate_per_second",
    "rollout_steps_per_second",
]
