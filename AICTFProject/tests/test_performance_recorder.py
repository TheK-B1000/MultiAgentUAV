"""Unit tests for the PerformanceRecorder class."""

from __future__ import annotations

import unittest
from rl.custom_ppo.telemetry.performance import PerformanceRecorder
from rl.custom_ppo.telemetry.models import PerformanceSummary


class PerformanceRecorderTests(unittest.TestCase):
    def test_performance_recorder_aggregation(self) -> None:
        recorder = PerformanceRecorder(run_id="test_run", telemetry_mode="basic")

        # Record rollout timings
        recorder.measure_rollout(duration_seconds=2.0, steps=100)
        recorder.measure_rollout(duration_seconds=3.0, steps=150)

        # Record optimization timings
        recorder.measure_optimization(duration_seconds=4.0, samples=400)
        recorder.measure_optimization(duration_seconds=6.0, samples=600)

        # Check summary stats
        summary = recorder.summary(
            device="cpu",
            pytorch_version="2.0",
            environment_count=1,
            rollout_length=100,
            total_training_duration=15.0,
        )

        self.assertIsInstance(summary, PerformanceSummary)
        self.assertEqual(summary.run_id, "test_run")
        self.assertEqual(summary.telemetry_mode, "basic")
        
        # rollout: total steps = 250, total duration = 5.0 -> mean env transitions/sec = 50.0
        self.assertAlmostEqual(summary.environment_transitions_per_second, 50.0)
        
        # medians
        # rollout: [100/2, 150/3] = [50, 50] -> median rollout transitions/sec = 50.0
        self.assertAlmostEqual(summary.median_rollout_transitions_per_second, 50.0)
        
        # opt: [400/4, 600/6] = [100, 100] -> median optimization samples/sec = 100.0
        self.assertAlmostEqual(summary.median_optimization_samples_per_second, 100.0)
