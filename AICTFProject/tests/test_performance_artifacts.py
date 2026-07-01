"""Tests for atomic file writing of performance summaries and CSV samples."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rl.custom_ppo.training_telemetry import TrainingTelemetry


class PerformanceArtifactsTests(unittest.TestCase):
    def test_atomic_performance_summary_writing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            summary_path = Path(td) / "perf_summary.json"
            cfg = SimpleNamespace(
                performance_summary_path=str(summary_path),
                training_telemetry_mode="basic",
                device="cpu",
                cli_preset=None,
            )

            runtime = SimpleNamespace(
                global_step=100,
                env=SimpleNamespace(
                    num_envs=16,
                )
            )

            telemetry = TrainingTelemetry(
                cfg=cfg,
                hparams=SimpleNamespace(use_latent_strategy=False, run_id="run_perf", run_pid=123),
                curriculum=None,
                reward_shaping_coef=lambda: 0.0,
                runtime=runtime,
            )

            # Record some performance stats
            telemetry.performance_recorder.measure_rollout(duration_seconds=1.5, steps=100)
            telemetry.performance_recorder.measure_optimization(duration_seconds=2.0, samples=200)

            # Write summary
            res_path = telemetry.write_performance_summary(training_duration_seconds=10.0)
            self.assertEqual(res_path, str(summary_path))
            self.assertTrue(summary_path.exists())

            # Read and verify content
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(data["run_id"], "run_perf")
            self.assertEqual(data["telemetry_mode"], "basic")
            self.assertEqual(data["environment_count"], 16)
            self.assertAlmostEqual(data["total_training_duration"], 10.0)
