"""Tests for training lifecycle event emissions under different outcomes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rl.custom_ppo.training_telemetry import TrainingTelemetry
from rl.custom_ppo.telemetry.events import (
    TrainingCompleted,
    TrainingFailed,
    TrainingInterrupted,
    TrainingStarted,
)


class TrainingLifecycleEventsTests(unittest.TestCase):
    def test_lifecycle_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            events_path = Path(td) / "events.jsonl"
            cfg = SimpleNamespace(
                training_events_jsonl_path=str(events_path),
                training_telemetry_mode="basic",
                device="cpu",
                cli_preset=None,
            )

            # Test Success Lifecycle
            telemetry = TrainingTelemetry(
                cfg=cfg,
                hparams=SimpleNamespace(use_latent_strategy=False, run_id="run_success"),
                curriculum=None,
                reward_shaping_coef=lambda: 0.0,
                runtime=SimpleNamespace(global_step=0),
            )
            telemetry.emit_training_started(total_timesteps=100)
            telemetry.emit_training_completed(total_timesteps=100, duration_seconds=1.0)
            telemetry.close_e3_step_telemetry()

            # Test Failure Lifecycle
            telemetry2 = TrainingTelemetry(
                cfg=cfg,
                hparams=SimpleNamespace(use_latent_strategy=False, run_id="run_failed"),
                curriculum=None,
                reward_shaping_coef=lambda: 0.0,
                runtime=SimpleNamespace(global_step=0),
            )
            telemetry2.emit_training_started(total_timesteps=100)
            telemetry2.emit_training_failed(total_timesteps=100, duration_seconds=1.0, error=RuntimeError("test error"))
            telemetry2.close_e3_step_telemetry()

            # Test Interrupted Lifecycle
            telemetry3 = TrainingTelemetry(
                cfg=cfg,
                hparams=SimpleNamespace(use_latent_strategy=False, run_id="run_interrupted"),
                curriculum=None,
                reward_shaping_coef=lambda: 0.0,
                runtime=SimpleNamespace(global_step=0),
            )
            telemetry3.emit_training_started(total_timesteps=100)
            telemetry3.emit_training_interrupted(total_timesteps=100, duration_seconds=1.0, reason="KeyboardInterrupt")
            telemetry3.close_e3_step_telemetry()
