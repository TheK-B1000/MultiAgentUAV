"""Integration tests for Phase 6.1 telemetry modes and event pipelines."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.training_telemetry import TrainingTelemetry
from rl.custom_ppo.telemetry.schemas import TrainingTelemetryMode


class Phase6TelemetryIntegrationTests(unittest.TestCase):
    def test_telemetry_modes_parsing(self) -> None:
        cfg = PPOConfig()
        cfg.training_telemetry_mode = "off"
        telemetry = TrainingTelemetry(
            cfg=cfg,
            hparams=SimpleNamespace(use_latent_strategy=False, run_id="run_off"),
            curriculum=None,
            reward_shaping_coef=lambda: 0.0,
            runtime=SimpleNamespace(global_step=0),
        )
        self.assertEqual(telemetry.telemetry_mode, TrainingTelemetryMode.OFF)
        self.assertFalse(telemetry.optional_telemetry_enabled())

        cfg.training_telemetry_mode = "basic"
        telemetry = TrainingTelemetry(
            cfg=cfg,
            hparams=SimpleNamespace(use_latent_strategy=False, run_id="run_basic"),
            curriculum=None,
            reward_shaping_coef=lambda: 0.0,
            runtime=SimpleNamespace(global_step=0),
        )
        self.assertEqual(telemetry.telemetry_mode, TrainingTelemetryMode.BASIC)
        self.assertTrue(telemetry.optional_telemetry_enabled())

    def test_lifecycle_event_emission_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            events_path = Path(td) / "events.jsonl"
            cfg = PPOConfig()
            cfg.training_events_jsonl_path = str(events_path)
            cfg.training_telemetry_mode = "basic"

            telemetry = TrainingTelemetry(
                cfg=cfg,
                hparams=SimpleNamespace(use_latent_strategy=False, run_id="run_lifecycle"),
                curriculum=None,
                reward_shaping_coef=lambda: 0.0,
                runtime=SimpleNamespace(global_step=10),
            )

            # Emit started and completed
            telemetry.emit_training_started(total_timesteps=100)
            telemetry.emit_training_completed(total_timesteps=100, duration_seconds=5.0)
            telemetry.close_e3_step_telemetry()

            self.assertTrue(events_path.exists())
            lines = events_path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
