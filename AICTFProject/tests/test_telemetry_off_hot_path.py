"""OFF-mode telemetry hot-path inactivity tests."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rl.custom_ppo.training_telemetry import TrainingTelemetry


class TelemetryOffHotPathTests(unittest.TestCase):
    def _runtime(self) -> SimpleNamespace:
        return SimpleNamespace(
            global_step=0,
            _updates_completed=0,
            env=SimpleNamespace(num_envs=1),
        )

    def test_off_mode_constructs_no_observability_backends(self) -> None:
        cfg = SimpleNamespace(
            training_telemetry_mode="off",
            training_events_jsonl_path="",
            telemetry_events_jsonl_path="",
            gpu_monitor_enabled=True,
            checkpoint_dir=".",
        )
        with patch("rl.custom_ppo.training_telemetry.JSONLineEventWriter") as writer, patch(
            "rl.custom_ppo.training_telemetry.build_gpu_monitor"
        ) as monitor, patch("subprocess.run") as subprocess_run:
            telemetry = TrainingTelemetry(
                cfg=cfg,
                hparams=SimpleNamespace(use_latent_strategy=False, run_id="off_hot_path"),
                curriculum=None,
                reward_shaping_coef=lambda: 0.0,
                runtime=self._runtime(),
            )

        self.assertFalse(telemetry.optional_telemetry_enabled())
        writer.assert_not_called()
        monitor.assert_not_called()
        subprocess_run.assert_not_called()

    def test_off_mode_boundary_methods_do_not_emit_or_time(self) -> None:
        cfg = SimpleNamespace(
            training_telemetry_mode="off",
            training_events_jsonl_path="",
            telemetry_events_jsonl_path="",
            gpu_monitor_enabled=False,
            checkpoint_dir=".",
        )
        telemetry = TrainingTelemetry(
            cfg=cfg,
            hparams=SimpleNamespace(use_latent_strategy=False, run_id="off_hot_path"),
            curriculum=None,
            reward_shaping_coef=lambda: 0.0,
            runtime=self._runtime(),
        )
        telemetry.event_sink.emit = MagicMock()
        with patch("rl.custom_ppo.training_telemetry.TelemetryEnvelope") as envelope, patch(
            "rl.custom_ppo.training_telemetry.time.perf_counter"
        ) as perf_counter, patch("torch.cuda.synchronize") as synchronize, patch(
            "torch.cuda.reset_peak_memory_stats"
        ) as reset_peak, patch("torch.cuda.max_memory_allocated") as max_alloc, patch(
            "torch.cuda.max_memory_reserved"
        ) as max_reserved:
            telemetry.emit_training_started(total_timesteps=10)
            telemetry.emit_training_completed(total_timesteps=10, duration_seconds=1.0, checkpoint_path=None)
            telemetry.emit_training_failed(total_timesteps=10, duration_seconds=1.0, checkpoint_path=None, error=RuntimeError("x"))
            telemetry.emit_training_interrupted(total_timesteps=10, duration_seconds=1.0, checkpoint_path=None, reason="x")

        telemetry.event_sink.emit.assert_not_called()
        envelope.assert_not_called()
        perf_counter.assert_not_called()
        synchronize.assert_not_called()
        reset_peak.assert_not_called()
        max_alloc.assert_not_called()
        max_reserved.assert_not_called()
