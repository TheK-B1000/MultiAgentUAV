from __future__ import annotations

import dataclasses
import importlib
import json
import math
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from rl.custom_ppo.telemetry import (
    BufferedTelemetrySink,
    CheckpointLoaded,
    CheckpointSaved,
    CompositeTelemetrySink,
    NullTelemetrySink,
    OptimizationCompleted,
    PerformanceSample,
    RewardSummary,
    RolloutCompleted,
    TrainingCompleted,
    TrainingFailed,
    TrainingStarted,
    TrainingTelemetryAggregator,
)
from rl.custom_ppo.telemetry.errors import TelemetryValidationError
from rl.custom_ppo.telemetry.gpu_monitor import build_gpu_monitor
from rl.custom_ppo.telemetry.performance import (
    environment_transitions_per_second,
    optimization_samples_per_second,
    rollout_steps_per_second,
)
from rl.custom_ppo.telemetry.schemas import (
    PERFORMANCE_METRICS_SCHEMA_VERSION,
    TRAINING_EVENTS_SCHEMA_VERSION,
    TrainingTelemetryMode,
    coerce_telemetry_mode,
)
from rl.custom_ppo.telemetry.timing import PhaseTimer
from rl.custom_ppo.telemetry.validation import EventOrderValidator, validate_event
from rl.custom_ppo.telemetry.writers.json_writer import JSONLineEventWriter, event_to_record

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def _optimization_event(global_step: int = 8) -> OptimizationCompleted:
    return OptimizationCompleted(
        run_id="run",
        global_step=global_step,
        optimization_index=1,
        duration_seconds=0.25,
        samples_processed=32,
        minibatches_processed=4,
        optimizer_updates=1,
        optimization_samples_per_second=128.0,
        minibatches_per_second=16.0,
        optimizer_updates_per_second=4.0,
        policy_loss=0.1,
        value_loss=0.2,
        entropy=0.3,
        approx_kl=0.01,
        clip_fraction=0.125,
        explained_variance=None,
        gpu_memory_allocated_peak_bytes=None,
        gpu_memory_reserved_peak_bytes=None,
    )


class Phase6TelemetryModelTests(unittest.TestCase):
    def test_typed_events_are_immutable(self) -> None:
        event = _optimization_event()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            event.global_step = 9  # type: ignore[misc]

    def test_event_serialization_is_deterministic(self) -> None:
        event = _optimization_event()
        first = json.dumps(event_to_record(event), sort_keys=True, separators=(",", ":"))
        second = json.dumps(event_to_record(event), sort_keys=True, separators=(",", ":"))
        self.assertEqual(first, second)
        self.assertIn('"event_type":"OptimizationCompleted"', first)
        self.assertIn(f'"schema_version":{TRAINING_EVENTS_SCHEMA_VERSION}', first)

    def test_unavailable_metrics_remain_none(self) -> None:
        record = event_to_record(_optimization_event())
        self.assertIsNone(record["payload"]["explained_variance"])

    def test_nan_and_inf_are_rejected(self) -> None:
        bad = dataclasses.replace(_optimization_event(), value_loss=math.inf)
        with self.assertRaises(TelemetryValidationError):
            validate_event(bad)

    def test_negative_duration_fails_validation(self) -> None:
        bad = dataclasses.replace(_optimization_event(), duration_seconds=-0.1)
        with self.assertRaises(TelemetryValidationError):
            validate_event(bad)

    def test_global_steps_cannot_regress(self) -> None:
        validator = EventOrderValidator()
        validator.consume(_optimization_event(global_step=10))
        with self.assertRaises(TelemetryValidationError):
            validator.consume(_optimization_event(global_step=9))

    def test_modes_coerce_without_config_schema_drift(self) -> None:
        self.assertEqual(coerce_telemetry_mode("off"), TrainingTelemetryMode.OFF)
        self.assertEqual(coerce_telemetry_mode("basic"), TrainingTelemetryMode.BASIC)
        self.assertEqual(coerce_telemetry_mode("full"), TrainingTelemetryMode.FULL)
        self.assertEqual(coerce_telemetry_mode("unknown"), TrainingTelemetryMode.FULL)

    def test_lifecycle_and_checkpoint_events_serialize(self) -> None:
        events = [
            TrainingStarted(
                run_id="run",
                timestamp_seconds=1.0,
                global_step=0,
                requested_total_steps=128,
                device="cpu",
                preset_name=None,
                preset_hash=None,
                checkpoint_path=None,
                checkpoint_hash=None,
                telemetry_mode="basic",
            ),
            TrainingCompleted(
                run_id="run",
                timestamp_seconds=1.0,
                final_global_step=128,
                duration_seconds=1.5,
                checkpoint_path=None,
                status="completed",
            ),
            TrainingFailed(
                run_id="run",
                timestamp_seconds=1.0,
                final_global_step=64,
                duration_seconds=0.75,
                checkpoint_path=None,
                exception_type="RuntimeError",
                exception_message="boom",
                phase="phase_1",
            ),
            CheckpointSaved(
                run_id="run",
                timestamp_seconds=1.0,
                global_step=64,
                checkpoint_path="ckpt.zip",
                checkpoint_hash=None,
                save_duration_seconds=0.1,
                checkpoint_size_bytes=None,
                parent_checkpoint_hash=None,
                preset_hash=None,
            ),
            CheckpointLoaded(
                run_id="run",
                timestamp_seconds=1.0,
                global_step=64,
                checkpoint_path="ckpt.zip",
                checkpoint_hash=None,
                load_duration_seconds=0.2,
                source_observation_channels=None,
                target_observation_channels=None,
                migration_ids=(),
                behavioral_equivalence_result=None,
                device="cpu",
            ),
            PerformanceSample(
                timestamp_seconds=1.0,
                global_step=64,
                phase="update",
                environment_steps_per_second=10.0,
                rollout_steps_per_second=8.0,
                optimization_samples_per_second=12.0,
                gpu_utilization_percent=None,
                gpu_memory_allocated_bytes=None,
                gpu_memory_reserved_bytes=None,
            ),
        ]
        for event in events:
            record = event_to_record(event)
            self.assertEqual(record["event_type"], type(event).__name__)


class Phase6TelemetrySinkTests(unittest.TestCase):
    def test_null_sink_does_not_mutate_training_event(self) -> None:
        event = _optimization_event()
        before = dataclasses.asdict(event)
        NullTelemetrySink().emit(event)
        self.assertEqual(dataclasses.asdict(event), before)

    def test_composite_sink_sends_identical_event_to_each_child(self) -> None:
        first = BufferedTelemetrySink()
        second = BufferedTelemetrySink()
        event = _optimization_event()
        CompositeTelemetrySink([first, second]).emit(event)
        self.assertIs(first.events[0], event)
        self.assertIs(second.events[0], event)

    def test_jsonl_events_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "events.jsonl"
            writer = JSONLineEventWriter(str(path))
            writer.emit(_optimization_event())
            writer.close()
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[0]["event_type"], "OptimizationCompleted")
        self.assertEqual(rows[0]["payload"]["explained_variance"], None)

    def test_aggregator_tracks_coarse_counts(self) -> None:
        aggregator = TrainingTelemetryAggregator()
        reward = RewardSummary(
            actor_reward_mean=1.0,
            router_reward_mean=None,
            sparse_reward_mean=0.5,
            shaping_reward_mean=0.25,
            component_means={"reward_total": 1.0},
        )
        aggregator.consume(
            RolloutCompleted(
                run_id="run",
                global_step=32,
                rollout_index=1,
                vector_environment_count=2,
                vector_steps=16,
                environment_transitions=32,
                agent_transitions=None,
                duration_seconds=1.0,
                environment_step_duration_seconds=0.5,
                policy_inference_duration_seconds=0.3,
                transition_build_duration_seconds=0.1,
                buffer_write_duration_seconds=0.05,
                episode_bookkeeping_duration_seconds=0.05,
                environment_transitions_per_second=32.0,
                rollout_transitions_per_second=32.0,
                completed_episode_count=2,
                episode_return_mean=1.0,
                episode_length_mean=20.0,
                gpu_memory_allocated_peak_bytes=None,
                gpu_memory_reserved_peak_bytes=None,
                reward_summary=reward,
                latent_summary=None,
            )
        )
        aggregator.consume(_optimization_event(global_step=32))
        snap = aggregator.snapshot()
        self.assertEqual(snap.rollout_count, 1)
        self.assertEqual(snap.optimization_count, 1)
        self.assertEqual(snap.environment_steps, 32)

    def test_training_telemetry_opt_in_jsonl_event_path(self) -> None:
        from rl.custom_ppo.training_telemetry import TrainingTelemetry

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "training_events.jsonl"
            telemetry = TrainingTelemetry(
                cfg=SimpleNamespace(training_events_jsonl_path=str(path)),
                hparams=SimpleNamespace(use_latent_strategy=False),
                curriculum=None,
                reward_shaping_coef=lambda: 0.0,
                runtime=SimpleNamespace(global_step=12),
            )
            telemetry._emit_optimization_completed(
                {"explained_variance": 0.5},
                {"policy_loss": 0.1, "value_loss": 0.2, "approx_kl": 0.01},
                SimpleNamespace(pos=2, n_envs=3),
            )
            telemetry.close_e3_step_telemetry()
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[0]["event_type"], "OptimizationCompleted")
        self.assertEqual(rows[0]["payload"]["samples_processed"], 6)

    def test_training_telemetry_default_off_writes_no_optional_artifact(self) -> None:
        from rl.custom_ppo.training_telemetry import TrainingTelemetry

        with tempfile.TemporaryDirectory() as td:
            telemetry = TrainingTelemetry(
                cfg=SimpleNamespace(checkpoint_dir=td),
                hparams=SimpleNamespace(use_latent_strategy=False, run_id="run", run_pid=1),
                curriculum=None,
                reward_shaping_coef=lambda: 0.0,
                runtime=SimpleNamespace(global_step=12, _updates_completed=1),
            )
            self.assertEqual(telemetry.telemetry_mode, TrainingTelemetryMode.OFF)
            self.assertIsNone(telemetry.write_performance_summary(training_duration_seconds=1.0))
            self.assertFalse((Path(td) / "performance_summary.json").exists())

    def test_training_telemetry_basic_writes_performance_summary(self) -> None:
        from rl.custom_ppo.training_telemetry import TrainingTelemetry

        with tempfile.TemporaryDirectory() as td:
            telemetry = TrainingTelemetry(
                cfg=SimpleNamespace(checkpoint_dir=td, training_telemetry_mode="basic"),
                hparams=SimpleNamespace(use_latent_strategy=False, run_id="run", run_pid=1),
                curriculum=None,
                reward_shaping_coef=lambda: 0.0,
                runtime=SimpleNamespace(global_step=12, _updates_completed=1),
            )
            telemetry._total_transitions_collected = 100
            path = telemetry.write_performance_summary(training_duration_seconds=2.0)
            self.assertEqual(path, str(Path(td) / "performance_summary.json"))
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual(payload["telemetry_mode"], "basic")
        self.assertEqual(payload["environment_transitions_per_second"], 50.0)


class Phase6PerformanceTests(unittest.TestCase):
    def test_throughput_uses_none_for_unavailable_duration(self) -> None:
        self.assertIsNone(environment_transitions_per_second(100, 0.0))
        self.assertEqual(rollout_steps_per_second(100, 2.0), 50.0)
        self.assertEqual(optimization_samples_per_second(64, 4.0), 16.0)

    def test_timer_returns_finite_nonnegative_duration(self) -> None:
        with PhaseTimer("unit-test") as timer:
            time.sleep(0.001)
        self.assertIsNotNone(timer.result)
        self.assertGreaterEqual(timer.result.duration_seconds, 0.0)
        self.assertEqual(timer.result.timing_method, "wall_clock")

    def test_gpu_monitor_disabled_does_not_start_nvml(self) -> None:
        monitor = build_gpu_monitor(False)
        self.assertEqual(monitor.status, "unavailable")
        monitor.start()
        self.assertEqual(monitor.samples(), [])

    def test_performance_schema_version_is_public(self) -> None:
        self.assertEqual(PERFORMANCE_METRICS_SCHEMA_VERSION, 1)


class Phase6TrainingInvarianceTests(unittest.TestCase):
    def _tiny_config(self, *, run_tag: str, checkpoint_dir: str) -> object:
        from rl.train_ppo import PPOConfig

        cfg = PPOConfig()
        cfg.seed = 123
        cfg.total_timesteps = 4
        cfg.n_envs = 1
        cfg.n_steps = 4
        cfg.batch_size = 4
        cfg.n_epochs = 1
        cfg.max_decision_steps = 1
        cfg.device = "cpu"
        cfg.use_latent_strategy = True
        cfg.latent_resample_every_n = 0
        cfg.enable_metrics_csv = True
        cfg.enable_episode_csv = True
        cfg.enable_progress_bar = False
        cfg.fresh_metrics_csv = True
        cfg.run_tag = run_tag
        cfg.checkpoint_dir = checkpoint_dir
        return cfg

    def test_basic_telemetry_preserves_tiny_training_state(self) -> None:
        from rl.train_ppo import train_ppo

        with tempfile.TemporaryDirectory() as td:
            off_cfg = self._tiny_config(run_tag="telemetry_off", checkpoint_dir=td)
            basic_cfg = self._tiny_config(run_tag="telemetry_basic", checkpoint_dir=td)
            setattr(basic_cfg, "training_telemetry_mode", "basic")
            setattr(basic_cfg, "training_events_jsonl_path", str(Path(td) / "training_events.jsonl"))

            train_ppo(off_cfg)
            train_ppo(basic_cfg)

            import torch

            off_payload = torch.load(
                str(Path(td) / "final_telemetry_off.zip"),
                map_location="cpu",
                weights_only=False,
            )
            basic_payload = torch.load(
                str(Path(td) / "final_telemetry_basic.zip"),
                map_location="cpu",
                weights_only=False,
            )

            self.assertEqual(off_payload["global_step"], basic_payload["global_step"])
            self.assertEqual(off_payload["updates_completed"], basic_payload["updates_completed"])
            self.assertEqual(
                set(off_payload["model_state_dict"]),
                set(basic_payload["model_state_dict"]),
            )
            for key, tensor in off_payload["model_state_dict"].items():
                self.assertTrue(
                    torch.equal(tensor, basic_payload["model_state_dict"][key]),
                    msg=key,
                )
            self.assertTrue((Path(td) / "training_events.jsonl").is_file())
            self.assertTrue((Path(td) / "performance_summary.json").is_file())


class Phase6ArchitectureTests(unittest.TestCase):
    def test_compatibility_imports_remain_functional(self) -> None:
        module = importlib.import_module("rl.custom_ppo.training_telemetry")
        self.assertTrue(hasattr(module, "TrainingTelemetry"))

    def test_writers_do_not_import_rollout_or_trainer_modules(self) -> None:
        writer_modules = (
            "rl.custom_ppo.telemetry.writers.console",
            "rl.custom_ppo.telemetry.writers.csv_writer",
            "rl.custom_ppo.telemetry.writers.json_writer",
            "rl.custom_ppo.telemetry.writers.artifact_writer",
        )
        forbidden = ("rollout_collector", "rollout.collector", "trainer", "ppo_updater")
        for module_name in writer_modules:
            module = importlib.import_module(module_name)
            names = " ".join(sorted(module.__dict__))
            for item in forbidden:
                self.assertNotIn(item, names)

    def test_telemetry_models_do_not_import_policy_implementations(self) -> None:
        module = importlib.import_module("rl.custom_ppo.telemetry.models")
        names = " ".join(sorted(module.__dict__))
        self.assertNotIn("SharedActorCentralizedCritic", names)
        self.assertNotIn("LatentConditionedActor", names)


if __name__ == "__main__":
    unittest.main()
