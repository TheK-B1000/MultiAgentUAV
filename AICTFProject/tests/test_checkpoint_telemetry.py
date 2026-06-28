"""Tests for checkpoint save/load event telemetry, sizing, and lineage hashing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from rl.custom_ppo.training_telemetry import TrainingTelemetry
from rl.custom_ppo.telemetry.events import CheckpointLoaded, CheckpointSaved


class MockModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.grid_shape = (8, 64, 64)
        self.param = nn.Parameter(torch.zeros(1))


class CheckpointTelemetryTests(unittest.TestCase):
    def test_checkpoint_lineage_tracking(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            events_path = Path(td) / "events.jsonl"
            cfg = SimpleNamespace(
                training_events_jsonl_path=str(events_path),
                training_telemetry_mode="basic",
                device="cpu",
                cli_preset=None,
            )

            model = MockModel()
            runtime = SimpleNamespace(
                global_step=50,
                model=model,
                env=SimpleNamespace(
                    observation_space=SimpleNamespace(
                        spaces={
                            "grid": SimpleNamespace(
                                shape=(8, 64, 64)
                            )
                        }
                    )
                ),
            )

            telemetry = TrainingTelemetry(
                cfg=cfg,
                hparams=SimpleNamespace(use_latent_strategy=False, run_id="run_ckpt"),
                curriculum=None,
                reward_shaping_coef=lambda: 0.0,
                runtime=runtime,
            )

            # Create a mock checkpoint file
            ckpt_file = Path(td) / "ckpt_0.zip"
            payload = {
                "model_state_dict": model.state_dict(),
                "global_step": 50,
            }
            torch.save(payload, str(ckpt_file))

            # Emit loaded
            telemetry.emit_checkpoint_loaded(path=str(ckpt_file), duration_seconds=0.5)

            # Check that parent checkpoint hash was stored
            self.assertIsNotNone(telemetry._parent_checkpoint_hash)
            first_hash = telemetry._parent_checkpoint_hash

            # Emit saved
            new_ckpt_file = Path(td) / "ckpt_1.zip"
            torch.save(payload, str(new_ckpt_file))
            telemetry.emit_checkpoint_saved(path=str(new_ckpt_file), duration_seconds=0.3)

            # Verify the output events have the parent hash linkage
            telemetry.close_e3_step_telemetry()
            import json
            lines = events_path.read_text().splitlines()
            loaded_event = json.loads(lines[0])["payload"]
            saved_event = json.loads(lines[1])["payload"]

            self.assertEqual(saved_event["parent_checkpoint_hash"], first_hash)
            self.assertIsNotNone(saved_event["checkpoint_hash"])
            self.assertIsNotNone(saved_event["checkpoint_size_bytes"])
            self.assertGreater(saved_event["checkpoint_size_bytes"], 0)

    def test_checkpoint_loaded_uses_non_empty_run_id_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            events_path = Path(td) / "events.jsonl"
            cfg = SimpleNamespace(
                training_events_jsonl_path=str(events_path),
                training_telemetry_mode="basic",
                device="cpu",
                cli_preset=None,
                run_tag="benchmark_run",
            )
            model = MockModel()
            runtime = SimpleNamespace(
                global_step=0,
                model=model,
                env=SimpleNamespace(
                    observation_space=SimpleNamespace(
                        spaces={"grid": SimpleNamespace(shape=(8, 64, 64))}
                    )
                ),
            )
            telemetry = TrainingTelemetry(
                cfg=cfg,
                hparams=SimpleNamespace(use_latent_strategy=False, run_id=""),
                curriculum=None,
                reward_shaping_coef=lambda: 0.0,
                runtime=runtime,
            )
            ckpt_file = Path(td) / "ckpt_0.zip"
            torch.save({"model_state_dict": model.state_dict(), "global_step": 0}, str(ckpt_file))

            telemetry.emit_checkpoint_loaded(path=str(ckpt_file), duration_seconds=0.1)
            telemetry.close_e3_step_telemetry()

            import json
            row = json.loads(events_path.read_text().splitlines()[0])
            self.assertEqual(row["run_id"], "benchmark_run")
            self.assertEqual(row["payload"]["run_id"], "benchmark_run")

