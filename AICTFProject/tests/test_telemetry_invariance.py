"""Invariance tests verifying that telemetry modes do not alter RNG or weight updates."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import copy
import torch
import numpy as np

from game_field_gpu import GPUCTFVecEnv
from rl.config.ppo_config import PPOConfig
from rl.custom_ppo import CustomPPOTrainer
from rl.train_ppo import (
    _clamp_runtime_config_for_team_size,
    _ensure_cuda_or_fallback,
    _resolve_initial_opponent_and_phase,
    set_global_seed,
)
from rl.training.env_factory import build_training_env


class TelemetryInvarianceTests(unittest.TestCase):
    def _run_mini_training(self, mode: str, output_dir: str) -> tuple[dict, list[torch.Tensor]]:
        # Set global seed for determinism
        set_global_seed(42)

        cfg = PPOConfig()
        cfg.seed = 42
        cfg.device = "cpu"
        cfg.n_envs = 2
        cfg.n_steps = 4
        cfg.batch_size = 8
        cfg.n_epochs = 1
        cfg.training_telemetry_mode = mode
        cfg.load_path = None
        cfg.checkpoint_dir = output_dir
        cfg.mode = "FIXED_OPPONENT"
        cfg.fixed_opponent_tag = "OP3"
        cfg.use_latent_strategy = True
        cfg.enable_metrics_csv = False
        cfg.gpu_native_env = True
        
        cfg.training_events_jsonl_path = str(Path(output_dir) / f"events_{mode}.jsonl")
        cfg.performance_samples_path = str(Path(output_dir) / f"perf_{mode}.csv")
        cfg.performance_summary_path = str(Path(output_dir) / f"summary_{mode}.json")

        max_agents = max(1, int(getattr(cfg, "max_blue_agents", 2)))
        curriculum, initial_phase, initial_opponent_tag = _resolve_initial_opponent_and_phase(cfg, max_agents)
        _clamp_runtime_config_for_team_size(cfg, max_agents)

        env = build_training_env(
            cfg,
            initial_phase=initial_phase,
            initial_opponent_tag=initial_opponent_tag,
        )

        trainer = CustomPPOTrainer(
            env=env,
            cfg=cfg,
            learning_rate=3e-4,
            clip_range=0.2,
            ent_coef=0.01,
            n_epochs=1,
            batch_size=cfg.batch_size,
            value_clip_range=0.2,
            curriculum=curriculum,
        )

        # 1. Collect rollout
        rollout = trainer.collect_rollout()
        
        # Capture buffer contents (clone tensors for comparison)
        buffer_data = {}
        for key, tensor in rollout.fields.items():
            buffer_data[key] = tensor.clone()

        # 2. Run update
        trainer.update(rollout, total_timesteps=1000)
        
        # Capture model weights after update
        weights = [p.clone() for p in trainer.model.parameters()]

        env.close()
        trainer.telemetry.close_e3_step_telemetry()
        
        return buffer_data, weights

    def test_telemetry_invariance_across_modes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            # Run under OFF mode
            off_buffer, off_weights = self._run_mini_training("off", td)
            
            # Run under BASIC mode
            basic_buffer, basic_weights = self._run_mini_training("basic", td)
            
            # Run under FULL mode
            full_buffer, full_weights = self._run_mini_training("full", td)

            # Compare buffers between OFF and BASIC
            for key in off_buffer:
                self.assertTrue(
                    torch.allclose(off_buffer[key], basic_buffer[key]),
                    f"Buffer mismatch for key {key} between off and basic modes"
                )

            # Compare buffers between OFF and FULL
            for key in off_buffer:
                self.assertTrue(
                    torch.allclose(off_buffer[key], full_buffer[key]),
                    f"Buffer mismatch for key {key} between off and full modes"
                )

            # Compare model weights between OFF and BASIC
            for w_off, w_basic in zip(off_weights, basic_weights):
                self.assertTrue(
                    torch.allclose(w_off, w_basic),
                    "Model weights mismatch between off and basic modes after update"
                )

            # Compare model weights between OFF and FULL
            for w_off, w_full in zip(off_weights, full_weights):
                self.assertTrue(
                    torch.allclose(w_off, w_full),
                    "Model weights mismatch between off and full modes after update"
                )
