"""Pre-launch validation: ownership, checkpoint resume, compressed staged diagnostic."""

from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.curriculum.types import GateFamilyResult
from rl.custom_ppo.curriculum_gates import V6I1CurriculumController
from rl.custom_ppo.gate_protocol import GATE_STATUS_FAIL, V6I2_GATE_PROTOCOL
from rl.custom_ppo.ppo_updater import PPOUpdater, set_model_requires_grad_for_phase
from rl.custom_ppo.trainer import CustomPPOTrainer
from rl.custom_ppo.update.actor_intervention import ActorInterventionEvidenceUpdater
from rl.custom_ppo.update.loss_result import measurement_from_pair_tensor
from rl.custom_ppo.update.param_registry import (
    OptimizerRegistry,
    ParameterRegistry,
    validate_model_parameter_ownership,
)
from rl.custom_ppo.update.phase_policy import PhaseTrainingPolicy
from rl.custom_ppo.v6i1_phase_runtime import (
    latent_state_v6i1_checkpoint,
    load_v6i1_curriculum_state,
    restore_latent_state_v6i1_checkpoint,
    v6i1_curriculum_state_dict,
)
from rl.presets import apply_preset


def _compressed_v6i2_cfg() -> PPOConfig:
    cfg = apply_preset(PPOConfig(), "v6i2")
    cfg.seed = 0
    cfg.total_timesteps = 256
    cfg.curriculum_nominal_timesteps = 256
    cfg.phase_a_earliest_end_fraction = 0.25
    cfg.phase_a_max_end_fraction = 0.50
    cfg.phase_c_start_fraction = 0.75
    cfg.phase_a_gate_check_interval = 32
    cfg.n_envs = 1
    cfg.n_steps = 16
    cfg.batch_size = 16
    cfg.n_epochs = 1
    cfg.device = "cpu"
    cfg.enable_tensorboard = False
    cfg.enable_checkpoints = True
    cfg.enable_eval = False
    cfg.verbose_training = False
    cfg.enable_progress_bar = False
    cfg.max_blue_agents = 2
    cfg.gpu_native_env = True
    cfg.checkpoint_dir = tempfile.mkdtemp()
    cfg.phase_boundary_gate_mode = "observe_only"
    cfg.curriculum_gate_run_boundary_eval = False
    cfg.curriculum_gate_run_probe = False
    return cfg


def _param_delta(model: nn.Module, before: dict[str, torch.Tensor]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for name, param in model.named_parameters():
        if name not in before:
            continue
        deltas[name] = float((param.detach() - before[name]).abs().max().item())
    return deltas


def _group_delta(deltas: dict[str, float], *, group: str) -> float:
    keys = {
        "actor": ("actor_cnn", "latent_actor"),
        "critic": ("critic",),
        "router": (
            "strategy_encoder",
            "selector_gru",
            "episode_strategy_value_head",
            "strategy_aux_return_head",
            "phase_predictor",
        ),
    }[group]
    vals = [
        v
        for name, v in deltas.items()
        if any(k in name for k in keys) and "episode_strategy_value_head" not in name
        or (group == "router" and "episode_strategy_value_head" in name)
        or (group == "critic" and "critic" in name and "episode_strategy_value_head" not in name)
    ]
    return max(vals) if vals else 0.0


class ParameterOwnershipTests(unittest.TestCase):
    def test_v6i2_model_has_complete_non_overlapping_ownership(self) -> None:
        cfg = _compressed_v6i2_cfg()
        env = GPUCTFVecEnv(
            GPUFieldConfig(
                n_envs=1,
                n_agents_per_team=2,
                max_decision_steps=32,
                device="cpu",
                seed=0,
                map_layout=cfg.map_layout,
            )
        )
        try:
            trainer = CustomPPOTrainer(
                env,
                cfg,
                learning_rate=3e-4,
                clip_range=0.2,
                ent_coef=0.01,
                n_epochs=1,
                batch_size=16,
            )
            for phase in ("A", "B", "C"):
                set_model_requires_grad_for_phase(trainer.model, phase)
                registry = validate_model_parameter_ownership(trainer.model)
                policy = PhaseTrainingPolicy.from_phase(phase)
                for group, trainable in (
                    ("actor", policy.actor_trainable),
                    ("critic", policy.critic_trainable),
                    ("router", policy.router_trainable),
                ):
                    params = getattr(registry, group)
                    if trainable:
                        self.assertGreater(len(params), 0, msg=f"phase={phase} {group}")
                    for param in params:
                        self.assertEqual(bool(param.requires_grad), trainable, msg=f"{phase}/{group}")
                OptimizerRegistry.from_runtime(trainer, trainer.optimizer).validate_against(registry)
        finally:
            env.close()


class CheckpointResumeParityTests(unittest.TestCase):
    def test_curriculum_latent_and_updater_state_roundtrip(self) -> None:
        cfg = _compressed_v6i2_cfg()
        env = GPUCTFVecEnv(
            GPUFieldConfig(
                n_envs=1,
                n_agents_per_team=2,
                max_decision_steps=32,
                device="cpu",
                seed=0,
                map_layout=cfg.map_layout,
            )
        )
        try:
            trainer = CustomPPOTrainer(
                env,
                cfg,
                learning_rate=3e-4,
                clip_range=0.2,
                ent_coef=0.01,
                n_epochs=1,
                batch_size=16,
            )
            assert trainer.v6i1_curriculum is not None
            curriculum = trainer.v6i1_curriculum
            curriculum.phase = "A"
            curriculum.next_gate_step = 64
            curriculum.last_gate_step_run = 32
            curriculum.gate_check_history = [{"global_step": 32, "gate_passed": False}]
            trainer.latent_state.update_cf_pair_jsd_ema([0.002] * 6, 32)
            trainer.latent_state.actor_intervention_consecutive_updates = 2

            gen = trainer.updater._z_separation_generator
            gen.manual_seed(99)
            pre = torch.rand(4, generator=gen)
            saved = trainer.updater.state_dict()

            ckpt_path = tempfile.mktemp(suffix=".zip")
            trainer.save(ckpt_path)

            trainer2 = CustomPPOTrainer(
                env,
                cfg,
                learning_rate=3e-4,
                clip_range=0.2,
                ent_coef=0.01,
                n_epochs=1,
                batch_size=16,
            )
            trainer2.load(ckpt_path)

            self.assertEqual(trainer2.v6i1_curriculum.phase, "A")
            self.assertEqual(trainer2.v6i1_curriculum.next_gate_step, 64)
            self.assertEqual(trainer2.v6i1_curriculum.last_gate_step_run, 32)
            self.assertEqual(len(trainer2.v6i1_curriculum.gate_check_history), 1)
            self.assertEqual(int(trainer2.latent_state.actor_intervention_consecutive_updates), 2)

            gen2 = trainer2.updater._z_separation_generator
            gen2.manual_seed(99)
            _ = torch.rand(4, generator=gen2)
            np.testing.assert_allclose(
                torch.rand(4, generator=gen2).numpy(),
                torch.rand(4, generator=gen).numpy(),
            )
            trainer2.updater.load_state_dict(saved)
        finally:
            env.close()


class ActorInterventionEvidenceSmokeTests(unittest.TestCase):
    def test_invalid_cf_does_not_update_gate(self) -> None:
        latent = SimpleNamespace(
            update_cf_pair_jsd_ema=mock.Mock(return_value=True),
            actor_intervention_consecutive_updates=0,
            cf_pair_jsd_valid_updates=0,
        )
        cfg = PPOConfig()
        cfg.gate_protocol_version = V6I2_GATE_PROTOCOL
        cfg.experiment_id = "v6i2"
        measurement = measurement_from_pair_tensor(
            None, active_fraction=0.0, valid_groups=0, reason="separation_disabled"
        )
        result = ActorInterventionEvidenceUpdater().update(latent, measurement, cfg=cfg, global_step=1)
        self.assertFalse(result.gate_updated)
        latent.update_cf_pair_jsd_ema.assert_not_called()

    def test_valid_finite_pairs_update_once(self) -> None:
        latent = SimpleNamespace(
            update_cf_pair_jsd_ema=mock.Mock(return_value=True),
            actor_intervention_consecutive_updates=0,
            cf_pair_jsd_valid_updates=0,
        )
        cfg = PPOConfig()
        cfg.gate_protocol_version = V6I2_GATE_PROTOCOL
        cfg.experiment_id = "v6i2"
        pairs = torch.tensor([0.002, 0.003, 0.001, 0.002, 0.004, 0.002])
        measurement = measurement_from_pair_tensor(pairs, active_fraction=1.0, valid_groups=3)
        result = ActorInterventionEvidenceUpdater().update(latent, measurement, cfg=cfg, global_step=7)
        self.assertTrue(result.measurement_valid)
        latent.update_cf_pair_jsd_ema.assert_called_once()

    def test_zero_jsd_is_valid_not_missing(self) -> None:
        pairs = torch.zeros(6)
        measurement = measurement_from_pair_tensor(pairs, active_fraction=1.0, valid_groups=2)
        self.assertTrue(measurement.valid)
        self.assertEqual(measurement.as_list(), [0.0] * 6)


class CompressedStagedDiagnosticTests(unittest.TestCase):
    def test_observe_only_advances_a_to_b_to_c_with_phase_deltas(self) -> None:
        cfg = _compressed_v6i2_cfg()
        env = GPUCTFVecEnv(
            GPUFieldConfig(
                n_envs=1,
                n_agents_per_team=2,
                max_decision_steps=32,
                device="cpu",
                seed=0,
                map_layout=cfg.map_layout,
            )
        )
        try:
            trainer = CustomPPOTrainer(
                env,
                cfg,
                learning_rate=3e-4,
                clip_range=0.2,
                ent_coef=0.01,
                n_epochs=1,
                batch_size=16,
            )
            curriculum = trainer.v6i1_curriculum
            assert curriculum is not None

            def _run_phase(phase: str, steps: int) -> None:
                curriculum.phase = phase
                for _ in range(steps):
                    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
                    rollout = trainer.collect_rollout()
                    trainer.update(rollout, total_timesteps=int(cfg.total_timesteps))
                    deltas = _param_delta(trainer.model, before)
                    actor_moved = _group_delta(deltas, group="actor") > 0.0
                    critic_moved = _group_delta(deltas, group="critic") > 0.0
                    router_moved = _group_delta(deltas, group="router") > 0.0
                    if phase == "A":
                        self.assertTrue(actor_moved or critic_moved)
                    elif phase == "B":
                        self.assertTrue(critic_moved or router_moved)
                        self.assertFalse(actor_moved)
                    elif phase == "C":
                        self.assertTrue(actor_moved or critic_moved or router_moved)

            _run_phase("A", 2)
            trainer.global_step = int(curriculum.phase_a_min_end)
            curriculum.maybe_apply_nominal_phase_transition()
            self.assertEqual(curriculum.phase, "B")
            _run_phase("B", 2)
            trainer.global_step = int(curriculum.phase_c_nominal_start)
            curriculum.maybe_apply_nominal_phase_transition()
            self.assertEqual(curriculum.phase, "C")
            _run_phase("C", 2)
        finally:
            env.close()

    def test_failed_gate_before_max_continues_phase_a(self) -> None:
        cfg = _compressed_v6i2_cfg()
        cfg.phase_boundary_gate_mode = "enforce"
        cfg.curriculum_gate_run_boundary_eval = True
        cfg.curriculum_gate_run_probe = True
        trainer = SimpleNamespace(
            cfg=cfg,
            global_step=0,
            latent_k=4,
            latent_state=SimpleNamespace(
                cf_episode_counts=[60] * 4,
                recent_z_history=[0, 1, 2, 3] * 25,
                pair_jsd_ema=np.array([0.02] * 6, dtype=np.float32),
                jsd_gate_consecutive_updates=5,
                pairwise_ema_valid_updates=5,
                cf_J=np.array([10.0] * 4, dtype=np.float32),
                cf_return_var=1.0,
                router_optimizer_step_count=0,
                compute_competence_scores=lambda: (np.array([0.55] * 4), True),
                cf_pair_jsd_ema=np.array([0.002] * 6, dtype=np.float32),
                cf_pair_jsd_valid_updates=1,
                actor_intervention_consecutive_updates=0,
                update_cf_pair_jsd_ema=lambda *a, **k: True,
            ),
            latent_episode_strategy_ppo=False,
            last_stats={"latent_forced_z_step_fraction": 1.0},
            save=mock.Mock(),
            model=mock.Mock(training=True, train=mock.Mock()),
            device=torch.device("cpu"),
        )
        ctrl = V6I1CurriculumController(trainer)
        trainer.global_step = int(ctrl.phase_a_min_end)
        fail_result = GateFamilyResult(status=GATE_STATUS_FAIL)
        with mock.patch.object(ctrl, "_evaluate_online_gates", return_value={}), mock.patch.object(
            ctrl,
            "_evaluate_behavioral_realization_gate",
            return_value=fail_result,
        ), mock.patch.object(
            ctrl, "_run_learnability_probe", return_value=fail_result
        ), mock.patch(
            "rl.custom_ppo.curriculum.controller.GateIsolationBoundary"
        ) as mock_boundary:
            boundary = mock.MagicMock()
            boundary.__enter__.return_value = boundary
            boundary.eval_model = trainer.model
            boundary.policy.return_value = mock.Mock()
            boundary.assert_unchanged = mock.Mock()
            mock_boundary.return_value = boundary
            promoted = ctrl.check_and_run_gate()
        self.assertFalse(promoted)
        self.assertEqual(ctrl.phase, "A")


if __name__ == "__main__":
    unittest.main()
