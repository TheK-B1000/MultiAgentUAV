"""Integration tests for V6I1 staged curriculum wiring."""

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.curriculum_gates import V6I1CurriculumController, is_staged_v6i1_curriculum
from rl.custom_ppo.ppo_updater import set_model_requires_grad_for_phase
from rl.custom_ppo.schedules import (
    resolve_v6i1_cf_coef,
    resolve_v6i1_exploration_epsilon,
    resolve_v6i1_forced_fraction,
    resolve_v6i1_usage_coef,
)
from rl.custom_ppo.trainer import CustomPPOTrainer
from rl.custom_ppo.v6i1_phase_runtime import (
    is_v6i1_staged_trainer,
    resolve_v6i1_cf_coef_current,
    resolve_v6i1_episode_forced_frac,
    resolve_v6i1_rollout_usage_coef,
)
from rl.presets import apply_preset
from rl.training.banner import _episode_credit_training_active, _print_latent_strategy_banner


def _actor_param_requires_grad(model: nn.Module) -> bool:
    for name, param in model.named_parameters():
        if "actor_cnn" in name or "latent_actor" in name:
            return bool(param.requires_grad)
    raise AssertionError("actor parameters not found")


def _router_param_requires_grad(model: nn.Module) -> bool:
    for name, param in model.named_parameters():
        if "strategy_encoder" in name or "selector_gru" in name:
            return bool(param.requires_grad)
    raise AssertionError("router parameters not found")


def _v6i1_smoke_cfg(*, observe_only: bool = True) -> PPOConfig:
    cfg = apply_preset(PPOConfig(), "v6i1_staged")
    cfg.seed = 0
    cfg.total_timesteps = 128
    cfg.curriculum_nominal_timesteps = 128
    cfg.n_envs = 1
    cfg.n_steps = 16
    cfg.batch_size = 16
    cfg.n_epochs = 1
    cfg.use_stable_marl_ppo = False
    cfg.device = "cpu"
    cfg.enable_tensorboard = False
    cfg.enable_checkpoints = False
    cfg.enable_eval = False
    cfg.verbose_training = False
    cfg.enable_progress_bar = False
    cfg.max_blue_agents = 2
    cfg.gpu_native_env = True
    cfg.checkpoint_dir = tempfile.mkdtemp()
    if observe_only:
        cfg.phase_boundary_gate_mode = "observe_only"
        cfg.curriculum_gate_run_boundary_eval = False
        cfg.curriculum_gate_run_probe = False
    return cfg


class V6I1PresetTests(unittest.TestCase):
    def test_production_preset_activates_staged_controller(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i1_staged_team_intent_curriculum")
        self.assertTrue(is_staged_v6i1_curriculum(cfg))
        self.assertTrue(cfg.use_v6i1_curriculum)
        self.assertEqual(cfg.training_mode, "staged_team_intent_curriculum")
        self.assertTrue(cfg.curriculum_gate_run_boundary_eval)
        self.assertTrue(cfg.curriculum_gate_run_probe)

    def test_production_preset_disables_legacy_episode_credit(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i1")
        self.assertFalse(cfg.latent_episode_strategy_ppo)
        self.assertEqual(cfg.latent_episode_strategy_coef, 0.0)
        self.assertEqual(cfg.latent_episode_strategy_warmup_decision_steps, 0)

    def test_repertoire_ablation_does_not_activate_staged_controller(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i1_repertoire_only_ablation")
        self.assertFalse(is_staged_v6i1_curriculum(cfg))
        self.assertEqual(cfg.training_mode, "repertoire_only_ablation")
        self.assertEqual(cfg.latent_forced_z_episode_frac_start, 1.0)


class V6I1EpisodeCreditBannerTests(unittest.TestCase):
    def test_v6i1_preset_does_not_emit_opponent_blind_warning(self) -> None:
        import io
        import contextlib

        cfg = apply_preset(PPOConfig(), "v6i1")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _print_latent_strategy_banner(cfg)
        out = buf.getvalue()
        self.assertNotIn("MI(z; opponent) structurally bounded", out)

    def test_episode_credit_warning_requires_nonzero_coef(self) -> None:
        cfg = PPOConfig()
        cfg.use_latent_strategy = True
        cfg.latent_episode_strategy_ppo = True
        cfg.latent_episode_strategy_coef = 0.0
        self.assertFalse(_episode_credit_training_active(cfg))

        cfg.latent_episode_strategy_coef = 0.30
        self.assertTrue(_episode_credit_training_active(cfg))

    def test_bookkeeping_flag_without_coef_does_not_warn(self) -> None:
        import io
        import contextlib

        cfg = PPOConfig()
        cfg.use_latent_strategy = True
        cfg.latent_k = 4
        cfg.latent_episode_strategy_ppo = True
        cfg.latent_episode_strategy_coef = 0.0
        cfg.latent_episode_strategy_warmup_decision_steps = 0
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _print_latent_strategy_banner(cfg)
        self.assertNotIn("MI(z; opponent) structurally bounded", buf.getvalue())

    def test_active_episode_credit_without_warmup_warns(self) -> None:
        import io
        import contextlib

        cfg = PPOConfig()
        cfg.use_latent_strategy = True
        cfg.latent_k = 4
        cfg.latent_episode_strategy_ppo = True
        cfg.latent_episode_strategy_coef = 0.30
        cfg.latent_episode_strategy_warmup_decision_steps = 0
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _print_latent_strategy_banner(cfg)
        self.assertIn("MI(z; opponent) structurally bounded", buf.getvalue())


class V6I1ScheduleResolverTests(unittest.TestCase):
    def test_phase_schedules_at_nominal_boundaries(self) -> None:
        n = 100_000
        t_a = 40_000
        self.assertEqual(resolve_v6i1_forced_fraction("A", 20_000, t_a, n), 1.0)
        self.assertAlmostEqual(resolve_v6i1_forced_fraction("B", t_a, t_a, n), 0.50)
        self.assertEqual(resolve_v6i1_forced_fraction("C", 80_000, t_a, n), 0.25)
        self.assertEqual(resolve_v6i1_cf_coef("A", 10_000, t_a, n, 0.01), 0.0)
        self.assertAlmostEqual(resolve_v6i1_cf_coef("A", 15_000, t_a, n, 0.01), 0.005)
        self.assertEqual(resolve_v6i1_cf_coef("A", 25_000, t_a, n, 0.01), 0.01)
        self.assertEqual(resolve_v6i1_cf_coef("B", 50_000, t_a, n, 0.01), 0.0)
        self.assertAlmostEqual(resolve_v6i1_cf_coef("C", 80_000, t_a, n, 0.01), 0.0025)
        self.assertEqual(resolve_v6i1_exploration_epsilon("A", 50_000, t_a, n), 0.0)
        self.assertAlmostEqual(resolve_v6i1_exploration_epsilon("B", t_a, t_a, n), 0.20)
        self.assertEqual(resolve_v6i1_usage_coef("A", 50_000, t_a, n), 0.0)
        self.assertAlmostEqual(resolve_v6i1_usage_coef("B", t_a, t_a, n), 0.003)


class V6I1PhaseTransitionTests(unittest.TestCase):
    def _trainer_stub(self, cfg: PPOConfig, *, step: int) -> SimpleNamespace:
        return SimpleNamespace(
            cfg=cfg,
            global_step=step,
            latent_k=4,
            save=lambda _path: None,
            model=SimpleNamespace(training=True, train=lambda *_a, **_k: None),
        )

    def test_observe_only_nominal_a_b_c_transitions(self) -> None:
        cfg = PPOConfig()
        cfg.checkpoint_dir = tempfile.mkdtemp()
        cfg.curriculum_nominal_timesteps = 100_000
        cfg.phase_boundary_gate_mode = "observe_only"
        cfg.training_mode = "staged_team_intent_curriculum"
        cfg.use_v6i1_curriculum = True
        trainer = self._trainer_stub(cfg, step=39_999)
        ctrl = V6I1CurriculumController(trainer)
        self.assertFalse(ctrl.maybe_apply_nominal_phase_transition())
        self.assertEqual(ctrl.phase, "A")

        trainer.global_step = 40_000
        self.assertTrue(ctrl.maybe_apply_nominal_phase_transition())
        self.assertEqual(ctrl.phase, "B")
        self.assertEqual(ctrl.t_A, 40_000)

        trainer.global_step = 70_000
        self.assertTrue(ctrl.maybe_apply_nominal_phase_transition())
        self.assertEqual(ctrl.phase, "C")


class V6I1RequiresGradTests(unittest.TestCase):
    def _model(self) -> nn.Module:
        model = nn.Module()
        model.actor_cnn = nn.Linear(2, 2)
        model.latent_actor = nn.Linear(2, 2)
        model.critic = nn.Linear(2, 1)
        model.strategy_encoder = nn.Linear(2, 2)
        for p in model.parameters():
            p.requires_grad = True
        return model

    def test_actor_freeze_only_in_phase_b(self) -> None:
        model = self._model()
        for phase, actor_trainable, router_trainable in (
            ("A", True, False),
            ("B", False, True),
            ("C", True, True),
        ):
            set_model_requires_grad_for_phase(model, phase)
            self.assertEqual(model.actor_cnn.weight.requires_grad, actor_trainable, msg=phase)
            self.assertEqual(model.strategy_encoder.weight.requires_grad, router_trainable, msg=phase)
            self.assertTrue(model.critic.weight.requires_grad, msg=phase)


class V6I1TrainerSmokeTests(unittest.TestCase):
    def test_trainer_mounts_three_optimizers_and_resolvers(self) -> None:
        cfg = _v6i1_smoke_cfg()
        env = GPUCTFVecEnv(
            GPUFieldConfig(
                n_envs=1,
                n_agents_per_team=2,
                max_decision_steps=64,
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
                value_clip_range=0.2,
            )
            self.assertIsNotNone(trainer.v6i1_curriculum)
            self.assertTrue(trainer.v6i1_three_optimizer_mode)
            self.assertIsNotNone(trainer.router_optimizer)
            self.assertTrue(is_v6i1_staged_trainer(trainer))

            phases: list[str] = []
            for phase, step in (("A", 8), ("B", 48), ("C", 96)):
                trainer.v6i1_curriculum.phase = phase
                trainer.v6i1_curriculum.t_A = 40
                trainer.global_step = step
                rollout = trainer.collect_rollout()
                stats = trainer.update(rollout, total_timesteps=int(cfg.total_timesteps))
                phases.append(phase)
                self.assertAlmostEqual(
                    float(stats.get("latent_forced_z_episode_frac_current", -1.0)),
                    float(resolve_v6i1_episode_forced_frac(trainer)),
                )
                self.assertAlmostEqual(
                    float(stats.get("v6i1_cf_coef_current", -1.0)),
                    float(resolve_v6i1_cf_coef_current(trainer)),
                )
                if phase == "B":
                    self.assertFalse(_actor_param_requires_grad(trainer.model))
                    self.assertTrue(_router_param_requires_grad(trainer.model))
                    self.assertGreater(float(stats.get("v6i1_usage_coef_current", 0.0)), 0.0)
                if phase == "C":
                    self.assertTrue(_actor_param_requires_grad(trainer.model))
                    self.assertAlmostEqual(
                        float(stats.get("v6i1_usage_coef_current", -1.0)),
                        float(resolve_v6i1_rollout_usage_coef(trainer)),
                    )
            self.assertEqual(phases, ["A", "B", "C"])
        finally:
            env.close()

    def test_repertoire_ablation_does_not_mount_controller(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i1_repertoire_only_ablation")
        cfg.seed = 0
        cfg.total_timesteps = 32
        cfg.n_envs = 1
        cfg.n_steps = 8
        cfg.batch_size = 8
        cfg.n_epochs = 1
        cfg.device = "cpu"
        cfg.enable_tensorboard = False
        cfg.enable_checkpoints = False
        cfg.enable_eval = False
        cfg.verbose_training = False
        cfg.enable_progress_bar = False
        cfg.max_blue_agents = 2
        cfg.gpu_native_env = True
        cfg.checkpoint_dir = tempfile.mkdtemp()
        env = GPUCTFVecEnv(
            GPUFieldConfig(
                n_envs=1,
                n_agents_per_team=2,
                max_decision_steps=64,
                device="cpu",
                seed=1,
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
                batch_size=8,
                value_clip_range=0.2,
            )
            self.assertIsNone(trainer.v6i1_curriculum)
            self.assertFalse(trainer.v6i1_three_optimizer_mode)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
