"""End-to-end integration smoke for v6i2 staged curriculum wiring."""

from __future__ import annotations

import io
import contextlib
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.curriculum_gates import V6I1CurriculumController, is_staged_v6i1_curriculum
from rl.custom_ppo.curriculum.schedule import (
    format_staged_curriculum_budget_contract,
    resolve_schedule,
)
from rl.custom_ppo.gate_protocol import (
    V6I2_GATE_PROTOCOL,
    gate_lineage_audit_fields,
    is_staged_v6_team_intent_curriculum,
    is_v6i2_gate_protocol,
    resolve_gate_protocol_version,
)
from rl.custom_ppo.trainer import CustomPPOTrainer
from rl.custom_ppo.trainer_config import resolve_q_phi_input_dim_from_cfg
from rl.custom_ppo.v6i1_phase_runtime import (
    apply_v6i1_learning_rates,
    is_v6i1_staged_trainer,
    load_v6i1_curriculum_state,
    resolve_v6i1_lr_progress_remaining,
)
from rl.latent_marl import CONTEXT_STATE_DIM
from rl.presets import apply_preset
from rl.training.banner import print_training_banner


def _actor_param_requires_grad(model: nn.Module) -> bool:
    for name, param in model.named_parameters():
        if "actor_cnn" in name or "latent_actor" in name:
            return bool(param.requires_grad)
    raise AssertionError("actor parameters not found")


def _v6i2_smoke_cfg(*, observe_only: bool = True) -> PPOConfig:
    cfg = apply_preset(PPOConfig(), "v6i2")
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


class V6I2PresetWiringTests(unittest.TestCase):
    def test_v6i2_counts_as_staged_for_optimizer_and_router_contract(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i2")
        self.assertTrue(is_staged_v6i1_curriculum(cfg))
        self.assertTrue(is_staged_v6_team_intent_curriculum(cfg))
        self.assertTrue(is_v6i2_gate_protocol(cfg))
        hidden = int(cfg.v6i1_recurrent_selector_hidden)
        self.assertEqual(
            resolve_q_phi_input_dim_from_cfg(cfg),
            int(CONTEXT_STATE_DIM) + hidden,
        )

    def test_startup_banner_shows_nominal_and_effective_terminal(self) -> None:
        cfg = _v6i2_smoke_cfg()
        cfg.total_timesteps = 200
        lines = format_staged_curriculum_budget_contract(cfg)
        self.assertTrue(any("Nominal curriculum budget: 128" in line for line in lines))
        self.assertTrue(any("Current effective terminal: 200" in line for line in lines))

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_training_banner(cfg, curriculum=None, max_agents=2, team_size="2v2")
        out = buf.getvalue()
        self.assertIn("Nominal curriculum budget: 128", out)
        self.assertIn("Current effective terminal: 200", out)


class V6I2TerminalExtensionTests(unittest.TestCase):
    def _trainer_stub(self, cfg: PPOConfig, *, step: int) -> SimpleNamespace:
        return SimpleNamespace(
            cfg=cfg,
            global_step=step,
            latent_k=4,
            save=lambda _path: None,
            model=SimpleNamespace(training=True, train=lambda *_a, **_k: None),
            last_stats={},
        )

    def test_late_phase_a_promotion_extends_controller_and_cfg(self) -> None:
        cfg = _v6i2_smoke_cfg()
        schedule = resolve_schedule(cfg)
        late_step = schedule.phase_a_max_end
        expected_terminal = schedule.terminal_step_if_promoted_at(late_step)
        trainer = self._trainer_stub(cfg, step=late_step)
        ctrl = V6I1CurriculumController(trainer)
        self.assertEqual(ctrl.training_terminal_step, 128)
        ctrl._transition_to_phase_b(late_step, nominal=False)
        self.assertEqual(ctrl.training_terminal_step, expected_terminal)
        self.assertEqual(cfg.total_timesteps, expected_terminal)
        self.assertGreater(expected_terminal, 128)

    def test_extension_banner_printed_on_late_promotion(self) -> None:
        cfg = _v6i2_smoke_cfg()
        schedule = resolve_schedule(cfg)
        late_step = schedule.phase_a_max_end
        expected_terminal = schedule.terminal_step_if_promoted_at(late_step)
        trainer = self._trainer_stub(cfg, step=late_step)
        ctrl = V6I1CurriculumController(trainer)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ctrl._transition_to_phase_b(late_step, nominal=False)
        out = buf.getvalue()
        self.assertIn(f"Phase A promoted at: {late_step:,}", out)
        self.assertIn(f"Effective terminal extended to: {expected_terminal:,}", out)
        self.assertIn(f"Phase B budget: {schedule.phase_b_budget_steps:,}", out)
        self.assertIn(f"Phase C budget: {schedule.phase_c_budget_steps:,}", out)


class V6I2TrainerSmokeTests(unittest.TestCase):
    def _make_env(self, cfg: PPOConfig) -> GPUCTFVecEnv:
        return GPUCTFVecEnv(
            GPUFieldConfig(
                n_envs=1,
                n_agents_per_team=2,
                max_decision_steps=64,
                device="cpu",
                seed=0,
                map_layout=cfg.map_layout,
            )
        )

    def test_trainer_mounts_v6i2_controller_and_three_optimizers(self) -> None:
        cfg = _v6i2_smoke_cfg()
        env = self._make_env(cfg)
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
            self.assertEqual(resolve_gate_protocol_version(cfg), V6I2_GATE_PROTOCOL)
        finally:
            env.close()

    def test_learn_honors_extended_terminal_after_late_promotion(self) -> None:
        cfg = _v6i2_smoke_cfg()
        schedule = resolve_schedule(cfg)
        late_step = schedule.phase_a_max_end
        extended_terminal = schedule.terminal_step_if_promoted_at(late_step)
        env = self._make_env(cfg)
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
            ctrl = trainer.v6i1_curriculum
            assert ctrl is not None
            trainer.global_step = max(0, late_step - cfg.n_steps)
            rollout = trainer.collect_rollout()
            trainer.update(rollout, total_timesteps=int(cfg.total_timesteps))
            ctrl._transition_to_phase_b(late_step, nominal=False)
            self.assertEqual(ctrl.effective_training_terminal_step(), extended_terminal)
            trainer.learn(total_timesteps=int(cfg.total_timesteps))
            self.assertGreaterEqual(trainer.global_step, extended_terminal)
            self.assertEqual(cfg.total_timesteps, extended_terminal)
        finally:
            env.close()

    def test_checkpoint_and_resume_preserve_extended_terminal(self) -> None:
        cfg = _v6i2_smoke_cfg()
        schedule = resolve_schedule(cfg)
        late_step = schedule.phase_a_max_end
        extended_terminal = schedule.terminal_step_if_promoted_at(late_step)
        env = self._make_env(cfg)
        ckpt_path = tempfile.mktemp(suffix=".zip")
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
            ctrl = trainer.v6i1_curriculum
            assert ctrl is not None
            ctrl._transition_to_phase_b(late_step, nominal=False)
            trainer.save(ckpt_path)
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            latent_payload = dict(payload.get("latent_state_v6i1", {}) or {})
            self.assertEqual(latent_payload.get("gate_protocol_version"), V6I2_GATE_PROTOCOL)
            for key, value in gate_lineage_audit_fields(cfg).items():
                self.assertEqual(latent_payload.get(key), value)
            curriculum_payload = dict(payload.get("v6i1_curriculum_state", {}) or {})
            self.assertEqual(int(curriculum_payload.get("training_terminal_step", 0)), extended_terminal)

            cfg2 = _v6i2_smoke_cfg()
            cfg2.checkpoint_dir = tempfile.mkdtemp()
            env2 = self._make_env(cfg2)
            try:
                trainer2 = CustomPPOTrainer(
                    env2,
                    cfg2,
                    learning_rate=3e-4,
                    clip_range=0.2,
                    ent_coef=0.01,
                    n_epochs=1,
                    batch_size=16,
                    value_clip_range=0.2,
                )
                load_v6i1_curriculum_state(trainer2.v6i1_curriculum, curriculum_payload)
                self.assertEqual(
                    trainer2.v6i1_curriculum.effective_training_terminal_step(),
                    extended_terminal,
                )
                total_for_learn = max(
                    int(cfg2.total_timesteps),
                    int(trainer2.v6i1_curriculum.effective_training_terminal_step()),
                )
                self.assertEqual(total_for_learn, extended_terminal)
            finally:
                env2.close()
        finally:
            env.close()
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)


class V6I2ScheduleClockTests(unittest.TestCase):
    def _trainer_stub(self, cfg: PPOConfig, *, step: int, phase: str = "A") -> SimpleNamespace:
        curriculum = V6I1CurriculumController(
            SimpleNamespace(
                cfg=cfg,
                global_step=step,
                latent_k=4,
                save=lambda _path: None,
                model=SimpleNamespace(training=True, train=lambda *_a, **_k: None),
                last_stats={},
            )
        )
        curriculum.phase = phase
        if phase != "A":
            curriculum.t_A = int(resolve_schedule(cfg).phase_a_max_end)
        trainer = SimpleNamespace(
            cfg=cfg,
            global_step=step,
            latent_k=4,
            hparams=SimpleNamespace(latent_episode_strategy_lr=None),
            v6i1_curriculum=curriculum,
            v6i1_three_optimizer_mode=True,
            optimizers=SimpleNamespace(
                v6i1_three_optimizer_mode=True,
                actor=SimpleNamespace(param_groups=[{"lr": 0.0, "params": []}]),
                critic=SimpleNamespace(param_groups=[{"lr": 0.0, "params": []}]),
                router=SimpleNamespace(param_groups=[{"lr": 0.0, "params": []}]),
            ),
        )
        return trainer

    def test_phase_a_lr_progress_ignores_extended_terminal(self) -> None:
        cfg = _v6i2_smoke_cfg()
        schedule = resolve_schedule(cfg)
        step = schedule.phase_a_max_end - 1
        trainer = self._trainer_stub(cfg, step=step, phase="A")
        before = resolve_v6i1_lr_progress_remaining(trainer, training_terminal=128)
        trainer.v6i1_curriculum.training_terminal_step = schedule.terminal_step_if_promoted_at(
            schedule.phase_a_max_end
        )
        trainer.cfg.total_timesteps = int(trainer.v6i1_curriculum.training_terminal_step)
        after = resolve_v6i1_lr_progress_remaining(
            trainer, training_terminal=int(trainer.cfg.total_timesteps)
        )
        self.assertAlmostEqual(before, after)
        wrong_global = max(0.0, 1.0 - float(step) / float(trainer.cfg.total_timesteps))
        self.assertGreater(wrong_global, after)

    def test_phase_a_lrs_do_not_rebound_when_terminal_extended(self) -> None:
        cfg = _v6i2_smoke_cfg()
        schedule = resolve_schedule(cfg)
        step = schedule.phase_a_max_end - 1
        base_lr = 3e-4
        trainer = self._trainer_stub(cfg, step=step, phase="A")
        lrs_before = apply_v6i1_learning_rates(trainer, base_lr=base_lr, training_terminal=128)
        trainer.v6i1_curriculum.training_terminal_step = schedule.terminal_step_if_promoted_at(
            schedule.phase_a_max_end
        )
        trainer.cfg.total_timesteps = int(trainer.v6i1_curriculum.training_terminal_step)
        lrs_after = apply_v6i1_learning_rates(
            trainer,
            base_lr=base_lr,
            training_terminal=int(trainer.cfg.total_timesteps),
        )
        for key in ("actor_lr", "critic_lr"):
            self.assertAlmostEqual(float(lrs_before[key]), float(lrs_after[key]))

    def test_phase_b_uses_local_budget_clock(self) -> None:
        cfg = _v6i2_smoke_cfg()
        schedule = resolve_schedule(cfg)
        t_a = schedule.phase_a_max_end
        trainer = self._trainer_stub(cfg, step=t_a, phase="B")
        trainer.v6i1_curriculum.t_A = t_a
        at_start = resolve_v6i1_lr_progress_remaining(trainer, training_terminal=165)
        self.assertAlmostEqual(at_start, 1.0)
        mid = t_a + schedule.phase_b_budget_steps // 2
        trainer.global_step = mid
        at_mid = resolve_v6i1_lr_progress_remaining(trainer, training_terminal=165)
        self.assertAlmostEqual(at_mid, 0.5, places=2)

    def test_promotion_does_not_raise_lrs_via_global_denominator(self) -> None:
        cfg = _v6i2_smoke_cfg()
        schedule = resolve_schedule(cfg)
        late_step = schedule.phase_a_max_end
        extended_terminal = schedule.terminal_step_if_promoted_at(late_step)
        base_lr = 3e-4
        lr_floor = max(0.0, min(float(cfg.lr_floor_frac), 1.0))
        trainer_a = self._trainer_stub(cfg, step=late_step, phase="A")
        lrs_end_a = apply_v6i1_learning_rates(
            trainer_a, base_lr=base_lr, training_terminal=extended_terminal
        )
        trainer_b = self._trainer_stub(cfg, step=late_step, phase="B")
        trainer_b.v6i1_curriculum.t_A = late_step
        lrs_start_b = apply_v6i1_learning_rates(
            trainer_b, base_lr=base_lr, training_terminal=extended_terminal
        )
        wrong_global_progress = max(0.0, 1.0 - float(late_step) / float(extended_terminal))
        wrong_critic_lr = base_lr * max(wrong_global_progress, lr_floor)
        self.assertGreater(wrong_critic_lr, float(lrs_end_a["critic_lr"]))
        self.assertAlmostEqual(
            float(lrs_start_b["critic_lr"]),
            base_lr * max(1.0, lr_floor),
        )
        self.assertGreater(float(lrs_start_b["critic_lr"]), wrong_critic_lr)
        self.assertAlmostEqual(
            float(lrs_start_b["router_lr"]),
            float(cfg.v6i1_router_lr or 5e-3) * max(1.0, lr_floor),
        )

    def test_phase_a_lr_monotonic_non_increasing(self) -> None:
        cfg = _v6i2_smoke_cfg()
        base_lr = 3e-4
        env = V6I2TrainerSmokeTests()._make_env(cfg)
        try:
            trainer = CustomPPOTrainer(
                env,
                cfg,
                learning_rate=base_lr,
                clip_range=0.2,
                ent_coef=0.01,
                n_epochs=1,
                batch_size=16,
                value_clip_range=0.2,
            )
            prev_critic = float("inf")
            prev_actor = float("inf")
            while trainer.global_step < resolve_schedule(cfg).phase_a_max_end:
                rollout = trainer.collect_rollout()
                stats = trainer.update(
                    rollout, total_timesteps=int(cfg.curriculum_nominal_timesteps)
                )
                critic_lr = float(stats.get("critic_lr", 0.0))
                actor_lr = float(stats.get("actor_lr", 0.0))
                self.assertLessEqual(critic_lr, prev_critic + 1e-12)
                self.assertLessEqual(actor_lr, prev_actor + 1e-12)
                prev_critic = critic_lr
                prev_actor = actor_lr
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
