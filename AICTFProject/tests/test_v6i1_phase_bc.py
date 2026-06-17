"""Unit tests for V6I1 Phase B/C training mechanics."""

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.curriculum_gates import V6I1CurriculumController, is_staged_v6i1_curriculum
from rl.custom_ppo.latent_strategy_state import LatentStrategyState
from rl.custom_ppo.schedules import (
    resolve_v6i1_exploration_epsilon,
    resolve_v6i1_forced_fraction,
    resolve_v6i1_usage_coef,
)
from rl.custom_ppo.v6i1_phase_runtime import (
    apply_v6i1_learning_rates,
    build_v6i1_optimizers,
    latent_state_v6i1_checkpoint,
    restore_latent_state_v6i1_checkpoint,
    resolve_v6i1_rollout_usage_coef,
    v6i1_curriculum_state_dict,
    v6i1_macro_router_active,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor_cnn = nn.Linear(4, 4)
        self.latent_actor = nn.Linear(4, 4)
        self.critic = nn.Linear(4, 1)
        self.strategy_encoder = nn.Linear(4, 4)
        self.selector_gru = nn.GRUCell(4, 8)
        self.episode_strategy_value_head = nn.Linear(8, 1)
        self.global_state_dim = 4
        self.recurrent_selector_hidden_dim = 8
        self.latent_k = 4
        self.uses_latent_strategy = True
        self._sampling_gen_strategy = None

    def strategy_logits(self, global_state: torch.Tensor, *, selector_hidden=None) -> torch.Tensor:
        return self.strategy_encoder(global_state)

    def episode_strategy_value(self, global_state: torch.Tensor, z_idx: torch.Tensor) -> torch.Tensor:
        z_one_hot = torch.nn.functional.one_hot(z_idx.long(), num_classes=4).float()
        return self.episode_strategy_value_head(torch.cat([global_state, z_one_hot], dim=-1)).squeeze(-1)


def _v6i1_cfg() -> PPOConfig:
    cfg = PPOConfig()
    cfg.use_v6i1_curriculum = True
    cfg.training_mode = "staged_team_intent_curriculum"
    cfg.experiment_family = "v6"
    cfg.experiment_id = "v6i1"
    cfg.curriculum_nominal_timesteps = 1_000_000
    cfg.phase_boundary_gate_mode = "observe_only"
    cfg.latent_entropy_mode = "marginal"
    return cfg


class V6I1ScheduleWiringTests(unittest.TestCase):
    def test_usage_coef_positive_in_phase_b(self) -> None:
        coef = resolve_v6i1_usage_coef("B", 500_000, 400_000, 1_000_000)
        self.assertGreater(coef, 0.0)
        self.assertLessEqual(coef, 0.003)

    def test_forced_fraction_phase_c_is_quarter(self) -> None:
        self.assertEqual(resolve_v6i1_forced_fraction("C", 900_000, 400_000, 1_000_000), 0.25)

    def test_exploration_epsilon_phase_c_floor(self) -> None:
        self.assertEqual(resolve_v6i1_exploration_epsilon("C", 900_000, 400_000, 1_000_000), 0.05)


class MacroRouterLifecycleTests(unittest.TestCase):
    def _make_state(self, *, phase: str = "B") -> LatentStrategyState:
        cfg = _v6i1_cfg()
        model = _TinyModel()
        curriculum = SimpleNamespace(
            phase=phase,
            t_A=400_000,
            nominal_steps=1_000_000,
            resolve_phase=lambda _step=None: phase,
        )
        trainer = SimpleNamespace(
            cfg=cfg,
            device=torch.device("cpu"),
            env=SimpleNamespace(num_envs=2),
            model=model,
            use_latent_strategy=True,
            fixed_latent_strategy=False,
            latent_k=4,
            global_step=500_000,
            v6i1_curriculum=curriculum,
            latent_episode_strategy_ppo=False,
            latent_arc_credit_enabled=False,
            router_optimizer=torch.optim.AdamW(model.strategy_encoder.parameters(), lr=1e-3),
        )
        return LatentStrategyState(trainer)

    def test_macro_finalize_open_records_segment(self) -> None:
        state = self._make_state()
        mask = torch.tensor([True, False], dtype=torch.bool)
        gs = torch.zeros((2, 4), dtype=torch.float32)
        z = torch.tensor([1, 0], dtype=torch.long)
        logp = torch.tensor([-0.5, -0.5], dtype=torch.float32)
        state.macro_open(mask, global_state=gs, z_idx=z, z_log_prob=logp)
        state.macro_accumulate_step(torch.tensor([1.0, 0.0], dtype=torch.float32))
        state.macro_finalize(mask, reason="boundary")
        self.assertEqual(len(state.rollout_strategy_macro_records), 1)
        rec = state.rollout_strategy_macro_records[0]
        self.assertAlmostEqual(rec["macro_return"], 1.0)
        self.assertEqual(rec["z"], 1)

    def test_apply_macro_strategy_ppo_runs_in_phase_b(self) -> None:
        state = self._make_state(phase="B")
        state.rollout_strategy_macro_records.append(
            {
                "global_state_0": torch.zeros(4),
                "z": 2,
                "z_logprob_old": -1.0,
                "macro_return": 3.0,
                "macro_length": 64,
                "reason": "boundary",
            }
        )
        stats = state.apply_macro_strategy_ppo()
        self.assertEqual(stats["latent_macro_count"], 1.0)
        self.assertGreater(stats["latent_macro_grad_norm"], 0.0)


class ThreeOptimizerAndCheckpointTests(unittest.TestCase):
    def test_build_v6i1_optimizers_splits_param_groups(self) -> None:
        cfg = _v6i1_cfg()
        model = _TinyModel()
        trainer = SimpleNamespace(
            cfg=cfg,
            model=model,
            base_learning_rate=3e-4,
            latent_episode_strategy_lr=5e-3,
            v6i1_three_optimizer_mode=False,
            latent_router_optimizer=None,
        )
        build_v6i1_optimizers(trainer, base_lr=3e-4)
        self.assertTrue(trainer.v6i1_three_optimizer_mode)
        self.assertIsNotNone(trainer.actor_optimizer)
        self.assertIsNotNone(trainer.critic_optimizer)
        self.assertIsNotNone(trainer.router_optimizer)

    def test_phase_c_actor_lr_scaled(self) -> None:
        cfg = _v6i1_cfg()
        model = _TinyModel()
        curriculum = SimpleNamespace(
            phase="C",
            t_A=400_000,
            nominal_steps=1_000_000,
            resolve_phase=lambda _step=None: "C",
        )
        trainer = SimpleNamespace(
            cfg=cfg,
            model=model,
            global_step=800_000,
            v6i1_curriculum=curriculum,
            v6i1_three_optimizer_mode=True,
        )
        build_v6i1_optimizers(trainer, base_lr=1e-3)
        stats = apply_v6i1_learning_rates(trainer, base_lr=1e-3, progress_remaining=1.0)
        self.assertAlmostEqual(stats["actor_lr"], 0.05 * 1e-3)
        self.assertAlmostEqual(stats["critic_lr"], 1e-3)

    def test_checkpoint_round_trip_curriculum_and_latent_state(self) -> None:
        cfg = _v6i1_cfg()
        latent_state = self._make_state_for_checkpoint()
        curriculum = V6I1CurriculumController(
            SimpleNamespace(
                cfg=cfg,
                global_step=450_000,
                latent_k=4,
                latent_state=latent_state,
                latent_episode_strategy_ppo=False,
                last_stats={},
                save=lambda _path: None,
            )
        )
        curriculum.phase = "B"
        curriculum.t_A = 420_000
        latent_state.cf_J[0] = 12.5
        latent_state.pair_jsd_ema[0] = 0.02
        payload = {
            "curriculum": v6i1_curriculum_state_dict(curriculum),
            "latent": latent_state_v6i1_checkpoint(latent_state),
        }
        curriculum.phase = "A"
        curriculum.t_A = -1
        latent_state.cf_J[:] = 0.0
        from rl.custom_ppo.v6i1_phase_runtime import load_v6i1_curriculum_state

        load_v6i1_curriculum_state(curriculum, payload["curriculum"])
        restore_latent_state_v6i1_checkpoint(latent_state, payload["latent"])
        self.assertEqual(curriculum.phase, "B")
        self.assertEqual(curriculum.t_A, 420_000)
        self.assertAlmostEqual(float(latent_state.cf_J[0]), 12.5)

    def _make_state_for_checkpoint(self) -> LatentStrategyState:
        cfg = _v6i1_cfg()
        model = _TinyModel()
        trainer = SimpleNamespace(
            cfg=cfg,
            device=torch.device("cpu"),
            env=SimpleNamespace(num_envs=1),
            model=model,
            use_latent_strategy=True,
            fixed_latent_strategy=False,
            latent_k=4,
            global_step=450_000,
            v6i1_curriculum=SimpleNamespace(phase="B", t_A=400_000, nominal_steps=1_000_000, resolve_phase=lambda _s=None: "B"),
            latent_episode_strategy_ppo=False,
            latent_arc_credit_enabled=False,
        )
        return LatentStrategyState(trainer)


class PhaseTransitionTests(unittest.TestCase):
    def test_nominal_phase_b_to_c_after_duration(self) -> None:
        cfg = _v6i1_cfg()
        cfg.phase_boundary_gate_mode = "observe_only"
        trainer = SimpleNamespace(
            cfg=cfg,
            global_step=700_000,
            latent_k=4,
            latent_state=SimpleNamespace(
                cf_episode_counts=[0, 0, 0, 0],
                recent_z_history=[],
                pair_jsd_ema=[0.0] * 6,
                jsd_gate_consecutive_updates=0,
                cf_J=[0.0] * 4,
                cf_return_var=1.0,
                router_optimizer_step_count=0,
                compute_competence_scores=lambda: ([0.0, 0.0, 0.0, 0.0], False),
            ),
            latent_episode_strategy_ppo=False,
            last_stats={},
            save=lambda _path: None,
        )
        ctrl = V6I1CurriculumController(trainer)
        ctrl.phase = "B"
        ctrl.t_A = 400_000
        transitioned = ctrl.maybe_apply_phase_transitions()
        self.assertTrue(transitioned)
        self.assertEqual(ctrl.phase, "C")


class MacroActiveGateTests(unittest.TestCase):
    def test_macro_inactive_in_phase_a(self) -> None:
        cfg = _v6i1_cfg()
        trainer = SimpleNamespace(
            cfg=cfg,
            global_step=100_000,
            v6i1_curriculum=SimpleNamespace(
                phase="A",
                t_A=-1,
                nominal_steps=1_000_000,
                resolve_phase=lambda _s=None: "A",
            ),
        )
        self.assertFalse(v6i1_macro_router_active(trainer))

    def test_rollout_usage_zero_in_phase_a(self) -> None:
        cfg = _v6i1_cfg()
        trainer = SimpleNamespace(
            cfg=cfg,
            global_step=100_000,
            v6i1_curriculum=SimpleNamespace(
                phase="A",
                t_A=-1,
                nominal_steps=1_000_000,
                resolve_phase=lambda _s=None: "A",
            ),
        )
        self.assertEqual(resolve_v6i1_rollout_usage_coef(trainer), 0.0)


if __name__ == "__main__":
    unittest.main()
