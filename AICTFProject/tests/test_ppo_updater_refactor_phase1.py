"""Phase 0/1 characterization tests for PPO updater refactor."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.gate_protocol import V6I2_GATE_PROTOCOL
from rl.custom_ppo.ppo_updater import PPOUpdater
from rl.custom_ppo.update.actor_intervention import ActorInterventionEvidenceUpdater
from rl.custom_ppo.update.loss_result import measurement_from_pair_tensor
from rl.custom_ppo.update.minibatch_updater import ACTOR_INTERVENTION_REASON_CODES
from rl.custom_ppo.update.optimizer_stepper import ThreeOptimizerStepper, clip_optimizer_grad_norm
from rl.custom_ppo.update.pair_utils import latent_pair_count, validate_v6_protocol_latent_k
from rl.custom_ppo.update.phase_policy import PhaseTrainingPolicy, resolve_training_phase
from rl.custom_ppo.update.separation_objectives import SeparationObjective
from rl.custom_ppo.update.update_context import PPOUpdateContextBuilder


def _minimal_hparams(**overrides: object) -> SimpleNamespace:
    base = dict(
        latent_strategy_ppo_coef=0.1,
        use_latent_strategy=True,
        fixed_latent_strategy=False,
        latent_k=4,
        ent_coef=0.01,
        learning_rate=3e-4,
        n_epochs=1,
        batch_size=64,
        clip_range=0.2,
        vf_coef=0.5,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class PairCountTests(unittest.TestCase):
    def test_latent_pair_count(self) -> None:
        self.assertEqual(latent_pair_count(4), 6)
        self.assertEqual(latent_pair_count(1), 0)

    def test_v6_protocol_rejects_non_four_k(self) -> None:
        cfg = PPOConfig()
        cfg.use_v6i1_curriculum = True
        cfg.training_mode = "staged_team_intent_curriculum"
        cfg.experiment_family = "v6"
        cfg.experiment_id = "v6i1"
        with self.assertRaises(ValueError):
            validate_v6_protocol_latent_k(cfg, 3)


class PhasePolicyTests(unittest.TestCase):
    def test_phase_b_disables_actor_and_counterfactual(self) -> None:
        policy = PhaseTrainingPolicy.from_phase("B")
        self.assertFalse(policy.actor_step_enabled)
        self.assertTrue(policy.router_step_enabled)
        self.assertFalse(policy.counterfactual_active)

    @mock.patch("rl.custom_ppo.v6i1_phase_runtime.v6i1_schedule_context")
    def test_resolve_training_phase_raises_on_mismatch(self, mock_sched) -> None:
        mock_sched.return_value = ("B", 0, 0, 0)
        runtime = SimpleNamespace(
            global_step=100,
            v6i1_curriculum=SimpleNamespace(resolve_phase=lambda _s: "A"),
        )
        with self.assertRaises(RuntimeError):
            resolve_training_phase(runtime, global_step=100)


class OptimizerClipTests(unittest.TestCase):
    def test_clip_only_touches_optimizer_params(self) -> None:
        owned = nn.Linear(2, 2)
        other = nn.Linear(2, 2)
        opt = torch.optim.Adam(owned.parameters(), lr=1e-3)
        x = torch.randn(3, 2)
        loss = owned(x).sum()
        loss.backward()
        other.weight.grad = torch.ones_like(other.weight)
        norm = clip_optimizer_grad_norm(opt, 1.0)
        self.assertGreater(norm, 0.0)
        self.assertIsNotNone(owned.weight.grad)
        self.assertTrue(torch.allclose(other.weight.grad, torch.ones_like(other.weight)))


class ActorInterventionEvidenceTests(unittest.TestCase):
    def test_invalid_measurement_does_not_update_gate(self) -> None:
        latent = SimpleNamespace(
            update_cf_pair_jsd_ema=mock.Mock(return_value=True),
            actor_intervention_consecutive_updates=0,
        )
        cfg = PPOConfig()
        cfg.gate_protocol_version = V6I2_GATE_PROTOCOL
        cfg.experiment_id = "v6i2"
        measurement = measurement_from_pair_tensor(
            None,
            active_fraction=0.0,
            valid_groups=0,
            reason="missing_pair_jsd",
        )
        result = ActorInterventionEvidenceUpdater().update(
            latent,
            measurement,
            cfg=cfg,
            global_step=10,
        )
        self.assertFalse(result.gate_updated)
        latent.update_cf_pair_jsd_ema.assert_not_called()

    def test_valid_measurement_updates_gate(self) -> None:
        latent = SimpleNamespace(
            update_cf_pair_jsd_ema=mock.Mock(return_value=True),
        )
        cfg = PPOConfig()
        cfg.gate_protocol_version = V6I2_GATE_PROTOCOL
        cfg.experiment_id = "v6i2"
        pairs = torch.tensor([0.05, 0.06, 0.04, 0.03, 0.02, 0.01])
        measurement = measurement_from_pair_tensor(
            pairs,
            active_fraction=1.0,
            valid_groups=4,
        )
        result = ActorInterventionEvidenceUpdater().update(
            latent,
            measurement,
            cfg=cfg,
            global_step=42,
        )
        self.assertTrue(result.measurement_valid)
        latent.update_cf_pair_jsd_ema.assert_called_once_with(
            [float(v) for v in pairs.tolist()],
            42,
        )


class UpdaterRngCheckpointTests(unittest.TestCase):
    def test_separation_generator_roundtrip(self) -> None:
        updater = PPOUpdater(
            model=nn.Linear(2, 2),
            optimizer=torch.optim.Adam(nn.Linear(2, 2).parameters()),
            device=torch.device("cpu"),
            cfg=PPOConfig(),
            hparams=_minimal_hparams(),
            latent_state=SimpleNamespace(),
            runtime=SimpleNamespace(global_step=0),
        )
        gen = updater._z_separation_generator
        gen.manual_seed(1)
        _ = torch.rand(5, generator=gen)
        saved = updater.state_dict()
        gen.manual_seed(1)
        _ = torch.rand(5, generator=gen)
        expected_next = torch.rand(5, generator=gen)
        updater.load_state_dict(saved)
        actual_next = torch.rand(5, generator=gen)
        self.assertTrue(torch.allclose(actual_next, expected_next))


class UpdateContextTests(unittest.TestCase):
    @mock.patch("rl.custom_ppo.v6i1_phase_runtime.v6i1_macro_router_active", return_value=False)
    def test_strategy_kl_stop_requires_main_loop_qphi(self, _macro) -> None:
        cfg = PPOConfig()
        cfg.latent_strategy_ppo_coef = 0.0
        hparams = _minimal_hparams(latent_strategy_ppo_coef=0.0, use_latent_strategy=True)
        runtime = SimpleNamespace(
            global_step=0,
            latent_router_optimizer=None,
        )
        buffer = SimpleNamespace(pos=0, fields={})
        ctx = PPOUpdateContextBuilder(cfg=cfg, hparams=hparams, runtime=runtime).build(
            total_timesteps=1_000_000,
            primary_lr=3e-4,
            latent_lam_h=0.001,
            curr_sep_coef=0.0,
            curr_adapter_scale=0.0,
            buffer=buffer,
        )
        self.assertFalse(ctx.strategy_kl_stop_enabled)

    @mock.patch("rl.custom_ppo.v6i1_phase_runtime.v6i1_schedule_context")
    @mock.patch("rl.custom_ppo.v6i1_phase_runtime.v6i1_macro_router_active", return_value=False)
    def test_action_kl_stop_disabled_in_phase_b(self, _macro, mock_sched) -> None:
        mock_sched.return_value = ("B", 0, 0, 0)
        cfg = PPOConfig()
        hparams = _minimal_hparams()
        runtime = SimpleNamespace(
            global_step=0,
            latent_router_optimizer=None,
            v6i1_curriculum=SimpleNamespace(resolve_phase=lambda _s: "B"),
        )
        buffer = SimpleNamespace(pos=0, fields={})
        ctx = PPOUpdateContextBuilder(cfg=cfg, hparams=hparams, runtime=runtime).build(
            total_timesteps=1_000_000,
            primary_lr=3e-4,
            latent_lam_h=0.001,
            curr_sep_coef=0.0,
            curr_adapter_scale=0.0,
            buffer=buffer,
        )
        self.assertFalse(ctx.action_kl_stop_enabled)
        self.assertEqual(ctx.phase, "B")


class PairTelemetryValidityTests(unittest.TestCase):
    def test_missing_measurement_is_not_valid(self) -> None:
        m = measurement_from_pair_tensor(
            None, active_fraction=0.0, valid_groups=0, reason="missing_pair_jsd"
        )
        self.assertFalse(m.valid)
        self.assertIsNone(m.as_list())

    def test_zero_jsd_is_valid_measurement(self) -> None:
        pairs = torch.zeros(6, dtype=torch.float32)
        m = measurement_from_pair_tensor(pairs, active_fraction=1.0, valid_groups=4)
        self.assertTrue(m.valid)
        self.assertEqual(m.as_list(), [0.0] * 6)

    def test_non_finite_pair_jsd_is_invalid(self) -> None:
        pairs = torch.tensor([0.1, float("nan"), 0.2, 0.3, 0.4, 0.5])
        m = measurement_from_pair_tensor(pairs, active_fraction=1.0, valid_groups=4)
        self.assertFalse(m.valid)
        self.assertEqual(m.reason, "invalid_pair_jsd")


class ActorInterventionScenarioTests(unittest.TestCase):
    def _separation(
        self,
        *,
        separation_coef: float,
        counterfactual_active: bool,
    ) -> SimpleNamespace:
        device = torch.device("cpu")
        zero = torch.zeros((), dtype=torch.float32, device=device)
        objective = SeparationObjective(
            model=nn.Linear(2, 2),
            cfg=PPOConfig(),
            hparams=_minimal_hparams(),
            runtime=SimpleNamespace(global_step=0),
            latent_state=SimpleNamespace(
                compute_competence_scores=lambda: (torch.zeros(4), True)
            ),
            subsample_generator=torch.Generator(device=device),
        )
        return objective.compute(
            obs_batch={
                "grid": torch.zeros(2, 2, 7, 20, 20),
                "vec": torch.zeros(2, 10),
                "agent_mask": torch.ones(2, 4),
                "mask": torch.ones(2, 4),
            },
            batch={"global_state": torch.zeros(2, 20)},
            advantages=torch.zeros(2),
            entropy=torch.zeros(2),
            z_idx=torch.zeros(2, dtype=torch.long),
            separation_coef=separation_coef,
            counterfactual_active=counterfactual_active,
            device=device,
            zero_scalar=zero,
        )

    def test_separation_disabled_does_not_produce_valid_measurement(self) -> None:
        result = self._separation(separation_coef=0.0, counterfactual_active=True)
        self.assertFalse(result.pairwise_measurement.valid)
        self.assertEqual(result.pairwise_measurement.reason, "separation_disabled")

    def test_phase_b_counterfactual_inactive_does_not_produce_valid_measurement(self) -> None:
        result = self._separation(separation_coef=0.01, counterfactual_active=False)
        self.assertFalse(result.pairwise_measurement.valid)
        self.assertEqual(result.pairwise_measurement.reason, "phase_counterfactual_inactive")

    def test_valid_finite_pairs_can_update_gate(self) -> None:
        latent = SimpleNamespace(update_cf_pair_jsd_ema=mock.Mock(return_value=True))
        cfg = PPOConfig()
        cfg.gate_protocol_version = V6I2_GATE_PROTOCOL
        cfg.experiment_id = "v6i2"
        pairs = torch.tensor([0.002] * 6)
        measurement = measurement_from_pair_tensor(pairs, active_fraction=1.0, valid_groups=8)
        evidence = ActorInterventionEvidenceUpdater().update(
            latent, measurement, cfg=cfg, global_step=100
        )
        self.assertTrue(evidence.measurement_valid)
        self.assertTrue(evidence.gate_updated)
        latent.update_cf_pair_jsd_ema.assert_called_once()

    def test_reason_codes_cover_separation_paths(self) -> None:
        self.assertIn("separation_disabled", ACTOR_INTERVENTION_REASON_CODES)
        self.assertIn("phase_counterfactual_inactive", ACTOR_INTERVENTION_REASON_CODES)
        self.assertIn("no_active_rows", ACTOR_INTERVENTION_REASON_CODES)


class PhaseOptimizerStepTests(unittest.TestCase):
    @mock.patch("rl.custom_ppo.v6i1_phase_runtime.step_v6i1_optimizers")
    def test_phase_policy_controls_optimizer_steps(self, mock_step) -> None:
        mock_step.return_value = {
            "actor_grad_norm": 0.1,
            "critic_grad_norm": 0.2,
            "router_grad_norm": 0.3,
        }
        model = nn.Linear(2, 1)
        x = torch.randn(3, 2)
        loss = model(x).sum()
        runtime = SimpleNamespace(
            v6i1_three_optimizer_mode=True,
            actor_optimizer=mock.Mock(),
            critic_optimizer=mock.Mock(),
            router_optimizer=mock.Mock(),
        )
        stepper = ThreeOptimizerStepper(runtime)
        ctx = SimpleNamespace(phase="A")
        policy = PhaseTrainingPolicy.from_phase("A")
        stepper.step(
            total_loss=loss,
            ppo_actor_loss=loss,
            value_loss=loss,
            policy_loss=loss,
            entropy_loss=loss,
            latent_loss=torch.tensor(0.0),
            ent_coef=0.01,
            vf_coef=0.5,
            context=ctx,
            phase_policy=policy,
            model=model,
            latent_state=SimpleNamespace(strategy_encoder_grad_norm=lambda: 0.0),
            epoch_idx=0,
            mb_idx=0,
            max_grad_norm=0.5,
        )
        mock_step.assert_called_once()
        kwargs = mock_step.call_args.kwargs
        self.assertTrue(kwargs["actor_step"])
        self.assertTrue(kwargs["critic_step"])
        self.assertFalse(kwargs["router_step"])

        mock_step.reset_mock()
        model.zero_grad(set_to_none=True)
        loss = model(x).sum()
        policy_b = PhaseTrainingPolicy.from_phase("B")
        stepper.step(
            total_loss=loss,
            ppo_actor_loss=loss,
            value_loss=loss,
            policy_loss=loss,
            entropy_loss=loss,
            latent_loss=torch.tensor(0.0),
            ent_coef=0.01,
            vf_coef=0.5,
            context=SimpleNamespace(phase="B"),
            phase_policy=policy_b,
            model=model,
            latent_state=SimpleNamespace(strategy_encoder_grad_norm=lambda: 0.0),
            epoch_idx=0,
            mb_idx=0,
            max_grad_norm=0.5,
        )
        kwargs_b = mock_step.call_args.kwargs
        self.assertFalse(kwargs_b["actor_step"])
        self.assertTrue(kwargs_b["critic_step"])
        self.assertTrue(kwargs_b["router_step"])
