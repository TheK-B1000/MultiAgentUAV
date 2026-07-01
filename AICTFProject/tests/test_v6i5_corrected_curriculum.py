"""Pins v6i5 corrected team-intent curriculum wiring."""

from __future__ import annotations

import unittest
from dataclasses import asdict
from types import SimpleNamespace

import torch
import torch.nn as nn

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.csv_writers import _update_fieldnames
from rl.custom_ppo.curriculum_gates import is_staged_v6i1_curriculum
from rl.custom_ppo.gate_protocol import (
    V6I2_GATE_PROTOCOL,
    is_staged_v6_team_intent_curriculum,
    resolve_gate_protocol_version,
)
from rl.custom_ppo.latent.router_sampling import (
    build_current_plus_delta_router_context,
    current_opportunity_features,
)
from rl.custom_ppo.latent_diagnostics import _latent_rollout_stats
from rl.custom_ppo.update.telemetry import build_metric_schema
from rl.custom_ppo.update.update_order import update_order_jsd_metrics
from rl.custom_ppo.trainer_config import resolve_q_phi_input_dim_from_cfg
from rl.custom_ppo.trainer_optimizers import TrainerOptimizerBundle
from rl.training.cli import cfg_from_args, parse_train_args
from rl.training.config_validation import normalize_and_validate_training_config
from rl.global_state import GLOBAL_STATE_DIM
from rl.latent_losses import compute_v6i5_router_loss
from rl.presets import apply_preset


class _RouterOnlyModel(nn.Module):
    def __init__(self, latent_k: int = 4) -> None:
        super().__init__()
        self.strategy_encoder = nn.Linear(68, latent_k)
        self.strategy_tau = 1.0

    def strategy_logits(self, context: torch.Tensor) -> torch.Tensor:
        return self.strategy_encoder(context.float()) / self.strategy_tau


class _TinyV6Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor_cnn = nn.Linear(2, 2)
        self.latent_actor = nn.Linear(2, 2)
        self.critic = nn.Linear(2, 1)
        self.strategy_encoder = nn.Linear(2, 4)


class V6I5PresetTests(unittest.TestCase):
    def test_v6i5_exact_locked_diff_vs_v6i2(self) -> None:
        v6i2 = asdict(apply_preset(PPOConfig(), "v6i2"))
        v6i5 = asdict(apply_preset(PPOConfig(), "v6i5"))
        diff = {key for key in v6i5 if v6i5[key] != v6i2[key]}
        self.assertEqual(
            diff,
            {
                "experiment_id",
                "latent_entropy_mode",
                "latent_resample_every_n",
                "v6i1_router_lr",
                "latent_strategy_ppo_coef",
                "strategy_target_kl",
                "router_context_mode",
                "router_context_dimension",
                "router_persistence_mode",
                "router_marginal_entropy_coefficient",
                "run_tag",
            },
        )
        self.assertEqual(v6i5["experiment_id"], "v6i5")
        self.assertEqual(v6i5["latent_entropy_mode"], "marginal")
        self.assertEqual(v6i5["latent_resample_every_n"], 32)
        self.assertEqual(v6i5["v6i1_router_lr"], 0.001)
        self.assertEqual(v6i5["latent_strategy_ppo_coef"], 0.20)
        self.assertEqual(v6i5["strategy_target_kl"], 0.015)
        self.assertEqual(v6i5["router_context_mode"], "current_plus_delta")
        self.assertEqual(v6i5["router_context_dimension"], 68)
        self.assertEqual(v6i5["router_persistence_mode"], "expected_switch_detached_previous")
        self.assertEqual(v6i5["router_marginal_entropy_coefficient"], 0.001)
        self.assertEqual(v6i5["router_conditional_entropy_coefficient"], 0.0)

    def test_v6i5_single_alias_and_gate_lineage(self) -> None:
        resolved = asdict(apply_preset(PPOConfig(), "v6i5"))
        self.assertEqual(resolved["experiment_id"], "v6i5")
        for alias in (
            "v6i5_full",
            "v6i5_a1_entropy",
            "v6i5_a2_persistence",
            "v6i5_a3_router_ppo",
            "v6i5_a4_interval32",
        ):
            with self.assertRaises(ValueError):
                apply_preset(PPOConfig(), alias)
        cfg = apply_preset(PPOConfig(), "v6i5")
        self.assertTrue(is_staged_v6i1_curriculum(cfg))
        self.assertTrue(is_staged_v6_team_intent_curriculum(cfg))
        self.assertEqual(resolve_gate_protocol_version(cfg), V6I2_GATE_PROTOCOL)

    def test_v6i5_q_phi_dim_is_corrected_router_context(self) -> None:
        self.assertEqual(resolve_q_phi_input_dim_from_cfg(apply_preset(PPOConfig(), "v6i5")), 68)
        self.assertNotEqual(
            resolve_q_phi_input_dim_from_cfg(apply_preset(PPOConfig(), "v6i2")),
            68,
        )


class V6I5RouterContextTests(unittest.TestCase):
    def test_router_context_selects_current_frame_and_delta(self) -> None:
        batch = 3
        frames = [
            torch.full((batch, GLOBAL_STATE_DIM), float(i + 1))
            for i in range(5)
        ]
        ctx170 = torch.cat(frames, dim=-1)
        previous = torch.zeros((batch, GLOBAL_STATE_DIM))
        router_context = build_current_plus_delta_router_context(ctx170, previous)
        self.assertTrue(torch.equal(current_opportunity_features(ctx170), frames[0]))
        self.assertTrue(torch.equal(router_context[:, :GLOBAL_STATE_DIM], frames[0]))
        self.assertTrue(torch.equal(router_context[:, GLOBAL_STATE_DIM:], frames[0]))

    def test_first_delta_zero_then_delta_from_previous_opportunity(self) -> None:
        first = torch.arange(GLOBAL_STATE_DIM, dtype=torch.float32).reshape(1, -1)
        second = first + 10.0
        first_ctx = torch.cat([first, first, first, torch.zeros_like(first), torch.zeros_like(first)], dim=-1)
        second_ctx = torch.cat([second, second, second, torch.zeros_like(second), torch.zeros_like(second)], dim=-1)
        first_router = build_current_plus_delta_router_context(first_ctx, first)
        second_router = build_current_plus_delta_router_context(second_ctx, first)
        self.assertTrue(torch.equal(first_router[:, GLOBAL_STATE_DIM:], torch.zeros_like(first)))
        self.assertTrue(torch.equal(second_router[:, GLOBAL_STATE_DIM:], torch.full_like(first, 10.0)))


class V6I5RouterLossTests(unittest.TestCase):
    def test_marginal_entropy_step_uses_full_router_batch_only(self) -> None:
        torch.manual_seed(0)
        model = _RouterOnlyModel()
        context = torch.randn(8, 68)
        loss, stats = compute_v6i5_router_loss(
            model,
            router_contexts=context,
            previous_router_contexts=torch.zeros_like(context),
            executed_z=torch.zeros(8, dtype=torch.long),
            old_log_probs=torch.zeros(8),
            advantages=torch.ones(8),
            opportunity_mask=torch.ones(8, dtype=torch.bool),
            persistence_mask=torch.zeros(8, dtype=torch.bool),
            clip_range=0.2,
            latent_k=4,
            ppo_coef=0.0,
            persistence_coef=0.0,
            entropy_coef=0.5,
            entropy_objective="maximize",
            include_rollout_marginal_entropy=True,
            device=torch.device("cpu"),
        )
        self.assertTrue(loss.requires_grad)
        self.assertEqual(stats["application_count"], 1.0)
        self.assertEqual(stats["row_count"], 8.0)
        self.assertEqual(stats["conditional_entropy_nats"], 0.0)
        self.assertEqual(stats["effective_coefficient"], 0.5)
        loss.backward()
        grad = model.strategy_encoder.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum().item()), 0.0)

    def test_minibatch_router_loss_uses_expected_switch_persistence(self) -> None:
        torch.manual_seed(1)
        model = _RouterOnlyModel()
        context = torch.randn(6, 68)
        prev = torch.randn(6, 68)
        mask = torch.tensor([True, True, False, True, False, True])
        loss, stats = compute_v6i5_router_loss(
            model,
            router_contexts=context,
            previous_router_contexts=prev,
            executed_z=torch.tensor([0, 1, 2, 3, 0, 1]),
            old_log_probs=torch.zeros(6),
            advantages=torch.arange(6, dtype=torch.float32),
            opportunity_mask=mask,
            persistence_mask=mask,
            clip_range=0.2,
            latent_k=4,
            ppo_coef=0.2,
            persistence_coef=0.02,
            entropy_coef=0.0,
            entropy_objective="maximize",
            include_rollout_marginal_entropy=False,
            device=torch.device("cpu"),
        )
        self.assertTrue(loss.requires_grad)
        self.assertGreater(float(stats["persistence_loss"].detach().item()), 0.0)
        self.assertEqual(stats["application_count"], 0.0)
        loss.backward()
        grad = model.strategy_encoder.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum().item()), 0.0)

    def test_previous_context_has_no_gradient_and_invalid_persistence_rows_are_zero(self) -> None:
        torch.manual_seed(2)
        model = _RouterOnlyModel()
        context = torch.randn(4, 68, requires_grad=True)
        prev = torch.randn(4, 68, requires_grad=True)
        loss, stats = compute_v6i5_router_loss(
            model,
            router_contexts=context,
            previous_router_contexts=prev,
            executed_z=torch.tensor([0, 1, 2, 3]),
            old_log_probs=torch.zeros(4),
            advantages=torch.ones(4),
            opportunity_mask=torch.ones(4, dtype=torch.bool),
            persistence_mask=torch.zeros(4, dtype=torch.bool),
            clip_range=0.2,
            latent_k=4,
            ppo_coef=0.2,
            persistence_coef=0.02,
            entropy_coef=0.0,
            entropy_objective="maximize",
            include_rollout_marginal_entropy=False,
            device=torch.device("cpu"),
        )
        self.assertEqual(float(stats["persistence_loss"].detach().item()), 0.0)
        loss.backward()
        self.assertIsNotNone(context.grad)
        self.assertIsNone(prev.grad)


class V6I5TelemetryContractTests(unittest.TestCase):
    def test_required_diagnostic_columns_are_always_in_update_schema(self) -> None:
        fields = set(_update_fieldnames(use_latent_strategy=True, latent_k=4))
        required = {
            "actor_z_jsd_mean",
            "actor_z_jsd_min",
            "actor_z_jsd_max",
            "actor_z_pairs_total",
            "actor_z_pairs_above_margin",
            "actor_z_pairs_above_margin_fraction",
            "actor_z_eval_state_count",
            "actor_z_eval_pair_count",
            "cf_batch_pair_jsd_mean",
            "cf_batch_pair_jsd_min",
            "cf_batch_pair_jsd_max",
            "cf_batch_pairs_total",
            "cf_batch_pairs_above_margin",
            "cf_batch_pairs_above_margin_fraction",
            "cf_valid_sample_count",
            "cf_valid_pair_count",
            "actor_cf_valid_pair_count",
            "actor_grad_norm_total",
            "actor_grad_norm_ppo",
            "actor_grad_norm_cf",
            "actor_ppo_grad_norm",
            "actor_cf_grad_norm_scaled",
            "actor_cf_to_ppo_grad_ratio",
            "actor_grad_norm_other_aux",
            "actor_grad_ratio_cf_to_ppo",
            "actor_grad_ratio_cf_to_ppo_denominator_clamped",
            "actor_grad_ppo_valid",
            "actor_grad_cf_valid",
            "actor_cf_loss_evaluated",
            "actor_grad_cf_inactive_reason",
            "actor_pathway_grad_valid",
            "actor_pathway_grad_norm_local_encoder",
            "actor_pathway_grad_norm_z_embedding",
            "actor_pathway_grad_norm_film",
            "actor_pathway_grad_norm_policy_head",
            "actor_pathway_cf_grad_norm_local_encoder",
            "actor_pathway_cf_grad_norm_z_embedding",
            "actor_pathway_cf_grad_norm_film",
            "actor_pathway_cf_grad_norm_policy_head",
            "actor_z_embedding_cf_grad_norm",
            "actor_film_gamma_cf_grad_norm",
            "actor_film_beta_cf_grad_norm",
            "z_resampled_actual",
            "router_opportunity_count",
            "persistence_valid_pair_count",
            "router_marginal_entropy_application_count",
            "router_marginal_entropy_effective_coefficient",
            "router_marginal_entropy_row_count",
            "actor_cf_update_mode",
            "actor_cf_update_mode_code",
            "actor_ppo_optimizer_step_count",
            "actor_cf_optimizer_step_count",
            "actor_jsd_before_substeps",
            "actor_jsd_after_ppo",
            "actor_jsd_after_cf",
            "actor_jsd_after_first_substep",
            "actor_jsd_after_second_substep",
            "actor_jsd_update_start",
            "ppo_jsd_delta",
            "cf_jsd_delta",
            "cf_gain",
            "retained_cf_gain",
            "cf_retention_ratio",
            "cf_retention_reason",
            "cf_retention_reason_code",
            "actor_kl_after_ppo",
            "actor_kl_after_cf",
            "actor_kl_after_second_substep",
        }
        self.assertFalse(required - fields)

    def test_update_accumulator_schema_tracks_canonical_cf_fields(self) -> None:
        schema = build_metric_schema(latent_k=4, pair_count=6)
        for key in (
            "cf_batch_pair_jsd_mean",
            "cf_batch_pair_jsd_min",
            "cf_batch_pair_jsd_max",
            "actor_grad_ratio_cf_to_ppo",
            "actor_cf_to_ppo_grad_ratio",
            "actor_pathway_cf_grad_norm_z_embedding",
            "actor_z_embedding_cf_grad_norm",
            "router_marginal_entropy_row_count",
            "actor_cf_update_mode_code",
            "actor_ppo_optimizer_step_count",
            "actor_cf_optimizer_step_count",
            "actor_jsd_before_substeps",
            "actor_jsd_after_ppo",
            "actor_jsd_after_cf",
            "actor_jsd_update_start",
            "ppo_jsd_delta",
            "cf_jsd_delta",
            "cf_retention_ratio",
            "actor_kl_after_second_substep",
        ):
            self.assertIn(key, schema)

    def test_actor_cf_update_mode_config_and_cli_contract(self) -> None:
        for mode in ("combined", "ppo_then_cf", "cf_then_ppo"):
            cfg = PPOConfig(actor_cf_update_mode=mode)
            self.assertEqual(normalize_and_validate_training_config(cfg).actor_cf_update_mode, mode)

        cfg = PPOConfig(actor_cf_update_mode="bad_mode")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            normalize_and_validate_training_config(cfg)

        parsed = parse_train_args(["--preset", "v6i5", "--actor-cf-update-mode", "cf_then_ppo"])
        self.assertEqual(cfg_from_args(parsed).actor_cf_update_mode, "cf_then_ppo")

        parsed_legacy = parse_train_args(["--preset", "v6i5", "--latent-cf-sequential-update"])
        self.assertEqual(cfg_from_args(parsed_legacy).actor_cf_update_mode, "ppo_then_cf")

    def test_v6_actor_cf_optimizer_has_independent_state_over_actor_params(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i5")
        hparams = SimpleNamespace(
            learning_rate=1e-3,
            use_latent_strategy=True,
            fixed_latent_strategy=False,
            latent_episode_strategy_lr=None,
        )
        bundle = TrainerOptimizerBundle.build(model=_TinyV6Model(), cfg=cfg, hparams=hparams)
        self.assertIsNot(bundle.actor, bundle.actor_cf)
        self.assertIsNotNone(bundle.actor_cf)
        actor_params = [id(p) for group in bundle.actor.param_groups for p in group["params"]]
        cf_params = [id(p) for group in bundle.actor_cf.param_groups for p in group["params"]]
        self.assertEqual(actor_params, cf_params)
        self.assertEqual(bundle.actor.state, {})
        self.assertEqual(bundle.actor_cf.state, {})

    def test_update_order_retention_math(self) -> None:
        ppo_then_cf = update_order_jsd_metrics(
            mode="ppo_then_cf",
            before=0.10,
            after_ppo=0.07,
            after_cf=0.12,
        )
        self.assertAlmostEqual(float(ppo_then_cf["ppo_jsd_delta"]), -0.03)
        self.assertAlmostEqual(float(ppo_then_cf["cf_jsd_delta"]), 0.05)
        self.assertAlmostEqual(float(ppo_then_cf["cf_gain"]), 0.05)
        self.assertAlmostEqual(float(ppo_then_cf["retained_cf_gain"]), 0.05)
        self.assertAlmostEqual(float(ppo_then_cf["cf_retention_ratio"]), 1.0)
        self.assertEqual(ppo_then_cf["cf_retention_reason"], "")

        cf_then_ppo = update_order_jsd_metrics(
            mode="cf_then_ppo",
            before=0.10,
            after_cf=0.15,
            after_ppo=0.11,
        )
        self.assertAlmostEqual(float(cf_then_ppo["cf_jsd_delta"]), 0.05)
        self.assertAlmostEqual(float(cf_then_ppo["ppo_jsd_delta"]), -0.04)
        self.assertAlmostEqual(float(cf_then_ppo["retained_cf_gain"]), 0.01)
        self.assertAlmostEqual(float(cf_then_ppo["cf_retention_ratio"]), 0.2)

        no_gain = update_order_jsd_metrics(
            mode="cf_then_ppo",
            before=0.10,
            after_cf=0.10,
            after_ppo=0.09,
        )
        self.assertTrue(torch.isnan(torch.tensor(float(no_gain["cf_retention_ratio"]))))
        self.assertEqual(no_gain["cf_retention_reason"], "no_measurable_cf_gain")

    def test_rollout_stats_count_actual_router_opportunities(self) -> None:
        buffer = SimpleNamespace(
            pos=2,
            fields={
                "z": torch.tensor([[0, 1], [1, 2]]),
                "prev_z": torch.tensor([[0, 1], [0, 2]]),
                "z_persist_mask": torch.tensor([[False, False], [True, False]]),
                "z_resampled": torch.zeros((2, 2), dtype=torch.bool),
                "z_resampled_actual": torch.tensor([[True, True], [False, True]]),
                "persistence_valid": torch.tensor([[False, False], [True, False]]),
            },
        )
        trainer = SimpleNamespace(use_latent_strategy=True, latent_k=4)
        stats = _latent_rollout_stats(trainer, buffer)
        self.assertEqual(stats["strategy_resample_count"], 3.0)
        self.assertEqual(stats["router_opportunity_count"], 3.0)
        self.assertEqual(stats["persistence_valid_pair_count"], 1.0)


if __name__ == "__main__":
    unittest.main()
