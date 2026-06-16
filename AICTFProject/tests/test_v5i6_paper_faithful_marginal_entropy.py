"""Focused tests for the v5i6 paper-faithful marginal-entropy preset."""

from __future__ import annotations

import dataclasses
import io
import math
import unittest
from contextlib import redirect_stdout

import torch

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.csv_writers import _update_fieldnames
from rl.custom_ppo.schedules import resolve_latent_forced_z_frac
from rl.latent_losses import (
    rollout_marginal_entropy_loss,
    rollout_router_soft_diagnostics,
)
from rl.presets import apply_preset
from rl.presets.plan_faithful import (
    apply_plan_faithful_latent_v5i4_end_to_end,
    apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor,
    apply_plan_faithful_latent_v5i6_paper_faithful_marginal_entropy,
)
from rl.training.banner import _maybe_print_paper_faithful_audit


V5I6_ALIASES = (
    "v5i6",
    "v5i6_paper_faithful",
    "v5i6_paper_faithful_marginal_entropy",
    "v5i6_marginal_entropy",
    "paper_faithful_marginal_entropy",
    "latent_v5i6_paper_faithful",
    "latent_v5i6_paper_faithful_marginal_entropy",
    "latent_v5i6_marginal_entropy",
    "plan_faithful_latent_v5i6_paper_faithful_marginal_entropy",
    "plan_faithful_latent_v5i6_marginal_entropy",
)


def _resolved(name: str) -> PPOConfig:
    return apply_preset(PPOConfig(), name)


class V5i6PresetInheritanceTests(unittest.TestCase):
    def test_v5i6_diff_vs_v5i4_is_entropy_reduction_schedule_floor_and_tag(self) -> None:
        v5i4 = dataclasses.asdict(apply_plan_faithful_latent_v5i4_end_to_end(PPOConfig()))
        v5i6 = dataclasses.asdict(
            apply_plan_faithful_latent_v5i6_paper_faithful_marginal_entropy(
                PPOConfig()
            )
        )
        diffs = {
            k: (v5i4.get(k), v5i6.get(k))
            for k in set(v5i4) | set(v5i6)
            if v5i4.get(k) != v5i6.get(k)
        }
        self.assertEqual(set(diffs), {"latent_entropy_mode", "latent_lam_h_end", "run_tag"})
        self.assertEqual(diffs["latent_entropy_mode"], ("conditional", "marginal"))
        self.assertAlmostEqual(float(v5i4["latent_lam_h_end"]), 0.0002)
        self.assertAlmostEqual(float(v5i6["latent_lam_h_end"]), 0.001)

    def test_v5i6_diff_vs_v5i5_is_entropy_reduction_only_plus_tag(self) -> None:
        v5i5 = dataclasses.asdict(
            apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor(PPOConfig())
        )
        v5i6 = dataclasses.asdict(
            apply_plan_faithful_latent_v5i6_paper_faithful_marginal_entropy(
                PPOConfig()
            )
        )
        diffs = {
            k: (v5i5.get(k), v5i6.get(k))
            for k in set(v5i5) | set(v5i6)
            if v5i5.get(k) != v5i6.get(k)
        }
        self.assertEqual(set(diffs), {"latent_entropy_mode", "run_tag"})
        self.assertEqual(diffs["latent_entropy_mode"], ("conditional", "marginal"))

    def test_v5i6_entropy_contract(self) -> None:
        cfg = _resolved("v5i6")
        self.assertEqual(str(cfg.latent_entropy_mode), "marginal")
        self.assertEqual(str(cfg.latent_entropy_objective), "maximize")
        self.assertAlmostEqual(float(cfg.latent_lam_h_start), 0.003)
        self.assertAlmostEqual(float(cfg.latent_lam_h), 0.003)
        self.assertAlmostEqual(float(cfg.latent_lam_h_end), 0.001)
        self.assertEqual(int(cfg.latent_entropy_anneal_start), 0)
        self.assertEqual(int(cfg.latent_entropy_anneal_end), 300_000)
        self.assertAlmostEqual(float(cfg.latent_usage_balance_coef), 0.0)

    def test_v5i6_keeps_v5i4_core_contract(self) -> None:
        cfg = _resolved("v5i6")
        self.assertTrue(bool(cfg.use_latent_strategy))
        self.assertEqual(int(cfg.latent_k), 4)
        self.assertEqual(int(cfg.latent_z_embed_dim), 16)
        self.assertEqual(int(cfg.latent_resample_every_n), 64)
        self.assertFalse(bool(cfg.latent_resample_on_flag))
        self.assertAlmostEqual(float(cfg.latent_lam_p), 0.03)
        self.assertAlmostEqual(float(cfg.latent_strategy_ppo_coef), 0.10)
        self.assertFalse(bool(cfg.enable_actor_z_film))
        self.assertFalse(bool(cfg.latent_actor_z_adapter_enabled))
        self.assertFalse(bool(cfg.latent_actor_z_onehot_enabled))
        self.assertFalse(bool(cfg.latent_episode_strategy_ppo))
        self.assertIsNone(cfg.latent_episode_strategy_lr)
        self.assertFalse(bool(cfg.latent_arc_credit_enabled))
        self.assertFalse(bool(cfg.latent_router_distill_enabled))
        self.assertFalse(bool(cfg.latent_strategy_aux_return_head))
        self.assertAlmostEqual(float(cfg.latent_strategy_aux_predict_phase_coef), 0.0)
        self.assertAlmostEqual(float(cfg.latent_conditional_entropy_min_coef), 0.0)
        self.assertAlmostEqual(float(cfg.latent_marginal_balance_coef), 0.0)

    def test_forced_z_resolves_to_zero_at_every_step(self) -> None:
        cfg = _resolved("v5i6")
        for step in (0, 10_000, 200_000, 500_000, 1_000_000):
            self.assertAlmostEqual(
                resolve_latent_forced_z_frac(cfg, global_step=step),
                0.0,
                places=8,
            )

    def test_v5i6_run_tag_advertises_marginal_entropy_and_correct_budget(self) -> None:
        cfg = _resolved("v5i6")
        self.assertEqual(int(cfg.total_timesteps), 1_000_000)
        self.assertIn("v5i6_paper_faithful_marginal_entropy", cfg.run_tag)
        self.assertIn("OP5_OP6_OP7", cfg.run_tag)
        self.assertIn("_1m_", cfg.run_tag)
        self.assertNotIn("_2m_", cfg.run_tag)


class V5i6AliasSnapshotTests(unittest.TestCase):
    def test_all_aliases_resolve_to_identical_config(self) -> None:
        baseline = dataclasses.asdict(_resolved(V5I6_ALIASES[0]))
        for name in V5I6_ALIASES[1:]:
            current = dataclasses.asdict(_resolved(name))
            self.assertEqual(current, baseline, f"alias {name!r} must match v5i6")


class V5i6PaperFaithfulAuditBannerTests(unittest.TestCase):
    def _capture(self, cfg: PPOConfig) -> str:
        buf = io.StringIO()
        with redirect_stdout(buf):
            _maybe_print_paper_faithful_audit(cfg)
        return buf.getvalue()

    def test_audit_banner_fires_for_v5i6_with_marginal_mode(self) -> None:
        cfg = _resolved("v5i6")
        out = self._capture(cfg)
        self.assertIn("v5i6 paper-faithful audit", out)
        self.assertIn("q_phi task-reward PPO: ON", out)
        self.assertIn("persistence: ON", out)
        self.assertIn("entropy maximization: ON (mode=marginal", out)
        # v5i6 must aggregate p_bar over the FULL rollout resample subset,
        # not per-PPO-minibatch (Jensen bias). The banner advertises this so
        # reviewers see the aggregation choice on every run header.
        self.assertIn("aggregation=rollout", out)
        self.assertIn("resampling cadence: every 64 decisions", out)
        self.assertNotIn("v5i4 paper-faithful audit", out)
        self.assertNotIn("v5i5 paper-faithful audit", out)
        self.assertNotIn("v5i6 audit WARNING", out)

    def test_audit_banner_warns_when_v5i6_is_not_marginal(self) -> None:
        cfg = _resolved("v5i6")
        cfg.latent_entropy_mode = "conditional"
        out = self._capture(cfg)
        self.assertIn("v5i6 audit WARNING", out)
        self.assertIn("latent_entropy_mode='conditional'", out)


class V5i6TelemetrySchemaTests(unittest.TestCase):
    def test_metrics_csv_header_includes_marginal_entropy_loss_fields(self) -> None:
        fields = _update_fieldnames(use_latent_strategy=True, latent_k=4)
        for column in (
            "strategy_marginal_entropy_loss",
            "strategy_marginal_entropy_nats",
            "strategy_marginal_entropy_kl",
        ):
            self.assertIn(column, fields)

    def test_metrics_csv_header_includes_rollout_soft_router_diagnostics(self) -> None:
        """v5i6 telemetry contract: rollout-level soft-router diagnostics."""
        fields = _update_fieldnames(use_latent_strategy=True, latent_k=4)
        for column in (
            "router_rollout_soft_marginal_entropy_nats",
            "router_rollout_soft_conditional_entropy_nats",
            "router_rollout_soft_mi_proxy_nats",
            "router_rollout_soft_argmax_occupancy_max",
            "router_rollout_soft_argmax_occupancy_min",
            "router_rollout_soft_argmax_occupancy_ratio",
            "router_rollout_resample_count",
        ):
            self.assertIn(column, fields, f"missing column {column!r}")
        for k in range(4):
            self.assertIn(f"router_rollout_soft_p_bar_z{k}", fields)


class V5i6RolloutMarginalEntropyContractTests(unittest.TestCase):
    """The v5i6 contract: optimize ``KL( N^{-1} sum_i q_phi(z|s_i) || U )``
    over ALL rollout resample-decision points, not per-minibatch.
    """

    def test_perfect_specialization_with_uniform_state_distribution_yields_zero_kl(self) -> None:
        """The desired Summer outcome: 4 z's, perfectly disjoint state
        subsets of equal size, perfectly confident logits. The rollout-level
        loss must be ~0 (we are already at the optimum); a per-minibatch
        scheme would still report nonzero loss on minibatches that happen
        to oversample one z.
        """
        K = 4
        large = 50.0
        logits = torch.zeros((4 * 16, K), dtype=torch.float32)
        for z in range(K):
            logits[z * 16 : (z + 1) * 16, z] = large
        loss, stats = rollout_marginal_entropy_loss(
            logits,
            objective="maximize",
            lam_h=0.001,
            latent_k=K,
            device=torch.device("cpu"),
        )
        self.assertLess(float(loss.item()), 1e-5)
        self.assertLess(stats["rollout_marginal_entropy_kl"], 1e-4)
        self.assertAlmostEqual(stats["rollout_marginal_entropy_nats"], math.log(K), places=4)
        self.assertLess(stats["rollout_conditional_entropy_nats"], 1e-3)
        self.assertAlmostEqual(stats["rollout_mi_proxy_nats"], math.log(K), places=3)

    def test_collapse_attracts_strong_gradient(self) -> None:
        """Bias case: most mass on z=1 but not saturated (so softmax has
        non-vanishing Jacobian). Gradient must point AWAY from the
        collapsed mode: dLoss/dlogits[:,1] > 0 (decrease) and
        dLoss/dlogits[:,non-1] < 0 (increase).
        """
        K = 4
        logits = torch.zeros((64, K), dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            # Logit gap of ~3 -> softmax ~ [0.046, 0.86, 0.046, 0.046]:
            # marginal still ~one-hot but Jacobian is non-degenerate.
            logits[:, 1] = 3.0
        loss, stats = rollout_marginal_entropy_loss(
            logits,
            objective="maximize",
            lam_h=1.0,
            latent_k=K,
            device=torch.device("cpu"),
        )
        # KL(p_bar || U) = sum p_bar log(p_bar K) > 0 for non-uniform p_bar.
        self.assertGreater(stats["rollout_marginal_entropy_kl"], 0.5)
        loss.backward()
        # Sum of grad on z=1 column must be POSITIVE (push logits[:,1] down).
        self.assertGreater(float(logits.grad[:, 1].sum().item()), 0.0)
        # And on the underrepresented z's, the gradient on each non-1 column
        # should be NEGATIVE on aggregate (push their logits up).
        for z in (0, 2, 3):
            self.assertLess(float(logits.grad[:, z].sum().item()), 0.0)

    def test_csv_column_documents_aggregation_unit(self) -> None:
        """Schema must clearly distinguish soft-router rollout aggregation
        (this test) from sampled-z occupancy (latent_occupancy_*) so
        analysts don't conflate them.
        """
        fields = _update_fieldnames(use_latent_strategy=True, latent_k=4)
        soft_columns = {f for f in fields if f.startswith("router_rollout_soft_")}
        sampled_columns = {
            "latent_marginal_entropy_nats",
            "effective_num_latents",
            "latent_occupancy_min",
            "latent_occupancy_max",
            "latent_occupancy_ratio",
        }
        # Both groups must coexist with no name clobbering.
        self.assertGreater(len(soft_columns), 0)
        self.assertEqual(soft_columns & sampled_columns, set())


if __name__ == "__main__":
    unittest.main()
