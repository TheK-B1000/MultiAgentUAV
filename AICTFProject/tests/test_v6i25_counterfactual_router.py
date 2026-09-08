"""Unit tests for V6I25 counterfactual geometry→z router (corrected protocol)."""
from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from rl.router.counterfactual_router import (
    advantages_from_returns,
    assert_valid_geometry_context,
    assign_cross_fitted_z,
    build_geometry_q_table,
    counterfactual_router_loss,
    decide_v6i25_verdict,
    geometry_context_report,
    geometry_key,
    soft_q_router_loss,
    soft_target_from_q,
    soft_targets_from_geometry_q,
    stage_a_signal_validation,
    stage_b_router_eval,
    StageAResult,
    StageBResult,
    train_test_split_indices,
)


class SoftTargetLossTests(unittest.TestCase):
    def test_soft_target_uniform_when_q_flat(self):
        q = torch.zeros(2, 4)
        p = soft_target_from_q(q, temperature=1.0)
        self.assertTrue(torch.allclose(p, torch.full_like(p, 0.25), atol=1e-5))

    def test_soft_target_peaks_on_best_z(self):
        q = torch.tensor([[0.0, 10.0, 0.0, 0.0]])
        p = soft_target_from_q(q, temperature=0.5)
        self.assertEqual(int(p.argmax(dim=-1).item()), 1)
        self.assertGreater(float(p[0, 1]), 0.9)

    def test_soft_q_loss_zero_at_perfect_match(self):
        logits = torch.tensor([[0.0, 5.0, 0.0, 0.0]])
        targets = soft_target_from_q(torch.tensor([[0.0, 5.0, 0.0, 0.0]]), temperature=1.0)
        loss = soft_q_router_loss(logits, targets, spread_floor=1e-6)
        # CE at matching peaked distributions is small but not exactly 0
        self.assertLess(float(loss.item()), 0.2)

    def test_soft_q_loss_masks_negligible_spread(self):
        logits = torch.randn(4, 4, requires_grad=True)
        targets = soft_target_from_q(torch.zeros(4, 4), temperature=1.0)
        q = torch.zeros(4, 4)
        loss = soft_q_router_loss(logits, targets, spread_floor=0.1, q_for_mask=q)
        self.assertEqual(float(loss.item()), 0.0)
        loss.backward()
        self.assertTrue(torch.allclose(logits.grad, torch.zeros_like(logits.grad)))

    def test_advantage_ablation_helper_still_works(self):
        returns = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        adv = advantages_from_returns(returns)
        self.assertAlmostEqual(float(adv.sum().item()), 0.0, places=5)
        logits = torch.zeros(1, 4, requires_grad=True)
        loss = counterfactual_router_loss(logits, adv)
        loss.backward()
        self.assertIsNotNone(logits.grad)


class GeometryOracleTests(unittest.TestCase):
    def test_geometry_key_quantizes(self):
        a = geometry_key(np.array([1.23456, 2.0]), decimals=2)
        b = geometry_key(np.array([1.23499, 2.001]), decimals=2)
        self.assertEqual(a, b)

    def test_assert_rejects_all_zero(self):
        with self.assertRaises(ValueError):
            assert_valid_geometry_context(np.zeros(35))

    def test_assert_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            assert_valid_geometry_context(np.array([1.0, np.nan, 0.0]))

    def test_build_q_aggregates_same_geometry_across_opponents(self):
        # Two opponents, identical start geometry, conflicting winners → average.
        ctx = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        ret = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # OP8: z0 wins
                [0.0, 1.0, 0.0, 0.0],  # OP9: z1 wins
            ],
            dtype=np.float64,
        )
        table = build_geometry_q_table(ctx, ret, decimals=4)
        self.assertEqual(len(table.q_by_key), 1)
        q = next(iter(table.q_by_key.values()))
        np.testing.assert_allclose(q, np.array([0.5, 0.5, 0.0, 0.0]))

    def test_cross_fitted_oracle_uses_train_not_hindsight(self):
        # Train: geometry A prefers z=0; held-out episode for A happened to win on z=3.
        train_ctx = np.array([[1.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        train_ret = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        table = build_geometry_q_table(train_ctx, train_ret, decimals=4)
        held_ctx = np.array([[1.0, 0.0]], dtype=np.float64)
        held_ret = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)  # hindsight max = z3
        z_star = assign_cross_fitted_z(held_ctx, table, decimals=4)
        self.assertEqual(int(z_star[0]), 0)  # train Q, not hindsight
        # Context-oracle return is R(held, z*=0)=0, not hindsight 1.
        self.assertEqual(float(held_ret[0, z_star[0]]), 0.0)
        self.assertEqual(float(held_ret.max()), 1.0)

    def test_stage_a_detects_predictable_gap(self):
        rng = np.random.default_rng(0)
        # Two geometries; each has a clear best z on train and held-out.
        train_ctx = np.vstack(
            [np.tile([1.0, 0.0], (20, 1)), np.tile([0.0, 1.0], (20, 1))]
        )
        train_ret = np.zeros((40, 4))
        train_ret[:20, 0] = 1.0
        train_ret[20:, 1] = 1.0
        test_ctx = train_ctx.copy()
        test_ret = train_ret.copy()
        # Add a little noise but keep mean ordering.
        test_ret = np.clip(test_ret + rng.normal(0, 0.01, size=test_ret.shape), 0, 1)
        table = build_geometry_q_table(train_ctx, train_ret, decimals=4)
        stage_a = stage_a_signal_validation(
            test_ret,
            test_ctx,
            table,
            train_returns_for_best_fixed=train_ret,
            decimals=4,
            n_bootstrap=200,
            seed=0,
        )
        self.assertTrue(stage_a.signal_ok)
        self.assertGreater(stage_a.delta, 0.2)

    def test_stage_a_fails_when_no_geometry_signal(self):
        # Returns independent of geometry; best fixed equals any oracle.
        rng = np.random.default_rng(1)
        ctx = rng.normal(size=(40, 3))
        ret = np.tile(np.array([0.6, 0.55, 0.5, 0.45]), (40, 1))
        table = build_geometry_q_table(ctx[:30], ret[:30], decimals=4)
        stage_a = stage_a_signal_validation(
            ret[30:],
            ctx[30:],
            table,
            train_returns_for_best_fixed=ret[:30],
            decimals=4,
            n_bootstrap=200,
            seed=1,
        )
        self.assertFalse(stage_a.signal_ok)

    def test_geometry_report_unique_and_duplicate_rate(self):
        ctx = np.array([[1.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        report = geometry_context_report(ctx)
        self.assertEqual(report["n_unique_contexts"], 2)
        self.assertAlmostEqual(report["duplicate_context_rate"], 1.0 / 3.0, places=5)


class VerdictTests(unittest.TestCase):
    def _stage_a_ok(self) -> StageAResult:
        return StageAResult(
            context_oracle_mean=0.7,
            best_fixed_mean=0.4,
            best_fixed_z=0,
            delta=0.3,
            ci_low=0.1,
            ci_high=0.5,
            signal_ok=True,
            n=40,
        )

    def test_fail_signal(self):
        a = self._stage_a_ok()
        a.signal_ok = False
        self.assertEqual(decide_v6i25_verdict(a, None), "FAIL_SIGNAL")

    def test_pass(self):
        b = StageBResult(
            router_mean=0.6,
            uniform_mean=0.4,
            best_fixed_mean=0.4,
            context_oracle_mean=0.7,
            best_fixed_z=0,
            delta_router_minus_best_fixed=0.2,
            router_ci_low=0.05,
            router_ci_high=0.35,
            router_beats_best_fixed=True,
            gap_recovery=0.2 / 0.3,
            n=40,
        )
        self.assertGreaterEqual(b.gap_recovery, 0.5)
        self.assertEqual(decide_v6i25_verdict(self._stage_a_ok(), b), "PASS")

    def test_partial_low_recovery(self):
        b = StageBResult(
            router_mean=0.5,
            uniform_mean=0.4,
            best_fixed_mean=0.4,
            context_oracle_mean=0.7,
            best_fixed_z=0,
            delta_router_minus_best_fixed=0.1,
            router_ci_low=0.02,
            router_ci_high=0.2,
            router_beats_best_fixed=True,
            gap_recovery=0.1 / 0.3,
            n=40,
        )
        self.assertLess(b.gap_recovery, 0.5)
        self.assertEqual(decide_v6i25_verdict(self._stage_a_ok(), b), "PARTIAL")

    def test_fail_router(self):
        b = StageBResult(
            router_mean=0.35,
            uniform_mean=0.4,
            best_fixed_mean=0.4,
            context_oracle_mean=0.7,
            best_fixed_z=0,
            delta_router_minus_best_fixed=-0.05,
            router_ci_low=-0.2,
            router_ci_high=0.1,
            router_beats_best_fixed=False,
            gap_recovery=-0.05 / 0.3,
            n=40,
        )
        self.assertEqual(decide_v6i25_verdict(self._stage_a_ok(), b), "FAIL_ROUTER")


class SoftTargetsFromTableTests(unittest.TestCase):
    def test_targets_follow_q_table(self):
        ctx = np.array([[1.0], [2.0]], dtype=np.float64)
        ret = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
        table = build_geometry_q_table(ctx, ret, decimals=4)
        targets, q_rows = soft_targets_from_geometry_q(ctx, table, temperature=0.25)
        self.assertEqual(int(targets[0].argmax()), 0)
        self.assertEqual(int(targets[1].argmax()), 2)
        np.testing.assert_allclose(q_rows[0], ret[0])


class SplitTests(unittest.TestCase):
    def test_split_disjoint(self):
        train, test = train_test_split_indices(20, test_frac=0.25, seed=0)
        self.assertEqual(len(train) + len(test), 20)
        self.assertEqual(len(set(train) & set(test)), 0)


class TinyRouterTrainSmoke(unittest.TestCase):
    def test_soft_train_moves_logits(self):
        from rl.router.counterfactual_router import train_counterfactual_router

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.uses_latent_strategy = True
                self.strategy_encoder = nn.Linear(4, 4)
                self.selector_gru = None
                self.global_state_dim = 4
                self.q_phi_input_dim = 4

            def strategy_logits(self, global_state, selector_hidden=None):
                return self.strategy_encoder(global_state)

        model = Tiny()
        ctx = torch.randn(32, 4)
        # Geometry ≈ feature[0]; Q prefers z=0 when feature[0]>0 else z=1
        q = torch.zeros(32, 4)
        q[:, 0] = (ctx[:, 0] > 0).float()
        q[:, 1] = (ctx[:, 0] <= 0).float()
        targets = soft_target_from_q(q, temperature=0.5)
        before = model.strategy_encoder.weight.detach().clone()
        train_counterfactual_router(
            model,
            ctx,
            targets,
            q_values=q,
            n_steps=50,
            batch_size=16,
            lr=1e-2,
            spread_floor=1e-6,
            device="cpu",
            seed=0,
        )
        after = model.strategy_encoder.weight.detach()
        self.assertFalse(torch.allclose(before, after))


if __name__ == "__main__":
    unittest.main()
