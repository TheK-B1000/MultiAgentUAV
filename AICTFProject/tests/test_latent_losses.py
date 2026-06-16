"""Parity tests for ``rl.latent_losses``.

Each test exercises one pure helper against a hand-coded reference
computation, asserting bitwise tensor equality. Together they pin the
contract that the latent strategy loss math is unchanged from when it lived
inline inside ``CustomPPOTrainer.update``.
"""
from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from rl.latent_losses import (
    rollout_marginal_entropy_loss,
    rollout_router_soft_diagnostics,
    strategy_aux_return_loss,
    strategy_entropy_loss,
    strategy_kl_consecutive_loss,
    strategy_marginal_entropy_loss,
    strategy_persistence_loss,
    strategy_phase_aux_loss,
    strategy_ppo_loss,
)
from rl.latent_marl import expected_strategy_switch_penalty
from rl.ppo_core import ppo_policy_loss


DEVICE = torch.device("cpu")


def _set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)


def _exact_eq(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(torch.equal(a, b))


class StrategyEntropyLossTests(unittest.TestCase):
    def test_maximize_objective_is_negative_lam_h_times_masked_mean(self) -> None:
        _set_seed(0)
        h = torch.tensor([0.3, 0.5, 0.7, 0.9], dtype=torch.float32)
        mask = torch.tensor([True, False, True, True])
        loss, stats = strategy_entropy_loss(h, mask, objective="maximize", lam_h=0.01, device=DEVICE)
        expected = -0.01 * h[mask].mean()
        self.assertTrue(_exact_eq(loss, expected), f"loss={loss}, expected={expected}")
        self.assertAlmostEqual(stats["strategy_entropy_term_mean"], float(h[mask].mean()))

    def test_minimize_objective_flips_sign(self) -> None:
        h = torch.tensor([0.4, 0.6], dtype=torch.float32)
        mask = torch.tensor([True, True])
        loss, _ = strategy_entropy_loss(h, mask, objective="minimize", lam_h=0.02, device=DEVICE)
        expected = 0.02 * h.mean()
        self.assertTrue(_exact_eq(loss, expected))

    def test_none_objective_returns_zero_no_grad(self) -> None:
        h = torch.tensor([0.1, 0.2], dtype=torch.float32, requires_grad=True)
        mask = torch.tensor([True, True])
        loss, _ = strategy_entropy_loss(h, mask, objective="none", lam_h=0.5, device=DEVICE)
        self.assertEqual(loss.item(), 0.0)
        self.assertFalse(loss.requires_grad)

    def test_zero_lam_h_returns_zero_loss_regardless_of_objective(self) -> None:
        h = torch.tensor([0.1, 0.2], dtype=torch.float32)
        mask = torch.tensor([True, True])
        loss, _ = strategy_entropy_loss(h, mask, objective="maximize", lam_h=0.0, device=DEVICE)
        self.assertEqual(loss.item(), 0.0)

    def test_empty_mask_uses_zero_h_mean(self) -> None:
        h = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        mask = torch.tensor([False, False, False])
        loss, stats = strategy_entropy_loss(h, mask, objective="maximize", lam_h=0.01, device=DEVICE)
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(stats["strategy_entropy_term_mean"], 0.0)


class StrategyMarginalEntropyLossTests(unittest.TestCase):
    def test_maximize_objective_minimizes_batch_marginal_kl_to_uniform(self) -> None:
        _set_seed(0)
        logits = torch.randn(6, 4)
        mask = torch.tensor([True, False, True, True, False, True])
        loss, stats = strategy_marginal_entropy_loss(
            logits,
            mask,
            objective="maximize",
            lam_h=0.003,
            latent_k=4,
            device=DEVICE,
        )
        p_bar = torch.softmax(logits[mask], dim=-1).mean(dim=0).clamp_min(1e-8)
        ref_kl = (p_bar * (torch.log(p_bar) + torch.log(p_bar.new_tensor(4.0)))).sum()
        ref_h = -(p_bar * torch.log(p_bar)).sum()
        self.assertTrue(_exact_eq(loss, 0.003 * ref_kl))
        self.assertAlmostEqual(stats["strategy_marginal_entropy_kl"], float(ref_kl))
        self.assertAlmostEqual(stats["strategy_marginal_entropy_nats"], float(ref_h))

    def test_minimize_objective_flips_marginal_kl_sign(self) -> None:
        logits = torch.tensor(
            [[3.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        mask = torch.tensor([True, True])
        loss, _ = strategy_marginal_entropy_loss(
            logits,
            mask,
            objective="minimize",
            lam_h=0.01,
            latent_k=4,
            device=DEVICE,
        )
        p_bar = torch.softmax(logits, dim=-1).mean(dim=0).clamp_min(1e-8)
        ref_kl = (p_bar * (torch.log(p_bar) + torch.log(p_bar.new_tensor(4.0)))).sum()
        self.assertTrue(_exact_eq(loss, -0.01 * ref_kl))

    def test_empty_mask_returns_zero_no_grad(self) -> None:
        logits = torch.randn(3, 4, requires_grad=True)
        mask = torch.zeros(3, dtype=torch.bool)
        loss, stats = strategy_marginal_entropy_loss(
            logits,
            mask,
            objective="maximize",
            lam_h=0.01,
            latent_k=4,
            device=DEVICE,
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertFalse(loss.requires_grad)
        self.assertEqual(stats["strategy_marginal_entropy_kl"], 0.0)
        self.assertEqual(stats["strategy_marginal_entropy_nats"], 0.0)


class RolloutMarginalEntropyLossTests(unittest.TestCase):
    """v5i6 rollout-level marginal entropy loss.

    Distinct from ``StrategyMarginalEntropyLossTests`` above (which pins the
    deprecated per-minibatch helper kept only for parity). These tests pin
    the gradient path that v5i6 actually optimizes.
    """

    def test_maximize_objective_minimizes_rollout_marginal_kl_to_uniform(self) -> None:
        _set_seed(0)
        logits = torch.randn(32, 4)
        loss, stats = rollout_marginal_entropy_loss(
            logits,
            objective="maximize",
            lam_h=0.003,
            latent_k=4,
            device=DEVICE,
        )
        probs = torch.softmax(logits, dim=-1)
        p_bar = probs.mean(dim=0).clamp_min(1e-8)
        ref_kl = (p_bar * (torch.log(p_bar) + torch.log(p_bar.new_tensor(4.0)))).sum()
        ref_h_marg = -(p_bar * torch.log(p_bar)).sum()
        ref_h_cond = -(probs.clamp_min(1e-8) * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()
        self.assertTrue(_exact_eq(loss, 0.003 * ref_kl))
        self.assertAlmostEqual(stats["rollout_marginal_entropy_kl"], float(ref_kl), places=5)
        self.assertAlmostEqual(stats["rollout_marginal_entropy_nats"], float(ref_h_marg), places=5)
        self.assertAlmostEqual(stats["rollout_conditional_entropy_nats"], float(ref_h_cond), places=5)
        self.assertAlmostEqual(
            stats["rollout_mi_proxy_nats"],
            float(ref_h_marg - ref_h_cond),
            places=5,
        )
        self.assertAlmostEqual(stats["rollout_resample_count"], 32.0)

    def test_minimize_objective_flips_sign(self) -> None:
        logits = torch.tensor(
            [[3.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        loss, _ = rollout_marginal_entropy_loss(
            logits,
            objective="minimize",
            lam_h=0.01,
            latent_k=4,
            device=DEVICE,
        )
        p_bar = torch.softmax(logits, dim=-1).mean(dim=0).clamp_min(1e-8)
        ref_kl = (p_bar * (torch.log(p_bar) + torch.log(p_bar.new_tensor(4.0)))).sum()
        self.assertTrue(_exact_eq(loss, -0.01 * ref_kl))

    def test_empty_input_returns_zero_no_grad(self) -> None:
        logits = torch.zeros((0, 4), dtype=torch.float32, requires_grad=True)
        loss, stats = rollout_marginal_entropy_loss(
            logits,
            objective="maximize",
            lam_h=0.01,
            latent_k=4,
            device=DEVICE,
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertFalse(loss.requires_grad)
        self.assertEqual(stats["rollout_marginal_entropy_kl"], 0.0)
        self.assertEqual(stats["rollout_marginal_entropy_nats"], 0.0)
        self.assertEqual(stats["rollout_conditional_entropy_nats"], 0.0)
        self.assertEqual(stats["rollout_mi_proxy_nats"], 0.0)
        self.assertEqual(stats["rollout_resample_count"], 0.0)

    def test_gradient_flows_to_logits(self) -> None:
        logits = torch.randn(8, 4, requires_grad=True)
        loss, _ = rollout_marginal_entropy_loss(
            logits,
            objective="maximize",
            lam_h=0.5,
            latent_k=4,
            device=DEVICE,
        )
        loss.backward()
        self.assertIsNotNone(logits.grad)
        # Gradient must be nonzero for non-uniform p_bar (otherwise nothing
        # learns).
        self.assertGreater(float(logits.grad.abs().sum().item()), 0.0)

    def test_jensen_demo_per_minibatch_upper_bounds_rollout_level(self) -> None:
        """Demonstrates the bug the rollout-level path fixes.

        Construct 4 minibatches of 1 state each, with one-hot logits such
        that the true rollout-marginal is exactly uniform but every
        per-minibatch marginal is one-hot (max KL). Per-minibatch mean KL
        is ``log K = ln 4`` while the rollout-level KL is exactly 0 — the
        Jensen gap that the per-minibatch loss systematically over-applies.
        """
        K = 4
        large = 1e3  # effectively one-hot under softmax
        per_minibatch_logits = [
            torch.tensor([[large, 0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, large, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, large, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0, large]]),
        ]

        per_mb_kls: list[float] = []
        for mb in per_minibatch_logits:
            _, mb_stats = strategy_marginal_entropy_loss(
                mb,
                torch.ones(mb.shape[0], dtype=torch.bool),
                objective="maximize",
                lam_h=1.0,
                latent_k=K,
                device=DEVICE,
            )
            per_mb_kls.append(float(mb_stats["strategy_marginal_entropy_kl"]))

        rollout_logits = torch.cat(per_minibatch_logits, dim=0)
        _, rollout_stats = rollout_marginal_entropy_loss(
            rollout_logits,
            objective="maximize",
            lam_h=1.0,
            latent_k=K,
            device=DEVICE,
        )

        mean_per_minibatch_kl = sum(per_mb_kls) / len(per_mb_kls)
        rollout_kl = float(rollout_stats["rollout_marginal_entropy_kl"])
        # Per-minibatch mean is ln(K); rollout-level is essentially 0.
        self.assertAlmostEqual(mean_per_minibatch_kl, math.log(K), places=4)
        self.assertLess(rollout_kl, 1e-4)
        self.assertGreater(mean_per_minibatch_kl, rollout_kl + 1.0)


class RolloutRouterSoftDiagnosticsTests(unittest.TestCase):
    def test_perfect_specialization_yields_uniform_p_bar_and_zero_conditional(self) -> None:
        """v5i6 happy path: 4 z's, perfectly disjoint state subsets,
        confident logits. Rollout-level diagnostics should report
        ``H_marginal = log K`` and ``H_conditional ~= 0``.
        """
        K = 4
        large = 50.0
        logits = torch.zeros((4 * 16, K), dtype=torch.float32)
        for z in range(K):
            logits[z * 16 : (z + 1) * 16, z] = large
        diag = rollout_router_soft_diagnostics(logits, latent_k=K)
        self.assertAlmostEqual(
            diag["router_rollout_soft_marginal_entropy_nats"],
            math.log(K),
            places=4,
        )
        self.assertLess(diag["router_rollout_soft_conditional_entropy_nats"], 1e-4)
        self.assertAlmostEqual(
            diag["router_rollout_soft_mi_proxy_nats"],
            math.log(K),
            places=4,
        )
        for z in range(K):
            self.assertAlmostEqual(
                diag[f"router_rollout_soft_p_bar_z{z}"], 0.25, places=4
            )
        self.assertAlmostEqual(
            diag["router_rollout_soft_argmax_occupancy_max"], 0.25, places=4
        )
        self.assertAlmostEqual(
            diag["router_rollout_soft_argmax_occupancy_min"], 0.25, places=4
        )
        self.assertAlmostEqual(
            diag["router_rollout_soft_argmax_occupancy_ratio"], 1.0, places=4
        )
        self.assertEqual(diag["router_rollout_resample_count"], 64.0)

    def test_total_collapse_yields_one_hot_p_bar(self) -> None:
        K = 4
        logits = torch.zeros((32, K), dtype=torch.float32)
        logits[:, 1] = 50.0  # everyone picks z=1
        diag = rollout_router_soft_diagnostics(logits, latent_k=K)
        self.assertAlmostEqual(diag["router_rollout_soft_p_bar_z1"], 1.0, places=4)
        self.assertLess(diag["router_rollout_soft_marginal_entropy_nats"], 1e-4)
        self.assertLess(diag["router_rollout_soft_conditional_entropy_nats"], 1e-4)
        self.assertAlmostEqual(
            diag["router_rollout_soft_mi_proxy_nats"], 0.0, places=4
        )
        self.assertAlmostEqual(
            diag["router_rollout_soft_argmax_occupancy_max"], 1.0, places=4
        )
        self.assertAlmostEqual(
            diag["router_rollout_soft_argmax_occupancy_min"], 0.0, places=4
        )
        # min == 0 -> ratio is the finite cap, not inf.
        self.assertGreater(diag["router_rollout_soft_argmax_occupancy_ratio"], 0.0)
        self.assertTrue(math.isfinite(diag["router_rollout_soft_argmax_occupancy_ratio"]))

    def test_uniform_logits_yield_log_k_marginal_and_log_k_conditional(self) -> None:
        K = 4
        logits = torch.zeros((32, K), dtype=torch.float32)
        diag = rollout_router_soft_diagnostics(logits, latent_k=K)
        self.assertAlmostEqual(
            diag["router_rollout_soft_marginal_entropy_nats"], math.log(K), places=4
        )
        self.assertAlmostEqual(
            diag["router_rollout_soft_conditional_entropy_nats"], math.log(K), places=4
        )
        self.assertAlmostEqual(
            diag["router_rollout_soft_mi_proxy_nats"], 0.0, places=4
        )

    def test_empty_input_returns_zeros(self) -> None:
        logits = torch.zeros((0, 4), dtype=torch.float32)
        diag = rollout_router_soft_diagnostics(logits, latent_k=4)
        self.assertEqual(diag["router_rollout_soft_marginal_entropy_nats"], 0.0)
        self.assertEqual(diag["router_rollout_soft_conditional_entropy_nats"], 0.0)
        self.assertEqual(diag["router_rollout_soft_mi_proxy_nats"], 0.0)
        self.assertEqual(diag["router_rollout_resample_count"], 0.0)
        for z in range(4):
            self.assertEqual(diag[f"router_rollout_soft_p_bar_z{z}"], 0.0)


class StrategyPersistenceLossTests(unittest.TestCase):
    def test_matches_expected_strategy_switch_penalty_mean(self) -> None:
        _set_seed(0)
        logits = torch.randn(8, 4)
        prev_z = torch.randint(0, 4, (8,))
        mask = torch.tensor([True, False, True, True, False, False, True, True])
        loss, stats = strategy_persistence_loss(
            logits, prev_z, mask, lam_p=0.025, device=DEVICE
        )
        ref_switch = expected_strategy_switch_penalty(logits, prev_z)
        ref_persist = ref_switch[mask].mean()
        ref_loss = 0.025 * ref_persist
        self.assertTrue(_exact_eq(loss, ref_loss))
        self.assertAlmostEqual(stats["persist_term"], float(ref_persist))

    def test_empty_mask_returns_zero_tensor(self) -> None:
        logits = torch.randn(4, 4)
        prev_z = torch.randint(0, 4, (4,))
        mask = torch.zeros(4, dtype=torch.bool)
        loss, stats = strategy_persistence_loss(logits, prev_z, mask, lam_p=0.5, device=DEVICE)
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(stats["persist_term"], 0.0)

    def test_zero_lam_p_still_emits_persist_stat(self) -> None:
        _set_seed(1)
        logits = torch.randn(3, 4)
        prev_z = torch.tensor([0, 1, 2])
        mask = torch.tensor([True, True, True])
        loss, stats = strategy_persistence_loss(logits, prev_z, mask, lam_p=0.0, device=DEVICE)
        self.assertEqual(loss.item(), 0.0)
        ref_persist = expected_strategy_switch_penalty(logits, prev_z).mean()
        self.assertAlmostEqual(stats["persist_term"], float(ref_persist))


class StrategyKLConsecutiveLossTests(unittest.TestCase):
    def test_matches_masked_kl_mean(self) -> None:
        _set_seed(0)
        z_logits = torch.randn(6, 4)
        z_logits_prev = torch.randn(6, 4)
        valid = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        loss, stats = strategy_kl_consecutive_loss(z_logits, z_logits_prev, valid, coef=0.3)
        log_p = F.log_softmax(z_logits, dim=-1)
        log_q = F.log_softmax(z_logits_prev.detach(), dim=-1)
        p = log_p.exp()
        kl = (p * (log_p - log_q)).sum(-1)
        denom = valid.sum().clamp_min(1.0)
        kl_m = (kl * valid).sum() / denom
        expected = 0.3 * kl_m
        self.assertTrue(_exact_eq(loss, expected))
        self.assertAlmostEqual(stats["kl_mean"], float(kl_m))

    def test_zero_coef_short_circuits(self) -> None:
        z = torch.randn(2, 3, requires_grad=True)
        zp = torch.randn(2, 3, requires_grad=True)
        v = torch.tensor([1.0, 1.0])
        loss, stats = strategy_kl_consecutive_loss(z, zp, v, coef=0.0)
        self.assertEqual(loss.item(), 0.0)
        self.assertFalse(loss.requires_grad)
        self.assertEqual(stats["kl_mean"], 0.0)


class StrategyPhaseAuxLossTests(unittest.TestCase):
    def test_matches_cross_entropy_times_coef(self) -> None:
        _set_seed(0)
        logits = torch.randn(8, 5)
        target = torch.randint(0, 5, (8,))
        loss, stats = strategy_phase_aux_loss(logits, target, coef=0.2)
        ref = 0.2 * F.cross_entropy(logits, target.long())
        self.assertTrue(_exact_eq(loss, ref))
        self.assertAlmostEqual(stats["phase_term"], float(F.cross_entropy(logits, target.long())))

    def test_zero_coef_short_circuits(self) -> None:
        logits = torch.randn(4, 3, requires_grad=True)
        target = torch.tensor([0, 1, 2, 0])
        loss, _ = strategy_phase_aux_loss(logits, target, coef=0.0)
        self.assertEqual(loss.item(), 0.0)
        self.assertFalse(loss.requires_grad)


class StrategyPPOLossTests(unittest.TestCase):
    def test_matches_ppo_policy_loss_on_resample_subset(self) -> None:
        _set_seed(0)
        logp = torch.randn(10).requires_grad_(True)
        logp_old = torch.randn(10)
        advantages = torch.randn(10)
        resample = torch.tensor([True, False, True, False, True, True, False, True, False, True])
        loss, stats = strategy_ppo_loss(
            logp, logp_old, advantages, resample,
            clip_range=0.2, coef=0.5, device=DEVICE,
        )
        sub_adv = advantages[resample].detach()
        sub_adv = (sub_adv - sub_adv.mean()) / (sub_adv.std(unbiased=False) + 1e-8)
        ref_pol, ref_stats = ppo_policy_loss(logp[resample], logp_old[resample], sub_adv, 0.2)
        self.assertTrue(_exact_eq(loss, 0.5 * ref_pol))
        self.assertTrue(_exact_eq(stats["policy_loss"], ref_pol))
        self.assertTrue(_exact_eq(stats["approx_kl"], ref_stats["approx_kl"]))
        self.assertTrue(_exact_eq(stats["clip_fraction"], ref_stats["clip_fraction"]))

    def test_empty_resample_returns_zero_loss_and_neutral_stats(self) -> None:
        logp = torch.randn(4)
        logp_old = torch.randn(4)
        advantages = torch.randn(4)
        resample = torch.zeros(4, dtype=torch.bool)
        loss, stats = strategy_ppo_loss(
            logp, logp_old, advantages, resample,
            clip_range=0.2, coef=0.5, device=DEVICE,
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(stats["approx_kl"].item(), 0.0)
        self.assertEqual(stats["clip_fraction"].item(), 0.0)
        self.assertEqual(stats["ratio"].numel(), 1)
        self.assertEqual(stats["ratio"].item(), 1.0)
        self.assertEqual(stats["policy_loss"].item(), 0.0)

    def test_single_resample_row_skips_normalization(self) -> None:
        """When only one row is resampled, the std-normalization branch is skipped.

        We pin this behavior by computing the reference WITHOUT the
        ``(adv - mean) / std`` rescaling and verifying the helper matches.
        """
        _set_seed(2)
        logp = torch.randn(5)
        logp_old = torch.randn(5)
        advantages = torch.randn(5)
        resample = torch.tensor([False, True, False, False, False])
        loss, stats = strategy_ppo_loss(
            logp, logp_old, advantages, resample,
            clip_range=0.2, coef=1.0, device=DEVICE,
        )
        ref_pol, _ = ppo_policy_loss(
            logp[resample], logp_old[resample], advantages[resample].detach(), 0.2
        )
        self.assertTrue(_exact_eq(loss, 1.0 * ref_pol))
        self.assertTrue(_exact_eq(stats["policy_loss"], ref_pol))


class StrategyAuxReturnLossTests(unittest.TestCase):
    def test_matches_mse_on_gathered_predictions(self) -> None:
        _set_seed(0)
        B, K = 8, 4
        pred_all = torch.randn(B, K)
        z = torch.randint(0, K, (B,))
        resample = torch.tensor([True, False, True, True, False, True, False, False])
        # Caller pre-masks the returns tensor.
        returns_normalized = torch.randn(int(resample.sum()))
        loss, stats = strategy_aux_return_loss(
            pred_all, z, returns_normalized, resample,
            latent_k=K, coef=0.7, device=DEVICE,
        )
        z_sel = z[resample].long().clamp(min=0, max=K - 1)
        pred_selected = pred_all[resample].gather(1, z_sel.reshape(-1, 1)).squeeze(1)
        mse = F.mse_loss(pred_selected, returns_normalized)
        self.assertTrue(_exact_eq(loss, 0.7 * mse))
        self.assertAlmostEqual(stats["aux_return_term"], float(mse))

    def test_zero_coef_short_circuits(self) -> None:
        pred_all = torch.randn(3, 4, requires_grad=True)
        z = torch.tensor([0, 1, 2])
        resample = torch.tensor([True, True, True])
        ret = torch.randn(3)
        loss, _ = strategy_aux_return_loss(
            pred_all, z, ret, resample, latent_k=4, coef=0.0, device=DEVICE,
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertFalse(loss.requires_grad)

    def test_empty_resample_returns_zero(self) -> None:
        pred_all = torch.randn(3, 4)
        z = torch.tensor([0, 1, 2])
        resample = torch.zeros(3, dtype=torch.bool)
        ret = torch.empty(0)
        loss, stats = strategy_aux_return_loss(
            pred_all, z, ret, resample, latent_k=4, coef=1.0, device=DEVICE,
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(stats["aux_return_term"], 0.0)


class LatentLossCompositionTests(unittest.TestCase):
    """End-to-end parity: hand-compose all the loss terms in the same order as
    the trainer and verify the helpers sum to the same scalar as the legacy
    inline math.
    """

    def test_full_composition_matches_inline_legacy(self) -> None:
        _set_seed(7)
        B, K = 6, 4
        logits = torch.randn(B, K, requires_grad=True)
        logits_prev = torch.randn(B, K, requires_grad=True)
        h = torch.rand(B) * 1.3  # strategy_entropy per row
        prev_z = torch.randint(0, K, (B,))
        z = torch.randint(0, K, (B,))
        resample = torch.tensor([True, False, True, False, True, False])
        persist_mask = torch.tensor([False, True, True, False, False, True])
        valid = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        phase_logits = torch.randn(B, 5)
        phase_target = torch.randint(0, 5, (B,))
        log_prob = torch.randn(B, requires_grad=True)
        log_prob_old = torch.randn(B)
        advantages = torch.randn(B)
        pred_all = torch.randn(B, K)
        ret_target = torch.randn(int(resample.sum()))

        # --- helper composition (matches the trainer's order) ---
        ent_loss, _ = strategy_entropy_loss(
            h, resample, objective="maximize", lam_h=0.003, device=DEVICE,
        )
        persist_loss, _ = strategy_persistence_loss(
            logits, prev_z, persist_mask, lam_p=0.025, device=DEVICE,
        )
        composed = persist_loss + ent_loss
        kl_loss, _ = strategy_kl_consecutive_loss(logits, logits_prev, valid, coef=0.1)
        composed = composed + kl_loss
        phase_loss, _ = strategy_phase_aux_loss(phase_logits, phase_target, coef=0.05)
        composed = composed + phase_loss
        ppo_loss, ppo_stats = strategy_ppo_loss(
            log_prob, log_prob_old, advantages, resample,
            clip_range=0.18, coef=0.3, device=DEVICE,
        )
        composed = composed + ppo_loss
        aux_loss, _ = strategy_aux_return_loss(
            pred_all, z, ret_target, resample,
            latent_k=K, coef=1.0, device=DEVICE,
        )
        composed = composed + aux_loss

        # --- legacy inline reference ---
        h_mean = h[resample].mean()
        ent_ref = -0.003 * h_mean
        switch = expected_strategy_switch_penalty(logits, prev_z)
        persist_ref = 0.025 * switch[persist_mask].mean()
        legacy = persist_ref + ent_ref
        log_p = F.log_softmax(logits, dim=-1)
        log_q = F.log_softmax(logits_prev.detach(), dim=-1)
        p = log_p.exp()
        kl = (p * (log_p - log_q)).sum(dim=-1)
        denom = valid.sum().clamp_min(1.0)
        kl_m = (kl * valid).sum() / denom
        legacy = legacy + 0.1 * kl_m
        phase_ce = F.cross_entropy(phase_logits, phase_target.long())
        legacy = legacy + 0.05 * phase_ce
        sub_adv = advantages[resample].detach()
        sub_adv = (sub_adv - sub_adv.mean()) / (sub_adv.std(unbiased=False) + 1e-8)
        ref_pol, _ = ppo_policy_loss(
            log_prob[resample], log_prob_old[resample], sub_adv, 0.18,
        )
        legacy = legacy + 0.3 * ref_pol
        z_sel = z[resample].long().clamp(min=0, max=K - 1)
        pred_selected = pred_all[resample].gather(1, z_sel.reshape(-1, 1)).squeeze(1)
        mse = F.mse_loss(pred_selected, ret_target)
        legacy = legacy + 1.0 * mse

        self.assertTrue(_exact_eq(composed, legacy),
                        f"composed={composed.item()}, legacy={legacy.item()}")


if __name__ == "__main__":
    unittest.main()
