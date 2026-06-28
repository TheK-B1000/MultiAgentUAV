"""Phase 7: Latent Diagnostics Refactor — 26 targeted tests.

Coverage:
    Group A (1-7):   DiagnosticStatus, DiagnosticResult, DiagnosticError
    Group B (8-13):  entropy.py — pure math: Shannon entropy, MI
    Group C (14-15): entropy.py — bucket_z_fracs, fill_zero_z_fracs
    Group D (16-17): switching.py — flag_return_indices
    Group E (18-20): counterfactual.py — jsd_from_logits properties
    Group F (21):    occupancy.py — compute_occupancy_stats
    Group G (22-23): competence.py — critic variance, adapter grad norms
    Group H (24-25): validation.py — typed validators
    Group I (26):    Facade equivalence — latent_diagnostics re-exports same objects
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Group A: DiagnosticStatus, DiagnosticResult, DiagnosticError
# ---------------------------------------------------------------------------


class TestDiagnosticStatus(unittest.TestCase):
    def test_status_all_five_values(self):
        from rl.custom_ppo.diagnostics.results import DiagnosticStatus

        self.assertEqual(DiagnosticStatus.PASS, "pass")
        self.assertEqual(DiagnosticStatus.FAIL, "fail")
        self.assertEqual(DiagnosticStatus.WARN, "warn")
        self.assertEqual(DiagnosticStatus.INCONCLUSIVE, "inconclusive")
        self.assertEqual(DiagnosticStatus.ERROR, "error")
        self.assertEqual(len(DiagnosticStatus), 5)

    def test_result_is_pass_is_fail_properties(self):
        from rl.custom_ppo.diagnostics.results import DiagnosticResult, DiagnosticStatus

        r_pass = DiagnosticResult(status=DiagnosticStatus.PASS, value=1.0, sample_count=100)
        self.assertTrue(r_pass.is_pass)
        self.assertFalse(r_pass.is_fail)
        self.assertTrue(r_pass.is_available)

        r_fail = DiagnosticResult(status=DiagnosticStatus.FAIL, value=0.001, sample_count=50)
        self.assertFalse(r_fail.is_pass)
        self.assertTrue(r_fail.is_fail)
        self.assertTrue(r_fail.is_available)

    def test_result_is_frozen(self):
        from rl.custom_ppo.diagnostics.results import DiagnosticResult, DiagnosticStatus
        import dataclasses

        r = DiagnosticResult(status=DiagnosticStatus.PASS, value=3.14, sample_count=10)
        with self.assertRaises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            r.value = 0.0  # type: ignore[misc]

    def test_result_map_transforms_value(self):
        from rl.custom_ppo.diagnostics.results import DiagnosticResult, DiagnosticStatus

        original = DiagnosticResult(
            status=DiagnosticStatus.WARN,
            value=0.5,
            sample_count=64,
            reason="some reason",
        )
        doubled = original.map(lambda x: x * 2)
        self.assertAlmostEqual(doubled.value, 1.0)
        self.assertEqual(doubled.status, DiagnosticStatus.WARN)
        self.assertEqual(doubled.sample_count, 64)
        self.assertEqual(doubled.reason, "some reason")

    def test_result_unavailable_classmethod(self):
        from rl.custom_ppo.diagnostics.results import DiagnosticResult, DiagnosticStatus

        r = DiagnosticResult.unavailable("no data")
        self.assertEqual(r.status, DiagnosticStatus.INCONCLUSIVE)
        self.assertIsNone(r.value)
        self.assertEqual(r.sample_count, 0)
        self.assertFalse(r.is_available)

    def test_result_from_error_populates_error_field(self):
        from rl.custom_ppo.diagnostics.results import DiagnosticResult, DiagnosticStatus

        exc = ValueError("something went wrong")
        r = DiagnosticResult.from_error(exc)
        self.assertEqual(r.status, DiagnosticStatus.ERROR)
        self.assertIsNone(r.value)
        self.assertIsNotNone(r.error)
        self.assertIn("something went wrong", r.error.message)
        self.assertEqual(r.error.exc_type, "ValueError")

    def test_result_is_available_property(self):
        from rl.custom_ppo.diagnostics.results import DiagnosticResult, DiagnosticStatus

        # sample_count == 0 → not available
        r_zero = DiagnosticResult(status=DiagnosticStatus.PASS, value=1.0, sample_count=0)
        self.assertFalse(r_zero.is_available)

        # ERROR status → not available
        r_err = DiagnosticResult.from_error(RuntimeError("err"))
        self.assertFalse(r_err.is_available)

        # Normal WARN with count > 0 → available
        r_warn = DiagnosticResult(status=DiagnosticStatus.WARN, value=0.001, sample_count=1)
        self.assertTrue(r_warn.is_available)


# ---------------------------------------------------------------------------
# Group B: entropy.py — Shannon entropy and MI
# ---------------------------------------------------------------------------


class TestShannonEntropy(unittest.TestCase):
    def test_shannon_entropy_uniform_golden_value(self):
        """Uniform distribution over K categories has entropy = ln(K)."""
        from rl.custom_ppo.diagnostics.entropy import _shannon_entropy_nats

        arr = np.array([0, 1, 2, 3] * 25)  # 100 samples, uniform over 4 cats
        result = _shannon_entropy_nats(arr, 4)
        expected = math.log(4)  # ≈ 1.3862943611198906
        self.assertAlmostEqual(result, expected, places=10)

    def test_shannon_entropy_deterministic_is_zero(self):
        """One-hot distribution has entropy = 0."""
        from rl.custom_ppo.diagnostics.entropy import _shannon_entropy_nats

        arr = np.zeros(50, dtype=np.int64)
        result = _shannon_entropy_nats(arr, 4)
        self.assertAlmostEqual(result, 0.0, places=12)

    def test_shannon_entropy_none_returns_zero(self):
        from rl.custom_ppo.diagnostics.entropy import _shannon_entropy_nats

        self.assertEqual(_shannon_entropy_nats(None, 4), 0.0)

    def test_shannon_entropy_empty_array_returns_zero(self):
        from rl.custom_ppo.diagnostics.entropy import _shannon_entropy_nats

        self.assertEqual(_shannon_entropy_nats(np.array([], dtype=np.int64), 4), 0.0)

    def test_mi_z_vs_perfectly_correlated_equals_entropy(self):
        """When z == x deterministically, MI(z; x) = H(z)."""
        from rl.custom_ppo.diagnostics.entropy import _mi_z_vs, _shannon_entropy_nats

        rng = np.random.default_rng(42)
        K = 4
        z = rng.integers(0, K, size=200)
        x = z.copy()  # perfect correlation

        mi = _mi_z_vs(z, K, x, K)
        h_z = _shannon_entropy_nats(z.astype(np.int64), K)
        self.assertAlmostEqual(mi, h_z, places=10)

    def test_mi_z_vs_independent_near_zero(self):
        """MI(z; constant_x) = 0 because x carries no information about z."""
        from rl.custom_ppo.diagnostics.entropy import _mi_z_vs

        rng = np.random.default_rng(7)
        z = rng.integers(0, 4, size=500)
        x = np.zeros(500, dtype=np.int64)  # constant x → no information

        mi = _mi_z_vs(z, 4, x, 1)
        self.assertAlmostEqual(mi, 0.0, places=10)


# ---------------------------------------------------------------------------
# Group C: bucket_z_fracs and fill_zero_z_fracs
# ---------------------------------------------------------------------------


class TestBucketZFracs(unittest.TestCase):
    def test_bucket_z_fracs_sums_to_one_per_bucket(self):
        from rl.custom_ppo.diagnostics.entropy import _bucket_z_fracs

        K = 3
        z = np.array([0, 1, 2, 0, 1, 0])  # bucket 0 has z=[0,0,0] wait...
        # bucket 0: positions 0,3,5 → z=[0,0,0] → P(z=0|b=0)=1
        # bucket 1: positions 1,4   → z=[1,1]   → P(z=1|b=1)=1
        # bucket 2: position 2      → z=[2]      → P(z=2|b=2)=1
        bucket = np.array([0, 1, 2, 0, 1, 0])
        out: dict[str, float] = {}
        _bucket_z_fracs(out, z, K, bucket, 3, lambda b, k: f"b{b}_z{k}")

        for b in range(3):
            total = sum(out[f"b{b}_z{k}"] for k in range(K))
            self.assertAlmostEqual(total, 1.0, places=10,
                                   msg=f"bucket {b} fracs should sum to 1")

    def test_fill_zero_z_fracs_produces_all_zeros(self):
        from rl.custom_ppo.diagnostics.entropy import _fill_zero_z_fracs

        out: dict[str, float] = {}
        _fill_zero_z_fracs(out, K=4, n_buckets=3, key=lambda b, k: f"b{b}_z{k}")
        for b in range(3):
            for k in range(4):
                self.assertEqual(out[f"b{b}_z{k}"], 0.0)


# ---------------------------------------------------------------------------
# Group D: switching.py
# ---------------------------------------------------------------------------


class TestFlagReturnIndices(unittest.TestCase):
    def test_flag_return_indices_detects_carrier_drop(self):
        """Returns index where blue flag carrier dropped the flag without scoring."""
        from rl.custom_ppo.diagnostics.switching import _flag_return_indices

        # t=0: blue carries; t=1: blue drops without scoring; t=2: neutral
        blue_cap = np.array([True, False, False])
        red_cap = np.array([False, False, False])
        abs_rsp = np.array([0.0, 0.0, 0.0])  # no score

        idx = _flag_return_indices(blue_cap, red_cap, abs_rsp)
        self.assertIn(1, idx)

    def test_flag_return_indices_empty_for_short_sequence(self):
        from rl.custom_ppo.diagnostics.switching import _flag_return_indices

        blue_cap = np.array([True])
        red_cap = np.array([False])
        abs_rsp = np.array([0.0])

        idx = _flag_return_indices(blue_cap, red_cap, abs_rsp)
        self.assertEqual(len(idx), 0)


# ---------------------------------------------------------------------------
# Group E: counterfactual.py — JSD properties
# ---------------------------------------------------------------------------


class TestJsdFromLogits(unittest.TestCase):
    def test_jsd_from_logits_self_is_zero(self):
        """JSD(p, p) = 0 for any distribution."""
        from rl.custom_ppo.diagnostics.counterfactual import _jsd_from_logits

        torch.manual_seed(0)
        logits = torch.randn(10, 5)
        jsd = _jsd_from_logits(logits, logits)
        self.assertTrue((jsd.abs() < 1e-6).all(),
                        msg=f"JSD(p, p) should be 0, got max={jsd.abs().max().item():.2e}")

    def test_jsd_from_logits_symmetric(self):
        """JSD(p, q) == JSD(q, p)."""
        from rl.custom_ppo.diagnostics.counterfactual import _jsd_from_logits

        torch.manual_seed(1)
        a = torch.randn(8, 4)
        b = torch.randn(8, 4)
        jsd_ab = _jsd_from_logits(a, b)
        jsd_ba = _jsd_from_logits(b, a)
        self.assertTrue(torch.allclose(jsd_ab, jsd_ba, atol=1e-6),
                        msg="JSD must be symmetric")

    def test_jsd_from_logits_nonnegative(self):
        """JSD >= 0 for all inputs."""
        from rl.custom_ppo.diagnostics.counterfactual import _jsd_from_logits

        torch.manual_seed(2)
        for _ in range(5):
            a = torch.randn(16, 6)
            b = torch.randn(16, 6)
            jsd = _jsd_from_logits(a, b)
            self.assertTrue((jsd >= -1e-7).all(),
                            msg=f"JSD should be >= 0, got min={jsd.min().item():.2e}")


# ---------------------------------------------------------------------------
# Group F: occupancy.py
# ---------------------------------------------------------------------------


class TestComputeOccupancyStats(unittest.TestCase):
    def test_compute_occupancy_stats_uniform(self):
        """Uniform counts → entropy = ln(K), effective_num_latents = K, ratio = 1."""
        from rl.custom_ppo.diagnostics.occupancy import compute_occupancy_stats

        counts = torch.ones(4)
        stats = compute_occupancy_stats(counts, 4)

        self.assertAlmostEqual(float(stats["latent_marginal_entropy_nats"]), math.log(4), places=8)
        self.assertAlmostEqual(float(stats["effective_num_latents"]), 4.0, places=6)
        self.assertAlmostEqual(float(stats["latent_occupancy_ratio"]), 1.0, places=8)
        self.assertAlmostEqual(float(stats["latent_occupancy_min"]), 0.25, places=8)
        self.assertAlmostEqual(float(stats["latent_occupancy_max"]), 0.25, places=8)
        for k in range(4):
            self.assertAlmostEqual(float(stats[f"strategy_occupancy_{k}"]), 0.25, places=8)

    def test_compute_occupancy_stats_collapse(self):
        """Single occupied latent → entropy = 0, effective = 1, ratio is large."""
        from rl.custom_ppo.diagnostics.occupancy import compute_occupancy_stats

        counts = torch.tensor([100.0, 0.0, 0.0, 0.0])
        stats = compute_occupancy_stats(counts, 4)

        self.assertAlmostEqual(float(stats["latent_marginal_entropy_nats"]), 0.0, places=8)
        self.assertAlmostEqual(float(stats["effective_num_latents"]), 1.0, places=6)
        self.assertAlmostEqual(float(stats["latent_occupancy_min"]), 0.0, places=8)
        self.assertAlmostEqual(float(stats["latent_occupancy_max"]), 1.0, places=8)


# ---------------------------------------------------------------------------
# Group G: competence.py
# ---------------------------------------------------------------------------


class TestComputeCriticValueVariance(unittest.TestCase):
    def test_critic_value_variance_nan_when_k_less_than_2(self):
        """K < 2 → 'critic_value_var_z' is NaN (not enough latents to compare)."""
        from rl.custom_ppo.diagnostics.competence import compute_critic_value_variance

        class MockModel:
            latent_k = 1
            uses_latent_strategy = True

            def values(self, gs, z_idx=None):
                return torch.zeros(gs.shape[0])

        result = compute_critic_value_variance(MockModel(), torch.rand(5, 10))
        self.assertTrue(math.isnan(result["critic_value_var_z"]),
                        msg="K=1 should yield NaN")

    def test_adapter_grad_norms_empty_without_residual(self):
        """Returns {} when latent_actor is None or enable_latent_z_residual is False."""
        from rl.custom_ppo.diagnostics.competence import compute_adapter_grad_norms

        class MockModelNoActor:
            latent_actor = None

        class MockModelNoResidual:
            class la:
                enable_latent_z_residual = False
            latent_actor = la

        self.assertEqual(compute_adapter_grad_norms(MockModelNoActor()), {})
        self.assertEqual(compute_adapter_grad_norms(MockModelNoResidual()), {})


# ---------------------------------------------------------------------------
# Group H: validation.py
# ---------------------------------------------------------------------------


class TestValidators(unittest.TestCase):
    def test_validate_occupancy_entropy_pass_and_fail(self):
        from rl.custom_ppo.diagnostics.validation import validate_occupancy_entropy
        from rl.custom_ppo.diagnostics.results import DiagnosticStatus

        high_entropy = {
            "latent_marginal_entropy_nats": 1.2,
            "strategy_unique_count": 4,
        }
        result_pass = validate_occupancy_entropy(high_entropy, min_entropy_nats=0.5)
        self.assertEqual(result_pass.status, DiagnosticStatus.PASS)
        self.assertTrue(result_pass.is_pass)
        self.assertAlmostEqual(result_pass.value, 1.2)

        low_entropy = {
            "latent_marginal_entropy_nats": 0.05,
            "strategy_unique_count": 1,
        }
        result_fail = validate_occupancy_entropy(low_entropy, min_entropy_nats=0.5)
        self.assertEqual(result_fail.status, DiagnosticStatus.FAIL)
        self.assertTrue(result_fail.is_fail)

    def test_validate_jsd_separation_pass(self):
        from rl.custom_ppo.diagnostics.validation import validate_jsd_separation
        from rl.custom_ppo.diagnostics.results import DiagnosticStatus

        stats = {"actor_z_jsd_mean": 0.05, "actor_z_pairs_total": 6}
        result = validate_jsd_separation(stats, min_jsd=0.001)
        self.assertEqual(result.status, DiagnosticStatus.PASS)
        self.assertAlmostEqual(result.value, 0.05)
        self.assertEqual(result.sample_count, 6)

    def test_validate_unique_latents_fail(self):
        from rl.custom_ppo.diagnostics.validation import validate_unique_latents
        from rl.custom_ppo.diagnostics.results import DiagnosticStatus

        stats = {"strategy_unique_count": 1}
        result = validate_unique_latents(stats, min_unique=2)
        self.assertEqual(result.status, DiagnosticStatus.FAIL)
        self.assertEqual(result.value, 1)

    def test_validate_mi_proxy_unavailable_when_key_missing(self):
        from rl.custom_ppo.diagnostics.validation import validate_mi_proxy
        from rl.custom_ppo.diagnostics.results import DiagnosticStatus

        result = validate_mi_proxy({})  # key not present
        self.assertEqual(result.status, DiagnosticStatus.INCONCLUSIVE)
        self.assertFalse(result.is_available)


# ---------------------------------------------------------------------------
# Group I: Facade equivalence
# ---------------------------------------------------------------------------


class TestFacadeEquivalence(unittest.TestCase):
    def test_legacy_facade_imports_are_same_function_objects(self):
        """latent_diagnostics.py re-exports the identical function objects from diagnostics.*

        Equivalence is proven by identity (``is``), not just value equality.
        This guarantees that the facade is a pure re-export with no logic duplication.
        """
        import rl.custom_ppo.latent_diagnostics as facade
        import rl.custom_ppo.diagnostics.entropy as entropy_mod
        import rl.custom_ppo.diagnostics.counterfactual as cf_mod
        import rl.custom_ppo.diagnostics.competence as comp_mod
        import rl.custom_ppo.diagnostics.aggregation as agg_mod
        import rl.custom_ppo.diagnostics.specialization as spec_mod
        import rl.custom_ppo.diagnostics.switching as sw_mod

        pairs = [
            (facade._shannon_entropy_nats, entropy_mod._shannon_entropy_nats),
            (facade._mi_z_vs, entropy_mod._mi_z_vs),
            (facade._bucket_z_fracs, entropy_mod._bucket_z_fracs),
            (facade._fill_zero_z_fracs, entropy_mod._fill_zero_z_fracs),
            (facade._jsd_from_logits, cf_mod._jsd_from_logits),
            (facade._policy_z_sensitivity_kl, cf_mod._policy_z_sensitivity_kl),
            (facade.compute_pairwise_actor_jsd, cf_mod.compute_pairwise_actor_jsd),
            (facade.compute_adapter_grad_norms, comp_mod.compute_adapter_grad_norms),
            (facade.compute_critic_value_variance, comp_mod.compute_critic_value_variance),
            (facade._v6i8_residual_adapter_stats, comp_mod._v6i8_residual_adapter_stats),
            (facade._latent_rollout_stats, agg_mod._latent_rollout_stats),
            (facade._latent_opponent_rollout_diag, agg_mod._latent_opponent_rollout_diag),
            (facade._behavior_diversity_stats, spec_mod._behavior_diversity_stats),
            (facade._flag_return_indices, sw_mod._flag_return_indices),
        ]
        for facade_fn, impl_fn in pairs:
            self.assertIs(facade_fn, impl_fn,
                          msg=f"Facade re-export of {facade_fn.__name__} "
                              f"must be the same object as {impl_fn.__module__}.{impl_fn.__name__}")


class TestPublicDiagnosticsPackageAPI(unittest.TestCase):
    def test_top_level_package_imports(self):
        """All public API names are importable from rl.custom_ppo.diagnostics."""
        from rl.custom_ppo.diagnostics import (
            DiagnosticStatus,
            DiagnosticResult,
            DiagnosticError,
            shannon_entropy_nats,
            mi_z_vs,
            jsd_from_logits,
            compute_pairwise_actor_jsd,
            compute_adapter_grad_norms,
            compute_critic_value_variance,
            validate_occupancy_entropy,
            validate_jsd_separation,
            validate_mi_proxy,
            validate_unique_latents,
            compute_occupancy_stats,
        )
        # Spot-check that they're callable
        for obj in [
            DiagnosticStatus, DiagnosticResult, DiagnosticError,
            shannon_entropy_nats, mi_z_vs, jsd_from_logits,
        ]:
            self.assertTrue(callable(obj) or isinstance(obj, type))


if __name__ == "__main__":
    unittest.main()
