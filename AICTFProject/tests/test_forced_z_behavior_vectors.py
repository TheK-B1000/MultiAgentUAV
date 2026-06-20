"""Unit tests for forced-z behavior vector telemetry."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from macro_actions import MacroAction
from rl.behavior_telemetry import BEHAVIOR_TELEMETRY_NAMES, N_TELEMETRY
from rl.forced_z_behavior_vectors import (
    CF_REGIME_COSMETIC,
    CF_REGIME_PRODUCTIVE,
    CF_REGIME_UNDERPOWERED,
    FORCED_Z_BEHAVIOR_VECTOR_NAMES,
    INTERVENTION_QUADRANT_COLLAPSE,
    INTERVENTION_QUADRANT_COSMETIC,
    INTERVENTION_QUADRANT_GENUINE,
    PhaseABehaviorTrendTracker,
    actor_pair_stats_from_update,
    behavior_vector_from_macro_probs,
    behavior_vector_from_telemetry_row,
    build_behavior_distance_profile,
    classify_cf_training_regime,
    classify_intervention_quadrant,
    component_scale_and_validity,
    normalize_behavior_vectors,
    opportunity_conditioned_z_returns,
    pairwise_behavior_distances,
    per_z_vector_telemetry,
    phase_a_diagnostic_telemetry,
    phase_a_stats_snapshot,
)


class BehaviorVectorMappingTests(unittest.TestCase):
    def test_telemetry_row_has_seven_dims(self) -> None:
        row = np.zeros((N_TELEMETRY,), dtype=np.float64)
        row[BEHAVIOR_TELEMETRY_NAMES.index("num_attackers")] = 3.0
        row[BEHAVIOR_TELEMETRY_NAMES.index("avg_blue_to_enemy_flag")] = 0.8
        vec = behavior_vector_from_telemetry_row(row)
        self.assertEqual(vec.shape, (len(FORCED_Z_BEHAVIOR_VECTOR_NAMES),))
        self.assertAlmostEqual(vec[0], 0.8)
        self.assertAlmostEqual(vec[5], 0.75)

    def test_normalization_bounds_to_unit_interval(self) -> None:
        raw = [np.array([2.0, -0.1, 0.5, 0.2, 0.1, 0.5, 0.5])]
        normed = normalize_behavior_vectors(raw, source="macro")
        self.assertTrue(np.all(normed[0] >= 0.0))
        self.assertTrue(np.all(normed[0] <= 1.0))

    def test_component_scale_and_validity_reported(self) -> None:
        z0 = torch.zeros(5)
        z0[int(MacroAction.GET_FLAG)] = 0.9
        raw = [behavior_vector_from_macro_probs(z0)]
        meta = component_scale_and_validity(raw, source="macro")
        self.assertIn("behavior_component_scale_attack_lane_preference", meta)
        self.assertEqual(meta["behavior_component_valid_attack_lane_preference"], 1.0)


class PairwiseDistanceTests(unittest.TestCase):
    def test_pairwise_keys_for_k4_normalized(self) -> None:
        raw = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ]
        normed = normalize_behavior_vectors(raw, source="macro")
        stats, agg = pairwise_behavior_distances(normed, pair_count=6, already_normalized=True, pair_threshold=0.1)
        self.assertEqual(len(agg), 6)
        self.assertIn("forced_z_behavior_pair_distance_min", stats)
        self.assertIn("forced_z_behavior_pairs_above_threshold", stats)
        self.assertGreater(stats["forced_z_behavior_pair_distance_min"], 0.0)

    def test_high_mean_can_fail_weakest_pair_gate(self) -> None:
        raw = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            np.array([0.5, 0.5, 0.5, 0.5, 0.5001, 0.5, 0.5]),
        ]
        profile = build_behavior_distance_profile(raw, source="macro", pair_count=6, latent_k=4, pair_threshold=0.9)
        self.assertGreater(profile["forced_z_behavior_pair_distance_mean"], 0.0)
        self.assertLess(profile["forced_z_behavior_pair_distance_min"], 0.1)


class InterventionQuadrantTests(unittest.TestCase):
    def test_collapse(self) -> None:
        code, label = classify_intervention_quadrant(0.0, 0.0)
        self.assertEqual(code, INTERVENTION_QUADRANT_COLLAPSE)
        self.assertEqual(label, "collapse")

    def test_cosmetic(self) -> None:
        code, label = classify_intervention_quadrant(0.5, 0.0)
        self.assertEqual(code, INTERVENTION_QUADRANT_COSMETIC)
        self.assertEqual(label, "cosmetic")

    def test_genuine_uses_weakest_pair_distance(self) -> None:
        code, _ = classify_intervention_quadrant(0.5, 0.06, behavior_distance_threshold=0.05)
        self.assertEqual(code, INTERVENTION_QUADRANT_GENUINE)


class CfRegimeTests(unittest.TestCase):
    def test_underpowered(self) -> None:
        code, label = classify_cf_training_regime(
            cf_to_ppo_ratio=0.001,
            competence_min=0.5,
            behavior_distance=0.1,
            behavior_slope=0.0,
            actor_slope=0.0,
            actor_jsd=0.0,
        )
        self.assertEqual(code, CF_REGIME_UNDERPOWERED)
        self.assertEqual(label, "underpowered")

    def test_productive_with_slopes(self) -> None:
        code, label = classify_cf_training_regime(
            cf_to_ppo_ratio=0.1,
            competence_min=0.5,
            behavior_distance=0.2,
            behavior_slope=0.01,
            actor_slope=0.01,
            actor_jsd=0.05,
        )
        self.assertEqual(code, CF_REGIME_PRODUCTIVE)
        self.assertEqual(label, "productive")

    def test_cosmetic(self) -> None:
        code, label = classify_cf_training_regime(
            cf_to_ppo_ratio=0.1,
            competence_min=0.5,
            behavior_distance=0.1,
            behavior_slope=0.0,
            actor_slope=0.0,
            actor_jsd=0.5,
        )
        self.assertEqual(code, CF_REGIME_COSMETIC)
        self.assertEqual(label, "cosmetic")


class TrendTrackerTests(unittest.TestCase):
    def test_slope_requires_window(self) -> None:
        tracker = PhaseABehaviorTrendTracker(window=5)
        for i in range(5):
            tracker.record(
                global_step=i,
                actor_jsd=0.01 * i,
                behavior_dist=0.02 * i,
                actor_valid=True,
                behavior_valid=True,
            )
        telem = tracker.telemetry()
        self.assertGreater(telem["phase_a_actor_jsd_slope_20"], 0.0)
        self.assertGreater(telem["phase_a_behavior_distance_slope_20"], 0.0)
        self.assertEqual(telem["phase_a_actor_jsd_valid_updates"], 5.0)


class PhaseADiagnosticTests(unittest.TestCase):
    def _base_stats(self) -> dict[str, float]:
        stats: dict[str, float] = {
            "latent_actor_z_separation_jsd": 0.05,
            "forced_z_behavior_pair_distance_mean": 0.5,
            "forced_z_behavior_pair_distance_min": 0.4,
            "forced_z_behavior_components_valid": 1.0,
            "forced_z_behavior_all_z_represented": 1.0,
            "phase_a_behavior_pair_gate_pass": 1.0,
            "pairwise_profile_available": 1.0,
            "cf_competence_z0": 0.5,
            "cf_competence_z1": 0.5,
            "cf_competence_z2": 0.5,
            "cf_competence_z3": 0.5,
            "cf_to_ppo_grad_ratio": 0.1,
        }
        for i in range(6):
            stats[f"forced_z_pair_jsd_{i}"] = 0.02
        return stats

    def test_corridor_requires_pair_gates_and_slopes(self) -> None:
        tracker = PhaseABehaviorTrendTracker(window=5)
        out: dict[str, float] = {}
        for step in range(5):
            stats = self._base_stats()
            stats["latent_actor_z_separation_jsd"] = 0.01 + 0.01 * step
            stats["forced_z_behavior_pair_distance_mean"] = 0.2 + 0.05 * step
            stats["forced_z_behavior_pair_distance_min"] = 0.15 + 0.05 * step
            out = phase_a_diagnostic_telemetry(
                stats,
                trend_tracker=tracker,
                global_step=step,
            )
        self.assertEqual(out["phase_a_actor_pair_gate_pass"], 1.0)
        self.assertEqual(out["phase_a_corridor_viable"], 1.0)

    def test_snapshot_rejects_stale_stats(self) -> None:
        stats = phase_a_diagnostic_telemetry(self._base_stats(), global_step=100)
        snap = phase_a_stats_snapshot(stats, gate_step=500_000)
        self.assertEqual(snap["phase_a_stats_stale"], 1.0)
        self.assertEqual(snap["phase_a_snapshot_usable"], 0.0)
        self.assertEqual(snap["phase_a_intervention_quadrant_label"], "not_run")

    def test_snapshot_accepts_fresh_stats(self) -> None:
        stats = self._base_stats()
        stats.update(phase_a_diagnostic_telemetry(stats, global_step=1000))
        snap = phase_a_stats_snapshot(stats, gate_step=1000)
        self.assertEqual(snap["phase_a_snapshot_usable"], 1.0)

    def test_actor_pair_fields_copy_actor_gate_details(self) -> None:
        stats = self._base_stats()
        for i in range(6):
            stats[f"forced_z_pair_jsd_{i}"] = 0.00001
        gate_details = {
            "batch_pairs_above_margin": 5,
            "cf_pair_jsd_last_batch": [0.002, 0.002, 0.002, 0.002, 0.002, 0.0004],
            "single_update_ok": True,
        }
        out = phase_a_diagnostic_telemetry(
            stats,
            global_step=1000,
            actor_gate_details=gate_details,
            actor_jsd_margin=0.001,
        )
        self.assertEqual(out["phase_a_actor_pairs_above_margin"], 5.0)
        self.assertEqual(out["phase_a_actor_weakest_pair_jsd"], 0.0004)
        self.assertEqual(out["phase_a_actor_pair_gate_pass"], 1.0)


class ActorPairStatsTests(unittest.TestCase):
    def test_actor_pair_gate_counts(self) -> None:
        stats = {f"cf_batch_pair_jsd_{i}": 0.02 for i in range(6)}
        stats["cf_batch_evidence_valid"] = 1.0
        out = actor_pair_stats_from_update(stats, margin=0.01)
        self.assertEqual(out["phase_a_actor_pairs_above_margin"], 6.0)
        self.assertEqual(out["phase_a_actor_pair_gate_pass"], 1.0)

    def test_actor_pair_gate_uses_cf_pairs_when_validity_flag_missing(self) -> None:
        stats = {f"cf_batch_pair_jsd_{i}": 0.02 for i in range(6)}
        stats.update({f"forced_z_pair_jsd_{i}": 0.00001 for i in range(6)})
        out = actor_pair_stats_from_update(stats, margin=0.01)
        self.assertEqual(out["phase_a_actor_pairs_above_margin"], 6.0)
        self.assertEqual(out["phase_a_actor_pair_gate_pass"], 1.0)
        self.assertEqual(out["phase_a_actor_weakest_pair_jsd"], 0.02)

    def test_actor_pair_gate_uses_actor_margin_scale(self) -> None:
        stats = {f"cf_batch_pair_jsd_{i}": 0.002 for i in range(6)}
        out = actor_pair_stats_from_update(stats, margin=0.001)
        self.assertEqual(out["phase_a_actor_pairs_above_margin"], 6.0)
        self.assertEqual(out["phase_a_actor_pair_gate_pass"], 1.0)
        self.assertEqual(out["phase_a_actor_weakest_pair_jsd"], 0.002)

    def test_actor_pair_gate_falls_back_to_forced_z_when_cf_pairs_absent(self) -> None:
        stats = {f"forced_z_pair_jsd_{i}": 0.02 for i in range(6)}
        out = actor_pair_stats_from_update(stats, margin=0.01)
        self.assertEqual(out["phase_a_actor_pairs_above_margin"], 6.0)
        self.assertEqual(out["phase_a_actor_pair_gate_pass"], 1.0)


class OpportunityForkTests(unittest.TestCase):
    def _make_buffer(self, *, repeats: int = 3) -> object:
        class _Buf:
            fields: dict[str, torch.Tensor]

        rows: list[tuple[float, int, int]] = []
        for _ in range(repeats):
            for zk in range(4):
                rows.append((float(zk), zk, 0))
        rets, zs, pbs = zip(*rows)
        buf = _Buf()
        buf.pos = len(rets)
        buf.fields = {
            "returns": torch.tensor(rets, dtype=torch.float32).reshape(-1, 1),
            "z": torch.tensor(zs, dtype=torch.long).reshape(-1, 1),
            "opponent_id": torch.zeros(len(rets), 1, dtype=torch.long),
            "pressure_bucket_id": torch.tensor(pbs, dtype=torch.long).reshape(-1, 1),
            "role_bucket_id": torch.zeros(len(rets), 1, dtype=torch.long),
            "spread_bucket_id": torch.zeros(len(rets), 1, dtype=torch.long),
            "phase_id": torch.zeros(len(rets), 1, dtype=torch.long),
            "blue_ahead": torch.zeros(len(rets), 1),
        }
        return buf

    def test_valid_fork_requires_support_and_margin(self) -> None:
        out = opportunity_conditioned_z_returns(self._make_buffer(), latent_k=4, min_samples_per_z=3)
        self.assertGreater(out["opportunity_eligible_cell_count"], 0.0)
        self.assertIn("opportunity_cell_0_count_z0", out)
        self.assertIn("opportunity_cell_0_best_margin", out)
        self.assertGreaterEqual(out["opportunity_fork_fraction_valid"], 0.0)

    def test_sparse_cells_not_counted_as_eligible(self) -> None:
        class _Buf:
            pos = 4
            fields: dict[str, torch.Tensor]

        buf = _Buf()
        buf.fields = {
            "returns": torch.tensor([0.0, 1.0, 1.0, 0.0]).reshape(4, 1),
            "z": torch.tensor([0, 1, 0, 1]).reshape(4, 1),
            "opponent_id": torch.zeros(4, 1, dtype=torch.long),
            "pressure_bucket_id": torch.tensor([0, 0, 1, 1]).reshape(4, 1),
            "role_bucket_id": torch.zeros(4, 1, dtype=torch.long),
            "spread_bucket_id": torch.zeros(4, 1, dtype=torch.long),
            "phase_id": torch.zeros(4, 1, dtype=torch.long),
            "blue_ahead": torch.zeros(4, 1),
        }
        out = opportunity_conditioned_z_returns(buf, latent_k=4, min_samples_per_z=3)
        self.assertEqual(out["opportunity_eligible_cell_count"], 0.0)


if __name__ == "__main__":
    unittest.main()
