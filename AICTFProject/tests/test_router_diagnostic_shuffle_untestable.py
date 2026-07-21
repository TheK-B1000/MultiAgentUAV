"""Pins the vacuous-shuffle labeling in the v6i9 router diagnostic ablation.

A shuffle control is *vacuous* when the permutation unit holds a single latent:
permuting a constant sequence is the identity, so the shuffled condition is
byte-identical to learned. The evaluator must report that the control was
inapplicable (it failed to construct an intervention) rather than as a
conventional trust failure (intervened and found no effect).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.eval_v6i9_router_diagnostic_ablation import _build_v2_trust_checks


def _rows(conditions: list[str]) -> list[dict]:
    rows: list[dict] = []
    for cond in conditions:
        row = {
            "opponent": "OP8",
            "map": "map_b",
            "condition": cond,
            "episode_seed": 15000,
        }
        if cond == "fixed_z2":
            row["strategy_dominant"] = 2
            row["z_sequence"] = "2"
        rows.append(row)
    return rows


def _trace_rows(learned_seq: str, cross_seq: str) -> list[dict]:
    def _mk(cond: str, seq: str) -> dict:
        return {
            "condition": cond,
            "initial_z": seq.split()[0] if seq else "",
            "z_sequence": seq,
        }

    return [
        _mk("learned_qphi_switching", learned_seq),
        _mk("shuffled_qphi_outputs", learned_seq),
        _mk("shuffled_qphi_cross_episode", cross_seq),
        _mk("fixed_z2", "2"),
    ]


_CONDITIONS = [
    "learned_qphi_switching",
    "fixed_z2",
    "uniform_random_at_router_opportunities",
    "shuffled_qphi_outputs",
    "shuffled_qphi_cross_episode",
]
_FROZEN = {"frozen_tensor_hash_match": True}


class VacuousShuffleLabelingTests(unittest.TestCase):
    def test_constant_z_shuffle_is_untestable_not_failed(self) -> None:
        # Router emits a single latent for every opportunity -> both shuffle
        # units are constant, so neither control can reassign.
        trace_comparison = {
            "learned_qphi_switching__vs__uniform_random_at_router_opportunities": {
                "same_z_sequence_fraction": 0.9,
            },
            "learned_qphi_switching__vs__shuffled_qphi_outputs": {
                "same_z_sequence_fraction": 1.0,
            },
            "learned_qphi_switching__vs__shuffled_qphi_cross_episode": {
                "same_z_sequence_fraction": 1.0,
            },
        }
        checks = _build_v2_trust_checks(
            _rows(_CONDITIONS),
            _trace_rows("3 3 3", "3 3 3"),
            trace_comparison,
            _FROZEN,
            episodes_per_cell=1,
            shuffled_meta={"can_reassign": False, "mean_displacement_fraction": 0.0},
            cross_episode_meta={"can_reassign": False, "mean_reassignment_fraction": 0.0},
        )

        self.assertEqual(checks["shuffle_test_status"], "UNTESTABLE")
        self.assertEqual(
            checks["shuffle_test_reason"],
            "fewer than 2 unique latent assignments inside the permutation unit",
        )
        cross = checks["shuffle_control"]["cross_episode"]
        self.assertFalse(cross["applicable"])
        self.assertIsNone(cross["passed"])
        self.assertEqual(cross["reason"], "constant_z_within_shuffle_unit")
        within = checks["shuffle_control"]["within_episode"]
        self.assertFalse(within["applicable"])
        self.assertIsNone(within["passed"])
        # The causal test still cannot establish trust, but for the truthful
        # reason (no intervention), not a conventional negative-control failure.
        self.assertFalse(checks["v2_trustworthy"])

    def test_applicable_shuffle_that_differs_is_testable_and_passes(self) -> None:
        trace_comparison = {
            "learned_qphi_switching__vs__uniform_random_at_router_opportunities": {
                "same_z_sequence_fraction": 0.9,
            },
            "learned_qphi_switching__vs__shuffled_qphi_outputs": {
                "same_z_sequence_fraction": 0.7,
            },
            "learned_qphi_switching__vs__shuffled_qphi_cross_episode": {
                "same_z_sequence_fraction": 0.6,
            },
        }
        checks = _build_v2_trust_checks(
            _rows(_CONDITIONS),
            _trace_rows("3 1 3", "1 3 1"),
            trace_comparison,
            _FROZEN,
            episodes_per_cell=1,
            shuffled_meta={"can_reassign": True, "mean_displacement_fraction": 0.27},
            cross_episode_meta={"can_reassign": True, "mean_reassignment_fraction": 0.31},
        )

        self.assertEqual(checks["shuffle_test_status"], "TESTABLE")
        self.assertIsNone(checks["shuffle_test_reason"])
        self.assertTrue(checks["shuffle_control"]["cross_episode"]["applicable"])
        self.assertTrue(checks["shuffle_control"]["cross_episode"]["passed"])
        self.assertTrue(checks["shuffle_control"]["within_episode"]["passed"])


if __name__ == "__main__":
    unittest.main()
