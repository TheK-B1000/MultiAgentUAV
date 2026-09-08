#!/usr/bin/env python3
"""Unit tests for G0 weakness-sweep competence + gate logic.

Edge fixtures requested before freezing the analyzer:

1. One G0 seed survives the candidate  → fails all_seeds_negative
2. Upper CI exactly zero               → fails UCB95 < 0 (strict)
3. Exactly three negative opponents    → AMBIGUOUS, no C1 selected

Also covers COMPETENT (0–2 negatives) and INCOMPETENT (4–7 negatives).
"""
from __future__ import annotations

import unittest

import numpy as np

from experiments.analyze_g0_weakness import (
    analyze_results,
    competence_verdict,
    qualifies_weakness,
    select_c1,
)


def _opp(family_mean: float, per_seed, hi: float, *, w: float | None = None,
         sat: float = 0.0, tie: float = 0.0, n_ep: int = 8) -> dict:
    per = np.asarray(per_seed, dtype=float)
    # Synthetic arr so pooled diagnostic has something to chew on.
    arr = np.tile(per.reshape(-1, 1), (1, n_ep))
    return {
        "seeds": [901001, 901002, 901003],
        "per_seed_mean": per,
        "family_mean": float(family_mean),
        "lo": float(family_mean) - 0.2,
        "hi": float(hi),
        "W": float(w if w is not None else per.max()),
        "win": 0.0,
        "tie": float(tie),
        "loss": 1.0,
        "saturation": float(sat),
        "behavior": {},
        "n_ep": n_ep,
        "arr": arr,
    }


class CompetenceVerdictTests(unittest.TestCase):
    def test_competent_zero_to_two(self):
        for n in (0, 1, 2):
            self.assertEqual(competence_verdict(n), "COMPETENT")

    def test_ambiguous_exactly_three(self):
        self.assertEqual(competence_verdict(3), "AMBIGUOUS")

    def test_incompetent_four_to_seven(self):
        for n in (4, 5, 6, 7):
            self.assertEqual(competence_verdict(n), "INCOMPETENT")


class WeaknessGateEdgeFixtures(unittest.TestCase):
    def test_one_seed_survives_fails_all_seeds_negative(self):
        # seed means: -1.0, -0.8, +0.1
        g = qualifies_weakness(np.array([-1.0, -0.8, 0.1]), family_hi=-0.2)
        self.assertFalse(g["all_seeds_negative"])
        self.assertTrue(g["ucb95_strictly_negative"])
        self.assertFalse(g["qualifies"])

    def test_upper_ci_exactly_zero_fails(self):
        g = qualifies_weakness(np.array([-1.0, -0.8, -0.5]), family_hi=0.0)
        self.assertTrue(g["all_seeds_negative"])
        self.assertFalse(g["ucb95_strictly_negative"])
        self.assertFalse(g["qualifies"])

    def test_strict_negative_ucb_passes(self):
        g = qualifies_weakness(np.array([-1.0, -0.8, -0.5]), family_hi=-1e-9)
        self.assertTrue(g["qualifies"])


class CompetenceSelectionFixtures(unittest.TestCase):
    def test_exactly_three_negative_is_ambiguous_no_c1(self):
        # 3 negative family means, 4 positive → AMBIGUOUS; even a perfect
        # weakness must not be selected.
        results = {
            "OP6": _opp(-1.0, [-1.2, -1.0, -0.8], hi=-0.5, w=-0.8),
            "OP7": _opp(-0.8, [-1.0, -0.9, -0.7], hi=-0.3, w=-0.7),
            "OP8": _opp(-0.5, [-0.6, -0.5, -0.4], hi=-0.1, w=-0.4),
            "OP9": _opp(+1.0, [0.8, 1.0, 1.2], hi=+1.5, w=+1.2),
            "OP10": _opp(+1.2, [1.0, 1.2, 1.4], hi=+1.6, w=+1.4),
            "OP11": _opp(+0.5, [0.3, 0.5, 0.7], hi=+0.9, w=+0.7),
            "OP12": _opp(+0.8, [0.6, 0.8, 1.0], hi=+1.1, w=+1.0),
        }
        rng = np.random.default_rng(0)
        summary = analyze_results(results, n_boot=200, alpha=0.05, rng=rng)
        self.assertEqual(summary["n_negative_opponents"], 3)
        self.assertEqual(summary["competence"], "AMBIGUOUS")
        self.assertIsNone(summary["selected_c1"])
        # Gate may still mark OP6–OP8 as qualifiers informationally.
        self.assertGreaterEqual(len(summary["qualified"]), 1)

    def test_competent_isolated_failure_selects_c1(self):
        # 1 negative opponent that clears the gate → COMPETENT + C1.
        results = {
            "OP6": _opp(-1.0, [-1.2, -1.0, -0.8], hi=-0.4, w=-0.8),
            "OP7": _opp(+1.0, [0.8, 1.0, 1.2], hi=+1.4, w=+1.2),
            "OP8": _opp(+0.9, [0.7, 0.9, 1.1], hi=+1.3, w=+1.1),
            "OP9": _opp(+0.5, [0.3, 0.5, 0.7], hi=+0.9, w=+0.7),
            "OP10": _opp(+1.5, [1.3, 1.5, 1.7], hi=+1.9, w=+1.7),
            "OP11": _opp(+0.4, [0.2, 0.4, 0.6], hi=+0.8, w=+0.6),
            "OP12": _opp(+0.6, [0.4, 0.6, 0.8], hi=+1.0, w=+0.8),
        }
        rng = np.random.default_rng(0)
        summary = analyze_results(results, n_boot=200, alpha=0.05, rng=rng)
        self.assertEqual(summary["competence"], "COMPETENT")
        self.assertEqual(summary["selected_c1"], "OP6")

    def test_incompetent_blocks_c1_even_if_gate_clears(self):
        # 5 negative opponents → INCOMPETENT; no C1.
        results = {
            f"OP{i}": _opp(-0.5, [-0.7, -0.5, -0.3], hi=-0.1, w=-0.3)
            for i in range(6, 11)
        }
        results["OP11"] = _opp(+1.0, [0.8, 1.0, 1.2], hi=+1.4, w=+1.2)
        results["OP12"] = _opp(+0.5, [0.3, 0.5, 0.7], hi=+0.9, w=+0.7)
        rng = np.random.default_rng(0)
        summary = analyze_results(results, n_boot=200, alpha=0.05, rng=rng)
        self.assertEqual(summary["n_negative_opponents"], 5)
        self.assertEqual(summary["competence"], "INCOMPETENT")
        self.assertIsNone(summary["selected_c1"])

    def test_surviving_seed_blocks_selection_under_competent(self):
        results = {
            "OP6": _opp(-0.5, [-1.0, -0.8, 0.1], hi=-0.2, w=+0.1),  # fails all_neg
            "OP7": _opp(+1.0, [0.8, 1.0, 1.2], hi=+1.4, w=+1.2),
            "OP8": _opp(+0.9, [0.7, 0.9, 1.1], hi=+1.3, w=+1.1),
            "OP9": _opp(+0.5, [0.3, 0.5, 0.7], hi=+0.9, w=+0.7),
            "OP10": _opp(+1.5, [1.3, 1.5, 1.7], hi=+1.9, w=+1.7),
            "OP11": _opp(+0.4, [0.2, 0.4, 0.6], hi=+0.8, w=+0.6),
            "OP12": _opp(+0.6, [0.4, 0.6, 0.8], hi=+1.0, w=+0.8),
        }
        # Family mean for OP6 is negative so n_neg=1 → COMPETENT, but gate fails.
        rng = np.random.default_rng(0)
        summary = analyze_results(results, n_boot=200, alpha=0.05, rng=rng)
        self.assertEqual(summary["competence"], "COMPETENT")
        self.assertIsNone(summary["selected_c1"])
        self.assertFalse(results["OP6"]["gate"]["all_seeds_negative"])

    def test_select_c1_direct_helper_respects_competence(self):
        results = {
            "OP6": _opp(-1.0, [-1.2, -1.0, -0.8], hi=-0.4, w=-0.8),
        }
        results["OP6"]["gate"] = qualifies_weakness(
            results["OP6"]["per_seed_mean"], results["OP6"]["hi"]
        )
        self.assertIsNone(select_c1(results, ["OP6"], "AMBIGUOUS"))
        self.assertEqual(select_c1(results, ["OP6"], "COMPETENT"), "OP6")


if __name__ == "__main__":
    unittest.main()
