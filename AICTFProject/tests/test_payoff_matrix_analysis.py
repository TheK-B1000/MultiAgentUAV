"""Pinning tests for pool-admissibility ``delta_pool`` (cross-fitted gate)."""

from __future__ import annotations

import unittest

from experiments.payoff_matrix_analysis import (
    analyze_pool,
    cells_from_rows,
    validate_cells,
    _synthetic_counters,
    _synthetic_saturated,
)


class PayoffMatrixValidationTests(unittest.TestCase):
    def test_rejects_unequal_episode_counts(self) -> None:
        cells = {
            ("rush", "bait"): [1.0, 0.0, -1.0],
            ("turtle", "bait"): [1.0, 0.0],  # mismatched
            ("rush", "race"): [0.0, 1.0, 0.0],
            ("turtle", "race"): [0.0, 1.0, 0.0],
        }
        with self.assertRaises(ValueError):
            validate_cells(cells)

    def test_cells_from_rows_sorts_by_episode_index(self) -> None:
        rows = [
            {"blue_style": "rush", "red_style": "bait", "episode_index": 2, "win_margin": 1.0},
            {"blue_style": "rush", "red_style": "bait", "episode_index": 0, "win_margin": -1.0},
            {"blue_style": "rush", "red_style": "bait", "episode_index": 1, "win_margin": 0.0},
            {"blue_style": "turtle", "red_style": "bait", "episode_index": 0, "win_margin": 0.5},
            {"blue_style": "turtle", "red_style": "bait", "episode_index": 1, "win_margin": 0.5},
            {"blue_style": "turtle", "red_style": "bait", "episode_index": 2, "win_margin": 0.5},
        ]
        cells = cells_from_rows(rows)
        self.assertEqual(list(cells[("rush", "bait")]), [-1.0, 0.0, 1.0])


class PayoffMatrixCrossfitGateTests(unittest.TestCase):
    """In-sample max is winner's-curse biased; cross-fit must kill structureless pools."""

    def test_saturated_pool_fails_delta_lcb(self) -> None:
        rep = analyze_pool(_synthetic_saturated(n=64, seed=0), n_boot=400, seed=0)
        # In-sample selective value is biased upward — must not be the gate.
        self.assertGreater(rep.v_selective_insample, rep.v_best_fixed_insample)
        self.assertFalse(rep.gates["delta_pool_lcb_positive"])
        self.assertLessEqual(rep.delta_pool_lcb, 0.0)
        self.assertFalse(rep.admissible)

    def test_counter_structured_pool_passes_delta_lcb(self) -> None:
        # Relax WR-band / tie-rate / degenerate-red for the synthetic counter
        # means (they are designed for delta structure, not saturation band).
        rep = analyze_pool(
            _synthetic_counters(n=64, seed=0),
            n_boot=400,
            seed=0,
            wr_band=(0.0, 1.0),
            max_tie_rate=1.0,
            min_br_diversity=4,
        )
        self.assertTrue(rep.gates["delta_pool_lcb_positive"])
        self.assertGreater(rep.delta_pool_lcb, 0.0)
        self.assertGreaterEqual(rep.br_diversity, 2)
        self.assertTrue(rep.gates["all_blues_protected"])
        self.assertEqual(rep.unprotected_blues, [])
        self.assertIsNone(rep.dominating_blue_style)
        # Full admissible may still fail no_degenerate_red depending on noise;
        # the load-bearing gates are delta_pool_lcb_positive + all_blues_protected.
        self.assertTrue(rep.gates["delta_pool_lcb_positive"])

    def test_two_niche_split_pattern_fails_all_blues_protected(self) -> None:
        # OP6→TURTLE, OP7–OP10→SPLIT: only two useful niches.
        n = 32
        cells = {}
        reds = ("op6", "op7", "op8", "op9", "op10")
        for r in reds:
            turtle = 1.0 if r == "op6" else -0.5
            split = -0.5 if r == "op6" else 1.0
            cells[("BLUE_TURTLE", r)] = [turtle] * n
            cells[("BLUE_SPLIT", r)] = [split] * n
            cells[("BLUE_RUSH", r)] = [-1.0] * n
            cells[("BLUE_ESCORT", r)] = [-1.0] * n
        rep = analyze_pool(
            cells,
            n_boot=200,
            seed=1,
            wr_band=(0.0, 1.0),
            max_tie_rate=1.0,
            min_br_diversity=4,
        )
        self.assertEqual(rep.br_diversity, 2)
        self.assertFalse(rep.gates["all_blues_protected"])
        self.assertIn("BLUE_RUSH", rep.unprotected_blues)
        self.assertIn("BLUE_ESCORT", rep.unprotected_blues)
        self.assertFalse(rep.gates["best_response_diversity"])
        self.assertFalse(rep.admissible)

    def test_dominating_row_forces_zero_structure(self) -> None:
        # rush strictly dominates every column.
        n = 48
        cells = {}
        for r in ("bait", "race", "collapse", "flank"):
            cells[("rush", r)] = [1.0] * n
            cells[("turtle", r)] = [0.0] * n
            cells[("split", r)] = [-0.5] * n
            cells[("escort", r)] = [-1.0] * n
        rep = analyze_pool(
            cells,
            n_boot=200,
            seed=1,
            wr_band=(0.0, 1.0),
            max_tie_rate=1.0,
        )
        self.assertEqual(rep.dominating_blue_style, "rush")
        self.assertFalse(rep.gates["no_dominating_blue_style"])
        self.assertLessEqual(rep.delta_pool_lcb, 0.05)


if __name__ == "__main__":
    unittest.main()
