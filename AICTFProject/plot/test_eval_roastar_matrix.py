"""Unit tests for ROA-Star shared-eval discovery, match score, and paired bootstrap."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from eval_rollout import (  # noqa: E402
    compute_aggregates,
    match_score_from_wld,
    paired_bootstrap_seed_mean,
    shared_episode_seeds,
)
from eval_roastar_matrix import (  # noqa: E402
    aggregate_setting_opponent,
    discover_roastar_finals,
    parse_seed,
    parse_setting,
)


class TestParse(unittest.TestCase):
    def test_parse_seed_setting(self):
        name = "final_ppo_roastar_pfsp_3v3_seed44.zip"
        self.assertEqual(parse_seed(name), 44)
        self.assertEqual(parse_setting(name), "3v3")


class TestDiscovery(unittest.TestCase):
    def test_discovers_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            for setting in ("2v2", "3v3", "4v4"):
                d = os.path.join(tmp, setting)
                os.makedirs(d)
                for seed in (42, 43, 44):
                    open(
                        os.path.join(d, f"final_ppo_roastar_pfsp_{setting}_seed{seed}.zip"),
                        "wb",
                    ).close()
            found = discover_roastar_finals(
                tmp, settings=["2v2", "3v3", "4v4"], seeds=[42, 43, 44]
            )
            self.assertEqual(set(found), {"2v2", "3v3", "4v4"})
            for setting in found:
                self.assertEqual(set(found[setting]), {42, 43, 44})


class TestMatchScore(unittest.TestCase):
    def test_wld_to_match_score(self):
        # 2W 1D 1L -> (2+0.5)/4 = 0.625 -> 62.5%
        self.assertAlmostEqual(match_score_from_wld(2, 1, 1), 62.5)

    def test_compute_aggregates_includes_match_score(self):
        episodes = [
            {"success": 1, "blue_score": 2, "red_score": 1, "steps": 10, "return": 1.0,
             "zone_coverage": 0.0, "collision_free": 1, "win_margin": 1,
             "time_to_first_score": float("nan"), "mean_inter_robot_dist": float("nan")},
            {"success": 0, "blue_score": 1, "red_score": 1, "steps": 10, "return": 0.0,
             "zone_coverage": 0.0, "collision_free": 1, "win_margin": 0,
             "time_to_first_score": float("nan"), "mean_inter_robot_dist": float("nan")},
            {"success": 0, "blue_score": 0, "red_score": 1, "steps": 10, "return": -1.0,
             "zone_coverage": 0.0, "collision_free": 1, "win_margin": -1,
             "time_to_first_score": float("nan"), "mean_inter_robot_dist": float("nan")},
        ]
        agg = compute_aggregates(episodes)
        self.assertEqual(agg["wins"], 1)
        self.assertEqual(agg["draws"], 1)
        self.assertEqual(agg["losses"], 1)
        self.assertAlmostEqual(agg["match_score"], 50.0)
        self.assertIn("match_score_ci_lo", agg)


class TestSharedSeeds(unittest.TestCase):
    def test_op3_op4_disjoint(self):
        a = shared_episode_seeds(5, 42, "OP3")
        b = shared_episode_seeds(5, 42, "OP4")
        self.assertEqual(a, [42, 43, 44, 45, 46])
        self.assertTrue(set(a).isdisjoint(set(b)))


class TestPairedBootstrap(unittest.TestCase):
    def test_equal_length(self):
        pts = [
            np.array([1.0, 0.0, 0.5, 1.0]),
            np.array([1.0, 1.0, 0.0, 0.5]),
            np.array([0.5, 0.5, 0.5, 0.5]),
        ]
        mean, lo, hi = paired_bootstrap_seed_mean(pts, n_boot=200, rng=np.random.default_rng(0))
        self.assertTrue(0.0 <= lo <= mean <= hi <= 100.0)

    def test_aggregate_setting_opponent(self):
        aggs = [
            {
                "n_episodes": 4,
                "win_rate": 50.0,
                "loss_rate": 25.0,
                "draw_rate": 25.0,
                "match_score": 62.5,
                "success_rate": 50.0,
                "mean_steps": 100.0,
                "mean_captures": 1.0,
                "defense_shutout_rate": 10.0,
                "collision_free_rate": 100.0,
                "win_margin_mean": 0.5,
            },
            {
                "n_episodes": 4,
                "win_rate": 75.0,
                "loss_rate": 25.0,
                "draw_rate": 0.0,
                "match_score": 75.0,
                "success_rate": 75.0,
                "mean_steps": 110.0,
                "mean_captures": 1.2,
                "defense_shutout_rate": 20.0,
                "collision_free_rate": 100.0,
                "win_margin_mean": 0.8,
            },
        ]
        points = [
            np.array([1.0, 0.5, 0.0, 1.0]),
            np.array([1.0, 1.0, 0.0, 1.0]),
        ]
        row = aggregate_setting_opponent("2v2", "OP3", aggs, points, n_boot=100)
        self.assertEqual(row["n_seeds"], 2)
        self.assertAlmostEqual(row["match_score_mean"], 68.75)
        self.assertTrue(row["match_score_ci_lo"] <= row["match_score_mean"] <= row["match_score_ci_hi"])


if __name__ == "__main__":
    unittest.main()
