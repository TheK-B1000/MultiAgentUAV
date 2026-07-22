"""Unit tests for ablation shared-eval discovery and seed aggregation."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from eval_ablations import (  # noqa: E402
    ABLATION_ARMS,
    aggregate_across_seeds,
    discover_all_checkpoints,
    discover_arm_checkpoints,
    parse_seed_from_filename,
)


class TestParseSeed(unittest.TestCase):
    def test_legacy_seed42(self):
        self.assertEqual(parse_seed_from_filename("final_ppo_ablate_ours_2v2.zip"), 42)

    def test_explicit_seed(self):
        self.assertEqual(parse_seed_from_filename("final_ppo_ablate_ours_seed43_2v2.zip"), 43)
        self.assertEqual(
            parse_seed_from_filename("final_ppo_ablate_no_shaping_seed44_rew_no_shaping_2v2.zip"),
            44,
        )


class TestDiscovery(unittest.TestCase):
    def test_discovers_arms_and_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            names = [
                "final_ppo_ablate_ours_2v2.zip",
                "final_ppo_ablate_ours_seed43_2v2.zip",
                "final_ppo_ablate_no_league_2v2.zip",
                "final_ppo_ablate_no_league_seed43_2v2.zip",
                "final_ppo_ablate_no_curriculum_seed42_2v2.zip",
                "final_ppo_ablate_no_shaping_seed42_rew_no_shaping_2v2.zip",
                "final_ppo_league_2v2.zip",  # distractor
            ]
            for name in names:
                open(os.path.join(tmp, name), "wb").close()

            all_ckpts = discover_all_checkpoints(tmp)
            self.assertEqual(set(all_ckpts), {"ours", "no_league", "no_curriculum", "no_shaping"})
            self.assertEqual(set(all_ckpts["ours"]), {42, 43})
            self.assertEqual(set(all_ckpts["no_league"]), {42, 43})
            self.assertEqual(set(all_ckpts["no_curriculum"]), {42})
            self.assertEqual(set(all_ckpts["no_shaping"]), {42})

            ours_only = discover_all_checkpoints(tmp, arms=["ours"], seeds=[43])
            self.assertEqual(list(ours_only), ["ours"])
            self.assertEqual(set(ours_only["ours"]), {43})

    def test_arm_patterns_cover_all_keys(self):
        keys = {a.key for a in ABLATION_ARMS}
        self.assertEqual(keys, {"ours", "no_league", "no_curriculum", "no_shaping"})


class TestAggregateAcrossSeeds(unittest.TestCase):
    def test_mean_std_across_seeds(self):
        aggs = [
            {"success_rate": 60.0, "success_rate_std": 1.0, "mean_steps": 100.0, "mean_steps_std": 0.0,
             "collision_free_rate": 100.0, "collision_free_rate_std": 0.0, "return_var": 1.0, "return_var_std": 0.0,
             "coverage_efficiency": 0.0, "coverage_efficiency_std": 0.0, "win_margin_mean": 1.0, "win_margin_std": 0.0,
             "time_to_first_score_mean": float("nan"), "time_to_first_score_std": 0.0,
             "mean_inter_robot_dist_mean": float("nan"), "mean_inter_robot_dist_std": 0.0},
            {"success_rate": 70.0, "success_rate_std": 1.0, "mean_steps": 120.0, "mean_steps_std": 0.0,
             "collision_free_rate": 100.0, "collision_free_rate_std": 0.0, "return_var": 1.0, "return_var_std": 0.0,
             "coverage_efficiency": 0.0, "coverage_efficiency_std": 0.0, "win_margin_mean": 1.0, "win_margin_std": 0.0,
             "time_to_first_score_mean": float("nan"), "time_to_first_score_std": 0.0,
             "mean_inter_robot_dist_mean": float("nan"), "mean_inter_robot_dist_std": 0.0},
        ]
        row = aggregate_across_seeds("2v2", "Ours", "OP3", aggs)
        self.assertEqual(row["method"], "Ours")
        self.assertEqual(row["opponent"], "OP3")
        self.assertAlmostEqual(float(row["success_rate_mean"]), 65.0)
        # sample std of [60, 70] = sqrt(50) ≈ 7.071
        self.assertAlmostEqual(float(row["success_rate_std"]), 50.0 ** 0.5, places=5)
        self.assertAlmostEqual(float(row["mean_steps_mean"]), 110.0)

    def test_single_seed_std_zero(self):
        aggs = [{"success_rate": 55.0, "mean_steps": 90.0, "collision_free_rate": 100.0,
                 "return_var": 0.0, "coverage_efficiency": 0.0, "win_margin_mean": 0.0,
                 "time_to_first_score_mean": float("nan"), "mean_inter_robot_dist_mean": float("nan")}]
        row = aggregate_across_seeds("2v2", "No league", "OP4", aggs)
        self.assertAlmostEqual(float(row["success_rate_mean"]), 55.0)
        self.assertAlmostEqual(float(row["success_rate_std"]), 0.0)


if __name__ == "__main__":
    unittest.main()
