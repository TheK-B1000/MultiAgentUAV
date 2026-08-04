"""Offline unit tests for paper eval tools (no GPU / no CUDA).

Covers discovery helpers, matrix aggregation, and tournament top-k selection.
"""

from __future__ import annotations

import csv
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

from checkpoint_tournament import (  # noqa: E402
    aggregate_crossplay_matrix,
    discover_tournament_candidates,
    load_metrics_scores,
    mean_crossplay_score,
    select_topk_indices,
)
from eval_crossplay import (  # noqa: E402
    build_payoff_matrix,
    discover_crossplay_policies,
    format_matrix_table,
    matrix_rows_for_csv,
    wld_to_cell,
)
from eval_exploitability import (  # noqa: E402
    discover_blue_targets,
    exploiter_run_tag,
    exploiter_win_rate_from_blue_wld,
    summarize_exploit_eval,
)
from eval_rollout import match_score_from_wld  # noqa: E402


class TestTournamentDiscovery(unittest.TestCase):
    def test_discovers_snapshots_and_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            tag = "ppo_league_2v2"
            for ep in (100, 300, 200):
                open(
                    os.path.join(tmp, f"{tag}_league_snapshot_ep{ep:06d}.zip"),
                    "wb",
                ).close()
            open(os.path.join(tmp, f"final_{tag}.zip"), "wb").close()
            open(os.path.join(tmp, "distractor_league_snapshot_ep000001.zip"), "wb").close()

            cands = discover_tournament_candidates(tmp, tag, include_final=True)
            eps = [c["episode"] for c in cands if c["kind"] == "snapshot"]
            self.assertEqual(eps, [100, 200, 300])
            self.assertEqual(cands[-1]["kind"], "final")
            self.assertEqual(len(cands), 4)

    def test_metrics_scores_from_success_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            tag = "ppo_league_2v2"
            snap = os.path.join(tmp, f"{tag}_league_snapshot_ep000010.zip")
            open(snap, "wb").close()
            metrics = os.path.join(tmp, f"{tag}_metrics.csv")
            with open(metrics, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["episode_id", "success"])
                w.writeheader()
                for i in range(1, 21):
                    w.writerow({"episode_id": i, "success": 1 if i > 10 else 0})
            cands = discover_tournament_candidates(tmp, tag, include_final=False)
            scores = load_metrics_scores(metrics, cands, window=5)
            self.assertIn(os.path.abspath(snap), scores)
            # Window ending at ep 10: episodes 6..10 all success=0 -> 0%
            self.assertAlmostEqual(scores[os.path.abspath(snap)], 0.0)

    def test_select_topk_fraction_and_min_keep(self):
        scores = [10.0, 90.0, 50.0, 70.0, 20.0, 80.0, 40.0, 60.0, 30.0, 55.0]
        # top 30% of 10 = 3
        idx = select_topk_indices(scores, top_frac=0.30, min_keep=3)
        self.assertEqual(len(idx), 3)
        self.assertEqual([scores[i] for i in idx], [90.0, 80.0, 70.0])

        # min_keep dominates when fraction is tiny
        idx2 = select_topk_indices(scores, top_frac=0.01, min_keep=3)
        self.assertEqual(len(idx2), 3)

    def test_aggregate_crossplay_means(self):
        cells = {
            ("A", "B"): (2, 0, 0),  # A beats B -> 100
            ("A", "C"): (0, 0, 2),  # draw -> 50
            ("B", "A"): (0, 2, 0),
            ("B", "C"): (1, 1, 0),  # 50
            ("C", "A"): (0, 0, 2),
            ("C", "B"): (1, 1, 0),
        }
        means = aggregate_crossplay_matrix(cells)
        self.assertAlmostEqual(means["A"], 75.0)
        self.assertAlmostEqual(means["B"], 25.0)  # (0 + 50) / 2
        self.assertAlmostEqual(mean_crossplay_score([100.0, 50.0]), 75.0)


class TestCrossplayMatrix(unittest.TestCase):
    def test_wld_to_cell(self):
        cell = wld_to_cell(2, 1, 1)
        self.assertEqual(cell["wins"], 2)
        self.assertEqual(cell["n_episodes"], 4)
        self.assertAlmostEqual(cell["match_score"], 62.5)
        self.assertAlmostEqual(match_score_from_wld(2, 1, 1), 62.5)

    def test_build_payoff_matrix_from_fake_wld(self):
        cell_wlds = {
            ("ours_seed42", "roastar_seed42"): (6, 2, 2),  # MS = (6+1)/10 = 70
            ("roastar_seed42", "ours_seed42"): (2, 6, 2),  # MS = (2+1)/10 = 30
        }
        ids = ["ours_seed42", "roastar_seed42"]
        mat = build_payoff_matrix(cell_wlds, ids, ids)
        self.assertEqual(len(mat), 2)
        self.assertAlmostEqual(mat[0][0], 50.0)  # diagonal default
        self.assertAlmostEqual(mat[0][1], 70.0)
        self.assertAlmostEqual(mat[1][0], 30.0)
        self.assertAlmostEqual(mat[1][1], 50.0)

        policies = [
            {"id": "ours_seed42", "label": "SEA-GUARD", "seed": 42, "basename": "a.zip"},
            {"id": "roastar_seed42", "label": "ROA-Star", "seed": 42, "basename": "b.zip"},
        ]
        rows = matrix_rows_for_csv(cell_wlds, policies)
        self.assertEqual(len(rows), 2)
        by_pair = {(r["blue_id"], r["red_id"]): r for r in rows}
        self.assertAlmostEqual(by_pair[("ours_seed42", "roastar_seed42")]["match_score"], 70.0)

        table = format_matrix_table(mat, ["SEA-GUARD", "ROA-Star"], ["SEA-GUARD", "ROA-Star"])
        self.assertIn("SEA-GUARD", table)
        self.assertIn("70.0", table)

    def test_discover_crossplay_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            names = [
                "final_ppo_ablate_ours_2v2.zip",
                "final_ppo_roastar_pfsp_2v2_seed42.zip",
                "final_ppo_ablate_no_league_2v2.zip",
                "final_ppo_ablate_no_curriculum_seed42_2v2.zip",
                "final_ppo_ablate_no_shaping_seed42_rew_no_shaping_2v2.zip",
                "final_ppo_self_play_2v2.zip",
                "final_ppo_league_2v2.zip",  # ours fallback; ablate preferred
            ]
            for name in names:
                open(os.path.join(tmp, name), "wb").close()
            policies = discover_crossplay_policies(tmp, seeds=[42], setting="2v2")
            keys = {p["key"] for p in policies}
            self.assertEqual(
                keys,
                {"ours", "roastar_pfsp", "no_league", "no_curriculum", "no_shaping", "self_play"},
            )
            ours = [p for p in policies if p["key"] == "ours"][0]
            self.assertIn("ablate_ours", ours["basename"])


class TestExploitabilityHelpers(unittest.TestCase):
    def test_discover_blue_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "final_ppo_ablate_ours_2v2.zip"), "wb").close()
            open(os.path.join(tmp, "final_ppo_roastar_pfsp_2v2_seed42.zip"), "wb").close()
            open(os.path.join(tmp, "final_ppo_self_play_2v2.zip"), "wb").close()
            blues = discover_blue_targets(tmp, setting="2v2")
            self.assertEqual({b["key"] for b in blues}, {"ours", "roastar_seed42", "self_play"})

    def test_summarize_exploit_eval(self):
        episodes = [
            {"blue_score": 2, "red_score": 1},  # blue win
            {"blue_score": 0, "red_score": 1},  # blue loss
            {"blue_score": 1, "red_score": 1},  # draw
            {"blue_score": 0, "red_score": 2},  # blue loss
        ]
        row = summarize_exploit_eval(
            episodes,
            method="SEA-GUARD",
            blue_ckpt="/x/final_ppo_ablate_ours_2v2.zip",
            exploiter_ckpt="/x/exploiter_vs_ours_2v2_steps100000.zip",
            steps=100_000,
        )
        # W=1 L=2 D=1 -> blue MS = (1+0.5)/4 = 37.5; exploiter WR = 2/4 = 50
        self.assertAlmostEqual(row["blue_match_score_vs_exploiter"], 37.5)
        self.assertAlmostEqual(row["exploiter_win_rate"], 50.0)
        self.assertAlmostEqual(exploiter_win_rate_from_blue_wld(1, 2, 1), 50.0)
        self.assertIn("exploiter_vs_ours", exploiter_run_tag(
            {"key": "ours"}, steps=100_000, agents=2
        ))


if __name__ == "__main__":
    unittest.main()
