import csv
import os
import tempfile
import unittest

from aggregate_ablation_results import aggregate_by_arm, parse_run_tag, summarize_csv

_FIELDNAMES = [
    "episode_id", "success", "time_to_first_score", "time_to_game_over",
    "collisions_per_episode", "near_misses_per_episode", "collision_free_episode",
    "mean_inter_robot_dist", "std_inter_robot_dist", "zone_coverage",
    "phase_name", "opponent_kind", "scripted_tag", "blue_score", "red_score",
    "opponent_switch_count", "vec_schema_version",
]


def _write_metrics_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _row(episode_id: int, success: int, phase: str = "OP3", switch: int = 1) -> dict:
    return {
        "episode_id": episode_id, "success": success, "time_to_first_score": "",
        "time_to_game_over": "", "collisions_per_episode": 0, "near_misses_per_episode": 0,
        "collision_free_episode": 1, "mean_inter_robot_dist": "", "std_inter_robot_dist": "",
        "zone_coverage": 0, "phase_name": phase, "opponent_kind": "scripted",
        "scripted_tag": phase, "blue_score": 3, "red_score": 2 if success else 3,
        "opponent_switch_count": switch, "vec_schema_version": 2,
    }


class ParseRunTagTests(unittest.TestCase):
    def test_parses_default_seed_tag(self):
        self.assertEqual(parse_run_tag("ppo_ablate_ours_2v2_metrics.csv"), ("ours", None, "2v2"))

    def test_parses_explicit_seed_tag(self):
        self.assertEqual(parse_run_tag("ppo_ablate_ours_seed43_2v2_metrics.csv"), ("ours", 43, "2v2"))

    def test_parses_multi_word_arm_name(self):
        self.assertEqual(parse_run_tag("ppo_ablate_no_curriculum_seed44_2v2_metrics.csv"), ("no_curriculum", 44, "2v2"))

    def test_rejects_unrelated_filename(self):
        self.assertIsNone(parse_run_tag("some_other_file.csv"))


class SummarizeCsvTests(unittest.TestCase):
    def test_computes_overall_and_tail_win_rate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ppo_ablate_ours_2v2_metrics.csv")
            rows = [_row(i, 1 if i >= 5 else 0) for i in range(10)]  # 5/10 overall, last 3 all wins
            _write_metrics_csv(path, rows)

            summary = summarize_csv(path, tail_window=3)

            self.assertEqual(summary.arm, "ours")
            self.assertIsNone(summary.seed)
            self.assertEqual(summary.n_episodes, 10)
            self.assertAlmostEqual(summary.overall_success_rate, 50.0)
            self.assertAlmostEqual(summary.tail_success_rate, 100.0)
            self.assertTrue(summary.reached_op3)

    def test_tolerates_malformed_trailing_row(self):
        """Guards against a partially-flushed last line from a still-running
        training process -- must not raise, just skip the bad row."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ppo_ablate_ours_2v2_metrics.csv")
            rows = [_row(i, 1) for i in range(5)]
            _write_metrics_csv(path, rows)
            with open(path, "a", encoding="utf-8") as f:
                f.write("6,")  # truncated row, no success value

            summary = summarize_csv(path, tail_window=100)
            self.assertEqual(summary.n_episodes, 5)
            self.assertAlmostEqual(summary.overall_success_rate, 100.0)

    def test_returns_none_for_unparseable_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "not_an_ablation_file.csv")
            _write_metrics_csv(path, [_row(0, 1)])
            self.assertIsNone(summarize_csv(path, tail_window=10))


class AggregateByArmTests(unittest.TestCase):
    def test_aggregates_mean_and_std_across_seeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = {
                42: os.path.join(tmpdir, "ppo_ablate_ours_2v2_metrics.csv"),
                43: os.path.join(tmpdir, "ppo_ablate_ours_seed43_2v2_metrics.csv"),
                44: os.path.join(tmpdir, "ppo_ablate_ours_seed44_2v2_metrics.csv"),
            }
            win_rates = {42: 50, 43: 60, 44: 70}  # -> mean 60, easy to sanity check
            for seed, path in paths.items():
                wr = win_rates[seed]
                rows = [_row(i, 1 if i < wr else 0) for i in range(100)]
                _write_metrics_csv(path, rows)

            summaries = [summarize_csv(p, tail_window=100) for p in paths.values()]
            arm_rows = aggregate_by_arm(summaries)

            self.assertEqual(len(arm_rows), 1)
            row = arm_rows[0]
            self.assertEqual(row["arm"], "ours")
            self.assertEqual(row["n_seeds"], 3)
            self.assertAlmostEqual(row["overall_success_rate_mean"], 60.0, places=1)
            self.assertTrue(row["all_reached_op3"])


if __name__ == "__main__":
    unittest.main()
