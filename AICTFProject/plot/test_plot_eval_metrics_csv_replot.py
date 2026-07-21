import csv
import os
import tempfile
import unittest
from unittest import mock

from plot_eval_metrics import main as plot_eval_metrics_main

_FIELDNAMES = [
    "setting", "method", "opponent",
    "success_rate_mean", "success_rate_std",
    "mean_steps_mean", "mean_steps_std",
    "collision_free_mean", "collision_free_std",
    "return_variance_mean", "return_variance_std",
    "coverage_efficiency_mean", "coverage_efficiency_std",
    "win_margin_mean", "win_margin_std",
    "time_to_first_score_mean", "time_to_first_score_std",
    "mean_inter_robot_dist_mean", "mean_inter_robot_dist_std",
]


def _row(method: str, success: float) -> dict:
    return {
        "setting": "2v2", "method": method, "opponent": "OP4",
        "success_rate_mean": success, "success_rate_std": 10.0,
        "mean_steps_mean": 90.0, "mean_steps_std": 10.0,
        "collision_free_mean": 100.0, "collision_free_std": 0.0,
        "return_variance_mean": 1.0, "return_variance_std": 0.0,
        "coverage_efficiency_mean": 0.0, "coverage_efficiency_std": 0.0,
        "win_margin_mean": 1.5, "win_margin_std": 0.5,
        "time_to_first_score_mean": "nan", "time_to_first_score_std": 0.0,
        "mean_inter_robot_dist_mean": "nan", "mean_inter_robot_dist_std": 0.0,
    }


class ReplotFromCsvRegressionTests(unittest.TestCase):
    """
    Regression guard for a pre-existing bug where --metrics-csv always crashed
    with NameError (n_episodes only defined on the live-eval path), and for the
    fix that lets a 4th method (e.g. "ROA-Star (PFSP)") in the CSV actually
    reach the printed table / --table-out CSV / bar charts instead of being
    silently dropped in favor of the hardcoded Ours/Jacob et al./Self-play trio.
    """

    def test_four_method_csv_replots_without_crashing_and_keeps_all_methods(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "synthetic.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
                writer.writeheader()
                writer.writerows([
                    _row("Ours", 90.0),
                    _row("Jacob et al.", 85.0),
                    _row("Self-play", 80.0),
                    _row("ROA-Star (PFSP)", 92.0),
                ])

            out_base = os.path.join(tmpdir, "plot")
            table_out = os.path.join(tmpdir, "table.csv")
            argv = [
                "plot_eval_metrics.py",
                "--metrics-csv", csv_path,
                "--modes", "2v2",
                "--table-opponent", "OP4",
                "--out", out_base,
                "--table-out", table_out,
            ]
            with mock.patch("sys.argv", argv):
                plot_eval_metrics_main()  # must not raise

            self.assertTrue(os.path.isfile(table_out))
            with open(table_out, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            methods = {r["method"] for r in rows}
            self.assertEqual(methods, {"Ours", "Jacob et al.", "Self-play", "ROA-Star (PFSP)"})


if __name__ == "__main__":
    unittest.main()
