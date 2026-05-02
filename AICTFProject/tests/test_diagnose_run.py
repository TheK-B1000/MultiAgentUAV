from __future__ import annotations

import csv
import subprocess
import sys
import unittest
from pathlib import Path

from tools.diagnose_run import _safe_ratio


_TEST_RUNS = Path(__file__).resolve().parents[1] / ".test_runs" / "diagnose_run"


def _write_metrics(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    _TEST_RUNS.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class DiagnoseRunTests(unittest.TestCase):
    def test_safe_ratio_handles_zero_baseline(self) -> None:
        self.assertEqual(_safe_ratio(0.0, 0.0), 0.0)
        self.assertEqual(_safe_ratio(1.0, 0.0), float("inf"))
        self.assertEqual(_safe_ratio(1.5, 3.0), 0.5)

    def test_diagnose_run_flags_large_z_switch_adv_std_ratio(self) -> None:
        path = _TEST_RUNS / "z_switch_ratio_fail_metrics.csv"
        try:
            _write_metrics(
                path,
                [
                    "rollout_win_rate",
                    "strategy_entropy",
                    "strategy_persist_loss",
                    "clip_fraction",
                    "explained_variance",
                    "strategy_occupancy_0",
                    "strategy_occupancy_1",
                    "rollout_adv_std_at_z_switch",
                    "rollout_adv_std_not_z_switch",
                ],
                [
                    {
                        "rollout_win_rate": "0.0",
                        "strategy_entropy": "0.1",
                        "strategy_persist_loss": "0.1",
                        "clip_fraction": "0.1",
                        "explained_variance": "0.8",
                        "strategy_occupancy_0": "0.5",
                        "strategy_occupancy_1": "0.5",
                        "rollout_adv_std_at_z_switch": "1.0",
                        "rollout_adv_std_not_z_switch": "0.5",
                    },
                    {
                        "rollout_win_rate": "0.2",
                        "strategy_entropy": "0.1",
                        "strategy_persist_loss": "0.1",
                        "clip_fraction": "0.1",
                        "explained_variance": "0.8",
                        "strategy_occupancy_0": "0.5",
                        "strategy_occupancy_1": "0.5",
                        "rollout_adv_std_at_z_switch": "2.0",
                        "rollout_adv_std_not_z_switch": "1.0",
                    },
                ],
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "tools/diagnose_run.py",
                    str(path),
                    "--window",
                    "2",
                    "--max-z-switch-adv-std-ratio",
                    "1.5",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )
        finally:
            if path.exists():
                path.unlink()

        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL: z_switch_adv_std_ratio: 2.0000 <= 1.5000", result.stdout)

    def test_diagnose_run_skips_missing_z_switch_columns(self) -> None:
        path = _TEST_RUNS / "missing_z_switch_metrics.csv"
        try:
            _write_metrics(
                path,
                [
                    "rollout_win_rate",
                    "strategy_entropy",
                    "strategy_persist_loss",
                    "clip_fraction",
                    "explained_variance",
                    "strategy_occupancy_0",
                    "strategy_occupancy_1",
                ],
                [
                    {
                        "rollout_win_rate": "0.0",
                        "strategy_entropy": "0.1",
                        "strategy_persist_loss": "0.1",
                        "clip_fraction": "0.1",
                        "explained_variance": "0.8",
                        "strategy_occupancy_0": "0.5",
                        "strategy_occupancy_1": "0.5",
                    },
                    {
                        "rollout_win_rate": "0.2",
                        "strategy_entropy": "0.1",
                        "strategy_persist_loss": "0.1",
                        "clip_fraction": "0.1",
                        "explained_variance": "0.8",
                        "strategy_occupancy_0": "0.5",
                        "strategy_occupancy_1": "0.5",
                    },
                ],
            )
            result = subprocess.run(
                [sys.executable, "tools/diagnose_run.py", str(path), "--window", "2"],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )
        finally:
            if path.exists():
                path.unlink()

        self.assertEqual(result.returncode, 0)
        self.assertIn("SKIP: z_switch_adv_std_ratio", result.stdout)


if __name__ == "__main__":
    unittest.main()
