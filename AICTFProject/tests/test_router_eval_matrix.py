"""Tests for offline q_phi router-quality ledger construction."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from tools.router_eval_matrix import build_router_ledger, load_eval_cells


def _write_eval_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "map_set",
        "latent_selection",
        "fixed_latent_id",
        "latent_resample_every",
        "opponent",
        "episodes",
        "success_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class RouterEvalMatrixTests(unittest.TestCase):
    def test_router_ledger_separates_train_and_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval_aggregate.csv"
            rows: list[dict[str, object]] = []
            payoff = {
                "OP5": [0.20, 0.80, 0.40, 0.30],
                "OP6": [0.75, 0.25, 0.35, 0.20],
                "OP4": [0.65, 0.30, 0.20, 0.10],
            }
            router = {"OP5": 0.78, "OP6": 0.72, "OP4": 0.25}
            uniform_episode = {"OP5": 0.40, "OP6": 0.36, "OP4": 0.28}
            uniform_router = {"OP5": 0.42, "OP6": 0.38, "OP4": 0.30}
            no_switch = {"OP5": 0.55, "OP6": 0.65, "OP4": 0.40}
            shuffled = {"OP5": 0.50, "OP6": 0.60, "OP4": 0.35}
            calibration = {
                "OP5": [0.20, 0.82, 0.40, 0.30],
                "OP6": [0.70, 0.30, 0.35, 0.20],
            }
            for opponent, fixed_values in calibration.items():
                for z, success_rate in enumerate(fixed_values):
                    rows.append(
                        {
                            "map_set": "calibration",
                            "latent_selection": "fixed",
                            "fixed_latent_id": z,
                            "latent_resample_every": "",
                            "opponent": opponent,
                            "episodes": 50,
                            "success_rate": success_rate,
                        }
                    )
            for opponent, fixed_values in payoff.items():
                rows.append(
                    {
                        "map_set": "eval",
                        "latent_selection": "learned_qphi_switching",
                        "fixed_latent_id": "",
                        "latent_resample_every": "",
                        "opponent": opponent,
                        "episodes": 100,
                        "success_rate": router[opponent],
                    }
                )
                rows.append(
                    {
                        "map_set": "eval",
                        "latent_selection": "uniform_episode_fixed",
                        "fixed_latent_id": "",
                        "latent_resample_every": "",
                        "opponent": opponent,
                        "episodes": 100,
                        "success_rate": uniform_episode[opponent],
                    }
                )
                rows.append(
                    {
                        "map_set": "eval",
                        "latent_selection": "uniform_random_at_router_opportunities",
                        "fixed_latent_id": "",
                        "latent_resample_every": "",
                        "opponent": opponent,
                        "episodes": 100,
                        "success_rate": uniform_router[opponent],
                    }
                )
                rows.append(
                    {
                        "map_set": "eval",
                        "latent_selection": "qphi_initial_only_no_switch",
                        "fixed_latent_id": "",
                        "latent_resample_every": 0,
                        "opponent": opponent,
                        "episodes": 100,
                        "success_rate": no_switch[opponent],
                    }
                )
                rows.append(
                    {
                        "map_set": "eval",
                        "latent_selection": "shuffled_qphi_outputs",
                        "fixed_latent_id": "",
                        "latent_resample_every": "",
                        "opponent": opponent,
                        "episodes": 100,
                        "success_rate": shuffled[opponent],
                    }
                )
                for z, success_rate in enumerate(fixed_values):
                    rows.append(
                        {
                            "map_set": "eval",
                            "latent_selection": "fixed",
                            "fixed_latent_id": z,
                            "latent_resample_every": "",
                            "opponent": opponent,
                            "episodes": 100,
                            "success_rate": success_rate,
                        }
                    )
            _write_eval_csv(path, rows)

            cells = load_eval_cells([path])
            pair_rows, split_rows = build_router_ledger(
                cells,
                latent_k=4,
                holdout_opponents=["OP4"],
            )

        by_split = {str(r["split"]): r for r in split_rows}
        self.assertEqual(len(pair_rows), 3)
        self.assertAlmostEqual(float(by_split["train"]["g_available"]), 0.25)
        self.assertGreater(float(by_split["train"]["g_realized"]), 0.0)
        self.assertGreater(float(by_split["train"]["g_no_switch"]), 0.0)
        self.assertGreater(float(by_split["train"]["delta_vs_uniform_episode_fixed"]), 0.0)
        self.assertGreater(
            float(by_split["train"]["delta_vs_uniform_random_at_router_opportunities"]),
            0.0,
        )
        self.assertGreater(float(by_split["train"]["delta_router_primary"]), 0.0)
        self.assertTrue(by_split["train"]["router_beats_preselected_global_fixed_z"])
        self.assertTrue(by_split["train"]["router_beats_no_switch"])
        self.assertTrue(by_split["train"]["router_beats_primary_baseline"])
        self.assertIn("holdout", by_split)
        self.assertLess(float(by_split["holdout"]["g_realized"]), 0.0)
        self.assertLess(float(by_split["holdout"]["delta_router_primary"]), 0.0)
        self.assertFalse(by_split["holdout"]["router_beats_preselected_global_fixed_z"])

    def test_missing_fixed_z_cell_omits_incomplete_opponent(self):
        cells = load_eval_cells([])
        pair_rows, split_rows = build_router_ledger(cells, latent_k=4)
        self.assertEqual(pair_rows, [])
        self.assertEqual(split_rows, [])

    def test_router_r0_rows_normalize_to_no_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval_aggregate.csv"
            _write_eval_csv(
                path,
                [
                    {
                        "map_set": "eval",
                        "latent_selection": "router",
                        "fixed_latent_id": "",
                        "latent_resample_every": 0,
                        "opponent": "OP5",
                        "episodes": 10,
                        "success_rate": 0.5,
                    }
                ],
            )
            cells = load_eval_cells([path])
        self.assertEqual(cells[0].latent_selection, "qphi_initial_only_no_switch")


if __name__ == "__main__":
    unittest.main()
