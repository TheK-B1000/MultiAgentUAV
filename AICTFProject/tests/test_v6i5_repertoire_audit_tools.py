"""Tests for v6i5 Phase-A audit and forced-repertoire report helpers."""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl.behavior_telemetry import BEHAVIOR_TELEMETRY_NAMES
from rl.eval_forced_latent_repertoire import (
    _resolve_map_layout,
    build_behavior_matrix,
    build_episode_results,
    build_pairwise_distances,
    build_readiness_report,
    build_summary,
)
from tools.v6i5_phase_a_audit import build_report, compute_cross_row_retention
from tools.v6i5_state_conditioned_branch_eval import (
    compute_advantage_rows,
    compute_oracle_rows,
    state_bucket_labels,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class V6I5PhaseAAuditToolTests(unittest.TestCase):
    def test_cross_row_retention_uses_next_update_start(self) -> None:
        rows = [
            {
                "update": "1",
                "actor_jsd_after_ppo": "0.10",
                "actor_jsd_after_cf": "0.16",
            },
            {
                "update": "2",
                "actor_jsd_update_start": "0.13",
                "actor_jsd_after_ppo": "0.12",
                "actor_jsd_after_cf": "0.18",
            },
        ]
        out = compute_cross_row_retention(rows)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["cf_gain"], 0.06)
        self.assertAlmostEqual(out[0]["retained_gain"], 0.03)
        self.assertAlmostEqual(out[0]["cross_row_retention_ratio"], 0.5)

    def test_phase_a_report_passes_healthy_latest_row(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "metrics.csv"
            _write_csv(
                path,
                [
                    {
                        "update": "1",
                        "actor_z_jsd_mean": "0.002",
                        "actor_z_pairs_above_margin": "6",
                        "actor_cf_to_ppo_grad_ratio": "0.02",
                        "actor_jsd_update_start": "0.10",
                        "actor_jsd_after_ppo": "0.09",
                        "actor_jsd_after_cf": "0.12",
                        "cf_jsd_delta": "0.03",
                        "ppo_jsd_delta": "-0.01",
                        "actor_kl_after_ppo": "0.01",
                        "actor_kl_after_cf": "0.01",
                        "win_rate": "0.50",
                    },
                    {
                        "update": "2",
                        "actor_z_jsd_mean": "0.003",
                        "actor_z_pairs_above_margin": "6",
                        "actor_cf_to_ppo_grad_ratio": "0.02",
                        "actor_jsd_update_start": "0.11",
                        "actor_jsd_after_ppo": "0.10",
                        "actor_jsd_after_cf": "0.13",
                        "cf_jsd_delta": "0.03",
                        "ppo_jsd_delta": "-0.01",
                        "actor_kl_after_ppo": "0.01",
                        "actor_kl_after_cf": "0.01",
                        "win_rate": "0.51",
                        "actor_cf_optimizer_step_count": "1",
                    },
                ],
            )
            report = build_report(path)
        self.assertEqual(report["recommendation"], "continue_current_run")
        self.assertTrue(report["pass_checks"]["actor_z_jsd_above_margin"])
        self.assertTrue(report["pass_checks"]["pairs_above_margin"])


class V6I5ForcedRepertoireToolTests(unittest.TestCase):
    def test_episode_results_group_step_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "steps.csv"
            _write_csv(
                path,
                [
                    {"opponent": "OP5", "mode": "fixed_z", "fixed_z_id": "0", "episode_idx": "0", "step": "0", "blue_score": "0", "red_score": "0"},
                    {"opponent": "OP5", "mode": "fixed_z", "fixed_z_id": "0", "episode_idx": "0", "step": "1", "blue_score": "1", "red_score": "0"},
                ],
            )
            rows = build_episode_results(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["blue_win"], 1)
        self.assertEqual(rows[0]["decision_steps"], 2)

    def test_behavior_matrix_and_pairwise_distances(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "by_z.csv"
            base = {
                "opponent": "OP5",
                "mode": "fixed_z",
                "n_steps": "10",
                "blue_win_rate": "0.5",
                "n_episodes_touched": "2",
                "blue_scores_per_episode": "1.0",
                "red_scores_per_episode": "0.0",
            }
            row0 = dict(base, z="0")
            row1 = dict(base, z="1")
            for name in BEHAVIOR_TELEMETRY_NAMES:
                row0[f"{name}_mean"] = "0.0"
                row1[f"{name}_mean"] = "1.0"
            _write_csv(path, [row0, row1])
            matrix = build_behavior_matrix(path)
            pairwise = build_pairwise_distances(matrix)
            summary = build_summary(path)
        self.assertEqual(len(matrix), 2)
        self.assertEqual(len(pairwise), 1)
        self.assertTrue(math.isfinite(float(pairwise[0]["pairwise_behavior_distance"])))
        report = build_readiness_report(
            checkpoint=Path("dummy.zip"),
            opponents=["OP5"],
            behavior_matrix=matrix,
            pairwise=pairwise,
            summary_rows=summary,
            behavior_margin=0.01,
            competence_floor=0.2,
        )
        self.assertTrue(report["readiness"]["different_z_behaviors"])
        self.assertTrue(report["readiness"]["competence_floor_pass"])

    def test_map_layout_resolves_from_metrics_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            metrics = root / "run_tag_metrics.csv"
            metrics.write_text("update\n1\n", encoding="utf-8")
            (root / "run_tag_run_config.json").write_text(
                json.dumps({"resolved_ppo_config": {"map_layout": "map_b_split_lane_v2"}}),
                encoding="utf-8",
            )

            self.assertEqual(_resolve_map_layout(None, metrics), "map_b_split_lane_v2")
            self.assertEqual(_resolve_map_layout("map_a_open", metrics), "map_a_open")


class V6I5StateConditionedBranchToolTests(unittest.TestCase):
    class _Core:
        def __init__(self) -> None:
            self.max_dist = 100.0
            self.blue_score = torch.tensor([1], dtype=torch.long)
            self.red_score = torch.tensor([2], dtype=torch.long)
            self.blue_carrying = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
            self.red_carrying = torch.tensor([[1, 0, 0, 0]], dtype=torch.long)
            self.blue_alive = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
            self.red_alive = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
            self.blue_x = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
            self.blue_y = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
            self.red_x = torch.tensor([[88.0, 12.0, 80.0, 70.0]])
            self.red_y = torch.tensor([[88.0, 12.0, 80.0, 70.0]])
            self.blue_flag_pos = torch.tensor([[10.0, 10.0]])
            self.red_flag_pos = torch.tensor([[90.0, 90.0]])

    def test_state_bucket_labels_detect_enemy_pressure_and_trailing_late(self) -> None:
        core = self._Core()
        labels = state_bucket_labels(core, step=1200, max_steps=1500)
        self.assertIn("enemy_carrying_team_flag", labels)
        self.assertIn("trailing_late", labels)
        self.assertIn("high_enemy_pressure", labels)
        self.assertIn("carrier_near_capture_zone", labels)

    def test_advantage_and_oracle_compare_against_z3(self) -> None:
        rows = []
        for z, terminal_delta in [(0, 1), (1, -1), (2, 3), (3, 2)]:
            rows.append(
                {
                    "opponent": "OP6",
                    "bucket": "team_carrying_enemy_flag",
                    "forced_z": z,
                    "short_return": float(terminal_delta),
                    "short_score_delta": terminal_delta,
                    "terminal_score_delta": terminal_delta,
                    "terminal_blue_won": int(terminal_delta > 0),
                }
            )
        adv = compute_advantage_rows(rows, baseline_z=3)
        z2 = next(r for r in adv if int(r["forced_z"]) == 2)
        self.assertEqual(z2["is_bucket_best_z"], 1)
        self.assertAlmostEqual(z2["latent_advantage_vs_z3"], 1.0)
        oracle = compute_oracle_rows(adv, baseline_z=3)
        self.assertEqual(len(oracle), 1)
        self.assertEqual(oracle[0]["best_z"], 2)
        self.assertAlmostEqual(oracle[0]["oracle_minus_z3"], 1.0)


if __name__ == "__main__":
    unittest.main()
