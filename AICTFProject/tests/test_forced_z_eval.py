"""Tests for canonical forced-z episode I/O and analysis."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.forced_z_eval.analysis.complementarity import build_complementarity_report
from experiments.forced_z_eval.analysis.stage_c import build_stage_c_report
from experiments.forced_z_eval.io import load_episode_results, write_run_artifacts
from experiments.forced_z_eval.protocol import DEFAULT_LATENTS, ForcedZProtocol


def _fake_ep(success: int, margin: int, ret: float, phase: str = "attack") -> dict:
    return {
        "success": success,
        "win_margin": margin,
        "return": ret,
        "episode_start_phase": phase,
        "behavior_num_attackers": 1.0,
        "behavior_num_defenders": 1.0,
        "behavior_team_spread": 0.2,
        "behavior_carrier_escort_count": 0.0,
        "behavior_n_intercept_near_enemy_carrier": 0.1,
        "behavior_avg_blue_to_enemy_flag": 0.3,
        "behavior_avg_blue_to_own_flag": 0.4,
        "behavior_intercept_pressure": 0.2,
        "behavior_attack_defense_ratio": 0.5,
        "behavior_num_go_to": 0.0,
        "behavior_nearest_blue_to_carrier": 0.0,
        "behavior_nearest_blue_to_enemy_carrier": 0.0,
        "behavior_defense_pressure": 0.0,
    }


class ForcedZEvalIOTests(unittest.TestCase):
    def test_roundtrip_and_oracle_gap(self) -> None:
        protocol = ForcedZProtocol(
            checkpoint="fake.zip",
            opponents=("OP8",),
            maps=("map_b",),
            latents=DEFAULT_LATENTS,
            episodes_per_cell=2,
            base_seed=42,
        )
        cells = {
            ("OP8", 0, "map_b"): [_fake_ep(1, 1, 1.0), _fake_ep(0, -1, -2.0)],
            ("OP8", 1, "map_b"): [_fake_ep(1, 2, 2.0), _fake_ep(1, 0, 0.5)],
            ("OP8", 2, "map_b"): [_fake_ep(0, -2, -3.0), _fake_ep(1, 1, 1.5)],
            ("OP8", 3, "map_b"): [_fake_ep(1, 0, 0.2), _fake_ep(0, 0, -0.5)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_run_artifacts(run_dir, protocol=protocol, cells=cells)
            loaded_protocol, loaded_cells = load_episode_results(run_dir)
            self.assertEqual(loaded_protocol.base_seed, 42)
            self.assertEqual(len(loaded_cells[("OP8", 0, "map_b")]), 2)
            stage_c = build_stage_c_report(
                loaded_cells,
                opponents=["OP8"],
                maps=["map_b"],
                latents=DEFAULT_LATENTS,
            )
            comp = build_complementarity_report(
                loaded_cells,
                opponents=["OP8"],
                maps=["map_b"],
                latents=DEFAULT_LATENTS,
                metric="return",
            )
            self.assertGreater(comp["oracle_gap"], 0.0)
            self.assertEqual(stage_c["oracle_wr"], 1.0)
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["deterministic_actions"])
            self.assertEqual(manifest["max_decision_steps"], 400)


if __name__ == "__main__":
    unittest.main()
