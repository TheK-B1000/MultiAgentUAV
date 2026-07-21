"""Tests for canonical forced-z episode I/O and analysis."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from experiments.forced_z_eval.analysis.complementarity import build_complementarity_report
from experiments.forced_z_eval.analysis.stage_c import build_stage_c_report
from experiments.forced_z_eval.io import append_episode_rows, load_episode_results, write_manifest, write_run_artifacts
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

    def test_incremental_manifest_and_episode_append(self) -> None:
        protocol = ForcedZProtocol(
            checkpoint="fake.zip",
            opponents=("OP8",),
            maps=("map_b",),
            latents=DEFAULT_LATENTS,
            episodes_per_cell=2,
            base_seed=42,
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_manifest(
                run_dir,
                protocol=protocol,
                status="running",
                completed_conditions=[],
                episode_count=0,
            )
            append_episode_rows(
                run_dir,
                protocol=protocol,
                cells={("OP8", 0, "map_b"): [_fake_ep(1, 1, 1.0), _fake_ep(0, -1, -2.0)]},
            )
            write_manifest(
                run_dir,
                protocol=protocol,
                status="running",
                completed_conditions=[{"opponent": "OP8", "latent_z": 0, "map": "map_b", "episodes": 2}],
                episode_count=2,
            )

            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "running")
            self.assertEqual(manifest["episode_count"], 2)
            self.assertEqual(manifest["completed_condition_count"], 1)
            self.assertTrue((run_dir / "episode_results.csv").is_file())
            _, loaded_cells = load_episode_results(run_dir)
            self.assertEqual(len(loaded_cells[("OP8", 0, "map_b")]), 2)


class ForcedZEnvOverrideTests(unittest.TestCase):
    def test_v6i18_run_config_maps_surface_fields(self) -> None:
        from experiments.forced_z_eval.env_overrides import (
            env_reward_kwargs_from_resolved_config,
            find_run_config_for_checkpoint,
            resolve_forced_z_env_overrides,
        )

        ckpt = (
            PROJECT_ROOT
            / "artifacts/v6i18_margin_tempo_surface_5u_seed1/final_v6i18_margin_tempo_surface_5u_seed1_2v2.zip"
        )
        if not ckpt.is_file():
            self.skipTest("v6i18 artifact not present")
        run_cfg = find_run_config_for_checkpoint(ckpt)
        self.assertIsNotNone(run_cfg)
        steps, env_kwargs, source = resolve_forced_z_env_overrides(
            checkpoint=str(ckpt),
            inherit_training_config=True,
        )
        self.assertEqual(steps, 240)
        self.assertAlmostEqual(env_kwargs["surface_score_margin_coef"], 0.15)
        self.assertEqual(env_kwargs["stalemate_max_steps"], 80)
        self.assertIsNotNone(source)

        flat = {
            "max_decision_steps": 240,
            "env_surface_score_margin_coef": 0.15,
            "env_stalemate_max_steps": 80,
        }
        self.assertEqual(
            env_reward_kwargs_from_resolved_config(flat)["surface_score_margin_coef"],
            0.15,
        )


if __name__ == "__main__":
    unittest.main()
