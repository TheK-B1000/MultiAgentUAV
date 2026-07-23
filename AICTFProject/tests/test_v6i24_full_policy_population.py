"""Tests for V6I24 lean full-policy population diagnostic.

Pins:
  - Preset resolution / alias equality
  - Resolved-config diff vs v6i21j parent
  - Latent / adapter / PopulationTrainer flags off
  - Fixed cell-pressure helpers (both maps, normalized, complementary)
  - Freeze return-norm after load helper contract

Classification: DIAGNOSTIC (Path C fallback; not PAPER-FAITHFUL).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.return_normalization import ReturnNormalizer


def _resolve(name: str) -> PPOConfig:
    from rl.presets import PRESET_REGISTRY

    return PRESET_REGISTRY[name](PPOConfig())


class V6i24PresetResolutionTests(unittest.TestCase):
    ALIASES = [
        "v6i24_full_policy_population",
        "v6i24",
        "latent_v6i24_full_policy_population",
        "plan_faithful_latent_v6i24_full_policy_population",
    ]

    def test_all_aliases_resolve(self) -> None:
        for alias in self.ALIASES:
            with self.subTest(alias=alias):
                self.assertIsInstance(_resolve(alias), PPOConfig)

    def test_alias_equality(self) -> None:
        configs = [asdict(_resolve(a)) for a in self.ALIASES]
        for i, alias_i in enumerate(self.ALIASES):
            for j, alias_j in enumerate(self.ALIASES):
                if i >= j:
                    continue
                self.assertEqual(configs[i], configs[j], f"{alias_i} != {alias_j}")


class V6i24ConfigDiffTests(unittest.TestCase):
    def test_lean_population_flags(self) -> None:
        cfg = _resolve("v6i24")
        self.assertFalse(cfg.population_training_enabled)
        self.assertEqual(cfg.population_k, 4)
        self.assertEqual(cfg.population_pressure_rotation_interval, 0)
        self.assertEqual(cfg.population_round_robin_updates_per_cycle, 0)
        self.assertTrue(cfg.freeze_return_norm_after_load)
        self.assertTrue(cfg.opponent_randomize)
        self.assertEqual(cfg.v6i9_training_stage, "generalist")

    def test_latent_scaffold_frozen_z0(self) -> None:
        """Keep latent concat arch for warm-start; freeze z=0; no adapters."""
        cfg = _resolve("v6i24")
        self.assertTrue(cfg.use_latent_strategy)
        self.assertTrue(cfg.fixed_latent_strategy)
        self.assertEqual(cfg.fixed_latent_strategy_id, 0)
        self.assertEqual(cfg.latent_assignment_mode, "fixed")
        self.assertFalse(cfg.enable_latent_z_residual)
        self.assertEqual(cfg.latent_z_residual_alpha, 0.0)
        self.assertFalse(cfg.latent_population_birth_active_z_only)
        self.assertFalse(cfg.latent_population_birth_per_z_action_heads)
        self.assertEqual(cfg.latent_strategy_ppo_coef, 0.0)

    def test_experiment_id_and_run_tag(self) -> None:
        cfg = _resolve("v6i24")
        self.assertEqual(cfg.experiment_id, "v6i24")
        self.assertIn("v6i24", cfg.run_tag)
        self.assertIn("OP8", cfg.run_tag)
        self.assertIn("OP12", cfg.run_tag)

    def test_minimal_diff_vs_v6i21j(self) -> None:
        v6i24 = asdict(_resolve("v6i24"))
        v6i21j = asdict(_resolve("v6i21j"))
        expected = {
            "enable_latent_z_residual",
            "fixed_latent_strategy",
            "freeze_return_norm_after_load",
            "latent_assignment_mode",
            "latent_lam_h_end",
            "latent_lam_h_start",
            "latent_strategy_ppo_coef",
            "opponent_randomize",
            "population_pressure_rotation_interval",
            "population_round_robin_updates_per_cycle",
            "v6i9_training_stage",
            "experiment_id",
            "run_tag",
        }
        actual = {k for k in v6i24 if v6i24[k] != v6i21j.get(k)}
        self.assertEqual(
            actual,
            expected,
            f"Unexpected config diff vs v6i21j: {actual ^ expected}",
        )


class V6i24ForbiddenChannelsTests(unittest.TestCase):
    def test_no_episode_credit(self) -> None:
        cfg = _resolve("v6i24")
        self.assertEqual(cfg.latent_episode_strategy_coef, 0.0)
        self.assertFalse(cfg.latent_episode_strategy_ppo)

    def test_no_dedicated_router_optimizer(self) -> None:
        cfg = _resolve("v6i24")
        self.assertIsNone(getattr(cfg, "latent_episode_strategy_lr", None))


class V6i24PressureHelperTests(unittest.TestCase):
    def _toy_report(self, path: Path) -> None:
        cells = []
        for opp, wr, red in [
            ("OP8", 0.76, 1.2),
            ("OP9", 0.40, 2.5),
            ("OP10", 0.88, 0.8),
            ("OP11", 0.52, 2.0),
            ("OP12", 0.60, 1.5),
        ]:
            for mp in ("map_b", "map_b_split_lane_v2"):
                cells.append(
                    {
                        "opponent": opp,
                        "map": mp,
                        "episodes": 25,
                        "win_rate": wr,
                        "blue_score_mean": 3.0,
                        "red_score_mean": red,
                    }
                )
        path.write_text(json.dumps({"cells": cells}), encoding="utf-8")

    def test_pressures_normalized_both_maps_no_op3(self) -> None:
        from experiments.v6i24_population_config import build_member_pressures

        with tempfile.TemporaryDirectory() as td:
            report = Path(td) / "calib.json"
            self._toy_report(report)
            pressures = build_member_pressures(report_path=report)
        self.assertEqual(len(pressures), 4)
        self.assertEqual([p.label for p in pressures], [
            "balanced",
            "failure_cells",
            "high_variance",
            "complementary",
        ])
        for p in pressures:
            weights = [w for _, _, w in p.cell_weights]
            self.assertAlmostEqual(sum(weights), 1.0, places=6)
            maps = {m for _, m, _ in p.cell_weights}
            # map_b normalizes to map_b_split_lane
            self.assertTrue(any("split_lane" in m and "v2" not in m for m in maps) or "map_b_split_lane" in maps)
            self.assertTrue(any(m.endswith("v2") or "v2" in m for m in maps))
            map_mass = {}
            for _, m, w in p.cell_weights:
                map_mass[m] = map_mass.get(m, 0.0) + w
            for mass in map_mass.values():
                self.assertGreaterEqual(mass, 0.05 - 1e-9)
            opps = {o for o, _, _ in p.cell_weights}
            self.assertTrue(opps.isdisjoint({"OP3", "OP4", "OP5", "OP6", "OP7"}))


class V6i24ReturnNormFreezeTests(unittest.TestCase):
    def test_freeze_blocks_update_keeps_normalize(self) -> None:
        import torch

        rn = ReturnNormalizer(enabled=True)
        rn.mean = 2.0
        rn.var = 4.0
        rn.count = 100.0
        rn.freeze()
        before = (rn.mean, rn.var, rn.count)
        rn.update(torch.tensor([10.0, 12.0, 14.0]))
        self.assertEqual((rn.mean, rn.var, rn.count), before)
        out = rn.normalize(torch.tensor([4.0]))
        self.assertAlmostEqual(float(out.item()), (4.0 - 2.0) / 2.0, places=5)


class V6i24StrategicGateUnitTests(unittest.TestCase):
    def test_stricter_payoff_gates(self) -> None:
        from experiments.run_v6i24_population_eval_gates import evaluate_strategic_separation
        import numpy as np

        # Two clear specialist cells with margin >= 0.10 and distinct winners
        M = np.array(
            [
                [1.0, 0.2, 0.5, 0.5],
                [0.2, 1.0, 0.5, 0.5],
                [0.4, 0.4, 0.4, 0.4],
                [0.3, 0.3, 0.3, 0.3],
            ],
            dtype=np.float64,
        )
        result = evaluate_strategic_separation(
            M,
            ["c0", "c1", "c2", "c3"],
            ["balanced", "failure", "variance", "complement"],
        )
        self.assertTrue(result["gate_row_distance"])
        self.assertTrue(result["gate_different_best_with_margin"])
        self.assertTrue(result["gate_oracle_above_fixed"])


if __name__ == "__main__":
    unittest.main()
