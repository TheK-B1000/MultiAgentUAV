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
        self.assertEqual(cfg.recurrent_selector_hidden_dim, 0)

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
            "recurrent_selector_hidden_dim",
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


class V6i24SharedCoreTests(unittest.TestCase):
    def test_filter_keeps_trunk_discards_adapters(self) -> None:
        from experiments.v6i24_shared_core import (
            filter_shared_core_state_dict,
            is_shared_core_parameter,
        )
        import torch

        self.assertTrue(is_shared_core_parameter("actor_cnn.conv.0.weight"))
        self.assertTrue(is_shared_core_parameter("latent_actor.body.0.weight"))
        self.assertTrue(is_shared_core_parameter("latent_actor.action_head.weight"))
        self.assertTrue(is_shared_core_parameter("critic.net.0.weight"))
        self.assertTrue(is_shared_core_parameter("latent_actor.strategy_embedding.weight"))
        self.assertFalse(is_shared_core_parameter("latent_actor.latent_adapters.0.weight"))
        self.assertFalse(is_shared_core_parameter("latent_actor.latent_action_heads.1.weight"))
        self.assertFalse(is_shared_core_parameter("strategy_encoder.net.0.weight"))
        self.assertFalse(is_shared_core_parameter("selector_gru.cell.weight_ih"))

        src = {
            "actor_cnn.conv.0.weight": torch.zeros(1),
            "latent_actor.latent_adapters.0.weight": torch.zeros(1),
            "latent_actor.strategy_embedding.weight": torch.zeros(1),
            "only_in_source": torch.zeros(1),
        }
        tgt = {
            "actor_cnn.conv.0.weight": torch.zeros(1),
            "latent_actor.strategy_embedding.weight": torch.zeros(1),
            "latent_actor.latent_adapters.0.weight": torch.zeros(1),
        }
        shared, report = filter_shared_core_state_dict(src, tgt)
        self.assertIn("actor_cnn.conv.0.weight", shared)
        self.assertIn("latent_actor.strategy_embedding.weight", shared)
        self.assertNotIn("latent_actor.latent_adapters.0.weight", shared)
        self.assertTrue(report["kept_strategy_embedding"])

    def test_materialize_shared_core_from_v6i23(self) -> None:
        from experiments.v6i24_shared_core import materialize_shared_core_member_checkpoint

        donor = Path(
            "artifacts/v6i23_population_birth_5u_seed1/"
            "final_v6i23_population_birth_5u_seed1_2v2.zip"
        )
        if not donor.is_file():
            self.skipTest(f"donor missing: {donor}")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "init.zip"
            result = materialize_shared_core_member_checkpoint(
                source_checkpoint=donor,
                output_path=out,
                seed=1,
                mode="shared-core",
            )
            self.assertTrue(out.is_file())
            self.assertGreater(result.report["n_shared_loaded"], 10)
            self.assertTrue(result.report["kept_strategy_embedding"])
            # Adapters must have been ignored from donor
            ignored = set(result.report.get("ignored_latent_keys") or [])
            self.assertTrue(any("latent_adapters" in k for k in ignored))
            self.assertTrue(any("latent_action_heads" in k for k in ignored))


class V6i24StrategicGateUnitTests(unittest.TestCase):
    def test_stricter_payoff_gates_require_cross_fitted_ci(self) -> None:
        from experiments.run_v6i24_population_eval_gates import (
            evaluate_cross_fitted_teacher_oracle,
            evaluate_strategic_separation,
        )
        import numpy as np

        M = np.array(
            [
                [1.0, 0.2, 0.5, 0.5],
                [0.2, 1.0, 0.5, 0.5],
                [0.4, 0.4, 0.4, 0.4],
                [0.3, 0.3, 0.3, 0.3],
            ],
            dtype=np.float64,
        )
        # Without episode tensor, cross-fitted primary gate stays closed.
        result = evaluate_strategic_separation(
            M,
            ["c0", "c1", "c2", "c3"],
            ["balanced", "failure", "variance", "complement"],
        )
        self.assertTrue(result["gate_row_distance"])
        self.assertTrue(result["gate_different_best_with_margin"])
        self.assertFalse(result["gate_cross_fitted_oracle"])
        self.assertGreater(result["hindsight_oracle_gap"], 0.0)

        # Synthetic matched episodes: cell0 prefers π0, cell1 prefers π1.
        rng = np.random.default_rng(0)
        k, c, e = 4, 4, 40
        returns = np.zeros((k, c, e), dtype=np.float64)
        returns[0, 0, :] = 1.0
        returns[1, 1, :] = 1.0
        returns[0, 1, :] = 0.2
        returns[1, 0, :] = 0.2
        returns[2, :, :] = 0.4
        returns[3, :, :] = 0.3
        returns += rng.normal(0.0, 0.01, size=returns.shape)
        cross = evaluate_cross_fitted_teacher_oracle(
            returns,
            member_labels=["balanced", "failure", "variance", "complement"],
            context_labels=["c0", "c1", "c2", "c3"],
            test_frac=0.25,
            seed=0,
            n_bootstrap=200,
        )
        self.assertTrue(cross["gate_cross_fitted_oracle"])
        self.assertGreater(cross["delta"], 0.1)

        full = evaluate_strategic_separation(
            returns.mean(axis=2),
            ["c0", "c1", "c2", "c3"],
            ["balanced", "failure", "variance", "complement"],
            returns_kce=returns,
            seed=0,
            n_bootstrap=200,
        )
        self.assertTrue(full["gate_different_best_with_margin"])
        self.assertTrue(full["gate_cross_fitted_oracle"])
        self.assertTrue(full["gate_oracle_above_fixed"])


if __name__ == "__main__":
    unittest.main()
