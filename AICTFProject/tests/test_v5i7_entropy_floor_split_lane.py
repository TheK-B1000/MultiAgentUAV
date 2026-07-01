"""Focused tests for the v5i7 entropy-floor split-lane preset."""

from __future__ import annotations

import dataclasses
import io
import unittest
from types import SimpleNamespace
from contextlib import redirect_stdout

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.csv_writers import _update_fieldnames
from rl.custom_ppo.ppo_updater import _populate_main_loop_qphi_telemetry
from rl.custom_ppo.schedules import resolve_latent_forced_z_frac
from rl.presets import apply_preset
from rl.presets.plan_faithful import (
    apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor,
    apply_plan_faithful_latent_v5i7_entropy_floor_split_lane,
)
from rl.training.banner import _maybe_print_paper_faithful_audit


V5I7_ALIASES = (
    "v5i7",
    "v5i7_split_lane",
    "v5i7_entropy_floor_split_lane",
    "v5i7_summer_faithful_entropy_floor_split_lane",
    "v5i7_summer_faithful_split_lane",
    "latent_v5i7_split_lane",
    "latent_v5i7_entropy_floor_split_lane",
    "latent_v5i7_summer_faithful_entropy_floor_split_lane",
    "latent_v5i7_summer_faithful_split_lane",
    "plan_faithful_latent_v5i7_split_lane",
    "plan_faithful_latent_v5i7_entropy_floor_split_lane",
    "plan_faithful_latent_v5i7_summer_faithful_entropy_floor_split_lane",
    "plan_faithful_latent_v5i7_summer_faithful_split_lane",
)


def _resolved(name: str) -> PPOConfig:
    return apply_preset(PPOConfig(), name)


class V5i7PresetInheritanceTests(unittest.TestCase):
    def test_v5i7_diff_vs_v5i5_is_map_layout_and_tag_only(self) -> None:
        v5i5 = dataclasses.asdict(
            apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor(
                PPOConfig()
            )
        )
        v5i7 = dataclasses.asdict(
            apply_plan_faithful_latent_v5i7_entropy_floor_split_lane(PPOConfig())
        )
        diffs = {
            k: (v5i5.get(k), v5i7.get(k))
            for k in set(v5i5) | set(v5i7)
            if v5i5.get(k) != v5i7.get(k)
        }
        self.assertEqual(set(diffs), {"map_layout", "run_tag"})
        self.assertEqual(diffs["map_layout"], ("map_a_open", "map_b_split_lane"))

    def test_v5i7_keeps_v5i5_latent_contract(self) -> None:
        cfg = _resolved("v5i7")
        self.assertTrue(bool(cfg.use_latent_strategy))
        self.assertEqual(int(cfg.latent_k), 4)
        self.assertEqual(int(cfg.latent_z_embed_dim), 16)
        self.assertEqual(str(cfg.latent_entropy_mode), "conditional")
        self.assertEqual(str(cfg.latent_entropy_objective), "maximize")
        self.assertAlmostEqual(float(cfg.latent_lam_h_start), 0.003)
        self.assertAlmostEqual(float(cfg.latent_lam_h_end), 0.001)
        self.assertAlmostEqual(float(cfg.latent_lam_p), 0.03)
        self.assertAlmostEqual(float(cfg.latent_strategy_ppo_coef), 0.10)
        self.assertEqual(int(cfg.latent_resample_every_n), 64)
        self.assertFalse(bool(cfg.latent_resample_on_flag))
        self.assertFalse(bool(cfg.enable_actor_z_film))
        self.assertFalse(bool(cfg.latent_actor_z_adapter_enabled))
        self.assertFalse(bool(cfg.latent_actor_z_onehot_enabled))
        self.assertFalse(bool(cfg.latent_episode_strategy_ppo))
        self.assertIsNone(cfg.latent_episode_strategy_lr)
        self.assertFalse(bool(cfg.latent_arc_credit_enabled))
        self.assertFalse(bool(cfg.latent_router_distill_enabled))
        self.assertFalse(bool(cfg.latent_strategy_aux_return_head))
        self.assertAlmostEqual(float(cfg.latent_strategy_aux_predict_phase_coef), 0.0)

    def test_v5i7_uses_split_lane_map_and_entropy_floor_tag(self) -> None:
        cfg = _resolved("v5i7")
        self.assertEqual(str(cfg.map_layout), "map_b_split_lane")
        self.assertIn("v5i7_summer_faithful_entropy_floor_split_lane", cfg.run_tag)
        self.assertIn("OP5_OP6_OP7", cfg.run_tag)
        self.assertIn("_1m_", cfg.run_tag)
        self.assertIn("summer_faithful", cfg.run_tag)
        self.assertNotIn("marginal_entropy", cfg.run_tag)
        self.assertNotIn("_2m_", cfg.run_tag)

    def test_forced_z_resolves_to_zero_at_every_step(self) -> None:
        cfg = _resolved("v5i7")
        for step in (0, 10_000, 200_000, 500_000, 1_000_000):
            self.assertAlmostEqual(
                resolve_latent_forced_z_frac(cfg, global_step=step),
                0.0,
                places=8,
            )

    def test_audit_banner_detects_v5i7_as_conditional_entropy_family(self) -> None:
        cfg = _resolved("v5i7")
        stream = io.StringIO()
        with redirect_stdout(stream):
            _maybe_print_paper_faithful_audit(cfg)
        out = stream.getvalue()
        self.assertIn("[PPO] v5i7 paper-faithful audit:", out)
        self.assertIn(
            "entropy maximization: ON (mode=conditional, aggregation=per-state",
            out,
        )
        self.assertNotIn("audit WARNING", out)


class V5i7AliasSnapshotTests(unittest.TestCase):
    def test_all_aliases_resolve_to_identical_config(self) -> None:
        baseline = dataclasses.asdict(_resolved(V5I7_ALIASES[0]))
        for name in V5I7_ALIASES[1:]:
            current = dataclasses.asdict(_resolved(name))
            self.assertEqual(current, baseline, f"alias {name!r} must match v5i7")


class V5i7MainLoopQPhiTelemetryTests(unittest.TestCase):
    def test_metrics_header_contains_main_loop_qphi_fields(self) -> None:
        fields = _update_fieldnames(use_latent_strategy=True, latent_k=4)
        self.assertIn("main_loop_q_phi_train_active", fields)
        self.assertIn("main_loop_q_phi_grad_norm", fields)

    def test_main_loop_qphi_telemetry_backfills_legacy_smoke_fields(self) -> None:
        cfg = _resolved("v5i7")
        row = {
            "strategy_grad_norm": 0.125,
            "latent_q_phi_train_active": 0.0,
            "q_phi_grad_norm": 0.0,
            "q_phi_strategy_encoder_grad_norm": 0.0,
        }
        hparams = SimpleNamespace(
            use_latent_strategy=True,
            fixed_latent_strategy=False,
            latent_kl_consecutive=0.0,
        )
        runtime = SimpleNamespace(latent_router_optimizer=None)

        _populate_main_loop_qphi_telemetry(
            row,
            cfg=cfg,
            hparams=hparams,
            runtime=runtime,
            latent_lam_h=0.001,
        )

        self.assertEqual(row["main_loop_q_phi_train_active"], 1.0)
        self.assertAlmostEqual(row["main_loop_q_phi_grad_norm"], 0.125)
        self.assertEqual(row["latent_q_phi_train_active"], 1.0)
        self.assertAlmostEqual(row["q_phi_grad_norm"], 0.125)
        self.assertAlmostEqual(row["q_phi_strategy_encoder_grad_norm"], 0.125)

    def test_main_loop_qphi_telemetry_preserves_extension_grad_fields(self) -> None:
        cfg = _resolved("v5i7")
        row = {
            "strategy_grad_norm": 0.125,
            "latent_q_phi_train_active": 1.0,
            "q_phi_grad_norm": 0.5,
            "q_phi_strategy_encoder_grad_norm": 0.4,
        }
        hparams = SimpleNamespace(
            use_latent_strategy=True,
            fixed_latent_strategy=False,
            latent_kl_consecutive=0.0,
        )
        runtime = SimpleNamespace(latent_router_optimizer=None)

        _populate_main_loop_qphi_telemetry(
            row,
            cfg=cfg,
            hparams=hparams,
            runtime=runtime,
            latent_lam_h=0.001,
        )

        self.assertAlmostEqual(row["main_loop_q_phi_grad_norm"], 0.125)
        self.assertAlmostEqual(row["q_phi_grad_norm"], 0.5)
        self.assertAlmostEqual(row["q_phi_strategy_encoder_grad_norm"], 0.4)


if __name__ == "__main__":
    unittest.main()
