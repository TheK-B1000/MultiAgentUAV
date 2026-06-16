"""Focused tests for the v5i8 split-lane v2 task-pressure preset."""

from __future__ import annotations

import dataclasses
import io
import unittest
from contextlib import redirect_stdout

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.schedules import resolve_latent_forced_z_frac
from rl.presets import apply_preset
from rl.presets.plan_faithful import (
    apply_plan_faithful_latent_v5i7_entropy_floor_split_lane,
    apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure,
)
from rl.training.banner import _maybe_print_paper_faithful_audit


V5I8_ALIASES = (
    "v5i8",
    "v5i8_split_lane_v2",
    "v5i8_split_lane_v2_task_pressure",
    "v5i8_summer_faithful_split_lane_v2",
    "latent_v5i8_split_lane_v2",
    "latent_v5i8_split_lane_v2_task_pressure",
    "latent_v5i8_summer_faithful_split_lane_v2",
    "plan_faithful_latent_v5i8_split_lane_v2",
    "plan_faithful_latent_v5i8_split_lane_v2_task_pressure",
    "plan_faithful_latent_v5i8_summer_faithful_split_lane_v2",
)


def _resolved(name: str) -> PPOConfig:
    return apply_preset(PPOConfig(), name)


class V5i8PresetInheritanceTests(unittest.TestCase):
    def test_v5i8_diff_vs_v5i7_is_map_layout_and_tag_only(self) -> None:
        v5i7 = dataclasses.asdict(
            apply_plan_faithful_latent_v5i7_entropy_floor_split_lane(PPOConfig())
        )
        v5i8 = dataclasses.asdict(
            apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure(
                PPOConfig()
            )
        )
        diffs = {
            k: (v5i7.get(k), v5i8.get(k))
            for k in set(v5i7) | set(v5i8)
            if v5i7.get(k) != v5i8.get(k)
        }
        self.assertEqual(set(diffs), {"map_layout", "run_tag"})
        self.assertEqual(diffs["map_layout"], ("map_b_split_lane", "map_b_split_lane_v2"))

    def test_v5i8_keeps_v5i7_latent_contract(self) -> None:
        cfg = _resolved("v5i8")
        parent = _resolved("v5i7")
        for key in (
            "use_latent_strategy",
            "latent_k",
            "latent_z_embed_dim",
            "latent_entropy_mode",
            "latent_entropy_objective",
            "latent_lam_h_start",
            "latent_lam_h_end",
            "latent_lam_p",
            "latent_strategy_ppo_coef",
            "latent_resample_every_n",
            "latent_resample_on_flag",
            "enable_actor_z_film",
            "latent_actor_z_adapter_enabled",
            "latent_actor_z_onehot_enabled",
            "latent_episode_strategy_ppo",
            "latent_episode_strategy_lr",
            "latent_arc_credit_enabled",
            "latent_router_distill_enabled",
            "latent_strategy_aux_return_head",
            "latent_strategy_aux_predict_phase_coef",
        ):
            self.assertEqual(getattr(cfg, key), getattr(parent, key), key)

    def test_v5i8_uses_split_lane_v2_map_and_task_pressure_tag(self) -> None:
        cfg = _resolved("v5i8")
        self.assertEqual(str(cfg.map_layout), "map_b_split_lane_v2")
        self.assertIn("v5i8_summer_faithful_split_lane_v2_task_pressure", cfg.run_tag)
        self.assertIn("OP5_OP6_OP7", cfg.run_tag)
        self.assertIn("_1m_", cfg.run_tag)
        self.assertIn("summer_faithful", cfg.run_tag)
        self.assertNotIn("marginal_entropy", cfg.run_tag)
        self.assertNotIn("_2m_", cfg.run_tag)

    def test_forced_z_resolves_to_zero_at_every_step(self) -> None:
        cfg = _resolved("v5i8")
        for step in (0, 10_000, 200_000, 500_000, 1_000_000):
            self.assertAlmostEqual(
                resolve_latent_forced_z_frac(cfg, global_step=step),
                0.0,
                places=8,
            )

    def test_audit_banner_detects_v5i8_as_conditional_entropy_family(self) -> None:
        cfg = _resolved("v5i8")
        stream = io.StringIO()
        with redirect_stdout(stream):
            _maybe_print_paper_faithful_audit(cfg)
        out = stream.getvalue()
        self.assertIn("[PPO] v5i8 paper-faithful audit:", out)
        self.assertIn(
            "entropy maximization: ON (mode=conditional, aggregation=per-state",
            out,
        )
        self.assertNotIn("audit WARNING", out)


class V5i8AliasSnapshotTests(unittest.TestCase):
    def test_all_aliases_resolve_to_identical_config(self) -> None:
        baseline = dataclasses.asdict(_resolved(V5I8_ALIASES[0]))
        for name in V5I8_ALIASES[1:]:
            current = dataclasses.asdict(_resolved(name))
            self.assertEqual(current, baseline, f"alias {name!r} must match v5i8")


if __name__ == "__main__":
    unittest.main()
