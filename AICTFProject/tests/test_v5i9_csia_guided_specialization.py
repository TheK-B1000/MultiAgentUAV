"""Focused tests for the v5i9 CSIA-guided specialization extension."""

from __future__ import annotations

import dataclasses
import io
import unittest
from contextlib import redirect_stdout

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.csv_writers import _update_fieldnames
from rl.custom_ppo.schedules import resolve_latent_forced_z_frac
from rl.presets import apply_preset
from rl.presets.plan_faithful import (
    apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure,
    apply_plan_faithful_latent_v5i9_csia_guided_specialization,
)
from rl.training.banner import _maybe_print_paper_faithful_audit


V5I9_ALIASES = (
    "v5i9",
    "v5i9_csia",
    "v5i9_csia_guided_specialization",
    "latent_v5i9_csia",
    "latent_v5i9_csia_guided_specialization",
    "plan_faithful_latent_v5i9_csia",
    "plan_faithful_latent_v5i9_csia_guided_specialization",
)


def _resolved(name: str) -> PPOConfig:
    return apply_preset(PPOConfig(), name)


class V5i9PresetInheritanceTests(unittest.TestCase):
    def test_v5i9_diff_vs_v5i8_is_csia_and_tag_only(self) -> None:
        v5i8 = dataclasses.asdict(
            apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure(PPOConfig())
        )
        v5i9 = dataclasses.asdict(
            apply_plan_faithful_latent_v5i9_csia_guided_specialization(
                PPOConfig()
            )
        )
        diffs = {
            k: (v5i8.get(k), v5i9.get(k))
            for k in set(v5i8) | set(v5i9)
            if v5i8.get(k) != v5i9.get(k)
        }
        self.assertEqual(set(diffs), {"csia_enabled", "csia_reward_coef", "run_tag"})
        self.assertEqual(diffs["csia_enabled"], (False, True))
        self.assertEqual(diffs["csia_reward_coef"], (0.0, 0.02))

    def test_v5i9_keeps_v5i8_latent_and_map_contract(self) -> None:
        cfg = _resolved("v5i9")
        parent = _resolved("v5i8")
        for key in (
            "map_layout",
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

    def test_v5i9_is_not_labeled_paper_or_summer_faithful(self) -> None:
        cfg = _resolved("v5i9")
        self.assertIn("v5i9_csia_guided_specialization", cfg.run_tag)
        self.assertNotIn("paper_faithful", cfg.run_tag)
        self.assertNotIn("summer_faithful", cfg.run_tag)

        stream = io.StringIO()
        with redirect_stdout(stream):
            _maybe_print_paper_faithful_audit(cfg)
        self.assertEqual(stream.getvalue(), "")

    def test_v5i9_forced_z_coverage_curriculum_stays_off(self) -> None:
        cfg = _resolved("v5i9")
        for step in (0, 10_000, 200_000, 500_000, 1_000_000):
            self.assertAlmostEqual(
                resolve_latent_forced_z_frac(cfg, global_step=step),
                0.0,
                places=8,
            )

    def test_v5i9_metrics_header_contains_csia_fields(self) -> None:
        fields = _update_fieldnames(use_latent_strategy=True, latent_k=4)
        for name in (
            "reward_csia_mean",
            "csia_interaction_strength",
            "centered_advantage_matrix",
            "oracle_best_z_per_opponent",
            "router_oracle_gap",
            "routing_gain",
            "gate_A_pass",
            "gate_B_pass",
            "gate_C_pass",
            "csia_bonus_active",
        ):
            self.assertIn(name, fields)


class V5i9AliasSnapshotTests(unittest.TestCase):
    def test_all_aliases_resolve_to_identical_config(self) -> None:
        baseline = dataclasses.asdict(_resolved(V5I9_ALIASES[0]))
        for name in V5I9_ALIASES[1:]:
            current = dataclasses.asdict(_resolved(name))
            self.assertEqual(current, baseline, f"alias {name!r} must match v5i9")


if __name__ == "__main__":
    unittest.main()
