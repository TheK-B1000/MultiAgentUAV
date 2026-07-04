"""Pinning tests for V6I20 asymmetric consequence surface."""

from __future__ import annotations

import dataclasses
import unittest

from gpu_env._maps import MAP_B_SPLIT_LANE, MAP_B_SPLIT_LANE_V2
from rl.config.ppo_config import PPOConfig
from rl.presets import apply_preset
from rl.training.config_validation import normalize_and_validate_training_config
from rl.training.env_factory import _gpu_env_reward_kwargs


class V6i20AsymmetryHandicapPresetTests(unittest.TestCase):
    def test_aliases_resolve_to_asymmetry_arm(self) -> None:
        aliases = [
            "v6i20",
            "v6i20_asymmetry_handicap_surface_diagnostic",
            "v6i20_asymmetry_handicap_surface",
            "v6i20_handicap_surface",
            "latent_v6i20_asymmetry_handicap_surface_diagnostic",
            "plan_faithful_latent_v6i20_asymmetry_handicap_surface_diagnostic",
        ]
        base = dataclasses.asdict(apply_preset(PPOConfig(), aliases[0]))
        for alias in aliases[1:]:
            self.assertEqual(base, dataclasses.asdict(apply_preset(PPOConfig(), alias)))

    def test_v6i20_is_exact_asymmetry_surface_diff_over_v6i19(self) -> None:
        parent_cfg = apply_preset(PPOConfig(), "v6i19")
        normalize_and_validate_training_config(parent_cfg)
        parent = dataclasses.asdict(parent_cfg)
        cfg_obj = apply_preset(PPOConfig(), "v6i20")
        normalize_and_validate_training_config(cfg_obj)
        cfg = dataclasses.asdict(cfg_obj)
        changed = {k for k in parent if parent[k] != cfg[k]}
        self.assertEqual(
            changed,
            {
                "experiment_id",
                "env_surface_blue_capture_tempo_bonus",
                "env_surface_red_flag_touch_penalty",
                "env_surface_red_carrier_progress_penalty",
                "env_surface_blue_near_cap_bonus",
                "run_tag",
            },
        )
        self.assertEqual(cfg_obj.experiment_id, "v6i20")
        self.assertAlmostEqual(float(cfg_obj.env_surface_blue_capture_tempo_bonus), 0.45)
        self.assertAlmostEqual(float(cfg_obj.env_surface_red_flag_touch_penalty), 0.50)
        self.assertAlmostEqual(float(cfg_obj.env_surface_red_carrier_progress_penalty), 0.075)
        self.assertAlmostEqual(float(cfg_obj.env_surface_blue_near_cap_bonus), 0.035)
        self.assertAlmostEqual(float(cfg_obj.env_surface_score_margin_coef), 0.15)

    def test_v6i20_preserves_v6i19_specialist_scaffold(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i20")
        normalize_and_validate_training_config(cfg)
        self.assertEqual(tuple(str(x).upper() for x in cfg.opponent_pool), ("OP8", "OP9", "OP10", "OP11", "OP12"))
        self.assertEqual(tuple(cfg.map_pool), (MAP_B_SPLIT_LANE, MAP_B_SPLIT_LANE_V2))
        self.assertEqual(cfg.map_layout, MAP_B_SPLIT_LANE)
        self.assertEqual(cfg.max_decision_steps, 240)
        self.assertEqual(cfg.env_stalemate_max_steps, 80)
        self.assertTrue(cfg.latent_contract_specialist_enabled)
        self.assertEqual(cfg.latent_contract_specialist_variant, "sharp")
        self.assertAlmostEqual(float(cfg.latent_contract_specialist_coef), 0.75)
        self.assertTrue(cfg.enable_latent_z_residual)
        self.assertTrue(cfg.latent_actor_z_adapter_enabled)
        self.assertEqual(cfg.latent_assignment_mode, "balanced_episode")
        self.assertFalse(cfg.train_router_when_forced)
        self.assertFalse(cfg.train_router_critic_when_forced)
        self.assertEqual(cfg.v6i9_training_stage, "repertoire")

    def test_asymmetry_reward_overrides_forward_to_gpu_config(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i20")
        kwargs = _gpu_env_reward_kwargs(cfg)
        self.assertEqual(kwargs["stalemate_max_steps"], 80)
        self.assertAlmostEqual(kwargs["surface_score_margin_coef"], 0.15)
        self.assertAlmostEqual(kwargs["surface_blue_capture_tempo_bonus"], 0.45)
        self.assertAlmostEqual(kwargs["surface_red_flag_touch_penalty"], 0.50)
        self.assertAlmostEqual(kwargs["surface_red_carrier_progress_penalty"], 0.075)
        self.assertAlmostEqual(kwargs["surface_blue_near_cap_bonus"], 0.035)


if __name__ == "__main__":
    unittest.main()
