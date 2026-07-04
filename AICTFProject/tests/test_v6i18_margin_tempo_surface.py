"""Pinning tests for V6I18 margin/tempo consequence surface."""

from __future__ import annotations

import dataclasses
import unittest

import torch

from gpu_env._config import RewardConfig
from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.reward_composition import _compose_training_reward_components
from rl.presets import apply_preset
from rl.training.config_validation import normalize_and_validate_training_config
from rl.training.env_factory import _gpu_env_reward_kwargs


class V6i18MarginTempoPresetTests(unittest.TestCase):
    def test_aliases_resolve_to_margin_tempo_arm(self) -> None:
        aliases = [
            "v6i18",
            "v6i18_margin_tempo_surface_diagnostic",
            "v6i18_margin_tempo_surface",
            "latent_v6i18_margin_tempo_surface_diagnostic",
            "plan_faithful_latent_v6i18_margin_tempo_surface_diagnostic",
        ]
        base = dataclasses.asdict(apply_preset(PPOConfig(), aliases[0]))
        for alias in aliases[1:]:
            self.assertEqual(base, dataclasses.asdict(apply_preset(PPOConfig(), alias)))

    def test_v6i18_is_exact_consequence_surface_diff_over_v6i17(self) -> None:
        parent = dataclasses.asdict(apply_preset(PPOConfig(), "v6i17"))
        cfg_obj = apply_preset(PPOConfig(), "v6i18")
        cfg = dataclasses.asdict(cfg_obj)
        changed = {k for k in parent if parent[k] != cfg[k]}
        self.assertEqual(
            changed,
            {
                "experiment_id",
                "max_decision_steps",
                "env_stalemate_max_steps",
                "env_surface_score_margin_coef",
                "env_surface_blue_capture_tempo_bonus",
                "env_surface_red_flag_touch_penalty",
                "env_surface_red_carrier_progress_penalty",
                "env_surface_blue_near_cap_bonus",
                "run_tag",
            },
        )
        self.assertEqual(cfg_obj.experiment_id, "v6i18")
        self.assertEqual(cfg_obj.max_decision_steps, 240)
        self.assertEqual(cfg_obj.env_stalemate_max_steps, 80)
        self.assertAlmostEqual(float(cfg_obj.env_surface_score_margin_coef), 0.15)
        self.assertAlmostEqual(float(cfg_obj.env_surface_blue_capture_tempo_bonus), 0.25)
        self.assertAlmostEqual(float(cfg_obj.env_surface_red_flag_touch_penalty), 0.20)
        self.assertAlmostEqual(float(cfg_obj.env_surface_red_carrier_progress_penalty), 0.025)
        self.assertAlmostEqual(float(cfg_obj.env_surface_blue_near_cap_bonus), 0.015)

    def test_v6i18_preserves_v6i17_specialist_scaffold(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i18")
        normalize_and_validate_training_config(cfg)
        self.assertEqual(tuple(str(x).upper() for x in cfg.opponent_pool), ("OP8", "OP9", "OP10", "OP11", "OP12"))
        self.assertTrue(cfg.latent_contract_specialist_enabled)
        self.assertEqual(cfg.latent_contract_specialist_variant, "sharp")
        self.assertAlmostEqual(float(cfg.latent_contract_specialist_coef), 0.75)
        self.assertTrue(cfg.enable_latent_z_residual)
        self.assertTrue(cfg.latent_actor_z_adapter_enabled)
        self.assertEqual(cfg.latent_assignment_mode, "balanced_episode")
        self.assertFalse(cfg.train_router_when_forced)
        self.assertFalse(cfg.train_router_critic_when_forced)
        self.assertEqual(cfg.v6i9_training_stage, "repertoire")

    def test_surface_reward_overrides_forward_to_gpu_config(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i18")
        kwargs = _gpu_env_reward_kwargs(cfg)
        self.assertEqual(kwargs["stalemate_max_steps"], 80)
        self.assertAlmostEqual(kwargs["surface_score_margin_coef"], 0.15)
        self.assertAlmostEqual(kwargs["surface_blue_capture_tempo_bonus"], 0.25)
        self.assertAlmostEqual(kwargs["surface_red_flag_touch_penalty"], 0.20)
        self.assertAlmostEqual(kwargs["surface_red_carrier_progress_penalty"], 0.025)
        self.assertAlmostEqual(kwargs["surface_blue_near_cap_bonus"], 0.015)


class SurfaceRewardDefaultTests(unittest.TestCase):
    def test_new_reward_fields_are_default_off(self) -> None:
        cfg = RewardConfig()
        self.assertEqual(cfg.surface_score_margin_coef, 0.0)
        self.assertEqual(cfg.surface_blue_capture_tempo_bonus, 0.0)
        self.assertEqual(cfg.surface_red_flag_touch_penalty, 0.0)
        self.assertEqual(cfg.surface_red_carrier_progress_penalty, 0.0)
        self.assertEqual(cfg.surface_blue_near_cap_bonus, 0.0)

    def test_trainer_reward_composition_accepts_surface_pressure_inside_offense_channel(self) -> None:
        comp = {
            "reward_terminal": torch.tensor([0.0]),
            "reward_sparse": torch.tensor([0.0]),
            "reward_failure": torch.tensor([0.0]),
            "reward_offense": torch.tensor([0.25]),
            "reward_pbrs": torch.tensor([0.0]),
            "reward_team": torch.tensor([0.0]),
        }
        out = _compose_training_reward_components(
            comp,
            dense_weight=0.25,
            reward_scale=4.0,
            reward_clip=1.0,
            shaping_coef=1.0,
        )
        self.assertGreater(float(out["reward_total"][0]), 0.0)


if __name__ == "__main__":
    unittest.main()
