"""Pinning tests for V6I22D stronger behavior-diversity repertoire birth."""
from __future__ import annotations

import dataclasses
import unittest

from rl.config.ppo_config import PPOConfig
from rl.presets import apply_preset
from rl.training.config_validation import normalize_and_validate_training_config


class V6i22DStrongBehaviorDiversityTests(unittest.TestCase):
    def test_primary_aliases_resolve_to_same_config(self) -> None:
        aliases = [
            "v6i22d",
            "v6i22d_strong_behavior_diversity",
            "v6i22d_behavior_diversity_coef010",
            "latent_v6i22d_strong_behavior_diversity",
            "plan_faithful_latent_v6i22d_strong_behavior_diversity",
        ]
        base = dataclasses.asdict(apply_preset(PPOConfig(), aliases[0]))
        for alias in aliases[1:]:
            self.assertEqual(base, dataclasses.asdict(apply_preset(PPOConfig(), alias)))

    def test_v6i22d_diff_vs_v6i22_is_behavior_contrast_only(self) -> None:
        parent_cfg = apply_preset(PPOConfig(), "v6i22")
        normalize_and_validate_training_config(parent_cfg)
        parent = dataclasses.asdict(parent_cfg)
        cfg_obj = apply_preset(PPOConfig(), "v6i22d")
        normalize_and_validate_training_config(cfg_obj)
        cfg = dataclasses.asdict(cfg_obj)
        changed = {k for k in parent if parent[k] != cfg[k]}
        self.assertEqual(
            changed,
            {
                "experiment_id",
                "latent_behavior_contrast_coef",
                "latent_behavior_contrast_margin",
                "run_tag",
            },
        )
        self.assertEqual(cfg_obj.experiment_id, "v6i22d_coef010")
        self.assertAlmostEqual(float(cfg_obj.latent_behavior_contrast_coef), 0.10)
        self.assertAlmostEqual(float(cfg_obj.latent_behavior_contrast_margin), 0.06)
        self.assertIn("v6i22d_strong_behavior_diversity_coef010", cfg_obj.run_tag)

    def test_v6i22d_stays_label_free_and_router_off(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i22d")
        normalize_and_validate_training_config(cfg)
        self.assertEqual(cfg.latent_assignment_mode, "balanced_episode")
        self.assertFalse(cfg.train_router_when_forced)
        self.assertFalse(cfg.train_router_critic_when_forced)
        self.assertFalse(cfg.latent_contract_specialist_enabled)
        self.assertEqual(float(cfg.latent_contract_specialist_coef), 0.0)
        self.assertEqual(cfg.latent_contract_specialist_variant, "base")
        self.assertFalse(cfg.latent_router_distill_enabled)
        self.assertEqual(float(cfg.latent_strategy_aux_predict_phase_coef), 0.0)
        self.assertFalse(cfg.latent_strategy_aux_return_head)
        self.assertEqual(float(cfg.latent_outcome_diversity_coef), 0.0)

    def test_sweep_arms_only_change_behavior_contrast_strength(self) -> None:
        parent_obj = apply_preset(PPOConfig(), "v6i22")
        normalize_and_validate_training_config(parent_obj)
        parent = dataclasses.asdict(parent_obj)
        for alias, coef, experiment_id in [
            ("v6i22d_coef005", 0.05, "v6i22d_coef005"),
            ("v6i22d", 0.10, "v6i22d_coef010"),
        ]:
            cfg_obj = apply_preset(PPOConfig(), alias)
            normalize_and_validate_training_config(cfg_obj)
            cfg = dataclasses.asdict(cfg_obj)
            changed = {k for k in parent if parent[k] != cfg[k]}
            self.assertEqual(
                changed,
                {
                    "experiment_id",
                    "latent_behavior_contrast_coef",
                    "latent_behavior_contrast_margin",
                    "run_tag",
                },
            )
            self.assertEqual(cfg_obj.experiment_id, experiment_id)
            self.assertAlmostEqual(float(cfg_obj.latent_behavior_contrast_coef), coef)
            self.assertAlmostEqual(float(cfg_obj.latent_behavior_contrast_margin), 0.06)


if __name__ == "__main__":
    unittest.main()
