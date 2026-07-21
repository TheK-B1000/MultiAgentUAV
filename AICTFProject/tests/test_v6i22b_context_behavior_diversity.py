"""Pinning tests for V6I22B label-free behavior-diversity repertoire birth."""
from __future__ import annotations

import dataclasses
import unittest

from rl.config.ppo_config import PPOConfig
from rl.presets import apply_preset
from rl.training.config_validation import normalize_and_validate_training_config


class V6i22BContextBehaviorDiversityTests(unittest.TestCase):
    def test_primary_aliases_resolve_to_same_config(self) -> None:
        aliases = [
            "v6i22b",
            "v6i22b_context_behavior_diversity",
            "v6i22b_behavior_diversity_coef003",
            "latent_v6i22b_context_behavior_diversity",
            "plan_faithful_latent_v6i22b_context_behavior_diversity",
        ]
        base = dataclasses.asdict(apply_preset(PPOConfig(), aliases[0]))
        for alias in aliases[1:]:
            self.assertEqual(base, dataclasses.asdict(apply_preset(PPOConfig(), alias)))

    def test_v6i22b_diff_vs_v6i22_is_behavior_contrast_only(self) -> None:
        parent_cfg = apply_preset(PPOConfig(), "v6i22")
        normalize_and_validate_training_config(parent_cfg)
        parent = dataclasses.asdict(parent_cfg)
        cfg_obj = apply_preset(PPOConfig(), "v6i22b")
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
        self.assertEqual(cfg_obj.experiment_id, "v6i22b_coef003")
        self.assertAlmostEqual(float(cfg_obj.latent_behavior_contrast_coef), 0.03)
        self.assertAlmostEqual(float(cfg_obj.latent_behavior_contrast_margin), 0.06)
        self.assertIn("v6i22b_context_behavior_diversity_coef003", cfg_obj.run_tag)

    def test_v6i22b_stays_label_free_and_router_off(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i22b")
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

    def test_sweep_arms_only_change_behavior_contrast_strength(self) -> None:
        parent_obj = apply_preset(PPOConfig(), "v6i22")
        normalize_and_validate_training_config(parent_obj)
        parent = dataclasses.asdict(parent_obj)
        for alias, coef in [
            ("v6i22b_coef001", 0.01),
            ("v6i22b", 0.03),
            ("v6i22b_coef005", 0.05),
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
            self.assertAlmostEqual(float(cfg_obj.latent_behavior_contrast_coef), coef)
            self.assertAlmostEqual(float(cfg_obj.latent_behavior_contrast_margin), 0.06)


if __name__ == "__main__":
    unittest.main()
