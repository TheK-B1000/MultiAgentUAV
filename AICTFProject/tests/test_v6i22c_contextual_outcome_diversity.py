"""Pinning tests for V6I22C label-free contextual outcome diversity."""
from __future__ import annotations

import dataclasses
import unittest

from rl.config.ppo_config import PPOConfig
from rl.presets import apply_preset
from rl.training.config_validation import normalize_and_validate_training_config


class V6i22CContextualOutcomeDiversityTests(unittest.TestCase):
    def test_primary_aliases_resolve_to_same_config(self) -> None:
        aliases = [
            "v6i22c",
            "v6i22c_contextual_outcome_diversity",
            "v6i22c_outcome_diversity_coef003",
            "latent_v6i22c_contextual_outcome_diversity",
            "plan_faithful_latent_v6i22c_contextual_outcome_diversity",
        ]
        base = dataclasses.asdict(apply_preset(PPOConfig(), aliases[0]))
        for alias in aliases[1:]:
            self.assertEqual(base, dataclasses.asdict(apply_preset(PPOConfig(), alias)))

    def test_v6i22c_diff_vs_v6i22_is_outcome_diversity_only(self) -> None:
        parent_obj = apply_preset(PPOConfig(), "v6i22")
        normalize_and_validate_training_config(parent_obj)
        parent = dataclasses.asdict(parent_obj)
        cfg_obj = apply_preset(PPOConfig(), "v6i22c")
        normalize_and_validate_training_config(cfg_obj)
        cfg = dataclasses.asdict(cfg_obj)
        changed = {k for k in parent if parent[k] != cfg[k]}
        self.assertEqual(
            changed,
            {
                "experiment_id",
                "latent_outcome_diversity_coef",
                "run_tag",
            },
        )
        self.assertEqual(cfg_obj.experiment_id, "v6i22c_coef003")
        self.assertAlmostEqual(float(cfg_obj.latent_outcome_diversity_coef), 0.03)
        self.assertAlmostEqual(float(cfg_obj.latent_outcome_diversity_margin), 1.0)
        self.assertTrue(cfg_obj.latent_outcome_diversity_success_only)
        self.assertIn("v6i22c_contextual_outcome_diversity_coef003", cfg_obj.run_tag)

    def test_v6i22c_stays_label_free_and_router_off(self) -> None:
        parent = apply_preset(PPOConfig(), "v6i22")
        normalize_and_validate_training_config(parent)
        cfg = apply_preset(PPOConfig(), "v6i22c")
        normalize_and_validate_training_config(cfg)
        self.assertEqual(cfg.latent_assignment_mode, "balanced_episode")
        self.assertFalse(cfg.train_router_when_forced)
        self.assertFalse(cfg.train_router_critic_when_forced)
        self.assertFalse(cfg.latent_contract_specialist_enabled)
        self.assertEqual(float(cfg.latent_contract_specialist_coef), 0.0)
        self.assertEqual(cfg.latent_contract_specialist_variant, "base")
        self.assertEqual(float(cfg.latent_behavior_contrast_coef), 0.0)
        self.assertEqual(float(cfg.latent_behavior_contrast_margin), 0.25)
        self.assertFalse(cfg.latent_router_distill_enabled)
        self.assertEqual(float(cfg.latent_strategy_aux_predict_phase_coef), 0.0)
        self.assertFalse(cfg.latent_strategy_aux_return_head)
        self.assertFalse(cfg.latent_v3i3_event_preference_enabled)
        self.assertEqual(float(cfg.latent_preference_coef), float(parent.latent_preference_coef))
        self.assertEqual(float(cfg.latent_preference_commit_coef), float(parent.latent_preference_commit_coef))


if __name__ == "__main__":
    unittest.main()
