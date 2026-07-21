"""Pinning tests for V6I22 label-free adaptive hardpool repertoire birth."""
from __future__ import annotations

import dataclasses
import unittest

from gpu_env._maps import MAP_B_SPLIT_LANE, MAP_B_SPLIT_LANE_V2
from rl.config.ppo_config import PPOConfig
from rl.presets import apply_preset
from rl.training.config_validation import normalize_and_validate_training_config


class V6i22AdaptiveHardpoolRepertoireBirthTests(unittest.TestCase):
    def test_aliases_resolve_to_same_repertoire_birth_config(self) -> None:
        aliases = [
            "v6i22",
            "v6i22_adaptive_hardpool_repertoire_birth",
            "v6i22_repertoire_birth",
            "latent_v6i22_adaptive_hardpool_repertoire_birth",
            "plan_faithful_latent_v6i22_adaptive_hardpool_repertoire_birth",
        ]
        base = dataclasses.asdict(apply_preset(PPOConfig(), aliases[0]))
        for alias in aliases[1:]:
            self.assertEqual(base, dataclasses.asdict(apply_preset(PPOConfig(), alias)))

    def test_v6i22_is_label_free_repertoire_diff_over_v6i21j(self) -> None:
        parent_cfg = apply_preset(PPOConfig(), "v6i21j")
        normalize_and_validate_training_config(parent_cfg)
        parent = dataclasses.asdict(parent_cfg)
        cfg_obj = apply_preset(PPOConfig(), "v6i22")
        normalize_and_validate_training_config(cfg_obj)
        cfg = dataclasses.asdict(cfg_obj)
        changed = {k for k in parent if parent[k] != cfg[k]}
        self.assertEqual(
            changed,
            {
                "experiment_id",
                "latent_contract_specialist_coef",
                "latent_contract_specialist_enabled",
                "latent_contract_specialist_variant",
                "run_tag",
            },
        )
        self.assertEqual(cfg_obj.experiment_id, "v6i22")
        self.assertEqual(
            cfg_obj.run_tag,
            "v6i22_adaptive_hardpool_repertoire_birth_OP8_OP9_OP10_OP11_OP12",
        )

    def test_v6i22_preserves_hardpool_surface_and_blocks_router(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i22")
        normalize_and_validate_training_config(cfg)
        self.assertEqual(tuple(str(x).upper() for x in cfg.opponent_pool), ("OP8", "OP9", "OP10", "OP11", "OP12"))
        self.assertEqual(tuple(cfg.map_pool), (MAP_B_SPLIT_LANE, MAP_B_SPLIT_LANE_V2))
        self.assertEqual(cfg.map_layout, MAP_B_SPLIT_LANE)
        self.assertEqual(cfg.latent_assignment_mode, "balanced_episode")
        self.assertFalse(cfg.train_router_when_forced)
        self.assertFalse(cfg.train_router_critic_when_forced)
        self.assertEqual(cfg.v6i9_training_stage, "repertoire")

    def test_v6i22_has_no_handcrafted_contract_scaffold(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i22")
        normalize_and_validate_training_config(cfg)
        self.assertFalse(cfg.latent_contract_specialist_enabled)
        self.assertEqual(float(cfg.latent_contract_specialist_coef), 0.0)
        self.assertEqual(cfg.latent_contract_specialist_variant, "base")
        self.assertTrue(cfg.enable_latent_z_residual)
        self.assertTrue(cfg.latent_actor_z_adapter_enabled)
        self.assertFalse(cfg.latent_router_distill_enabled)
        self.assertEqual(float(cfg.latent_strategy_aux_predict_phase_coef), 0.0)
        self.assertFalse(cfg.latent_strategy_aux_return_head)


if __name__ == "__main__":
    unittest.main()
