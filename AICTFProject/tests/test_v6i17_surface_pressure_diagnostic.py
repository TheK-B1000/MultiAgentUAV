"""Pinning tests for V6I17 surface-pressure diagnostic."""

from __future__ import annotations

import dataclasses
import unittest

from rl.config.ppo_config import PPOConfig
from rl.presets import apply_preset
from rl.training.config_validation import normalize_and_validate_training_config


class V6i17SurfacePressurePresetTests(unittest.TestCase):
    def test_aliases_resolve_to_harder_asymmetric_surface_arm(self) -> None:
        aliases = [
            "v6i17",
            "v6i17_surface_pressure_diagnostic",
            "v6i17_harder_asymmetric_opponents",
            "latent_v6i17_surface_pressure_diagnostic",
            "plan_faithful_latent_v6i17_surface_pressure_diagnostic",
        ]
        base = dataclasses.asdict(apply_preset(PPOConfig(), aliases[0]))
        for alias in aliases[1:]:
            self.assertEqual(base, dataclasses.asdict(apply_preset(PPOConfig(), alias)))

    def test_v6i17_is_exact_surface_diff_over_v6i16(self) -> None:
        parent = dataclasses.asdict(apply_preset(PPOConfig(), "v6i16_capacity_sharp_contracts"))
        cfg_obj = apply_preset(PPOConfig(), "v6i17_surface_pressure_diagnostic")
        cfg = dataclasses.asdict(cfg_obj)
        changed = {k for k in parent if parent[k] != cfg[k]}
        self.assertEqual(changed, {"experiment_id", "opponent_pool", "run_tag"})

        self.assertEqual(cfg_obj.experiment_id, "v6i17")
        self.assertEqual(
            tuple(str(x).upper() for x in cfg_obj.opponent_pool),
            ("OP8", "OP9", "OP10", "OP11", "OP12"),
        )
        self.assertEqual(tuple(cfg_obj.opponent_pool_weights), ())
        self.assertEqual(cfg_obj.run_tag, "v6i17_surface_pressure_diagnostic_OP8_OP9_OP10_OP11_OP12")

    def test_v6i17_preserves_contract_specialist_scaffold_and_blocks_router(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i17")
        self.assertTrue(cfg.latent_contract_specialist_enabled)
        self.assertEqual(cfg.latent_contract_specialist_variant, "sharp")
        self.assertAlmostEqual(float(cfg.latent_contract_specialist_coef), 0.75)
        self.assertTrue(cfg.enable_latent_z_residual)
        self.assertTrue(cfg.latent_actor_z_adapter_enabled)
        self.assertEqual(cfg.latent_assignment_mode, "balanced_episode")
        self.assertFalse(cfg.train_router_when_forced)
        self.assertFalse(cfg.train_router_critic_when_forced)
        self.assertEqual(cfg.v6i9_training_stage, "repertoire")

    def test_training_validation_preserves_op11_op12_surface(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i17")
        normalize_and_validate_training_config(cfg)
        self.assertEqual(
            tuple(str(x).upper() for x in cfg.opponent_pool),
            ("OP8", "OP9", "OP10", "OP11", "OP12"),
        )


if __name__ == "__main__":
    unittest.main()
