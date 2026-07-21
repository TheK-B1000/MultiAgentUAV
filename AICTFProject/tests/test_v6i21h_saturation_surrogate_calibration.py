"""Pinning tests for V6I21H saturation surrogate calibration preset."""
from __future__ import annotations

import dataclasses
import unittest

from gpu_env._maps import MAP_B_SPLIT_LANE, MAP_B_SPLIT_LANE_V2
from rl.config.ppo_config import PPOConfig
from rl.presets import apply_preset
from rl.training.config_validation import normalize_and_validate_training_config


class V6i21hSaturationSurrogatePresetTests(unittest.TestCase):
    def test_aliases_resolve_to_surrogate_arm(self) -> None:
        aliases = [
            "v6i21h",
            "v6i21h_saturation_surrogate_calibration",
            "v6i21h_saturation_surrogate",
            "latent_v6i21h_saturation_surrogate_calibration",
            "plan_faithful_latent_v6i21h_saturation_surrogate_calibration",
        ]
        base = dataclasses.asdict(apply_preset(PPOConfig(), aliases[0]))
        for alias in aliases[1:]:
            self.assertEqual(base, dataclasses.asdict(apply_preset(PPOConfig(), alias)))

    def test_v6i21h_is_exact_identity_diff_over_v6i21g(self) -> None:
        parent_cfg = apply_preset(PPOConfig(), "v6i21g")
        normalize_and_validate_training_config(parent_cfg)
        parent = dataclasses.asdict(parent_cfg)
        cfg_obj = apply_preset(PPOConfig(), "v6i21h")
        normalize_and_validate_training_config(cfg_obj)
        cfg = dataclasses.asdict(cfg_obj)
        changed = {k for k in parent if parent[k] != cfg[k]}
        self.assertEqual(changed, {"experiment_id", "run_tag"})
        self.assertEqual(cfg_obj.experiment_id, "v6i21h")
        self.assertEqual(cfg_obj.run_tag, "v6i21h_saturation_surrogate_calibration")

    def test_v6i21h_preserves_v6i21_scaffold_and_op_pool(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i21h")
        normalize_and_validate_training_config(cfg)
        self.assertEqual(tuple(str(x).upper() for x in cfg.opponent_pool), ("OP8", "OP9", "OP10", "OP11", "OP12"))
        self.assertEqual(tuple(cfg.map_pool), (MAP_B_SPLIT_LANE, MAP_B_SPLIT_LANE_V2))
        self.assertEqual(cfg.map_layout, MAP_B_SPLIT_LANE)
        self.assertFalse(cfg.train_router_when_forced)
        self.assertFalse(cfg.train_router_critic_when_forced)


if __name__ == "__main__":
    unittest.main()
