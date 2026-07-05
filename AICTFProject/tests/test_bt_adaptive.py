"""Tests for OP8-OP12 adaptive hardpool v2 memory and profiles."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu_env._core._bt_adaptive import _BTAdaptiveMixin
from gpu_env._core._bt_profiles import build_profile_tensors, profile_for_level
from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
from opponent_params import sample_batched_opponent_params


class BTAdaptiveProfileTests(unittest.TestCase):
    def test_op8_through_op12_adaptive_enabled(self) -> None:
        for lvl in range(8, 13):
            self.assertTrue(profile_for_level(lvl).adaptive_enabled, f"level {lvl}")

    def test_op5_through_op7_adaptive_disabled(self) -> None:
        for lvl in range(5, 8):
            self.assertFalse(profile_for_level(lvl).adaptive_enabled, f"level {lvl}")

    def test_profile_tensors_export_adaptive_flag(self) -> None:
        keys = ["OP8", "OP9", "OP10", "OP11", "OP12"]
        prof = build_profile_tensors(keys, device=torch.device("cpu"), batch_size=5)
        self.assertTrue(bool(prof["adaptive_enabled"].all().item()))

    def test_v6i21d_brutal_denial_constants_are_active(self) -> None:
        self.assertGreaterEqual(_BTAdaptiveMixin._NEAR_CAP_DIST, 12.0)
        self.assertLessEqual(_BTAdaptiveMixin._REPEAT_LANE_STREAK, 2)
        self.assertLessEqual(_BTAdaptiveMixin._HIGH_OVERCOMMIT, 0.25)
        self.assertLessEqual(_BTAdaptiveMixin._BLUE_CARRIER_SPEED_MULT, 0.75)
        self.assertGreaterEqual(_BTAdaptiveMixin._RED_INTERCEPTOR_NEAR_FLAG_BOOST, 1.35)
        self.assertLessEqual(_BTAdaptiveMixin._RED_RESPAWN_MULT, 0.50)
        self.assertGreaterEqual(profile_for_level(8).intercept_block_base, 0.82)
        self.assertGreaterEqual(profile_for_level(9).intercept_block_base, 0.84)
        self.assertGreaterEqual(profile_for_level(12).lock_counter, 30)

    def test_op8_through_op12_2v2_speed_ranges_boost_red(self) -> None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(123)
        for key in ("OP8", "OP9", "OP10", "OP11", "OP12"):
            params = sample_batched_opponent_params(
                "SCRIPTED",
                key,
                n_agents=2,
                batch_size=32,
                device="cpu",
                generator=gen,
            )
            self.assertGreaterEqual(float(params["speed_mult"].min().item()), 1.20, key)

    def test_speed_overdrive_mask_allows_hardpool_red_to_exceed_base_cap(self) -> None:
        env = GPUCTFVecEnv(
            GPUFieldConfig(
                n_envs=1,
                max_blue_agents=2,
                max_red_agents=2,
                map_layout="map_b",
                max_decision_steps=16,
                aquaticus_profile=True,
                rules_profile="OURS",
                device="cpu",
                seed=123,
            )
        )
        try:
            x = torch.full((1, 2), 5.0)
            y = torch.full((1, 2), 5.0)
            heading = torch.zeros((1, 2))
            speed = torch.full((1, 2), float(env.cfg.max_speed_cps))
            alive = torch.ones((1, 2), dtype=torch.bool)
            tx = torch.full((1, 2), 20.0)
            ty = torch.full((1, 2), 5.0)
            cap = torch.full((1, 2), float(env.cfg.max_speed_cps) * 1.20)
            mask = torch.ones((1, 2), dtype=torch.bool)
            _, _, _, speed2, _, _ = env.core._integrate_side(
                x,
                y,
                heading,
                speed,
                alive,
                tx,
                ty,
                speed_cap=cap,
                speed_overdrive_mask=mask,
            )
            self.assertGreater(float(speed2.max().item()), float(env.cfg.max_speed_cps))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
