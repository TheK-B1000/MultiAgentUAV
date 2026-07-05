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

    def test_v6i21b_pressure_constants_are_active(self) -> None:
        self.assertLessEqual(_BTAdaptiveMixin._NEAR_CAP_DIST, 7.0)
        self.assertLessEqual(_BTAdaptiveMixin._REPEAT_LANE_STREAK, 4)
        self.assertLessEqual(_BTAdaptiveMixin._HIGH_OVERCOMMIT, 0.40)
        self.assertAlmostEqual(_BTAdaptiveMixin._BLUE_CARRIER_SPEED_MULT, 0.95)
        self.assertGreaterEqual(profile_for_level(8).intercept_block_base, 0.66)
        self.assertGreaterEqual(profile_for_level(9).intercept_block_base, 0.70)
        self.assertGreaterEqual(profile_for_level(12).lock_counter, 20)

    def test_op8_through_op12_2v2_speed_ranges_do_not_slow_red(self) -> None:
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
            self.assertGreaterEqual(float(params["speed_mult"].min().item()), 1.0, key)


if __name__ == "__main__":
    unittest.main()
