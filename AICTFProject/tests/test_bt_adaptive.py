"""Tests for OP8-OP12 adaptive hardpool v2 memory and profiles."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu_env._core._bt_profiles import build_profile_tensors, profile_for_level


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


if __name__ == "__main__":
    unittest.main()
