"""Unit tests for OP5..OP12 tactical profile registry."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu_env._core._bt_profiles import (
    BT_OPPONENT_KEYS,
    BT_PROFILES,
    build_profile_tensors,
    is_bt_opponent,
    normalize_bt_level,
    profile_for_level,
)


class TestBTProfileRegistry(unittest.TestCase):
    def test_levels_5_through_12_exist(self) -> None:
        for lvl in range(5, 13):
            self.assertIn(lvl, BT_PROFILES)

    def test_opponent_keys_resolve(self) -> None:
        for key in ("OP5", "OP6", "OP7", "OP8", "OP9", "OP10", "OP11", "OP12"):
            self.assertTrue(is_bt_opponent(key))
            self.assertEqual(normalize_bt_level(key), int(key[2:]))

    def test_non_bt_keys_rejected(self) -> None:
        self.assertFalse(is_bt_opponent("OP3"))
        self.assertIsNone(normalize_bt_level("OP4"))

    def test_profile_tensors_shape(self) -> None:
        keys = ["OP5", "OP12", "OP7", "OP9"]
        prof = build_profile_tensors(keys, device=torch.device("cpu"), batch_size=4)
        self.assertEqual(tuple(prof["bt_level"].shape), (4,))
        self.assertEqual(int(prof["bt_level"][0].item()), 5)
        self.assertEqual(int(prof["bt_level"][1].item()), 12)
        self.assertTrue(prof["enable_escort"][2].item())   # OP7
        self.assertFalse(prof["enable_counter"][3].item())  # OP9

    def test_curriculum_escort_progression(self) -> None:
        self.assertFalse(profile_for_level(5).enable_escort)
        self.assertTrue(profile_for_level(7).enable_escort)
        self.assertTrue(profile_for_level(10).enable_escort)

    def test_op12_counter_always(self) -> None:
        self.assertTrue(profile_for_level(12).counter_always)
        self.assertFalse(profile_for_level(11).counter_always)

    def test_all_bt_keys_in_registry(self) -> None:
        for key in BT_OPPONENT_KEYS:
            self.assertIsNotNone(normalize_bt_level(key))


if __name__ == "__main__":
    unittest.main()
