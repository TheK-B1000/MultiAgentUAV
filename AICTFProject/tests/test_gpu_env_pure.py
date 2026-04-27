from __future__ import annotations

import unittest

import torch

from gpu_env._core._dynamics import align_speed_cap_to_speed
from gpu_env._core._scripted_red import macro_commit_ticks
from macro_actions import MacroAction


class GPUEnvPureHelperTests(unittest.TestCase):
    def test_align_speed_cap_expands_batch_vector(self) -> None:
        speed = torch.zeros((2, 3), dtype=torch.float32)
        cap = align_speed_cap_to_speed(speed, torch.tensor([1.0, 2.0]))
        self.assertEqual(tuple(cap.shape), (2, 3))
        self.assertTrue(torch.equal(cap[0], torch.full((3,), 1.0)))
        self.assertTrue(torch.equal(cap[1], torch.full((3,), 2.0)))

    def test_align_speed_cap_accepts_scalar(self) -> None:
        speed = torch.zeros((2, 3), dtype=torch.float32)
        cap = align_speed_cap_to_speed(speed, torch.tensor(1.5))
        self.assertTrue(torch.equal(cap, torch.full_like(speed, 1.5)))

    def test_align_speed_cap_rejects_incompatible_shape(self) -> None:
        speed = torch.zeros((2, 3), dtype=torch.float32)
        with self.assertRaises(RuntimeError):
            align_speed_cap_to_speed(speed, torch.ones((4,)))

    def test_macro_commit_ticks_maps_each_macro(self) -> None:
        macro = torch.tensor(
            [
                int(MacroAction.GO_TO),
                int(MacroAction.GRAB_MINE),
                int(MacroAction.GET_FLAG),
                int(MacroAction.PLACE_MINE),
                int(MacroAction.GO_HOME),
            ],
            dtype=torch.int64,
        )
        ticks = macro_commit_ticks(
            macro,
            go_to_ticks=1,
            grab_ticks=2,
            get_flag_ticks=3,
            place_ticks=4,
            go_home_ticks=5,
        )
        self.assertTrue(torch.equal(ticks, torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)))

    def test_macro_commit_ticks_clamps_to_one(self) -> None:
        macro = torch.tensor([int(MacroAction.GO_TO)], dtype=torch.int64)
        ticks = macro_commit_ticks(
            macro,
            go_to_ticks=0,
            grab_ticks=0,
            get_flag_ticks=0,
            place_ticks=0,
            go_home_ticks=0,
        )
        self.assertTrue(torch.equal(ticks, torch.tensor([1], dtype=torch.int32)))
