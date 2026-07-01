"""OP5 coordinated_attack / attack_sync_window reach the GPU core and scripted red."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from game_field_gpu import GPUFieldConfig
from gpu_env._core_class import BatchedCTFCore
from opponent_params import sample_batched_opponent_params


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


class Op5CoordinationWiringTests(unittest.TestCase):
    def test_sample_includes_coordination_keys(self) -> None:
        p = sample_batched_opponent_params(
            kind="SCRIPTED",
            key="OP5_RUSHER",
            phase="OP5_RUSHER",
            n_agents=2,
            batch_size=8,
            device="cpu",
            generator=_gen(7),
        )
        self.assertIn("coordinated_attack", p)
        self.assertIn("attack_sync_window", p)
        self.assertEqual(tuple(p["coordinated_attack"].shape), (8,))
        self.assertEqual(tuple(p["attack_sync_window"].shape), (8,))
        self.assertEqual(p["coordinated_attack"].dtype, torch.bool)
        self.assertEqual(p["attack_sync_window"].dtype, torch.int32)

    def test_core_applies_coordination_from_sample(self) -> None:
        core = BatchedCTFCore(GPUFieldConfig(n_envs=2, n_agents_per_team=2, device="cpu", seed=101))
        for i in range(2):
            core._opponent_kind[i] = "SCRIPTED"
            core._opponent_key[i] = "OP5_RUSHER"
        fixed = {
            "deception_prob": torch.zeros(2),
            "speed_mult": torch.ones(2),
            "attacker_style": torch.ones(2, dtype=torch.int32),
            "defender_style": torch.zeros(2, dtype=torch.int32),
            "role_switch_prob": torch.zeros(2),
            "coordinated_attack": torch.tensor([True, False], dtype=torch.bool),
            "attack_sync_window": torch.tensor([5, 3], dtype=torch.int32),
            "noise_sigma": torch.zeros(2),
        }
        with mock.patch("gpu_env._core._dynamics.sample_batched_opponent_params", return_value=fixed):
            core._apply_opponent_params_for_mask(torch.ones((2,), dtype=torch.bool, device=core.device))
        self.assertTrue(bool(core.red_coordinated_attack[0].item()))
        self.assertFalse(bool(core.red_coordinated_attack[1].item()))
        self.assertEqual(int(core.red_attack_sync_window[0].item()), 5)
        self.assertEqual(int(core.red_attack_sync_window[1].item()), 3)
        self.assertEqual(int(core.red_coord_ticks_left[0].item()), 0)
        self.assertEqual(int(core.red_coord_ticks_left[1].item()), 0)

    def test_coordinated_red_locks_striker_to_shared_aim(self) -> None:
        core = BatchedCTFCore(GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=202))
        core._opponent_kind[0] = "SCRIPTED"
        core._opponent_key[0] = "OP5_RUSHER"
        core._apply_opponent_params_for_mask(torch.ones((1,), dtype=torch.bool, device=core.device))
        core.red_coordinated_attack[:] = True
        core.red_attack_sync_window[:] = 20
        core.red_coord_ticks_left[:] = 0
        # Blue flag at home (left); coordinated red attackers aim at mirrored blue_flag_pos.
        fx = float(core.blue_flag_pos[0, 0].item())
        fy = float(core.blue_flag_pos[0, 1].item())
        tx, ty = core._assign_scripted_targets_by_role("red")
        flip = bool(core.red_script_role_flip[0].item())
        striker_j = 0 if flip else 1
        s1x, s1y = float(tx[0, striker_j].item()), float(ty[0, striker_j].item())
        self.assertAlmostEqual(s1x, fx, places=4)
        self.assertAlmostEqual(s1y, fy, places=4)
        self.assertGreater(int(core.red_coord_ticks_left[0].item()), 0)


if __name__ == "__main__":
    unittest.main()
