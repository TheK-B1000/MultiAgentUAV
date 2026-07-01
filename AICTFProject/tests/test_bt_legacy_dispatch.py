"""BT vs legacy scripted dispatch: no legacy side effects on pure BT rows."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _core(opponent: str, *, seed: int = 0):
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig  # type: ignore[import]

    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=2,
        max_red_agents=2,
        map_layout="map_b",
        max_decision_steps=400,
        aquaticus_profile=True,
        rules_profile="OURS",
        device="cpu",
        seed=seed,
    )
    env = GPUCTFVecEnv(cfg)
    env.reset()
    c = env.core
    c.set_next_opponent("SCRIPTED", opponent, env_indices=[0])
    c.red_coordinated_attack[0] = True
    c.red_coord_ticks_left[0] = 10
    c.red_coord_aim_x[0] = 99.0
    c.red_coord_aim_y[0] = 99.0
    c.red_deception_prob[0] = 0.5
    c.red_role_switch_prob[0] = 0.5
    return c, env


class TestBTLegacyDispatchIsolation(unittest.TestCase):
    def test_bt_only_skips_coord_tick_decrement(self) -> None:
        c, env = _core("OP11")
        ticks_before = int(c.red_coord_ticks_left[0].item())
        c._assign_scripted_targets_by_role("red")
        ticks_after = int(c.red_coord_ticks_left[0].item())
        self.assertEqual(ticks_before, ticks_after)
        env.close()

    def test_bt_only_targets_are_deterministic(self) -> None:
        outs = []
        for _ in range(2):
            c, env = _core("OP8", seed=99)
            c._assign_scripted_targets_by_role("red")
            outs.append(c._debug_red_target_x[0].clone())
            env.close()
        self.assertTrue(torch.allclose(outs[0], outs[1]))

    def test_legacy_op5_still_runs_coord_logic(self) -> None:
        c, env = _core("OP5", seed=7)
        c.red_coordinated_attack[0] = True
        c.red_coord_ticks_left[0] = 5
        c._assign_scripted_targets_by_role("red")
        self.assertLess(int(c.red_coord_ticks_left[0].item()), 5)
        env.close()


if __name__ == "__main__":
    unittest.main()
