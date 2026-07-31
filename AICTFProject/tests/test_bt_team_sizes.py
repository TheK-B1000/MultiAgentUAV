"""Team-size validation for behavior-tree opponents (2v2, 4v4, max supported)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu_env._core._bt_red import (
    ROLE_ATTACKER,
    ROLE_DEFENDER,
    ROLE_ESCORT,
    ROLE_INTERCEPTOR,
)


def _make(n_red: int, n_blue: int, opponent: str = "OP11", *, n_envs: int = 1, seed: int = 0):
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig  # type: ignore[import]

    cfg = GPUFieldConfig(
        n_envs=n_envs,
        max_blue_agents=n_blue,
        max_red_agents=n_red,
        map_layout="map_b",
        max_decision_steps=400,
        aquaticus_profile=True,
        rules_profile="OURS",
        device="cpu",
        seed=seed,
    )
    env = GPUCTFVecEnv(cfg)
    env.reset()
    core = env.core
    for i in range(n_envs):
        core._opponent_key[i] = opponent
        core.red_script_role_flip[i] = False
        core.red_coordinated_attack[i] = False
        core.red_alive[i] = True
        core.blue_alive[i] = True
        core.bt_role_lock_ticks[i] = 0
    return core, env


class Test2v2RoleSpread(unittest.TestCase):
    def test_attacker_and_defender_coexist(self) -> None:
        core, _ = _make(2, 2, "OP7")
        midline = float(core.cols) * 0.5
        core.blue_x[0, 0] = midline + 2.0
        core.blue_y[0, 0] = 10.0
        core._get_bt_targets()
        roles = set(core.bt_red_role[0].tolist())
        self.assertIn(ROLE_DEFENDER, roles)
        self.assertGreater(len(roles), 1)


class Test4v4RoleSpread(unittest.TestCase):
    def test_not_all_agents_same_role(self) -> None:
        core, _ = _make(4, 4, "OP11")
        midline = float(core.cols) * 0.5
        core.blue_x[0, 0] = midline + 2.0
        core.blue_y[0, 0] = 8.0
        core.blue_x[0, 1] = midline + 3.0
        core.blue_y[0, 1] = 12.0
        core._get_bt_targets()
        roles = core.bt_red_role[0].tolist()
        self.assertGreater(len(set(roles)), 1, f"All agents same role: {roles}")

    def test_targets_not_identical(self) -> None:
        core, _ = _make(4, 4, "OP11")
        midline = float(core.cols) * 0.5
        core.blue_x[0, 0] = midline + 2.0
        core.blue_y[0, 0] = 8.0
        core.blue_x[0, 1] = midline + 3.0
        core.blue_y[0, 1] = 12.0
        core.red_carrying[0, 0] = True
        core.red_x[0, 0] = 14.0
        core.red_y[0, 0] = 10.0
        core._assign_scripted_targets_by_role("red")
        tx = core._debug_red_target_x[0]
        ty = core._debug_red_target_y[0]
        pairs = {
            (round(float(tx[i].item()), 1), round(float(ty[i].item()), 1))
            for i in range(4)
        }
        self.assertGreater(len(pairs), 1)

    def test_escort_limited_to_one_support_agent(self) -> None:
        # OP12 carries the escort gate; OP10 is the pure interceptor niche.
        core, _ = _make(4, 4, "OP12")
        # Past OP12's stage-1 opening window, which forces ATTACKER.
        core.step_count[0] = 25
        core.sim_step_count[0] = 25
        core.red_carrying[0, 0] = True
        core.red_x[0, 0] = 14.0
        core.red_y[0, 0] = 10.0
        core.blue_x[0, 0] = 13.0
        core.blue_y[0, 0] = 10.0
        core._get_bt_targets()
        roles = core.bt_red_role[0].tolist()
        self.assertEqual(roles.count(ROLE_ESCORT), 1)

    def test_valid_action_bounds(self) -> None:
        core, _ = _make(4, 4, "OP12")
        core._assign_scripted_targets_by_role("red")
        max_x = float(core.cols - 1)
        max_y = float(core.rows - 1)
        tx = core._debug_red_target_x
        ty = core._debug_red_target_y
        self.assertFalse(torch.isnan(tx).any())
        self.assertTrue((tx >= 0).all() and (tx <= max_x).all())
        self.assertTrue((ty >= 0).all() and (ty <= max_y).all())


class Test4v4VectorizedIsolation(unittest.TestCase):
    def test_mixed_opponents_isolated(self) -> None:
        core, _ = _make(2, 2, "OP5", n_envs=2, seed=3)
        core.set_next_opponent("SCRIPTED", "OP5", env_indices=[0])
        core.set_next_opponent("SCRIPTED", "OP12", env_indices=[1])
        core.red_flag_pos[0, 0] = 10.0
        core.red_flag_pos[0, 1] = 10.0
        core.blue_carrying[1, 0] = True
        core.blue_x[1, 0] = 1.0
        core.blue_y[1, 0] = 10.0
        core.red_x[1, 0] = 18.0
        core.red_y[1, 0] = 5.0
        core._get_bt_targets()
        self.assertNotEqual(core.bt_red_role[0].tolist(), core.bt_red_role[1].tolist())


class TestResetClearsBTState(unittest.TestCase):
    def test_reset_zeros_telemetry_and_roles(self) -> None:
        core, env = _make(4, 4, "OP11")
        core.red_carrying[0, 0] = True
        core._get_bt_targets()
        self.assertGreater(int(core.bt_tel_escort_attempts[0].item()), 0)
        env.reset()
        self.assertEqual(int(core.bt_tel_escort_attempts[0].item()), 0)
        self.assertTrue((core.bt_red_role[0] == ROLE_ATTACKER).all())


class TestMaxTeamSmoke(unittest.TestCase):
    def test_8v8_produces_valid_targets(self) -> None:
        core, _ = _make(8, 8, "OP11")
        core._assign_scripted_targets_by_role("red")
        self.assertEqual(tuple(core._debug_red_target_x.shape), (1, 8))


if __name__ == "__main__":
    unittest.main()
