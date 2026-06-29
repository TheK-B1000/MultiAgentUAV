"""OP9 deliberate mine placement through the behavior-tree route layer."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu_env._core._bt_profiles import profile_for_level
from gpu_env._core._bt_red import ROLE_DEFENDER, ROLE_ESCORT, ROLE_INTERCEPTOR
from macro_actions import MacroAction


def _core(opponent: str = "OP9", *, n_red: int = 2, seed: int = 0):
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig  # type: ignore[import]

    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=2,
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
    c = env.core
    c.set_next_opponent("SCRIPTED", opponent, env_indices=[0])
    c.red_script_role_flip[0] = False
    c.red_coordinated_attack[0] = False
    c.red_deception_prob[0] = 0.0
    c.red_role_switch_prob[0] = 0.0
    c.red_alive[0] = True
    c.blue_alive[0] = True
    c.bt_role_lock_ticks[0] = 0
    c.bt_mine_lock_ticks.zero_()
    c.bt_want_mine.zero_()
    c.sim_step_count.zero_()
    c.step_count.zero_()
    return c, env


def _in_mine_window(c) -> None:
    prof = profile_for_level(9)
    c.sim_step_count[0] = int(prof.mine_approach_lead_steps - 1)


class _BTMineTestCase(unittest.TestCase):
    _env = None

    def tearDown(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


class TestOP9MineProfile(_BTMineTestCase):
    def test_op9_enables_mines(self) -> None:
        self.assertTrue(profile_for_level(9).enable_mines)
        self.assertFalse(profile_for_level(8).enable_mines)


class TestOP9DefenderChokeMine(_BTMineTestCase):
    def test_defender_routes_to_approach_lane(self) -> None:
        c, self._env = _core("OP9", seed=101)
        _in_mine_window(c)
        c.red_mine_charges[0, 1] = 1
        c.red_x[0, 0] = 4.0
        c.red_y[0, 0] = 10.0
        c.red_x[0, 1] = 18.0
        c.red_y[0, 1] = 10.0
        c.blue_x[0, 0] = 12.0
        c.blue_y[0, 0] = 10.0
        c.blue_x[0, 1] = 14.0
        c.blue_y[0, 1] = 12.0

        c._assign_scripted_targets_by_role("red")
        roles = c.bt_red_role[0].tolist()
        self.assertIn(ROLE_DEFENDER, roles)
        agent = roles.index(ROLE_DEFENDER)
        self.assertTrue(bool(c.bt_want_mine[0, agent].item()))

        home_x = float(c.red_flag_home[0, 0].item())
        midline = float(c.cols) * 0.5
        expected_x = home_x + (midline - home_x) * 0.4
        tx = float(c._debug_red_target_x[0, agent].item())
        ty = float(c._debug_red_target_y[0, agent].item())
        self.assertAlmostEqual(tx, expected_x, places=1)
        self.assertAlmostEqual(ty, float(c.red_flag_home[0, 1].item()), places=1)


class TestOP9InterceptorLaneMine(_BTMineTestCase):
    def test_interceptor_mines_escape_lane(self) -> None:
        c, self._env = _core("OP9", seed=102)
        _in_mine_window(c)
        c.red_mine_charges[0, 0] = 1
        c.blue_carrying[0, 0] = True
        c.blue_x[0, 0] = 12.0
        c.blue_y[0, 0] = 10.0
        c.red_x[0, 0] = 6.0
        c.red_y[0, 0] = 10.0
        c.red_x[0, 1] = 7.0
        c.red_y[0, 1] = 11.0

        c._assign_scripted_targets_by_role("red")
        roles = c.bt_red_role[0].tolist()
        self.assertIn(ROLE_INTERCEPTOR, roles)
        agent = roles.index(ROLE_INTERCEPTOR)
        self.assertTrue(bool(c.bt_want_mine[0, agent].item()))

        home_x = float(c.blue_flag_home[0, 0].item())
        ec_x = float(c.blue_x[0, 0].item())
        expected_x = ec_x + (home_x - ec_x) * 0.5
        self.assertAlmostEqual(float(c.bt_mine_target_x[0, agent].item()), expected_x, places=1)


class TestOP9MineGuards(_BTMineTestCase):
    def test_no_mine_without_charge(self) -> None:
        c, self._env = _core("OP9", seed=103)
        _in_mine_window(c)
        c.red_mine_charges[0] = 0
        c.blue_x[0, 0] = 12.0
        c.blue_y[0, 0] = 10.0
        c.red_x[0, 0] = 4.0
        c.red_y[0, 0] = 10.0
        c._assign_scripted_targets_by_role("red")
        self.assertFalse(bool(c.bt_want_mine[0].any().item()))

    def test_carrier_does_not_mine(self) -> None:
        c, self._env = _core("OP9", seed=104)
        _in_mine_window(c)
        c.red_mine_charges[0, 0] = 1
        c.red_carrying[0, 0] = True
        c.red_x[0, 0] = 12.0
        c.red_y[0, 0] = 10.0
        c._assign_scripted_targets_by_role("red")
        self.assertFalse(bool(c.bt_want_mine[0, 0].item()))

    def test_escort_role_skips_mines(self) -> None:
        c, self._env = _core("OP10", seed=105)
        _in_mine_window(c)
        c.red_mine_charges[0, 1] = 1
        c.red_carrying[0, 0] = True
        c.red_x[0, 0] = 12.0
        c.red_y[0, 0] = 10.0
        c.red_x[0, 1] = 11.0
        c.red_y[0, 1] = 10.0
        c._assign_scripted_targets_by_role("red")
        roles = c.bt_red_role[0].tolist()
        if ROLE_ESCORT in roles:
            agent = roles.index(ROLE_ESCORT)
            self.assertFalse(bool(c.bt_want_mine[0, agent].item()))


class TestOP9PlaceMineMacro(_BTMineTestCase):
    def test_place_mine_macro_when_at_site(self) -> None:
        c, self._env = _core("OP9", seed=106)
        _in_mine_window(c)
        c.red_mine_charges[0, 1] = 1
        c.blue_x[0, 0] = 12.0
        c.blue_y[0, 0] = 10.0
        c.red_x[0, 0] = 4.0
        c.red_y[0, 0] = 10.0
        c.red_x[0, 1] = 18.0
        c.red_y[0, 1] = 10.0
        c._assign_scripted_targets_by_role("red")
        roles = c.bt_red_role[0].tolist()
        agent = roles.index(ROLE_DEFENDER)
        c.red_x[0, agent] = c.bt_mine_target_x[0, agent]
        c.red_y[0, agent] = c.bt_mine_target_y[0, agent]

        macro = c._bt_scripted_red_macros()
        self.assertIsNotNone(macro)
        assert macro is not None
        self.assertEqual(int(macro[0, agent].item()), int(MacroAction.PLACE_MINE))


if __name__ == "__main__":
    unittest.main()
