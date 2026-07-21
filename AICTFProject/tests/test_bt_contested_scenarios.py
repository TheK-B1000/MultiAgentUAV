"""Deterministic contested CTF scenarios for OP5..OP12 behavior-tree opponents.

Each scenario forces a meaningful tactical choice and asserts the decision
(role, branch, target geometry) — not win rate alone.
"""
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
    ROLE_COUNTER,
    ROLE_DEFENDER,
    ROLE_ESCORT,
    ROLE_FLAG_RETR,
    ROLE_INTERCEPTOR,
    ROLE_2V1_WING,
)


def _core(opponent: str = "OP11", *, n_red: int = 2, n_blue: int = 2, seed: int = 0):
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig  # type: ignore[import]

    cfg = GPUFieldConfig(
        n_envs=1,
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
    c = env.core
    c._opponent_key[0] = opponent
    c.red_script_role_flip[0] = False
    c.red_coordinated_attack[0] = False
    c.red_deception_prob[0] = 0.0
    c.red_role_switch_prob[0] = 0.0
    c.red_alive[0] = True
    c.blue_alive[0] = True
    c.bt_role_lock_ticks[0] = 0
    return c, env


def _bt(c, opponent: str | None = None):
    if opponent:
        c._opponent_key[0] = opponent
    tx, ty = c._get_bt_targets()
    roles = c.bt_red_role[0].tolist()
    branches = c.bt_active_branch[0].tolist()
    return roles, branches, tx[0].tolist(), ty[0].tolist()


class TestScenario01InterceptFeasible(unittest.TestCase):
    def test_interceptor_role_and_midpoint_target(self) -> None:
        c, _ = _core("OP8")
        c.blue_carrying[0, 0] = True
        c.blue_x[0, 0] = 10.0
        c.blue_y[0, 0] = 10.0
        c.red_x[0, 0] = 8.0
        c.red_y[0, 0] = 9.0
        c.red_x[0, 1] = 8.0
        c.red_y[0, 1] = 11.0
        roles, branches, tx, ty = _bt(c)
        self.assertIn(ROLE_INTERCEPTOR, roles)
        agent = roles.index(ROLE_INTERCEPTOR)
        self.assertEqual(branches[agent], ROLE_INTERCEPTOR)
        home_x = float(c.blue_flag_home[0, 0].item())
        carrier_x = float(c.blue_x[0, 0].item())
        lo, hi = min(home_x, carrier_x), max(home_x, carrier_x)
        self.assertGreater(tx[agent], lo)
        self.assertLess(tx[agent], hi)


class TestScenario02InterceptInfeasibleCounter(unittest.TestCase):
    def test_op12_counter_not_interceptor(self) -> None:
        c, _ = _core("OP12")
        c.blue_carrying[0, 0] = True
        c.blue_x[0, 0] = 1.0
        c.blue_y[0, 0] = 10.0
        c.red_x[0, 0] = 18.0
        c.red_y[0, 0] = 5.0
        c.red_x[0, 1] = 18.0
        c.red_y[0, 1] = 15.0
        roles, branches, _, _ = _bt(c)
        self.assertIn(ROLE_COUNTER, roles)
        self.assertNotIn(ROLE_INTERCEPTOR, roles)
        agent = roles.index(ROLE_COUNTER)
        self.assertEqual(branches[agent], ROLE_COUNTER)

    def test_op12_counter_pushes_on_blue_overcommit_before_flag_grab(self) -> None:
        c, _ = _core("OP12")
        c.bt_adapt_overcommit[0] = 0.8
        c.blue_carrying[0] = False
        c.red_flag_pos[0] = c.red_flag_home[0].clone()
        c.blue_x[0, 0] = 15.0
        c.blue_y[0, 0] = 8.0
        c.blue_x[0, 1] = 16.0
        c.blue_y[0, 1] = 12.0
        c.red_x[0, 0] = 18.0
        c.red_y[0, 0] = 5.0
        c.red_x[0, 1] = 18.0
        c.red_y[0, 1] = 15.0
        roles, branches, _, _ = _bt(c)
        self.assertIn(ROLE_COUNTER, roles)
        agent = roles.index(ROLE_COUNTER)
        self.assertEqual(branches[agent], ROLE_COUNTER)


class TestScenario03CarrierPursuedEscort(unittest.TestCase):
    def test_escort_interposes_between_carrier_and_threat(self) -> None:
        c, _ = _core("OP10")
        c.red_carrying[0, 0] = True
        c.red_x[0, 0] = 14.0
        c.red_y[0, 0] = 10.0
        c.blue_x[0, 0] = 13.0
        c.blue_y[0, 0] = 10.0
        roles, branches, tx, ty = _bt(c)
        self.assertIn(ROLE_ESCORT, roles)
        esc = roles.index(ROLE_ESCORT)
        self.assertEqual(branches[esc], ROLE_ESCORT)
        mid_x = (14.0 + 13.0) * 0.5
        self.assertAlmostEqual(tx[esc], mid_x, delta=1.5)


class TestScenario04CarrierSafeRoute(unittest.TestCase):
    def test_carrier_deviates_from_blocked_direct_path(self) -> None:
        c, _ = _core("OP11")
        c.red_carrying[0, 0] = True
        c.red_x[0, 0] = 12.0
        c.red_y[0, 0] = 10.0
        home_y = float(c.red_flag_home[0, 1].item())
        mid_x = (12.0 + float(c.red_flag_home[0, 0].item())) * 0.5
        c.blue_x[0, 0] = mid_x
        c.blue_y[0, 0] = 10.0
        roles, _, tx, ty = _bt(c)
        carrier = 0
        self.assertTrue(c.red_carrying[0, carrier])
        self.assertGreater(abs(ty[carrier] - home_y), 0.4)


class TestScenario05TwoVsOnePressure(unittest.TestCase):
    def test_2v1_wing_when_two_red_near_one_blue(self) -> None:
        c, _ = _core("OP11")
        c.red_x[0, 0] = 7.0
        c.red_y[0, 0] = 10.0
        c.red_x[0, 1] = 8.0
        c.red_y[0, 1] = 9.0
        c.blue_x[0, 0] = 7.5
        c.blue_y[0, 0] = 9.5
        c.blue_x[0, 1] = 1.0
        c.blue_y[0, 1] = 1.0
        roles, branches, _, _ = _bt(c)
        self.assertIn(ROLE_2V1_WING, roles)
        wing = roles.index(ROLE_2V1_WING)
        self.assertEqual(branches[wing], ROLE_2V1_WING)


class TestScenario06DualFlagStandoff(unittest.TestCase):
    def test_both_carriers_trigger_escort_and_intercept_roles(self) -> None:
        c, _ = _core("OP11")
        c.red_carrying[0, 0] = True
        c.red_x[0, 0] = 14.0
        c.red_y[0, 0] = 10.0
        c.blue_carrying[0, 0] = True
        c.blue_x[0, 0] = 6.0
        c.blue_y[0, 0] = 10.0
        c.red_x[0, 1] = 8.0
        c.red_y[0, 1] = 10.0
        roles, _, _, _ = _bt(c)
        self.assertTrue(
            ROLE_ESCORT in roles or ROLE_INTERCEPTOR in roles,
            f"Expected escort or intercept in dual-carry state, got {roles}",
        )


class TestScenario07DefendVsCounter(unittest.TestCase):
    def test_trailing_infeasible_op11_counters_not_all_attack(self) -> None:
        c, _ = _core("OP11", seed=1)
        c.red_score[0] = 0
        c.blue_score[0] = 1
        c.blue_carrying[0, 0] = True
        c.blue_x[0, 0] = 1.0
        c.blue_y[0, 0] = 10.0
        c.red_x[0, 0] = 18.0
        c.red_y[0, 0] = 5.0
        c.red_x[0, 1] = 18.0
        c.red_y[0, 1] = 15.0
        roles, _, _, _ = _bt(c)
        self.assertIn(ROLE_COUNTER, roles)
        self.assertNotEqual(roles.count(ROLE_ATTACKER), len(roles))


class TestScenario08DefenderChokePoint(unittest.TestCase):
    def test_defender_targets_zone_not_flag_home(self) -> None:
        c, _ = _core("OP6")
        midline = float(c.cols) * 0.5
        c.blue_x[0, 0] = midline + 2.0
        c.blue_y[0, 0] = 10.0
        roles, _, tx, ty = _bt(c)
        self.assertIn(ROLE_DEFENDER, roles)
        d = roles.index(ROLE_DEFENDER)
        flag_x = float(c.red_flag_home[0, 0].item())
        self.assertGreater(abs(tx[d] - flag_x), 0.5)


class TestScenario09EscortEndsAfterCarrierTagged(unittest.TestCase):
    def test_no_escort_when_carrier_tagged(self) -> None:
        c, _ = _core("OP10")
        c.red_carrying[0, 0] = True
        c.red_tagged[0, 0] = True
        c.red_x[0, 0] = 14.0
        c.red_y[0, 0] = 10.0
        c.blue_x[0, 0] = 13.0
        c.blue_y[0, 0] = 10.0
        roles, _, _, _ = _bt(c)
        self.assertNotIn(ROLE_ESCORT, roles)


class TestScenario10RouteReplanOnFlagTransition(unittest.TestCase):
    def test_objective_changes_when_flag_captured(self) -> None:
        c, _ = _core("OP7")
        c.red_flag_pos[0, 0] = 10.0
        c.red_flag_pos[0, 1] = 10.0
        roles_before, _, _, _ = _bt(c)
        self.assertIn(ROLE_FLAG_RETR, roles_before)
        c.red_flag_pos[0, 0] = float(c.red_flag_home[0, 0])
        c.red_flag_pos[0, 1] = float(c.red_flag_home[0, 1])
        c.bt_role_lock_ticks[0] = 0
        prev_changes = int(c.bt_tel_objective_changes[0].item())
        roles_after, _, _, _ = _bt(c)
        new_changes = int(c.bt_tel_objective_changes[0].item())
        self.assertNotIn(ROLE_FLAG_RETR, roles_after)
        self.assertGreater(new_changes, prev_changes)


class TestScriptedDispatchUsesBT(unittest.TestCase):
    def test_assign_scripted_sets_bt_roles_for_op11(self) -> None:
        c, _ = _core("OP11")
        c.red_carrying[0, 0] = True
        c.red_x[0, 0] = 14.0
        c.red_y[0, 0] = 10.0
        c.blue_x[0, 0] = 13.0
        c.blue_y[0, 0] = 10.0
        c._assign_scripted_targets_by_role("red")
        roles = c.bt_red_role[0].tolist()
        self.assertIn(ROLE_ESCORT, roles)


if __name__ == "__main__":
    unittest.main()
