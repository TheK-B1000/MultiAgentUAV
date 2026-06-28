"""Curriculum and isolation tests for OP5..OP12 behavior-tree opponents."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu_env._core._bt_profiles import is_bt_opponent
from gpu_env._core._bt_red import (
    ROLE_ATTACKER,
    ROLE_COUNTER,
    ROLE_DEFENDER,
    ROLE_ESCORT,
    ROLE_FLAG_RETR,
    ROLE_INTERCEPTOR,
)


def _make_core(opponent: str, *, seed: int = 0, max_steps: int = 400,
               step: int = 0, red_score: int = 0, blue_score: int = 0):
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig  # type: ignore[import]

    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2,
        map_layout="map_b", max_decision_steps=max_steps,
        aquaticus_profile=True, rules_profile="OURS",
        device="cpu", seed=seed,
    )
    env = GPUCTFVecEnv(cfg)
    env.reset()
    core = env.core
    core.blue_score[0] = blue_score
    core.red_score[0] = red_score
    core.step_count[0] = step
    core.sim_step_count[0] = step
    core._opponent_key[0] = opponent
    core.red_script_role_flip[0] = False
    core.red_coordinated_attack[0] = False
    core.red_alive[0] = True
    core.blue_alive[0] = True
    core.red_deception_prob[0] = 0.0
    core.red_role_switch_prob[0] = 0.0
    return core, env


def _run_bt(core, opponent: str = "OP11") -> tuple:
    core._opponent_key[0] = opponent
    tx, ty = core._get_bt_targets()
    roles = core.bt_red_role[0].tolist()
    return roles, tx[0].tolist(), ty[0].tolist()


class TestOP5NotBlindRush(unittest.TestCase):
    """OP5 should react to own-flag theft instead of everyone rushing the enemy flag."""

    def test_flag_retrieval_when_own_flag_stolen(self) -> None:
        core, _ = _make_core("OP5")
        core.red_flag_pos[0, 0] = 10.0
        core.red_flag_pos[0, 1] = 10.0
        core.bt_role_lock_ticks[0] = 0
        roles, _, _ = _run_bt(core, "OP5")
        self.assertIn(
            ROLE_FLAG_RETR,
            roles,
            f"OP5 should retrieve stolen flag, got roles={roles}",
        )


class TestOP6DefensiveIdentity(unittest.TestCase):
    """OP6 (turtle) disables counter-capture — hopeless chase should not flip to COUNTER."""

    def test_no_counter_when_infeasible(self) -> None:
        core, _ = _make_core("OP6", red_score=0, blue_score=1)
        core.blue_carrying[0, 0] = True
        core.blue_x[0, 0] = 1.0
        core.blue_y[0, 0] = 10.0
        core.red_x[0, 0] = 18.0
        core.red_y[0, 0] = 5.0
        core.red_x[0, 1] = 18.0
        core.red_y[0, 1] = 15.0
        core.bt_role_lock_ticks[0] = 0
        roles, _, _ = _run_bt(core, "OP6")
        self.assertNotIn(ROLE_COUNTER, roles)


class TestOP7EscortCoordination(unittest.TestCase):
    def test_escort_with_carrier(self) -> None:
        core, _ = _make_core("OP7")
        core.red_carrying[0, 0] = True
        core.red_x[0, 0] = 14.0
        core.red_y[0, 0] = 10.0
        core.blue_x[0, 0] = 13.0
        core.blue_y[0, 0] = 10.0
        roles, _, _ = _run_bt(core, "OP7")
        self.assertIn(ROLE_ESCORT, roles)


class TestOP10EscortInterpose(unittest.TestCase):
    def test_escort_role_with_carrier(self) -> None:
        core, _ = _make_core("OP10")
        core.red_carrying[0, 0] = True
        core.red_x[0, 0] = 14.0
        core.red_y[0, 0] = 10.0
        core.blue_x[0, 0] = 13.0
        core.blue_y[0, 0] = 10.0
        roles, _, _ = _run_bt(core, "OP10")
        self.assertIn(ROLE_ESCORT, roles)


class TestAllOP5PlusValidActions(unittest.TestCase):
    def test_dispatch_bounds_for_curriculum(self) -> None:
        for opp in ("OP5", "OP6", "OP7", "OP8", "OP9", "OP10", "OP11", "OP12"):
            with self.subTest(opponent=opp):
                core, _ = _make_core(opp)
                core._assign_scripted_targets_by_role("red")
                tx = core._debug_red_target_x
                ty = core._debug_red_target_y
                max_x = float(core.cols - 1)
                max_y = float(core.rows - 1)
                self.assertFalse(torch.isnan(tx).any())
                self.assertFalse(torch.isnan(ty).any())
                self.assertTrue((tx >= 0.0).all() and (tx <= max_x).all())
                self.assertTrue((ty >= 0.0).all() and (ty <= max_y).all())


class TestVectorizedEnvIsolation(unittest.TestCase):
    """Tactical state must not leak between batched environments."""

    def test_mixed_opponent_roles_isolated(self) -> None:
        from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig  # type: ignore[import]

        cfg = GPUFieldConfig(
            n_envs=2,
            max_blue_agents=2,
            max_red_agents=2,
            map_layout="map_b",
            max_decision_steps=400,
            aquaticus_profile=True,
            rules_profile="OURS",
            device="cpu",
            seed=7,
        )
        env = GPUCTFVecEnv(cfg)
        env.reset()
        core = env.core
        core.set_next_opponent("SCRIPTED", "OP5", env_indices=[0])
        core.set_next_opponent("SCRIPTED", "OP12", env_indices=[1])

        # Distinct flag states per env.
        core.red_flag_pos[0, 0] = 10.0
        core.red_flag_pos[0, 1] = 10.0
        core.red_flag_pos[1, 0] = float(core.red_flag_home[1, 0])
        core.red_flag_pos[1, 1] = float(core.red_flag_home[1, 1])
        core.blue_carrying[1, 0] = True
        core.blue_x[1, 0] = 1.0
        core.blue_y[1, 0] = 10.0
        core.red_x[1, 0] = 18.0
        core.red_y[1, 0] = 5.0
        core.red_x[1, 1] = 18.0
        core.red_y[1, 1] = 15.0
        core.bt_role_lock_ticks[:] = 0

        core._get_bt_targets()
        roles0 = core.bt_red_role[0].tolist()
        roles1 = core.bt_red_role[1].tolist()
        self.assertIn(ROLE_FLAG_RETR, roles0)
        self.assertIn(ROLE_COUNTER, roles1)
        self.assertNotEqual(roles0, roles1)


class TestStrategicDiversity(unittest.TestCase):
    """Same match state should yield different tactics for different profiles."""

    def _infeasible_leading_state(self, opponent: str):
        core, _ = _make_core(opponent, red_score=1, blue_score=0)
        core.blue_carrying[0, 0] = True
        core.blue_x[0, 0] = 1.0
        core.blue_y[0, 0] = 10.0
        core.red_x[0, 0] = 18.0
        core.red_y[0, 0] = 5.0
        core.red_x[0, 1] = 18.0
        core.red_y[0, 1] = 15.0
        core.bt_role_lock_ticks[0] = 0
        return core

    def test_op5_vs_op12_differ_when_leading_infeasible(self) -> None:
        roles5, _, _ = _run_bt(self._infeasible_leading_state("OP5"), "OP5")
        roles12, _, _ = _run_bt(self._infeasible_leading_state("OP12"), "OP12")
        self.assertNotIn(ROLE_COUNTER, roles5)
        self.assertIn(ROLE_COUNTER, roles12)


class TestRoleHysteresisCurriculum(unittest.TestCase):
    def test_op5_role_lock_prevents_flip(self) -> None:
        core, _ = _make_core("OP5")
        core.bt_red_role[0, 0] = ROLE_DEFENDER
        core.bt_role_lock_ticks[0, 0] = 15
        _run_bt(core, "OP5")
        self.assertEqual(int(core.bt_red_role[0, 0].item()), ROLE_DEFENDER)


class TestBTDispatchMask(unittest.TestCase):
    def test_all_curriculum_keys_use_bt(self) -> None:
        for key in ("OP5", "OP6", "OP7", "OP8", "OP9", "OP10", "OP11", "OP12"):
            self.assertTrue(is_bt_opponent(key))


class TestFlagStateTransition(unittest.TestCase):
    def test_role_changes_after_carrier_removed(self) -> None:
        core, _ = _make_core("OP7")
        core.red_carrying[0, 0] = True
        core.red_x[0, 0] = 14.0
        core.red_y[0, 0] = 10.0
        core.bt_role_lock_ticks[0] = 0
        roles_before, _, _ = _run_bt(core, "OP7")
        self.assertIn(ROLE_ESCORT, roles_before)
        core.red_carrying[0] = False
        core.bt_role_lock_ticks[0] = 0
        roles_after, _, _ = _run_bt(core, "OP7")
        self.assertNotIn(ROLE_ESCORT, roles_after)


class TestInterceptorFeasibleOP8(unittest.TestCase):
    def test_interceptor_on_feasible_carry(self) -> None:
        core, _ = _make_core("OP8")
        core.blue_carrying[0, 0] = True
        core.blue_x[0, 0] = 10.0
        core.blue_y[0, 0] = 10.0
        core.red_x[0, 0] = 8.0
        core.red_y[0, 0] = 9.0
        core.red_x[0, 1] = 8.0
        core.red_y[0, 1] = 11.0
        roles, _, _ = _run_bt(core, "OP8")
        self.assertIn(ROLE_INTERCEPTOR, roles)


class TestDefenderRoleOP6(unittest.TestCase):
    def test_defender_when_intruder_on_own_half(self) -> None:
        core, _ = _make_core("OP6")
        midline = float(core.cols) * 0.5
        core.blue_x[0, 0] = midline + 2.0
        core.blue_y[0, 0] = 10.0
        core.bt_role_lock_ticks[0] = 0
        roles, _, _ = _run_bt(core, "OP6")
        self.assertIn(ROLE_DEFENDER, roles)


if __name__ == "__main__":
    unittest.main()
