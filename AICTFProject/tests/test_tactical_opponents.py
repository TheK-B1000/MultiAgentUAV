"""Deterministic scenario tests for tactical score/time awareness in OP8/9/10.

Each test creates a minimal BatchedCTFCore with a fixed game state and asserts
that the scripted red team responds to the tactical context (score, time).
Tests are fast (CPU, n_envs=1, no rollout) and fully deterministic.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _make_core(opponent: str, *, blue_score: int = 0, red_score: int = 0,
               step: int = 0, max_steps: int = 400, seed: int = 42):
    """Return a 2v2 core (n_envs=1) with fixed scores/time and the given opponent."""
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

    # Assign opponent key so the behavioral masks fire.
    core._opponent_key[0] = opponent
    # Guarantee guardian is index 0 (no role flip).
    core.red_script_role_flip[0] = False
    # Ensure all agents alive.
    core.red_alive[0] = True
    core.blue_alive[0] = True

    return core, env


class TestOP8AdaptiveBlock(unittest.TestCase):
    """OP8 block fraction: 0.5 when leading, 0.7 when trailing (BT interceptor)."""

    def _interceptor_target_x(self, *, red_score: int, blue_score: int) -> float:
        """Return interceptor agent target_x after one BT target update."""
        from gpu_env._core._bt_red import ROLE_INTERCEPTOR

        core, env = _make_core(
            "OP8", red_score=red_score, blue_score=blue_score, step=100,
        )
        core.blue_carrying[0, 0] = True
        core.blue_x[0, 0] = 8.0
        core.blue_y[0, 0] = 10.0
        core._assign_scripted_targets_by_role("red")
        roles = core.bt_red_role[0].tolist()
        if ROLE_INTERCEPTOR not in roles:
            self.skipTest("No interceptor assigned in this geometry")
        agent = roles.index(ROLE_INTERCEPTOR)
        return float(core._debug_red_target_x[0, agent].item())

    def test_trailing_guardian_closer_to_home_than_leading(self) -> None:
        """When trailing, interceptor block point is closer to home than when leading."""
        leading_tx = self._interceptor_target_x(red_score=1, blue_score=0)
        trailing_tx = self._interceptor_target_x(red_score=0, blue_score=1)
        # Blue home is left side (low x). Trailing block (frac=0.7) is closer to home → lower x.
        self.assertLess(trailing_tx, leading_tx,
                        f"Trailing block ({trailing_tx:.2f}) should be closer to home than leading ({leading_tx:.2f})")

    def test_leading_block_fraction_approx_half(self) -> None:
        """Leading: interceptor should be ~50% of carrier-to-home distance from carrier."""
        from gpu_env._core._bt_red import ROLE_INTERCEPTOR

        core, env = _make_core("OP8", red_score=1, blue_score=0, step=100)
        core.blue_carrying[0, 0] = True
        core.blue_x[0, 0] = 8.0
        core.blue_y[0, 0] = 10.0
        core._assign_scripted_targets_by_role("red")
        roles = core.bt_red_role[0].tolist()
        if ROLE_INTERCEPTOR not in roles:
            self.skipTest("No interceptor assigned in this geometry")
        agent = roles.index(ROLE_INTERCEPTOR)
        tx = float(core._debug_red_target_x[0, agent].item())
        carrier_x = 8.0
        home_x = float(core.blue_flag_home[0, 0].item())
        denom = home_x - carrier_x
        if abs(denom) < 1e-3:
            self.skipTest("Degenerate geometry")
        frac = (tx - carrier_x) / denom
        self.assertAlmostEqual(frac, 0.5, delta=0.2,
                               msg=f"Leading block fraction should be ~0.5, got {frac:.3f}")


class TestOP9SplitLaneFeint(unittest.TestCase):
    """OP9's striker runs a split-lane feint, not a straight line at the flag.

    The feint is the OP9 niche in the profile table (``lane_amplitude_frac``
    is the highest of any level), and it is unconditional: the route does not
    depend on the score or on how much of the episode has elapsed.
    """

    def _striker_target_x(self, *, step: int, red_score: int, blue_score: int) -> float:
        """Return attacker (red agent 1) target_x after one BT target update."""
        from gpu_env._core._bt_red import ROLE_ATTACKER

        core, env = _make_core(
            "OP9", red_score=red_score, blue_score=blue_score,
            step=step, max_steps=400,
        )
        core.red_attacker_style[0] = 0
        core.red_coordinated_attack[0] = False
        core.red_x[0, 1] = 12.0
        core.red_y[0, 1] = 5.0
        core.blue_x[0, 0] = 13.0
        core.blue_y[0, 0] = 5.0
        core._assign_scripted_targets_by_role("red")
        roles = core.bt_red_role[0].tolist()
        agent = roles.index(ROLE_ATTACKER) if ROLE_ATTACKER in roles else 1
        return float(core._debug_red_target_x[0, agent].item())

    def _enemy_flag_x(self) -> float:
        core, env = _make_core("OP9")
        return float(core.blue_flag_pos[0, 0].item())

    def test_striker_deviates_from_direct_flag_line(self) -> None:
        efx = self._enemy_flag_x()
        tx = self._striker_target_x(step=320, red_score=0, blue_score=1)
        self.assertNotAlmostEqual(tx, efx, delta=0.5,
                                  msg=f"OP9 striker should deviate from flag ({efx:.1f}), got {tx:.1f}")

    def test_feint_does_not_depend_on_score_or_phase(self) -> None:
        early_trailing = self._striker_target_x(step=50, red_score=0, blue_score=1)
        late_trailing = self._striker_target_x(step=320, red_score=0, blue_score=1)
        late_leading = self._striker_target_x(step=320, red_score=1, blue_score=0)
        self.assertAlmostEqual(early_trailing, late_trailing, delta=0.01)
        self.assertAlmostEqual(late_trailing, late_leading, delta=0.01)


class TestDebugTargetsStored(unittest.TestCase):
    """_debug_red_target_x/y must be populated after _assign_scripted_targets_by_role."""

    def test_debug_targets_exist(self) -> None:
        core, env = _make_core("OP8", step=10)
        core._assign_scripted_targets_by_role("red")
        self.assertTrue(hasattr(core, "_debug_red_target_x"))
        self.assertTrue(hasattr(core, "_debug_red_target_y"))
        self.assertEqual(tuple(core._debug_red_target_x.shape), (1, 2))

    def test_tactical_context_extract(self) -> None:
        from gpu_env._core._tactical_context import extract_tactical_context
        # step=320 > 0.75 * 400 → late_game is True in both core and TacticalContext.
        core, env = _make_core("OP9", red_score=1, blue_score=2, step=320)
        core._assign_scripted_targets_by_role("red")
        tc = extract_tactical_context(core, env_idx=0)
        self.assertEqual(tc.red_score, 1)
        self.assertEqual(tc.blue_score, 2)
        self.assertTrue(tc.red_trailing)
        self.assertFalse(tc.red_leading)
        # step=320, max_steps=400 → time_remaining_frac = 0.20 < 0.25 → late_game True
        self.assertTrue(tc.late_game, f"Expected late_game=True, got {tc.time_remaining_frac:.3f}")
        self.assertIsNotNone(tc.red_target_x)


if __name__ == "__main__":
    unittest.main()
