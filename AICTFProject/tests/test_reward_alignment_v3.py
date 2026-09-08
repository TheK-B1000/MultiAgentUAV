"""Reward-alignment ordering constraints for Reward V3.

These tests pin the *intended* economy: objective progress must dominate
farmable / passive alternatives. They are deliberately independent of the
live V3 training processes.

Two layers:
  1. Static composition math (instant; no env).
  2. Optional scripted trajectory gate (marked slow; run by the scan script).

The scan script experiments/scan_reward_v3_economy.py is the authoritative
whole-trajectory measurement; these tests keep regressions from reopening
the same exploit class.
"""
from __future__ import annotations

import math
import unittest

from game_manager import (
    ENEMY_MAV_KILL_REWARD,
    SPARSE_FLAG_CAPTURE_POINTS,
    SPARSE_TAG_NO_FLAG_POINTS,
    WIN_TEAM_REWARD,
)


# Copied V3 knobs — must stay in sync with run_reward_v3_probe.py
V3_TAG_NOFLAG = 0.0
V3_TAG_CARRIER = 0.0
V3_FAILED = -0.004
GAMMA = 0.995
HORIZON = 240
SPARSE_W = 1.0
DENSE_W = 0.25


def sparse_norm(points: float) -> float:
    return SPARSE_W * (points / 100.0)


def compose_raw(*, terminal=0.0, offense=0.0, failure=0.0,
                pbrs=0.0, team=0.0, sparse_points=0.0) -> float:
    """Pre-tanh composed raw matching _reward_total."""
    return (
        terminal
        + offense
        + failure
        + DENSE_W * (pbrs + team)
        + sparse_norm(sparse_points)
    )


class RewardAlignmentV3StaticTests(unittest.TestCase):
    def test_v3_zeros_sparse_tag_family(self):
        self.assertEqual(V3_TAG_NOFLAG, 0.0)
        self.assertEqual(V3_TAG_CARRIER, 0.0)
        # Baseline still documents the exploit magnitude that V3 closed.
        self.assertEqual(SPARSE_TAG_NO_FLAG_POINTS, 100.0)
        self.assertEqual(SPARSE_FLAG_CAPTURE_POINTS, 100.0)

    def test_discounted_terminal_is_materially_smaller_than_face_value(self):
        disc = WIN_TEAM_REWARD * (GAMMA ** HORIZON)
        self.assertLess(disc, 0.35)
        self.assertGreater(disc, 0.25)  # ~0.301 at gamma=0.995, T=240

    def test_one_immediate_kill_reward_rivals_discounted_terminal(self):
        """Documents the hidden offense channel: enemy_mav_kill is NOT /100."""
        disc_term = WIN_TEAM_REWARD * (GAMMA ** HORIZON)
        self.assertGreater(ENEMY_MAV_KILL_REWARD, disc_term)
        # At design tag rate 17, kill-farm (or being-tagged cost) dominates.
        self.assertGreater(ENEMY_MAV_KILL_REWARD * 17.0, WIN_TEAM_REWARD * 5.0)

    def test_winning_capture_raw_exceeds_passive_loss_raw(self):
        """Ordering on composed RAW (pre-tanh), idealized single-event episodes."""
        win_capture = compose_raw(
            terminal=WIN_TEAM_REWARD,
            offense=0.5,  # flag_carry_home
            sparse_points=SPARSE_FLAG_CAPTURE_POINTS,
        )
        passive_loss = compose_raw(
            terminal=-1.0,
            # camping while red scores once against us
            offense=-0.5,
            sparse_points=-SPARSE_FLAG_CAPTURE_POINTS,
            team=0.03 * 60,  # ~60 steps of defense-presence while flag stolen
            failure=V3_FAILED * 20,
        )
        self.assertGreater(win_capture, passive_loss)

    def test_capture_raw_exceeds_routine_tag_sequence_under_v3_sparse(self):
        """With sparse tags zeroed, a capture must beat a pure tag farm on sparse alone.

        NOTE: offense enemy_mav_kill still pays +0.5/tag — this test isolates the
        sparse ledger. The scan script measures the full composed path.
        """
        capture_sparse = sparse_norm(SPARSE_FLAG_CAPTURE_POINTS)
        tag_farm_sparse = sparse_norm(V3_TAG_NOFLAG) * 20.0
        self.assertGreater(capture_sparse, tag_farm_sparse)

    def test_failed_commit_v3_budget_bounded(self):
        """At the measured 184 events/ep, V3 failure cost stays under 1.0."""
        self.assertLess(abs(V3_FAILED * 184.0), 1.0)
        self.assertGreater(abs(V3_FAILED * 184.0), 0.5)  # still noticeable

    def test_pbrs_gamma_matches_ppo_gamma_contract(self):
        from gpu_env._config import RewardConfig
        from rl.config.ppo_config import PPOConfig

        self.assertAlmostEqual(RewardConfig().pbrs_gamma, PPOConfig().gamma)

    def test_objective_progress_beats_inactivity_on_synthetic_discounted(self):
        """Synthetic discounted returns: early dense camping vs late win+capture."""
        # Camper: +0.0075 team*dense per step for 240 steps, then lose terminal
        camp_step = DENSE_W * 0.03
        camp_disc = sum((GAMMA ** t) * camp_step for t in range(HORIZON))
        camp_disc += (GAMMA ** (HORIZON - 1)) * (-1.0)

        # Attacker: small negative offense noise early, capture+win at t=180
        att = 0.0
        for t in range(180):
            att += (GAMMA ** t) * (-0.001)  # mild failed-commit trickle
        event_raw = compose_raw(
            terminal=WIN_TEAM_REWARD,
            offense=0.5,
            sparse_points=SPARSE_FLAG_CAPTURE_POINTS,
        )
        att += (GAMMA ** 180) * event_raw
        self.assertGreater(att, camp_disc)


class RewardAlignmentV3HiddenChannelTests(unittest.TestCase):
    def test_enemy_mav_kill_still_nonzero_under_default_config(self):
        """V3 did not close this channel — pin so future budgets cannot forget it."""
        from gpu_env._config import RewardConfig

        cfg = RewardConfig()
        self.assertEqual(cfg.enemy_mav_kill_reward, ENEMY_MAV_KILL_REWARD)
        self.assertGreater(abs(cfg.enemy_mav_kill_reward), 0.0)
        # Sparse tags ARE zeroable; kill reward is a separate path.
        cfg2 = RewardConfig(sparse_tag_no_flag_points=0.0, sparse_tag_with_flag_points=0.0)
        self.assertEqual(cfg2.enemy_mav_kill_reward, ENEMY_MAV_KILL_REWARD)

    def test_sparse_oob_path_still_uses_module_constant(self):
        """Documentation pin: cfg.sparse_oob_points is not yet wired into _sparse_reward_points."""
        import inspect
        from gpu_env._core import _rewards as rewards_mod

        src = inspect.getsource(rewards_mod._RewardsMixin._sparse_reward_points)
        self.assertIn("SPARSE_OOB_POINTS", src)
        # The cfg knob exists for instrumentation / future budgeting.
        from gpu_env._config import RewardConfig
        self.assertEqual(RewardConfig().sparse_oob_points, -100.0)


if __name__ == "__main__":
    unittest.main()
