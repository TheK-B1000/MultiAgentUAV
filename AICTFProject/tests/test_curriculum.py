"""Guards for the restored Jacob paper curriculum."""

from __future__ import annotations

import unittest

from rl.curriculum import CurriculumConfig, CurriculumState, jacob_paper_curriculum_state, phase_from_tag


class CurriculumTests(unittest.TestCase):
    def test_curriculum_promotes_op1_to_op2_after_gate(self) -> None:
        state = CurriculumState(
            CurriculumConfig(
                phases=["OP1", "OP2", "OP3"],
                min_episodes={"OP1": 2, "OP2": 2, "OP3": 2},
                min_winrate={"OP1": 1.0, "OP2": 0.5, "OP3": 0.5},
                winrate_window=2,
                required_win_by={"OP1": 0, "OP2": 1, "OP3": 1},
            )
        )
        state.phase_episode_count += 1
        state.record_result("OP1", 1.0)
        self.assertFalse(state.advance_if_ready(win_by=0))
        state.phase_episode_count += 1
        state.record_result("OP1", 1.0)
        self.assertTrue(state.advance_if_ready(win_by=0))
        self.assertEqual(state.phase, "OP2")
        self.assertEqual(state.phase_episode_count, 0)

    def test_jacob_curriculum_restores_old_thresholds(self) -> None:
        two_v_two = jacob_paper_curriculum_state(2)
        four_v_four = jacob_paper_curriculum_state(4)
        self.assertEqual(two_v_two.config.min_episodes["OP1"], 200)
        self.assertEqual(two_v_two.recent_results["OP1"].maxlen, 50)
        self.assertEqual(four_v_four.config.min_episodes["OP1"], 350)
        self.assertEqual(four_v_four.recent_results["OP1"].maxlen, 80)
        self.assertEqual(phase_from_tag("OP4"), "OP3")


if __name__ == "__main__":
    unittest.main()
