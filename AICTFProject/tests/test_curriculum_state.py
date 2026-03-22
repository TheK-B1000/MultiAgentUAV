import unittest

from rl.curriculum import CurriculumConfig, CurriculumState


class CurriculumStateTests(unittest.TestCase):
    def test_advancing_phase_clears_next_phase_window(self):
        cfg = CurriculumConfig(
            phases=["OP1", "OP2", "OP3"],
            min_episodes={"OP1": 1, "OP2": 1, "OP3": 1},
            min_winrate={"OP1": 1.0, "OP2": 1.0, "OP3": 1.0},
            winrate_window=5,
            required_win_by={"OP1": 0, "OP2": 0, "OP3": 0},
            elo_margin=0.0,
        )
        state = CurriculumState(cfg)
        state.recent_results["OP2"].extend([1.0, 0.0, 1.0])
        state.phase_episode_count = 1
        state.record_result("OP1", 1.0)

        advanced = state.advance_if_ready(
            learner_rating=1200.0,
            opponent_rating=1200.0,
            win_by=1,
            skip_elo_check=True,
        )

        self.assertTrue(advanced)
        self.assertEqual(state.phase, "OP2")
        self.assertEqual(len(state.recent_results["OP2"]), 0)
        self.assertEqual(state.phase_episode_count, 0)


if __name__ == "__main__":
    unittest.main()
