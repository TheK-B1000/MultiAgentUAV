import unittest

from rl.episode_result import parse_episode_result, scores_from_info


class ScoresFromInfoTests(unittest.TestCase):
    def test_reads_nested_episode_result(self):
        """GPUCTFVecEnv.step_wait shape."""
        info = {"episode_result": {"blue_score": 3, "red_score": 1}}
        self.assertEqual(scores_from_info(info), (3, 1))

    def test_reads_flat_core_info(self):
        """BatchedCTFCore.step shape -- what two-policy stepping actually sees.

        Regression guard: parse_episode_result returns None here, which silently
        produced zero episodes and NaN match scores in every two-policy caller.
        """
        info = {
            "blue_score": 0,
            "red_score": 2,
            "decision_steps": 86,
            "phase": "OP3",
            "opponent_kind": "snapshot",
        }
        self.assertIsNone(parse_episode_result(info))
        self.assertEqual(scores_from_info(info), (0, 2))

    def test_nested_wins_over_flat_when_both_present(self):
        info = {"blue_score": 9, "red_score": 9, "episode_result": {"blue_score": 1, "red_score": 0}}
        self.assertEqual(scores_from_info(info), (1, 0))

    def test_returns_none_when_scores_are_absent(self):
        self.assertIsNone(scores_from_info({"phase": "OP3"}))
        self.assertIsNone(scores_from_info({}))

    def test_returns_none_on_unparseable_scores(self):
        self.assertIsNone(scores_from_info({"blue_score": "n/a", "red_score": 1}))


if __name__ == "__main__":
    unittest.main()
