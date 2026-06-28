from __future__ import annotations

import unittest
from argparse import Namespace

from rl.evaluation.matched_seed import matched_seed_evaluation


class EvaluationMatchedSeedTests(unittest.TestCase):
    def test_loop_order_is_map_opponent_episode_policy(self) -> None:
        calls = []
        args = Namespace(maps=["m1", "m2"], opponents=["op"], episodes=2, seed_start=10, device="cpu", max_decision_steps=4)
        def run_episode_fn(**kwargs):
            calls.append((kwargs["map_name"], kwargs["opponent"], kwargs["seed"], kwargs["policy_name"]))
            return {"policy": kwargs["policy_name"], "map": kwargs["map_name"], "resolved_opponent": kwargs["opponent"], "blue_score": 1.0, "red_score": 0.0}
        rows = matched_seed_evaluation(args, "b", "c", 2, run_episode_fn=run_episode_fn, validate_opponent_name=lambda value: value.upper())
        self.assertEqual(len(rows), 8)
        self.assertEqual(calls[0], ("m1", "OP", 10, "baseline"))
        self.assertEqual(calls[1], ("m1", "OP", 10, "candidate"))
        self.assertEqual(calls[2], ("m1", "OP", 11, "baseline"))
        self.assertEqual(calls[-1], ("m2", "OP", 11, "candidate"))


if __name__ == "__main__":
    unittest.main()
