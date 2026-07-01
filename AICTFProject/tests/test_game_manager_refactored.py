from __future__ import annotations

import unittest
from game_manager import GameManager, FlagState, DynamicsManager, MetricTracker, RewardRouter


class DummyAgent:
    def __init__(self, agent_id: int, side: str = "blue") -> None:
        self.agent_id = agent_id
        self.side = side
        self.unique_id = f"{self.side}_{self.agent_id}"
        self.float_pos = (5.0, 10.0)
        self._carrying = False
        self._carrying_scored = False

    def isEnabled(self) -> bool:
        return True

    def setCarryingFlag(self, val: bool, scored: bool = False) -> None:
        self._carrying = val
        self._carrying_scored = scored


class GameManagerRefactoredTests(unittest.TestCase):
    def test_game_manager_initialization_and_properties(self) -> None:
        gm = GameManager(cols=20, rows=20)
        
        # Test basic dimensions
        self.assertEqual(gm.cols, 20)
        self.assertEqual(gm.rows, 20)
        
        # Verify delegates are initialized
        self.assertIsInstance(gm.blue_flag, FlagState)
        self.assertIsInstance(gm.red_flag, FlagState)
        self.assertIsInstance(gm.dynamics_manager, DynamicsManager)
        self.assertIsInstance(gm.metric_tracker, MetricTracker)
        self.assertIsInstance(gm.reward_router, RewardRouter)
        
        # Verify property facades
        self.assertEqual(gm.blue_flag_position, (2, 10))
        self.assertEqual(gm.red_flag_position, (17, 10))
        self.assertFalse(gm.blue_flag_taken)
        self.assertFalse(gm.red_flag_taken)
        self.assertIsNone(gm.blue_flag_carrier)
        
        # Verify property setter
        gm.blue_flag_position = (5, 5)
        self.assertEqual(gm.blue_flag.position, (5, 5))
        self.assertEqual(gm.blue_flag_position, (5, 5))

    def test_flag_pickup_and_scoring_delegation(self) -> None:
        gm = GameManager(cols=20, rows=20)
        agent = DummyAgent(agent_id=1, side="blue")
        
        # Place agent close to red flag position (17, 10)
        agent.float_pos = (17.2, 9.8)
        
        # Pickup enemy flag
        success = gm.try_pickup_enemy_flag(agent)
        self.assertTrue(success)
        self.assertTrue(gm.red_flag_taken)
        self.assertEqual(gm.red_flag_carrier, agent)
        self.assertTrue(agent._carrying)
        
        # Carrier positions update on tick sanity check
        agent.float_pos = (12.0, 8.0)
        gm.sanity_check_flags()
        self.assertEqual(gm.red_flag_position, (12, 8))
        
        # Bring it home to score at blue flag home (2, 10)
        agent.float_pos = (2.1, 9.9)
        gm.sanity_check_flags()
        
        score_success = gm.try_score_if_carrying_and_home(agent)
        self.assertTrue(score_success)
        self.assertEqual(gm.blue_score, 1)
        self.assertFalse(gm.red_flag_taken)
        self.assertEqual(gm.red_flag_position, gm.red_flag_home)
        self.assertFalse(agent._carrying)
        self.assertTrue(agent._carrying_scored)

    def test_dynamics_and_metrics_delegation(self) -> None:
        gm = GameManager(cols=20, rows=20)
        
        # Dynamics config
        gm.set_dynamics_config({"blue_speed_mult": 1.25, "red_speed_mult": 0.85})
        self.assertEqual(gm.get_team_speed_multiplier("blue"), 1.25)
        self.assertEqual(gm.get_team_speed_multiplier("red"), 0.85)
        
        summary = gm.get_episode_dynamics_summary()
        self.assertEqual(summary["blue_speed_mult"], 1.25)
        self.assertEqual(summary["red_speed_mult"], 0.85)
        
        # Metrics
        gm.record_tick_metrics(collision_delta=2, near_miss_delta=5, blue_inter_robot_dist=4.5)
        self.assertEqual(gm.collision_count_this_episode, 2)
        self.assertEqual(gm.near_miss_count_this_episode, 5)
        self.assertEqual(gm.blue_inter_robot_distances, [4.5])


if __name__ == "__main__":
    unittest.main()
