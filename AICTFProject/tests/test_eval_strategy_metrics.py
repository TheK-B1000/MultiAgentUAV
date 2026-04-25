import unittest

from plot.eval_rollout import _strategy_phase_from_global_state, compute_aggregates


class EvalStrategyMetricsTests(unittest.TestCase):
    def test_strategy_phase_uses_global_state_flag_bits(self) -> None:
        base = [0.0] * 14
        self.assertEqual(_strategy_phase_from_global_state(base), "neutral")
        blue_attack = base[:]
        blue_attack[11] = 1.0
        self.assertEqual(_strategy_phase_from_global_state(blue_attack), "blue_attack")
        blue_defense = base[:]
        blue_defense[10] = 1.0
        self.assertEqual(_strategy_phase_from_global_state(blue_defense), "blue_defense")

    def test_compute_aggregates_includes_strategy_diagnostics(self) -> None:
        episodes = [
            {
                "success": 1,
                "blue_score": 2,
                "red_score": 0,
                "steps": 12,
                "return": 1.5,
                "zone_coverage": 0.8,
                "collision_free": 1,
                "win_margin": 2,
                "strategy_switch_rate": 0.0,
                "strategy_resample_rate": 0.25,
                "strategy_unique_count": 1,
                "strategy_entropy_mean": 0.7,
                "strategy_occupancy_0": 1.0,
                "strategy_occupancy_1": 0.0,
                "strategy_phase_neutral_occupancy_0": 1.0,
                "strategy_phase_neutral_occupancy_1": 0.0,
            },
            {
                "success": 0,
                "blue_score": 0,
                "red_score": 1,
                "steps": 20,
                "return": -0.5,
                "zone_coverage": 0.4,
                "collision_free": 0,
                "win_margin": -1,
                "strategy_switch_rate": 0.5,
                "strategy_resample_rate": 0.75,
                "strategy_unique_count": 2,
                "strategy_entropy_mean": 0.9,
                "strategy_occupancy_0": 0.25,
                "strategy_occupancy_1": 0.75,
                "strategy_phase_neutral_occupancy_0": 0.5,
                "strategy_phase_neutral_occupancy_1": 0.5,
            },
        ]

        agg = compute_aggregates(episodes)

        self.assertAlmostEqual(agg["strategy_switch_rate_mean"], 0.25)
        self.assertAlmostEqual(agg["strategy_resample_rate_mean"], 0.5)
        self.assertAlmostEqual(agg["strategy_unique_count_mean"], 1.5)
        self.assertAlmostEqual(agg["strategy_entropy_step_mean"], 0.8)
        self.assertAlmostEqual(agg["strategy_occupancy_0_mean"], 0.625)
        self.assertAlmostEqual(agg["strategy_occupancy_1_mean"], 0.375)
        self.assertAlmostEqual(agg["strategy_phase_neutral_occupancy_0_mean"], 0.75)
        self.assertAlmostEqual(agg["strategy_phase_neutral_occupancy_1_mean"], 0.25)


if __name__ == "__main__":
    unittest.main()
