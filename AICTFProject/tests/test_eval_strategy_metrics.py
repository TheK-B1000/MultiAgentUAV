import unittest

import numpy as np

from plot.eval_rollout import (
    _safe_pearson,
    compute_episode_coordination_metrics,
    compute_aggregates,
    _strategy_phase_from_global_state,
)


class EvalStrategyMetricsTests(unittest.TestCase):
    def test_safe_pearson_constant_is_nan(self) -> None:
        x = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        self.assertFalse(np.isfinite(_safe_pearson(x, x)))

    def test_compute_episode_coordination_metrics_locked_macros(self) -> None:
        # Two agents: perfectly opposite alternating macros -> r -> -1.
        t = 20
        traj = np.zeros((t, 2, 2), dtype=np.int64)
        traj[:, 0, 0] = np.arange(t, dtype=np.int64) % 2
        traj[:, 1, 0] = 1 - traj[:, 0, 0]
        m = compute_episode_coordination_metrics(traj)
        self.assertAlmostEqual(m["coord_head0_team_agreement_rate"], 0.0)
        self.assertAlmostEqual(m["coord_trajectory_steps"], float(t))
        self.assertLess(m["coord_pairwise_head0_pearson_mean"], -0.99)

    def test_compute_episode_coordination_metrics_perfect_team_sync(self) -> None:
        t = 15
        macros = (np.arange(t, dtype=np.int64) % 3).reshape(t, 1)
        h1 = np.zeros((t, 1), dtype=np.int64)
        a0 = np.concatenate([macros, h1], axis=1)
        a1 = np.concatenate([macros, h1], axis=1)
        traj = np.stack([a0, a1], axis=1)
        m = compute_episode_coordination_metrics(traj)
        self.assertAlmostEqual(m["coord_head0_team_agreement_rate"], 1.0)
        self.assertAlmostEqual(m["coord_full_action_team_agreement_rate"], 1.0)
        self.assertAlmostEqual(m["coord_pairwise_head0_pearson_mean"], 1.0)

    def test_compute_aggregates_includes_coord_diagnostics(self) -> None:
        episodes = [
            {
                "success": 1,
                "blue_score": 2,
                "red_score": 0,
                "steps": 10,
                "return": 1.0,
                "zone_coverage": 0.5,
                "collision_free": 1,
                "win_margin": 2,
                "coord_pairwise_head0_pearson_mean": 0.8,
                "coord_head0_team_agreement_rate": 0.9,
                "coord_full_action_team_agreement_rate": 0.7,
                "coord_trajectory_steps": 100.0,
            },
            {
                "success": 1,
                "blue_score": 1,
                "red_score": 0,
                "steps": 12,
                "return": 0.5,
                "zone_coverage": 0.6,
                "collision_free": 1,
                "win_margin": 1,
                "coord_pairwise_head0_pearson_mean": 0.2,
                "coord_head0_team_agreement_rate": 0.4,
                "coord_full_action_team_agreement_rate": 0.3,
                "coord_trajectory_steps": 120.0,
            },
        ]
        agg = compute_aggregates(episodes)
        self.assertAlmostEqual(agg["coord_pairwise_head0_pearson_mean_mean"], 0.5)
        self.assertAlmostEqual(agg["coord_head0_team_agreement_rate_mean"], 0.65)
        self.assertAlmostEqual(agg["coord_full_action_team_agreement_rate_mean"], 0.5)
        self.assertAlmostEqual(agg["coord_trajectory_steps_mean"], 110.0)

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
