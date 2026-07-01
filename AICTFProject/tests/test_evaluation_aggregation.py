from __future__ import annotations

import unittest

from rl.evaluation.aggregation import aggregate_conditions


class EvaluationAggregationTests(unittest.TestCase):
    def test_missing_metrics_remain_none_not_zero(self) -> None:
        rows = [
            {"policy": "candidate", "map": "map", "resolved_opponent": "OP8", "blue_score": 1.0, "red_score": 0.0, "win": 1, "collision_metric_source": "unavailable", "stuck_metric_source": "unavailable", "route_metric_source": "unavailable"}
        ]
        summary = aggregate_conditions(rows)[0]
        self.assertIsNone(summary["wall_collisions"])
        self.assertEqual(summary["collision_metric_source"], "unavailable")


if __name__ == "__main__":
    unittest.main()
