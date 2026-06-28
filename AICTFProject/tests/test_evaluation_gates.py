from __future__ import annotations

import unittest
from argparse import Namespace

from rl.custom_ppo.probe_result import PROBE_SUCCESS, CounterfactualProbeResult, GradientProbeResult, WeightProbeResult
from rl.evaluation.gates import build_summary


class EvaluationGateTests(unittest.TestCase):
    def test_verdict_preserves_inconclusive_when_navigation_evidence_missing(self) -> None:
        args = Namespace(obs_weight_threshold=1e-4, gradient_threshold=0.0, counterfactual_action_threshold=0.01, counterfactual_kl_threshold=1e-5, navigation_improvement_threshold=0.1, route_difference_threshold=0.1, minimum_win_rate=0.6, competence_retention_tolerance=0.05, saturation_win_rate=0.95, maps=["map_a_open"], episodes=2)
        probes = {
            "candidate_weights": WeightProbeResult(status=PROBE_SUCCESS, has_obstacle_channel=True, cnn_channels=8, obstacle_weight_l2=1.0),
            "candidate_gradient": GradientProbeResult(status=PROBE_SUCCESS, obstacle_gradient_l2=1.0),
            "candidate_counterfactual": CounterfactualProbeResult(status=PROBE_SUCCESS, states_evaluated=1, mean_action_kl=1.0, mean_logit_l2=1.0, argmax_action_change_rate=1.0),
        }
        episodes = [
            {"policy": "baseline", "map": "map_a_open", "win": 1, "collision_metric_source": "unavailable", "stuck_metric_source": "unavailable", "route_metric_source": "unavailable"},
            {"policy": "candidate", "map": "map_a_open", "win": 1, "collision_metric_source": "unavailable", "stuck_metric_source": "unavailable", "route_metric_source": "unavailable"},
        ]
        conditions = [{"policy": "candidate", "map": "map_a_open", "win": 1, "route_metric_source": "unavailable"}]
        summary = build_summary(args, probes, episodes, conditions)
        self.assertEqual(summary["verdict"], "INCONCLUSIVE: ADD MISSING TELEMETRY OR MORE EPISODES")


if __name__ == "__main__":
    unittest.main()
