from __future__ import annotations

import unittest
from argparse import Namespace

from rl.custom_ppo.probe_result import PROBE_ERROR, PROBE_SUCCESS, CounterfactualProbeResult, GradientProbeResult, WeightProbeResult
from rl.evaluation.gates import build_summary


class EvaluationGateTests(unittest.TestCase):
    def _probes(self) -> dict:
        return {
            "candidate_weights": WeightProbeResult(
                status=PROBE_SUCCESS, has_obstacle_channel=True, cnn_channels=8, obstacle_weight_l2=1.0
            ),
            "candidate_gradient": GradientProbeResult(status=PROBE_SUCCESS, obstacle_gradient_l2=1.0),
            "candidate_counterfactual": CounterfactualProbeResult(
                status=PROBE_SUCCESS,
                states_evaluated=1,
                mean_action_kl=1.0,
                mean_logit_l2=1.0,
                argmax_action_change_rate=1.0,
            ),
        }

    def _args(self, **overrides) -> Namespace:
        base = dict(
            obs_weight_threshold=1e-4,
            gradient_threshold=0.0,
            counterfactual_action_threshold=0.01,
            counterfactual_kl_threshold=1e-5,
            navigation_improvement_threshold=0.1,
            route_difference_threshold=0.1,
            minimum_win_rate=0.6,
            competence_retention_tolerance=0.05,
            saturation_win_rate=0.95,
            allow_saturated_pool=False,
            maps=["map_a_open"],
            episodes=2,
        )
        base.update(overrides)
        return Namespace(**base)

    def test_required_gates_ready_even_when_navigation_telemetry_missing(self) -> None:
        episodes = [
            {
                "policy": "baseline",
                "map": "map_a_open",
                "win": 1,
                "collision_metric_source": "unavailable",
                "stuck_metric_source": "unavailable",
                "route_metric_source": "unavailable",
            },
            {
                "policy": "candidate",
                "map": "map_a_open",
                "win": 1,
                "collision_metric_source": "unavailable",
                "stuck_metric_source": "unavailable",
                "route_metric_source": "unavailable",
            },
        ]
        conditions = [{"policy": "candidate", "map": "map_a_open", "win": 1, "route_metric_source": "unavailable"}]
        summary = build_summary(self._args(), self._probes(), episodes, conditions)
        self.assertEqual(summary["verdict"], "READY FOR STAGE B")
        self.assertTrue(summary["stage2_eligible"])
        self.assertEqual(summary["diagnostics"]["map_route_signal"], "INCONCLUSIVE")

    def test_saturated_pool_does_not_block_stage2(self) -> None:
        episodes = [
            {"policy": "baseline", "map": "map_a_open", "win": 1},
            {"policy": "candidate", "map": "map_a_open", "win": 1},
        ]
        conditions = [{"policy": "candidate", "map": "map_a_open", "win": 1.0}]
        summary = build_summary(self._args(), self._probes(), episodes, conditions)
        self.assertEqual(summary["diagnostics"]["pool_saturation"], "SATURATED")
        self.assertEqual(summary["verdict"], "READY FOR STAGE B")
        self.assertTrue(summary["stage2_eligible"])

    def test_saturated_pool_label_when_override_enabled(self) -> None:
        episodes = [
            {"policy": "baseline", "map": "map_a_open", "win": 1},
            {"policy": "candidate", "map": "map_a_open", "win": 1},
        ]
        conditions = [{"policy": "candidate", "map": "map_a_open", "win": 1.0}]
        summary = build_summary(
            self._args(allow_saturated_pool=True),
            self._probes(),
            episodes,
            conditions,
        )
        self.assertEqual(summary["verdict"], "READY FOR STAGE B - SATURATED POOL")

    def test_probe_error_blocks_stage2(self) -> None:
        probes = self._probes()
        probes["candidate_gradient"] = GradientProbeResult(status=PROBE_ERROR, error="boom")
        episodes = [
            {"policy": "baseline", "map": "map_a_open", "win": 1},
            {"policy": "candidate", "map": "map_a_open", "win": 1},
        ]
        conditions = [{"policy": "candidate", "map": "map_a_open", "win": 1.0}]
        summary = build_summary(self._args(), probes, episodes, conditions)
        self.assertIn("PROBE ERROR", summary["verdict"])
        self.assertFalse(summary["stage2_eligible"])


if __name__ == "__main__":
    unittest.main()
