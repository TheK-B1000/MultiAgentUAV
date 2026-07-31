from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import DEFAULT, patch

from rl.custom_ppo.probe_result import (
    PROBE_SUCCESS,
    CounterfactualProbeResult,
    GradientProbeResult,
    WeightProbeResult,
)
from rl.evaluation.config import MapAwarenessEvaluationConfig
from rl.evaluation.orchestrator import EvaluationRuntime, run_evaluation
from rl.evaluation.policy_loader import LoadedEvaluationPolicy


class EvaluationOrchestratorTests(unittest.TestCase):
    def _config(self, root: Path) -> MapAwarenessEvaluationConfig:
        baseline = root / "baseline.zip"
        candidate = root / "candidate.zip"
        baseline.write_bytes(b"baseline")
        candidate.write_bytes(b"candidate")
        return MapAwarenessEvaluationConfig(
            baseline_checkpoint=baseline,
            candidate_checkpoint=candidate,
            maps=("map_a_open",),
            opponents=("OP8",),
            episodes_per_cell=1,
            seed_start=7000,
            device="cpu",
            output_dir=root / "out",
            max_decision_steps=4,
            counterfactual_steps=1,
            obs_weight_threshold=1e-4,
            gradient_threshold=0.0,
            counterfactual_kl_threshold=1e-5,
            counterfactual_action_threshold=0.01,
            navigation_improvement_threshold=0.1,
            route_difference_threshold=0.1,
            minimum_win_rate=0.6,
            competence_retention_tolerance=0.05,
            saturation_win_rate=0.95,
            require_formal_identity=False,
        )

    def _runtime(self, calls: list[str], *, fail_distribution: bool = False, fail_write: bool = False) -> EvaluationRuntime:
        def write_json(path: Path, payload) -> None:
            calls.append(f"write:{path.name}")
            if fail_write and path.name == "final_report.json":
                raise RuntimeError("artifact write failed")
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        def preflight_distribution(policy, *, label: str) -> None:
            calls.append(f"preflight_distribution:{label}")
            if fail_distribution:
                raise RuntimeError("bad policy")

        return EvaluationRuntime(
            project_root=Path.cwd(),
            command=["eval.py"],
            validate_opponent_name=lambda opponent: opponent.upper(),
            preflight_opponents=lambda **kwargs: calls.append("preflight_opponents"),
            preflight_distribution_contract=preflight_distribution,
            inspect_obstacle_weights=lambda policy: calls.append("weights") or WeightProbeResult(status=PROBE_SUCCESS, has_obstacle_channel=True, cnn_channels=8, obstacle_weight_l2=1.0),
            gradient_probe=lambda *args, **kwargs: calls.append("gradient") or GradientProbeResult(status=PROBE_SUCCESS, obstacle_gradient_l2=1.0),
            obstacle_counterfactual=lambda *args, **kwargs: calls.append("counterfactual") or CounterfactualProbeResult(status=PROBE_SUCCESS, states_evaluated=1, mean_action_kl=1.0, mean_logit_l2=1.0, argmax_action_change_rate=1.0),
            run_episode=lambda **kwargs: calls.append(f"episode:{kwargs['policy_name']}") or {
                "policy": kwargs["policy_name"],
                "map": kwargs["map_name"],
                "requested_opponent": kwargs["opponent"],
                "resolved_opponent": kwargs["opponent"],
                "opponent": kwargs["opponent"],
                "seed": kwargs["seed"],
                "blue_score": 1.0,
                "red_score": 0.0,
                "win": 1,
                "loss": 0,
                "draw": 0,
                "score_margin": 1.0,
                "collision_metric_source": "environment_exact",
                "stuck_metric_source": "environment_exact",
                "route_metric_source": "environment_exact",
                "wall_collisions": 1.0,
                "blocked_movement_events": 1.0,
                "stuck_steps": 1.0,
                "upper_lane_use": 1.0,
                "lower_lane_use": 1.0,
                "episode_steps": 1,
            },
            write_json_text=write_json,
        )

    def _patch_loaders(self):
        return patch.multiple(
            "rl.evaluation.orchestrator",
            read_checkpoint_dimensions=DEFAULT,
            load_evaluation_policy=DEFAULT,
        )

    def test_orchestrator_call_order_and_manifest_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self._patch_loaders() as mocks:
            root = Path(tmp)
            calls: list[str] = []
            mocks["read_checkpoint_dimensions"].side_effect = [({}, 2, 5, 50), ({}, 2, 5, 50)]
            mocks["load_evaluation_policy"].side_effect = [
                LoadedEvaluationPolicy("baseline", "b", object(), {}, 2, 5, 50, 7),
                LoadedEvaluationPolicy("candidate", "c", object(), {}, 2, 5, 50, 8),
            ]
            result = run_evaluation(self._config(root), self._runtime(calls))

            self.assertEqual(result.exit_code, 0)
            self.assertLess(calls.index("preflight_opponents"), calls.index("preflight_distribution:baseline"))
            self.assertLess(calls.index("counterfactual"), calls.index("episode:baseline"))
            self.assertIn("write:final_report.json", calls)
            self.assertTrue((root / "out" / "final_report.json").is_file())
            manifest = json.loads((root / "out" / "evaluation_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(result.manifest.terminal_write_count, 1)

    def test_policy_preflight_failure_prevents_episode_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self._patch_loaders() as mocks:
            root = Path(tmp)
            calls: list[str] = []
            mocks["read_checkpoint_dimensions"].side_effect = [({}, 2, 5, 50), ({}, 2, 5, 50)]
            mocks["load_evaluation_policy"].side_effect = [
                LoadedEvaluationPolicy("baseline", "b", object(), {}, 2, 5, 50, 7),
                LoadedEvaluationPolicy("candidate", "c", object(), {}, 2, 5, 50, 8),
            ]
            with self.assertRaises(RuntimeError):
                run_evaluation(self._config(root), self._runtime(calls, fail_distribution=True))
            self.assertFalse(any(call.startswith("episode:") for call in calls))
            manifest = json.loads((root / "out" / "evaluation_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["error"], "RuntimeError: bad policy")

    def test_artifact_failure_finalizes_failure_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self._patch_loaders() as mocks:
            root = Path(tmp)
            calls: list[str] = []
            mocks["read_checkpoint_dimensions"].side_effect = [({}, 2, 5, 50), ({}, 2, 5, 50)]
            mocks["load_evaluation_policy"].side_effect = [
                LoadedEvaluationPolicy("baseline", "b", object(), {}, 2, 5, 50, 7),
                LoadedEvaluationPolicy("candidate", "c", object(), {}, 2, 5, 50, 8),
            ]
            with self.assertRaises(RuntimeError):
                run_evaluation(self._config(root), self._runtime(calls, fail_write=True))
            manifest = json.loads((root / "out" / "evaluation_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["error"], "RuntimeError: artifact write failed")


if __name__ == "__main__":
    unittest.main()
