from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch

from rl.custom_ppo.distributions import ActionHead, MultiHeadActionDistribution
from rl.evaluation.config import MapAwarenessEvaluationConfig, config_from_namespace
from rl.evaluation.errors import (
    EvaluationCheckpointError,
    EvaluationConfigError,
    EvaluationPreflightError,
)
from rl.evaluation.policy_loader import load_evaluation_policy
from rl.evaluation.preflight import (
    validate_distribution_contract,
    validate_distribution_result,
)


class Phase10EvaluationConfigTests(unittest.TestCase):
    def _namespace(self, root: Path) -> Namespace:
        baseline = root / "baseline.zip"
        candidate = root / "candidate.zip"
        baseline.write_bytes(b"baseline")
        candidate.write_bytes(b"candidate")
        return Namespace(
            baseline=str(baseline),
            candidate=str(candidate),
            maps=["map_a_open", "map_b_split_lane"],
            opponents=["OP8", "OP9"],
            episodes=2,
            seed_start=7000,
            device="cpu",
            output_dir=str(root / "out"),
            max_decision_steps=400,
            counterfactual_steps=64,
            obs_weight_threshold=1e-4,
            gradient_threshold=0.0,
            counterfactual_kl_threshold=1e-5,
            counterfactual_action_threshold=0.01,
            navigation_improvement_threshold=0.1,
            route_difference_threshold=0.1,
            minimum_win_rate=0.6,
            competence_retention_tolerance=0.05,
            saturation_win_rate=0.95,
        )

    def test_config_from_namespace_preserves_cli_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = config_from_namespace(self._namespace(Path(tmp)))

        self.assertIsInstance(config, MapAwarenessEvaluationConfig)
        self.assertEqual(config.maps, ("map_a_open", "map_b_split_lane"))
        self.assertEqual(config.opponents, ("OP8", "OP9"))
        self.assertEqual(config.reference_map, "map_b_split_lane")
        self.assertEqual(config.reference_opponent, "OP8")
        self.assertEqual(config.baseline_cnn_channels, 7)
        self.assertEqual(config.candidate_cnn_channels, 8)

    def test_config_validation_uses_typed_error_for_invalid_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._namespace(Path(tmp))
            args.episodes = 0
            with self.assertRaises(EvaluationConfigError):
                config_from_namespace(args)


class Phase10PolicyLoaderTests(unittest.TestCase):
    def test_load_evaluation_policy_wraps_checkpoint_facts(self) -> None:
        policy = object()
        metadata = {"format": "test"}
        with patch(
            "rl.evaluation.policy_loader.read_checkpoint_dimensions",
            return_value=(metadata, 2, 5, 50),
        ), patch(
            "rl.evaluation.policy_loader.load_policy",
            return_value=policy,
        ):
            loaded = load_evaluation_policy(
                "candidate", "candidate.zip", device="cpu", cnn_channels=8
            )

        self.assertEqual(loaded.label, "candidate")
        self.assertIs(loaded.policy, policy)
        self.assertEqual(loaded.metadata, metadata)
        self.assertEqual(loaded.n_agents, 2)
        self.assertEqual(loaded.n_macros, 5)
        self.assertEqual(loaded.n_targets, 50)
        self.assertEqual(loaded.cnn_channels, 8)

    def test_load_evaluation_policy_raises_typed_checkpoint_error(self) -> None:
        with patch(
            "rl.evaluation.policy_loader.read_checkpoint_dimensions",
            side_effect=ValueError("bad metadata"),
        ):
            with self.assertRaises(EvaluationCheckpointError):
                load_evaluation_policy(
                    "baseline", "missing.zip", device="cpu", cnn_channels=7
                )


class _ContractModel:
    def get_distribution(self, obs, *, z_idx=None):
        return MultiHeadActionDistribution(
            [ActionHead(torch.zeros((1, 2), requires_grad=True))]
        )


class _ContractPolicy:
    def __init__(self) -> None:
        self.model = _ContractModel()

    def get_distribution(self, obs, *, z_idx=None):
        return self.model.get_distribution(obs, z_idx=z_idx)


class Phase10PreflightTests(unittest.TestCase):
    def test_distribution_contract_accepts_wrapper_and_model(self) -> None:
        validate_distribution_contract(_ContractPolicy(), label="candidate")

    def test_distribution_contract_rejects_missing_contract(self) -> None:
        class BrokenPolicy:
            model = object()

        with self.assertRaises(EvaluationPreflightError):
            validate_distribution_contract(BrokenPolicy(), label="broken")

    def test_distribution_result_preserves_typed_distribution_requirement(self) -> None:
        dist = MultiHeadActionDistribution(
            [ActionHead(torch.zeros((1, 2), requires_grad=True))]
        )
        validate_distribution_result(dist, label="candidate")
        with self.assertRaises(EvaluationPreflightError):
            validate_distribution_result(object(), label="broken")


if __name__ == "__main__":
    unittest.main()
