from __future__ import annotations

import unittest

import torch

from rl.custom_ppo.distributions import ActionHead, MultiHeadActionDistribution
from rl.evaluation.probes.counterfactual import obstacle_counterfactual
from rl.evaluation.probes.obstacle_weights import inspect_obstacle_weights
from rl.evaluation.probes.runtime import ObstacleProbeRuntime


class _Model:
    def __init__(self, channels: int = 8) -> None:
        self.training = False
        self.weight = torch.ones((2, channels, 1, 1), requires_grad=True)
    def eval(self):
        self.training = False
    def train(self, value=True):
        self.training = value
    def get_observation_encoder_input_weights(self):
        return self.weight


class EvaluationProbeModuleTests(unittest.TestCase):
    def test_weight_probe_reports_obstacle_channel_without_zero_filling_errors(self) -> None:
        model = _Model()
        runtime = ObstacleProbeRuntime(
            make_env=lambda **kwargs: None,
            model=lambda policy: model,
            policy_device=lambda policy, fallback: torch.device("cpu"),
            reset_obs=lambda value: value,
            set_opponent=lambda env, opponent: opponent,
            to_torch=lambda obs, device: obs,
            zero_obstacle_channel=lambda obs: (obs, "grid"),
            head_argmax_change_rate=lambda real, zero: 0.0,
            predict=lambda policy, obs: None,
            unpack_step=lambda value: value,
            done=lambda value: True,
        )
        result = inspect_obstacle_weights(object(), runtime=runtime)
        self.assertTrue(result.is_success)
        self.assertTrue(result.has_obstacle_channel)
        self.assertEqual(result.cnn_channels, 8)
        self.assertIsNotNone(result.obstacle_weight_l2)

    def test_counterfactual_probe_error_preserves_none_metrics(self) -> None:
        class Env:
            def reset(self):
                raise RuntimeError("reset failed")
            def close(self):
                pass
        runtime = ObstacleProbeRuntime(
            make_env=lambda **kwargs: Env(),
            model=lambda policy: _Model(),
            policy_device=lambda policy, fallback: torch.device("cpu"),
            reset_obs=lambda value: value,
            set_opponent=lambda env, opponent: opponent,
            to_torch=lambda obs, device: obs,
            zero_obstacle_channel=lambda obs: (obs, "grid"),
            head_argmax_change_rate=lambda real, zero: 0.0,
            predict=lambda policy, obs: None,
            unpack_step=lambda value: value,
            done=lambda value: True,
        )
        result = obstacle_counterfactual(object(), runtime=runtime, device="cpu", map_name="map", opponent="OP8", n_agents=2, steps=1)
        self.assertFalse(result.is_success)
        self.assertIsNone(result.mean_action_kl)
        self.assertIn("reset failed", result.error or "")


if __name__ == "__main__":
    unittest.main()
