"""Obstacle-channel weight probe."""
from __future__ import annotations

from typing import Any

import torch

from rl.custom_ppo.probe_result import PROBE_SUCCESS, WeightProbeResult
from rl.evaluation.probes.runtime import ObstacleProbeRuntime


def inspect_obstacle_weights(
    policy: Any,
    *,
    runtime: ObstacleProbeRuntime,
) -> WeightProbeResult:
    """Return typed weight inspection result via the diagnostics contract."""
    weight = runtime.model(policy).get_observation_encoder_input_weights()
    channels = int(weight.shape[1])

    if channels < 8:
        return WeightProbeResult(
            status=PROBE_SUCCESS,
            has_obstacle_channel=False,
            cnn_channels=channels,
        )

    obstacle_weights = weight[:, 7].detach()
    return WeightProbeResult(
        status=PROBE_SUCCESS,
        has_obstacle_channel=True,
        cnn_channels=channels,
        obstacle_weight_l2=float(torch.linalg.vector_norm(obstacle_weights).item()),
        obstacle_weight_abs_mean=float(obstacle_weights.abs().mean().item()),
        obstacle_weight_abs_max=float(obstacle_weights.abs().max().item()),
        obstacle_weight_nonzero_fraction=float(
            (obstacle_weights.abs() > 0).float().mean().item()
        ),
    )


__all__ = ["inspect_obstacle_weights"]
