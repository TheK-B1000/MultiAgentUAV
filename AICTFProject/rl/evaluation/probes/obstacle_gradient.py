"""Obstacle-channel gradient probe."""
from __future__ import annotations

from typing import Any

import torch

from rl.custom_ppo.probe_result import PROBE_ERROR, PROBE_SUCCESS, GradientProbeResult
from rl.evaluation.probes.runtime import ObstacleProbeRuntime


def gradient_probe(
    policy: Any,
    *,
    runtime: ObstacleProbeRuntime,
    device: str,
    map_name: str,
    opponent: str,
    n_agents: int,
) -> GradientProbeResult:
    """Measure gradient flow through CNN channel 7 via the public contract."""
    env = runtime.make_env(
        n_agents=n_agents,
        map_name=map_name,
        device=device,
        seed=4242,
        max_steps=64,
        instrumented=False,
    )
    model = runtime.model(policy)
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)

    try:
        runtime.set_opponent(env, opponent)
        obs = runtime.reset_obs(env.reset())
        obs_t = runtime.to_torch(obs, runtime.policy_device(policy, device))

        batch = int(obs_t["grid"].shape[0])
        z_probe = torch.zeros(batch, dtype=torch.long, device=obs_t["grid"].device)
        dist = model.get_distribution(obs_t, z_idx=z_probe)

        diagnostic_loss = sum(
            head.logits.softmax(dim=-1).square().mean()
            for head in dist.heads
        )
        diagnostic_loss.backward()

        weight = model.get_observation_encoder_input_weights()
        if int(weight.shape[1]) < 8:
            return GradientProbeResult(
                status=PROBE_ERROR,
                error="Candidate policy has fewer than 8 CNN input channels.",
            )

        if weight.grad is None:
            return GradientProbeResult(
                status=PROBE_ERROR,
                error="First CNN convolution gradient is None after backward().",
            )

        obstacle_gradient = weight.grad[:, 7]
        return GradientProbeResult(
            status=PROBE_SUCCESS,
            obstacle_gradient_l2=float(torch.linalg.vector_norm(obstacle_gradient).item()),
            obstacle_gradient_abs_mean=float(obstacle_gradient.abs().mean().item()),
            diagnostic_loss=float(diagnostic_loss.detach().cpu().item()),
        )
    except Exception as exc:
        return GradientProbeResult(
            status=PROBE_ERROR,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        env.close()


__all__ = ["gradient_probe"]
