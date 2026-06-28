"""Obstacle-channel counterfactual probe."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rl.custom_ppo.probe_result import PROBE_ERROR, PROBE_SUCCESS, CounterfactualProbeResult
from rl.evaluation.probes.runtime import ObstacleProbeRuntime


def obstacle_counterfactual(
    policy: Any,
    *,
    runtime: ObstacleProbeRuntime,
    device: str,
    map_name: str,
    opponent: str,
    n_agents: int,
    steps: int,
) -> CounterfactualProbeResult:
    """Compare real vs. zeroed-obstacle-channel distributions."""
    env = runtime.make_env(
        n_agents=n_agents,
        map_name=map_name,
        device=device,
        seed=4343,
        max_steps=max(steps + 8, 64),
        instrumented=False,
    )
    model = runtime.model(policy)
    was_training = model.training
    model.eval()

    kls: list[float] = []
    l2_values: list[float] = []
    change_rates: list[float] = []
    tensor_key: str | None = None

    try:
        runtime.set_opponent(env, opponent)
        obs = runtime.reset_obs(env.reset())

        for _ in range(steps):
            obs_t = runtime.to_torch(obs, runtime.policy_device(policy, device))
            zero_t, tensor_key = runtime.zero_obstacle_channel(obs_t)

            batch = int(obs_t["grid"].shape[0])
            z_probe = torch.zeros(batch, dtype=torch.long, device=obs_t["grid"].device)

            with torch.no_grad():
                real_dist = model.get_distribution(obs_t, z_idx=z_probe)
                zero_dist = model.get_distribution(zero_t, z_idx=z_probe)

            if len(real_dist.heads) != len(zero_dist.heads):
                raise RuntimeError(
                    "Distribution head count changed during the obstacle counterfactual."
                )

            per_head_kl = []
            per_head_l2 = []

            for real_head, zero_head in zip(real_dist.heads, zero_dist.heads):
                real_lp = real_head.logits.log_softmax(dim=-1)
                zero_lp = zero_head.logits.log_softmax(dim=-1)
                per_head_kl.append(
                    (real_lp.exp() * (real_lp - zero_lp)).sum(dim=-1).mean()
                )
                per_head_l2.append(
                    torch.linalg.vector_norm(
                        real_head.logits - zero_head.logits, dim=-1
                    ).mean()
                )

            kls.append(float(torch.stack(per_head_kl).mean().detach().cpu().item()))
            l2_values.append(float(torch.stack(per_head_l2).mean().detach().cpu().item()))
            change_rates.append(
                runtime.head_argmax_change_rate(
                    [h.logits for h in real_dist.heads],
                    [h.logits for h in zero_dist.heads],
                )
            )

            action = runtime.predict(policy, obs)
            obs, _, done, _ = runtime.unpack_step(env.step(action))
            if runtime.done(done):
                break

        if not kls:
            raise RuntimeError("No counterfactual states were evaluated.")

        return CounterfactualProbeResult(
            status=PROBE_SUCCESS,
            states_evaluated=len(kls),
            observation_tensor=tensor_key,
            mean_action_kl=float(np.mean(kls)),
            max_action_kl=float(np.max(kls)),
            mean_logit_l2=float(np.mean(l2_values)),
            max_logit_l2=float(np.max(l2_values)),
            argmax_action_change_rate=float(np.mean(change_rates)),
        )
    except Exception as exc:
        return CounterfactualProbeResult(
            status=PROBE_ERROR,
            states_evaluated=len(kls),
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        model.train(was_training)
        env.close()


__all__ = ["obstacle_counterfactual"]
