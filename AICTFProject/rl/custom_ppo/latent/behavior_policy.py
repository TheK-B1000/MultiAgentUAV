"""Behavior-policy probability helpers for router sampling."""

from __future__ import annotations

import torch
from torch.distributions import Categorical

from rl.custom_ppo.latent.types import RouterActionSource


def epsilon_behavior_probs(
    router_probs: torch.Tensor,
    *,
    epsilon: float,
    latent_k: int,
) -> torch.Tensor:
    eps = float(epsilon)
    if eps <= 0.0:
        return router_probs
    uniform = 1.0 / float(max(1, int(latent_k)))
    return (1.0 - eps) * router_probs + eps * uniform


def uniform_behavior_probs(
    shape: tuple[int, ...],
    *,
    latent_k: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.full(
        (*shape, int(latent_k)),
        1.0 / float(max(1, int(latent_k))),
        dtype=dtype,
        device=device,
    )


def behavior_log_prob_from_probs(
    behavior_probs: torch.Tensor,
    executed_z: torch.Tensor,
) -> torch.Tensor:
    dist = Categorical(probs=behavior_probs.clamp_min(1e-8))
    return dist.log_prob(executed_z.long())


def resolve_action_sources(
    *,
    forced_mask: torch.Tensor,
    rehearsal_mask: torch.Tensor,
    epsilon_override_mask: torch.Tensor,
    event_refresh_mask: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> list[RouterActionSource]:
    sources: list[RouterActionSource] = []
    for i in range(int(batch_size)):
        if bool(forced_mask[i].item()):
            sources.append(
                RouterActionSource.FORCED_REHEARSAL
                if bool(rehearsal_mask[i].item())
                else RouterActionSource.FORCED_REHEARSAL
            )
        elif bool(epsilon_override_mask[i].item()):
            sources.append(RouterActionSource.EPSILON_MIXTURE)
        elif bool(event_refresh_mask[i].item()):
            sources.append(RouterActionSource.EVENT_REFRESH)
        else:
            sources.append(RouterActionSource.ROUTER)
    return sources
