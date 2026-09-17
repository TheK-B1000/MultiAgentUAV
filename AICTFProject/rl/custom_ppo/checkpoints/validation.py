from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .models import BehavioralEquivalenceReport

def _probe_obs_bank(observation_space: Any, *, batch_size: int, n_agents: int, device: torch.device) -> dict[str, torch.Tensor]:
    grid_shape = observation_space.spaces["grid"].shape
    vec_shape = observation_space.spaces["vec"].shape
    mask_shape = observation_space.spaces["mask"].shape
    grid = torch.linspace(
        0.0,
        1.0,
        steps=batch_size * n_agents * grid_shape[1] * grid_shape[2] * grid_shape[3],
        device=device,
    ).reshape(batch_size, n_agents, *grid_shape[1:])
    vec = torch.linspace(
        -0.5, 0.5, steps=batch_size * n_agents * vec_shape[1], device=device
    ).reshape(batch_size, n_agents, vec_shape[1])
    return {
        "grid": grid,
        "vec": vec,
        "agent_mask": torch.ones((batch_size, n_agents), device=device),
        "mask": torch.ones((batch_size, mask_shape[0]), device=device),
    }


def _entity_kwargs_for_probe(model: nn.Module, *, batch_size: int, device: torch.device) -> dict[str, Any]:
    """Zero entity tensors when the probe model requires them (g≡0 at zero-init proj)."""
    if getattr(model, "entity_encoder", None) is None:
        return {}
    from rl.custom_ppo.entity_residual import ENTITY_FEATURES

    n = int(getattr(model, "n_agents", 4))
    k_t = max(1, n - 1)
    k_e = n
    feat = int(ENTITY_FEATURES)
    return {
        "teammates": torch.zeros(batch_size, n, k_t, feat, device=device),
        "teammates_valid": torch.zeros(batch_size, n, k_t, dtype=torch.bool, device=device),
        "enemies": torch.zeros(batch_size, n, k_e, feat, device=device),
        "enemies_valid": torch.zeros(batch_size, n, k_e, dtype=torch.bool, device=device),
    }


def _logit_agreement(
    src_logits: torch.Tensor,
    tgt_logits: torch.Tensor,
    *,
    batch_size: int,
    n_agents: int,
    per_agent_action_dims: list[int],
) -> tuple[list[float], list[float], int]:
    src_flat = src_logits.reshape(batch_size * n_agents, -1)
    tgt_flat = tgt_logits.reshape(batch_size * n_agents, -1)
    all_kls: list[float] = []
    all_max_logit_diffs: list[float] = []
    total_argmax_disagreements = 0
    offset = 0
    for dim in per_agent_action_dims:
        src_chunk = src_flat[:, offset : offset + dim]
        tgt_chunk = tgt_flat[:, offset : offset + dim]
        src_dist = Categorical(logits=src_chunk)
        tgt_dist = Categorical(logits=tgt_chunk)
        kl = torch.distributions.kl.kl_divergence(src_dist, tgt_dist)
        all_kls.extend(kl.cpu().tolist())
        all_max_logit_diffs.append(float(torch.max(torch.abs(src_chunk - tgt_chunk)).item()))
        total_argmax_disagreements += int(torch.sum(torch.argmax(src_chunk, dim=-1) != torch.argmax(tgt_chunk, dim=-1)).item())
        offset += dim
    return all_kls, all_max_logit_diffs, total_argmax_disagreements


def run_behavioral_equivalence_probe(
    source_model: nn.Module,
    target_model: nn.Module,
    observation_space: Any,
    allowed_latents: list[int],
    device: torch.device
) -> tuple[float, float, float, int]:
    """Run a behavioral probe check on a fixed probe bank for the specified allowed latents.
    
    Returns: (mean_kl, max_kl, max_logit_diff, argmax_disagreement)

    Role-conditioning contracts:
    * Warm-start (source role OFF, target role ON): target logits with ``r=0``
      and ``r=1`` must both match the source (zero new columns ⇒ role inert
      at t=0).
    * Same-architecture resume/eval (both role ON): compare source vs target
      under matching role vectors (do not withhold roles from source).
    """
    source_model.eval()
    target_model.eval()
    
    batch_size = 5
    n_agents = getattr(source_model, "n_agents", 4)
    obs = _probe_obs_bank(observation_space, batch_size=batch_size, n_agents=n_agents, device=device)
    src_entity = _entity_kwargs_for_probe(source_model, batch_size=batch_size, device=device)
    tgt_entity = _entity_kwargs_for_probe(target_model, batch_size=batch_size, device=device)
    src_role_on = bool(getattr(source_model, "role_conditioning_enabled", False))
    tgt_role_on = bool(getattr(target_model, "role_conditioning_enabled", False))
    
    all_kls = []
    all_max_logit_diffs = []
    total_argmax_disagreements = 0
    
    with torch.no_grad():
        for z in allowed_latents:
            z_idx = torch.full((batch_size,), z, dtype=torch.long, device=device)
            # Warm-start: source has no role bit; target must match at r∈{0,1}.
            # Same-arch: both models need matching roles on every call.
            if tgt_role_on and not src_role_on:
                role_pairs = [
                    (None, torch.zeros(batch_size, n_agents, device=device)),
                    (None, torch.ones(batch_size, n_agents, device=device)),
                ]
            elif tgt_role_on and src_role_on:
                role_pairs = [
                    (torch.zeros(batch_size, n_agents, device=device),
                     torch.zeros(batch_size, n_agents, device=device)),
                    (torch.ones(batch_size, n_agents, device=device),
                     torch.ones(batch_size, n_agents, device=device)),
                ]
            else:
                role_pairs = [(None, None)]

            for src_roles, tgt_roles in role_pairs:
                sk = dict(src_entity)
                tk = dict(tgt_entity)
                if src_roles is not None:
                    sk["roles"] = src_roles
                if tgt_roles is not None:
                    tk["roles"] = tgt_roles
                try:
                    src_logits = source_model.policy_logits(obs, z_idx=z_idx, **sk)
                except TypeError:
                    src_logits = source_model.policy_logits(obs, **sk)
                try:
                    tgt_logits = target_model.policy_logits(obs, z_idx=z_idx, **tk)
                except TypeError:
                    tgt_logits = target_model.policy_logits(obs, **tk)
                kls, diffs, argmax_d = _logit_agreement(
                    src_logits,
                    tgt_logits,
                    batch_size=batch_size,
                    n_agents=n_agents,
                    per_agent_action_dims=list(source_model.per_agent_action_dims),
                )
                all_kls.extend(kls)
                all_max_logit_diffs.extend(diffs)
                total_argmax_disagreements += argmax_d
                
    mean_kl = float(np.mean(all_kls)) if all_kls else 0.0
    max_kl = float(np.max(all_kls)) if all_kls else 0.0
    max_logit_diff = float(np.max(all_max_logit_diffs)) if all_max_logit_diffs else 0.0
    
    return mean_kl, max_kl, max_logit_diff, total_argmax_disagreements


def behavioral_equivalence_report(source_model: nn.Module, target_model: nn.Module, observation_space: Any, allowed_latents: list[int], device: torch.device, tolerance: float = 1e-6) -> BehavioralEquivalenceReport:
    mean_kl, max_kl, max_logit_diff, argmax_diff = run_behavioral_equivalence_probe(source_model, target_model, observation_space, allowed_latents, device)
    sample_count = max(1, len(allowed_latents) * 5 * int(getattr(source_model, "n_agents", 4)))
    return BehavioralEquivalenceReport(
        passed=mean_kl <= tolerance and max_kl <= tolerance and max_logit_diff <= tolerance and argmax_diff == 0,
        mean_kl=mean_kl,
        max_kl=max_kl,
        max_logit_difference=max_logit_diff,
        argmax_difference_rate=float(argmax_diff) / float(sample_count),
        sample_count=sample_count,
        tolerance=tolerance,
    )
