from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .models import BehavioralEquivalenceReport

def run_behavioral_equivalence_probe(
    source_model: nn.Module,
    target_model: nn.Module,
    observation_space: Any,
    allowed_latents: list[int],
    device: torch.device
) -> tuple[float, float, float, int]:
    """Run a behavioral probe check on a fixed probe bank for the specified allowed latents.
    
    Returns: (mean_kl, max_kl, max_logit_diff, argmax_disagreement)
    """
    source_model.eval()
    target_model.eval()
    
    batch_size = 5
    n_agents = getattr(source_model, "n_agents", 4)
    
    grid_shape = observation_space.spaces["grid"].shape
    vec_shape = observation_space.spaces["vec"].shape
    mask_shape = observation_space.spaces["mask"].shape
    
    grid = torch.linspace(0.0, 1.0, steps=batch_size * n_agents * grid_shape[1] * grid_shape[2] * grid_shape[3], device=device).reshape(batch_size, n_agents, *grid_shape[1:])
    vec = torch.linspace(-0.5, 0.5, steps=batch_size * n_agents * vec_shape[1], device=device).reshape(batch_size, n_agents, vec_shape[1])
    agent_mask = torch.ones((batch_size, n_agents), device=device)
    mask = torch.ones((batch_size, mask_shape[0]), device=device)
    
    obs = {
        "grid": grid,
        "vec": vec,
        "agent_mask": agent_mask,
        "mask": mask
    }
    
    all_kls = []
    all_max_logit_diffs = []
    total_argmax_disagreements = 0
    
    with torch.no_grad():
        for z in allowed_latents:
            z_idx = torch.full((batch_size,), z, dtype=torch.long, device=device)
            
            src_logits = source_model.policy_logits(obs, z_idx=z_idx)
            tgt_logits = target_model.policy_logits(obs, z_idx=z_idx)
            
            src_flat = src_logits.reshape(batch_size * n_agents, -1)
            tgt_flat = tgt_logits.reshape(batch_size * n_agents, -1)
            
            offset = 0
            for dim in source_model.per_agent_action_dims:
                src_chunk = src_flat[:, offset : offset + dim]
                tgt_chunk = tgt_flat[:, offset : offset + dim]
                
                src_dist = Categorical(logits=src_chunk)
                tgt_dist = Categorical(logits=tgt_chunk)
                
                kl = torch.distributions.kl.kl_divergence(src_dist, tgt_dist)
                all_kls.extend(kl.cpu().tolist())
                
                all_max_logit_diffs.append(torch.max(torch.abs(src_chunk - tgt_chunk)).item())
                
                src_argmax = torch.argmax(src_chunk, dim=-1)
                tgt_argmax = torch.argmax(tgt_chunk, dim=-1)
                total_argmax_disagreements += torch.sum(src_argmax != tgt_argmax).item()
                
                offset += dim
                
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
