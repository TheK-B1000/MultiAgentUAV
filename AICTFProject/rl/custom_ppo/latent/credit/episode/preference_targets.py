"""Counterfactual preference target precomputation for episode PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class PreferenceTargets:
    coef: float
    target_probs: torch.Tensor
    mask: torch.Tensor
    active_buckets: int
    target_entropy_sum: float
    unique_keys: set[int]
    key_to_target_probs: dict[int, np.ndarray | None]


def build_preference_targets(
    *,
    trainer: Any,
    host: Any,
    batch_size: int,
    opponent_ids: torch.Tensor,
    bucket_ids: torch.Tensor,
    device: torch.device,
    latent_k: int,
) -> PreferenceTargets:
    coef = float(getattr(trainer, "latent_preference_coef", 0.0) or 0.0)
    target_probs = torch.zeros((batch_size, latent_k), dtype=torch.float32, device=device)
    mask = torch.zeros((batch_size,), dtype=torch.bool, device=device)
    active_buckets = 0
    target_entropy_sum = 0.0
    unique_keys: set[int] = set()
    key_to_target_probs: dict[int, np.ndarray | None] = {}

    if coef <= 0.0 or len(host.latent_preference_buffer) == 0:
        return PreferenceTargets(
            coef=coef,
            target_probs=target_probs,
            mask=mask,
            active_buckets=0,
            target_entropy_sum=0.0,
            unique_keys=unique_keys,
            key_to_target_probs=key_to_target_probs,
        )

    batch_keys = (opponent_ids * 256 + bucket_ids).detach().cpu().numpy().tolist()
    unique_keys = set(int(k) for k in batch_keys)
    buffer_by_key: dict[int, list[dict[str, Any]]] = {}
    for record in host.latent_preference_buffer:
        key = int(record["opponent"] * 256 + record["context_bucket"])
        buffer_by_key.setdefault(key, []).append(record)

    min_bucket_count = int(getattr(trainer, "latent_preference_min_bucket_count", 8) or 8)
    min_distinct_z = int(getattr(trainer, "latent_preference_min_distinct_z", 2) or 2)
    temperature = float(getattr(trainer, "latent_preference_temperature", 0.75) or 0.75)

    for key in unique_keys:
        matching = buffer_by_key.get(int(key), [])
        distinct_zs = {r["z"] for r in matching}
        if len(matching) < min_bucket_count or len(distinct_zs) < min_distinct_z:
            key_to_target_probs[key] = None
            continue
        active_buckets += 1
        returns_for_z = {z_idx: [] for z_idx in range(latent_k)}
        for record in matching:
            returns_for_z[record["z"]].append(record["return"])
        avg_return_by_z: dict[int, float] = {}
        for z_idx in range(latent_k):
            if returns_for_z[z_idx]:
                avg_return_by_z[z_idx] = sum(returns_for_z[z_idx]) / len(returns_for_z[z_idx])
        sampled_avgs = [avg_return_by_z[z_idx] for z_idx in range(latent_k) if z_idx in avg_return_by_z]
        fallback_val = min(sampled_avgs) if sampled_avgs else 0.0
        for z_idx in range(latent_k):
            if z_idx not in avg_return_by_z:
                avg_return_by_z[z_idx] = fallback_val
        avg_returns = np.array(
            [avg_return_by_z[z_idx] for z_idx in range(latent_k)], dtype=np.float32
        )
        exp_returns = np.exp((avg_returns - np.max(avg_returns)) / temperature)
        target_prob = exp_returns / np.sum(exp_returns)
        key_to_target_probs[key] = target_prob

    for i, key in enumerate(batch_keys):
        target = key_to_target_probs.get(int(key))
        if target is None:
            continue
        target_probs[i] = torch.as_tensor(target, dtype=torch.float32, device=device)
        mask[i] = True
        target_entropy_sum += float(-np.sum(target * np.log(target + 1e-12)))

    return PreferenceTargets(
        coef=coef,
        target_probs=target_probs,
        mask=mask,
        active_buckets=active_buckets,
        target_entropy_sum=target_entropy_sum,
        unique_keys=unique_keys,
        key_to_target_probs=key_to_target_probs,
    )
