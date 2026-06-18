"""Fixed episode-router advantage computation."""

from __future__ import annotations

from typing import Any, Optional

import torch

from rl.custom_ppo.latent.context_buckets import episode_bucket_baseline_keys
from rl.custom_ppo.latent_value_baselines import compute_z_marginal_strategy_value


def normalize_episode_advantages(
    advantages: torch.Tensor,
    *,
    bucket_keys: torch.Tensor | None,
    enabled: bool,
) -> torch.Tensor:
    if not enabled or advantages.numel() <= 1:
        return advantages
    if bucket_keys is not None:
        normalized = torch.zeros_like(advantages)
        unique_keys_tensor, counts_tensor = torch.unique(bucket_keys, return_counts=True)
        unique_keys = unique_keys_tensor.detach().cpu().tolist()
        counts = counts_tensor.detach().cpu().tolist()
        for key, count in zip(unique_keys, counts):
            mask = bucket_keys == key
            if count > 1:
                sub_adv = advantages[mask]
                normalized[mask] = (sub_adv - sub_adv.mean()) / (
                    sub_adv.std(unbiased=False) + 1e-8
                )
            else:
                normalized[mask] = advantages[mask]
        return normalized
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)


def compute_fixed_episode_advantages(
    *,
    trainer: Any,
    model: Any,
    states: torch.Tensor,
    executed_z: torch.Tensor,
    episode_returns: torch.Tensor,
    selector_hidden: torch.Tensor | None,
    bucket_baseline_vector: torch.Tensor | None,
    bucket_mode: str | None,
    opponent_ids: torch.Tensor | None,
    bucket_ids: torch.Tensor | None,
    return_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return (fixed_advantages, bucket_keys_used_for_norm)."""
    with torch.no_grad():
        initial_v_z = model.episode_strategy_value(
            states, executed_z, selector_hidden=selector_hidden
        )
        if bucket_baseline_vector is not None:
            baseline = bucket_baseline_vector
            bucket_keys = None
            if return_norm and bucket_mode is not None and opponent_ids is not None and bucket_ids is not None:
                bucket_keys = episode_bucket_baseline_keys(
                    mode=str(bucket_mode),
                    states=states,
                    opponent_ids=opponent_ids,
                    bucket_ids=bucket_ids,
                )
        elif getattr(trainer.cfg, "latent_q_phi_marginal_baseline", False):
            baseline = compute_z_marginal_strategy_value(
                model,
                states,
                trainer.latent_k,
                policy_weighted=False,
                selector_hidden=selector_hidden,
            )
            bucket_keys = None
        else:
            baseline = initial_v_z.detach()
            bucket_keys = None

        fixed = episode_returns - baseline
        fixed = normalize_episode_advantages(
            fixed,
            bucket_keys=bucket_keys,
            enabled=return_norm,
        ).detach()
    return fixed, bucket_keys
