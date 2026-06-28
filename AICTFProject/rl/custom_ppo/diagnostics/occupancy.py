"""Pure occupancy computations extracted from the latent rollout stats pipeline."""

from __future__ import annotations

import math

import torch


def compute_occupancy_stats(counts: torch.Tensor, K: int) -> dict[str, object]:
    """Return occupancy-derived metrics from a per-latent count tensor.

    ``counts`` must have shape ``(K,)`` and be non-negative integers (or floats).
    Returns a dict with occupancy fractions plus derived scalar metrics.
    """
    occupancy = counts / counts.sum().clamp_min(1.0)
    occ = occupancy.detach().cpu()
    occ_clamped = occ.clamp_min(1e-12)
    marginal_entropy = float((-(occ_clamped * occ_clamped.log()).sum()).item())
    occ_list = [float(v) for v in occ.tolist()]
    occ_min = float(min(occ_list)) if occ_list else 0.0
    occ_max = float(max(occ_list)) if occ_list else 0.0
    result: dict[str, object] = {
        "occupancy": occupancy,
        "latent_marginal_entropy_nats": marginal_entropy,
        "effective_num_latents": float(math.exp(marginal_entropy)),
        "latent_occupancy_min": occ_min,
        "latent_occupancy_max": occ_max,
        "latent_occupancy_ratio": float(occ_max / max(occ_min, 1e-8)),
    }
    for idx, value in enumerate(occ_list):
        result[f"strategy_occupancy_{idx}"] = float(value)
    return result


__all__ = ["compute_occupancy_stats"]
