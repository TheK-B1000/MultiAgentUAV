"""Small scalar schedules used by custom PPO training."""

from __future__ import annotations

from typing import Any


def linear_anneal(
    step: int | float,
    start_value: float,
    end_value: float,
    start_step: int,
    end_step: int,
) -> float:
    """Linearly interpolate between two scalar values inside a step window."""
    step_f = float(step)
    start_step_i = int(start_step)
    end_step_i = int(end_step)
    start_value_f = float(start_value)
    end_value_f = float(end_value)

    if end_step_i <= start_step_i:
        return end_value_f if step_f >= end_step_i else start_value_f
    if step_f <= start_step_i:
        return start_value_f
    if step_f >= end_step_i:
        return end_value_f

    progress = (step_f - start_step_i) / float(end_step_i - start_step_i)
    return start_value_f + progress * (end_value_f - start_value_f)


def resolve_latent_lam_h(cfg: Any, *, global_step: int | float, total_timesteps: int) -> float:
    """Resolve the current latent entropy coefficient without changing old configs."""
    lam_h_start = getattr(cfg, "latent_lam_h_start", None)
    if lam_h_start is None:
        lam_h_start = getattr(cfg, "latent_lam_h", 0.0) or 0.0
    lam_h_start = max(0.0, float(lam_h_start))

    lam_h_end = getattr(cfg, "latent_lam_h_end", None)
    if lam_h_end is None:
        lam_h_end = lam_h_start
    lam_h_end = max(0.0, float(lam_h_end))

    anneal_start = getattr(cfg, "latent_entropy_anneal_start", None)
    if anneal_start is None:
        anneal_start = 0

    anneal_end = getattr(cfg, "latent_entropy_anneal_end", None)
    if anneal_end is None:
        anneal_end = int(total_timesteps)

    return linear_anneal(
        global_step,
        lam_h_start,
        lam_h_end,
        int(anneal_start),
        int(anneal_end),
    )
