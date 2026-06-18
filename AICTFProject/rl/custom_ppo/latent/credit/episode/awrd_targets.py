"""Advantage-weighted router distillation target precomputation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from rl.custom_ppo.latent.preferences import advantage_weighted_target_from_records
from rl.custom_ppo.latent.preferences import warmup_ramp_coef_scale as _warmup_ramp_coef_scale
from rl.custom_ppo.latent.schedule_steps import resolve_schedule_step


@dataclass
class AwrdTargets:
    enabled: bool
    coef: float
    coef_scale: float
    soft_margin: bool
    target_probs: torch.Tensor
    mask: torch.Tensor
    per_sample_coefs: torch.Tensor
    active_buckets: int
    target_entropy_sum: float
    margin_sum: float
    wr_spread_sum: float
    best_z_sum: float
    best_z_matches: float
    effective_coef_sum: float
    key_stats: dict[int, dict[str, float]]


def resolve_awrd_coef(
    *,
    trainer: Any,
    base_coef: float,
    soft_margin: bool,
    global_step: int,
) -> float:
    if not soft_margin or base_coef <= 0.0:
        return base_coef
    abs_step = int(getattr(trainer.cfg, "latent_awrd_boost_after_steps", 0) or 0)
    frac = float(getattr(trainer.cfg, "latent_awrd_boost_after_fraction", 0.0) or 0.0)
    boost_start = resolve_schedule_step(
        absolute_step=abs_step if abs_step > 0 else None,
        fraction=frac if frac > 0.0 else None,
        nominal_steps=int(getattr(trainer.cfg, "curriculum_nominal_timesteps", 1_000_000)),
    )
    multiplier = float(getattr(trainer.cfg, "latent_awrd_boost_multiplier", 1.0) or 1.0)
    if boost_start is not None and global_step >= boost_start and multiplier > 1.0:
        return base_coef * multiplier
    return base_coef


def build_awrd_targets(
    *,
    trainer: Any,
    host: Any,
    batch_size: int,
    executed_z: torch.Tensor,
    opponent_ids: torch.Tensor,
    bucket_ids: torch.Tensor,
    device: torch.device,
    latent_k: int,
) -> AwrdTargets:
    enabled = bool(getattr(trainer, "latent_awrd_enabled", False))
    coef_scale = _warmup_ramp_coef_scale(
        global_step=int(getattr(trainer, "global_step", 0) or 0),
        warmup_steps=int(getattr(trainer, "latent_awrd_warmup_steps", 0) or 0),
        ramp_steps=int(getattr(trainer, "latent_awrd_ramp_steps", 0) or 0),
    )
    base_coef = float(getattr(trainer, "latent_awrd_coef", 0.0) or 0.0) * coef_scale
    soft_margin = bool(getattr(trainer, "latent_awrd_soft_margin_gating", False))
    coef = resolve_awrd_coef(
        trainer=trainer,
        base_coef=base_coef,
        soft_margin=soft_margin,
        global_step=int(getattr(trainer, "global_step", 0) or 0),
    )
    target_probs = torch.zeros((batch_size, latent_k), dtype=torch.float32, device=device)
    mask = torch.zeros((batch_size,), dtype=torch.bool, device=device)
    per_sample_coefs = torch.zeros((batch_size,), dtype=torch.float32, device=device)
    key_stats: dict[int, dict[str, float]] = {}
    active_buckets = 0
    target_entropy_sum = 0.0
    margin_sum = 0.0
    wr_spread_sum = 0.0
    best_z_sum = 0.0
    best_z_matches = 0.0
    effective_coef_sum = 0.0

    if (
        not enabled
        or coef <= 0.0
        or len(host.latent_preference_buffer) == 0
    ):
        return AwrdTargets(
            enabled=enabled,
            coef=coef,
            coef_scale=coef_scale,
            soft_margin=soft_margin,
            target_probs=target_probs,
            mask=mask,
            per_sample_coefs=per_sample_coefs,
            active_buckets=0,
            target_entropy_sum=0.0,
            margin_sum=0.0,
            wr_spread_sum=0.0,
            best_z_sum=0.0,
            best_z_matches=0.0,
            effective_coef_sum=0.0,
            key_stats=key_stats,
        )

    batch_keys = (opponent_ids * 256 + bucket_ids).detach().cpu().numpy().tolist()
    buffer_by_key: dict[int, list[dict[str, Any]]] = {}
    for record in host.latent_preference_buffer:
        key = int(record["opponent"] * 256 + record["context_bucket"])
        buffer_by_key.setdefault(key, []).append(record)

    min_count = int(getattr(trainer, "latent_awrd_min_bucket_count", 8) or 8)
    min_distinct = int(getattr(trainer, "latent_awrd_min_distinct_z", 2) or 2)
    temperature = float(getattr(trainer, "latent_awrd_temperature", 0.35) or 0.35)
    threshold = float(getattr(trainer, "latent_awrd_margin_threshold", 0.15) or 0.15)
    use_return = soft_margin
    key_to_target: dict[int, Optional[np.ndarray]] = {}

    for key in set(int(k) for k in batch_keys):
        target, stats = advantage_weighted_target_from_records(
            buffer_by_key.get(int(key), []),
            latent_k=latent_k,
            min_count=min_count,
            min_distinct_z=min_distinct,
            temperature=temperature,
            margin_threshold=threshold,
            soft_margin_gating=soft_margin,
            use_return=use_return,
        )
        key_to_target[int(key)] = target
        key_stats[int(key)] = stats
        if target is not None:
            active_buckets += 1

    for i, key in enumerate(batch_keys):
        target = key_to_target.get(int(key))
        if target is None:
            continue
        target_probs[i] = torch.as_tensor(target, dtype=torch.float32, device=device)
        mask[i] = True
        target_entropy_sum += float(-np.sum(target * np.log(target + 1e-12)))
        stats = key_stats.get(int(key), {})
        margin_sum += float(stats.get("margin", 0.0))
        wr_spread_sum += float(stats.get("wr_spread", 0.0))
        best_z_sum += float(stats.get("best_z", -1.0))
        z_picked = int(executed_z[i].item())
        best_z = int(stats.get("best_z", -1))
        if z_picked == best_z:
            best_z_matches += 1.0
        if soft_margin:
            margin = float(stats.get("margin", 0.0))
            scale = float(getattr(trainer, "latent_awrd_margin_scale", 3.0) or 3.0)
            min_margin = float(getattr(trainer, "latent_awrd_min_margin", 0.08) or 0.08)
            eff_coef = coef * (1.0 + scale * margin)
            if margin < min_margin:
                eff_coef = coef * 0.25
            per_sample_coefs[i] = eff_coef
            effective_coef_sum += eff_coef

    return AwrdTargets(
        enabled=enabled,
        coef=coef,
        coef_scale=coef_scale,
        soft_margin=soft_margin,
        target_probs=target_probs,
        mask=mask,
        per_sample_coefs=per_sample_coefs,
        active_buckets=active_buckets,
        target_entropy_sum=target_entropy_sum,
        margin_sum=margin_sum,
        wr_spread_sum=wr_spread_sum,
        best_z_sum=best_z_sum,
        best_z_matches=best_z_matches,
        effective_coef_sum=effective_coef_sum,
        key_stats=key_stats,
    )
