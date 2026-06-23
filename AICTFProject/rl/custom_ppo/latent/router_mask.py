"""Allowed-latent masking for frozen-repertoire router experiments."""

from __future__ import annotations

from typing import Any

import torch


MASKED_LOGIT = -1.0e8


def allowed_latents_from_cfg(cfg: Any, latent_k: int) -> tuple[int, ...]:
    raw = tuple(getattr(cfg, "router_allowed_latents", ()) or ())
    if not raw:
        return tuple(range(int(latent_k)))
    allowed = tuple(int(z) for z in raw)
    if len(set(allowed)) != len(allowed):
        raise ValueError(f"router_allowed_latents contains duplicates: {allowed!r}")
    bad = [z for z in allowed if z < 0 or z >= int(latent_k)]
    if bad:
        raise ValueError(
            f"router_allowed_latents {allowed!r} outside latent range [0, {int(latent_k)})"
        )
    return allowed


def router_effective_latent_k(cfg: Any, latent_k: int) -> int:
    return len(allowed_latents_from_cfg(cfg, latent_k))


def apply_router_allowed_latent_mask(
    logits: torch.Tensor,
    *,
    cfg: Any,
    latent_k: int,
) -> torch.Tensor:
    allowed = allowed_latents_from_cfg(cfg, latent_k)
    if len(allowed) == int(latent_k):
        return logits
    if logits.dim() < 1 or int(logits.shape[-1]) != int(latent_k):
        raise ValueError(
            f"router logits last dimension must equal latent_k={int(latent_k)}, got {tuple(logits.shape)}"
        )
    masked = logits.clone()
    disallowed = [z for z in range(int(latent_k)) if z not in allowed]
    masked[..., disallowed] = MASKED_LOGIT
    return masked


def masked_uniform_logits(
    batch_size: int,
    *,
    cfg: Any,
    latent_k: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    logits = torch.full(
        (int(batch_size), int(latent_k)),
        MASKED_LOGIT,
        dtype=dtype,
        device=device,
    )
    logits[:, list(allowed_latents_from_cfg(cfg, latent_k))] = 0.0
    return logits


def assert_fixed_latent_allowed(cfg: Any, latent_k: int, fixed_z: int) -> None:
    allowed = allowed_latents_from_cfg(cfg, latent_k)
    if int(fixed_z) not in allowed:
        raise ValueError(
            f"fixed latent z{int(fixed_z)} is not in router_allowed_latents={allowed!r}"
        )
