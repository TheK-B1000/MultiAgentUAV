"""Latent pair-count helpers (configuration-aware, not hardcoded six)."""

from __future__ import annotations

from rl.custom_ppo.gate_protocol import is_v6_protocol


def latent_pair_count(latent_k: int) -> int:
    k = int(latent_k)
    if k <= 1:
        return 0
    return k * (k - 1) // 2


def validate_v6_protocol_latent_k(cfg: object, latent_k: int) -> None:
    if not is_v6_protocol(cfg):
        return
    if int(latent_k) != 4:
        raise ValueError(
            f"V6 gate protocols require latent_k=4, got {int(latent_k)!r}."
        )
