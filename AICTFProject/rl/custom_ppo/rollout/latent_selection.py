"""Latent-strategy selection helpers used by rollout collection."""

from __future__ import annotations

from typing import Any, Optional

import torch


def fixed_latent_bootstrap_z(
    hparams: Any,
    z_t: torch.Tensor,
) -> Optional[torch.Tensor]:
    if not bool(getattr(hparams, "fixed_latent_strategy", False)):
        return None
    return torch.full_like(
        z_t,
        int(getattr(hparams, "fixed_latent_strategy_id", 0)),
        dtype=torch.long,
    )


__all__ = ["fixed_latent_bootstrap_z"]
