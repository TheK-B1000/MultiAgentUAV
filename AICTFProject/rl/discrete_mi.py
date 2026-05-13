"""Discrete mutual information helpers for telemetry (e.g. MI(z; opponent_id))."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def discrete_mi_plugin(counts: Any, *, eps: float = 1e-12) -> float:
    """Plug-in mutual information from joint counts matrix ``counts[z, o]``.

    Uses natural log (nats). Empty or degenerate tables return 0.0.
    """
    p = np.asarray(counts, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("counts must be 2-D (n_z, n_o)")
    s = float(p.sum())
    if s <= 0.0:
        return 0.0
    p = p / s
    p_z = p.sum(axis=1, keepdims=True)
    p_o = p.sum(axis=0, keepdims=True)
    denom = (p_z @ p_o).clip(min=eps)
    ratio = np.clip(p, eps, None) / denom
    # Sum only where joint mass exists (avoid NaNs from 0*log)
    mask = p > eps
    mi = float(np.sum(np.where(mask, p * np.log(ratio), 0.0)))
    if not math.isfinite(mi):
        return 0.0
    return mi
