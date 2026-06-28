"""Pure entropy and mutual-information helpers.

All functions here are pure: they operate on numpy arrays or primitive types
and have no side effects.  They are the innermost computational layer of the
latent-strategy diagnostics stack.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from rl.discrete_mi import discrete_mi_plugin


# ---------------------------------------------------------------------------
# Buffer-extraction utilities (depend on the rollout-buffer protocol only)
# ---------------------------------------------------------------------------


def _flat_long_np(buffer: Any, name: str, length: int) -> np.ndarray | None:
    """Return ``buffer.fields[name][:length]`` flattened to int64 numpy, or None."""
    if name not in buffer.fields:
        return None
    return buffer.fields[name][:length].reshape(-1).long().cpu().numpy()


def _flat_float_np(buffer: Any, name: str, length: int) -> np.ndarray | None:
    """Return ``buffer.fields[name][:length]`` flattened to float32 numpy, or None."""
    if name not in buffer.fields:
        return None
    return buffer.fields[name][:length].reshape(-1).float().cpu().numpy()


# ---------------------------------------------------------------------------
# Pure entropy / MI computations
# ---------------------------------------------------------------------------


def _shannon_entropy_nats(arr: np.ndarray | None, num_categories: int) -> float:
    """Plug-in Shannon entropy in nats. Returns 0.0 when ``arr`` is missing/empty."""
    if arr is None or arr.size == 0:
        return 0.0
    counts = np.bincount(arr, minlength=num_categories).astype(np.float64)
    total = counts.sum()
    if total <= 0.0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def _mi_z_vs(z: np.ndarray, K: int, x: np.ndarray | None, n_x: int) -> float:
    """Plug-in MI(z; x) in nats. Returns 0.0 when ``x`` is missing or empty."""
    if x is None:
        return 0.0
    valid = (z >= 0) & (z < K) & (x >= 0) & (x < n_x)
    if not bool(valid.any()):
        return 0.0
    idx = z[valid].astype(np.int64) * n_x + x[valid].astype(np.int64)
    joint = np.bincount(idx, minlength=K * n_x).reshape(K, n_x).astype(np.float64)
    return float(discrete_mi_plugin(joint))


def _bucket_z_fracs(
    out: dict[str, float],
    z: np.ndarray,
    K: int,
    bucket: np.ndarray,
    n_buckets: int,
    key: Callable[[int, int], str],
) -> None:
    """Write ``out[key(b, k)] = P(z=k | bucket=b)`` for every (b, k); zeros when empty."""
    for b in range(n_buckets):
        mask = bucket == b
        if bool(mask.any()):
            z_sub = np.clip(z[mask], 0, K - 1)
            for k in range(K):
                out[key(b, k)] = float((z_sub == k).mean())
        else:
            for k in range(K):
                out[key(b, k)] = 0.0


def _fill_zero_z_fracs(
    out: dict[str, float], K: int, n_buckets: int, key: Callable[[int, int], str]
) -> None:
    """Default branch for ``_bucket_z_fracs`` when the bucket field is absent."""
    for b in range(n_buckets):
        for k in range(K):
            out[key(b, k)] = 0.0


# ---------------------------------------------------------------------------
# Public names (no leading underscore) for use in the diagnostics package API
# ---------------------------------------------------------------------------
shannon_entropy_nats = _shannon_entropy_nats
mi_z_vs = _mi_z_vs
bucket_z_fracs = _bucket_z_fracs
fill_zero_z_fracs = _fill_zero_z_fracs
flat_long_np = _flat_long_np
flat_float_np = _flat_float_np


__all__ = [
    "_flat_long_np",
    "_flat_float_np",
    "_shannon_entropy_nats",
    "_mi_z_vs",
    "_bucket_z_fracs",
    "_fill_zero_z_fracs",
    "flat_long_np",
    "flat_float_np",
    "shannon_entropy_nats",
    "mi_z_vs",
    "bucket_z_fracs",
    "fill_zero_z_fracs",
]
