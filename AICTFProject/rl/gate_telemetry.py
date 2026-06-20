"""Shared telemetry mappers for gate-owned diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def phase_a_actor_pair_telemetry_from_actor_gate_details(
    details: dict[str, Any] | None,
) -> dict[str, float]:
    """Phase A actor-pair CSV/report fields copied from actor gate details."""
    src = dict(details or {})
    batch_pairs = float(
        src.get("batch_pairs_above_margin", src.get("num_pairs_above_margin", 0.0)) or 0.0
    )
    pair_values = src.get("cf_pair_jsd_last_batch")
    if pair_values is None:
        pair_values = src.get("cf_pair_jsd_ema")
    finite_pairs: list[float] = []
    if pair_values is not None:
        try:
            finite_pairs = [float(v) for v in list(pair_values) if np.isfinite(float(v))]
        except (TypeError, ValueError):
            finite_pairs = []
    weakest = (
        float(min(finite_pairs))
        if finite_pairs
        else float(src.get("min_cf_pair_jsd_ema", src.get("min_pair_jsd_ema", 0.0)) or 0.0)
    )
    return {
        "phase_a_actor_pairs_above_margin": batch_pairs,
        "phase_a_actor_weakest_pair_jsd": weakest,
        "phase_a_actor_pair_gate_pass": 1.0 if bool(src.get("single_update_ok", False)) else 0.0,
    }


__all__ = ["phase_a_actor_pair_telemetry_from_actor_gate_details"]
