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


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def phase_a_matched_seed_behavioral_telemetry_from_gate_details(
    details: dict[str, Any] | None,
) -> dict[str, Any]:
    """CSV/report fields copied from the matched-seed behavioral gate result."""
    src = dict(details or {})
    semantics = src.get("matched_seed_semantics_details")
    if isinstance(semantics, dict):
        sem = semantics
    else:
        sem = src
    opponents = sem.get("opponents", {})
    if not isinstance(opponents, dict):
        opponents = {}

    route_values: list[float] = []
    behavior_values: list[float] = []
    performance_values: list[float] = []
    aggregate_values: list[float] = []
    floor_pass_count = 0
    for report in opponents.values():
        if not isinstance(report, dict):
            continue
        route_values.append(_finite_float(report.get("route_distance")))
        behavior_values.append(_finite_float(report.get("task_behavior_distance")))
        performance_values.append(_finite_float(report.get("performance_spread")))
        aggregate_values.append(_finite_float(report.get("aggregate_effect")))
        floor_pass_count += int(bool(report.get("component_floor_pass", False)))

    gate_status = str(
        src.get(
            "behavioral_realization_gate_status",
            src.get("aggregate_result", src.get("matched_seed_semantics", "")),
        )
        or ""
    )
    semantics_status = str(src.get("matched_seed_semantics", sem.get("matched_seed_semantics", "")) or "")
    strong = _finite_float(sem.get("strong_opponent_count"))
    required = _finite_float(sem.get("behavioral_realization_min_opponents_pass", 0.0))
    if required <= 0.0:
        required = _finite_float(src.get("behavioral_realization_min_opponents_pass", 0.0))
    aggregate = _finite_float(sem.get("aggregate_effect", sem.get("aggregate_semantic_effect")))
    if aggregate == 0.0 and aggregate_values:
        aggregate = float(np.mean(aggregate_values))

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def _min(values: list[float]) -> float:
        return float(min(values)) if values else 0.0

    return {
        "matched_seed_behavioral_gate_status": gate_status,
        "matched_seed_behavioral_semantics_status": semantics_status,
        "matched_seed_behavioral_gate_pass": 1.0 if gate_status == "PASS" else 0.0,
        "matched_seed_behavioral_strong_opponents": strong,
        "matched_seed_behavioral_required_opponents": required,
        "matched_seed_behavioral_opponent_count": float(len(opponents)),
        "matched_seed_behavioral_component_floor_pass_count": float(floor_pass_count),
        "matched_seed_behavioral_aggregate_effect": aggregate,
        "matched_seed_behavioral_mean_route_distance": _mean(route_values),
        "matched_seed_behavioral_mean_task_behavior_distance": _mean(behavior_values),
        "matched_seed_behavioral_min_task_behavior_distance": _min(behavior_values),
        "matched_seed_behavioral_mean_performance_spread": _mean(performance_values),
        "matched_seed_behavioral_min_performance_spread": _min(performance_values),
    }


__all__ = [
    "phase_a_actor_pair_telemetry_from_actor_gate_details",
    "phase_a_matched_seed_behavioral_telemetry_from_gate_details",
]
