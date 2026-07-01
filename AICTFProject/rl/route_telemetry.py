"""Per-step route crossing telemetry for split-lane CTF layouts.

The GPU env accumulates lane crossings in episode ``info`` dicts (cumulative
counts). Qualitative / forced-z eval tools convert those to per-step deltas
and derive normalized lane-usage fractions for repertoire diagnostics.

Only meaningful on split-lane maps (``map_b_split_lane*``); on open maps all
signals are zero.
"""

from __future__ import annotations

from typing import Any

import numpy as np

ROUTE_CROSSING_KEYS: tuple[str, ...] = (
    "blue_attack_upper_crossings",
    "blue_attack_lower_crossings",
    "blue_return_upper_crossings",
    "blue_return_lower_crossings",
    "blue_intercept_upper_crossings",
    "blue_intercept_lower_crossings",
)

ROUTE_DERIVED_NAMES: tuple[str, ...] = (
    "upper_attack_fraction",
    "lower_attack_fraction",
    "upper_return_fraction",
    "lower_return_fraction",
    "intercept_lane_fraction",
    "route_switch_rate",
)

ROUTE_TELEMETRY_NAMES: tuple[str, ...] = ROUTE_CROSSING_KEYS + ROUTE_DERIVED_NAMES


def zero_route_cumulative() -> dict[str, int]:
    return {k: 0 for k in ROUTE_CROSSING_KEYS}


def cumulative_route_crossings_from_info(info: dict[str, Any]) -> dict[str, int]:
    """Read cumulative episode crossing counts from a post-step env info dict."""
    return {k: int(info.get(k, 0) or 0) for k in ROUTE_CROSSING_KEYS}


def route_step_crossings(
    prev_cumulative: dict[str, int],
    cumulative: dict[str, int],
) -> dict[str, float]:
    """Per-step crossing increments from cumulative episode counters."""
    return {
        k: float(max(0, int(cumulative.get(k, 0)) - int(prev_cumulative.get(k, 0))))
        for k in ROUTE_CROSSING_KEYS
    }


def route_derived_features(step_crossings: dict[str, float]) -> dict[str, float]:
    """Normalized lane-usage fractions for one decision step (excluding switch rate)."""
    au = float(step_crossings.get("blue_attack_upper_crossings", 0.0))
    al = float(step_crossings.get("blue_attack_lower_crossings", 0.0))
    ru = float(step_crossings.get("blue_return_upper_crossings", 0.0))
    rl = float(step_crossings.get("blue_return_lower_crossings", 0.0))
    iu = float(step_crossings.get("blue_intercept_upper_crossings", 0.0))
    il = float(step_crossings.get("blue_intercept_lower_crossings", 0.0))

    attack_total = au + al
    return_total = ru + rl
    intercept_total = iu + il
    all_total = attack_total + return_total + intercept_total

    def _upper_frac(upper: float, lower: float) -> float:
        total = upper + lower
        return float(upper / total) if total > 0.0 else 0.0

    return {
        "upper_attack_fraction": _upper_frac(au, al),
        "lower_attack_fraction": _upper_frac(al, au),
        "upper_return_fraction": _upper_frac(ru, rl),
        "lower_return_fraction": _upper_frac(rl, ru),
        "intercept_lane_fraction": (
            float(intercept_total / all_total) if all_total > 0.0 else 0.0
        ),
    }


def dominant_route_lane(step_crossings: dict[str, float]) -> int | None:
    """Index into ``ROUTE_CROSSING_KEYS`` for the lane with most crossings this step."""
    vals = [float(step_crossings.get(k, 0.0)) for k in ROUTE_CROSSING_KEYS]
    total = float(sum(vals))
    if total <= 0.0:
        return None
    return int(np.argmax(np.asarray(vals, dtype=np.float64)))


def route_switch_indicator(
    step_crossings: dict[str, float],
    prev_lane: int | None,
) -> float:
    """1.0 when the dominant crossing lane changes vs the previous active step."""
    lane = dominant_route_lane(step_crossings)
    if lane is None or prev_lane is None:
        return 0.0
    return 1.0 if lane != prev_lane else 0.0


def attach_route_telemetry_to_row(
    row: dict[str, Any],
    *,
    step_crossings: dict[str, float],
    prev_dominant_lane: int | None,
) -> int | None:
    """Write per-step route columns into ``row``; return updated dominant lane."""
    for k, v in step_crossings.items():
        row[k] = float(v)
    derived = route_derived_features(step_crossings)
    for k, v in derived.items():
        row[k] = float(v)
    row["route_switch_rate"] = route_switch_indicator(step_crossings, prev_dominant_lane)
    lane = dominant_route_lane(step_crossings)
    return lane if lane is not None else prev_dominant_lane
