"""Aggregation helpers for V6I9 map-awareness evaluation."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

NUMERIC_FIELDS = (
    "blue_score",
    "red_score",
    "win",
    "loss",
    "draw",
    "score_margin",
    "wall_collisions",
    "blocked_movement_events",
    "stuck_steps",
    "repeated_blocked_movement",
    "upper_lane_use",
    "lower_lane_use",
    "neutral_lane_use",
    "route_switches",
    "movement_attempts",
    "successful_movement_steps",
    "obstacle_collisions_per_1000_steps",
    "blocked_movements_per_1000_movement_attempts",
    "stuck_steps_per_1000_steps",
    "successful_movement_rate",
    "upper_lane_fraction",
    "lower_lane_fraction",
    "route_switches_per_episode",
    "episode_steps",
)


def _mean(values: Iterable[Any]) -> float | None:
    numbers: list[float] = []

    for value in values:
        if value is None or value == "":
            continue

        try:
            number = float(value)
        except (TypeError, ValueError):
            continue

        if math.isfinite(number):
            numbers.append(number)

    if not numbers:
        return None

    return float(np.mean(numbers))


def aggregate_conditions(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str],
        list[Mapping[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        grouped[
            (
                str(row["policy"]),
                str(row["map"]),
                str(row["resolved_opponent"]),
            )
        ].append(row)

    output: list[dict[str, Any]] = []

    for (
        policy,
        map_name,
        opponent,
    ), group in sorted(grouped.items()):
        aggregate: dict[str, Any] = {
            "policy": policy,
            "map": map_name,
            "requested_opponent": opponent,
            "resolved_opponent": opponent,
            "opponent": opponent,
            "episodes": len(group),
        }

        for field in NUMERIC_FIELDS:
            aggregate[field] = _mean(
                item.get(field) for item in group
            )

        for source_field in (
            "collision_metric_source",
            "stuck_metric_source",
            "route_metric_source",
        ):
            values = sorted({str(item.get(source_field, "unavailable")) for item in group})
            aggregate[source_field] = values[0] if len(values) == 1 else "mixed"

        upper = aggregate.get("upper_lane_use") or 0.0
        lower = aggregate.get("lower_lane_use") or 0.0
        neutral = aggregate.get("neutral_lane_use") or 0.0
        crossings = upper + lower

        aggregate["route_crossings"] = crossings
        lane_total = upper + lower + neutral
        aggregate["upper_lane_fraction"] = (
            upper / lane_total
            if lane_total > 0
            else None
        )
        aggregate["lower_lane_fraction"] = lower / lane_total if lane_total > 0 else None

        output.append(aggregate)

    return output


def _policy_rows(
    rows: Sequence[Mapping[str, Any]],
    policy_name: str,
    obstacle_maps_only: bool = False,
) -> list[Mapping[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("policy") == policy_name
    ]

    if obstacle_maps_only:
        selected = [
            row
            for row in selected
            if "open" not in str(
                row.get("map", "")
            ).lower()
        ]

    return selected


def _field_mean(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> float | None:
    return _mean(row.get(field) for row in rows)



__all__ = [
    "NUMERIC_FIELDS",
    "aggregate_conditions",
    "field_mean",
    "mean",
    "policy_rows",
]

mean = _mean
policy_rows = _policy_rows
field_mean = _field_mean
