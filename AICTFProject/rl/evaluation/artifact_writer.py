"""Artifact writing helpers for V6I9 map-awareness evaluation."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        return

    fieldnames: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def report_text(summary: Mapping[str, Any]) -> str:
    labels = (
        (
            "obstacle_weights_moved",
            "Obstacle weights moved",
        ),
        (
            "obstacle_gradient_connected",
            "Obstacle gradient connected",
        ),
        (
            "obstacle_counterfactual_effect",
            "Obstacle counterfactual effect",
        ),
        (
            "wall_collisions_improved",
            "Wall collisions improved",
        ),
        (
            "blocked_movement_improved",
            "Blocked movement improved",
        ),
        (
            "stuck_behavior_improved",
            "Stuck behavior improved",
        ),
        (
            "map_dependent_routes",
            "Map-dependent routes observed",
        ),
        (
            "hard_pool_competence_retained",
            "Hard-pool competence retained",
        ),
        (
            "universal_saturation_avoided",
            "Universal saturation avoided",
        ),
    )

    lines = [
        "V6I9 MAP-AWARENESS PROMOTION GATE",
        "",
    ]

    for key, label in labels:
        status = summary["gates"][key]["status"]
        lines.append(
            f"{label + ':':36s} {status}"
        )

    lines.extend(
        (
            "",
            f"VERDICT: {summary['verdict']}",
        )
    )

    if summary.get("warning"):
        lines.extend(
            (
                "",
                f"WARNING: {summary['warning']}",
            )
        )

    return "\n".join(lines)



__all__ = ["report_text", "write_csv"]
