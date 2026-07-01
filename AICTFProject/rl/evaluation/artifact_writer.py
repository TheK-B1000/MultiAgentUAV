"""Artifact writing helpers for V6I9 map-awareness evaluation."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl.evaluation.gates import DIAGNOSTIC_GATE_KEYS, REQUIRED_GATE_KEYS


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


def _gate_label(key: str) -> str:
    labels = {
        "obstacle_weights_moved": "Obstacle weights moved",
        "obstacle_gradient_connected": "Obstacle gradient connected",
        "obstacle_counterfactual_effect": "Obstacle counterfactual effect",
        "hard_pool_competence_retained": "Hard-pool competence retained",
        "wall_collisions_improved": "Wall collisions improved",
        "blocked_movement_improved": "Blocked movement improved",
        "stuck_behavior_improved": "Stuck behavior improved",
        "map_dependent_routes": "Map-dependent routes observed",
        "pool_saturation": "Pool saturation (diagnostic)",
        "universal_saturation_avoided": "Pool saturation (legacy alias)",
    }
    return labels.get(key, key.replace("_", " ").title())


def report_text(summary: Mapping[str, Any]) -> str:
    lines = [
        "V6I9 MAP-AWARENESS PROMOTION GATE",
        "",
        "Required gates:",
    ]

    required = summary.get("required_gates") or {
        key: summary["gates"][key] for key in REQUIRED_GATE_KEYS
    }
    for key in REQUIRED_GATE_KEYS:
        status = required[key]["status"]
        lines.append(f"  {_gate_label(key) + ':':34s} {status}")

    lines.extend(["", "Diagnostics:"])
    diagnostic = summary.get("diagnostic_gates") or {
        key: summary["gates"][key]
        for key in DIAGNOSTIC_GATE_KEYS
        if key in summary["gates"]
    }
    for key in DIAGNOSTIC_GATE_KEYS:
        if key not in diagnostic:
            continue
        status = diagnostic[key]["status"]
        lines.append(f"  {_gate_label(key) + ':':34s} {status}")

    diagnostics = summary.get("diagnostics")
    if diagnostics:
        lines.extend(["", "Diagnostic summary:"])
        for key, value in diagnostics.items():
            lines.append(f"  {key}: {value}")

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

    return "\n".join(lines) + "\n"


__all__ = ["report_text", "write_csv"]
