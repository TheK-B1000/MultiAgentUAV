"""Behavior distance analysis from forced-z episode telemetry."""
from __future__ import annotations

from typing import Any

import numpy as np

from experiments.forced_z_eval.io import CellEpisodes
from rl.behavior_telemetry import BEHAVIOR_TELEMETRY_NAMES
from rl.forced_z_behavior_vectors import (
    FORCED_Z_BEHAVIOR_VECTOR_NAMES,
    behavior_vector_from_telemetry_row,
    build_behavior_distance_profile,
)


def _mean_metric(eps: list[dict[str, Any]], key: str) -> float:
    vals = [float(e.get(key, np.nan)) for e in eps if key in e]
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def behavior_vectors_for_cell(eps: list[dict[str, Any]]) -> np.ndarray | None:
    if not eps:
        return None
    rows = []
    for ep in eps:
        row = np.asarray(
            [float(ep.get(f"behavior_{name}", 0.0)) for name in BEHAVIOR_TELEMETRY_NAMES],
            dtype=np.float64,
        )
        rows.append(behavior_vector_from_telemetry_row(row))
    return np.mean(np.stack(rows, axis=0), axis=0)


def build_behavior_report(
    cells: CellEpisodes,
    *,
    opponents: list[str],
    maps: list[str],
    latents: tuple[int, ...],
) -> dict[str, Any]:
    vectors: list[np.ndarray] = []
    per_z: dict[str, Any] = {}
    for z in latents:
        cell_vecs = []
        for opponent in opponents:
            for map_name in maps:
                vec = behavior_vectors_for_cell(cells.get((opponent, z, map_name), []))
                if vec is not None:
                    cell_vecs.append(vec)
        if cell_vecs:
            mean_vec = np.mean(np.stack(cell_vecs, axis=0), axis=0)
            vectors.append(mean_vec)
            per_z[f"z{z}"] = {name: float(mean_vec[i]) for i, name in enumerate(FORCED_Z_BEHAVIOR_VECTOR_NAMES)}
    pair_count = len(latents) * (len(latents) - 1) // 2
    summary = (
        build_behavior_distance_profile(
            vectors,
            source="telemetry",
            pair_count=pair_count,
            latent_k=len(latents),
        )
        if len(vectors) >= 2
        else {}
    )
    cell_behavior = {}
    for opponent in opponents:
        for map_name in maps:
            for z in latents:
                eps = cells.get((opponent, z, map_name), [])
                cell_behavior[f"{opponent}|{map_name}|z{z}"] = {
                    name: _mean_metric(eps, f"behavior_{name}") for name in BEHAVIOR_TELEMETRY_NAMES
                }
    return {
        "per_z_behavior_vectors": per_z,
        "pairwise_summary": summary,
        "cell_behavior_means": cell_behavior,
        "forced_z_behavior_vector_names": list(FORCED_Z_BEHAVIOR_VECTOR_NAMES),
    }


__all__ = ["build_behavior_report", "behavior_vectors_for_cell"]
