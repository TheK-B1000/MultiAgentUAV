"""Oracle metrics from matched forced-z episodes."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

import numpy as np

from experiments.forced_z_eval.io import CellEpisodes

MetricName = Literal["return", "win_margin", "success"]


def metric_value(ep: dict[str, Any], metric: MetricName) -> float:
    if metric == "return":
        return float(ep.get("return", 0.0))
    if metric == "win_margin":
        return float(ep.get("win_margin", 0.0))
    return float(ep.get("success", 0.0))


def per_episode_oracle(
    cells: CellEpisodes,
    opponents: list[str],
    maps: list[str],
    latents: tuple[int, ...],
    *,
    metric: MetricName = "return",
) -> tuple[float, float, list[float], list[float], int]:
    oracle_vals: list[float] = []
    fixed_by_z: dict[int, list[float]] = defaultdict(list)
    for opponent in opponents:
        for map_name in maps:
            ep_lists = [cells.get((opponent, z, map_name), []) for z in latents]
            n = min(len(eps) for eps in ep_lists)
            if n == 0:
                continue
            for i in range(n):
                oracle_vals.append(float(max(metric_value(ep_lists[z][i], metric) for z in latents)))
            for z in latents:
                for ep in ep_lists[z]:
                    fixed_by_z[z].append(metric_value(ep, metric))
    if not oracle_vals:
        return float("nan"), float("nan"), [], [], -1
    best_z = max(latents, key=lambda z: float(np.mean(fixed_by_z[z])) if fixed_by_z[z] else -1e9)
    fixed_vals = fixed_by_z[best_z]
    return float(np.mean(oracle_vals)), float(np.mean(fixed_vals)), oracle_vals, fixed_vals, int(best_z)


def build_oracle_report(
    cells: CellEpisodes,
    *,
    opponents: list[str],
    maps: list[str],
    latents: tuple[int, ...],
    metric: MetricName = "return",
) -> dict[str, Any]:
    oracle_mean, fixed_mean, oracle_eps, fixed_eps, best_z = per_episode_oracle(
        cells, opponents, maps, latents, metric=metric
    )
    gap = float(oracle_mean - fixed_mean) if oracle_mean == oracle_mean and fixed_mean == fixed_mean else float("nan")
    return {
        "oracle_metric": metric,
        "best_fixed_z": int(best_z),
        "best_fixed_mean": fixed_mean,
        "oracle_mean": oracle_mean,
        "oracle_gap": gap,
        "oracle_gap_std": float(np.std(np.asarray(oracle_eps, dtype=np.float64))) if oracle_eps else 0.0,
        "matched_episode_count": len(oracle_eps),
        "fixed_episode_count": len(fixed_eps),
    }


__all__ = ["MetricName", "build_oracle_report", "metric_value", "per_episode_oracle"]
