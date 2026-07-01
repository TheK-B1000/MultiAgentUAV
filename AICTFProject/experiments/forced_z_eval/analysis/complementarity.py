"""Complementarity ladder analysis from canonical forced-z episodes."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from experiments.forced_z_eval.analysis.behavior import build_behavior_report
from experiments.forced_z_eval.analysis.oracle import MetricName, build_oracle_report, metric_value
from experiments.forced_z_eval.io import CellEpisodes
from rl.behavior_telemetry import BEHAVIOR_TELEMETRY_NAMES


def _wr(eps: list[dict[str, Any]]) -> float:
    if not eps:
        return float("nan")
    return sum(int(e.get("success", 0)) for e in eps) / len(eps)


def _mean_margin(eps: list[dict[str, Any]]) -> float:
    if not eps:
        return float("nan")
    return sum(int(e.get("win_margin", 0)) for e in eps) / len(eps)


def _mean_metric(eps: list[dict[str, Any]], key: str) -> float:
    import numpy as np

    vals = [float(e.get(key, float("nan"))) for e in eps if key in e]
    vals = [v for v in vals if v == v]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _aggregate_cell(eps: list[dict[str, Any]]) -> dict[str, float]:
    out = {
        "win_rate": _wr(eps),
        "mean_margin": _mean_margin(eps),
        "mean_return": _mean_metric(eps, "return"),
        "mean_steps": _mean_metric(eps, "steps"),
        "mean_time_to_first_score": _mean_metric(eps, "time_to_first_score"),
        "mean_blue_score": _mean_metric(eps, "blue_score"),
        "mean_collisions": _mean_metric(eps, "collisions_per_episode"),
        "mean_blue_stuck_steps": _mean_metric(eps, "blue_stuck_steps"),
        "mean_blue_blocked_events": _mean_metric(eps, "blue_blocked_movement_events"),
        "episodes": float(len(eps)),
    }
    for name in BEHAVIOR_TELEMETRY_NAMES:
        out[f"mean_behavior_{name}"] = _mean_metric(eps, f"behavior_{name}")
    return out


def _best_z_per_context(
    cells: CellEpisodes,
    opponents: list[str],
    maps: list[str],
    latents: tuple[int, ...],
    *,
    metric: MetricName,
    context_key: str,
) -> dict[tuple[str, ...], int]:
    grouped: dict[tuple[str, ...], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (opponent, z, map_name), eps in cells.items():
        for ep in eps:
            ctx = (opponent, map_name, str(ep.get(context_key, "")))
            grouped[ctx][z].append(metric_value(ep, metric))
    best: dict[tuple[str, ...], int] = {}
    for ctx, by_z in grouped.items():
        ranked = sorted(
            ((z, sum(vals) / len(vals)) for z, vals in by_z.items() if vals),
            key=lambda t: t[1],
            reverse=True,
        )
        if ranked:
            best[ctx] = int(ranked[0][0])
    return best


def _ladder_verdict(*, oracle_gap: float, context_unique_best: int, behavior_pair_mean: float) -> str:
    if oracle_gap <= 1e-6:
        return "NO_ORACLE_GAP — latents differ, but not usefully under matched-seed oracle"
    if context_unique_best >= 2:
        return "ORACLE_GAP_PLUS_CONTEXT — proceed to router training"
    if behavior_pair_mean > 0.05 or oracle_gap > 0.0:
        return "ORACLE_GAP_ONLY — complementarity exists; routing may be difficult"
    return "INCONCLUSIVE"


def build_complementarity_report(
    cells: CellEpisodes,
    *,
    opponents: list[str],
    maps: list[str],
    latents: tuple[int, ...],
    metric: MetricName = "return",
) -> dict[str, Any]:
    oracle_report = build_oracle_report(cells, opponents=opponents, maps=maps, latents=latents, metric=metric)
    behavior_report = build_behavior_report(cells, opponents=opponents, maps=maps, latents=latents)
    best_by_cell = {
        f"{opp}|{m}": max(latents, key=lambda z: _wr(cells.get((opp, z, m), [])))
        for opp in opponents
        for m in maps
    }
    best_by_context = _best_z_per_context(
        cells, opponents, maps, latents, metric=metric, context_key="episode_start_phase"
    )
    context_unique = len(set(best_by_context.values()))
    behavior_pair_mean = float(
        behavior_report.get("pairwise_summary", {}).get("forced_z_behavior_pair_distance_mean", 0.0) or 0.0
    )
    oracle_gap = float(oracle_report.get("oracle_gap", 0.0) or 0.0)
    verdict = _ladder_verdict(
        oracle_gap=oracle_gap if oracle_gap == oracle_gap else 0.0,
        context_unique_best=context_unique,
        behavior_pair_mean=behavior_pair_mean,
    )
    cell_rows = []
    for (opponent, z, map_name), eps in sorted(cells.items()):
        cell_rows.append({"opponent": opponent, "latent_z": z, "map": map_name, **_aggregate_cell(eps)})
    return {
        **oracle_report,
        "best_z_per_cell": best_by_cell,
        "unique_best_z_cells": len(set(best_by_cell.values())),
        "context_best_z": {"|".join(k): v for k, v in best_by_context.items()},
        "context_unique_best_z": context_unique,
        "behavior_summary": behavior_report.get("pairwise_summary", {}),
        "cell_aggregates": cell_rows,
        "ladder_verdict": verdict,
    }


def print_complementarity_report(report: dict[str, Any]) -> None:
    metric = str(report.get("oracle_metric", "return"))
    print("\n=== Matched-seed forced-z summary ===")
    print(f"Best fixed z (global): z{report.get('best_fixed_z')}  {metric}={report.get('best_fixed_mean', float('nan')):.4f}")
    print(f"Hindsight oracle     : {metric}={report.get('oracle_mean', float('nan')):.4f}")
    print(f"Oracle gap           : {report.get('oracle_gap', float('nan')):+.4f}")
    print(f"Best z per map×opp   : {report.get('best_z_per_cell')}")
    print(f"Unique best-z cells  : {report.get('unique_best_z_cells')} / {len(report.get('best_z_per_cell', {}))}")
    print(
        f"Best z by start-phase: {len(report.get('context_best_z', {}))} contexts, "
        f"{report.get('context_unique_best_z')} unique winners"
    )
    print(
        f"Behavior pair mean   : "
        f"{report.get('behavior_summary', {}).get('forced_z_behavior_pair_distance_mean', 0.0):.4f}"
    )
    print(f"\nLADDER VERDICT: {report.get('ladder_verdict')}")


__all__ = ["build_complementarity_report", "print_complementarity_report"]
