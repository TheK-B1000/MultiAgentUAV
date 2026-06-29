#!/usr/bin/env python3
"""Matched-seed forced-z repertoire complementarity evaluation (V6I9 Stage 2).

Runs z0..z3 forced for entire episodes with matched map, opponent, seed,
initial state, and horizon. Reports win/margin/return plus navigation and
behavior telemetry, then scores the complementarity ladder:

  1. No oracle gap          -> latents differ cosmetically
  2. Oracle gap only        -> useful complementarity, routing may be hard
  3. Oracle gap + context   -> proceed to router training
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, DefaultDict, Dict, List, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.calibrate_hard_pool import (  # noqa: E402
    LATENTS,
    MAPS,
    OPPONENTS,
    CellEpisodes,
    run_forced_z_cells,
    _mean_margin,
    _wr,
)
from rl.behavior_telemetry import BEHAVIOR_TELEMETRY_NAMES  # noqa: E402
from rl.forced_z_behavior_vectors import (  # noqa: E402
    FORCED_Z_BEHAVIOR_VECTOR_NAMES,
    behavior_vector_from_telemetry_row,
    build_behavior_distance_profile,
)

MetricName = str


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I9 matched-seed forced-z complementarity eval")
    p.add_argument(
        "--checkpoint",
        default="checkpoints/2v2/final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip",
    )
    p.add_argument("--episodes", type=int, default=100, help="Matched episodes per (opponent, map, z) cell")
    p.add_argument("--device", default="cuda")
    p.add_argument("--opponents", nargs="+", default=list(OPPONENTS))
    p.add_argument("--maps", nargs="+", default=list(MAPS))
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--oracle-metric", choices=("return", "win_margin", "success"), default="return")
    p.add_argument("--stochastic", action="store_true")
    return p.parse_args()


def _metric_value(ep: Dict[str, Any], metric: MetricName) -> float:
    if metric == "return":
        return float(ep.get("return", 0.0))
    if metric == "win_margin":
        return float(ep.get("win_margin", 0.0))
    return float(ep.get("success", 0.0))


def _mean_metric(eps: List[Dict[str, Any]], key: str) -> float:
    vals = [float(e.get(key, np.nan)) for e in eps if key in e]
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def _aggregate_cell(eps: List[Dict[str, Any]]) -> Dict[str, float]:
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
        "mean_intercept_crossings": _mean_metric(eps, "blue_intercept_upper_crossings")
        + _mean_metric(eps, "blue_intercept_lower_crossings"),
        "mean_return_crossings": _mean_metric(eps, "blue_return_upper_crossings")
        + _mean_metric(eps, "blue_return_lower_crossings"),
        "mean_attack_crossings": _mean_metric(eps, "blue_attack_upper_crossings")
        + _mean_metric(eps, "blue_attack_lower_crossings"),
        "episodes": float(len(eps)),
    }
    for name in BEHAVIOR_TELEMETRY_NAMES:
        out[f"mean_behavior_{name}"] = _mean_metric(eps, f"behavior_{name}")
    return out


def _behavior_vectors_for_cell(eps: List[Dict[str, Any]]) -> np.ndarray | None:
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


def _per_episode_oracle(
    cells: CellEpisodes,
    opponents: List[str],
    maps: List[str],
    *,
    metric: MetricName,
) -> Tuple[float, float, List[float], List[float]]:
    """Return oracle mean, best-fixed mean, per-episode oracle values, per-episode fixed values."""
    oracle_vals: List[float] = []
    fixed_by_z: Dict[int, List[float]] = defaultdict(list)

    for opponent in opponents:
        for map_name in maps:
            ep_lists = [cells.get((opponent, z, map_name), []) for z in LATENTS]
            n = min(len(eps) for eps in ep_lists)
            if n == 0:
                continue
            for i in range(n):
                per_z = [_metric_value(ep_lists[z][i], metric) for z in LATENTS]
                oracle_vals.append(float(max(per_z)))
            for z in LATENTS:
                for ep in ep_lists[z]:
                    fixed_by_z[z].append(_metric_value(ep, metric))

    if not oracle_vals:
        return float("nan"), float("nan"), [], []

    best_z = max(LATENTS, key=lambda z: float(np.mean(fixed_by_z[z])) if fixed_by_z[z] else -1e9)
    fixed_vals = fixed_by_z[best_z]
    return (
        float(np.mean(oracle_vals)),
        float(np.mean(fixed_vals)),
        oracle_vals,
        fixed_vals,
    )


def _best_fixed_z(
    cells: CellEpisodes,
    opponents: List[str],
    maps: List[str],
    *,
    metric: MetricName,
) -> Tuple[int, float]:
    best_z, best_val = -1, -1e18
    for z in LATENTS:
        vals = []
        for opponent in opponents:
            for map_name in maps:
                for ep in cells.get((opponent, z, map_name), []):
                    vals.append(_metric_value(ep, metric))
        mean_val = float(np.mean(vals)) if vals else float("nan")
        if mean_val == mean_val and mean_val > best_val:
            best_z, best_val = int(z), mean_val
    return best_z, best_val


def _best_z_per_context(
    cells: CellEpisodes,
    opponents: List[str],
    maps: List[str],
    *,
    metric: MetricName,
    context_key: str,
) -> Dict[Tuple[str, ...], int]:
    grouped: DefaultDict[Tuple[str, ...], Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for (opponent, z, map_name), eps in cells.items():
        for ep in eps:
            ctx = (opponent, map_name, str(ep.get(context_key, "")))
            grouped[ctx][z].append(_metric_value(ep, metric))
    best: Dict[Tuple[str, ...], int] = {}
    for ctx, by_z in grouped.items():
        ranked = sorted(
            ((z, float(np.mean(vals))) for z, vals in by_z.items() if vals),
            key=lambda t: t[1],
            reverse=True,
        )
        if ranked:
            best[ctx] = int(ranked[0][0])
    return best


def _pairwise_behavior_summary(cells: CellEpisodes, opponents: List[str], maps: List[str]) -> Dict[str, float]:
    vectors: List[np.ndarray] = []
    for z in LATENTS:
        cell_vecs = []
        for opponent in opponents:
            for map_name in maps:
                vec = _behavior_vectors_for_cell(cells.get((opponent, z, map_name), []))
                if vec is not None:
                    cell_vecs.append(vec)
        if cell_vecs:
            vectors.append(np.mean(np.stack(cell_vecs, axis=0), axis=0))
    if len(vectors) < 2:
        return {}
    pair_count = len(LATENTS) * (len(LATENTS) - 1) // 2
    return build_behavior_distance_profile(
        vectors,
        source="telemetry",
        pair_count=pair_count,
        latent_k=len(LATENTS),
    )


def _ladder_verdict(
    *,
    oracle_gap: float,
    context_unique_best: int,
    behavior_pair_mean: float,
) -> str:
    if oracle_gap <= 1e-6:
        return "NO_ORACLE_GAP — latents differ, but not usefully under matched-seed oracle"
    if context_unique_best >= 2:
        return "ORACLE_GAP_PLUS_CONTEXT — proceed to router training"
    if behavior_pair_mean > 0.05 or oracle_gap > 0.0:
        return "ORACLE_GAP_ONLY — complementarity exists; routing may be difficult"
    return "INCONCLUSIVE"


def main() -> None:
    args = _parse_args()
    try:
        import plot.eval_rollout  # noqa: F401
    except ImportError as exc:
        print(f"ERROR: eval infrastructure unavailable: {exc}")
        sys.exit(1)

    out_dir = args.out_dir or os.path.join(SCRIPT_DIR, "repertoire_eval_runs")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(out_dir, f"v6i9_repertoire_complementarity_{stamp}.json")
    out_csv = os.path.join(out_dir, f"v6i9_repertoire_episodes_{stamp}.csv")

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Episodes   : {args.episodes} matched per (opponent, z, map)")
    print(f"Device     : {args.device}")
    print(f"Opponents  : {args.opponents}")
    print(f"Maps       : {args.maps}")
    print(f"Oracle metric: {args.oracle_metric}")
    print()

    cells = run_forced_z_cells(
        checkpoint=args.checkpoint,
        opponents=args.opponents,
        latents=tuple(LATENTS),
        maps=args.maps,
        n_episodes=args.episodes,
        device=args.device,
        deterministic=not args.stochastic,
        base_seed=int(args.base_seed),
        collect_behavior_mean=True,
    )

    cell_rows: List[Dict[str, Any]] = []
    episode_rows: List[Dict[str, Any]] = []
    for (opponent, z, map_name), eps in sorted(cells.items()):
        agg = _aggregate_cell(eps)
        row = {
            "opponent": opponent,
            "latent_z": z,
            "map": map_name,
            **{k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in agg.items()},
        }
        cell_rows.append(row)
        for ep_idx, ep in enumerate(eps):
            episode_rows.append(
                {
                    "opponent": opponent,
                    "latent_z": z,
                    "map": map_name,
                    "episode_index": ep_idx,
                    **ep,
                }
            )

    oracle_mean, fixed_mean, oracle_eps, _ = _per_episode_oracle(
        cells, args.opponents, args.maps, metric=args.oracle_metric
    )
    best_z, best_fixed = _best_fixed_z(cells, args.opponents, args.maps, metric=args.oracle_metric)
    oracle_gap = float(oracle_mean - fixed_mean) if oracle_mean == oracle_mean and fixed_mean == fixed_mean else float("nan")

    best_by_cell = {
        f"{opp}|{m}": max(LATENTS, key=lambda z: _wr(cells.get((opp, z, m), [])))
        for opp in args.opponents
        for m in args.maps
    }
    best_by_context = _best_z_per_context(
        cells, args.opponents, args.maps, metric=args.oracle_metric, context_key="episode_start_phase"
    )
    context_unique = len(set(best_by_context.values()))
    behavior_summary = _pairwise_behavior_summary(cells, args.opponents, args.maps)
    behavior_pair_mean = float(behavior_summary.get("forced_z_behavior_pair_distance_mean", 0.0) or 0.0)
    verdict = _ladder_verdict(
        oracle_gap=oracle_gap if oracle_gap == oracle_gap else 0.0,
        context_unique_best=context_unique,
        behavior_pair_mean=behavior_pair_mean,
    )

    print("\n=== Matched-seed forced-z summary ===")
    print(f"Best fixed z (global): z{best_z}  {args.oracle_metric}={best_fixed:.4f}")
    print(f"Hindsight oracle     : {args.oracle_metric}={oracle_mean:.4f}")
    print(f"Oracle gap           : {oracle_gap:+.4f}")
    print(f"Best z per map×opp   : {best_by_cell}")
    print(f"Unique best-z cells  : {len(set(best_by_cell.values()))} / {len(best_by_cell)}")
    print(f"Best z by start-phase: {len(best_by_context)} contexts, {context_unique} unique winners")
    print(f"Behavior pair mean   : {behavior_pair_mean:.4f}")
    print(f"\nLADDER VERDICT: {verdict}")

    print("\n--- Win-rate matrix ---")
    for map_name in args.maps:
        print(f"  map={map_name}")
        print("  " + f"{'':12s}" + "".join(f"  z={z}" for z in LATENTS))
        for opp in args.opponents:
            vals = [_wr(cells.get((opp, z, map_name), [])) for z in LATENTS]
            print(f"  {opp:<12s}" + "".join(f"  {v:5.1%}" if v == v else "   nan" for v in vals))

    print("\n--- Mean return matrix ---")
    for map_name in args.maps:
        print(f"  map={map_name}")
        print("  " + f"{'':12s}" + "".join(f"  z={z}" for z in LATENTS))
        for opp in args.opponents:
            vals = [_mean_metric(cells.get((opp, z, map_name), []), "return") for z in LATENTS]
            print(f"  {opp:<12s}" + "".join(f"  {v:7.2f}" if v == v else "     nan" for v in vals))

    report = {
        "checkpoint": args.checkpoint,
        "episodes_per_cell": int(args.episodes),
        "oracle_metric": args.oracle_metric,
        "best_fixed_z": int(best_z),
        "best_fixed_mean": fixed_mean,
        "oracle_mean": oracle_mean,
        "oracle_gap": oracle_gap,
        "oracle_gap_std": float(np.std(np.asarray(oracle_eps, dtype=np.float64))) if oracle_eps else 0.0,
        "best_z_per_cell": best_by_cell,
        "unique_best_z_cells": len(set(best_by_cell.values())),
        "context_best_z": { "|".join(k): v for k, v in best_by_context.items() },
        "context_unique_best_z": context_unique,
        "behavior_summary": behavior_summary,
        "forced_z_behavior_vector_names": list(FORCED_Z_BEHAVIOR_VECTOR_NAMES),
        "ladder_verdict": verdict,
        "cell_aggregates": cell_rows,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if episode_rows:
        fieldnames: List[str] = []
        for row in episode_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(episode_rows)

    print(f"\nWrote JSON report: {out_json}")
    print(f"Wrote episode CSV: {out_csv}")


if __name__ == "__main__":
    main()
