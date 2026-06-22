#!/usr/bin/env python3
"""Forced-latent repertoire evaluation for v6i5 checkpoints.

Pure evaluation. This tool freezes one checkpoint, runs natural and fixed-z
rollouts through the existing qualitative rollout harness, and writes a
v6i5-named artifact set for repertoire readiness review.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl.behavior_telemetry import BEHAVIOR_TELEMETRY_NAMES
from rl.custom_ppo.inference import read_custom_ppo_metadata
from tools import qualitative_rollout


def _to_float(value: Any) -> float:
    try:
        text = str(value).strip()
        if not text:
            return math.nan
        return float(text)
    except Exception:
        return math.nan


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _load_run_config_for_metrics(metrics_csv: Path | None) -> dict[str, Any]:
    if metrics_csv is None:
        return {}
    stem = metrics_csv.stem
    if stem.endswith("_metrics"):
        run_config = metrics_csv.with_name(f"{stem[:-8]}_run_config.json")
        if run_config.exists():
            try:
                return json.loads(run_config.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def _resolve_map_layout(cli_value: str | None, metrics_csv: Path | None) -> str:
    if cli_value:
        return str(cli_value).strip().lower()
    run_config = _load_run_config_for_metrics(metrics_csv)
    resolved = run_config.get("resolved_ppo_config") if isinstance(run_config, dict) else None
    if isinstance(resolved, dict):
        value = resolved.get("map_layout")
        if value:
            return str(value).strip().lower()
    return "map_a_open"


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(preferred or [])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_episode_results(steps_csv: Path) -> list[dict[str, Any]]:
    rows = _read_csv(steps_csv)
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row.get("opponent", ""),
            row.get("mode", ""),
            row.get("fixed_z_id", ""),
            row.get("episode_idx", ""),
        )
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (opponent, mode, fixed_z, episode_idx), ep_rows in sorted(groups.items()):
        ep_rows.sort(key=lambda r: int(_to_float(r.get("step", "0"))))
        last = ep_rows[-1]
        blue_score = _to_float(last.get("blue_score", ""))
        red_score = _to_float(last.get("red_score", ""))
        out.append(
            {
                "opponent": opponent,
                "mode": mode,
                "fixed_z_id": int(_to_float(fixed_z)) if math.isfinite(_to_float(fixed_z)) else -1,
                "episode_idx": int(_to_float(episode_idx)) if math.isfinite(_to_float(episode_idx)) else -1,
                "decision_steps": len(ep_rows),
                "blue_score": blue_score,
                "red_score": red_score,
                "score_diff": blue_score - red_score if math.isfinite(blue_score) and math.isfinite(red_score) else "",
                "blue_win": 1 if math.isfinite(blue_score) and math.isfinite(red_score) and blue_score > red_score else 0,
                "draw": 1 if math.isfinite(blue_score) and math.isfinite(red_score) and blue_score == red_score else 0,
                "return": blue_score - red_score if math.isfinite(blue_score) and math.isfinite(red_score) else "",
            }
        )
    return out


def build_summary(rollout_by_z_csv: Path) -> list[dict[str, Any]]:
    rows = _read_csv(rollout_by_z_csv)
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "opponent": row.get("opponent", ""),
                "mode": row.get("mode", ""),
                "z": row.get("z", ""),
                "n_episodes": row.get("n_episodes_touched", ""),
                "n_steps": row.get("n_steps", ""),
                "win_rate": row.get("blue_win_rate", ""),
                "return_mean": (
                    _to_float(row.get("blue_scores_per_episode", ""))
                    - _to_float(row.get("red_scores_per_episode", ""))
                ),
                "capture_rate": row.get("blue_scores_per_episode", ""),
                "score_differential": (
                    _to_float(row.get("blue_scores_per_episode", ""))
                    - _to_float(row.get("red_scores_per_episode", ""))
                ),
            }
        )
    return out


def build_behavior_matrix(rollout_by_z_csv: Path) -> list[dict[str, Any]]:
    rows = _read_csv(rollout_by_z_csv)
    matrix: list[dict[str, Any]] = []
    for row in rows:
        if row.get("mode") != "fixed_z":
            continue
        out: dict[str, Any] = {
            "opponent": row.get("opponent", ""),
            "z": row.get("z", ""),
            "n_steps": row.get("n_steps", ""),
            "win_rate": row.get("blue_win_rate", ""),
        }
        for name in BEHAVIOR_TELEMETRY_NAMES:
            out[name] = row.get(f"{name}_mean", "")
        matrix.append(out)
    return matrix


def _behavior_vector(row: dict[str, Any]) -> list[float]:
    return [_to_float(row.get(name, "")) for name in BEHAVIOR_TELEMETRY_NAMES]


def _euclidean(a: list[float], b: list[float]) -> float:
    vals = [(x - y) ** 2 for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    if not vals:
        return math.nan
    return math.sqrt(sum(vals))


def build_pairwise_distances(behavior_matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_opp: dict[str, list[dict[str, Any]]] = {}
    for row in behavior_matrix:
        by_opp.setdefault(str(row.get("opponent", "")), []).append(row)
    out: list[dict[str, Any]] = []
    for opponent, rows in sorted(by_opp.items()):
        rows = sorted(rows, key=lambda r: int(_to_float(r.get("z", "0"))))
        for left, right in combinations(rows, 2):
            left_vec = _behavior_vector(left)
            right_vec = _behavior_vector(right)
            behavior_distance = _euclidean(left_vec, right_vec)
            wr_left = _to_float(left.get("win_rate", ""))
            wr_right = _to_float(right.get("win_rate", ""))
            out.append(
                {
                    "opponent": opponent,
                    "z_i": left.get("z", ""),
                    "z_j": right.get("z", ""),
                    "pairwise_behavior_distance": behavior_distance,
                    "pairwise_action_distribution_jsd": "",
                    "win_rate_i": wr_left,
                    "win_rate_j": wr_right,
                    "win_rate_abs_diff": abs(wr_left - wr_right)
                    if math.isfinite(wr_left) and math.isfinite(wr_right)
                    else "",
                }
            )
    return out


def build_readiness_report(
    *,
    checkpoint: Path,
    opponents: list[str],
    behavior_matrix: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    behavior_margin: float,
    competence_floor: float,
) -> dict[str, Any]:
    finite_distances = [
        _to_float(row.get("pairwise_behavior_distance", ""))
        for row in pairwise
        if math.isfinite(_to_float(row.get("pairwise_behavior_distance", "")))
    ]
    pairs_above = sum(1 for v in finite_distances if v >= behavior_margin)
    fixed_rows = [row for row in summary_rows if row.get("mode") == "fixed_z"]
    win_rates = [_to_float(row.get("win_rate", "")) for row in fixed_rows]
    finite_wr = [v for v in win_rates if math.isfinite(v)]
    min_competence = min(finite_wr) if finite_wr else math.nan

    best_by_opp: dict[str, int] = {}
    for opponent in opponents:
        opp_rows = [row for row in fixed_rows if row.get("opponent") == opponent]
        if not opp_rows:
            continue
        best = max(opp_rows, key=lambda r: _to_float(r.get("return_mean", "")))
        z = _to_float(best.get("z", ""))
        if math.isfinite(z):
            best_by_opp[opponent] = int(z)
    contextual_advantage = len(set(best_by_opp.values())) > 1
    readiness = {
        "different_z_behaviors": bool(finite_distances and pairs_above > 0),
        "behavior_pairs_above_margin": pairs_above,
        "behavior_pairs_total": len(finite_distances),
        "min_latent_competence": min_competence,
        "competence_floor_pass": bool(math.isfinite(min_competence) and min_competence >= competence_floor),
        "contextual_advantage_detected": contextual_advantage,
        "best_z_by_opponent": best_by_opp,
    }
    readiness["ready_for_router_training"] = bool(
        readiness["different_z_behaviors"]
        and readiness["competence_floor_pass"]
        and readiness["contextual_advantage_detected"]
    )
    return {
        "protocol": "v6i5_forced_latent_repertoire_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "opponents": opponents,
        "behavior_margin": behavior_margin,
        "competence_floor": competence_floor,
        "readiness": readiness,
    }


def run(argv: list[str] | None = None) -> dict[str, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metrics-csv", default=None)
    parser.add_argument("--opponents", nargs="+", default=["OP5", "OP6", "OP7", "OP4"])
    parser.add_argument("--episodes-per-mode", type=int, default=25)
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--map-layout", default=None)
    parser.add_argument("--include-natural", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--behavior-margin", type=float, default=0.05)
    parser.add_argument("--competence-floor", type=float, default=0.20)
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    metrics_csv = Path(args.metrics_csv).expanduser().resolve() if args.metrics_csv else None
    if args.agents is None:
        meta = read_custom_ppo_metadata(str(checkpoint))
        agents = int(meta.get("n_blue", 4))
    else:
        agents = int(args.agents)
    map_layout = _resolve_map_layout(args.map_layout, metrics_csv)
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else checkpoint.parent / "v6i5_forced_latent_repertoire"
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = ["fixed_z"]
    if bool(args.include_natural):
        modes.insert(0, "natural")
    qualitative_outputs = qualitative_rollout.run(
        checkpoint=checkpoint,
        opponents=list(args.opponents),
        episodes_per_mode=int(args.episodes_per_mode),
        agents=agents,
        device=str(args.device),
        seed=int(args.seed),
        out_dir=out_dir,
        modes=modes,
        deterministic=not bool(args.stochastic),
        max_steps=int(args.max_steps),
        map_layout=map_layout,
    )
    steps_csv = qualitative_outputs["steps"]
    by_z_csv = qualitative_outputs["rollout_by_z"]
    episode_rows = build_episode_results(steps_csv)
    summary_rows = build_summary(by_z_csv)
    behavior_matrix = build_behavior_matrix(by_z_csv)
    pairwise = build_pairwise_distances(behavior_matrix)
    report = build_readiness_report(
        checkpoint=checkpoint,
        opponents=list(args.opponents),
        behavior_matrix=behavior_matrix,
        pairwise=pairwise,
        summary_rows=summary_rows,
        behavior_margin=float(args.behavior_margin),
        competence_floor=float(args.competence_floor),
    )

    prefix = checkpoint.stem
    manifest_path = out_dir / f"{prefix}_v6i5_repertoire_manifest.json"
    episode_path = out_dir / f"{prefix}_v6i5_repertoire_episode_results.csv"
    summary_path = out_dir / f"{prefix}_v6i5_repertoire_summary.csv"
    behavior_path = out_dir / f"{prefix}_v6i5_repertoire_behavior_matrix.csv"
    pairwise_path = out_dir / f"{prefix}_v6i5_repertoire_pairwise_distances.csv"
    report_path = out_dir / f"{prefix}_v6i5_repertoire_report.json"

    outputs = {
        "qualitative_steps": str(steps_csv),
        "qualitative_rollout_by_z": str(by_z_csv),
        "qualitative_strategy_evidence": str(qualitative_outputs["strategy_evidence"]),
        "v6i5_repertoire_episode_results": str(episode_path),
        "v6i5_repertoire_summary": str(summary_path),
        "v6i5_repertoire_behavior_matrix": str(behavior_path),
        "v6i5_repertoire_pairwise_distances": str(pairwise_path),
        "v6i5_repertoire_report": str(report_path),
    }
    manifest = {
        "protocol": "v6i5_forced_latent_repertoire_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "metrics_csv": str(metrics_csv) if metrics_csv else None,
        "opponents": list(args.opponents),
        "episodes_per_mode": int(args.episodes_per_mode),
        "agents": agents,
        "device": str(args.device),
        "seed": int(args.seed),
        "deterministic": not bool(args.stochastic),
        "map_layout": map_layout,
        "modes": modes,
        "outputs": outputs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(episode_path, episode_rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(behavior_path, behavior_matrix)
    _write_csv(pairwise_path, pairwise)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[v6i5_repertoire] manifest: {manifest_path}")
    print(f"[v6i5_repertoire] report: {report_path}")
    return {
        "manifest": manifest_path,
        "episode_results": episode_path,
        "summary": summary_path,
        "behavior_matrix": behavior_path,
        "pairwise_distances": pairwise_path,
        "report": report_path,
    }


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
