#!/usr/bin/env python3
"""Evaluate one custom PPO checkpoint against scripted opponents.

This is the ablation-friendly evaluator for the final audit phase. It writes
both per-episode CSV rows and aggregate CSV rows, including latent strategy
diagnostics when the checkpoint exposes them.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from eval_rollout import compute_aggregates, count_wld, run_eval_episodes
from opponent_params import OP5_RUSHER_TUNING_TAG
from rl.custom_ppo import read_custom_ppo_metadata


def _slug(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return clean.strip("_") or "checkpoint"


def _label_append_op5_tuning_tag(label: str, opponents: list[str], *, no_suffix: bool) -> str:
    """Ensure CSV labels encode the OP5 scripted tuning revision when OP5 is evaluated."""
    if no_suffix or not OP5_RUSHER_TUNING_TAG:
        return label
    tags = {str(o).strip().upper() for o in opponents}
    if "OP5" not in tags and "OP5_RUSHER" not in tags:
        return label
    tail = f"_op5_{OP5_RUSHER_TUNING_TAG}"
    if label.rstrip().endswith(tail):
        return label
    return f"{label.rstrip()}{tail}"


def _union_fieldnames(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    fields = list(preferred)
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    return fields


def _write_rows(path: str, rows: list[dict[str, Any]], preferred: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fields = _union_fieldnames(rows, preferred)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one local PPO checkpoint and write ablation CSVs.")
    parser.add_argument("--checkpoint", required=True, help="Path to custom PPO checkpoint .zip")
    parser.add_argument("--label", default=None, help="Short label used in CSVs/filenames")
    parser.add_argument(
        "--no-op5-tuning-suffix",
        action="store_true",
        help=(
            "Do not append _op5_<tuning_tag> to --label when OP5/OP5_RUSHER is in --opponents "
            f"(default tag from opponent_params: {OP5_RUSHER_TUNING_TAG!r})."
        ),
    )
    parser.add_argument("--agents", type=int, default=None, help="Agents per team; default reads checkpoint metadata")
    parser.add_argument("--opponents", nargs="+", default=["OP3", "OP4", "OP5_RUSHER"], help="Scripted opponents to evaluate")
    parser.add_argument("--map-sets", nargs="+", default=["train", "eval"], choices=["train", "eval"], help="Map splits to evaluate")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-coordination-metrics",
        action="store_true",
        help="Disable per-episode coord_* fields (macro trajectory agreement / correlation).",
    )
    parser.add_argument(
        "--e3-step-telemetry",
        action="store_true",
        help="Write step-level strategy/behavior CSV telemetry for eval episodes.",
    )
    parser.add_argument("--out-dir", type=str, default=None, help="CSV output dir (default: AICTFProject/csv)")
    # Default True: argmax / greedy match paper-style *evaluation* (stochastic is for ablations / debugging).
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", action="store_false", dest="deterministic")
    parser.add_argument(
        "--fixed-latent-id",
        type=int,
        default=None,
        help=(
            "Latent only: deployment ablation — clamp every episode to this strategy id "
            "(0..K-1); skips q_phi(s) routing."
        ),
    )
    parser.add_argument(
        "--latent-resample-every",
        type=int,
        default=None,
        help=(
            "Latent only: re-run q_phi and broadcast a new z every N decision steps "
            "(0 = episode start only, matching training default)."
        ),
    )
    args = parser.parse_args()

    checkpoint = os.path.abspath(args.checkpoint if args.checkpoint.endswith(".zip") else args.checkpoint + ".zip")
    if not os.path.isfile(checkpoint):
        sys.exit(f"[ERROR] checkpoint not found: {checkpoint}")

    meta = read_custom_ppo_metadata(checkpoint)
    agents = int(args.agents or meta.get("n_blue", 2))
    label = args.label or os.path.splitext(os.path.basename(checkpoint))[0]
    label = _label_append_op5_tuning_tag(label, list(args.opponents), no_suffix=bool(args.no_op5_tuning_suffix))
    label_slug = _slug(label)
    mode = f"{agents}v{agents}"
    out_dir = os.path.abspath(args.out_dir or os.path.join(PROJECT_ROOT, "csv"))
    os.makedirs(out_dir, exist_ok=True)

    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig

    aggregate_rows: list[dict[str, Any]] = []
    for map_idx, raw_map_set in enumerate(args.map_sets):
        map_set = str(raw_map_set).strip().lower()
        for opp_idx, raw_opp in enumerate(args.opponents):
            opponent = str(raw_opp).strip().upper()
            cfg = GPUFieldConfig(
                n_envs=1,
                max_blue_agents=agents,
                max_red_agents=agents,
                map_set=map_set,
                max_decision_steps=400,
                aquaticus_profile=True,
                rules_profile="OURS",
                device=args.device,
                seed=int(args.seed) + 1000 * map_idx + opp_idx,
            )
            env = GPUCTFVecEnv(cfg)
            episode_path = os.path.join(out_dir, f"eval_{label_slug}_{mode}_{map_set}_{opponent}_{int(args.episodes)}ep.csv")
            e3_path = episode_path.replace(".csv", "_e3_steps.csv") if args.e3_step_telemetry else None
            try:
                print(f"[eval_checkpoint] {label} {mode} map={map_set} vs {opponent}: {args.episodes} episode(s)")
                episodes = run_eval_episodes(
                    checkpoint,
                    env,
                    int(args.episodes),
                    args.device,
                    opponent,
                    deterministic=bool(args.deterministic),
                    coordination_metrics=not bool(args.no_coordination_metrics),
                    progress_every=max(1, int(args.episodes) // 10) if int(args.episodes) >= 10 else 0,
                    fixed_latent_id=args.fixed_latent_id,
                    latent_resample_every_n=args.latent_resample_every,
                    e3_step_telemetry_path=e3_path,
                )
            finally:
                env.close()

            for idx, row in enumerate(episodes, start=1):
                row["episode_id"] = idx
                row["label"] = label
                row["setting"] = mode
                row["map_set"] = map_set
                row["opponent"] = opponent
                row["checkpoint"] = checkpoint

            _write_rows(
                episode_path,
                episodes,
                ["label", "setting", "map_set", "opponent", "episode_id", "success", "blue_score", "red_score", "steps", "return"],
            )
            w, l, d = count_wld(episodes)
            agg = compute_aggregates(episodes)
            agg_row: dict[str, Any] = {
                "label": label,
                "setting": mode,
                "map_set": map_set,
                "opponent": opponent,
                "checkpoint": checkpoint,
                "episodes": len(episodes),
                "wins": w,
                "losses": l,
                "draws": d,
            }
            agg_row.update(agg)
            aggregate_rows.append(agg_row)
            print(
                f"  W={w} L={l} D={d} WR={100.0 * w / max(1, w + l + d):.1f}% | "
                f"episodes: {episode_path}"
            )

    aggregate_path = os.path.join(out_dir, f"eval_{label_slug}_{mode}_aggregate.csv")
    _write_rows(
        aggregate_path,
        aggregate_rows,
        ["label", "setting", "map_set", "opponent", "episodes", "wins", "losses", "draws", "success_rate", "success_rate_std"],
    )
    print(f"[eval_checkpoint] aggregate: {aggregate_path}")


if __name__ == "__main__":
    main()
