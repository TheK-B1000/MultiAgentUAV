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
from rl.custom_ppo import read_custom_ppo_metadata


def _slug(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return clean.strip("_") or "checkpoint"


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
    parser.add_argument("--agents", type=int, default=None, help="Agents per team; default reads checkpoint metadata")
    parser.add_argument("--opponents", nargs="+", default=["OP3", "OP4"], help="Scripted opponents to evaluate")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=None, help="CSV output dir (default: AICTFProject/csv)")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", action="store_false", dest="deterministic")
    args = parser.parse_args()

    checkpoint = os.path.abspath(args.checkpoint if args.checkpoint.endswith(".zip") else args.checkpoint + ".zip")
    if not os.path.isfile(checkpoint):
        sys.exit(f"[ERROR] checkpoint not found: {checkpoint}")

    meta = read_custom_ppo_metadata(checkpoint)
    agents = int(args.agents or meta.get("n_blue", 2))
    label = args.label or os.path.splitext(os.path.basename(checkpoint))[0]
    label_slug = _slug(label)
    mode = f"{agents}v{agents}"
    out_dir = os.path.abspath(args.out_dir or os.path.join(PROJECT_ROOT, "csv"))
    os.makedirs(out_dir, exist_ok=True)

    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig

    aggregate_rows: list[dict[str, Any]] = []
    for opp_idx, raw_opp in enumerate(args.opponents):
        opponent = str(raw_opp).strip().upper()
        cfg = GPUFieldConfig(
            n_envs=1,
            max_blue_agents=agents,
            max_red_agents=agents,
            max_decision_steps=400,
            aquaticus_profile=True,
            rules_profile="OURS",
            device=args.device,
            seed=int(args.seed) + opp_idx,
        )
        env = GPUCTFVecEnv(cfg)
        try:
            print(f"[eval_checkpoint] {label} {mode} vs {opponent}: {args.episodes} episode(s)")
            episodes = run_eval_episodes(
                checkpoint,
                env,
                int(args.episodes),
                args.device,
                opponent,
                deterministic=bool(args.deterministic),
                progress_every=max(1, int(args.episodes) // 10) if int(args.episodes) >= 10 else 0,
            )
        finally:
            env.close()

        for idx, row in enumerate(episodes, start=1):
            row["episode_id"] = idx
            row["label"] = label
            row["setting"] = mode
            row["opponent"] = opponent
            row["checkpoint"] = checkpoint

        episode_path = os.path.join(out_dir, f"eval_{label_slug}_{mode}_{opponent}_{int(args.episodes)}ep.csv")
        _write_rows(
            episode_path,
            episodes,
            ["label", "setting", "opponent", "episode_id", "success", "blue_score", "red_score", "steps", "return"],
        )
        w, l, d = count_wld(episodes)
        agg = compute_aggregates(episodes)
        agg_row: dict[str, Any] = {
            "label": label,
            "setting": mode,
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
        ["label", "setting", "opponent", "episodes", "wins", "losses", "draws", "success_rate", "success_rate_std"],
    )
    print(f"[eval_checkpoint] aggregate: {aggregate_path}")


if __name__ == "__main__":
    main()
