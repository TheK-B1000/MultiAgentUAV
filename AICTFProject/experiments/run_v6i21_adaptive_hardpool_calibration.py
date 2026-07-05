#!/usr/bin/env python3
"""Calibrate adaptive OP8-OP12 hardpool v2 against a strong blue checkpoint.

Goal: verify blue WR is not saturated at 100% (target band 35-65% average).
Does not train router or birth specialists.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any

from plot.eval_rollout import run_eval_episodes
from rl.evaluation.opponent_resolution import set_opponent

def _make_env(*, map_name: str, seed: int, device: str, max_decision_steps: int = 240) -> Any:
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig

    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=2,
        max_red_agents=2,
        map_layout=map_name,
        max_decision_steps=int(max_decision_steps),
        aquaticus_profile=True,
        rules_profile="OURS",
        device=device,
        seed=seed,
    )
    return GPUCTFVecEnv(cfg)


DEFAULT_OPPONENTS = ("OP8", "OP9", "OP10", "OP11", "OP12")
DEFAULT_MAPS = ("map_b", "map_b_split_lane_v2")
TARGET_WR_LOW = 0.35
TARGET_WR_HIGH = 0.65


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I21 adaptive hardpool WR calibration")
    p.add_argument(
        "--checkpoint",
        default="checkpoints/2v2/final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip",
    )
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--device", default="cuda")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--maps", nargs="+", default=list(DEFAULT_MAPS))
    p.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    p.add_argument("--out-dir", default="artifacts/v6i21_adaptive_hardpool_calibration")
    p.add_argument("--progress-every", type=int, default=5)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for opp_idx, opponent in enumerate(args.opponents):
        for map_idx, map_name in enumerate(args.maps):
            cell_seed = int(args.base_seed) + opp_idx * 1000 + map_idx * 100
            env = _make_env(map_name=map_name, seed=cell_seed, device=args.device)
            try:
                set_opponent(env, opponent)
                eps = run_eval_episodes(
                    args.checkpoint,
                    env,
                    int(args.episodes),
                    args.device,
                    opponent,
                    deterministic=True,
                    latent_eval_seed=cell_seed,
                    progress_every=int(args.progress_every),
                )
            finally:
                env.close()

            n = len(eps)
            wins = sum(int(e.get("success", 0)) for e in eps)
            wr = wins / n if n else float("nan")
            blue_scores = [int(e.get("blue_score", 0)) for e in eps]
            red_scores = [int(e.get("red_score", 0)) for e in eps]
            row = {
                "opponent": opponent,
                "map": map_name,
                "episodes": n,
                "win_rate": wr,
                "blue_score_mean": sum(blue_scores) / n if n else 0.0,
                "red_score_mean": sum(red_scores) / n if n else 0.0,
                "seed": cell_seed,
            }
            rows.append(row)
            print(
                f"{opponent:>4} {map_name:>22}: WR={wr:6.1%} "
                f"blue={row['blue_score_mean']:.2f} red={row['red_score_mean']:.2f} (n={n})"
            )

    valid_wr = [r["win_rate"] for r in rows if r["episodes"] > 0]
    mean_wr = sum(valid_wr) / len(valid_wr) if valid_wr else float("nan")
    saturated = sum(1 for w in valid_wr if w >= 0.95)
    in_band = sum(1 for w in valid_wr if TARGET_WR_LOW <= w <= TARGET_WR_HIGH)
    report = {
        "checkpoint": args.checkpoint,
        "opponents": list(args.opponents),
        "maps": list(args.maps),
        "episodes_per_cell": int(args.episodes),
        "cells": rows,
        "mean_win_rate": mean_wr,
        "cells_in_target_band": in_band,
        "cells_saturated_95plus": saturated,
        "target_band": [TARGET_WR_LOW, TARGET_WR_HIGH],
        "calibration_pass": bool(valid_wr) and TARGET_WR_LOW <= mean_wr <= TARGET_WR_HIGH and saturated == 0,
        "note": "OP8-OP12 upgraded in-place at v6i21; pre-v6i21 hardpool results not comparable.",
    }
    out_path = out_dir / "calibration_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nMean WR={mean_wr:.1%}  in-band cells={in_band}/{len(valid_wr)}  saturated={saturated}")
    print(f"Calibration pass={report['calibration_pass']}")
    print(f"Wrote {out_path}")
    return 0 if report["calibration_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
