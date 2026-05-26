#!/usr/bin/env python3
"""Evaluate hard-pool ablation checkpoints against OP4 (zero-shot held-out opponent).

Loads each ``final_..._hardpool_..._2v2.zip`` checkpoint and runs eval-only
episodes against scripted OP4. Optionally also evaluates against OP3 / OP5 / OP6
so you get a single comparison table covering training opponents + held-out OP4.

This script is a thin orchestrator around the existing eval infrastructure in
``plot/eval_rollout.py``:

    run_eval_episodes(model_path, env, n_episodes, device, opponent, ...)
    count_wld(episodes)
    compute_aggregates(episodes)

No training, no gradient steps, no checkpoint mutation. The eval environment
uses ``map_set=eval`` by default (held-out maps) and ``deterministic=True``
(greedy/argmax policy), matching the paper-style evaluation protocol.

Outputs (under --out-dir, default = checkpoints/2v2/eval_op4_zero_shot/):

    eval_<run_tag>_<opp>_<N>ep_episodes.csv      one row per evaluation episode
    op4_zero_shot_comparison.csv                  one row per (run_tag, opponent)
                                                  with WR, mean_return, mean_steps,
                                                  ±SE, etc.

Usage
-----

Default: eval all five hard-pool ablations against OP3, OP4, OP5, OP6 on
``eval`` maps with 100 episodes each:

    python experiments/eval_op4_zero_shot.py

OP4 only, faster, train maps:

    python experiments/eval_op4_zero_shot.py --opponents OP4 \\
        --map-set train --episodes 200

Stochastic policy (matches training-time action sampling):

    python experiments/eval_op4_zero_shot.py --stochastic

Specific checkpoints by run_tag:

    python experiments/eval_op4_zero_shot.py \\
        --run-tags plan_faithful_latent_persist_entropy_hardpool_1m_2v2 \\
                   plan_faithful_no_latent_hardpool_1m_2v2
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PLOT_DIR = os.path.join(PROJECT_ROOT, "plot")
for p in (PROJECT_ROOT, PLOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


DEFAULT_HARDPOOL_RUN_TAGS: tuple[str, ...] = (
    "plan_faithful_latent_persist_entropy_hardpool_1m_2v2",
    "plan_faithful_no_latent_hardpool_1m_2v2",
    "plan_faithful_latent_k1_hardpool_1m_2v2",
    "plan_faithful_latent_no_persistence_hardpool_1m_2v2",
    "plan_faithful_latent_no_entropy_hardpool_1m_2v2",
)


def _resolve_checkpoint(checkpoint_dir: str, run_tag: str) -> str | None:
    """Find ``final_<run_tag>.zip`` under ``checkpoint_dir``."""
    candidates = (
        os.path.join(checkpoint_dir, f"final_{run_tag}.zip"),
        os.path.join(checkpoint_dir, f"{run_tag}.zip"),
    )
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _write_rows(path: str, rows: list[dict[str, Any]], preferred: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if not rows:
        return
    fields = list(preferred)
    for row in rows:
        for k in row.keys():
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.path.join("checkpoints", "2v2"),
        help="Directory containing final_<run_tag>.zip (default: checkpoints/2v2).",
    )
    parser.add_argument(
        "--run-tags",
        nargs="+",
        default=None,
        help=f"Run tags whose final checkpoints to evaluate. Default: {len(DEFAULT_HARDPOOL_RUN_TAGS)} hard-pool tags.",
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["OP3", "OP4", "OP5_RUSHER", "OP6_TURTLE"],
        help="Scripted opponents to evaluate against (default: OP3 OP4 OP5_RUSHER OP6_TURTLE).",
    )
    parser.add_argument(
        "--map-set",
        type=str,
        default="eval",
        choices=("train", "eval"),
        help="Map split (default: eval — held-out maps for zero-shot).",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per (model, opponent) cell.")
    parser.add_argument("--agents", type=int, default=2, help="Agents per team (default: 2).")
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda (default: cuda).")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="CSV output dir (default: <checkpoint-dir>/eval_op4_zero_shot/).",
    )
    parser.add_argument("--deterministic", action="store_true", default=True, help="Greedy/argmax policy (default).")
    parser.add_argument("--stochastic", action="store_false", dest="deterministic", help="Sample actions instead.")
    parser.add_argument(
        "--no-coordination-metrics",
        action="store_true",
        help="Skip per-episode coord_* fields (faster, smaller CSVs).",
    )
    args = parser.parse_args()

    run_tags = list(args.run_tags) if args.run_tags else list(DEFAULT_HARDPOOL_RUN_TAGS)
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    out_dir = os.path.abspath(args.out_dir or os.path.join(checkpoint_dir, "eval_op4_zero_shot"))
    os.makedirs(out_dir, exist_ok=True)

    from eval_rollout import compute_aggregates, count_wld, run_eval_episodes  # noqa: E402
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402

    # Merge-mode: preserve rows in op4_zero_shot_comparison.csv that are not being re-evaluated this
    # invocation, so partial re-runs (e.g. only the 3 phase-aux variants) don't wipe out previous results.
    aggregate_rows: list[dict[str, Any]] = []
    comparison_path_for_merge = os.path.join(out_dir, "op4_zero_shot_comparison.csv")
    if os.path.isfile(comparison_path_for_merge):
        try:
            with open(comparison_path_for_merge, "r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rerun_tags = set(run_tags)
                preserved = [row for row in reader if str(row.get("run_tag", "")).strip() not in rerun_tags]
            if preserved:
                aggregate_rows.extend(preserved)
                print(
                    f"[eval_op4_zero_shot] preserving {len(preserved)} row(s) from existing comparison CSV "
                    f"(not in --run-tags this invocation)."
                )
        except Exception as exc:
            print(f"[eval_op4_zero_shot] could not read existing comparison CSV ({exc}); starting fresh.")

    for ri, run_tag in enumerate(run_tags):
        ckpt = _resolve_checkpoint(checkpoint_dir, run_tag)
        if ckpt is None:
            print(f"[eval_op4_zero_shot] SKIP {run_tag}: no final_{run_tag}.zip in {checkpoint_dir}")
            continue
        print(f"\n[eval_op4_zero_shot] === {run_tag} ===")
        print(f"  checkpoint: {ckpt}")
        for oi, raw_opp in enumerate(args.opponents):
            opponent = str(raw_opp).strip().upper()
            cfg = GPUFieldConfig(
                n_envs=1,
                max_blue_agents=int(args.agents),
                max_red_agents=int(args.agents),
                map_set=str(args.map_set).lower(),
                max_decision_steps=400,
                aquaticus_profile=True,
                rules_profile="OURS",
                device=str(args.device),
                seed=int(args.seed) + 1000 * ri + oi,
            )
            env = GPUCTFVecEnv(cfg)
            try:
                episodes = run_eval_episodes(
                    ckpt,
                    env,
                    int(args.episodes),
                    str(args.device),
                    opponent,
                    deterministic=bool(args.deterministic),
                    coordination_metrics=not bool(args.no_coordination_metrics),
                    progress_every=max(1, int(args.episodes) // 10) if int(args.episodes) >= 10 else 0,
                )
            finally:
                env.close()

            for idx, row in enumerate(episodes, start=1):
                row["episode_id"] = idx
                row["run_tag"] = run_tag
                row["opponent"] = opponent
                row["map_set"] = str(args.map_set).lower()
                row["checkpoint"] = ckpt

            ep_path = os.path.join(
                out_dir,
                f"eval_{run_tag}_{opponent}_{int(args.episodes)}ep.csv",
            )
            _write_rows(
                ep_path,
                episodes,
                [
                    "run_tag",
                    "opponent",
                    "map_set",
                    "episode_id",
                    "success",
                    "blue_score",
                    "red_score",
                    "steps",
                    "return",
                ],
            )
            w, l, d = count_wld(episodes)
            agg = compute_aggregates(episodes)
            n = max(1, w + l + d)
            agg_row: dict[str, Any] = {
                "run_tag": run_tag,
                "opponent": opponent,
                "map_set": str(args.map_set).lower(),
                "episodes": len(episodes),
                "wins": w,
                "losses": l,
                "draws": d,
                "win_rate": w / n,
                "loss_rate": l / n,
                "draw_rate": d / n,
                "mean_return": agg.get("mean_return", 0.0),
                "return_std": agg.get("return_std", 0.0),
                "mean_steps": agg.get("mean_steps", 0.0),
                "mean_steps_std": agg.get("mean_steps_std", 0.0),
                "success_rate_se": agg.get("success_rate_std", 0.0),
                "checkpoint": ckpt,
                "deterministic": bool(args.deterministic),
            }
            aggregate_rows.append(agg_row)
            print(
                f"  vs {opponent:<13}  W={w:>3} L={l:>3} D={d:>3}  WR={100.0 * w / n:5.1f}% "
                f"mean_return={agg.get('mean_return', 0.0):+.3f}  mean_steps={agg.get('mean_steps', 0.0):.1f}"
            )

    comparison_path = os.path.join(out_dir, "op4_zero_shot_comparison.csv")
    _write_rows(
        comparison_path,
        aggregate_rows,
        [
            "run_tag",
            "opponent",
            "map_set",
            "episodes",
            "wins",
            "losses",
            "draws",
            "win_rate",
            "loss_rate",
            "draw_rate",
            "mean_return",
            "mean_steps",
        ],
    )
    print(f"\n[eval_op4_zero_shot] comparison written: {comparison_path}")

    if aggregate_rows:
        print("\n[eval_op4_zero_shot] HEADLINE TABLE (WR % by run_tag x opponent)")
        opps = list(dict.fromkeys(r["opponent"] for r in aggregate_rows))
        run_tags_seen = list(dict.fromkeys(r["run_tag"] for r in aggregate_rows))
        header = f"  {'run_tag':<58}  " + "  ".join(f"{o:>10}" for o in opps)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for tag in run_tags_seen:
            cells: list[str] = []
            for o in opps:
                hit = next((r for r in aggregate_rows if r["run_tag"] == tag and r["opponent"] == o), None)
                if hit is None:
                    cells.append(f"{'-':>10}")
                else:
                    cells.append(f"{100.0 * float(hit['win_rate']):>9.1f}%")
            print(f"  {tag:<58}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
