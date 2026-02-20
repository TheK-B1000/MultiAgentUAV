from __future__ import annotations

"""
Run full multi-baseline evaluation (Fixed, Paper, League, Self-Play) vs OP3 and OP4,
and print detailed metrics (performance, coordination, stability, specialization,
and robotics-style safety/path-quality metrics) in one command.

Usage example:

  python run_full_metrics_eval.py --episodes 100 --headless \
    --league-model checkpoints_sb3/final_ppo_league_v3.zip

Outputs:
  - Per-episode CSVs in metrics/ for each baseline/opponent:
      metrics/full_fixed_op3_100ep.csv, metrics/full_fixed_op4_100ep.csv, ...
  - Printed summaries for each baseline/opponent using analyze_eval_metrics helpers.
"""

import argparse
import os
import sys
from typing import Any, Dict, List
import contextlib

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(_SCRIPT_DIR, "metrics")
sys.path.insert(0, _SCRIPT_DIR)

# 4v4 models (default/legacy - kept for backward compatibility with old pretrained models)
BASELINE_MODEL_PATHS: Dict[str, str] = {
    "fixed_op3": "checkpoints_sb3_4v4/final_ppo_fixed_op3_4v4.zip",
    "curriculum_no_league": "checkpoints_sb3_4v4/final_ppo_paper_4v4.zip",
    "curriculum_league": "checkpoints_sb3_4v4/final_ppo_league_4v4.zip",
    "self_play": "checkpoints_sb3_4v4/final_ppo_self_play_4v4.zip",
}
# 2v2 models (separate dir/tags so we don't overwrite pretrained 4v4)
BASELINE_MODEL_PATHS_2V2: Dict[str, str] = {
    "fixed_op3": "checkpoints_sb3_2v2/final_ppo_fixed_op3_2v2.zip",
    "curriculum_no_league": "checkpoints_sb3_2v2/final_ppo_paper_2v2.zip",
    "curriculum_league": "checkpoints_sb3_2v2/final_ppo_league_curriculum_2v2.zip",
    "self_play": "checkpoints_sb3_2v2/final_ppo_self_play_2v2.zip",
}

DISPLAY_NAMES: Dict[str, str] = {
    "fixed_op3": "Fixed OP3",
    "curriculum_no_league": "Paper",
    "curriculum_league": "League",
    "self_play": "Self-Play",
}

OPPONENTS = ["OP3", "OP4"]


def _run_eval_for_baseline(
    baseline_key: str,
    model_path: str,
    num_episodes: int,
    seed_base: int,
    headless: bool,
) -> Dict[str, str]:
    """
    For a single baseline model, run eval vs OP3 and OP4 using CTFViewer,
    saving per-episode CSVs for each. Returns a dict opponent->csv_path.
    """
    from ctfviewer import CTFViewer

    out_csv: Dict[str, str] = {}

    for opp in OPPONENTS:
        # Derive a reproducible seed sequence per opponent
        if opp == "OP3":
            seeds = [seed_base + i for i in range(num_episodes)]
        else:
            seeds = [seed_base + 2000 + i for i in range(num_episodes)]

        viewer = CTFViewer(ppo_model_path=model_path, viewer_use_obs_builder=True)
        if not viewer.blue_ppo_team.model_loaded:
            print(f"[WARN] {baseline_key}: failed to load model at {model_path!r}, skipping {opp}.")
            continue

        os.makedirs(METRICS_DIR, exist_ok=True)
        csv_name = f"full_{baseline_key}_{opp}_{num_episodes}ep.csv"
        csv_path = os.path.join(METRICS_DIR, csv_name)

        print(f"[Eval] {DISPLAY_NAMES.get(baseline_key, baseline_key)} vs {opp}: "
              f"{num_episodes} episodes, saving to {csv_name}")

        # Use evaluate_model with save_csv to produce per-episode metrics
        # Silence all verbose per-run prints from CTFViewer/evaluate_model;
        # we only want our own high-level summary at the end.
        try:
            with open(os.devnull, "w", encoding="utf-8") as _devnull, contextlib.redirect_stdout(_devnull):
                viewer.evaluate_model(
                    num_episodes=num_episodes,
                    headless=headless,
                    opponent=opp,
                    eval_model="ppo",
                    episode_seeds=seeds,
                    save_csv=csv_path,
                    quiet=True,
                )
            out_csv[opp] = csv_path
        except Exception as exc:
            print(f"[WARN] {baseline_key}: eval vs {opp} failed: {exc}")
            continue

    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full baseline eval (Fixed, Paper, League, Self-Play) vs OP3 and OP4 with detailed metrics."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Episodes per opponent per baseline (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed base (default: 42)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run evaluation without display (recommended)",
    )
    parser.add_argument(
        "--league-model",
        type=str,
        default=None,
        help="Optional League checkpoint path (overrides default curriculum_league path)",
    )
    parser.add_argument(
        "--2v2",
        action="store_true",
        dest="use_2v2",
        help="Use 2v2 baseline paths (checkpoints_sb3_2v2, *_2v2.zip)",
    )
    args = parser.parse_args()

    paths = BASELINE_MODEL_PATHS_2V2 if getattr(args, "use_2v2", False) else BASELINE_MODEL_PATHS
    if args.league_model:
        paths["curriculum_league"] = args.league_model

    from analyze_eval_metrics import load_episodes, summarize_episodes
    import math

    n = max(1, int(args.episodes))
    seed = int(args.seed)

    print(f"\n=== Full Metrics Eval: {n} episodes per baseline per opponent (OP3, OP4) ===\n")

    # Keep track of per-baseline summaries (collect silently, print at end)
    summaries: Dict[str, Dict[str, Any]] = {}

    for key, rel_path in paths.items():
        name = DISPLAY_NAMES.get(key, key)
        path = os.path.join(_SCRIPT_DIR, rel_path) if not os.path.isabs(rel_path) else rel_path

        if not os.path.exists(path):
            print(f"[WARN] Model for {name} not found at {path!r}, skipping.")
            continue

        csv_paths = _run_eval_for_baseline(
            baseline_key=key,
            model_path=path,
            num_episodes=n,
            seed_base=seed,
            headless=args.headless,
        )

        # For each opponent, load and summarize (silently)
        summaries[key] = {}
        for opp in OPPONENTS:
            csv_path = csv_paths.get(opp)
            if not csv_path:
                continue
            episodes = load_episodes(csv_path)
            summary = summarize_episodes(episodes)
            summaries[key][opp] = summary

    # Print one comprehensive table at the end
    print("\n" + "=" * 120)
    print("COMPREHENSIVE METRICS TABLE - All Baselines vs OP3 and OP4")
    print("=" * 120)
    print(f"Episodes per condition: {n}  |  Seed: {seed}\n")

    def _fmt_pct(v: float) -> str:
        if math.isnan(v) or math.isinf(v):
            return "N/A"
        return f"{v:.1%}"

    def _fmt_float(v: float, decimals: int = 2) -> str:
        if math.isnan(v) or math.isinf(v):
            return "N/A"
        return f"{v:.{decimals}f}"

    # Table header
    header = (
        f"{'Baseline':<20} "
        f"{'Success OP3':>12} {'Success OP4':>12} {'Drop':>8} "
        f"{'Time OP3':>10} {'Time OP4':>10} "
        f"{'Coll/100 OP3':>13} {'Coll/100 OP4':>13} "
        f"{'Coverage OP3':>13} {'Coverage OP4':>13}"
    )
    print(header)
    print("-" * 120)

    # Table rows
    for key in BASELINE_MODEL_PATHS:
        name = DISPLAY_NAMES.get(key, key)
        s = summaries.get(key, {})
        op3_summary = s.get("OP3", {})
        op4_summary = s.get("OP4", {})

        if not op3_summary or not op4_summary:
            print(f"{name:<20} {'MISSING DATA':>108}")
            continue

        # Extract metrics
        succ_op3 = float(op3_summary.get("success_rate", 0.0))
        succ_op4 = float(op4_summary.get("success_rate", 0.0))
        drop = succ_op3 - succ_op4

        time_op3 = float(op3_summary.get("mean_time_to_game_over", math.nan))
        time_op4 = float(op4_summary.get("mean_time_to_game_over", math.nan))

        coll_op3 = float(op3_summary.get("collisions_per_100_steps_mean", math.nan))
        coll_op4 = float(op4_summary.get("collisions_per_100_steps_mean", math.nan))

        cov_op3 = float(op3_summary.get("coverage_efficiency_mean", math.nan))
        cov_op4 = float(op4_summary.get("coverage_efficiency_mean", math.nan))

        row = (
            f"{name:<20} "
            f"{_fmt_pct(succ_op3):>12} {_fmt_pct(succ_op4):>12} {_fmt_pct(drop):>8} "
            f"{_fmt_float(time_op3, 1):>10} {_fmt_float(time_op4, 1):>10} "
            f"{_fmt_float(coll_op3, 2):>13} {_fmt_float(coll_op4, 2):>13} "
            f"{_fmt_float(cov_op3, 4):>13} {_fmt_float(cov_op4, 4):>13}"
        )
        print(row)

    print("=" * 120)

    # Additional detailed metrics table
    print("\n" + "=" * 120)
    print("DETAILED METRICS - Performance, Coordination, Stability, Specialization")
    print("=" * 120 + "\n")

    # Performance metrics
    print("PERFORMANCE METRICS:")
    print(f"{'Baseline':<20} {'Success OP3':>12} {'Success OP4':>12} {'T_first OP3':>12} {'T_first OP4':>12} {'T_end OP3':>12} {'T_end OP4':>12}")
    print("-" * 120)
    for key in BASELINE_MODEL_PATHS:
        name = DISPLAY_NAMES.get(key, key)
        s = summaries.get(key, {})
        op3 = s.get("OP3", {})
        op4 = s.get("OP4", {})
        if not op3 or not op4:
            continue
        print(
            f"{name:<20} "
            f"{_fmt_pct(float(op3.get('success_rate', 0.0))):>12} "
            f"{_fmt_pct(float(op4.get('success_rate', 0.0))):>12} "
            f"{_fmt_float(float(op3.get('mean_time_to_first_score', math.nan)), 1):>12} "
            f"{_fmt_float(float(op4.get('mean_time_to_first_score', math.nan)), 1):>12} "
            f"{_fmt_float(float(op3.get('mean_time_to_game_over', math.nan)), 1):>12} "
            f"{_fmt_float(float(op4.get('mean_time_to_game_over', math.nan)), 1):>12}"
        )

    # Coordination metrics
    print("\nCOORDINATION METRICS:")
    print(f"{'Baseline':<20} {'% Attack OP3':>14} {'% Attack OP4':>14} {'% Defend OP3':>14} {'% Defend OP4':>14} {'Coverage OP3':>14} {'Coverage OP4':>14}")
    print("-" * 120)
    for key in BASELINE_MODEL_PATHS:
        name = DISPLAY_NAMES.get(key, key)
        s = summaries.get(key, {})
        op3 = s.get("OP3", {})
        op4 = s.get("OP4", {})
        if not op3 or not op4:
            continue
        print(
            f"{name:<20} "
            f"{_fmt_float(float(op3.get('mean_pct_attacking', math.nan)), 1):>14} "
            f"{_fmt_float(float(op4.get('mean_pct_attacking', math.nan)), 1):>14} "
            f"{_fmt_float(float(op3.get('mean_pct_defending', math.nan)), 1):>14} "
            f"{_fmt_float(float(op4.get('mean_pct_defending', math.nan)), 1):>14} "
            f"{_fmt_float(float(op3.get('zone_coverage_mean', math.nan)), 3):>14} "
            f"{_fmt_float(float(op4.get('zone_coverage_mean', math.nan)), 3):>14}"
        )

    # Stability metrics
    print("\nSTABILITY METRICS:")
    print(f"{'Baseline':<20} {'Reward OP3':>12} {'Reward OP4':>12} {'Reward/Step OP3':>16} {'Reward/Step OP4':>16} {'Reward Std OP3':>15} {'Reward Std OP4':>15}")
    print("-" * 120)
    for key in BASELINE_MODEL_PATHS:
        name = DISPLAY_NAMES.get(key, key)
        s = summaries.get(key, {})
        op3 = s.get("OP3", {})
        op4 = s.get("OP4", {})
        if not op3 or not op4:
            continue
        print(
            f"{name:<20} "
            f"{_fmt_float(float(op3.get('reward_mean', math.nan)), 2):>12} "
            f"{_fmt_float(float(op4.get('reward_mean', math.nan)), 2):>12} "
            f"{_fmt_float(float(op3.get('reward_per_timestep_mean', math.nan)), 6):>16} "
            f"{_fmt_float(float(op4.get('reward_per_timestep_mean', math.nan)), 6):>16} "
            f"{_fmt_float(float(op3.get('reward_std', math.nan)), 2):>15} "
            f"{_fmt_float(float(op4.get('reward_std', math.nan)), 2):>15}"
        )

    # Specialization metrics
    print("\nSPECIALIZATION METRICS:")
    print(f"{'Baseline':<20} {'Flag Var OP3':>14} {'Flag Var OP4':>14} {'Flag Cap OP3':>14} {'Flag Cap OP4':>14}")
    print("-" * 120)
    for key in BASELINE_MODEL_PATHS:
        name = DISPLAY_NAMES.get(key, key)
        s = summaries.get(key, {})
        op3 = s.get("OP3", {})
        op4 = s.get("OP4", {})
        if not op3 or not op4:
            continue
        print(
            f"{name:<20} "
            f"{_fmt_float(float(op3.get('flag_capture_variance_mean', math.nan)), 3):>14} "
            f"{_fmt_float(float(op4.get('flag_capture_variance_mean', math.nan)), 3):>14} "
            f"{_fmt_float(float(op3.get('mean_flag_captures', math.nan)), 2):>14} "
            f"{_fmt_float(float(op4.get('mean_flag_captures', math.nan)), 2):>14}"
        )

    # Robotics/Safety metrics
    print("\nROBOTICS METRICS (Safety & Path Quality):")
    print(f"{'Baseline':<20} {'Coll/Ep OP3':>13} {'Coll/Ep OP4':>13} {'Coll/100 OP3':>14} {'Coll/100 OP4':>14} {'Coll-Free OP3':>15} {'Coll-Free OP4':>15}")
    print("-" * 120)
    for key in BASELINE_MODEL_PATHS:
        name = DISPLAY_NAMES.get(key, key)
        s = summaries.get(key, {})
        op3 = s.get("OP3", {})
        op4 = s.get("OP4", {})
        if not op3 or not op4:
            continue
        print(
            f"{name:<20} "
            f"{_fmt_float(float(op3.get('collisions_per_episode_mean', math.nan)), 2):>13} "
            f"{_fmt_float(float(op4.get('collisions_per_episode_mean', math.nan)), 2):>13} "
            f"{_fmt_float(float(op3.get('collisions_per_100_steps_mean', math.nan)), 2):>14} "
            f"{_fmt_float(float(op4.get('collisions_per_100_steps_mean', math.nan)), 2):>14} "
            f"{_fmt_pct(float(op3.get('collision_free_rate', math.nan))):>15} "
            f"{_fmt_pct(float(op4.get('collision_free_rate', math.nan))):>15}"
        )

    print("\n" + "=" * 120)
    print("Evaluation complete. Per-episode CSVs saved in metrics/ directory.")
    print("=" * 120)


if __name__ == "__main__":
    main()

