"""
Run all baselines (except self_play) under the same evaluation setting and produce comparison plots.

Same setting for all:
  - Same opponent (OP3)
  - Same episode count
  - Same episode seeds (so each baseline sees the exact same N scenarios)
  - Headless eval, deterministic actions

Baselines included:
  - fixed_op3 (FIXED_OPPONENT OP3)
  - curriculum_no_league (CURRICULUM_NO_LEAGUE)
  - curriculum_league (CURRICULUM_LEAGUE)

Excluded: self_play (still training).

Outputs (in metrics/ by default):
  - baseline_comparison_results.csv: summary table
  - baseline_comparison_win_rate.png
  - baseline_comparison_time_to_first_score.png
  - baseline_comparison_collision_free_rate.png
  - baseline_comparison_summary.txt: human-readable summary
"""
from __future__ import annotations

import argparse
import os
import sys
import csv
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(_SCRIPT_DIR, "metrics")
sys.path.insert(0, _SCRIPT_DIR)

# Baseline model paths (must match ctfviewer.BASELINE_MODEL_PATHS for non–self_play)
BASELINE_MODEL_PATHS = {
    "fixed_op3": "checkpoints_sb3/final_ppo_fixed_op3.zip",
    "curriculum_no_league": "checkpoints_sb3/final_ppo_noleague.zip",
    "curriculum_league": "rl/checkpoints_sb3/final_ppo_league_curriculum_v2.zip",
}

DISPLAY_NAMES = {
    "fixed_op3": "Fixed OP3",
    "curriculum_no_league": "Curriculum No-League",
    "curriculum_league": "Curriculum League",
}


def run_one_baseline(
    baseline_key: str,
    model_path: str,
    num_episodes: int,
    opponent: str,
    episode_seeds: List[int],
    headless: bool,
) -> Dict[str, Any]:
    """Run evaluation for one baseline; return summary dict (win_rate, wins, losses, draws, mean_time_to_first_score, collision_free_rate, etc.)."""
    from ctfviewer import CTFViewer

    # Resolve path relative to script dir
    full_path = os.path.join(_SCRIPT_DIR, model_path) if not os.path.isabs(model_path) else model_path
    if not os.path.exists(full_path):
        return {
            "win_rate": 0.0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "mean_time_to_first_score": None,
            "collision_free_rate": 0.0,
            "mean_reward_per_timestep": None,
            "mean_collisions_per_100_steps": None,
            "error": f"Model not found: {full_path}",
        }

    try:
        viewer = CTFViewer(ppo_model_path=full_path, viewer_use_obs_builder=True)
        if not viewer.blue_ppo_team.model_loaded:
            return {
                "win_rate": 0.0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "mean_time_to_first_score": None,
                "collision_free_rate": 0.0,
                "mean_reward_per_timestep": None,
                "mean_collisions_per_100_steps": None,
                "error": f"Failed to load model: {full_path}",
            }
        summary = viewer.evaluate_model(
            num_episodes=num_episodes,
            headless=headless,
            opponent=opponent,
            eval_model="ppo",
            episode_seeds=episode_seeds,
        )
        return {
            "win_rate": summary.get("win_rate", 0.0),
            "wins": summary.get("wins", 0),
            "losses": summary.get("losses", 0),
            "draws": summary.get("draws", 0),
            "mean_time_to_first_score": summary.get("mean_time_to_first_score"),
            "collision_free_rate": summary.get("collision_free_rate", 0.0),
            "mean_reward_per_timestep": summary.get("mean_reward_per_timestep"),
            "mean_collisions_per_100_steps": summary.get("mean_collisions_per_100_steps"),
            "error": None,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "win_rate": 0.0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "mean_time_to_first_score": None,
            "collision_free_rate": 0.0,
            "mean_reward_per_timestep": None,
            "mean_collisions_per_100_steps": None,
            "error": str(e),
        }


def save_results_csv(results: Dict[str, Dict[str, Any]], out_path: str) -> None:
    """Write baseline comparison results to CSV."""
    rows = []
    for baseline_key, data in results.items():
        row = {
            "baseline": baseline_key,
            "display_name": DISPLAY_NAMES.get(baseline_key, baseline_key),
            "win_rate": data.get("win_rate", 0.0),
            "wins": data.get("wins", 0),
            "losses": data.get("losses", 0),
            "draws": data.get("draws", 0),
            "mean_time_to_first_score": data.get("mean_time_to_first_score"),
            "collision_free_rate": data.get("collision_free_rate", 0.0),
            "mean_reward_per_timestep": data.get("mean_reward_per_timestep"),
            "mean_collisions_per_100_steps": data.get("mean_collisions_per_100_steps"),
            "error": data.get("error") or "",
        }
        rows.append(row)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[Saved] {out_path}")


def save_summary_txt(results: Dict[str, Dict[str, Any]], num_episodes: int, seed_base: int, out_path: str) -> None:
    """Write human-readable summary."""
    lines = [
        "Baseline comparison (same setting: same seeds, same opponent OP3)",
        f"Episodes: {num_episodes}  Seed base: {seed_base}",
        "",
    ]
    for baseline_key in BASELINE_MODEL_PATHS:
        name = DISPLAY_NAMES.get(baseline_key, baseline_key)
        data = results.get(baseline_key, {})
        if data.get("error"):
            lines.append(f"{name}: ERROR - {data['error']}")
        else:
            wr = data.get("win_rate", 0.0)
            w, l, d = data.get("wins", 0), data.get("losses", 0), data.get("draws", 0)
            tfs = data.get("mean_time_to_first_score")
            cfr = data.get("collision_free_rate", 0.0)
            tfs_str = f"{tfs:.2f}s" if tfs is not None else "N/A"
            lines.append(f"{name}: WR={wr:.2%} ({w}W/{l}L/{d}D)  TtFS={tfs_str}  CollisionFree={cfr:.2%}")
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Saved] {out_path}")


def plot_comparison(results: Dict[str, Dict[str, Any]], save_dir: str) -> None:
    """Generate comparison bar charts. Requires matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skipping plots. Install with: pip install matplotlib")
        return

    baseline_keys = [k for k in BASELINE_MODEL_PATHS if k in results and not results[k].get("error")]
    if not baseline_keys:
        print("[WARN] No valid results to plot.")
        return
    labels = [DISPLAY_NAMES.get(k, k) for k in baseline_keys]

    # 1) Win rate
    win_rates = [results[k]["win_rate"] for k in baseline_keys]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(baseline_keys))
    bars = ax.bar(x, win_rates, color=["#2ecc71", "#3498db", "#9b59b6"][: len(baseline_keys)], edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Win rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Baseline comparison: Win rate vs OP3 (same seeds)")
    for i, v in enumerate(win_rates):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=11)
    plt.tight_layout()
    p = os.path.join(save_dir, "baseline_comparison_win_rate.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"[Saved] {p}")

    # 2) Mean time to first score
    tfs_vals = []
    for k in baseline_keys:
        t = results[k].get("mean_time_to_first_score")
        tfs_vals.append(t if t is not None else 0.0)
    if any(t is not None for t in [results[k].get("mean_time_to_first_score") for k in baseline_keys]):
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(x, tfs_vals, color=["#2ecc71", "#3498db", "#9b59b6"][: len(baseline_keys)], edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean time to first score (s)")
        ax.set_title("Baseline comparison: Time to first score (same seeds)")
        for i, v in enumerate(tfs_vals):
            if v > 0:
                ax.text(i, v + 1, f"{v:.1f}s", ha="center", fontsize=10)
        plt.tight_layout()
        p = os.path.join(save_dir, "baseline_comparison_time_to_first_score.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"[Saved] {p}")

    # 3) Collision-free rate
    cfr_vals = [results[k].get("collision_free_rate", 0.0) for k in baseline_keys]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(x, cfr_vals, color=["#2ecc71", "#3498db", "#9b59b6"][: len(baseline_keys)], edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Collision-free rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Baseline comparison: Collision-free rate (same seeds)")
    for i, v in enumerate(cfr_vals):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=11)
    plt.tight_layout()
    p = os.path.join(save_dir, "baseline_comparison_collision_free_rate.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"[Saved] {p}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all baselines (except self_play) with same setting and produce comparison results/plots."
    )
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes per baseline (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for episode seeds (default: 42)")
    parser.add_argument("--headless", action="store_true", help="Run headless (no display)")
    parser.add_argument("--opponent", type=str, default="OP3", help="Red opponent (default: OP3)")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--out-dir", type=str, default=METRICS_DIR, help="Directory for CSV, summary .txt, and plots (default: metrics/)")
    args = parser.parse_args()

    num_episodes = max(1, args.episodes)
    seed_base = args.seed
    episode_seeds = [seed_base + i for i in range(num_episodes)]

    print("=" * 60)
    print("BASELINE COMPARISON (same setting: same seeds, same opponent)")
    print("=" * 60)
    print(f"Episodes: {num_episodes}  Seed base: {seed_base}  Opponent: {args.opponent}")
    print("Baselines: fixed_op3, curriculum_no_league, curriculum_league (self_play excluded)")
    print()

    results: Dict[str, Dict[str, Any]] = {}
    for baseline_key, model_path in BASELINE_MODEL_PATHS.items():
        print("-" * 60)
        print(f"Running: {DISPLAY_NAMES.get(baseline_key, baseline_key)} ({baseline_key})")
        print(f"Model: {model_path}")
        print("-" * 60)
        results[baseline_key] = run_one_baseline(
            baseline_key=baseline_key,
            model_path=model_path,
            num_episodes=num_episodes,
            opponent=args.opponent,
            episode_seeds=episode_seeds,
            headless=args.headless,
        )
        if results[baseline_key].get("error"):
            print(f"  ERROR: {results[baseline_key]['error']}")
        else:
            r = results[baseline_key]
            print(f"  Win rate: {r['win_rate']:.2%} ({r['wins']}W / {r['losses']}L / {r['draws']}D)")
            if r.get("mean_time_to_first_score") is not None:
                print(f"  Mean time to first score: {r['mean_time_to_first_score']:.2f}s")
            print(f"  Collision-free rate: {r.get('collision_free_rate', 0):.2%}")
        print()

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "baseline_comparison_results.csv")
    save_results_csv(results, csv_path)
    txt_path = os.path.join(args.out_dir, "baseline_comparison_summary.txt")
    save_summary_txt(results, num_episodes, seed_base, txt_path)
    if not args.no_plots:
        plot_comparison(results, args.out_dir)

    print()
    print("=" * 60)
    print("DONE. Review:")
    print(f"  {csv_path}")
    print(f"  {txt_path}")
    if not args.no_plots:
        print(f"  {os.path.join(args.out_dir, 'baseline_comparison_win_rate.png')}")
        print(f"  {os.path.join(args.out_dir, 'baseline_comparison_time_to_first_score.png')}")
        print(f"  {os.path.join(args.out_dir, 'baseline_comparison_collision_free_rate.png')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
