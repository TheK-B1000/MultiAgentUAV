"""
Run 3 evaluation suites for all baselines; separate outputs so nothing is overwritten.

Suites:
  1) vs OP3 only — same seeds, same opponent. League may tie if OP3 is solved.
  2) vs species set — BALANCED, RUSHER, CAMPER. League should shine.
  3) vs snapshot pool — league snapshots (ppo_league_league_snapshot_ep*.zip) round-robin. League should shine.

Expected: League >= No-League on (2) and (3).

Baselines: fixed_op3, curriculum_no_league, curriculum_league, self_play.

Outputs (in metrics/ by default; distinct filenames per suite):
  - baseline_comparison_OP3_* / baseline_comparison_species_* / baseline_comparison_snapshots_*
  - *_results.csv, *_summary.txt, *_win_rate.png, *_time_to_first_score.png, *_collision_free_rate.png
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import csv
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(_SCRIPT_DIR, "metrics")
CHECKPOINTS_DIR = os.path.join(_SCRIPT_DIR, "checkpoints_sb3")
sys.path.insert(0, _SCRIPT_DIR)

# Species set for suite 2 (League should shine vs diverse playstyles)
SPECIES_TAGS = ["BALANCED", "RUSHER", "CAMPER"]

# League snapshot glob for suite 3 (round-robin)
LEAGUE_SNAPSHOT_GLOB = "ppo_league_league_snapshot_ep*.zip"

# Baseline model paths (must match ctfviewer.BASELINE_MODEL_PATHS for consistency)
BASELINE_MODEL_PATHS = {
    "fixed_op3": "checkpoints_sb3/final_ppo_fixed_op3.zip",
    "curriculum_no_league": "checkpoints_sb3/final_ppo_noleague.zip",
    "curriculum_league": "checkpoints_sb3/final_ppo_league.zip",
    "self_play": "checkpoints_sb3/final_ppo_selfplay.zip",
}

DISPLAY_NAMES = {
    "fixed_op3": "Fixed OP3",
    "curriculum_no_league": "Curriculum No-League",
    "curriculum_league": "Curriculum League",
    "self_play": "Self-Play",
}


def _aggregate_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate multiple evaluate_model summaries into one (sum wins/losses/draws, average rates)."""
    if not summaries:
        return {
            "win_rate": 0.0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "mean_time_to_first_score": None,
            "collision_free_rate": 0.0,
            "mean_reward_per_timestep": None,
            "mean_collisions_per_100_steps": None,
            "error": None,
        }
    wins = sum(s.get("wins", 0) for s in summaries)
    losses = sum(s.get("losses", 0) for s in summaries)
    draws = sum(s.get("draws", 0) for s in summaries)
    total = wins + losses + draws
    tfs_list = [s.get("mean_time_to_first_score") for s in summaries if s.get("mean_time_to_first_score") is not None]
    cfr_list = [s.get("collision_free_rate", 0.0) for s in summaries]
    return {
        "win_rate": wins / total if total else 0.0,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "mean_time_to_first_score": sum(tfs_list) / len(tfs_list) if tfs_list else None,
        "collision_free_rate": sum(cfr_list) / len(cfr_list) if cfr_list else 0.0,
        "mean_reward_per_timestep": None,
        "mean_collisions_per_100_steps": None,
        "error": None,
    }


def _run_one_baseline_eval(
    full_path: str,
    num_episodes: int,
    episode_seeds: List[int],
    headless: bool,
    *,
    opponent: str = "OP3",
    red_species_tag: Optional[str] = None,
    red_model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Single evaluation run; optional red_species_tag or red_model_path (snapshot)."""
    from ctfviewer import CTFViewer

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
            "error": "Failed to load model",
        }
    summary = viewer.evaluate_model(
        num_episodes=num_episodes,
        headless=headless,
        opponent=opponent,
        eval_model="ppo",
        episode_seeds=episode_seeds,
        red_species_tag=red_species_tag,
        red_model_path=red_model_path,
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


def run_one_baseline(
    baseline_key: str,
    model_path: str,
    num_episodes: int,
    opponent: str,
    episode_seeds: List[int],
    headless: bool,
    *,
    red_species_tag: Optional[str] = None,
    red_model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run evaluation for one baseline; return summary dict. Optional: red_species_tag or red_model_path for suite 2/3."""
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
        return _run_one_baseline_eval(
            full_path,
            num_episodes,
            episode_seeds,
            headless,
            opponent=opponent,
            red_species_tag=red_species_tag,
            red_model_path=red_model_path,
        )
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


def run_suite_op3(
    num_episodes: int,
    episode_seeds: List[int],
    headless: bool,
) -> Dict[str, Dict[str, Any]]:
    """Suite 1: vs OP3 only (current behavior). League may tie if OP3 is solved."""
    results: Dict[str, Dict[str, Any]] = {}
    for baseline_key, model_path in BASELINE_MODEL_PATHS.items():
        print(f"  [OP3] {DISPLAY_NAMES.get(baseline_key, baseline_key)}...")
        results[baseline_key] = run_one_baseline(
            baseline_key=baseline_key,
            model_path=model_path,
            num_episodes=num_episodes,
            opponent="OP3",
            episode_seeds=episode_seeds,
            headless=headless,
        )
    return results


def run_suite_species(
    num_episodes: int,
    seed_base: int,
    headless: bool,
) -> Dict[str, Dict[str, Any]]:
    """Suite 2: vs species set (BALANCED, RUSHER, CAMPER). League should shine."""
    n = len(SPECIES_TAGS)
    per_species = max(1, num_episodes // n)
    results: Dict[str, Dict[str, Any]] = {}
    for baseline_key, model_path in BASELINE_MODEL_PATHS.items():
        print(f"  [species] {DISPLAY_NAMES.get(baseline_key, baseline_key)}...")
        full_path = os.path.join(_SCRIPT_DIR, model_path) if not os.path.isabs(model_path) else model_path
        if not os.path.exists(full_path):
            results[baseline_key] = {
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
            continue
        summaries: List[Dict[str, Any]] = []
        for i, tag in enumerate(SPECIES_TAGS):
            ep_count = per_species if i < n - 1 else (num_episodes - per_species * (n - 1))
            seeds = [seed_base + 1000 * i + j for j in range(ep_count)]
            try:
                s = _run_one_baseline_eval(
                    full_path,
                    ep_count,
                    seeds,
                    headless,
                    opponent="OP3",
                    red_species_tag=tag,
                )
                summaries.append(s)
            except Exception as e:
                summaries.append({
                    "wins": 0, "losses": 0, "draws": 0,
                    "mean_time_to_first_score": None,
                    "collision_free_rate": 0.0,
                })
        results[baseline_key] = _aggregate_summaries(summaries)
    return results


def discover_league_snapshots() -> List[str]:
    """Return full paths to league snapshot .zip files (sorted)."""
    pattern = os.path.join(CHECKPOINTS_DIR, LEAGUE_SNAPSHOT_GLOB)
    paths = glob.glob(pattern)
    paths = [p for p in paths if p.endswith(".zip") and os.path.isfile(p)]
    paths.sort()
    return paths


def run_suite_snapshots(
    num_episodes: int,
    seed_base: int,
    headless: bool,
) -> Dict[str, Dict[str, Any]]:
    """Suite 3: vs snapshot pool (league snapshots round-robin). League should shine."""
    snapshots = discover_league_snapshots()
    if not snapshots:
        print("[WARN] No league snapshots found; suite 3 will report errors.")
        return {
            k: {
                "win_rate": 0.0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "mean_time_to_first_score": None,
                "collision_free_rate": 0.0,
                "mean_reward_per_timestep": None,
                "mean_collisions_per_100_steps": None,
                "error": "No league snapshots (ppo_league_league_snapshot_ep*.zip) found",
            }
            for k in BASELINE_MODEL_PATHS
        }
    n = len(snapshots)
    per_snapshot = max(1, num_episodes // n)
    results: Dict[str, Dict[str, Any]] = {}
    for baseline_key, model_path in BASELINE_MODEL_PATHS.items():
        print(f"  [snapshots] {DISPLAY_NAMES.get(baseline_key, baseline_key)}...")
        full_path = os.path.join(_SCRIPT_DIR, model_path) if not os.path.isabs(model_path) else model_path
        if not os.path.exists(full_path):
            results[baseline_key] = {
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
            continue
        summaries = []
        for i, snap_path in enumerate(snapshots):
            ep_count = per_snapshot if i < n - 1 else (num_episodes - per_snapshot * (n - 1))
            seeds = [seed_base + 2000 + i * 500 + j for j in range(ep_count)]
            try:
                s = _run_one_baseline_eval(
                    full_path,
                    ep_count,
                    seeds,
                    headless,
                    opponent="OP3",
                    red_model_path=snap_path,
                )
                summaries.append(s)
            except Exception as e:
                summaries.append({
                    "wins": 0, "losses": 0, "draws": 0,
                    "mean_time_to_first_score": None,
                    "collision_free_rate": 0.0,
                })
        results[baseline_key] = _aggregate_summaries(summaries)
    return results


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


def save_summary_txt(
    results: Dict[str, Dict[str, Any]],
    num_episodes: int,
    seed_base: int,
    out_path: str,
    suite_label: str = "OP3",
) -> None:
    """Write human-readable summary."""
    lines = [
        f"Baseline comparison — Suite: {suite_label}",
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


def plot_comparison(results: Dict[str, Dict[str, Any]], save_dir: str, suffix: str = "") -> None:
    """Generate comparison bar charts. suffix is used in filenames (e.g. _OP3, _species, _snapshots)."""
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
    prefix = f"baseline_comparison{suffix}_"

    # 1) Win rate
    win_rates = [results[k]["win_rate"] for k in baseline_keys]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(baseline_keys))
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"][: len(baseline_keys)]
    bars = ax.bar(x, win_rates, color=colors, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Win rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Baseline comparison{suffix}: Win rate")
    for i, v in enumerate(win_rates):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=11)
    plt.tight_layout()
    p = os.path.join(save_dir, f"{prefix}win_rate.png")
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
        bars = ax.bar(x, tfs_vals, color=colors, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean time to first score (s)")
        ax.set_title(f"Baseline comparison{suffix}: Time to first score")
        for i, v in enumerate(tfs_vals):
            if v > 0:
                ax.text(i, v + 1, f"{v:.1f}s", ha="center", fontsize=10)
        plt.tight_layout()
        p = os.path.join(save_dir, f"{prefix}time_to_first_score.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"[Saved] {p}")

    # 3) Collision-free rate
    cfr_vals = [results[k].get("collision_free_rate", 0.0) for k in baseline_keys]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(x, cfr_vals, color=colors, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Collision-free rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Baseline comparison{suffix}: Collision-free rate")
    for i, v in enumerate(cfr_vals):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=11)
    plt.tight_layout()
    p = os.path.join(save_dir, f"{prefix}collision_free_rate.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"[Saved] {p}")


def _print_suite_results(results: Dict[str, Dict[str, Any]], suite_name: str) -> None:
    for baseline_key, model_path in BASELINE_MODEL_PATHS.items():
        r = results.get(baseline_key, {})
        if r.get("error"):
            print(f"  {DISPLAY_NAMES.get(baseline_key, baseline_key)}: ERROR - {r['error']}")
        else:
            print(f"  {DISPLAY_NAMES.get(baseline_key, baseline_key)}: WR={r['win_rate']:.2%} ({r['wins']}W/{r['losses']}L/{r['draws']}D)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 3 evaluation suites (vs OP3, vs species, vs snapshot pool) for all baselines; separate outputs so League can be compared on (2) and (3)."
    )
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes per baseline per suite (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed (default: 42)")
    parser.add_argument("--headless", action="store_true", help="Run headless (no display)")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--out-dir", type=str, default=METRICS_DIR, help="Output directory (default: metrics/)")
    args = parser.parse_args()

    num_episodes = max(1, args.episodes)
    seed_base = args.seed
    os.makedirs(args.out_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Suite 1: vs OP3 only (League may tie if OP3 is solved)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("SUITE 1: vs OP3 only")
    print("=" * 60)
    episode_seeds = [seed_base + i for i in range(num_episodes)]
    results_op3 = run_suite_op3(num_episodes, episode_seeds, args.headless)
    _print_suite_results(results_op3, "OP3")

    save_results_csv(results_op3, os.path.join(args.out_dir, "baseline_comparison_OP3_results.csv"))
    save_summary_txt(results_op3, num_episodes, seed_base, os.path.join(args.out_dir, "baseline_comparison_OP3_summary.txt"), suite_label="vs OP3 only")
    if not args.no_plots:
        plot_comparison(results_op3, args.out_dir, suffix="_OP3")

    # -------------------------------------------------------------------------
    # Suite 2: vs species set (BALANCED, RUSHER, CAMPER) — League should shine
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("SUITE 2: vs species set (BALANCED, RUSHER, CAMPER)")
    print("=" * 60)
    results_species = run_suite_species(num_episodes, seed_base, args.headless)
    _print_suite_results(results_species, "species")

    save_results_csv(results_species, os.path.join(args.out_dir, "baseline_comparison_species_results.csv"))
    save_summary_txt(results_species, num_episodes, seed_base, os.path.join(args.out_dir, "baseline_comparison_species_summary.txt"), suite_label="vs species set")
    if not args.no_plots:
        plot_comparison(results_species, args.out_dir, suffix="_species")

    # -------------------------------------------------------------------------
    # Suite 3: vs snapshot pool (league snapshots round-robin) — League should shine
    # -------------------------------------------------------------------------
    snapshots = discover_league_snapshots()
    print("=" * 60)
    print(f"SUITE 3: vs snapshot pool (round-robin, {len(snapshots)} snapshots)")
    print("=" * 60)
    results_snapshots = run_suite_snapshots(num_episodes, seed_base, args.headless)
    _print_suite_results(results_snapshots, "snapshots")

    save_results_csv(results_snapshots, os.path.join(args.out_dir, "baseline_comparison_snapshots_results.csv"))
    save_summary_txt(results_snapshots, num_episodes, seed_base, os.path.join(args.out_dir, "baseline_comparison_snapshots_summary.txt"), suite_label="vs snapshot pool")
    if not args.no_plots:
        plot_comparison(results_snapshots, args.out_dir, suffix="_snapshots")

    print("=" * 60)
    print("DONE. Expected: League >= No-League on suite 2 and 3.")
    print("Outputs (no overwrite):")
    print(f"  {args.out_dir}/baseline_comparison_OP3_*")
    print(f"  {args.out_dir}/baseline_comparison_species_*")
    print(f"  {args.out_dir}/baseline_comparison_snapshots_*")
    print("=" * 60)


if __name__ == "__main__":
    main()
