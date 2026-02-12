"""
MARL baseline evaluation: all baselines vs OP3 and/or vs OP4.
Quiet run; single results table and one CSV at the end.

  python run_baseline_comparison.py --episodes 100 --headless
  python run_baseline_comparison.py --op4-only --episodes 100 --headless   # test only vs OP4

Outputs: metrics/baseline_eval.csv, and results printed at end.
"""
from __future__ import annotations

import argparse
import os
import sys
import csv
from typing import Any, Dict, List, Optional

# Optional plotting (only if matplotlib is installed)
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - plotting is optional
    plt = None  # type: ignore

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(_SCRIPT_DIR, "metrics")
sys.path.insert(0, _SCRIPT_DIR)

OP4_OPPONENT = "OP4"  # Single elite opponent for testing

BASELINE_MODEL_PATHS = {
    "fixed_op3": "checkpoints_sb3/final_ppo_fixed_op3.zip",
    "curriculum_no_league": "checkpoints_sb3/final_ppo_noleague.zip",
    "curriculum_league": "checkpoints_sb3/final_ppo_league.zip",
    "self_play": "checkpoints_sb3/final_ppo_selfplay.zip",
}

DISPLAY_NAMES = {
    "fixed_op3": "Fixed OP3",
    "curriculum_no_league": "Paper",
    "curriculum_league": "League",
    "self_play": "Self-Play",
}


def _run_eval(
    full_path: str,
    num_episodes: int,
    seeds: List[int],
    headless: bool,
    opponent: str = "OP3",
) -> Dict[str, Any]:
    from ctfviewer import CTFViewer
    viewer = CTFViewer(ppo_model_path=full_path, viewer_use_obs_builder=True)
    if not viewer.blue_ppo_team.model_loaded:
        return {"win_rate": 0.0, "wins": 0, "losses": 0, "draws": 0, "error": "Failed to load model"}
    s = viewer.evaluate_model(
        num_episodes=num_episodes,
        headless=headless,
        opponent=opponent,
        eval_model="ppo",
        episode_seeds=seeds,
        quiet=True,
    )
    return {
        "win_rate": s.get("win_rate", 0.0),
        "wins": s.get("wins", 0),
        "losses": s.get("losses", 0),
        "draws": s.get("draws", 0),
        "error": None,
    }


def _run_suite_op3(num_episodes: int, seed_base: int, headless: bool) -> Dict[str, Dict[str, Any]]:
    seeds = [seed_base + i for i in range(num_episodes)]
    out = {}
    for key, rel_path in BASELINE_MODEL_PATHS.items():
        path = os.path.join(_SCRIPT_DIR, rel_path) if not os.path.isabs(rel_path) else rel_path
        if not os.path.exists(path):
            out[key] = {"win_rate": 0.0, "wins": 0, "losses": 0, "draws": 0, "error": "Model not found"}
            continue
        try:
            out[key] = _run_eval(path, num_episodes, seeds, headless, opponent="OP3")
        except Exception as e:
            out[key] = {"win_rate": 0.0, "wins": 0, "losses": 0, "draws": 0, "error": str(e)}
    return out


def _run_suite_op4(num_episodes: int, seed_base: int, headless: bool) -> Dict[str, Dict[str, Any]]:
    """Run all baselines vs single OP4 opponent."""
    seeds = [seed_base + 2000 + j for j in range(num_episodes)]
    out = {}
    for key, rel_path in BASELINE_MODEL_PATHS.items():
        path = os.path.join(_SCRIPT_DIR, rel_path) if not os.path.isabs(rel_path) else rel_path
        if not os.path.exists(path):
            out[key] = {"win_rate": 0.0, "wins": 0, "losses": 0, "draws": 0, "error": "Model not found"}
            continue
        try:
            out[key] = _run_eval(path, num_episodes, seeds, headless, opponent=OP4_OPPONENT)
        except Exception as e:
            out[key] = {"win_rate": 0.0, "wins": 0, "losses": 0, "draws": 0, "error": str(e)}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="MARL baseline eval: OP3 + OP4. Results at end.")
    p.add_argument("--episodes", type=int, default=100, help="Episodes per condition (default: 100)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--headless", action="store_true", help="Headless (no display)")
    p.add_argument("--op4-only", action="store_true", help="Evaluate only vs OP4; skip OP3")
    p.add_argument("--out-dir", type=str, default=METRICS_DIR, help="Output directory")
    p.add_argument("--league-model", type=str, default=None, help="League checkpoint path (overrides default)")
    args = p.parse_args()

    if args.league_model:
        BASELINE_MODEL_PATHS["curriculum_league"] = args.league_model

    n = max(1, args.episodes)
    seed = args.seed
    os.makedirs(args.out_dir, exist_ok=True)

    op3 = _run_suite_op3(n, seed, args.headless) if not args.op4_only else {}
    op4_results = _run_suite_op4(n, seed, args.headless)

    # Single results table
    print("\n" + "=" * 70)
    print("MARL BASELINE EVALUATION" + (" (OP4 only)" if args.op4_only else ""))
    print("=" * 70)
    print(f"Episodes per condition: {n}  |  Seed: {seed}")
    print()
    if args.op4_only:
        print(f"{'Baseline':<20} {'vs OP4':>12}")
        print("-" * 70)
        for key in BASELINE_MODEL_PATHS:
            name = DISPLAY_NAMES.get(key, key)
            r2 = op4_results.get(key, {})
            wr2 = r2.get("win_rate", 0.0) if not r2.get("error") else float("nan")
            s2 = f"{wr2:.1%}" if wr2 == wr2 else "ERROR"
            print(f"{name:<20} {s2:>12}")
    else:
        print(f"{'Baseline':<20} {'vs OP3':>12} {'vs OP4':>12}")
        print("-" * 70)
        for key in BASELINE_MODEL_PATHS:
            name = DISPLAY_NAMES.get(key, key)
            r1 = op3.get(key, {})
            r2 = op4_results.get(key, {})
            wr1 = r1.get("win_rate", 0.0) if not r1.get("error") else float("nan")
            wr2 = r2.get("win_rate", 0.0) if not r2.get("error") else float("nan")
            s1 = f"{wr1:.1%}" if wr1 == wr1 else "ERROR"
            s2 = f"{wr2:.1%}" if wr2 == wr2 else "ERROR"
            print(f"{name:<20} {s1:>12} {s2:>12}")
    print("=" * 70)

    # One CSV
    csv_path = os.path.join(args.out_dir, "baseline_eval.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if args.op4_only:
            w.writerow(["baseline", "op4_win_rate", "op4_wins", "op4_losses", "op4_draws"])
            for key in BASELINE_MODEL_PATHS:
                r2 = op4_results.get(key, {})
                w.writerow([
                    DISPLAY_NAMES.get(key, key),
                    r2.get("win_rate", 0.0) if not r2.get("error") else "",
                    r2.get("wins", 0), r2.get("losses", 0), r2.get("draws", 0),
                ])
        else:
            w.writerow(["baseline", "op3_win_rate", "op3_wins", "op3_losses", "op3_draws", "op4_win_rate", "op4_wins", "op4_losses", "op4_draws"])
            for key in BASELINE_MODEL_PATHS:
                r1, r2 = op3.get(key, {}), op4_results.get(key, {})
                w.writerow([
                    DISPLAY_NAMES.get(key, key),
                    r1.get("win_rate", 0.0) if not r1.get("error") else "",
                    r1.get("wins", 0), r1.get("losses", 0), r1.get("draws", 0),
                    r2.get("win_rate", 0.0) if not r2.get("error") else "",
                    r2.get("wins", 0), r2.get("losses", 0), r2.get("draws", 0),
                ])
    print(f"Results saved to {csv_path}\n")

    # ------------------------------------------------------------------
    # Compact research-style summary (per-baseline OP3/OP4 performance)
    # ------------------------------------------------------------------
    def _safe_wr(d: Dict[str, Any]) -> float:
        return float(d.get("win_rate", 0.0)) if not d.get("error") else float("nan")

    print("Research-style summary (Performance & Robustness)")
    print("-" * 70)

    # Performance & Robustness (success vs OP3 / OP4 and generalization drop)
    if args.op4_only:
        for key in BASELINE_MODEL_PATHS:
            name = DISPLAY_NAMES.get(key, key)
            r2 = op4_results.get(key, {})
            wr2 = _safe_wr(r2)
            print(f"  {name}: win rate vs OP4 = {wr2:.1%}")
    else:
        for key in BASELINE_MODEL_PATHS:
            name = DISPLAY_NAMES.get(key, key)
            r1, r2 = op3.get(key, {}), op4_results.get(key, {})
            wr1, wr2 = _safe_wr(r1), _safe_wr(r2)
            if wr1 == wr1 and wr2 == wr2:
                drop = wr1 - wr2
                print(f"  {name}: OP3={wr1:.1%}, OP4={wr2:.1%}, generalization drop={drop:.1%}")
            else:
                print(f"  {name}: OP3/OP4 eval error (see CSV).")

    print("-" * 70)

    # Optional plots
    if plt is not None:
        try:
            # Bar plot of win rates
            labels = [DISPLAY_NAMES.get(k, k) for k in BASELINE_MODEL_PATHS]
            if args.op4_only:
                op4_wr = [
                    op4_results.get(k, {}).get("win_rate", 0.0)
                    if not op4_results.get(k, {}).get("error") else float("nan")
                    for k in BASELINE_MODEL_PATHS
                ]
                x = range(len(labels))
                plt.figure(figsize=(6, 4))
                plt.bar(x, op4_wr, color="tab:red")
                plt.xticks(x, labels, rotation=20)
                plt.ylabel("Win rate vs OP4")
                plt.ylim(0.0, 1.0)
                plt.title("Baseline win rate vs OP4")
                plt.tight_layout()
                png_path = os.path.join(args.out_dir, "baseline_op4_win_rate.png")
                plt.savefig(png_path)
                plt.close()
            else:
                op3_wr = [
                    op3.get(k, {}).get("win_rate", 0.0)
                    if not op3.get(k, {}).get("error") else float("nan")
                    for k in BASELINE_MODEL_PATHS
                ]
                op4_wr = [
                    op4_results.get(k, {}).get("win_rate", 0.0)
                    if not op4_results.get(k, {}).get("error") else float("nan")
                    for k in BASELINE_MODEL_PATHS
                ]
                x = range(len(labels))
                width = 0.35
                plt.figure(figsize=(7, 4))
                plt.bar([xi - width / 2 for xi in x], op3_wr, width=width, label="OP3", color="tab:blue")
                plt.bar([xi + width / 2 for xi in x], op4_wr, width=width, label="OP4", color="tab:red")
                plt.xticks(list(x), labels, rotation=20)
                plt.ylabel("Win rate")
                plt.ylim(0.0, 1.0)
                plt.title("Baseline win rate vs OP3 vs OP4")
                plt.legend()
                plt.tight_layout()
                png_path = os.path.join(args.out_dir, "baseline_comparison_win_rate.png")
                plt.savefig(png_path)
                plt.close()
        except Exception:
            # Plotting is best-effort; don't crash eval if plotting fails
            pass


if __name__ == "__main__":
    main()
