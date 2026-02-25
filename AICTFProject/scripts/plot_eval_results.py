"""
Plot evaluation success rates (models in action) for 2v2, 4v4, or 8v8.

Uses CSVs from run_full_metrics_eval.py:
  metrics/full_{baseline}_{OP3|OP4}_{n}ep_{2v2|4v4|8v8}.csv

Usage:
  cd AICTFProject
  # After running eval for each team size:
  python run_full_metrics_eval.py --agents 2 --episodes 100 --headless
  python run_full_metrics_eval.py --agents 4 --episodes 100 --headless
  # Then plot:
  python scripts/plot_eval_results.py --agents 2 --episodes 100 --outdir plots
  python scripts/plot_eval_results.py --agents 4 --episodes 100 --outdir plots
  python scripts/plot_eval_results.py --agents 8 --episodes 100 --outdir plots
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Baseline keys and display names (match run_full_metrics_eval.py)
EVAL_BASELINE_ORDER = ["fixed_op3", "curriculum_no_league", "curriculum_league", "self_play"]
EVAL_DISPLAY_NAMES = {
    "fixed_op3": "Fixed OP3",
    "curriculum_no_league": "Paper",
    "curriculum_league": "League (Ours)",
    "self_play": "Self-Play",
}


def set_latex_style(use_tex: bool) -> None:
    if use_tex:
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
    else:
        plt.rc("text", usetex=False)


def plot_eval_success_rates(
    metrics_dir: str,
    team_tag: str,
    num_episodes: int = 100,
    out_path: Optional[str] = None,
    use_tex: bool = True,
) -> str:
    """Plot eval success rate per baseline vs OP3 and OP4 for one team size."""
    set_latex_style(use_tex)
    try:
        from analyze_eval_metrics import load_episodes
    except ImportError:
        raise RuntimeError("analyze_eval_metrics not found; run from AICTFProject root.")

    opponents = ["OP3", "OP4"]
    data: Dict[str, Dict[str, float]] = {}
    for key in EVAL_BASELINE_ORDER:
        data[key] = {}
        for opp in opponents:
            csv_name = f"full_{key}_{opp}_{num_episodes}ep_{team_tag}.csv"
            csv_path = os.path.join(metrics_dir, csv_name)
            if not os.path.exists(csv_path):
                data[key][opp] = math.nan
                continue
            episodes = load_episodes(csv_path)
            if not episodes:
                data[key][opp] = math.nan
                continue
            successes = [e.success for e in episodes]
            data[key][opp] = sum(successes) / len(successes) if successes else math.nan

    labels = [EVAL_DISPLAY_NAMES.get(k, k) for k in EVAL_BASELINE_ORDER]
    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    ax.bar(
        [i - width / 2 for i in x],
        [data[k].get("OP3", math.nan) for k in EVAL_BASELINE_ORDER],
        width, label="vs OP3", color="#1f77b4",
    )
    ax.bar(
        [i + width / 2 for i in x],
        [data[k].get("OP4", math.nan) for k in EVAL_BASELINE_ORDER],
        width, label="vs OP4", color="#ff7f0e",
    )

    ax.set_ylabel("Success rate (win rate)")
    ax.set_title(f"{team_tag} evaluation: success rate vs scripted opponents")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    if out_path is None:
        out_dir = os.path.join(_PROJECT_DIR, "plots")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{team_tag}_eval_success_rates.pdf")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot eval success rates for 2v2/4v4/8v8.")
    parser.add_argument("--agents", type=int, default=4, choices=[2, 4, 8], help="Team size (2v2, 4v4, 8v8)")
    parser.add_argument("--episodes", type=int, default=100, help="Eval episode count (match run_full_metrics_eval)")
    parser.add_argument("--outdir", type=str, default="plots", help="Output directory for figures")
    parser.add_argument("--no-usetex", action="store_true", help="Disable LaTeX rendering")
    args = parser.parse_args()

    team_tag = f"{args.agents}v{args.agents}"
    metrics_dir = os.path.join(_PROJECT_DIR, "metrics")
    out_dir = os.path.join(_PROJECT_DIR, args.outdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{team_tag}_eval_success_rates.pdf")

    use_tex = not args.no_usetex
    try:
        path = plot_eval_success_rates(
            metrics_dir, team_tag, num_episodes=args.episodes,
            out_path=out_path, use_tex=use_tex,
        )
        print(f"Eval plot saved: {path}")
    except Exception as e:
        if use_tex:
            print("LaTeX failed, retrying without...")
            path = plot_eval_success_rates(
                metrics_dir, team_tag, num_episodes=args.episodes,
                out_path=out_path, use_tex=False,
            )
            print(f"Eval plot saved: {path}")
        else:
            raise


if __name__ == "__main__":
    main()
