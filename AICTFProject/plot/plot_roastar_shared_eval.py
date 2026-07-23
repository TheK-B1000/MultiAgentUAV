#!/usr/bin/env python3
"""
Figures for frozen ROA-Star shared OP3/OP4 evaluation.

  Figure 1: Scaling curve (team size vs Match Score for OP3 and OP4)
  Figure 2: Seed-by-opponent heatmap of Match Score

Usage (from AICTFProject):

  python plot/plot_roastar_shared_eval.py \\
      --metrics-csv csv/eval_roastar_shared.csv \\
      --per-seed-csv csv/eval_roastar_shared_per_seed.csv \\
      --out-dir figures
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def _load_csv(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float:
    return float(row[key])


def plot_scaling_curve(method_rows: List[dict], out_path: str) -> None:
    import matplotlib.pyplot as plt

    by_opp: Dict[str, List[Tuple[int, float, float, float]]] = defaultdict(list)
    for row in method_rows:
        setting = row["setting"]
        agents = int(str(setting).split("v")[0])
        opp = row["opponent"]
        by_opp[opp].append(
            (
                agents,
                _f(row, "match_score_mean"),
                _f(row, "match_score_ci_lo"),
                _f(row, "match_score_ci_hi"),
            )
        )

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    markers = {"OP3": "o", "OP4": "s"}
    for opp in ("OP3", "OP4"):
        pts = sorted(by_opp.get(opp, []), key=lambda t: t[0])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        yerr = [[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]]
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            marker=markers.get(opp, "o"),
            capsize=4,
            label=f"vs {opp}",
            linewidth=2,
        )
    ax.set_xlabel("Agents per team")
    ax.set_ylabel("Match score (%)")
    ax.set_xticks([2, 3, 4])
    ax.set_xticklabels(["2v2", "3v3", "4v4"])
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("ROA-Star (PFSP): Match score vs team size")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_seed_heatmap(per_seed_rows: List[dict], out_path: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    settings = ["2v2", "3v3", "4v4"]
    seeds = sorted({int(r["seed"]) for r in per_seed_rows})
    opponents = ["OP3", "OP4"]
    row_labels = [f"{s} seed {seed}" for s in settings for seed in seeds]
    mat = np.full((len(row_labels), len(opponents)), np.nan)
    lookup = {(r["setting"], int(r["seed"]), r["opponent"]): _f(r, "match_score") for r in per_seed_rows}
    for i, (setting, seed) in enumerate((s, sd) for s in settings for sd in seeds):
        for j, opp in enumerate(opponents):
            mat[i, j] = lookup.get((setting, seed, opp), np.nan)

    fig, ax = plt.subplots(figsize=(5.5, 7.0))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=100, cmap="viridis")
    ax.set_xticks(range(len(opponents)))
    ax.set_xticklabels(opponents)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if val == val:
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", color="white", fontsize=9)
    ax.set_title("ROA-Star (PFSP): Match score by seed × opponent")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Match score (%)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--per-seed-csv", required=True)
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    method_rows = _load_csv(args.metrics_csv)
    per_seed_rows = _load_csv(args.per_seed_csv)
    os.makedirs(args.out_dir, exist_ok=True)
    plot_scaling_curve(
        method_rows,
        os.path.join(args.out_dir, "roastar_scaling_match_score.png"),
    )
    plot_seed_heatmap(
        per_seed_rows,
        os.path.join(args.out_dir, "roastar_seed_opponent_heatmap.png"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
