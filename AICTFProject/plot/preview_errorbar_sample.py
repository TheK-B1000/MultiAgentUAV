#!/usr/bin/env python3
"""Preview script: render a SAMPLE win-rate bar chart with std-dev (binomial SE) error bars.

This is ONLY a preview using hardcoded win-rate numbers taken from the existing
paper figures (2v2/5v5/8v8 vs OP3, N=100 episodes). It does NOT load any models
or re-run evaluation. Use it to approve the plot style, then we will wire the
same style into plot_*_winrate.py to replace the real paper plots.

Error bars: binomial standard error  SE = sqrt(p*(1-p)/N) * 100%
(same quantity reported as success_rate_std/sqrt(N) in plot/eval_rollout.py).

Run:
  python plot/preview_errorbar_sample.py
Outputs:
  figures/SAMPLE_winrate_errorbars_2v2.png
  figures/SAMPLE_winrate_errorbars_5v5.png
  figures/SAMPLE_winrate_errorbars_8v8.png
  figures/SAMPLE_winrate_errorbars_all.png   (combined 2v2|5v5|8v8 panels)
"""
from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LABELS = ["Ours", "Jacob et al.", "Self-play"]
COLORS = ["#2ecc71", "#3498db", "#9b59b6"]

SAMPLES = {
    "2v2": {"win_rates": [93.0, 91.0, 86.0], "n": 100, "opp": "OP3"},
    "5v5": {"win_rates": [95.0, 96.0, 91.0], "n": 100, "opp": "OP3"},
    "8v8": {"win_rates": [0.0, 93.0, 0.0], "n": 100, "opp": "OP3"},
}


def binomial_se(p_pct: float, n: int) -> float:
    p = max(0.0, min(1.0, p_pct / 100.0))
    if n <= 0:
        return 0.0
    return 100.0 * math.sqrt(p * (1.0 - p) / n)


def plot_single(ax, title: str, win_rates, ns, opp: str) -> None:
    x = np.arange(len(LABELS))
    errs = [binomial_se(wr, ns) for wr in win_rates]
    bars = ax.bar(
        x,
        win_rates,
        color=COLORS,
        edgecolor="black",
        linewidth=1.2,
        yerr=errs,
        capsize=6,
        error_kw={"elinewidth": 1.6, "ecolor": "black"},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylabel(f"Win rate vs {opp} (%)", fontsize=17)
    ax.set_title(f"{title} Win rate  (n={ns})", fontsize=18)
    ax.set_ylim(0, 115)
    for bar, wr, se in zip(bars, win_rates, errs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + se + 2.5,
            f"{wr:.1f}% ± {se:.1f}",
            ha="center",
            fontsize=14,
        )


def main() -> None:
    plt.rc("font", size=14)
    # Per-team-size figures (drop-in replacements for 2v2/5v5/8v8 paper figures).
    for tag, d in SAMPLES.items():
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        plot_single(ax, tag, d["win_rates"], d["n"], d["opp"])
        plt.tight_layout()
        out = os.path.join(FIG_DIR, f"SAMPLE_winrate_errorbars_{tag}.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")

    # Combined 3-panel preview.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, tag in zip(axes, ("2v2", "5v5", "8v8")):
        d = SAMPLES[tag]
        plot_single(ax, tag, d["win_rates"], d["n"], d["opp"])
    plt.suptitle(
        "Win rate with ±1 binomial SE  (SE = √(p(1-p)/N) · 100%)",
        fontsize=18,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(FIG_DIR, "SAMPLE_winrate_errorbars_all.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
