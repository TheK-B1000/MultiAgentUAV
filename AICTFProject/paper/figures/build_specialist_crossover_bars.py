"""Specialist crossover: DARS-style grouped bars for 2v2 (PASS) and 4v4 C2 (FAIL).

Each panel groups bars by opponent pole. Under Pole A / Pole B, the two specialists
pi_A and pi_B sit side by side so crossover is visible as a color-matched taller bar
(blue taller under Pole A, vermillion taller under Pole B). That pattern holds in 2v2
and collapses in 4v4 C2.

Win rates and asymmetric 95% CI whiskers are recomputed from sealed eval row CSVs with
the project-frozen bootstrap (n_boot=20000, alpha=0.05, rng_seed=7).

Run:  python paper/figures/build_specialist_crossover_bars.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.figures.figure_style import COLORS, TWO_COLUMN, apply_style, save_figure

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"

PANELS = [
    {
        "title": "2v2 specialists (PASS)",
        "csv": SD / "specialist_baseline_eval_rows.csv",
        "n_note": "n=64 matched seeds",
    },
    {
        "title": "4v4 confirmatory C2 (FAIL)",
        "csv": SD / "confirmatory_redesign_c2_4v4_specialist_crossover_eval_rows.csv",
        "n_note": "n=128 matched seeds",
    },
]

POLICY_COLOR = {"pi_A": COLORS["A"], "pi_B": COLORS["B"]}
POLICY_LABEL = {
    "pi_A": r"$\pi_A$ (trained vs Pole A)",
    "pi_B": r"$\pi_B$ (trained vs Pole B)",
}


def _load_cells(path: Path) -> dict[tuple[str, str], np.ndarray]:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[tuple[str, str], list[int]] = {}
    for r in rows:
        out.setdefault((r["policy"], r["pole"]), []).append(int(r["win"]))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def _mean_ci_pct(wins: np.ndarray, n_boot=20000, alpha=0.05, rng_seed=7):
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, wins.size, size=(n_boot, wins.size))
    boots = wins[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    mean = float(wins.mean())
    return mean * 100, max(0.0, (mean - lo) * 100), max(0.0, (hi - mean) * 100)


def main() -> dict:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COLUMN, 3.0), sharey=True)

    group_x = {"A": 0.0, "B": 1.0}
    mode_dx = {"pi_A": -0.19, "pi_B": 0.19}
    bar_w = 0.34

    for i, (ax, panel) in enumerate(zip(axes, PANELS)):
        cells = _load_cells(panel["csv"])
        for pole in ("A", "B"):
            for policy in ("pi_A", "pi_B"):
                wins = cells[(policy, pole)]
                value, err_lo, err_hi = _mean_ci_pct(wins)
                x = group_x[pole] + mode_dx[policy]
                ax.bar(
                    x, value, yerr=[[err_lo], [err_hi]], width=bar_w,
                    color=POLICY_COLOR[policy], edgecolor="black", linewidth=0.6,
                    capsize=3, error_kw={"elinewidth": 0.8, "capthick": 0.8},
                )
                ax.text(
                    x, value + err_hi + 2.0,
                    f"{value:.0f}%",
                    ha="center", va="bottom", fontsize=7.5,
                )
        ax.set_xticks([group_x["A"], group_x["B"]])
        ax.set_xticklabels(["vs Pole A", "vs Pole B"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, 100)
        if i == 0:
            ax.set_ylabel("Win rate (%)")
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(panel["title"], fontsize=9.5, fontweight="bold")
        ax.text(0.5, -0.18, panel["n_note"], transform=ax.transAxes,
                ha="center", va="top", fontsize=7.5, style="italic")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        mpatches.Patch(facecolor=POLICY_COLOR[p], edgecolor="black", label=POLICY_LABEL[p])
        for p in ("pi_A", "pi_B")
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.05),
        ncol=2, frameon=False, fontsize=8.5,
    )
    caption = (
        "Desired specialization: $\\pi_A$ taller under Pole A and $\\pi_B$ taller under Pole B. "
        "Error bars are 95% bootstrap CIs (paired percentile, 20{,}000 resamples)."
    )
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8, style="italic")
    fig.subplots_adjust(wspace=0.12, bottom=0.22, top=0.82)

    paths = save_figure(fig, "fig_specialist_crossover_bars")
    print(paths)
    return paths


if __name__ == "__main__":
    main()
