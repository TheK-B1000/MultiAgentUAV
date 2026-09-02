"""Figure 2: strategic crossover forest plot, Pole A and Pole B side by side.

Consumes ONLY paper/data/fig2_crossover.csv (produced by extract_fig2_data.py) -- no numbers
live in this file. Each row is a point estimate (delta) with a 95% CI whisker; the vertical
line at Delta=0 is the actual scientific question (does the CI clear zero), which is why this
is a forest plot and not a bar chart.

Run:  python paper/figures/build_fig2_crossover.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt

from paper.figures.figure_style import COLORS, MARKERS, TWO_COLUMN, apply_style, save_figure

DATA = Path(__file__).resolve().parents[1] / "data" / "fig2_crossover.csv"

# paper narrative order, top to bottom in the plot (first row drawn at the top)
METHOD_ORDER = ["Specialists", "One-Sided Latent", "Paired Latent", "Trajectory-Guided",
                "Private-Critic", "Final Latent Baseline"]


def load_rows() -> dict:
    by_method: dict = {}
    with DATA.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_method.setdefault(row["method"], {})[row["pole"]] = {
                "delta": float(row["delta"]), "lo": float(row["ci_low"]),
                "hi": float(row["ci_high"]),
            }
    missing = [m for m in METHOD_ORDER if m not in by_method]
    if missing:
        raise SystemExit(f"REFUSING: CSV is missing methods {missing}")
    return by_method


def _panel(ax, by_method: dict, pole: str) -> None:
    n = len(METHOD_ORDER)
    ys = list(range(n, 0, -1))
    color, marker = COLORS[pole], MARKERS[pole]
    for y, method in zip(ys, METHOD_ORDER):
        d = by_method[method][pole]
        ax.plot([d["lo"], d["hi"]], [y, y], color=color, lw=1.3, solid_capstyle="round",
                zorder=2)
        ax.plot(d["delta"], y, marker=marker, color=color, ms=5, mec="white", mew=0.5,
                zorder=3)
    ax.axvline(0.0, color=COLORS["zero"], lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(METHOD_ORDER)
    ax.set_xlabel(rf"$\Delta_{pole}$ (win-rate advantage)")
    ax.set_xlim(-0.5, 0.5)
    ax.set_title(f"Pole {pole}", fontsize=9.5, fontweight="bold")


def main() -> int:
    apply_style()
    by_method = load_rows()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(TWO_COLUMN, 2.3), sharey=True)
    _panel(axA, by_method, "A")
    _panel(axB, by_method, "B")
    # sharey=True means axA and axB share one y-Axis object/formatter -- set_yticklabels([])
    # on axB would silently clobber axA's labels too (confirmed by direct repro). tick_params
    # only toggles visibility, so it hides axB's duplicate labels without touching axA's.
    axB.tick_params(labelleft=False)
    fig.subplots_adjust(wspace=0.08)

    paths = save_figure(fig, "fig2_crossover")
    print(f"Figure 2 written:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
