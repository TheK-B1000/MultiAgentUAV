"""2v2 training strategy emergence only -- no 4v4 panel.

Own-pole rolling win rate over timesteps for contested 2v2 specialists
(stay near ~0.5--0.6). Companion to the sealed 2v2 crossover claim.

Run:  python paper/figures/build_training_emergence_2v2.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.figures.figure_style import COLORS, LINESTYLES, MARKERS, ONE_COLUMN, apply_style, save_figure

SERIES = [
    {
        "label": r"$\pi_A$ vs Pole A",
        "path": ROOT / "artifacts/strategic_demand/r1_training/r1_pi_A_specialist_seed7100001/metrics.csv",
        "color": COLORS["A"],
        "ls": LINESTYLES["A"],
        "marker": MARKERS["A"],
    },
    {
        "label": r"$\pi_B$ vs Pole B",
        "path": ROOT / "artifacts/strategic_demand/r1_training/r1_pi_B_specialist_seed7200001/metrics.csv",
        "color": COLORS["B"],
        "ls": LINESTYLES["B"],
        "marker": MARKERS["B"],
    },
]


def _load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ts, wr = [], []
    for r in rows:
        t = r.get("timesteps", "")
        y = r.get("rolling_win_rate_200ep", "")
        if t == "" or y == "":
            continue
        ts.append(float(t))
        wr.append(float(y) * 100.0)
    return np.asarray(ts), np.asarray(wr)


def _downsample(x: np.ndarray, y: np.ndarray, every: int = 8):
    idx = np.arange(0, len(x), every)
    if len(x) and idx[-1] != len(x) - 1:
        idx = np.append(idx, len(x) - 1)
    return x[idx], y[idx]


def main() -> dict:
    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(ONE_COLUMN + 0.8, 2.8))
    ax.set_title("2v2 specialists (contested)", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Own-pole rolling win rate (%)")
    ax.set_ylim(0, 100)
    ax.axhspan(35, 75, color="#EEEEEE", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for s in SERIES:
        x, y = _load_curve(s["path"])
        ax.plot(x, y, color=s["color"], ls=s["ls"], lw=1.2, label=s["label"], zorder=2)
        xm, ym = _downsample(x, y, every=max(1, len(x) // 12))
        ax.plot(
            xm, ym, ls="none", marker=s["marker"], color=s["color"],
            ms=4.5, mec="white", mew=0.4, zorder=3,
        )

    ax.legend(loc="lower right", frameon=False, fontsize=8)
    caption = (
        "Shaded band: contested ~35--75% specialization-pressure window. "
        "Source: training metrics.csv (rolling_win_rate_200ep)."
    )
    fig.text(0.5, -0.06, caption, ha="center", va="top", fontsize=8, style="italic")
    fig.subplots_adjust(bottom=0.22, top=0.88)

    paths = save_figure(fig, "fig_training_emergence_2v2")
    print(paths)
    return paths


if __name__ == "__main__":
    main()
