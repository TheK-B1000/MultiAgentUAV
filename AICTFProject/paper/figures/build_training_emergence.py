"""Training strategy emergence: own-pole rolling win rate over timesteps.

Compares the contested 2v2 specialist training curves (stay near ~0.5--0.6) against
the 4v4 confirmatory C2 curves where pi_B2 saturates near ~0.9 very early. This is
the visual companion to 4V4_SPECIALIZATION_PRESSURE_DIAGNOSTIC.json: specialization
pressure is visible from training telemetry long before the sealed crossover eval.

y = rolling_win_rate_200ep against the specialist's own training pole (in-training
telemetry, not crossover). x = timesteps.

Run:  python paper/figures/build_training_emergence.py
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

from paper.figures.figure_style import COLORS, LINESTYLES, MARKERS, TWO_COLUMN, apply_style, save_figure

SERIES = [
    {
        "label": r"2v2 $\pi_A$ vs Pole A",
        "path": ROOT / "artifacts/strategic_demand/r1_training/r1_pi_A_specialist_seed7100001/metrics.csv",
        "color": COLORS["A"],
        "ls": LINESTYLES["A"],
        "marker": MARKERS["A"],
        "panel": 0,
    },
    {
        "label": r"2v2 $\pi_B$ vs Pole B",
        "path": ROOT / "artifacts/strategic_demand/r1_training/r1_pi_B_specialist_seed7200001/metrics.csv",
        "color": COLORS["B"],
        "ls": LINESTYLES["B"],
        "marker": MARKERS["B"],
        "panel": 0,
    },
    {
        "label": r"4v4 $\pi_{A2}$ vs Pole A",
        "path": ROOT / "artifacts/scale_4v4_specialists/pi_A_specialist_4v4/metrics.csv",
        "color": COLORS["A"],
        "ls": LINESTYLES["A"],
        "marker": MARKERS["A"],
        "panel": 1,
    },
    {
        "label": r"4v4 $\pi_{B2}$ vs Pole B2",
        "path": ROOT / "artifacts/scale_4v4_specialists/pi_B_specialist_4v4/metrics.csv",
        "color": COLORS["B"],
        "ls": LINESTYLES["B"],
        "marker": MARKERS["B"],
        "panel": 1,
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
    """Keep early density readable; mark every Nth point for grayscale encoding."""
    idx = np.arange(0, len(x), every)
    if len(x) and idx[-1] != len(x) - 1:
        idx = np.append(idx, len(x) - 1)
    return x[idx], y[idx]


def main() -> dict:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COLUMN, 2.8), sharey=True)
    titles = ["2v2 (contested)", "4v4 confirmatory C2 (saturates)"]

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Training timesteps")
        ax.set_ylim(0, 100)
        ax.axhspan(35, 75, color="#EEEEEE", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Own-pole rolling win rate (%)")
    axes[1].tick_params(labelleft=False)

    for s in SERIES:
        x, y = _load_curve(s["path"])
        ax = axes[s["panel"]]
        ax.plot(x, y, color=s["color"], ls=s["ls"], lw=1.2, label=s["label"], zorder=2)
        xm, ym = _downsample(x, y, every=max(1, len(x) // 12))
        ax.plot(
            xm, ym, ls="none", marker=s["marker"], color=s["color"],
            ms=4.5, mec="white", mew=0.4, zorder=3,
        )

    for ax in axes:
        ax.legend(loc="lower right", frameon=False, fontsize=7.5)

    caption = (
        "Shaded band marks the contested ~35--75% specialization-pressure window. "
        "2v2 specialists remain contested; 4v4 $\\pi_{B2}$ climbs near 90% early. "
        "Source: training metrics.csv (rolling_win_rate_200ep)."
    )
    fig.text(0.5, -0.06, caption, ha="center", va="top", fontsize=8, style="italic")
    fig.subplots_adjust(wspace=0.12, bottom=0.22, top=0.88)

    paths = save_figure(fig, "fig_training_emergence")
    print(paths)
    return paths


if __name__ == "__main__":
    main()
