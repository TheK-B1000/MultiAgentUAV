"""2v2 deployment robustness -- readable for someone with no project context.

Implements ROBUSTNESS_2V2_RUNG1_RESULT.json as a grouped bar chart. One panel per
deployment condition (nominal, localization noise, motion error, control delay). Each
panel groups its four bars by OPPONENT ("Pole A", "Pole B") rather than by internal latent
ID, and a legend spells out what the two colors mean in plain language -- no "z0"/"z1"
anywhere on the figure itself.

The policy being tested has two internal modes; Mode A was trained to counter Pole A and
Mode B to counter Pole B. The scientific question a reader can see at a glance: under each
opponent group, is the color-matched mode (blue under "Pole A", orange under "Pole B")
the taller bar? That pattern holding is "crossover survives"; it blurring is degradation.

Uses figure_style directly (apply_style/save_figure/COLORS) rather than the flat-bar
dars_bar_style module, which was built for one un-grouped row of bars per panel (the
VGC-style baseline-comparison figure) and does not fit this figure's two-level grouping
(opponent, then mode) without being stretched past what it was designed for.

Win rate % and the plotted error (half the 95% CI width) are recomputed here from the raw
seed-level CSVs with the SAME frozen bootstrap used everywhere in this project
(n_boot=20000, alpha=0.05, rng_seed=7) -- not copied from the JSON result, so the figure is
independently reproducible from source data.

Run:  python paper/figures/robustness_2v2_bars.py
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

D = ROOT / "artifacts" / "strategic_demand" / "sppo" / "robustness_eval_rows"
CONDITIONS = [
    ("nominal", "nominal", "Nominal"),
    ("localization_noise", "medium", "Localization noise"),
    ("motion_error", "medium", "Motion error"),
    ("control_delay", "medium", "Control delay"),
]
MODE_COLOR = {"A": COLORS["A"], "B": COLORS["B"]}
MODE_LABEL = {"A": "Mode A (trained to counter Pole A)", "B": "Mode B (trained to counter Pole B)"}


def _load(pole: str, z: int, family: str, severity: str) -> np.ndarray:
    p = D / f"rung1_2v2__2v2__pole{pole}__z{z}__{family}__{severity}.csv"
    with p.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return np.array([int(r["win"]) for r in rows], dtype=float)


def _mean_ci_pct(wins: np.ndarray, n_boot=20000, alpha=0.05, rng_seed=7):
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, wins.size, size=(n_boot, wins.size))
    boots = wins[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return wins.mean() * 100, ((hi - lo) / 2) * 100


def main() -> dict:
    apply_style()
    n = len(CONDITIONS)
    fig, axes = plt.subplots(1, n, figsize=(TWO_COLUMN, 2.9), sharey=True)

    group_x = {"A": 0.0, "B": 1.0}   # two opponent groups per panel
    mode_dx = {"A": -0.19, "B": 0.19}   # two bars per group
    bar_w = 0.34

    for i, (ax, (family, severity, title)) in enumerate(zip(axes, CONDITIONS)):
        for pole in ("A", "B"):
            for mode in ("A", "B"):
                wins = _load(pole, 0 if mode == "A" else 1, family, severity)
                value, err = _mean_ci_pct(wins)
                x = group_x[pole] + mode_dx[mode]
                ax.bar(x, value, yerr=err, width=bar_w, color=MODE_COLOR[mode],
                      edgecolor="black", linewidth=0.6, capsize=3,
                      error_kw={"elinewidth": 0.8, "capthick": 0.8})
                ax.text(x, value + err + 2.5, f"{value:.0f}", ha="center", va="bottom",
                        fontsize=7)
        ax.set_xticks([group_x["A"], group_x["B"]])
        ax.set_xticklabels(["Pole A", "Pole B"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, 100)
        if i == 0:
            ax.set_ylabel("Win rate (%)")
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [mpatches.Patch(facecolor=MODE_COLOR[m], edgecolor="black", label=MODE_LABEL[m])
              for m in ("A", "B")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.06),
              ncol=2, frameon=False, fontsize=8.5)

    caption = (
        "Each panel is a different way the deployment environment can be imperfect\n"
        "(applied only after training, never during it). Under each condition, the SAME\n"
        "frozen policy is tested against two opponent styles, Pole A and Pole B. Good\n"
        "behavior is the blue bar (Mode A) winning more against Pole A and the orange bar\n"
        "(Mode B) winning more against Pole B, in every panel. 128 matched test scenarios\n"
        "per bar. Source: ROBUSTNESS_2V2_RUNG1_RESULT.json."
    )
    fig.text(0.5, -0.14, caption, ha="center", va="top", fontsize=8, style="italic")
    fig.subplots_adjust(wspace=0.12, bottom=0.22, top=0.86)

    paths = save_figure(fig, "robustness_2v2_rung1_bars")
    print(paths)
    return paths


if __name__ == "__main__":
    main()
