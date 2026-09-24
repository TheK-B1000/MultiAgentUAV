"""Main-paper role-allocation companion: z0 vs z1 under each regime.

Focused three-panel version of fig_latent_behavior_2v2 for the reviewer ask:
"do latent modes look like recognizable team strategies?"

Panels:
  (a) defender allocation
  (b) attacker allocation
  (c) attack/defense ratio

Source (exploratory, not a gate):
  Z0_Z1_BEHAVIOR_CHARACTERIZATION_final_ccp_successor_production.json
  n=24 episodes/cell; means only (no bootstrap CIs in the sealed record).

Caption discipline (from 2v2_results_package):
  Coarse telemetry is largely pole-driven; within-pole z gaps are smaller.
  Matched-state visual proof of z-flip remains fig_trajectory_strip /
  fig_qualitative_latent. Do not overclaim tactic cartoons from these bars.

Run:  python paper/figures/build_role_allocation_2v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.figures.figure_style import COLORS, TWO_COLUMN, apply_style, save_figure

SRC = ROOT / (
    "artifacts/strategic_demand/sppo/"
    "Z0_Z1_BEHAVIOR_CHARACTERIZATION_final_ccp_successor_production.json"
)
STEM = "fig_role_allocation_2v2"

MODE_COLOR = {"z0": COLORS["A"], "z1": COLORS["B"]}
MODE_HATCH = {"z0": "///", "z1": "xxx"}  # redundant encoding beyond color
MODE_LABEL = {
    "z0": r"$z_0$ (A-specialized mode)",
    "z1": r"$z_1$ (B-specialized mode)",
}

PANELS = [
    ("(a) Defender allocation", "Mean defenders / tick", "num_defenders", (0, 0.85)),
    ("(b) Attacker allocation", "Mean attackers / tick", "num_attackers", (0, 0.85)),
    ("(c) Attack / defense ratio", "Attack ÷ defense", "attack_defense_ratio", (0, 0.85)),
]


def main() -> dict:
    means = json.loads(SRC.read_text(encoding="utf-8"))["per_condition_means"]
    apply_style()

    fig, axes = plt.subplots(1, 3, figsize=(TWO_COLUMN, 2.7), sharey=False)
    group_x = {"A": 0.0, "B": 1.0}
    mode_dx = {"z0": -0.18, "z1": 0.18}
    bar_w = 0.32

    for ax, (title, ylabel, key, ylim) in zip(axes, PANELS):
        for pole in ("A", "B"):
            for z in ("z0", "z1"):
                value = float(means[f"{z}_pole{pole}"][key])
                ax.bar(
                    group_x[pole] + mode_dx[z], value, width=bar_w,
                    color=MODE_COLOR[z], hatch=MODE_HATCH[z],
                    edgecolor="black", linewidth=0.6,
                )
                ax.text(
                    group_x[pole] + mode_dx[z], value + 0.02, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=7,
                )
        ax.set_xticks([group_x["A"], group_x["B"]])
        ax.set_xticklabels(["Regime A", "Regime B"])
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=8.5)

    handles = [
        mpatches.Patch(
            facecolor=MODE_COLOR[z], hatch=MODE_HATCH[z],
            edgecolor="black", label=MODE_LABEL[z],
        )
        for z in ("z0", "z1")
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.08),
        ncol=2, frameon=False, fontsize=8,
    )
    fig.subplots_adjust(wspace=0.32, bottom=0.14, top=0.78, left=0.06, right=0.99)

    paths = save_figure(fig, STEM)
    print(paths)
    return paths


if __name__ == "__main__":
    main()
