"""2v2 latent behavioral differentiation: z0 vs z1 under each opponent pole.

Shows defender / attacker / intercept / carrier-support / flag-pressure proxies
so specialization is visible as *behavior*, not only as win-rate Delta.

Source: Z0_Z1_BEHAVIOR_CHARACTERIZATION_final_ccp_successor_production.json
(exploratory diagnostic, n=24 episodes/cell -- NOT a gate). Means only are
available in the sealed record (no episode-level vectors for bootstrap CIs).

This is the LILI/ROMA-style role-behavior companion to fig_latent_crossover_2v2.

Run:  python paper/figures/build_latent_behavior_2v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.figures.figure_style import COLORS, TWO_COLUMN, apply_style, save_figure

SRC = ROOT / (
    "artifacts/strategic_demand/sppo/"
    "Z0_Z1_BEHAVIOR_CHARACTERIZATION_final_ccp_successor_production.json"
)

MODE_COLOR = {"z0": COLORS["A"], "z1": COLORS["B"]}
MODE_LABEL = {
    "z0": r"$z_0$ (Pole-A code)",
    "z1": r"$z_1$ (Pole-B code)",
}

# (title, ylabel, feature_key, scale, ylim, note)
PANELS = [
    ("Defender allocation", "Mean defenders", "num_defenders", 1.0, (0, 1.0)),
    ("Attacker allocation", "Mean attackers", "num_attackers", 1.0, (0, 1.0)),
    ("Intercept near carrier", "Intercept count", "n_intercept_near_enemy_carrier", 1.0, (0, 0.6)),
    ("Carrier escort", "Escort count", "carrier_escort_count", 1.0, (0, 0.5)),
    ("Attack / defense ratio", "Attack÷defense", "attack_defense_ratio", 1.0, (0, 1.0)),
]


def main() -> dict:
    blob = json.loads(SRC.read_text(encoding="utf-8"))
    means = blob["per_condition_means"]
    apply_style()

    fig, axes = plt.subplots(1, len(PANELS), figsize=(TWO_COLUMN, 2.85), sharey=False)
    group_x = {"A": 0.0, "B": 1.0}
    mode_dx = {"z0": -0.18, "z1": 0.18}
    bar_w = 0.32

    for ax, (title, ylabel, key, scale, ylim) in zip(axes, PANELS):
        for pole in ("A", "B"):
            for z in ("z0", "z1"):
                cell = means[f"{z}_pole{pole}"]
                value = float(cell[key]) * scale
                x = group_x[pole] + mode_dx[z]
                ax.bar(
                    x, value, width=bar_w, color=MODE_COLOR[z],
                    edgecolor="black", linewidth=0.6,
                )
        ax.set_xticks([group_x["A"], group_x["B"]])
        ax.set_xticklabels(["vs Pole A", "vs Pole B"])
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=8.5, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        mpatches.Patch(facecolor=MODE_COLOR[z], edgecolor="black", label=MODE_LABEL[z])
        for z in ("z0", "z1")
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.05),
        ncol=2, frameon=False, fontsize=8,
    )
    caption = (
        r"Same shared latent policy; bars are forced $z$ under each opponent regime. "
        r"Coarse telemetry means are largely \emph{pole-driven} (situation), with smaller "
        r"within-pole $z_0$ vs $z_1$ gaps -- do not over-read these as tactic cartoons. "
        r"For matched-state visual proof that flipping $z$ changes spatial layout, see "
        r"fig\_qualitative\_latent\_2v2. "
        r"Source: exploratory $z_0/z_1$ characterization (n=24/cell; not a gate)."
    )
    fig.text(0.5, -0.06, caption, ha="center", va="top", fontsize=7.5, style="italic")
    fig.subplots_adjust(wspace=0.35, bottom=0.22, top=0.82)

    paths = save_figure(fig, "fig_latent_behavior_2v2")
    print(paths)
    return paths


if __name__ == "__main__":
    main()
