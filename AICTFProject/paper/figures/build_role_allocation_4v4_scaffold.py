"""Role-allocation companion for 4v4 A' scaffold vs pi_B.

Panels (same layout as fig_role_allocation_2v2):
  (a) defender-like allocation (near own flag home)
  (b) attacker-like allocation
  (c) attack/defense ratio

Source (exploratory, not a gate):
  artifacts/qualitative_capture/4v4_scaffold_trajectory_strip/role_allocation.json

Caption discipline: A' has two forced DEFEND targets by construction; near-home
counts are a behavioral readout. Do not overclaim learned role specialization.

Run:  ./.venv/Scripts/python.exe paper/figures/build_role_allocation_4v4_scaffold.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.figures.figure_style import COLORS, TWO_COLUMN, apply_style, save_figure

SRC = ROOT / "artifacts/qualitative_capture/4v4_scaffold_trajectory_strip/role_allocation.json"
STEM = "fig_role_allocation_4v4_scaffold"
COMBINED = ROOT / "paper" / "plots" / "combined"

MODE_COLOR = {"A_prime": COLORS["A"], "pi_B": COLORS["B"]}
MODE_HATCH = {"A_prime": "///", "pi_B": "xxx"}
MODE_LABEL = {
    "A_prime": r"$A'=\pi_A+2$D (scaffold)",
    "pi_B": r"$\pi_B$ (learned)",
}
PANELS = [
    ("(a) Defender-like allocation", "Mean near-home / tick", "num_defenders", (0, 3.2)),
    ("(b) Attacker-like allocation", "Mean forward / tick", "num_attackers", (0, 3.2)),
    ("(c) Attack / defense ratio", "Attack ÷ defense", "attack_defense_ratio", (0, 6.0)),
]


def main() -> dict:
    if not SRC.exists():
        raise SystemExit(f"REFUSING: missing {SRC}")
    means = json.loads(SRC.read_text(encoding="utf-8"))["per_condition_means"]
    apply_style()

    fig, axes = plt.subplots(1, 3, figsize=(TWO_COLUMN, 2.7), sharey=False)
    group_x = {"A": 0.0, "B": 1.0}
    mode_dx = {"A_prime": -0.18, "pi_B": 0.18}
    bar_w = 0.32

    for ax, (title, ylabel, key, ylim) in zip(axes, PANELS):
        for pole in ("A", "B"):
            for mode in ("A_prime", "pi_B"):
                value = float(means[f"{mode}_pole{pole}"][key])
                ax.bar(
                    group_x[pole] + mode_dx[mode], value, width=bar_w,
                    color=MODE_COLOR[mode], hatch=MODE_HATCH[mode],
                    edgecolor="black", linewidth=0.6,
                )
                ax.text(
                    group_x[pole] + mode_dx[mode], value + 0.05 * ylim[1],
                    f"{value:.2f}", ha="center", va="bottom", fontsize=7,
                )
        ax.set_xticks([group_x["A"], group_x["B"]])
        ax.set_xticklabels(["Regime A", "Regime B"])
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=8.5)

    handles = [
        mpatches.Patch(
            facecolor=MODE_COLOR[m], hatch=MODE_HATCH[m],
            edgecolor="black", label=MODE_LABEL[m],
        )
        for m in ("A_prime", "pi_B")
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.08),
        ncol=2, frameon=False, fontsize=8,
    )
    fig.subplots_adjust(wspace=0.32, bottom=0.14, top=0.78, left=0.06, right=0.99)

    paths = save_figure(fig, STEM)
    COMBINED.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        src = ROOT / "paper" / "generated" / f"{STEM}.{ext}"
        shutil.copy2(src, COMBINED / f"{STEM}.{ext}")
    print(paths)
    return paths


if __name__ == "__main__":
    main()
