"""Absolute win-rate ablation: our controller vs. every available variant.

Two panels:
  (a) 2v2 -- the FULL ladder plus the generalist baseline:
      pi_G (no strategy code) | Share-0 | Share-Encoder | Share-Backbone | Share-Macro
  (b) 6v6 -- zero-sharing teachers vs. the Share-Encoder student.
      NOTE: no generalist exists at 6v6 (never trained), so that bar is absent
      by construction, not omitted.

Each variant shows four cells: the A-side code and the B-side code, evaluated
against both opponent regimes. This is ABSOLUTE performance context only --
specialization is decided by the paired Delta criterion, not by these bars.

All values are read from paper/data/RESULTS_DATA_2v2_6v6.json (harvested from
sealed artifacts). Nothing is retyped.

Run:  ./.venv/Scripts/python.exe paper/figures/build_ablation_winrate_2v2_6v6.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.figures.figure_style import (  # noqa: E402
    COLORS,
    TWO_COLUMN,
    apply_style,
    save_figure,
)

DATA = ROOT / "paper" / "data" / "RESULTS_DATA_2v2_6v6.json"


def main() -> int:
    d = json.loads(DATA.read_text(encoding="utf-8"))
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COLUMN, 2.9),
                             gridspec_kw={"width_ratios": [1.45, 1.0]})

    bw = 0.20  # bar width

    # ------------------------------------------------------------------ (a) 2v2
    ax = axes[0]
    ladder = d["2v2"]["sharing_ladder"]
    pig = d["2v2"]["generalist_pi_G"]

    # column order: generalist first (it has no code), then the ladder
    cols = []
    cols.append(("$\\pi_G$\n(no code)",
                 {"A_code_A": None, "A_code_B": None,
                  "B_code_A": None, "B_code_B": None},
                 (pig["V_pi_G_A"]["mean"], pig["V_pi_G_B"]["mean"])))
    for label, key in (("Share-0", "Share-0"),
                       ("Share-\nEncoder", "Share-Encoder"),
                       ("Share-\nBackbone", "Share-Backbone"),
                       ("Share-\nMacro", "Share-Macro")):
        c = ladder[key]["cell_win_rates"]
        cols.append((label, c, None))

    xs = np.arange(len(cols), dtype=float)
    for i, (label, cells, gen) in enumerate(cols):
        if gen is not None:
            # generalist: one policy, two regimes -- draw as two hatched bars
            ax.bar(i - bw * 0.75, gen[0], bw * 1.5, color=COLORS["control"],
                   edgecolor="black", linewidth=0.6, hatch="//",
                   label="$\\pi_G$ vs A" if i == 0 else None)
            ax.bar(i + bw * 0.75, gen[1], bw * 1.5, color=COLORS["control"],
                   edgecolor="black", linewidth=0.6, hatch="..",
                   label="$\\pi_G$ vs B" if i == 0 else None)
            continue
        vals = [cells["z0_poleA"], cells["z1_poleA"],
                cells["z0_poleB"], cells["z1_poleB"]]
        offs = [-1.5 * bw, -0.5 * bw, 0.5 * bw, 1.5 * bw]
        cols_ = [COLORS["A"], COLORS["B"], COLORS["A"], COLORS["B"]]
        hatches = ["", "", "//", "//"]
        for off, v, c, h in zip(offs, vals, cols_, hatches):
            ax.bar(i + off, v, bw, color=c, edgecolor="black", linewidth=0.6,
                   hatch=h)

    ax.axhline(0.5, color="#999999", linewidth=0.7, linestyle=":", zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([c[0] for c in cols], fontsize=7.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("win rate")
    ax.set_title("(a) 2v2: full sharing ladder vs. generalist", loc="left")

    # Two-axis legend: colour = which code, hatch = which opponent regime.
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=COLORS["A"], edgecolor="black", lw=0.6, label="A-side code $z_0$"),
        Patch(facecolor=COLORS["B"], edgecolor="black", lw=0.6, label="B-side code $z_1$"),
        Patch(facecolor="white", edgecolor="black", lw=0.6, label="vs Regime A"),
        Patch(facecolor="white", edgecolor="black", lw=0.6, hatch="//", label="vs Regime B"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6.8, ncol=4,
              loc="upper center", handlelength=1.1, columnspacing=0.8,
              handletextpad=0.4)

    # ------------------------------------------------------------------ (b) 6v6
    ax = axes[1]
    teach = d["6v6"]["share0_teacher_diagnostic"]["cell_win_rates"]
    stud = d["6v6"]["crossover"]["cell_win_rates"]

    groups = [
        ("Share-0\n(teachers)",
         [teach["pi_A_poleA"], teach["pi_B_poleA"],
          teach["pi_A_poleB"], teach["pi_B_poleB"]]),
        ("Share-\nEncoder",
         [stud["z0_poleA"], stud["z1_poleA"],
          stud["z0_poleB"], stud["z1_poleB"]]),
    ]
    offs = [-1.5 * bw, -0.5 * bw, 0.5 * bw, 1.5 * bw]
    cols_ = [COLORS["A"], COLORS["B"], COLORS["A"], COLORS["B"]]
    hatches = ["", "", "//", "//"]
    for i, (label, vals) in enumerate(groups):
        for off, v, c, h in zip(offs, vals, cols_, hatches):
            ax.bar(i + off, v, bw, color=c, edgecolor="black",
                   linewidth=0.6, hatch=h)

    ax.axhline(0.5, color="#999999", linewidth=0.7, linestyle=":", zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[0] for g in groups], fontsize=7.5)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 1.08)
    ax.set_title("(b) 6v6: no generalist trained", loc="left")
    ax.text(0.5, 0.34, "every cell 0.84–0.99:\nuniformly strong,\nno crossover pattern",
            ha="center", va="center", fontsize=7.0, color="#444444",
            transform=ax.transAxes)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    report = save_figure(fig, "fig_ablation_winrate_2v2_6v6")
    print(f"  wrote: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
