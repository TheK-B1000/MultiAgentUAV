"""Figure 1: the maritime CTF task and its two strategic poles.

A clean top-down schematic, not a game-UI screenshot -- a reviewer should understand the task
in about ten seconds. Two condition panels sharing one arena layout (2v2, symmetric home/flag
per side, matching gpu_env/_maps.py's map_a_open and ctfviewer.py's left/right blue/red split),
differing only in the scripted opponent's posture -- which is exactly what makes the two poles
real strategic conditions rather than a cosmetic split:

  Pole A: opponent = OP6 (TURTLE archetype, docs/same-map-tactical-regimes.md), plays
          defensively near its own home -> GUARD is the favored response.
  Pole B: opponent = OP7 (SWITCHER archetype), pushes toward the opponent's home -> BREACH is
          the favored response.

Both the archetype names and the GUARD/BREACH result are real, not illustrative:
docs/research-progress-tracker.md records the actual gate result this claim rests on --
"OP7 BREACH-GUARD +0.531 PASS; OP6 GUARD-BREACH +0.094 FAIL" -- and
experiments/opponent_spec.py's pole_B_genome() confirms Pole B is canonical OP7.

Run:  python paper/figures/build_fig1_environment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, RegularPolygon

from paper.figures.figure_style import COLORS, TWO_COLUMN, apply_style, save_figure

WATER = "#eaf2f7"


def _agent_marker(ax, x, y, color, heading_right: bool):
    dx = 0.28 if heading_right else -0.28
    tri = Polygon([(x + dx, y), (x - dx * 0.5, y + 0.16), (x - dx * 0.5, y - 0.16)],
                 closed=True, facecolor=color, edgecolor="black", lw=0.5, zorder=4)
    ax.add_patch(tri)


def _flag(ax, x, y, color):
    ax.plot([x, x], [y - 0.18, y + 0.22], color="black", lw=0.9, zorder=4)
    ax.add_patch(Polygon([(x, y + 0.22), (x + 0.22, y + 0.13), (x, y + 0.05)],
                         closed=True, facecolor=color, edgecolor="black", lw=0.4, zorder=4))


def _panel(ax, title: str, opp_archetype: str, opp_posture: str, favored: str,
          specialist: str, spec_color: str, red_near_home: bool) -> None:
    ax.add_patch(FancyBboxPatch((0, 0), 10, 5, boxstyle="round,pad=0.02",
                                facecolor=WATER, edgecolor="#333333", lw=0.8, zorder=0))
    ax.axvline(5, color="#333333", lw=0.7, ls=(0, (4, 3)), zorder=1)

    ax.add_patch(FancyBboxPatch((0.15, 1.9), 0.7, 1.2, boxstyle="round,pad=0.03",
                                facecolor=COLORS["A"], alpha=0.25, edgecolor=COLORS["A"],
                                lw=1.0, zorder=1))
    ax.add_patch(FancyBboxPatch((9.15, 1.9), 0.7, 1.2, boxstyle="round,pad=0.03",
                                facecolor=COLORS["B"], alpha=0.25, edgecolor=COLORS["B"],
                                lw=1.0, zorder=1))
    ax.text(0.5, 4.55, "blue home", ha="center", fontsize=7.5, color=COLORS["A"])
    ax.text(9.5, 4.55, "red home", ha="center", fontsize=7.5, color=COLORS["B"])
    _flag(ax, 0.5, 2.5, COLORS["A"])
    _flag(ax, 9.5, 2.5, COLORS["B"])

    for y in (2.9, 2.1):
        _agent_marker(ax, 1.1, y, COLORS["A"], heading_right=True)
    red_x = (8.6 if red_near_home else 4.6)
    for y in (2.9, 2.1):
        _agent_marker(ax, red_x, y, COLORS["B"], heading_right=False)

    ax.annotate(f"{opp_archetype}\n{opp_posture}", xy=(red_x, 1.85),
               ha="center", va="top", fontsize=7.5, color=COLORS["B"], fontstyle="italic")
    ax.annotate(f"favored response: {favored}", xy=(5, 0.55), ha="center", va="center",
               fontsize=8.5, fontweight="bold")
    ax.annotate(f"learns {specialist}", xy=(5, 4.85), ha="center", va="center",
               fontsize=8, color=spec_color, fontweight="bold")

    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(-0.3, 5.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9.5, fontweight="bold")


def main() -> int:
    apply_style()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(TWO_COLUMN, 2.6))

    _panel(axA, "(a) Pole A: opponent OP6 (TURTLE)", "TURTLE", "holds near red home",
          "GUARD", r"$\pi_A$", COLORS["A"], red_near_home=True)
    _panel(axB, "(b) Pole B: opponent OP7 (SWITCHER)", "SWITCHER", "pushes toward blue home",
          "BREACH", r"$\pi_B$", COLORS["B"], red_near_home=False)

    fig.suptitle("2v2 maritime capture-the-flag: two opponents, two favored strategies",
                fontsize=9.5, y=0.98)
    fig.subplots_adjust(wspace=0.12, top=0.80)

    paths = save_figure(fig, "fig1_environment")
    print("Figure 1 written:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
