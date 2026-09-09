"""Method pipeline schematic for the successful ICRA story (not SP-PPO).

Boxes:
  strategic demand -> PPO experts -> verify crossover -> matched states
  -> distill z -> verify closed loop -> progressive sharing -> specialization boundary

Run:  python paper/figures/build_method_pipeline_2v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.figures.figure_style import TWO_COLUMN, apply_style, save_figure


STEPS = [
    ("1. Strategic\ndemand", "OP6 vs OP7\nregimes"),
    ("2. PPO\nexperts", r"$\pi_A$, $\pi_B$" + "\nindependent"),
    ("3. Verify\ncrossover", r"$\Delta_A,\Delta_B$" + "\nPASS"),
    ("4. Matched\nstates", "same $s$\nboth experts"),
    ("5. Distill\n$z$-modes", r"$\pi_\theta(a|o,z)$"),
    ("6. Closed-loop\ncheck", r"$z_0/z_1$" + "\nX-pattern"),
    ("7. Progressive\nsharing", "Enc→Back\n→Macro"),
    ("8. Boundary", "where $\\Delta$\ncollapses"),
]


def _box(ax, x, y, w, h, title, sub, face):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0.9, edgecolor="#222222", facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x, y + 0.12, title, ha="center", va="center", fontsize=7.2, fontweight="bold")
    ax.text(x, y - 0.18, sub, ha="center", va="center", fontsize=6.2, color="#333333")


def main() -> dict:
    apply_style()
    fig, ax = plt.subplots(figsize=(TWO_COLUMN, 2.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n = len(STEPS)
    xs = [0.07 + i * (0.86 / (n - 1)) for i in range(n)]
    y = 0.58
    w, h = 0.105, 0.52
    faces = ["#ECEFF1", "#E3F2FD", "#E3F2FD", "#FFF8E1", "#E8F5E9", "#E8F5E9", "#FBE9E7", "#F3E5F5"]

    for i, ((title, sub), x, face) in enumerate(zip(STEPS, xs, faces)):
        _box(ax, x, y, w, h, title, sub, face)
        if i < n - 1:
            ax.annotate(
                "", xy=(xs[i + 1] - w / 2 - 0.005, y), xytext=(x + w / 2 + 0.005, y),
                arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.0),
            )

    ax.text(
        0.5, 0.14,
        r"PPO = expert engine only  ·  distillation creates $z$  ·  sharing = primary experimental variable  ·  crossover = validation",
        ha="center", va="center", fontsize=7.5, color="#222222",
    )
    caption = (
        "Successful method pipeline (Claim A). Not a new PPO variant: opponent-specific "
        "PPO experts are distilled into discrete strategy modes under a common recipe, then "
        "progressively shared while closed-loop payoff specialization is measured. "
        "Share-0 is a bit-exact expert-dispatch control (not a distillation rung). "
        "SP-PPO/CSC/SPFT are excluded (sealed non-recovery; appendix only)."
    )
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=7.0, style="italic")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.18)

    paths = save_figure(fig, "fig_method_pipeline_2v2")
    print(paths)
    return paths


if __name__ == "__main__":
    main()
