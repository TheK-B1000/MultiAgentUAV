"""Qualitative matched-frame grid for 4v4 confirmatory C2 (seed 16400001).

Same sealed evaluation seed, same absolute tick, four policy x pole cells.
Fidelity already verified by replay_capture_qualitative.py against the sealed
crossover CSV (terminal scores MATCH).

Scientific claim this figure supports (and does NOT support):
  SUPPORTS: opponent-conditioned behavioral differentiation is visible even when
            the win-rate crossover gate FAILS.
  DOES NOT: confirm specialization / crossover survival. Keep that distinction sharp.

Matched tick = 30 for every cell (first shared post-opening frame on the 10-tick
capture lattice that still exists for the shortest episode, pi_B2@A which ends at 82).

Annotations use BLUE-side observables from the tick log (carrier, support distance,
flag pressure) plus a short tactical label describing the visible blue spatial layout.
Red BT role tags are reported as induced opponent response state, not as learned
blue roles.

Run:  python paper/figures/build_qualitative_matched_frames.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.image import imread

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.figures.figure_style import TWO_COLUMN, apply_style, save_figure

CAPTURE = ROOT / "artifacts/qualitative_capture/4v4_c2_matched_frames"
MATCHED_TICK = 30
SEED = 16400001

# (row, col) -> cell directory / tag / tactical label
# rows: Pole A, Pole B; cols: pi_A2, pi_B2
CELLS = [
    # row 0 -- vs Pole A
    {
        "dir": "piA2_poleA",
        "tag": "piA2_poleA_seed16400001",
        "policy": r"$\pi_{A2}$",
        "pole": "A",
        "tactical": "Mass forward pressure",
    },
    {
        "dir": "piB2_poleA",
        "tag": "piB2_poleA_seed16400001",
        "policy": r"$\pi_{B2}$",
        "pole": "A",
        "tactical": "Carrier + home support",
    },
    # row 1 -- vs Pole B
    {
        "dir": "piA2_poleB",
        "tag": "piA2_poleB_seed16400001",
        "policy": r"$\pi_{A2}$",
        "pole": "B",
        "tactical": "Staged penetration",
    },
    {
        "dir": "piB2_poleB",
        "tag": "piB2_poleB_seed16400001",
        "policy": r"$\pi_{B2}$",
        "pole": "B",
        "tactical": "Home defense / retrieval",
    },
]


def _frame_path(cell: dict) -> Path:
    return CAPTURE / cell["dir"] / "frames" / f"{cell['tag']}_t{MATCHED_TICK:03d}.png"


def _tick_row(cell: dict) -> dict:
    path = CAPTURE / cell["dir"] / f"{cell['tag']}_tick_log.json"
    ticks = json.loads(path.read_text(encoding="utf-8"))
    return ticks[MATCHED_TICK]


_ROLE_SHORT = {
    "ATTACKER": "Atk",
    "DEFENDER": "Def",
    "INTERCEPTOR": "Int",
    "FLAG_RETR": "Flg",
    "ESCORT": "Esc",
    "COUNTER": "Ctr",
    "2V1_WING": "2v1",
}


def _annotation(cell: dict, row: dict) -> str:
    roles = row.get("red_role_counts") or {}
    role_str = ", ".join(
        f"{_ROLE_SHORT.get(k, k)}×{v}" for k, v in sorted(roles.items())
    ) or "none"
    if row.get("carrying"):
        d = row.get("carrier_nearest_teammate_dist")
        carry = f"carrier (support {d:.1f})" if d is not None else "carrier"
    else:
        carry = "no carrier"
    press = "flag pressure" if row.get("blue_flag_pressure") else "no pressure"
    return f"{cell['tactical']}\n{carry}; {press}\nopponent BT: {role_str}"


def main() -> dict:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(TWO_COLUMN, 4.6))

    for ax, cell in zip(axes.ravel(), CELLS):
        img_path = _frame_path(cell)
        if not img_path.exists():
            raise SystemExit(f"REFUSING: missing matched frame {img_path}")
        img = imread(str(img_path))
        ax.imshow(img, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#333333")

        title = f"{cell['policy']}  vs  Pole {cell['pole']}"
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=4)

        row = _tick_row(cell)
        # annotation strip under the frame
        ax.text(
            0.5, -0.04, _annotation(cell, row),
            transform=ax.transAxes, ha="center", va="top",
            fontsize=7.0, linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#888888", linewidth=0.5),
        )

    # row labels on the left
    axes[0, 0].set_ylabel(f"vs Pole A\n(matched t={MATCHED_TICK})", fontsize=9,
                          fontweight="bold", labelpad=8)
    axes[1, 0].set_ylabel(f"vs Pole B2\n(matched t={MATCHED_TICK})", fontsize=9,
                          fontweight="bold", labelpad=8)

    # legend for team colors
    handles = [
        mpatches.Patch(facecolor="#4C78A8", edgecolor="black", label="Blue = learned policy"),
        mpatches.Patch(facecolor="#E45756", edgecolor="black", label="Red = scripted pole"),
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, frameon=False, fontsize=8,
    )

    caption = (
        f"Matched-frame grid on sealed C2 crossover seed {SEED} "
        f"(fidelity-checked replay; terminal scores MATCH the sealed CSV). "
        f"Every panel shows absolute tick {MATCHED_TICK} -- the same episode phase, "
        f"not cherry-picked highlights. "
        f"Annotations: blue-side carrier / support / flag pressure, plus a short "
        f"tactical label for the visible blue spatial layout; opponent BT tags are "
        f"induced red response state, not learned blue roles. "
        f"READ AS: opponent-conditioned behavioral differentiation is visible. "
        f"DO NOT READ AS: specialization / crossover confirmed -- the sealed C2 "
        f"win-rate gate FAILS."
    )
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=7.5, style="italic")
    fig.subplots_adjust(wspace=0.14, hspace=0.62, bottom=0.16, top=0.90,
                        left=0.10, right=0.98)

    paths = save_figure(fig, "fig_qualitative_matched_frames_c2")
    print(paths)
    return paths


if __name__ == "__main__":
    main()
