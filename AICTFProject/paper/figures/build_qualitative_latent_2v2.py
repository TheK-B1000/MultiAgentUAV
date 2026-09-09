"""2v2 latent matched-frame grid: same seed, only z changes.

Sealed Rung-1 sharing-ladder policy (final_rung1.pt), seed 11960003.
Fidelity-checked CUDA replay: terminal scores MATCH rung1_ladder_eval_rows.csv
for every (z, pole) cell.

Scientific claim:
  SUPPORTS: changing only z switches visible blue spatial strategy under a fixed
            matched seed / opponent pole (different situation -> different strategy
            is the pole axis; different z -> different strategy is the column axis).
  DOES NOT: replace the sealed Delta crossover gate (that is fig_latent_crossover_2v2).

Matched tick = 30 (shared 10-tick capture lattice).

Run:  python paper/figures/build_qualitative_latent_2v2.py
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

CAPTURE = ROOT / "artifacts/qualitative_capture/2v2_rung1_matched_frames"
MATCHED_TICK = 30
SEED = 11960003

CELLS = [
    {
        "dir": "z0_poleA",
        "tag": "z0_poleA_seed11960003",
        "z": r"$z_0$",
        "pole": "A",
        "tactical": r"$z_0$: hold home + midfield probe",
        "outcome": "wins 3–1",
    },
    {
        "dir": "z1_poleA",
        "tag": "z1_poleA_seed11960003",
        "z": r"$z_1$",
        "pole": "A",
        "tactical": r"$z_1$: double midfield commit",
        "outcome": "0–0",
    },
    {
        "dir": "z0_poleB",
        "tag": "z0_poleB_seed11960003",
        "z": r"$z_0$",
        "pole": "B",
        "tactical": r"$z_0$: single intruder, home-heavy",
        "outcome": "0–0",
    },
    {
        "dir": "z1_poleB",
        "tag": "z1_poleB_seed11960003",
        "z": r"$z_1$",
        "pole": "B",
        "tactical": r"$z_1$: deep breach / flag pressure",
        "outcome": "wins 1–0",
    },
]


def _frame_path(cell: dict) -> Path:
    return CAPTURE / cell["dir"] / "frames" / f"{cell['tag']}_t{MATCHED_TICK:03d}.png"


def _tick_row(cell: dict) -> dict:
    path = CAPTURE / cell["dir"] / f"{cell['tag']}_tick_log.json"
    ticks = json.loads(path.read_text(encoding="utf-8"))
    return next(r for r in ticks if int(r["tick"]) == MATCHED_TICK)


def _annotation(cell: dict, row: dict) -> str:
    press = "flag pressure ON" if row.get("blue_flag_pressure") else "no flag pressure"
    carry = "carrier present" if row.get("carrying") else "no carrier"
    return f"{cell['tactical']}\n{carry}; {press}\nsealed outcome: {cell['outcome']}"


def main() -> dict:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(TWO_COLUMN, 4.7))

    for ax, cell in zip(axes.ravel(), CELLS):
        img_path = _frame_path(cell)
        if not img_path.exists():
            raise SystemExit(f"REFUSING: missing matched frame {img_path}")
        ax.imshow(imread(str(img_path)), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#333333")
        ax.set_title(f"{cell['z']}  vs  Pole {cell['pole']}", fontsize=9.5, fontweight="bold", pad=4)
        ax.text(
            0.5, -0.04, _annotation(cell, _tick_row(cell)),
            transform=ax.transAxes, ha="center", va="top", fontsize=7.0, linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#888888", linewidth=0.5),
        )

    axes[0, 0].set_ylabel(f"vs Pole A\n(matched t={MATCHED_TICK})", fontsize=9,
                          fontweight="bold", labelpad=8)
    axes[1, 0].set_ylabel(f"vs Pole B\n(matched t={MATCHED_TICK})", fontsize=9,
                          fontweight="bold", labelpad=8)

    handles = [
        mpatches.Patch(facecolor="#4C78A8", edgecolor="black", label="Blue = shared latent policy"),
        mpatches.Patch(facecolor="#E45756", edgecolor="black", label="Red = scripted pole"),
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, frameon=False, fontsize=8,
    )
    caption = (
        f"Matched seed {SEED}, shared Rung-1 policy $\\pi_\\theta(a\\mid o,z)$; "
        f"columns differ only in forced $z$, rows differ only in opponent pole "
        f"(identical initial conditions within each row). "
        f"CUDA fidelity-checked replay (terminals MATCH sealed rung1_ladder_eval_rows.csv). "
        f"READ AS: tangible proof that flipping $z$ changes the team's spatial decision. "
        f"DO NOT READ AS: the statistical specialization gate "
        f"(that is fig_claim_a_2v2 / fig_sharing_ladder_2v2). "
        f"Avoid over-reading coarse telemetry averages (often pole-dominated)."
    )
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=7.5, style="italic")
    fig.subplots_adjust(wspace=0.14, hspace=0.62, bottom=0.16, top=0.90,
                        left=0.10, right=0.98)

    paths = save_figure(fig, "fig_qualitative_latent_2v2")
    print(paths)
    return paths


if __name__ == "__main__":
    main()
