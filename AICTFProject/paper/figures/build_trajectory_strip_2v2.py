"""Matched z0 vs z1 trajectory strip: overlay paths + first decision divergence.

Reads artifacts/qualitative_capture/2v2_rung1_trajectory_strip/trajectories.json
(produced by experiments/export_matched_z_trajectories_2v2.py with fidelity MATCH).

Layout: two panels (Pole A / Pole B). In each panel, blue agent paths under z0
(solid) and z1 (dashed) are overlaid from the same sealed seed. A star marks
the first action divergence tick (fallback: first position divergence).

Scientific claim:
  SUPPORTS: changing only z changes what the team actually does under identical
            initial conditions (connects quantitative crossover to visible behavior).
  DOES NOT: replace the sealed Delta gate (fig_claim_a / fig_sharing_ladder).

Run:  python paper/figures/build_trajectory_strip_2v2.py
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

from paper.figures.figure_style import COLORS, TWO_COLUMN, apply_style, save_figure

SRC = ROOT / "artifacts/qualitative_capture/2v2_rung1_trajectory_strip/trajectories.json"
SEED = 11960003


def _cell(blob: dict, z: int, pole: str) -> dict:
    for c in blob["cells"]:
        if c["z"] == z and c["pole"] == pole:
            return c
    raise KeyError((z, pole))


def _paths(cell: dict):
    ticks = cell["ticks"]
    b0 = np.array([[t["blue_x"][0], t["blue_y"][0]] for t in ticks], dtype=float)
    b1 = np.array([[t["blue_x"][1], t["blue_y"][1]] for t in ticks], dtype=float)
    return b0, b1


def _panel(ax, blob: dict, pole: str) -> None:
    z0, z1 = _cell(blob, 0, pole), _cell(blob, 1, pole)
    div = blob["divergence_by_pole"][pole]
    mark = div.get("mark_tick")
    b0_z0, b1_z0 = _paths(z0)
    b0_z1, b1_z1 = _paths(z1)

    # Draw z0 solid, z1 dashed; agent 0 thicker, agent 1 thinner
    ax.plot(b0_z0[:, 0], b0_z0[:, 1], color=COLORS["A"], ls="-", lw=1.6, label=r"$z_0$ agent0")
    ax.plot(b1_z0[:, 0], b1_z0[:, 1], color=COLORS["A"], ls="-", lw=1.0, alpha=0.75, label=r"$z_0$ agent1")
    ax.plot(b0_z1[:, 0], b0_z1[:, 1], color=COLORS["B"], ls="--", lw=1.6, label=r"$z_1$ agent0")
    ax.plot(b1_z1[:, 0], b1_z1[:, 1], color=COLORS["B"], ls="--", lw=1.0, alpha=0.75, label=r"$z_1$ agent1")

    # start markers (identical by construction)
    ax.plot(b0_z0[0, 0], b0_z0[0, 1], marker="o", color="#222", ms=5, zorder=5)
    ax.plot(b1_z0[0, 0], b1_z0[0, 1], marker="o", color="#222", ms=4, zorder=5)

    if mark is not None and mark < len(b0_z0) and mark < len(b0_z1):
        # mark both z paths at divergence
        ax.plot(b0_z0[mark, 0], b0_z0[mark, 1], marker="*", color=COLORS["A"], ms=11, zorder=6)
        ax.plot(b0_z1[mark, 0], b0_z1[mark, 1], marker="*", color=COLORS["B"], ms=11, zorder=6)
        ax.annotate(
            f"first action diverge t={mark}",
            xy=(b0_z0[mark, 0], b0_z0[mark, 1]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=7, color="#222",
        )

    term0 = z0["terminal"]
    term1 = z1["terminal"]
    ax.set_title(
        f"Pole {pole}  |  $z_0$ {term0['blue']}–{term0['red']}   $z_1$ {term1['blue']}–{term1['red']}",
        fontsize=9, fontweight="bold",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False, fontsize=6.5, ncol=2)


def main() -> dict:
    if not SRC.exists():
        raise SystemExit(
            f"REFUSING: missing {SRC}. Run "
            "python experiments/export_matched_z_trajectories_2v2.py --device cuda first."
        )
    blob = json.loads(SRC.read_text(encoding="utf-8"))
    assert blob.get("fidelity") == "ALL_MATCH", blob.get("fidelity")
    assert int(blob["seed"]) == SEED

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COLUMN, 3.2))
    _panel(axes[0], blob, "A")
    _panel(axes[1], blob, "B")

    caption = (
        f"Matched seed {SEED}, Share-Encoder $\\pi_\\theta(a\\mid o,z)$; "
        "identical initial conditions within each panel; only forced $z$ differs. "
        "Solid=$z_0$, dashed=$z_1$; stars mark first action divergence. "
        "CUDA fidelity MATCH vs sealed rung1_ladder_eval_rows.csv. "
        "READ AS: the code changes what the team does. "
        "DO NOT READ AS: the statistical specialization gate."
    )
    fig.text(0.5, -0.04, caption, ha="center", va="top", fontsize=7.2, style="italic")
    fig.subplots_adjust(wspace=0.22, bottom=0.20, top=0.88, left=0.06, right=0.99)

    paths = save_figure(fig, "fig_trajectory_strip_2v2")
    print(paths)
    return paths


if __name__ == "__main__":
    main()
