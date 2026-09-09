"""Progressive sharing ladder: does specialization survive as sharing increases?

Maps sealed Rung 0..3 results onto the paper's Share-0 / Share-Encoder /
Share-Backbone / Share-Macro labels. Holds experts, dataset, schedule, seeds,
and specialization criterion fixed; only shared modules change.

Left panel: Delta_A / Delta_B with 95% CI whiskers and gate trail.
Right panel: paired within-seed D_A / D_B vs Share-0 (detectable loss iff UCB95<0).

Sources:
  RUNG0_LADDER_REFERENCE.json
  RUNG1_LADDER_EVAL_RESULT.json
  RUNG2_LADDER_EVAL_RESULT.json
  RUNG3_LADDER_EVAL_RESULT.json

Run:  python paper/figures/build_sharing_ladder_2v2.py
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

from paper.figures.figure_style import COLORS, LINESTYLES, MARKERS, TWO_COLUMN, apply_style, save_figure

SD = ROOT / "artifacts/strategic_demand/sppo"

# Paper table labels <-> sealed rung records
LADDER = [
    {
        "label": "Share-0",
        "sub": "Independent experts",
        "delta_src": ("RUNG0_LADDER_REFERENCE.json", "POOLED_N128"),
        "d_src": None,  # reference
    },
    {
        "label": "Share-Encoder",
        "sub": "Shared CNN",
        "delta_src": ("RUNG1_LADDER_EVAL_RESULT.json", "OWN_GATE_N128"),
        "d_src": ("RUNG1_LADDER_EVAL_RESULT.json", "PRIMARY_WITHIN_SEED"),
    },
    {
        "label": "Share-Backbone",
        "sub": "+ MLP backbone",
        "delta_src": ("RUNG2_LADDER_EVAL_RESULT.json", "OWN_GATE_N128"),
        "d_src": ("RUNG2_LADDER_EVAL_RESULT.json", "PRIMARY_WITHIN_SEED"),
    },
    {
        "label": "Share-Macro",
        "sub": "+ macro outputs",
        "delta_src": ("RUNG3_LADDER_EVAL_RESULT.json", "OWN_GATE_N128"),
        "d_src": ("RUNG3_LADDER_EVAL_RESULT.json", "PRIMARY_WITHIN_SEED"),
    },
]


def _load(name: str) -> dict:
    return json.loads((SD / name).read_text(encoding="utf-8"))


def _delta_triplet(blob: dict, key: str):
    g = blob[key]
    da, db = g["delta_A"], g["delta_B"]
    gate = "PASS" if g.get("passes") else "FAIL"
    return (
        da["mean"] * 100, (da["mean"] - da["lcb95"]) * 100, (da["ucb95"] - da["mean"]) * 100,
        db["mean"] * 100, (db["mean"] - db["lcb95"]) * 100, (db["ucb95"] - db["mean"]) * 100,
        gate,
    )


def _d_triplet(blob: dict, key: str):
    g = blob[key]
    da, db = g["D_A"], g["D_B"]
    return (
        da["mean"] * 100, (da["mean"] - da["lcb95"]) * 100, (da["ucb95"] - da["mean"]) * 100,
        db["mean"] * 100, (db["mean"] - db["lcb95"]) * 100, (db["ucb95"] - db["mean"]) * 100,
    )


def main() -> dict:
    apply_style()
    xs = np.arange(len(LADDER))

    da, da_lo, da_hi = [], [], []
    db, db_lo, db_hi = [], [], []
    gates = []
    dA, dA_lo, dA_hi = [0.0], [0.0], [0.0]
    dB, dB_lo, dB_hi = [0.0], [0.0], [0.0]

    for i, rung in enumerate(LADDER):
        fname, key = rung["delta_src"]
        m_a, lo_a, hi_a, m_b, lo_b, hi_b, gate = _delta_triplet(_load(fname), key)
        da.append(m_a); da_lo.append(lo_a); da_hi.append(hi_a)
        db.append(m_b); db_lo.append(lo_b); db_hi.append(hi_b)
        gates.append(gate)
        if rung["d_src"] is not None:
            fname2, key2 = rung["d_src"]
            m_a, lo_a, hi_a, m_b, lo_b, hi_b = _d_triplet(_load(fname2), key2)
            dA.append(m_a); dA_lo.append(lo_a); dA_hi.append(hi_a)
            dB.append(m_b); dB_lo.append(lo_b); dB_hi.append(hi_b)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(TWO_COLUMN, 3.0))

    # --- left: absolute specialization ---
    ax0.axhline(0.0, color=COLORS["zero"], lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax0.errorbar(
        xs - 0.06, da, yerr=[da_lo, da_hi], color=COLORS["A"], ls=LINESTYLES["A"],
        marker=MARKERS["A"], ms=5, mec="white", mew=0.5, lw=1.3,
        capsize=3, elinewidth=0.8, label=r"$\Delta_A$", zorder=2,
    )
    ax0.errorbar(
        xs + 0.06, db, yerr=[db_lo, db_hi], color=COLORS["B"], ls=LINESTYLES["B"],
        marker=MARKERS["B"], ms=5, mec="white", mew=0.5, lw=1.3,
        capsize=3, elinewidth=0.8, label=r"$\Delta_B$", zorder=2,
    )
    ax0.set_xticks(xs)
    ax0.set_xticklabels([r["label"] for r in LADDER], fontsize=8)
    for i, rung in enumerate(LADDER):
        ax0.text(i, -22, rung["sub"], ha="center", va="top", fontsize=6.5, color="#555555")
    ax0.set_ylim(-28, 50)
    ax0.set_ylabel(r"Specialization $\Delta$ (pp)")
    ax0.set_title("Does the crossover gate survive?", fontsize=9.5, fontweight="bold")
    ax0.text(
        0.5, 0.02, "gate " + "·".join("P" if g == "PASS" else "F" for g in gates),
        transform=ax0.transAxes, ha="center", va="bottom", fontsize=7.5, color="#444444",
    )
    ax0.legend(loc="upper right", frameon=False, fontsize=8)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    # --- right: paired loss vs Share-0 ---
    ax1.axhline(0.0, color=COLORS["zero"], lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax1.errorbar(
        xs - 0.06, dA, yerr=[dA_lo, dA_hi], color=COLORS["A"], ls=LINESTYLES["A"],
        marker=MARKERS["A"], ms=5, mec="white", mew=0.5, lw=1.3,
        capsize=3, elinewidth=0.8, label=r"$D_A$ vs Share-0", zorder=2,
    )
    ax1.errorbar(
        xs + 0.06, dB, yerr=[dB_lo, dB_hi], color=COLORS["B"], ls=LINESTYLES["B"],
        marker=MARKERS["B"], ms=5, mec="white", mew=0.5, lw=1.3,
        capsize=3, elinewidth=0.8, label=r"$D_B$ vs Share-0", zorder=2,
    )
    ax1.set_xticks(xs)
    ax1.set_xticklabels([r["label"] for r in LADDER], fontsize=8)
    ax1.set_ylim(-40, 20)
    ax1.set_ylabel(r"Paired change $D$ (pp)")
    ax1.set_title(r"Detectable loss iff UCB$_{95}(D)<0$", fontsize=9.5, fontweight="bold")
    ax1.legend(loc="lower left", frameon=False, fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    caption = (
        r"Progressive sharing with everything else held fixed (experts, data, schedule, "
        r"matched 128 seeds, specialization criterion). Share-Encoder preserves the gate; "
        r"Share-Backbone / Share-Macro show detectable Pole-A loss vs Share-0. "
        r"P=PASS, F=FAIL."
    )
    fig.text(0.5, -0.04, caption, ha="center", va="top", fontsize=7.5, style="italic")
    fig.subplots_adjust(wspace=0.28, bottom=0.22, top=0.88)

    paths = save_figure(fig, "fig_sharing_ladder_2v2")
    print(paths)
    return paths


if __name__ == "__main__":
    main()
