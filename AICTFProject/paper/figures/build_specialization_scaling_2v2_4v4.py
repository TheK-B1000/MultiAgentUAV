"""Specialization contrast: 2v2 PASS vs 4v4 scaffolded PASS*.

Two conditions only (exact sealed Δ with paired 95% CIs):

  2v2 Share-0  |  4v4 A' scaffold

Native 4v4 FAIL is intentionally omitted — this figure compares the two
conditions that clear the joint gate. PASS* marks the scaffold as a
controller result, not a learned crossover.

Run:  ./.venv/Scripts/python.exe paper/figures/build_specialization_scaling_2v2_4v4.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.figures.figure_style import COLORS, ONE_COLUMN, apply_style, save_figure

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
STEM = "fig_specialization_scaling_2v2_4v4"
COMBINED = ROOT / "paper" / "plots" / "combined"


def _trip(block: dict) -> tuple[float, float, float]:
    return float(block["mean"]), float(block["lcb95"]), float(block["ucb95"])


def load_conditions() -> list[dict]:
    r0 = json.loads((SD / "RUNG0_LADDER_REFERENCE.json").read_text(encoding="utf-8"))
    scaffold = json.loads(
        (SD / "SCAFFOLDED_A_CROSSOVER_BRIDGE_4V4_RESULT.json").read_text(encoding="utf-8")
    )
    g0 = r0["POOLED_N128"]
    gs = scaffold["PRIMARY_WIN_RATE_CONTRASTS"]
    return [
        {
            "label": "2v2\nShare-0",
            "dA": _trip(g0["delta_A"]),
            "dB": _trip(g0["delta_B"]),
            "gate": "PASS",
        },
        {
            "label": "4v4\nA' scaffold",
            "dA": _trip(gs["Delta_A_prime"]),
            "dB": _trip(gs["Delta_B_prime"]),
            "gate": "PASS*",
        },
    ]


def main() -> int:
    apply_style()
    conds = load_conditions()
    xs = np.arange(len(conds))

    fig, ax = plt.subplots(figsize=(ONE_COLUMN + 1.2, 2.7))
    for i, c in enumerate(conds):
        for trip, color, marker, dx in (
            (c["dA"], COLORS["A"], "o", -0.12),
            (c["dB"], COLORS["B"], "s", +0.12),
        ):
            mean, lo, hi = trip
            ax.errorbar(
                i + dx, mean,
                yerr=[[mean - lo], [hi - mean]],
                fmt=marker, color=color, markersize=5.5, capsize=3,
                linewidth=1.2, markeredgewidth=1.0, zorder=3,
            )
        gate_color = "#1a7f37" if c["gate"] == "PASS" else "#b07000"
        ax.text(
            i, -0.08, c["gate"],
            ha="center", va="top", fontsize=8,
            color=gate_color, fontweight="bold",
        )

    ax.axhline(0.0, color=COLORS["zero"], linewidth=0.8, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([c["label"] for c in conds])
    ax.set_xlim(-0.55, len(conds) - 0.45)
    ax.set_ylim(-0.12, 0.55)
    ax.set_ylabel(r"specialization contrast $\Delta$")
    ax.plot([], [], "o", color=COLORS["A"], label=r"$\Delta_A$")
    ax.plot([], [], "s", color=COLORS["B"], label=r"$\Delta_B$")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    report = save_figure(fig, STEM)

    COMBINED.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        src = ROOT / "paper" / "generated" / f"{STEM}.{ext}"
        shutil.copy2(src, COMBINED / f"{STEM}.{ext}")

    manifest = {
        "stem": STEM,
        "role": "specialization_contrast_2v2_PASS_vs_4v4_scaffold_PASS",
        "note": (
            "Compares only gate-clearing conditions. PASS* = scaffolded controller "
            "(A'=pi_A+2D), not a learned crossover. Native 4v4 FAIL omitted."
        ),
        "conditions": [
            {
                "label": c["label"].replace("\n", " "),
                "delta_A": {"mean": c["dA"][0], "lcb95": c["dA"][1], "ucb95": c["dA"][2]},
                "delta_B": {"mean": c["dB"][0], "lcb95": c["dB"][1], "ucb95": c["dB"][2]},
                "gate": c["gate"],
            }
            for c in conds
        ],
        "sources": {
            "2v2_Share-0": "RUNG0_LADDER_REFERENCE.json#POOLED_N128",
            "4v4_A_prime": "SCAFFOLDED_A_CROSSOVER_BRIDGE_4V4_RESULT.json",
        },
    }
    man_path = ROOT / "paper" / "data" / "specialization_scaling_2v2_4v4.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"wrote: {report}")
    for c in conds:
        print(
            f"  {c['label'].replace(chr(10), ' ')}: "
            f"dA={c['dA'][0]:+.4f} dB={c['dB'][0]:+.4f} {c['gate']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
