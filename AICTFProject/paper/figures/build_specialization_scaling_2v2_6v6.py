"""Closed-loop specialization across team sizes (Results central scaling figure).

Four conditions on one axis:

  2v2 Share-0  |  2v2 Share-Encoder  |  6v6 Share-0 teachers  |  6v6 Share-Encoder

Exact sealed Δ_A / Δ_B with paired 95% bootstrap CIs. No estimated values.

Run:  ./.venv/Scripts/python.exe paper/figures/build_specialization_scaling_2v2_6v6.py
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

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
STEM = "fig_specialization_scaling_2v2_6v6"


def _trip(block: dict) -> tuple[float, float, float]:
    return float(block["mean"]), float(block["lcb95"]), float(block["ucb95"])


def _gate(da: dict, db: dict, passes_key=None) -> str:
    if passes_key is not None:
        return "PASS" if passes_key else "FAIL"
    ok = (
        da["mean"] > 0 and da["lcb95"] > 0
        and db["mean"] > 0 and db["lcb95"] > 0
    )
    return "PASS" if ok else "FAIL"


def load_conditions() -> list[dict]:
    r0 = json.loads((SD / "RUNG0_LADDER_REFERENCE.json").read_text(encoding="utf-8"))
    r1 = json.loads((SD / "RUNG1_LADDER_EVAL_RESULT.json").read_text(encoding="utf-8"))
    t0 = json.loads(
        (SD / "SHARE0_6V6_TEACHER_SPECIALIST_CROSSOVER_EVAL_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    s1 = json.loads((SD / "RUNG1_6V6_CROSSOVER_EVAL_RESULT.json").read_text(encoding="utf-8"))

    g0 = r0["POOLED_N128"]
    g1 = r1["OWN_GATE_N128"]
    gt = t0["PRIMARY_GATE"]
    gs = s1["PRIMARY_GATE"]

    return [
        {
            "label": "2v2\nShare-0",
            "dA": _trip(g0["delta_A"]),
            "dB": _trip(g0["delta_B"]),
            "gate": _gate(g0["delta_A"], g0["delta_B"], g0.get("passes")),
        },
        {
            "label": "2v2\nShare-Encoder",
            "dA": _trip(g1["delta_A"]),
            "dB": _trip(g1["delta_B"]),
            "gate": _gate(g1["delta_A"], g1["delta_B"], g1.get("passes")),
        },
        {
            "label": "6v6\nShare-0 teachers",
            "dA": _trip(gt["delta_A"]),
            "dB": _trip(gt["delta_B"]),
            "gate": _gate(gt["delta_A"], gt["delta_B"], gt.get("passes")),
        },
        {
            "label": "6v6\nShare-Encoder",
            "dA": _trip(gs["delta_A"]),
            "dB": _trip(gs["delta_B"]),
            "gate": _gate(gs["delta_A"], gs["delta_B"], gs.get("passes")),
        },
    ]


def main() -> int:
    apply_style()
    conds = load_conditions()
    xs = np.arange(len(conds))

    fig, ax = plt.subplots(figsize=(TWO_COLUMN, 2.7))
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
        ax.text(
            i, -0.20, c["gate"],
            ha="center", va="top", fontsize=8,
            color="#1a7f37" if c["gate"] == "PASS" else "#b00020",
            fontweight="bold",
        )

    ax.axhline(0.0, color=COLORS["zero"], linewidth=0.8, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([c["label"] for c in conds])
    ax.set_xlim(-0.55, len(conds) - 0.45)
    ax.set_ylim(-0.28, 0.48)
    ax.set_ylabel(r"specialization contrast $\Delta$")
    ax.plot([], [], "o", color=COLORS["A"], label=r"$\Delta_A$")
    ax.plot([], [], "s", color=COLORS["B"], label=r"$\Delta_B$")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    report = save_figure(fig, STEM)
    print(f"wrote: {report}")
    for c in conds:
        print(f"  {c['label'].replace(chr(10), ' ')}: "
              f"dA={c['dA'][0]:+.4f} dB={c['dB'][0]:+.4f} {c['gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
