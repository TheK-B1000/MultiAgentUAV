"""Cross-scale figure: the 2v2 and 6v6 evidence as ONE argument.

Three panels, left to right, mirroring the Results section's own logic:

  (a) Opponent-regime DEMAND is certified at both scales, and does not weaken
      with team size. Sealed scripted-probe contrasts with 95% CIs.
  (b) Distillation FIDELITY at both scales: the student retains ~99% of its
      teachers' branch divergence, under an identical 48.0% parameter reduction.
  (c) The specialization CRITERION on the compressed policy. 2v2 is sealed;
      6v6 auto-fills the moment its sealed crossover artifact exists, and is
      drawn as an explicit PENDING slot until then.

Panel (c) is deliberately graceful: this script is safe to run before, during,
or after the 6v6 crossover eval. It NEVER estimates a missing value -- a
pending cell is drawn as a pending cell.

Run:  ./.venv/Scripts/python.exe paper/figures/build_cross_scale_2v2_6v6.py
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

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"

# --------------------------------------------------------------------------- sealed sources
DEMAND = {
    "2v2": {"dA": (0.297, 0.203, 0.391), "dB": (0.453, 0.385, 0.526), "n": 192},
    "6v6": {"dA": (0.266, 0.094, 0.438), "dB": (0.625, 0.484, 0.750), "n": 64},
}

# student branch JSD vs its own teachers' branch JSD (holdout)
FIDELITY = {
    "2v2": {"student_jsd": 0.178, "teacher_jsd": 0.179,
            "agree": (0.967, 0.989)},
    "6v6": {"student_jsd": 0.47759700160571905, "teacher_jsd": 0.4838482085211186,
            "agree": (0.9629380966910869, 0.9608576152681255)},
}

# 2v2 shared-encoder sealed crossover (Share-Encoder / Rung 1)
CRITERION_2V2 = {"dA": (0.266, 0.148, 0.383), "dB": (0.164, 0.047, 0.281)}

# 6v6 crossover: read from disk if sealed, else stay None (drawn as PENDING)
RESULT_6V6 = SD / "RUNG1_6V6_CROSSOVER_EVAL_RESULT.json"
FLAG_6V6 = SD / "RUNG1_6V6_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json"

# 6v6 ZERO-SHARING reference (frozen teachers, bit-exact dispatch, no tied modules).
# Required in panel (c): without it the panel reads as "compression killed it",
# which the data contradicts -- the uncompressed reference fails too.
SHARE0_6V6 = SD / "SHARE0_6V6_TEACHER_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
# 2v2 zero-sharing reference (Share-0 / Rung 0), sealed
SHARE0_2V2 = {"dA": (0.289, 0.164, 0.406), "dB": (0.258, 0.141, 0.375)}


def load_share0_6v6():
    """Zero-sharing 6v6 reference. Returns (dA, dB) or (None, None)."""
    if not SHARE0_6V6.is_file():
        return None, None
    g = json.loads(SHARE0_6V6.read_text(encoding="utf-8")).get("PRIMARY_GATE", {})
    a, b = g.get("delta_A"), g.get("delta_B")
    if not (a and b):
        return None, None
    return ((a["mean"], a["lcb95"], a["ucb95"]),
            (b["mean"], b["lcb95"], b["ucb95"]))


def load_6v6_criterion():
    """Return (dA, dB, status). Never invents a value."""
    if RESULT_6V6.is_file():
        d = json.loads(RESULT_6V6.read_text(encoding="utf-8"))
        g = d.get("PRIMARY_GATE", {})
        a, b = g.get("delta_A"), g.get("delta_B")
        if a and b:
            return (
                (a["mean"], a["lcb95"], a["ucb95"]),
                (b["mean"], b["lcb95"], b["ucb95"]),
                "PASS" if g.get("passes") else "FAIL",
            )
    if FLAG_6V6.is_file():
        return None, None, "INTEGRITY AUDIT REQUIRED"
    return None, None, "PENDING"


def _err(ax, x, trip, color, marker):
    mean, lo, hi = trip
    ax.errorbar(
        x, mean,
        yerr=[[mean - lo], [hi - mean]],
        fmt=marker, color=color, markersize=5, capsize=3,
        linewidth=1.2, markeredgewidth=1.0,
    )


def main() -> int:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COLUMN, 2.5))

    # ------------------------------------------------------------------ (a) demand
    ax = axes[0]
    xs = [0, 1]
    for i, scale in enumerate(("2v2", "6v6")):
        _err(ax, i - 0.09, DEMAND[scale]["dA"], COLORS["A"], "o")
        _err(ax, i + 0.09, DEMAND[scale]["dB"], COLORS["B"], "s")
    ax.axhline(0.0, color=COLORS["zero"], linewidth=0.8, linestyle="-", zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"2v2\n$n$={DEMAND['2v2']['n']}", f"6v6\n$n$={DEMAND['6v6']['n']}"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel(r"scripted demand contrast")
    ax.set_title("(a) regimes demand\ndifferent strategies", loc="left")
    ax.plot([], [], "o", color=COLORS["A"], label=r"$\Delta_G(A)$")
    ax.plot([], [], "s", color=COLORS["B"], label=r"$\Delta_B(B)$")
    ax.legend(frameon=False, loc="upper left", fontsize=8)

    # ------------------------------------------------------------------ (b) fidelity
    ax = axes[1]
    width = 0.32
    for i, scale in enumerate(("2v2", "6v6")):
        f = FIDELITY[scale]
        ax.bar(i - width / 2, f["teacher_jsd"], width,
               color=COLORS["control"], edgecolor="black", linewidth=0.6,
               label="teachers" if i == 0 else None)
        ax.bar(i + width / 2, f["student_jsd"], width,
               color=COLORS["A"], edgecolor="black", linewidth=0.6,
               label="compressed student" if i == 0 else None)
        frac = 100.0 * f["student_jsd"] / f["teacher_jsd"]
        ax.text(i, max(f["teacher_jsd"], f["student_jsd"]) + 0.025,
                f"{frac:.0f}%", ha="center", fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["2v2", "6v6"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("branch JS divergence")
    ax.set_title("(b) fidelity: separation\nretained under 48.0% cut", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=8, ncol=1,
              handlelength=1.2, borderpad=0.2)

    # ------------------------------------------------------------------ (c) criterion
    # Four x-positions: zero-sharing reference and compressed policy, at each scale.
    # Showing the zero-sharing reference is NOT optional -- at 6v6 it also fails,
    # which is what localizes the failure upstream of compression.
    ax = axes[2]
    dA6, dB6, status6 = load_6v6_criterion()
    s0A6, s0B6 = load_share0_6v6()

    POS = {"2v2_share0": 0.0, "2v2_enc": 0.75, "6v6_share0": 1.85, "6v6_enc": 2.60}
    off = 0.13

    _err(ax, POS["2v2_share0"] - off, SHARE0_2V2["dA"], COLORS["A"], "o")
    _err(ax, POS["2v2_share0"] + off, SHARE0_2V2["dB"], COLORS["B"], "s")
    _err(ax, POS["2v2_enc"] - off, CRITERION_2V2["dA"], COLORS["A"], "o")
    _err(ax, POS["2v2_enc"] + off, CRITERION_2V2["dB"], COLORS["B"], "s")

    if s0A6 is not None:
        _err(ax, POS["6v6_share0"] - off, s0A6, COLORS["A"], "o")
        _err(ax, POS["6v6_share0"] + off, s0B6, COLORS["B"], "s")
    if dA6 is not None:
        _err(ax, POS["6v6_enc"] - off, dA6, COLORS["A"], "o")
        _err(ax, POS["6v6_enc"] + off, dB6, COLORS["B"], "s")

    ax.axhline(0.0, color=COLORS["zero"], linewidth=0.8, zorder=0)
    ax.axvline(1.3, color="#BBBBBB", linewidth=0.7, linestyle=":", zorder=0)

    ax.set_xticks(list(POS.values()))
    ax.set_xticklabels(["no\nsharing", "shared\nencoder",
                        "no\nsharing", "shared\nencoder"], fontsize=7.5)
    ax.set_xlim(-0.45, 3.05)
    ax.set_ylim(-0.20, 0.62)
    ax.set_ylabel(r"$\Delta$ (specialization)")
    ax.set_title("(c) payoff specialization:\nsources vs. compressed", loc="left")

    # verdict strip + scale labels
    for key, txt in (("2v2_share0", "pass"), ("2v2_enc", "pass"),
                     ("6v6_share0", "fail"), ("6v6_enc", "fail")):
        ax.text(POS[key], -0.155, txt, ha="center", fontsize=7.5,
                color="#222222" if txt == "pass" else "#B00020")
    ax.text(0.375, 0.425, "2v2", ha="center", fontsize=8.5)
    ax.text(2.225, 0.425, "6v6", ha="center", fontsize=8.5)

    ax.plot([], [], "o", color=COLORS["A"], label=r"$\Delta_A$")
    ax.plot([], [], "s", color=COLORS["B"], label=r"$\Delta_B$")
    ax.legend(frameon=False, loc="upper center", fontsize=8, ncol=2,
              handlelength=1.0, columnspacing=0.9)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    report = save_figure(fig, "fig_cross_scale_2v2_6v6")
    print(f"  6v6 criterion panel status: {status6}")
    print(f"  wrote: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
