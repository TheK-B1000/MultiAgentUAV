"""Claim A main figure: specialists -> latent z-modes -> progressive sharing.

Three panels on one canvas so Claim A is visually unmistakable and separate
from Claim B (SP-PPO ablation):

  (a) Specialist crossover  pi_A/pi_B under Pole A/B
  (b) Latent strategy existence  pi_G, pi_A, pi_B, z0, z1 (Share-Encoder / Rung 1)
  (c) Sharing ladder  Delta_A / Delta_B across Share-0..Macro

Supports the lead sentence:
  a strategy-conditioned multi-robot controller can represent distinct
  payoff-relevant latent strategies and characterize how progressive
  parameter sharing affects their preservation.

Does NOT support: SP-PPO caused Share-Encoder PASS (see fig_claim_b_spp_ablation).

Run:  python paper/figures/build_claim_a_2v2.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.figures.figure_style import COLORS, LINESTYLES, MARKERS, TWO_COLUMN, apply_style, save_figure

SD = ROOT / "artifacts/strategic_demand/sppo"

SHARE = [
    ("Share-0", "RUNG0_LADDER_REFERENCE.json", "POOLED_N128"),
    ("Share-Encoder", "RUNG1_LADDER_EVAL_RESULT.json", "OWN_GATE_N128"),
    ("Share-Backbone", "RUNG2_LADDER_EVAL_RESULT.json", "OWN_GATE_N128"),
    ("Share-Macro", "RUNG3_LADDER_EVAL_RESULT.json", "OWN_GATE_N128"),
]


def _mean_ci_pct(wins: np.ndarray, n_boot=20000, alpha=0.05, rng_seed=7):
    rng = np.random.default_rng(rng_seed)
    boots = wins[rng.integers(0, wins.size, size=(n_boot, wins.size))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    mean = float(wins.mean())
    return mean * 100, max(0.0, (mean - lo) * 100), max(0.0, (hi - mean) * 100)


def _csv_wins(path: Path, pred) -> np.ndarray:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return np.asarray([int(r["win"]) for r in rows if pred(r)], dtype=float)


def _panel_specialists(ax) -> None:
    path = SD / "specialist_baseline_eval_rows.csv"
    group_x = {"A": 0.0, "B": 1.0}
    dx = {"pi_A": -0.18, "pi_B": 0.18}
    for pole in ("A", "B"):
        for pol in ("pi_A", "pi_B"):
            wins = _csv_wins(path, lambda r, p=pol, po=pole: r["policy"] == p and r["pole"] == po)
            m, lo, hi = _mean_ci_pct(wins)
            x = group_x[pole] + dx[pol]
            ax.bar(x, m, yerr=[[lo], [hi]], width=0.32,
                   color=COLORS["A" if pol == "pi_A" else "B"],
                   edgecolor="black", linewidth=0.55, capsize=2,
                   error_kw={"elinewidth": 0.7})
            ax.text(x, m + hi + 1.5, f"{m:.0f}", ha="center", fontsize=6.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pole A", "Pole B"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Win rate (%)")
    ax.set_title(r"(a) Specialist crossover", fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_latent(ax) -> None:
    g = SD / "pi_g_eval_rows.csv"
    s = SD / "specialist_baseline_eval_rows.csv"
    z = SD / "rung1_ladder_eval_rows.csv"
    order = [
        ("pi_G", g, lambda r, p: r["policy"] == "pi_G" and r["pole"] == p, COLORS["control"], ""),
        ("pi_A", s, lambda r, p: r["policy"] == "pi_A" and r["pole"] == p, COLORS["A"], ""),
        ("pi_B", s, lambda r, p: r["policy"] == "pi_B" and r["pole"] == p, COLORS["B"], ""),
        ("z0", z, lambda r, p: r["z"] == "z0" and r["pole"] == p, COLORS["A"], "///"),
        ("z1", z, lambda r, p: r["z"] == "z1" and r["pole"] == p, COLORS["B"], "///"),
    ]
    # two groups: Pole A at base 0, Pole B at base 6
    for gi, pole in enumerate(("A", "B")):
        base = gi * 6.0
        for i, (name, path, pred, color, hatch) in enumerate(order):
            wins = _csv_wins(path, lambda r, pr=pred, po=pole: pr(r, po))
            m, lo, hi = _mean_ci_pct(wins)
            x = base + i
            ax.bar(x, m, yerr=[[lo], [hi]], width=0.85, color=color, hatch=hatch,
                   edgecolor="black", linewidth=0.5, capsize=2,
                   error_kw={"elinewidth": 0.7})
            ax.text(x, m + hi + 1.2, f"{m:.0f}", ha="center", fontsize=5.5)
    ax.set_xticks([2, 8])
    ax.set_xticklabels(["Pole A", "Pole B"])
    ax.set_ylim(0, 100)
    ax.set_title(r"(b) Latent $z$-modes (+ baselines)", fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.annotate(r"$z_0\!:\!70/44$", xy=(0.22, 0.92), xycoords="axes fraction",
                fontsize=7, color=COLORS["A"])
    ax.annotate(r"$z_1\!:\!44/60$", xy=(0.72, 0.92), xycoords="axes fraction",
                fontsize=7, color=COLORS["B"])


def _panel_sharing(ax) -> None:
    xs = np.arange(len(SHARE))
    da, da_lo, da_hi, db, db_lo, db_hi, gates = [], [], [], [], [], [], []
    for _label, fname, key in SHARE:
        g = json.loads((SD / fname).read_text(encoding="utf-8"))[key]
        a, b = g["delta_A"], g["delta_B"]
        da.append(a["mean"] * 100)
        da_lo.append((a["mean"] - a["lcb95"]) * 100)
        da_hi.append((a["ucb95"] - a["mean"]) * 100)
        db.append(b["mean"] * 100)
        db_lo.append((b["mean"] - b["lcb95"]) * 100)
        db_hi.append((b["ucb95"] - b["mean"]) * 100)
        if "passes" in g:
            gates.append("PASS" if g["passes"] else "FAIL")
        else:
            gates.append("PASS" if (a.get("passes") and b.get("passes")) else "FAIL")

    ax.axhline(0.0, color=COLORS["zero"], lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.errorbar(xs - 0.05, da, yerr=[da_lo, da_hi], color=COLORS["A"], ls=LINESTYLES["A"],
                marker=MARKERS["A"], ms=4.5, mec="white", mew=0.4, lw=1.2,
                capsize=2.5, elinewidth=0.7, label=r"$\Delta_A$", zorder=2)
    ax.errorbar(xs + 0.05, db, yerr=[db_lo, db_hi], color=COLORS["B"], ls=LINESTYLES["B"],
                marker=MARKERS["B"], ms=4.5, mec="white", mew=0.4, lw=1.2,
                capsize=2.5, elinewidth=0.7, label=r"$\Delta_B$", zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(["S0", "Enc", "Back", "Macro"], fontsize=8)
    ax.set_ylim(-15, 45)
    ax.set_ylabel(r"$\Delta$ (pp)")
    ax.set_title("(c) Progressive sharing", fontsize=9, fontweight="bold")
    ax.text(0.5, 0.02, "gate " + "·".join("P" if g == "PASS" else "F" for g in gates),
            transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color="#444")
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> dict:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COLUMN, 2.85))
    _panel_specialists(axes[0])
    _panel_latent(axes[1])
    _panel_sharing(axes[2])

    handles = [
        mpatches.Patch(facecolor=COLORS["control"], edgecolor="black", label=r"$\pi_G$"),
        mpatches.Patch(facecolor=COLORS["A"], edgecolor="black", label=r"$\pi_A$ / $z_0$"),
        mpatches.Patch(facecolor=COLORS["B"], edgecolor="black", label=r"$\pi_B$ / $z_1$"),
        mpatches.Patch(facecolor=COLORS["A"], hatch="///", edgecolor="black", label=r"latent (hatched)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=4, frameon=False, fontsize=7.5)

    caption = (
        "Claim A only. "
        "(a) Independent specialists establish payoff-relevant strategies. "
        r"(b) Forced $z$ in one shared Share-Encoder policy reproduces the X-pattern "
        r"alongside $\pi_G/\pi_A/\pi_B$. "
        "(c) Sharing ladder: Encoder tolerated; Backbone degrades; Macro fails the gate. "
        "This figure does not attribute (b)--(c) to SP-PPO/CSC/SPFT "
        "(see fig_claim_b_spp_ablation)."
    )
    fig.text(0.5, -0.08, caption, ha="center", va="top", fontsize=7.0, style="italic")
    fig.subplots_adjust(wspace=0.28, bottom=0.24, top=0.82, left=0.06, right=0.99)

    paths = save_figure(fig, "fig_claim_a_2v2")
    print(paths)
    return paths


if __name__ == "__main__":
    main()
