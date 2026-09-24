"""ICRA killer three-panel: strategies exist -> transfer into z -> sharing erodes them.

  (a) Specialist crossover  pi_A vs pi_B
  (b) Distilled latent crossover  z0 vs z1 (Share-Encoder / Rung 1)
  (c) Progressive sharing  sealed Delta_A/Delta_B with 95% CIs + gate labels

Panel (c) uses the same sealed rung records as fig_sharing_ladder_2v2 (absolute
gate + annotated means). The full D_A/D_B paired-loss panel lives in that
companion figure so this main figure stays readable in 8 pages.

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
    ("Share-0", "S0", "RUNG0_LADDER_REFERENCE.json", "POOLED_N128"),
    ("Share-Encoder", "Enc", "RUNG1_LADDER_EVAL_RESULT.json", "OWN_GATE_N128"),
    ("Share-Backbone", "Back", "RUNG2_LADDER_EVAL_RESULT.json", "OWN_GATE_N128"),
    ("Share-Macro", "Macro", "RUNG3_LADDER_EVAL_RESULT.json", "OWN_GATE_N128"),
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


def _gate(g: dict, a: dict, b: dict) -> str:
    if "passes" in g and isinstance(g["passes"], bool):
        return "PASS" if g["passes"] else "FAIL"
    if "passes" in a and "passes" in b:
        return "PASS" if (a["passes"] and b["passes"]) else "FAIL"
    ok = a["mean"] > 0 and a["lcb95"] > 0 and b["mean"] > 0 and b["lcb95"] > 0
    return "PASS" if ok else "FAIL"


def _panel_specialists(ax) -> None:
    path = SD / "specialist_baseline_eval_rows.csv"
    for pole_i, pole in enumerate(("A", "B")):
        for pol, dx in (("pi_A", -0.18), ("pi_B", 0.18)):
            wins = _csv_wins(path, lambda r, p=pol, po=pole: r["policy"] == p and r["pole"] == po)
            m, lo, hi = _mean_ci_pct(wins)
            x = pole_i + dx
            ax.bar(
                x, m, yerr=[[lo], [hi]], width=0.32,
                color=COLORS["A" if pol == "pi_A" else "B"],
                edgecolor="black", linewidth=0.55, capsize=2,
                error_kw={"elinewidth": 0.7},
            )
            ax.text(x, m + hi + 1.5, f"{m:.0f}", ha="center", fontsize=7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pole A", "Pole B"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Win rate (%)")
    ax.set_title(r"(a) Strategies exist: $\pi_A,\pi_B$", fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_latent(ax) -> None:
    """Clean distilled latent X only (z0 vs z1) -- the transfer claim."""
    z = SD / "rung1_ladder_eval_rows.csv"
    for pole_i, pole in enumerate(("A", "B")):
        for zi, dx, color in (("z0", -0.18, COLORS["A"]), ("z1", 0.18, COLORS["B"])):
            wins = _csv_wins(z, lambda r, zz=zi, po=pole: r["z"] == zz and r["pole"] == po)
            m, lo, hi = _mean_ci_pct(wins)
            x = pole_i + dx
            ax.bar(
                x, m, yerr=[[lo], [hi]], width=0.32, color=color, hatch="///",
                edgecolor="black", linewidth=0.55, capsize=2,
                error_kw={"elinewidth": 0.7},
            )
            ax.text(x, m + hi + 1.5, f"{m:.0f}", ha="center", fontsize=7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pole A", "Pole B"])
    ax.set_ylim(0, 100)
    ax.set_title(r"(b) Transfer into $z$: $z_0,z_1$", fontsize=9, fontweight="bold")
    ax.text(
        0.5, 0.95, r"$z_0$: 70/44   $z_1$: 44/60",
        transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#333",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_sharing(ax) -> None:
    xs = np.arange(len(SHARE))
    da, da_lo, da_hi, db, db_lo, db_hi, gates = [], [], [], [], [], [], []
    for _lab, _short, fname, key in SHARE:
        g = json.loads((SD / fname).read_text(encoding="utf-8"))[key]
        a, b = g["delta_A"], g["delta_B"]
        da.append(a["mean"] * 100)
        da_lo.append((a["mean"] - a["lcb95"]) * 100)
        da_hi.append((a["ucb95"] - a["mean"]) * 100)
        db.append(b["mean"] * 100)
        db_lo.append((b["mean"] - b["lcb95"]) * 100)
        db_hi.append((b["ucb95"] - b["mean"]) * 100)
        gates.append(_gate(g, a, b))

    ax.axhline(0.0, color=COLORS["zero"], lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.axhspan(-8, 0, color="#FFEBEE", alpha=0.5, zorder=0)
    ax.errorbar(
        xs - 0.06, da, yerr=[da_lo, da_hi], color=COLORS["A"], ls=LINESTYLES["A"],
        marker=MARKERS["A"], ms=5, mec="white", mew=0.4, lw=1.3,
        capsize=2.5, elinewidth=0.8, label=r"$\Delta_A$", zorder=2,
    )
    ax.errorbar(
        xs + 0.06, db, yerr=[db_lo, db_hi], color=COLORS["B"], ls=LINESTYLES["B"],
        marker=MARKERS["B"], ms=5, mec="white", mew=0.4, lw=1.3,
        capsize=2.5, elinewidth=0.8, label=r"$\Delta_B$", zorder=2,
    )
    for i, g in enumerate(gates):
        color = "#2E7D32" if g == "PASS" else "#C62828"
        ax.text(i, -4.5, g, ha="center", va="top", fontsize=7, fontweight="bold", color=color)
    ax.set_xticks(xs)
    ax.set_xticklabels([s[1] for s in SHARE], fontsize=8)
    ax.set_ylim(-10, 48)
    ax.set_ylabel(r"$\Delta$ (pp)")
    ax.set_title(r"(c) Sharing erodes $\Delta$ (95\% CI)", fontsize=9, fontweight="bold")
    ax.text(
        0.5, 1.02, "preserved → preserved → degraded → failed",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5, style="italic",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> dict:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COLUMN, 2.95))
    _panel_specialists(axes[0])
    _panel_latent(axes[1])
    _panel_sharing(axes[2])

    handles = [
        mpatches.Patch(facecolor=COLORS["A"], edgecolor="black", label=r"$\pi_A$ / $z_0$"),
        mpatches.Patch(facecolor=COLORS["B"], edgecolor="black", label=r"$\pi_B$ / $z_1$"),
        mpatches.Patch(facecolor=COLORS["A"], hatch="///", edgecolor="black", label=r"distilled $z$"),
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.05),
        ncol=3, frameon=False, fontsize=7.5,
    )

    caption = (
        r"Causal chain for Claim A. "
        r"(a)~Independent PPO experts establish complementary payoff strategies. "
        r"(b)~Matched-state distillation transfers them into discrete modes of one "
        r"Share-Encoder controller ($z_0\!\sim\!\pi_A$, $z_1\!\sim\!\pi_B$). "
        r"(c)~Sealed $\Delta_A,\Delta_B$ with 95\% CIs across progressive sharing "
        r"(n=128); Backbone still PASSes but is already degraded; Macro FAILs "
        r"(LCB$_{95}(\Delta_A)=0$). "
        r"Robotics takeaway: shared perception can be tolerable; strategy-specific "
        r"higher-level capacity is where specialization becomes fragile. "
        "Not an SP-PPO result (appendix ablation). "
        "Share-0 is expert dispatch (structural control); Enc→Back→Macro share one distill recipe."
    )
    fig.text(0.5, -0.10, caption, ha="center", va="top", fontsize=6.8, style="italic")
    fig.subplots_adjust(wspace=0.30, bottom=0.26, top=0.82, left=0.06, right=0.99)

    paths = save_figure(fig, "fig_claim_a_2v2")
    print(paths)
    return paths


if __name__ == "__main__":
    main()
