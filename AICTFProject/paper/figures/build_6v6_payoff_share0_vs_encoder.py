"""6v6 absolute payoff: Share-0 teachers vs Share-Encoder student.

Two panels (Regime A | Regime B). Within each panel, four sealed cell means
with paired bootstrap CIs over the evaluation seeds:

  pi_A / pi_B  (Share-0 teacher diagnostic, seeds 13680001-128)
  z0 / z1      (Share-Encoder student, seeds 13640001-128)

Not a paired within-seed D plot across conditions (different seed blocks);
absolute WR context that makes the flat teacher Δ and high student competence
visible alongside the sealed specialization gates.

Run:  ./.venv/Scripts/python.exe paper/figures/build_6v6_payoff_share0_vs_encoder.py
"""
from __future__ import annotations

import csv
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
STEM = "fig_6v6_payoff_share0_vs_encoder"

ORDER = ("pi_A", "pi_B", "z0", "z1")
TICKS = [r"$\pi_A$", r"$\pi_B$", r"$z_0$", r"$z_1$"]
STYLE = {
    "pi_A": {"color": COLORS["A"], "hatch": "", "label": r"Share-0 $\pi_A$"},
    "pi_B": {"color": COLORS["B"], "hatch": "", "label": r"Share-0 $\pi_B$"},
    "z0": {"color": COLORS["A"], "hatch": "///", "label": r"Share-Encoder $z_0$"},
    "z1": {"color": COLORS["B"], "hatch": "///", "label": r"Share-Encoder $z_1$"},
}


def _mean_ci_pct(wins: np.ndarray, n_boot=20000, alpha=0.05, rng_seed=7):
    rng = np.random.default_rng(rng_seed)
    boots = wins[rng.integers(0, wins.size, size=(n_boot, wins.size))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    mean = float(wins.mean())
    return mean * 100, max(0.0, (mean - lo) * 100), max(0.0, (hi - mean) * 100)


def _load_cells() -> dict:
    t_csv = SD / "share0_6v6_teacher_specialist_crossover_eval_rows.csv"
    s_csv = SD / "rung1_6v6_crossover_eval_rows.csv"
    out = {}
    with t_csv.open(encoding="utf-8") as fh:
        trows = list(csv.DictReader(fh))
    with s_csv.open(encoding="utf-8") as fh:
        srows = list(csv.DictReader(fh))
    for pole in ("A", "B"):
        for pol in ("pi_A", "pi_B"):
            wins = np.asarray(
                [int(r["win"]) for r in trows if r["policy"] == pol and r["pole"] == pole],
                dtype=float,
            )
            out[(pol, pole)] = _mean_ci_pct(wins)
        for z, name in (("0", "z0"), ("1", "z1")):
            wins = np.asarray(
                [int(r["win"]) for r in srows if r["z"] == z and r["pole"] == pole],
                dtype=float,
            )
            out[(name, pole)] = _mean_ci_pct(wins)
    return out


def _gate_banner() -> str:
    t = json.loads(
        (SD / "SHARE0_6V6_TEACHER_SPECIALIST_CROSSOVER_EVAL_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    s = json.loads((SD / "RUNG1_6V6_CROSSOVER_EVAL_RESULT.json").read_text(encoding="utf-8"))
    gt, gs = t["PRIMARY_GATE"], s["PRIMARY_GATE"]
    return (
        f"Share-0 teachers: FAIL  "
        f"$\\Delta_A$={gt['delta_A']['mean']:+.3f}, "
        f"$\\Delta_B$={gt['delta_B']['mean']:+.3f}   |   "
        f"Share-Encoder: FAIL  "
        f"$\\Delta_A$={gs['delta_A']['mean']:+.3f}, "
        f"$\\Delta_B$={gs['delta_B']['mean']:+.3f}"
    )


def main() -> int:
    apply_style()
    cells = _load_cells()
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COLUMN, 2.9), sharey=True)
    xs = np.arange(len(ORDER))
    bar_w = 0.72

    for ax, pole, letter in ((axes[0], "A", "a"), (axes[1], "B", "b")):
        for i, pol in enumerate(ORDER):
            mean, elo, ehi = cells[(pol, pole)]
            st = STYLE[pol]
            ax.bar(
                xs[i], mean, yerr=[[elo], [ehi]], width=bar_w,
                color=st["color"], hatch=st["hatch"], edgecolor="black",
                linewidth=0.6, capsize=2.5,
                error_kw={"elinewidth": 0.8, "capthick": 0.8},
            )
            ax.text(
                xs[i], mean + ehi + 1.2, f"{mean:.0f}",
                ha="center", va="bottom", fontsize=7.5,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(TICKS, fontsize=9)
        ax.set_ylim(0, 110)
        ax.set_title(f"({letter}) Regime {pole}", loc="left")
        ax.set_ylabel("win rate (%)" if pole == "A" else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=STYLE[p]["color"],
                      hatch=STYLE[p]["hatch"], edgecolor="black", linewidth=0.6,
                      label=STYLE[p]["label"])
        for p in ORDER
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, 1.02), fontsize=8,
    )
    fig.suptitle(_gate_banner(), y=1.12, fontsize=8)
    fig.tight_layout()
    report = save_figure(fig, STEM)
    print(f"wrote: {report}")
    for pole in ("A", "B"):
        print(f"  Regime {pole}:",
              {p: round(cells[(p, pole)][0], 1) for p in ORDER})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
