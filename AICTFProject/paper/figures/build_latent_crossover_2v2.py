"""2v2 latent-strategy crossover: one shared policy pi_theta(a | o, z).

This is NOT the specialist figure. Specialists are separate policies (pi_A, pi_B).
Here a single shared latent-conditioned policy is evaluated under forced z; only the
strategy code changes:

  z0 -> strategy suited to Pole A
  z1 -> strategy suited to Pole B

Source: sealed Rung-1 sharing-ladder eval (RUNG1_LADDER_EVAL_RESULT.json /
rung1_ladder_eval_rows.csv). Own-gate PASS at n=128 matched seeds.

Win rates and asymmetric 95% CI whiskers recomputed with the project-frozen
bootstrap (n_boot=20000, alpha=0.05, rng_seed=7).

Run:  python paper/figures/build_latent_crossover_2v2.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.figures.figure_style import COLORS, ONE_COLUMN, apply_style, save_figure

CSV = ROOT / "artifacts/strategic_demand/sppo/rung1_ladder_eval_rows.csv"
MODE_COLOR = {"z0": COLORS["A"], "z1": COLORS["B"]}
MODE_LABEL = {
    "z0": r"$z_0$ (Pole-A strategy code)",
    "z1": r"$z_1$ (Pole-B strategy code)",
}


def _load_cells(path: Path) -> dict[tuple[str, str], np.ndarray]:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[tuple[str, str], list[int]] = {}
    for r in rows:
        out.setdefault((r["z"], r["pole"]), []).append(int(r["win"]))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def _mean_ci_pct(wins: np.ndarray, n_boot=20000, alpha=0.05, rng_seed=7):
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, wins.size, size=(n_boot, wins.size))
    boots = wins[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    mean = float(wins.mean())
    return mean * 100, max(0.0, (mean - lo) * 100), max(0.0, (hi - mean) * 100)


def main() -> dict:
    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(ONE_COLUMN + 0.6, 3.15))
    cells = _load_cells(CSV)

    group_x = {"A": 0.0, "B": 1.0}
    mode_dx = {"z0": -0.19, "z1": 0.19}
    bar_w = 0.34

    for pole in ("A", "B"):
        for z in ("z0", "z1"):
            wins = cells[(z, pole)]
            value, err_lo, err_hi = _mean_ci_pct(wins)
            x = group_x[pole] + mode_dx[z]
            ax.bar(
                x, value, yerr=[[err_lo], [err_hi]], width=bar_w,
                color=MODE_COLOR[z], edgecolor="black", linewidth=0.6,
                capsize=3, error_kw={"elinewidth": 0.8, "capthick": 0.8},
            )
            ax.text(
                x, value + err_hi + 2.0, f"{value:.0f}%",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks([group_x["A"], group_x["B"]])
    ax.set_xticklabels(["vs Pole A", "vs Pole B"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Win rate (%)")
    ax.set_title(
        r"2v2 latent crossover: shared $\pi_\theta(a\mid o,z)$ (PASS)",
        fontsize=9.5, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        mpatches.Patch(facecolor=MODE_COLOR[z], edgecolor="black", label=MODE_LABEL[z])
        for z in ("z0", "z1")
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03),
        ncol=1, frameon=False, fontsize=8,
    )
    caption = (
        r"One shared policy; only the strategy code $z$ changes. "
        r"Desired: $z_0$ taller under Pole A and $z_1$ taller under Pole B. "
        r"n=128 matched seeds; 95\% bootstrap CIs. "
        r"Source: RUNG1\_LADDER\_EVAL\_RESULT.json (not the specialist ceiling)."
    )
    fig.text(0.5, -0.04, caption, ha="center", va="top", fontsize=7.5, style="italic")
    fig.subplots_adjust(bottom=0.22, top=0.76)

    paths = save_figure(fig, "fig_latent_crossover_2v2")
    print(paths)
    return paths


if __name__ == "__main__":
    main()
