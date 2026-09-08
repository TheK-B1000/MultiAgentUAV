"""2v2 deployment robustness dose-response (win rate + Delta), available tiers only.

Three panels:
  (a) localization noise  -- nominal (0) + medium (0.06) + high (0.12); low incomplete
  (b) motion error        -- same available ladder
  (c) control delay       -- nominal (0) + medium (2 ticks); low incomplete, high N/A

Each panel plots Mode A / Mode B win rates vs severity (lines + markers + 95% CI),
with a second twin showing Delta_A / Delta_B from the sealed result JSONs where present
is omitted here -- this figure answers "does the system still perform?" via win rate.
A companion Delta figure is produced in the same script as fig_robustness_delta_dose.

Bootstrap CIs recomputed from raw CSVs (n_boot=20000, alpha=0.05, rng_seed=7).

Run:  python paper/figures/build_robustness_dose_response.py
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

from paper.figures.figure_style import COLORS, LINESTYLES, MARKERS, TWO_COLUMN, apply_style, save_figure

D = ROOT / "artifacts" / "strategic_demand" / "sppo" / "robustness_eval_rows"
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"

# Matched-scale severity schedule from DEPLOYMENT_ROBUSTNESS_SPEC.json#TIERS
FAMILIES = [
    {
        "key": "localization_noise",
        "title": "Localization noise",
        "xlabel": r"Localization noise $\sigma$ (cells)",
        "points": [
            ("nominal", "nominal", 0.0),
            ("localization_noise", "medium", 0.06),
            ("localization_noise", "high", 0.12),
        ],
    },
    {
        "key": "motion_error",
        "title": "Motion error",
        "xlabel": r"Motion error $\sigma$ (cells)",
        "points": [
            ("nominal", "nominal", 0.0),
            ("motion_error", "medium", 0.06),
            ("motion_error", "high", 0.12),
        ],
    },
    {
        "key": "control_delay",
        "title": "Control delay",
        "xlabel": "Control delay (ticks)",
        "points": [
            ("nominal", "nominal", 0.0),
            ("control_delay", "medium", 2.0),
        ],
    },
]


def _load(pole: str, z: int, family: str, severity: str) -> np.ndarray:
    p = D / f"rung1_2v2__2v2__pole{pole}__z{z}__{family}__{severity}.csv"
    with p.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return np.array([int(r["win"]) for r in rows], dtype=float)


def _mean_ci_pct(wins: np.ndarray, n_boot=20000, alpha=0.05, rng_seed=7):
    rng = np.random.default_rng(rng_seed)
    boots = wins[rng.integers(0, wins.size, size=(n_boot, wins.size))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    mean = float(wins.mean())
    return mean * 100, max(0.0, (mean - lo) * 100), max(0.0, (hi - mean) * 100)


def _delta_ci(wins_on: np.ndarray, wins_off: np.ndarray, n_boot=20000, alpha=0.05, rng_seed=7):
    """Paired delta = on - off, in percent, with asymmetric 95% CI."""
    assert wins_on.shape == wins_off.shape
    d = wins_on - wins_off
    rng = np.random.default_rng(rng_seed)
    boots = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    mean = float(d.mean())
    return mean * 100, max(0.0, (mean - lo) * 100), max(0.0, (hi - mean) * 100)


def build_winrate_figure() -> dict:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COLUMN, 2.8), sharey=True)

    for ax, fam in zip(axes, FAMILIES):
        for mode, z, pole_for_own in (("A", 0, "A"), ("B", 1, "B")):
            xs, ys, elo, ehi = [], [], [], []
            for family, severity, x in fam["points"]:
                # Own-pole win rate of mode under that severity
                wins = _load(pole_for_own, z, family, severity)
                m, lo, hi = _mean_ci_pct(wins)
                xs.append(x)
                ys.append(m)
                elo.append(lo)
                ehi.append(hi)
            ax.errorbar(
                xs, ys, yerr=[elo, ehi], color=COLORS[mode], ls=LINESTYLES[mode],
                marker=MARKERS[mode], ms=5, mec="white", mew=0.5, lw=1.3,
                capsize=3, elinewidth=0.8, label=f"Mode {mode} vs Pole {mode}",
            )
        ax.set_title(fam["title"], fontsize=9.5, fontweight="bold")
        ax.set_xlabel(fam["xlabel"])
        ax.set_ylim(0, 100)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="lower left", frameon=False, fontsize=7)

    axes[0].set_ylabel("Own-pole win rate (%)")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    caption = (
        "Available sealed tiers only (low tier incomplete). Lines are Mode A / Mode B "
        "win rate against their own pole under each deployment perturbation. "
        "Error bars: 95% bootstrap CIs."
    )
    fig.text(0.5, -0.08, caption, ha="center", va="top", fontsize=8, style="italic")
    fig.subplots_adjust(wspace=0.14, bottom=0.24, top=0.88)
    return save_figure(fig, "fig_robustness_dose_response_winrate")


def build_delta_figure() -> dict:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COLUMN, 2.8), sharey=True)

    for ax, fam in zip(axes, FAMILIES):
        xs, da, da_lo, da_hi, db, db_lo, db_hi = [], [], [], [], [], [], []
        for family, severity, x in fam["points"]:
            # Delta_A = V(z0,A) - V(z1,A); Delta_B = V(z1,B) - V(z0,B)
            w_a0 = _load("A", 0, family, severity)
            w_a1 = _load("A", 1, family, severity)
            w_b0 = _load("B", 0, family, severity)
            w_b1 = _load("B", 1, family, severity)
            m_a, lo_a, hi_a = _delta_ci(w_a0, w_a1)
            m_b, lo_b, hi_b = _delta_ci(w_b1, w_b0)
            xs.append(x)
            da.append(m_a); da_lo.append(lo_a); da_hi.append(hi_a)
            db.append(m_b); db_lo.append(lo_b); db_hi.append(hi_b)

        ax.axhline(0.0, color=COLORS["zero"], lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.errorbar(
            xs, da, yerr=[da_lo, da_hi], color=COLORS["A"], ls=LINESTYLES["A"],
            marker=MARKERS["A"], ms=5, mec="white", mew=0.5, lw=1.3,
            capsize=3, elinewidth=0.8, label=r"$\Delta_A$", zorder=2,
        )
        ax.errorbar(
            xs, db, yerr=[db_lo, db_hi], color=COLORS["B"], ls=LINESTYLES["B"],
            marker=MARKERS["B"], ms=5, mec="white", mew=0.5, lw=1.3,
            capsize=3, elinewidth=0.8, label=r"$\Delta_B$", zorder=2,
        )
        ax.set_title(fam["title"], fontsize=9.5, fontweight="bold")
        ax.set_xlabel(fam["xlabel"])
        ax.set_ylim(-25, 45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="lower left", frameon=False, fontsize=7.5)

    axes[0].set_ylabel(r"Specialization $\Delta$ (pp)")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    caption = (
        r"Does specialization survive the perturbation? $\Delta_A=V(z_0,A)-V(z_1,A)$, "
        r"$\Delta_B=V(z_1,B)-V(z_0,B)$ in percentage points. "
        "CI crossing zero means the crossover gate fails at that severity."
    )
    fig.text(0.5, -0.08, caption, ha="center", va="top", fontsize=8, style="italic")
    fig.subplots_adjust(wspace=0.14, bottom=0.24, top=0.88)
    return save_figure(fig, "fig_robustness_delta_dose")


def main() -> None:
    # Touch sealed JSONs so provenance is obvious if a reader greps the script.
    for name in ("ROBUSTNESS_2V2_RUNG1_RESULT.json", "ROBUSTNESS_2V2_HIGH_TIER_RESULT.json"):
        assert (SD / name).exists(), name
    p1 = build_winrate_figure()
    p2 = build_delta_figure()
    print({"winrate": p1, "delta": p2})


if __name__ == "__main__":
    main()
