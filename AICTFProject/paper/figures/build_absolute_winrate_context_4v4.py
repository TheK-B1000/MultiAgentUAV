"""Absolute win-rate context for the 4v4 scaffolded crossover bridge.

Two panels (Regime A | Regime B). Bars within each panel:

  π_A (native)  |  A′ = π_A + 2D  |  π_B

Parallel to build_absolute_winrate_context_2v2.py but without π_G / Share-Encoder
(those rungs do not exist at 4v4). Descriptive context — the specialization
hypothesis test is the Δ′ gate on the scaling figure, not these bars.

Source: SCAFFOLDED_A_CROSSOVER_BRIDGE_4V4_ROWS.csv (n=128 paired seeds).

Run:  ./.venv/Scripts/python.exe paper/figures/build_absolute_winrate_context_4v4.py
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.figures.figure_style import COLORS, TWO_COLUMN, apply_style, save_figure

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
STEM = "fig_absolute_winrate_context_4v4"
COMBINED = ROOT / "paper" / "plots" / "combined"
CSV = SD / "SCAFFOLDED_A_CROSSOVER_BRIDGE_4V4_ROWS.csv"

POLICY_STYLE = {
    "pi_A": {"color": COLORS["A"], "hatch": "", "label": r"Native $\pi_A$"},
    "A_prime": {
        "color": COLORS["A"],
        "hatch": "///",
        "label": r"$A'=\pi_A+2$D (scaffold)",
    },
    "pi_B": {"color": COLORS["B"], "hatch": "", "label": r"Native $\pi_B$"},
}
POLICY_ORDER = ("pi_A", "A_prime", "pi_B")
TICK_LABELS = [r"$\pi_A$", r"$A'$", r"$\pi_B$"]


def _mean_ci_pct(wins: np.ndarray, n_boot=20000, alpha=0.05, rng_seed=7):
    rng = np.random.default_rng(rng_seed)
    boots = wins[rng.integers(0, wins.size, size=(n_boot, wins.size))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    mean = float(wins.mean())
    return mean * 100, max(0.0, (mean - lo) * 100), max(0.0, (hi - mean) * 100)


def _cells() -> dict[tuple[str, str], tuple[float, float, float]]:
    with CSV.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for pol in POLICY_ORDER:
        for pole in ("A", "B"):
            wins = np.asarray(
                [int(r["win"]) for r in rows if r["arm"] == pol and r["pole"] == pole],
                dtype=float,
            )
            if wins.size != 128:
                raise SystemExit(f"expected 128 rows for {pol}@{pole}, got {wins.size}")
            out[(pol, pole)] = _mean_ci_pct(wins)

    sealed = json.loads(
        (SD / "SCAFFOLDED_A_CROSSOVER_BRIDGE_4V4_RESULT.json").read_text(encoding="utf-8")
    )["CELL_MEANS"]
    key_map = {
        ("pi_A", "A"): "pi_A_poleA",
        ("pi_A", "B"): "pi_A_poleB",
        ("A_prime", "A"): "A_prime_poleA",
        ("A_prime", "B"): "A_prime_poleB",
        ("pi_B", "A"): "pi_B_poleA",
        ("pi_B", "B"): "pi_B_poleB",
    }
    for key, sealed_key in key_map.items():
        got = out[key][0] / 100.0
        want = float(sealed[sealed_key]["win"])
        if abs(got - want) > 1e-6:
            raise SystemExit(f"cell drift {key}: {got} vs sealed {want}")
    return out


def main() -> dict:
    apply_style()
    cells = _cells()

    fig, axes = plt.subplots(1, 2, figsize=(TWO_COLUMN, 2.85), sharey=True)
    xs = np.arange(len(POLICY_ORDER))
    bar_w = 0.72

    for ax, pole, letter in ((axes[0], "A", "a"), (axes[1], "B", "b")):
        for i, pol in enumerate(POLICY_ORDER):
            mean, elo, ehi = cells[(pol, pole)]
            st = POLICY_STYLE[pol]
            ax.bar(
                xs[i], mean, yerr=[[elo], [ehi]], width=bar_w,
                color=st["color"], hatch=st["hatch"], edgecolor="black",
                linewidth=0.6, capsize=2.5,
                error_kw={"elinewidth": 0.8, "capthick": 0.8},
            )
            ax.text(
                xs[i], mean + ehi + 1.6, f"{mean:.0f}",
                ha="center", va="bottom", fontsize=7.5,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(TICK_LABELS, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title(f"({letter}) Regime {pole}", fontsize=9.5, fontweight="bold")

    axes[0].set_ylabel("Win rate (%)")
    axes[0].annotate(
        r"$A'$ nearly preserves $\pi_A$",
        xy=(0.98, 0.93), xycoords="axes fraction", ha="right",
        fontsize=7.5, color=COLORS["A"],
    )
    axes[1].annotate(
        r"$A'$ collapses; $\pi_B$ mid",
        xy=(0.98, 0.93), xycoords="axes fraction", ha="right",
        fontsize=7.5, color=COLORS["B"],
    )

    handles = [
        mpatches.Patch(
            facecolor=POLICY_STYLE[p]["color"], hatch=POLICY_STYLE[p]["hatch"],
            edgecolor="black", label=POLICY_STYLE[p]["label"],
        )
        for p in POLICY_ORDER
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.08),
        ncol=3, frameon=False, fontsize=7.5,
    )
    fig.subplots_adjust(wspace=0.12, bottom=0.14, top=0.82, left=0.07, right=0.99)

    paths = save_figure(fig, STEM)
    COMBINED.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        src = ROOT / "paper" / "generated" / f"{STEM}.{ext}"
        shutil.copy2(src, COMBINED / f"{STEM}.{ext}")

    manifest = {
        "stem": STEM,
        "role": "absolute_performance_context_4v4_scaffold_not_specialization_test",
        "policies": list(POLICY_ORDER),
        "n": 128,
        "win_rate_pct": {
            f"{pol}_regime{pole}": round(cells[(pol, pole)][0], 2)
            for pol in POLICY_ORDER
            for pole in ("A", "B")
        },
        "source": "SCAFFOLDED_A_CROSSOVER_BRIDGE_4V4_ROWS.csv",
    }
    man_path = ROOT / "paper" / "data" / "absolute_winrate_context_4v4.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(paths)
    print(f"combined: {COMBINED / (STEM + '.png')}")
    print(f"manifest: {man_path}")
    return paths


if __name__ == "__main__":
    main()
