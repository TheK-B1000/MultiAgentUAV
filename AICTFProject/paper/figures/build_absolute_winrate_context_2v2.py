"""Main-paper absolute win-rate context (not the specialization hypothesis test).

Two panels (Regime A | Regime B). Bars within each panel:

  pi_G, pi_A, pi_B, Share-Encoder z0, Share-Encoder z1

CSC/SPFT/PPO-only recovery arms and Encoder/Backbone/Macro degradation are
intentionally omitted -- those live in Claim B and the Delta ladder figures.

Sources (sealed):
  pi_G   -- pi_g_eval_rows.csv                 (n=64)
  pi_A/B -- specialist_baseline_eval_rows.csv  (n=64)
  z0/z1  -- rung1_ladder_eval_rows.csv         (n=128; Share-Encoder)

Run:  python paper/figures/build_absolute_winrate_context_2v2.py
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

from paper.figures.figure_style import COLORS, TWO_COLUMN, apply_style, save_figure

SD = ROOT / "artifacts/strategic_demand/sppo"
STEM = "fig_absolute_winrate_context_2v2"

POLICY_STYLE = {
    "pi_G": {"color": COLORS["control"], "hatch": "", "label": r"Generalist $\pi_G$"},
    "pi_A": {"color": COLORS["A"], "hatch": "", "label": r"Specialist $\pi_A$"},
    "pi_B": {"color": COLORS["B"], "hatch": "", "label": r"Specialist $\pi_B$"},
    "z0": {"color": COLORS["A"], "hatch": "///", "label": r"Share-Encoder $z_0$"},
    "z1": {"color": COLORS["B"], "hatch": "///", "label": r"Share-Encoder $z_1$"},
}
POLICY_ORDER = ("pi_G", "pi_A", "pi_B", "z0", "z1")
TICK_LABELS = [r"$\pi_G$", r"$\pi_A$", r"$\pi_B$", r"$z_0$", r"$z_1$"]


def _mean_ci_pct(wins: np.ndarray, n_boot=20000, alpha=0.05, rng_seed=7):
    rng = np.random.default_rng(rng_seed)
    boots = wins[rng.integers(0, wins.size, size=(n_boot, wins.size))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    mean = float(wins.mean())
    return mean * 100, max(0.0, (mean - lo) * 100), max(0.0, (hi - mean) * 100)


def _load_wins(csv_path: Path, filter_fn) -> np.ndarray:
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return np.asarray([int(r["win"]) for r in rows if filter_fn(r)], dtype=float)


def _cells() -> dict[tuple[str, str], tuple[float, float, float]]:
    g_csv = SD / "pi_g_eval_rows.csv"
    s_csv = SD / "specialist_baseline_eval_rows.csv"
    z_csv = SD / "rung1_ladder_eval_rows.csv"
    out = {}
    for pole in ("A", "B"):
        out[("pi_G", pole)] = _mean_ci_pct(
            _load_wins(g_csv, lambda r, p=pole: r["policy"] == "pi_G" and r["pole"] == p)
        )
        out[("pi_A", pole)] = _mean_ci_pct(
            _load_wins(s_csv, lambda r, p=pole: r["policy"] == "pi_A" and r["pole"] == p)
        )
        out[("pi_B", pole)] = _mean_ci_pct(
            _load_wins(s_csv, lambda r, p=pole: r["policy"] == "pi_B" and r["pole"] == p)
        )
        out[("z0", pole)] = _mean_ci_pct(
            _load_wins(z_csv, lambda r, p=pole: r["z"] == "z0" and r["pole"] == p)
        )
        out[("z1", pole)] = _mean_ci_pct(
            _load_wins(z_csv, lambda r, p=pole: r["z"] == "z1" and r["pole"] == p)
        )
    # Sanity vs sealed Share-Encoder JSON means
    sealed = json.loads((SD / "RUNG1_LADDER_EVAL_RESULT.json").read_text(encoding="utf-8"))[
        "cell_win_rates"
    ]
    for z in ("z0", "z1"):
        for pole in ("A", "B"):
            sealed_pct = sealed[f"{z}_pole{pole}"] * 100
            got = out[(z, pole)][0]
            if abs(sealed_pct - got) > 0.05:
                raise SystemExit(f"CI mean drift {z} pole{pole}: {got} vs sealed {sealed_pct}")
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
        ax.axvline(2.5, color="#CCCCCC", lw=0.7, zorder=0)

    axes[0].set_ylabel("Win rate (%)")
    axes[0].annotate(
        r"want $\pi_A,\,z_0$ high",
        xy=(0.98, 0.93), xycoords="axes fraction", ha="right",
        fontsize=7.5, color=COLORS["A"],
    )
    axes[1].annotate(
        r"want $\pi_B,\,z_1$ high",
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
        ncol=5, frameon=False, fontsize=7.5,
    )
    fig.subplots_adjust(wspace=0.12, bottom=0.14, top=0.82, left=0.07, right=0.99)

    paths = save_figure(fig, STEM)
    # Compact numeric manifest for caption / text cross-checks
    manifest = {
        "stem": STEM,
        "role": "absolute_performance_context_not_specialization_test",
        "policies": list(POLICY_ORDER),
        "n": {"pi_G": 64, "pi_A": 64, "pi_B": 64, "z0": 128, "z1": 128},
        "win_rate_pct": {
            f"{pol}_regime{pole}": round(cells[(pol, pole)][0], 2)
            for pol in POLICY_ORDER
            for pole in ("A", "B")
        },
    }
    man_path = ROOT / "paper" / "data" / "absolute_winrate_context_2v2.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(paths)
    print(f"manifest: {man_path}")
    return paths


if __name__ == "__main__":
    main()
