"""2v2 centerpiece: baselines + strategy-conditioned performance across sharing.

Two panels (Pole A | Pole B). Within each panel, bars in this order:

  pi_G, pi_A, pi_B | Share-0 (z0,z1) | Encoder (z0,z1) | Backbone (z0,z1) | Macro (z0,z1)

Visual encoding:
  solid gray  = generalist
  solid blue/vermillion = specialists
  hatched blue/vermillion = forced z0/z1 of the shared policy at that rung

Desired X-pattern:
  Pole A: pi_A ~ z0 > z1
  Pole B: pi_B ~ z1 > z0

Sources (sealed):
  pi_G -- pi_g_eval_rows.csv (n=64)
  pi_A/B -- specialist_baseline_eval_rows.csv (n=64)
  Share-* cells -- RUNG0_LADDER_REFERENCE / RUNG1/2/3_LADDER_EVAL_RESULT (n=128)
  CIs for Share-* recomputed from ladder CSVs where available; Rung0 pooled
  cells use sealed means with bootstrap from rung0 + stability CSVs when present.

Run:  python paper/figures/build_2v2_baseline_sharing_performance.py
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

SHARE_ORDER = (
    ("Share-0", "RUNG0_LADDER_REFERENCE.json", "cell_win_rates_n128", None),
    ("Share-Encoder", "RUNG1_LADDER_EVAL_RESULT.json", "cell_win_rates", "rung1_ladder_eval_rows.csv"),
    ("Share-Backbone", "RUNG2_LADDER_EVAL_RESULT.json", "cell_win_rates", "rung2_ladder_eval_rows.csv"),
    ("Share-Macro", "RUNG3_LADDER_EVAL_RESULT.json", "cell_win_rates", "rung3_ladder_eval_rows.csv"),
)


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


def _ref_stats() -> dict[tuple[str, str], tuple[float, float, float]]:
    out = {}
    g = SD / "pi_g_eval_rows.csv"
    s = SD / "specialist_baseline_eval_rows.csv"
    for pole in ("A", "B"):
        out[("pi_G", pole)] = _mean_ci_pct(
            _csv_wins(g, lambda r, p=pole: r["policy"] == "pi_G" and r["pole"] == p)
        )
        out[("pi_A", pole)] = _mean_ci_pct(
            _csv_wins(s, lambda r, p=pole: r["policy"] == "pi_A" and r["pole"] == p)
        )
        out[("pi_B", pole)] = _mean_ci_pct(
            _csv_wins(s, lambda r, p=pole: r["policy"] == "pi_B" and r["pole"] == p)
        )
    return out


def _rung0_pooled_wins(z: str, pole: str) -> np.ndarray:
    """Concatenate sealed n=64 block + stability n=64 block (= matched n=128)."""
    blocks = [
        SD / "rung0_crossover_eval_rows.csv",
        SD / "rung0_stability_rerun_rows.csv",
    ]
    parts = []
    for path in blocks:
        parts.append(_csv_wins(path, lambda r, zz=z, p=pole: r["z"] == zz and r["pole"] == p))
    return np.concatenate(parts)


def _share_stats() -> dict[tuple[str, str, str], tuple[float, float, float]]:
    """(share_label, z, pole) -> (mean%, elo, ehi)."""
    out = {}
    for label, jname, cell_key, csv_name in SHARE_ORDER:
        if label == "Share-0":
            for z in ("z0", "z1"):
                for pole in ("A", "B"):
                    out[(label, z, pole)] = _mean_ci_pct(_rung0_pooled_wins(z, pole))
            continue
        csv_path = SD / csv_name
        for z in ("z0", "z1"):
            for pole in ("A", "B"):
                wins = _csv_wins(csv_path, lambda r, zz=z, p=pole: r["z"] == zz and r["pole"] == p)
                out[(label, z, pole)] = _mean_ci_pct(wins)
        # sanity vs sealed JSON means
        cells = json.loads((SD / jname).read_text(encoding="utf-8"))[cell_key]
        for z in ("z0", "z1"):
            for pole in ("A", "B"):
                sealed = cells[f"{z}_pole{pole}"] * 100
                got = out[(label, z, pole)][0]
                if abs(sealed - got) > 0.05:
                    raise SystemExit(f"CI mean drift {label} {z}{pole}: {got} vs sealed {sealed}")
    return out


def main() -> dict:
    apply_style()
    ref = _ref_stats()
    share = _share_stats()

    fig, axes = plt.subplots(1, 2, figsize=(TWO_COLUMN, 3.35), sharey=True)

    # x layout: 3 ref + gap + 4*(2 bars) with small gaps between share groups
    # positions computed explicitly
    def positions():
        xs = []
        labels = []
        groups = []  # for vertical separators
        x = 0.0
        for pol in ("pi_G", "pi_A", "pi_B"):
            xs.append(x)
            labels.append(pol)
            x += 1.0
        x += 0.55  # gap
        for si, (label, *_rest) in enumerate(SHARE_ORDER):
            groups.append(x - 0.35)
            for z in ("z0", "z1"):
                xs.append(x)
                labels.append(f"{label}|{z}")
                x += 0.85
            x += 0.35  # gap between share groups
        return xs, labels, groups

    xs, labels, groups = positions()

    for ax, pole in zip(axes, ("A", "B")):
        # reference
        for i, pol in enumerate(("pi_G", "pi_A", "pi_B")):
            mean, elo, ehi = ref[(pol, pole)]
            color = COLORS["control"] if pol == "pi_G" else COLORS["A" if pol.endswith("A") else "B"]
            ax.bar(xs[i], mean, yerr=[[elo], [ehi]], width=0.7, color=color,
                   edgecolor="black", linewidth=0.55, capsize=2,
                   error_kw={"elinewidth": 0.7, "capthick": 0.7})
            ax.text(xs[i], mean + ehi + 1.8, f"{mean:.0f}", ha="center", fontsize=6)

        # sharing z bars
        idx = 3
        for label, *_rest in SHARE_ORDER:
            for z in ("z0", "z1"):
                mean, elo, ehi = share[(label, z, pole)]
                ax.bar(
                    xs[idx], mean, yerr=[[elo], [ehi]], width=0.7,
                    color=COLORS["A" if z == "z0" else "B"], hatch="///",
                    edgecolor="black", linewidth=0.55, capsize=2,
                    error_kw={"elinewidth": 0.7, "capthick": 0.7},
                )
                ax.text(xs[idx], mean + ehi + 1.8, f"{mean:.0f}", ha="center", fontsize=6)
                idx += 1

        for g in groups:
            ax.axvline(g, color="#CCCCCC", lw=0.6, zorder=0)

        ax.set_ylim(0, 100)
        ax.set_title(f"vs Pole {pole}", fontsize=9.5, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # custom tick labels at group centers
        tick_pos = [xs[0], xs[1], xs[2]]
        tick_lab = [r"$\pi_G$", r"$\pi_A$", r"$\pi_B$"]
        # centers of each share pair
        base = 3
        for si, (label, *_r) in enumerate(SHARE_ORDER):
            tick_pos.append(0.5 * (xs[base + 2 * si] + xs[base + 2 * si + 1]))
            short = ["S0", "Enc", "Back", "Macro"][si]
            tick_lab.append(short)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, fontsize=7.5)

    axes[0].set_ylabel("Win rate (%)")
    axes[0].annotate(
        r"want $\pi_A\!\approx\!z_0 > z_1$",
        xy=(0.98, 0.92), xycoords="axes fraction", ha="right", fontsize=7, color=COLORS["A"],
    )
    axes[1].annotate(
        r"want $\pi_B\!\approx\!z_1 > z_0$",
        xy=(0.98, 0.92), xycoords="axes fraction", ha="right", fontsize=7, color=COLORS["B"],
    )

    handles = [
        mpatches.Patch(facecolor=COLORS["control"], edgecolor="black", label=r"Generalist $\pi_G$"),
        mpatches.Patch(facecolor=COLORS["A"], edgecolor="black", label=r"Specialist $\pi_A$"),
        mpatches.Patch(facecolor=COLORS["B"], edgecolor="black", label=r"Specialist $\pi_B$"),
        mpatches.Patch(facecolor=COLORS["A"], hatch="///", edgecolor="black", label=r"Latent $z_0$"),
        mpatches.Patch(facecolor=COLORS["B"], hatch="///", edgecolor="black", label=r"Latent $z_1$"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.05),
               ncol=5, frameon=False, fontsize=7.5)

    caption = (
        r"2v2 sealed package: generalist + specialists vs forced-$z$ codes under progressive sharing "
        r"(S0=Share-0, Enc=Share-Encoder, Back=Share-Backbone, Macro=Share-Macro). "
        r"Hatched = same shared policy, only $z$ changes. "
        r"Error bars: 95\% bootstrap CIs ($\pi_G$/specialists $n{=}64$; sharing $n{=}128$)."
    )
    fig.text(0.5, -0.06, caption, ha="center", va="top", fontsize=7.3, style="italic")
    fig.subplots_adjust(wspace=0.12, bottom=0.20, top=0.82, left=0.07, right=0.99)

    paths = save_figure(fig, "fig_2v2_baseline_sharing_performance")
    print(paths)
    return paths


if __name__ == "__main__":
    main()
