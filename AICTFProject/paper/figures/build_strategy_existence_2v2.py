"""Main-paper Figure: strategies exist (baselines vs latent) -- two panels.

Panel A -- Closed-loop performance (win rate % + 95% CI):
  Under each opponent pole: Generalist pi_G, Specialist pi_A, Specialist pi_B,
  Latent z0, Latent z1.

Desired visual:
  Pole A:  pi_A ~ z0  >  z1  (and pi_G as no-code baseline)
  Pole B:  pi_B ~ z1  >  z0

Panel B -- Behavioral differences for forced z0 vs z1 (defender / attacker /
  intercept / attack-defense ratio) under each pole.

Sharing ladder is intentionally NOT in this figure -- that is
fig_sharing_ladder_2v2 (how much sharing before specialization deteriorates).

Sources (separate sealed blocks; n noted in caption):
  pi_G   -- pi_g_eval_rows.csv              (n=64)
  pi_A/B -- specialist_baseline_eval_rows.csv (n=64)
  z0/z1  -- rung1_ladder_eval_rows.csv      (n=128; Share-Encoder / Rung 1)
  behavior -- Z0_Z1_BEHAVIOR_CHARACTERIZATION_*.json (exploratory, n=24/cell)

Run:  python paper/figures/build_strategy_existence_2v2.py
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
BEH = SD / "Z0_Z1_BEHAVIOR_CHARACTERIZATION_final_ccp_successor_production.json"

# Visual encoding: A-affinity blue, B-affinity vermillion, generalist gray.
# Hatch distinguishes specialists (solid) from latent codes (hatched).
POLICY_STYLE = {
    "pi_G": {"color": COLORS["control"], "hatch": "", "label": r"Generalist $\pi_G$"},
    "pi_A": {"color": COLORS["A"], "hatch": "", "label": r"Specialist $\pi_A$"},
    "pi_B": {"color": COLORS["B"], "hatch": "", "label": r"Specialist $\pi_B$"},
    "z0": {"color": COLORS["A"], "hatch": "///", "label": r"Latent $z_0$"},
    "z1": {"color": COLORS["B"], "hatch": "///", "label": r"Latent $z_1$"},
}
POLICY_ORDER = ("pi_G", "pi_A", "pi_B", "z0", "z1")

BEH_PANELS = [
    ("Defenders", "num_defenders", (0, 1.0)),
    ("Attackers", "num_attackers", (0, 1.0)),
    ("Intercept", "n_intercept_near_enemy_carrier", (0, 0.55)),
    ("Atk/Def", "attack_defense_ratio", (0, 1.0)),
]


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


def _performance_cells() -> dict[tuple[str, str], tuple[float, float, float]]:
    """(policy, pole) -> (mean%, err_lo, err_hi)."""
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
    return out


def main() -> dict:
    apply_style()
    cells = _performance_cells()
    beh = json.loads(BEH.read_text(encoding="utf-8"))["per_condition_means"]

    fig = plt.figure(figsize=(TWO_COLUMN, 5.2))
    # Top: two performance panels; bottom: four behavior panels
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.55, wspace=0.28)

    axA = fig.add_subplot(gs[0, 0:2])
    axB = fig.add_subplot(gs[0, 2:4], sharey=axA)
    beh_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]

    # ---- Panel A: closed-loop win rates ----
    n = len(POLICY_ORDER)
    xs = np.arange(n)
    bar_w = 0.72

    for ax, pole, letter in ((axA, "A", "a"), (axB, "B", "b")):
        for i, pol in enumerate(POLICY_ORDER):
            mean, elo, ehi = cells[(pol, pole)]
            st = POLICY_STYLE[pol]
            ax.bar(
                xs[i], mean, yerr=[[elo], [ehi]], width=bar_w,
                color=st["color"], hatch=st["hatch"], edgecolor="black",
                linewidth=0.6, capsize=2.5,
                error_kw={"elinewidth": 0.8, "capthick": 0.8},
            )
            ax.text(xs[i], mean + ehi + 2.0, f"{mean:.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [r"$\pi_G$", r"$\pi_A$", r"$\pi_B$", r"$z_0$", r"$z_1$"], fontsize=8.5
        )
        ax.set_ylim(0, 100)
        ax.set_title(f"({letter}) vs Pole {pole}", fontsize=9.5, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axA.set_ylabel("Win rate (%)")
    axB.tick_params(labelleft=False)

    # Highlight intended pattern with light guides
    axA.annotate(
        r"$\pi_A\!\approx\!z_0 > z_1$",
        xy=(0.72, 0.92), xycoords="axes fraction", ha="center", fontsize=7.5,
        color=COLORS["A"],
    )
    axB.annotate(
        r"$\pi_B\!\approx\!z_1 > z_0$",
        xy=(0.72, 0.92), xycoords="axes fraction", ha="center", fontsize=7.5,
        color=COLORS["B"],
    )

    handles = [
        mpatches.Patch(
            facecolor=POLICY_STYLE[p]["color"], hatch=POLICY_STYLE[p]["hatch"],
            edgecolor="black", label=POLICY_STYLE[p]["label"],
        )
        for p in POLICY_ORDER
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01),
        ncol=5, frameon=False, fontsize=7.5,
    )

    # ---- Panel B: behavior z0 vs z1 ----
    group_x = {"A": 0.0, "B": 1.0}
    mode_dx = {"z0": -0.18, "z1": 0.18}
    for ax, (title, key, ylim) in zip(beh_axes, BEH_PANELS):
        for pole in ("A", "B"):
            for z in ("z0", "z1"):
                val = float(beh[f"{z}_pole{pole}"][key])
                ax.bar(
                    group_x[pole] + mode_dx[z], val, width=0.32,
                    color=POLICY_STYLE[z]["color"], hatch=POLICY_STYLE[z]["hatch"],
                    edgecolor="black", linewidth=0.55,
                )
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["Pole A", "Pole B"], fontsize=7.5)
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=8.5, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    beh_axes[0].set_ylabel("Behavior mean")
    fig.text(
        0.5, 0.46, "(c) Are latent strategies behaviorally different?",
        ha="center", va="bottom", fontsize=9.5, fontweight="bold",
    )

    caption = (
        r"(a--b) Closed-loop win rates with 95\% bootstrap CIs. "
        r"Specialists establish the specialization pattern; $\pi_G$ is the no-strategy-code "
        r"baseline; $z_0/z_1$ are forced codes of one shared Rung-1 policy $\pi_\theta(a\mid o,z)$. "
        r"Hatched bars = latent codes. "
        r"(c) Forced-$z$ behavior proxies (exploratory characterization). "
        r"Seed blocks differ ($\pi_G$/specialists $n{=}64$; latent $n{=}128$) -- "
        r"pattern comparison, not a matched five-way test. "
        r"Sharing ladder: see fig\_sharing\_ladder\_2v2."
    )
    fig.text(0.5, -0.01, caption, ha="center", va="top", fontsize=7.2, style="italic")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.14)

    paths = save_figure(fig, "fig_strategy_existence_2v2")
    print(paths)
    return paths


if __name__ == "__main__":
    main()
