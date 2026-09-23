"""Payoff contrast: 2v2 Share-0 PASS vs 4v4 A' scaffold PASS*.

Two panels:
  (a) Absolute win rates of the gate-clearing mode pair under each pole
  (b) Specialization Delta_A / Delta_B with sealed 95% CIs

This is the payoff companion to the trajectory strip and role-allocation
figures. A' is a scaffolded controller — not a learned latent mode.

Run:  ./.venv/Scripts/python.exe paper/figures/build_payoff_contrast_2v2_4v4.py
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
STEM = "fig_payoff_contrast_2v2_4v4"
COMBINED = ROOT / "paper" / "plots" / "combined"


def _mean_ci_pct(wins: np.ndarray, n_boot=20000, alpha=0.05, rng_seed=7):
    rng = np.random.default_rng(rng_seed)
    boots = wins[rng.integers(0, wins.size, size=(n_boot, wins.size))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    mean = float(wins.mean())
    return mean * 100, max(0.0, (mean - lo) * 100), max(0.0, (hi - mean) * 100)


def _load_2v2() -> dict[tuple[str, str], np.ndarray]:
    rows = []
    for name in ("rung0_crossover_eval_rows.csv", "rung0_stability_rerun_rows.csv"):
        with (SD / name).open(encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    out: dict[tuple[str, str], list[int]] = {}
    for r in rows:
        mode = "home_A" if r["z"] == "z0" else "home_B"
        out.setdefault((mode, r["pole"]), []).append(int(r["win"]))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def _load_4v4() -> dict[tuple[str, str], np.ndarray]:
    with (SD / "SCAFFOLDED_A_CROSSOVER_BRIDGE_4V4_ROWS.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[tuple[str, str], list[int]] = {}
    for r in rows:
        if r["arm"] == "A_prime":
            mode = "home_A"
        elif r["arm"] == "pi_B":
            mode = "home_B"
        else:
            continue
        out.setdefault((mode, r["pole"]), []).append(int(r["win"]))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def _panel_abs(ax, cells, title: str, labels: dict[str, str]) -> None:
    colors = {"home_A": COLORS["A"], "home_B": COLORS["B"]}
    hatches = {"home_A": "///", "home_B": "xxx"}
    for pole_i, pole in enumerate(("A", "B")):
        for mode, dx in (("home_A", -0.18), ("home_B", 0.18)):
            m, lo, hi = _mean_ci_pct(cells[(mode, pole)])
            x = pole_i + dx
            ax.bar(
                x, m, yerr=[[lo], [hi]], width=0.32,
                color=colors[mode], hatch=hatches[mode],
                edgecolor="black", linewidth=0.55, capsize=2,
                error_kw={"elinewidth": 0.7},
            )
            ax.text(x, m + hi + 1.5, f"{m:.0f}", ha="center", fontsize=7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pole A", "Pole B"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Win rate (%)")
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles = [
        mpatches.Patch(
            facecolor=colors[m], hatch=hatches[m], edgecolor="black", label=labels[m]
        )
        for m in ("home_A", "home_B")
    ]
    ax.legend(handles=handles, loc="lower center", frameon=False, fontsize=6.5)


def _panel_delta(ax) -> None:
    r0 = json.loads((SD / "RUNG0_LADDER_REFERENCE.json").read_text(encoding="utf-8"))[
        "POOLED_N128"
    ]
    sc = json.loads(
        (SD / "SCAFFOLDED_A_CROSSOVER_BRIDGE_4V4_RESULT.json").read_text(encoding="utf-8")
    )["PRIMARY_WIN_RATE_CONTRASTS"]
    conds = [
        ("2v2\nShare-0", r0["delta_A"], r0["delta_B"], "PASS"),
        ("4v4\nA' scaffold", sc["Delta_A_prime"], sc["Delta_B_prime"], "PASS*"),
    ]
    for i, (lab, da, db, gate) in enumerate(conds):
        for block, color, marker, dx in (
            (da, COLORS["A"], "o", -0.12),
            (db, COLORS["B"], "s", +0.12),
        ):
            mean, lo, hi = float(block["mean"]), float(block["lcb95"]), float(block["ucb95"])
            ax.errorbar(
                i + dx, mean, yerr=[[mean - lo], [hi - mean]],
                fmt=marker, color=color, markersize=5.5, capsize=3,
                linewidth=1.2, markeredgewidth=1.0, zorder=3,
            )
        ax.text(
            i, -0.08, gate, ha="center", va="top", fontsize=8, fontweight="bold",
            color="#1a7f37" if gate == "PASS" else "#b07000",
        )
    ax.axhline(0.0, color="#000", lw=0.8, zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([c[0] for c in conds])
    ax.set_ylim(-0.12, 0.55)
    ax.set_ylabel(r"specialization contrast $\Delta$")
    ax.set_title(r"(c) Payoff $\Delta_A$, $\Delta_B$", fontsize=9, fontweight="bold")
    ax.plot([], [], "o", color=COLORS["A"], label=r"$\Delta_A$")
    ax.plot([], [], "s", color=COLORS["B"], label=r"$\Delta_B$")
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> dict:
    apply_style()
    c2 = _load_2v2()
    c4 = _load_4v4()

    fig, axes = plt.subplots(1, 3, figsize=(TWO_COLUMN, 2.9))
    _panel_abs(
        axes[0], c2, "(a) 2v2 Share-0 (PASS)",
        {"home_A": r"$z_0$", "home_B": r"$z_1$"},
    )
    _panel_abs(
        axes[1], c4, "(b) 4v4 A' scaffold (PASS*)",
        {"home_A": r"$A'$", "home_B": r"$\pi_B$"},
    )
    _panel_delta(axes[2])

    caption = (
        "Gate-clearing payoff only. 2v2: learned Share-0 modes. "
        r"4v4: $A'=\pi_A$ with imposed 2A/2D vs existing $\pi_B$ (controller, not latent $z$). "
        "n=128 paired seeds; 95% bootstrap CIs."
    )
    fig.text(0.5, -0.04, caption, ha="center", va="top", fontsize=7.0, style="italic")
    fig.subplots_adjust(wspace=0.28, bottom=0.20, top=0.86, left=0.06, right=0.99)

    paths = save_figure(fig, STEM)
    COMBINED.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        src = ROOT / "paper" / "generated" / f"{STEM}.{ext}"
        shutil.copy2(src, COMBINED / f"{STEM}.{ext}")
    print(paths)
    return paths


if __name__ == "__main__":
    main()
