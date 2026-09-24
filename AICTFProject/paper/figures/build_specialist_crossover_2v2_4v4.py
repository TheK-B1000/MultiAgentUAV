"""Absolute crossover win rates: 2v2 PASS vs 4v4 scaffolded PASS*.

Two panels (same bar layout as build_specialist_crossover_2v2.py):

  (a) 2v2 Share-0 modes z0 / z1 (n=128) — complementary ownership, joint PASS
  (b) 4v4 A' vs pi_B (n=128) — the arms that enter Delta'_A / Delta'_B

Native 4v4 specialists are omitted: that pair fails the joint gate. A' is
pi_A with imposed 2A/2D (controller), so panel (b) is not a learned
specialist pair — it is the passing scaffold comparison.

Run:  ./.venv/Scripts/python.exe paper/figures/build_specialist_crossover_2v2_4v4.py
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
STEM = "fig_specialist_crossover_2v2_4v4"
COMBINED = ROOT / "paper" / "plots" / "combined"

# Share-0 matched cells (same sealed source as the Delta PASS).
CSV_2V2_BLOCK1 = SD / "rung0_crossover_eval_rows.csv"
CSV_2V2_BLOCK2 = SD / "rung0_stability_rerun_rows.csv"
CSV_4V4 = SD / "SCAFFOLDED_A_CROSSOVER_BRIDGE_4V4_ROWS.csv"

# Map to A/B colors: home-A mode vs home-B mode.
MODE_COLOR = {"home_A": COLORS["A"], "home_B": COLORS["B"]}


def _mean_ci_pct(wins: np.ndarray, n_boot=20000, alpha=0.05, rng_seed=7):
    rng = np.random.default_rng(rng_seed)
    boots = wins[rng.integers(0, wins.size, size=(n_boot, wins.size))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    mean = float(wins.mean())
    return mean * 100, max(0.0, (mean - lo) * 100), max(0.0, (hi - mean) * 100)


def _load_2v2_share0() -> dict[tuple[str, str], np.ndarray]:
    """z0 = home-A mode, z1 = home-B mode; pool both matched seed blocks."""
    rows = []
    for path in (CSV_2V2_BLOCK1, CSV_2V2_BLOCK2):
        with path.open(encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    out: dict[tuple[str, str], list[int]] = {}
    for r in rows:
        # rung0 CSVs use policy in {z0,z1} or similar — inspect flexibly
        mode = r.get("z") or r.get("policy") or r.get("arm")
        if mode in ("z0", "pi_A", "0"):
            key = ("home_A", r["pole"])
        elif mode in ("z1", "pi_B", "1"):
            key = ("home_B", r["pole"])
        else:
            raise SystemExit(f"unexpected 2v2 mode field: {mode!r} in {r}")
        out.setdefault(key, []).append(int(r["win"]))
    cells = {k: np.asarray(v, dtype=float) for k, v in out.items()}
    for k, arr in cells.items():
        if arr.size != 128:
            raise SystemExit(f"2v2 cell {k} has n={arr.size}, expected 128")
    sealed = json.loads((SD / "RUNG0_LADDER_REFERENCE.json").read_text(encoding="utf-8"))[
        "cell_win_rates_n128"
    ]
    check = {
        ("home_A", "A"): sealed["z0_poleA"],
        ("home_A", "B"): sealed["z0_poleB"],
        ("home_B", "A"): sealed["z1_poleA"],
        ("home_B", "B"): sealed["z1_poleB"],
    }
    for k, want in check.items():
        got = float(cells[k].mean())
        if abs(got - want) > 5e-4:
            raise SystemExit(f"2v2 Share-0 drift {k}: {got} vs sealed {want}")
    return cells


def _load_4v4_scaffold() -> dict[tuple[str, str], np.ndarray]:
    """A' plays the home-A role in Delta'; pi_B plays the home-B role."""
    with CSV_4V4.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[tuple[str, str], list[int]] = {}
    for r in rows:
        if r["arm"] == "A_prime":
            key = ("home_A", r["pole"])
        elif r["arm"] == "pi_B":
            key = ("home_B", r["pole"])
        else:
            continue  # drop native pi_A from this PASS-only comparison
        out.setdefault(key, []).append(int(r["win"]))
    cells = {k: np.asarray(v, dtype=float) for k, v in out.items()}
    sealed = json.loads(
        (SD / "SCAFFOLDED_A_CROSSOVER_BRIDGE_4V4_RESULT.json").read_text(encoding="utf-8")
    )["CELL_MEANS"]
    check = {
        ("home_A", "A"): sealed["A_prime_poleA"]["win"],
        ("home_A", "B"): sealed["A_prime_poleB"]["win"],
        ("home_B", "A"): sealed["pi_B_poleA"]["win"],
        ("home_B", "B"): sealed["pi_B_poleB"]["win"],
    }
    for k, want in check.items():
        got = float(cells[k].mean())
        if abs(got - want) > 5e-4:
            raise SystemExit(f"4v4 scaffold drift {k}: {got} vs sealed {want}")
    return cells


def _draw_panel(
    ax,
    cells: dict[tuple[str, str], np.ndarray],
    title: str,
    letter: str,
    labels: dict[str, str],
) -> dict[str, float]:
    group_x = {"A": 0.0, "B": 1.0}
    mode_dx = {"home_A": -0.19, "home_B": 0.19}
    bar_w = 0.34
    rates: dict[str, float] = {}

    for pole in ("A", "B"):
        for mode in ("home_A", "home_B"):
            wins = cells[(mode, pole)]
            value, err_lo, err_hi = _mean_ci_pct(wins)
            rates[f"{mode}_pole{pole}"] = round(value, 2)
            x = group_x[pole] + mode_dx[mode]
            ax.bar(
                x, value, yerr=[[err_lo], [err_hi]], width=bar_w,
                color=MODE_COLOR[mode], edgecolor="black", linewidth=0.6,
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
    ax.set_title(f"({letter}) {title}", fontsize=9.5, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # store labels for legend via closure on caller
    ax._mode_labels = labels  # type: ignore[attr-defined]
    return rates


def main() -> int:
    apply_style()
    cells_2 = _load_2v2_share0()
    cells_4 = _load_4v4_scaffold()

    labels_2 = {
        "home_A": r"$z_0$ (Share-0, home A)",
        "home_B": r"$z_1$ (Share-0, home B)",
    }
    labels_4 = {
        "home_A": r"$A'=\pi_A+2$D (scaffold)",
        "home_B": r"$\pi_B$ (learned)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(TWO_COLUMN, 3.0), sharey=True)
    rates_2 = _draw_panel(axes[0], cells_2, "2v2 Share-0 (PASS)", "a", labels_2)
    rates_4 = _draw_panel(
        axes[1], cells_4, "4v4 A' scaffold (PASS*)", "b", labels_4
    )
    axes[0].set_ylabel("Win rate (%)")

    # Panel-specific legends (arms differ)
    for ax, labels in ((axes[0], labels_2), (axes[1], labels_4)):
        handles = [
            mpatches.Patch(
                facecolor=MODE_COLOR[m], edgecolor="black", label=labels[m]
            )
            for m in ("home_A", "home_B")
        ]
        ax.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
            ncol=1, frameon=False, fontsize=7.5,
        )

    fig.subplots_adjust(wspace=0.14, bottom=0.28, top=0.88, left=0.07, right=0.99)

    paths = save_figure(fig, STEM)
    COMBINED.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        src = ROOT / "paper" / "generated" / f"{STEM}.{ext}"
        shutil.copy2(src, COMBINED / f"{STEM}.{ext}")

    manifest = {
        "stem": STEM,
        "role": "absolute_crossover_2v2_PASS_vs_4v4_scaffold_PASS",
        "n": {"2v2": 128, "4v4": 128},
        "win_rate_pct": {"2v2_Share-0": rates_2, "4v4_A_prime_vs_pi_B": rates_4},
        "gates": {"2v2": "PASS", "4v4": "PASS* (scaffolded controller)"},
        "note": "Native 4v4 specialist FAIL omitted; panel (b) arms are A' and pi_B only.",
    }
    man_path = ROOT / "paper" / "data" / "specialist_crossover_2v2_4v4.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(paths)
    print(f"manifest: {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
