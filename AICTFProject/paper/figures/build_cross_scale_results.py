"""Cross-scale results: win rates + specialization Delta (2v2 / 4v4 / 6v6).

Discipline (PI rule):
  Each scale earns latent/sharing rows ONLY after learned specialist crossover is
  sealed PASS. Until then, show demand / specialist status -- never invent cells.

Today:
  2v2  -- FULL (pi_G, specialists PASS, Share-0..Macro sealed)
  4v4  -- demand CERTIFIED; confirmatory specialists sealed FAIL (C2);
          B3 crossover not sealed; no pi_G; no sharing ladder
  6v6  -- demand CERTIFIED; no learned specialists; no pi_G; no ladder

Figure A: fig_cross_scale_winrate
  Three scale columns. 2v2 shows reference + sharing-level z0/z1 under each pole.
  4v4/6v6 show status panels (no fake bars).

Figure B: fig_cross_scale_delta
  Delta_A / Delta_B vs sharing level per scale (2v2 filled; others pending).

Run:  python paper/figures/build_cross_scale_results.py
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

SHARE_LABELS = ("Share-0", "Share-Encoder", "Share-Backbone", "Share-Macro")
SHARE_FILES = {
    "Share-0": ("RUNG0_LADDER_REFERENCE.json", "cell_win_rates_n128", "POOLED_N128"),
    "Share-Encoder": ("RUNG1_LADDER_EVAL_RESULT.json", "cell_win_rates", "OWN_GATE_N128"),
    "Share-Backbone": ("RUNG2_LADDER_EVAL_RESULT.json", "cell_win_rates", "OWN_GATE_N128"),
    "Share-Macro": ("RUNG3_LADDER_EVAL_RESULT.json", "cell_win_rates", "OWN_GATE_N128"),
}


def _mean_ci_pct(wins: np.ndarray, n_boot=20000, alpha=0.05, rng_seed=7):
    rng = np.random.default_rng(rng_seed)
    boots = wins[rng.integers(0, wins.size, size=(n_boot, wins.size))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    mean = float(wins.mean())
    return mean * 100, max(0.0, (mean - lo) * 100), max(0.0, (hi - mean) * 100)


def _load_csv_wins(path: Path, pred) -> np.ndarray:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return np.asarray([int(r["win"]) for r in rows if pred(r)], dtype=float)


def _load_json(name: str) -> dict:
    return json.loads((SD / name).read_text(encoding="utf-8"))


def _2v2_reference() -> dict:
    """policy -> pole -> (mean, elo, ehi)."""
    g = SD / "pi_g_eval_rows.csv"
    s = SD / "specialist_baseline_eval_rows.csv"
    out = {}
    for pol, path, key in (
        ("pi_G", g, "policy"),
        ("pi_A", s, "policy"),
        ("pi_B", s, "policy"),
    ):
        out[pol] = {}
        for pole in ("A", "B"):
            wins = _load_csv_wins(path, lambda r, p=pol, po=pole, k=key: r[k] == p and r["pole"] == po)
            out[pol][pole] = _mean_ci_pct(wins)
    return out


def _2v2_share_cells() -> dict:
    """share_label -> {z0_A, z1_A, z0_B, z1_B} as percent means (from sealed JSON)."""
    out = {}
    for label, (fname, cell_key, _gate_key) in SHARE_FILES.items():
        blob = _load_json(fname)
        cells = blob[cell_key]
        out[label] = {
            "z0": {"A": cells["z0_poleA"] * 100, "B": cells["z0_poleB"] * 100},
            "z1": {"A": cells["z1_poleA"] * 100, "B": cells["z1_poleB"] * 100},
        }
    return out


def _2v2_share_deltas() -> dict:
    """share_label -> (da, da_lo, da_hi, db, db_lo, db_hi, gate) in pp."""
    out = {}
    for label, (fname, _cell_key, gate_key) in SHARE_FILES.items():
        g = _load_json(fname)[gate_key]
        da, db = g["delta_A"], g["delta_B"]
        gate = "PASS" if g.get("passes") else "FAIL"
        out[label] = (
            da["mean"] * 100,
            (da["mean"] - da["lcb95"]) * 100,
            (da["ucb95"] - da["mean"]) * 100,
            db["mean"] * 100,
            (db["mean"] - db["lcb95"]) * 100,
            (db["ucb95"] - db["mean"]) * 100,
            gate,
        )
    return out


def _status_panel(ax, title: str, lines: list[str], tone: str = "pending") -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    face = {"ok": "#E8F5E9", "fail": "#FFEBEE", "pending": "#FFF8E1"}[tone]
    edge = {"ok": "#2E7D32", "fail": "#C62828", "pending": "#F9A825"}[tone]
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.05, 0.08), 0.90, 0.84, boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=face, edgecolor=edge, linewidth=1.2, transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.text(0.5, 0.82, title, transform=ax.transAxes, ha="center", va="top",
            fontsize=9, fontweight="bold", color=edge)
    y = 0.62
    for line in lines:
        ax.text(0.5, y, line, transform=ax.transAxes, ha="center", va="top",
                fontsize=7.5, color="#333333")
        y -= 0.14


def build_winrate_figure() -> dict:
    apply_style()
    ref = _2v2_reference()
    share = _2v2_share_cells()

    fig = plt.figure(figsize=(TWO_COLUMN, 6.4))
    # 3 scale columns; within 2v2: reference row + 4 share rows, each with A|B
    # Simpler: top reference (2v2 only meaningful); then 4 share rows x 3 scales
    outer = fig.add_gridspec(5, 3, height_ratios=[1.15, 1, 1, 1, 1], hspace=0.55, wspace=0.28)

    # Column titles
    for col, name in enumerate(("2v2", "4v4", "6v6")):
        fig.text(
            0.18 + col * 0.30, 0.98, name, ha="center", va="top",
            fontsize=11, fontweight="bold",
        )

    # --- Row 0: reference policies (2v2 filled; others status) ---
    ax_ref_a = fig.add_subplot(outer[0, 0])
    # split 2v2 reference into A/B via twin layout inside column: use 2 mini axes
    # Rebuild row 0 as nested gridspec for 2v2 only
    # Actually draw reference as two bar groups on one axis for 2v2
    pols = ("pi_G", "pi_A", "pi_B")
    xs = np.arange(len(pols))
    width = 0.35
    for i, pol in enumerate(pols):
        ma, elo_a, ehi_a = ref[pol]["A"]
        mb, elo_b, ehi_b = ref[pol]["B"]
        c = COLORS["control"] if pol == "pi_G" else COLORS["A" if pol == "pi_A" else "B"]
        ax_ref_a.bar(i - width / 2, ma, width, yerr=[[elo_a], [ehi_a]], color=c,
                     edgecolor="black", linewidth=0.5, capsize=2,
                     error_kw={"elinewidth": 0.7}, label=("vs A" if i == 0 else None))
        ax_ref_a.bar(i + width / 2, mb, width, yerr=[[elo_b], [ehi_b]], color=c,
                     edgecolor="black", linewidth=0.5, capsize=2, alpha=0.45,
                     hatch="..", error_kw={"elinewidth": 0.7},
                     label=("vs B" if i == 0 else None))
        ax_ref_a.text(i - width / 2, ma + ehi_a + 1.5, f"{ma:.0f}", ha="center", fontsize=6)
        ax_ref_a.text(i + width / 2, mb + ehi_b + 1.5, f"{mb:.0f}", ha="center", fontsize=6)
    ax_ref_a.set_xticks(xs)
    ax_ref_a.set_xticklabels([r"$\pi_G$", r"$\pi_A$", r"$\pi_B$"], fontsize=8)
    ax_ref_a.set_ylim(0, 100)
    ax_ref_a.set_ylabel("Win rate (%)", fontsize=8)
    ax_ref_a.set_title("Reference (specialists PASS)", fontsize=8.5, fontweight="bold")
    ax_ref_a.legend(loc="upper right", frameon=False, fontsize=6.5, ncol=2)
    ax_ref_a.spines["top"].set_visible(False)
    ax_ref_a.spines["right"].set_visible(False)

    ax_ref_4 = fig.add_subplot(outer[0, 1])
    _status_panel(
        ax_ref_4,
        "4v4 reference",
        [
            "Demand: CERTIFIED (C2 and B3)",
            "C2 specialists: FAIL (sealed)",
            "B3 crossover: not sealed",
            "pi_G: not available",
            "Latent rows: WITHHELD",
        ],
        tone="fail",
    )

    ax_ref_6 = fig.add_subplot(outer[0, 2])
    _status_panel(
        ax_ref_6,
        "6v6 reference",
        [
            "Demand: CERTIFIED",
            "Learned specialists: none",
            "pi_G: not available",
            "Latent rows: WITHHELD",
            "Ladder: not run (by design)",
        ],
        tone="pending",
    )

    # --- Rows 1-4: sharing levels ---
    for r, label in enumerate(SHARE_LABELS):
        # 2v2: A and B as grouped z0/z1 bars on one axis
        ax = fig.add_subplot(outer[r + 1, 0])
        cells = share[label]
        # positions: Pole A group at 0, Pole B at 1
        for g_i, pole in enumerate(("A", "B")):
            for z_i, z in enumerate(("z0", "z1")):
                x = g_i + (z_i - 0.5) * 0.36
                val = cells[z][pole]
                ax.bar(
                    x, val, width=0.32,
                    color=COLORS["A" if z == "z0" else "B"],
                    hatch="///", edgecolor="black", linewidth=0.5,
                )
                ax.text(x, val + 1.5, f"{val:.0f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pole A", "Pole B"], fontsize=7.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Win %", fontsize=7.5)
        gate = _2v2_share_deltas()[label][-1]
        ax.set_title(f"{label}  [{gate}]", fontsize=8, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if r == 0:
            ax.annotate(
                r"$z_0>z_1$", xy=(0, 85), fontsize=6.5, color=COLORS["A"], ha="center"
            )
            ax.annotate(
                r"$z_1>z_0$", xy=(1, 85), fontsize=6.5, color=COLORS["B"], ha="center"
            )

        # 4v4 / 6v6: withheld
        for col, scale in ((1, "4v4"), (2, "6v6")):
            axp = fig.add_subplot(outer[r + 1, col])
            _status_panel(
                axp,
                f"{scale}: {label}",
                [
                    "Not earned yet",
                    "(needs specialist",
                    "crossover PASS first)",
                ],
                tone="pending",
            )

    handles = [
        mpatches.Patch(facecolor=COLORS["A"], hatch="///", edgecolor="black", label=r"$z_0$"),
        mpatches.Patch(facecolor=COLORS["B"], hatch="///", edgecolor="black", label=r"$z_1$"),
        mpatches.Patch(facecolor=COLORS["control"], edgecolor="black", label=r"$\pi_G$ / solid specialists"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=3, frameon=False, fontsize=7.5)

    caption = (
        "Cross-scale discipline: latent/sharing rows appear only after sealed learned-specialist "
        "crossover PASS. 2v2 is complete (Share-Macro OWN_GATE FAIL via $\\Delta_A$ LCB$=0$). "
        "4v4 C2 specialists FAIL -- B3 pending; 6v6 has demand only. "
        "Solid vs hatched: reference policies vs forced-$z$ latent codes."
    )
    fig.text(0.5, -0.01, caption, ha="center", va="top", fontsize=7.2, style="italic")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.08)

    return save_figure(fig, "fig_cross_scale_winrate")


def build_delta_figure() -> dict:
    apply_style()
    deltas = _2v2_share_deltas()
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COLUMN, 2.9), sharey=True)

    # 2v2 filled
    ax = axes[0]
    xs = np.arange(len(SHARE_LABELS))
    da = [deltas[k][0] for k in SHARE_LABELS]
    da_lo = [deltas[k][1] for k in SHARE_LABELS]
    da_hi = [deltas[k][2] for k in SHARE_LABELS]
    db = [deltas[k][3] for k in SHARE_LABELS]
    db_lo = [deltas[k][4] for k in SHARE_LABELS]
    db_hi = [deltas[k][5] for k in SHARE_LABELS]
    gates = [deltas[k][6] for k in SHARE_LABELS]

    ax.axhline(0.0, color=COLORS["zero"], lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.errorbar(
        xs - 0.06, da, yerr=[da_lo, da_hi], color=COLORS["A"], ls=LINESTYLES["A"],
        marker=MARKERS["A"], ms=5, mec="white", mew=0.5, lw=1.2,
        capsize=3, elinewidth=0.8, label=r"$\Delta_A$", zorder=2,
    )
    ax.errorbar(
        xs + 0.06, db, yerr=[db_lo, db_hi], color=COLORS["B"], ls=LINESTYLES["B"],
        marker=MARKERS["B"], ms=5, mec="white", mew=0.5, lw=1.2,
        capsize=3, elinewidth=0.8, label=r"$\Delta_B$", zorder=2,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(SHARE_LABELS, fontsize=6.5, rotation=15, ha="right")
    ax.set_ylim(-20, 45)
    ax.set_ylabel(r"Specialization $\Delta$ (pp)")
    ax.set_title("2v2 (complete)", fontsize=9.5, fontweight="bold")
    ax.text(
        0.5, 0.02, "gate " + "·".join("P" if g == "PASS" else "F" for g in gates),
        transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color="#444444",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 4v4 / 6v6 pending
    _status_panel(
        axes[1],
        "4v4 Δ summary",
        [
            "Demand CERTIFIED | C2 specialists FAIL",
            "B3 crossover not sealed",
            "No Share-* Delta yet",
            "Do not fill until specialist PASS",
        ],
        tone="fail",
    )
    axes[1].set_title("4v4 (incomplete)", fontsize=9.5, fontweight="bold")

    _status_panel(
        axes[2],
        "6v6 Delta summary",
        [
            "Demand CERTIFIED",
            "No learned specialists",
            "No Share-* Delta yet",
            "Ladder not authorized yet",
        ],
        tone="pending",
    )
    axes[2].set_title("6v6 (incomplete)", fontsize=9.5, fontweight="bold")

    caption = (
        r"Central question: as team size and parameter sharing increase, where does "
        r"$\Delta_A/\Delta_B$ survive? Only sealed PASS specialist scales unlock Share-* rows. "
        r"2v2: Share-Encoder preserves the gate; Share-Macro fails via $\Delta_A$."
    )
    fig.text(0.5, -0.06, caption, ha="center", va="top", fontsize=7.5, style="italic")
    fig.subplots_adjust(wspace=0.18, bottom=0.22, top=0.88, left=0.08, right=0.98)

    return save_figure(fig, "fig_cross_scale_delta")


def main() -> None:
    p1 = build_winrate_figure()
    p2 = build_delta_figure()
    print({"winrate": p1, "delta": p2})


if __name__ == "__main__":
    main()
