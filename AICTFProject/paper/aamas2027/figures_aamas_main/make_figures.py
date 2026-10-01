"""Main-paper AAMAS figure set, built around one claim: the role-allocated system keeps two
distinct, selectable opponent-conditioned strategies (Delta_A > 0 and Delta_B > 0).

Palette (fixed across every figure):
  ours = navy; baselines = graded grays; Strategy A / Delta_A = blue; Strategy B / Delta_B = amber.
All numbers come from the sealed n=128 crossover rows; intervals are 95% percentile bootstrap
over seeds (20000 resamples, rng 7), via the loaders in ../paper_figures.py.

Run (from AICTFProject):
  ./.venv/Scripts/python.exe paper/aamas2027/figures_aamas_main/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import paper_figures as pf  # noqa: E402

OURS = "#1F4E79"
# Progressive grayscale: light → dark so baselines stay distinguishable in print/PDF.
BASE = {"Generalist": "#E8E8E8", "Specialists (no roles)": "#B0B0B0", "Share-Encoder": "#787878",
        "Fully Shared+$z$": "#3A3A3A"}
COLOR = {**BASE, "Ours (heuristic roles)": OURS}
C_A, C_A_DARK, C_B = "#3A7DC0", "#0E2F57", "#D9922E"
INK, GRID, BAND = pf.INK, pf.GRID, "#EAF1F8"
LABEL = {"Generalist": "Generalist", "Specialists (no roles)": "No roles", "Share-Encoder": "Share-Encoder",
         "Fully Shared+$z$": "Fully Shared+$z$", "Ours (heuristic roles)": "Ours (roles)"}
SCALES, METHODS = pf.SCALES, pf.METHODS
MARKER = pf.SCALE_MARKER
EKW = dict(elinewidth=0.7, capsize=1.8, capthick=0.7)

# 2v2 mechanism comparison uses the role-allocated run that shares the no-role seed block.
OURS_PAIRED_2V2 = ("standardized_2v2_diag_split_specialist_crossover_eval_rows.csv", "spec")


def export(fig, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(HERE / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", HERE / f"{stem}.pdf")


def _grid_y(ax) -> None:
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def _err(st) -> list[list[float]]:
    return [[st[0] - st[1]], [st[2] - st[0]]]


# ----------------------------------------------------------------- Figure 1: complementarity
def fig1_complementarity(stats: dict) -> None:
    fig, ax = plt.subplots(figsize=(3.33, 3.55))
    lo, hi = -0.30, 0.70
    ax.fill_between([0, hi], 0, hi, color=BAND, lw=0, zorder=0)
    ax.axhline(0, color=INK, lw=0.8, zorder=1)
    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    for (scale, method), st in stats.items():
        ours = method.startswith("Ours")
        (ma, *_), (mb, *_) = st["A"], st["B"]
        col = COLOR[method]
        ax.errorbar(ma, mb, xerr=_err(st["A"]), yerr=_err(st["B"]), fmt="none", ecolor=col,
                    elinewidth=0.9 if ours else 0.6, alpha=0.9 if ours else 0.5, zorder=3 if ours else 2)
        ax.plot(ma, mb, marker=MARKER[scale], color=col, ms=7 if ours else 5.2, mec="white", mew=0.6,
                ls="none", zorder=5 if ours else 4)
    q = dict(fontsize=7, color="#555555", ha="center", va="center", style="italic")
    ax.text(0.40, 0.655, "each strategy wins\nits own pole", linespacing=1.1, **q)
    ax.text(0.53, -0.25, "B never worth\nselecting", linespacing=1.1, **q)
    ax.text(-0.15, 0.655, "A never worth\nselecting", linespacing=1.1, **q)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ticks = np.arange(-0.2, 0.61, 0.2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(axis="x", length=2.5)
    ax.set_xlabel(r"$\Delta_A$: Strategy A's advantage on Pole A")
    ax.set_ylabel(r"$\Delta_B$: Strategy B's advantage on Pole B")
    mh = [Line2D([], [], color=COLOR[m], marker="o", ls="none", ms=5.5) for m in METHODS]
    sh = [Line2D([], [], color=INK, marker=MARKER[s], ls="none", ms=5, mfc="white") for s in SCALES]
    leg = fig.legend(mh, [LABEL[m] for m in METHODS], loc="lower center", bbox_to_anchor=(0.5, -0.06),
                     ncol=4, frameon=False, handletextpad=0.2, columnspacing=0.8, fontsize=7.2)
    fig.add_artist(leg)
    fig.legend(sh, list(SCALES), loc="lower center", bbox_to_anchor=(0.5, -0.115), ncol=3, frameon=False,
               handletextpad=0.2, columnspacing=1.4, fontsize=7.2)
    export(fig, "fig1_complementarity")


# --------------------------------------------------------------------- Figure 2: mechanism
def fig2_mechanism(cells: dict) -> dict:
    """No roles -> ours: Pole-A strategy's win rate on Pole B, and Delta_B."""
    ours_cells = {"2v2": pf.load_cells(*OURS_PAIRED_2V2),
                  "4v4": cells[("4v4", "Ours (heuristic roles)")],
                  "6v6": cells[("6v6", "Ours (heuristic roles)")]}
    rows = {}
    for s in SCALES:
        nr, ou = cells[(s, "Specialists (no roles)")], ours_cells[s]
        rows[s] = {
            "wr": (pf.winrates(nr)[("pi_A", "B")], pf.winrates(ou)[("pi_A", "B")]),
            "dB": (pf.deltas(nr)["B"], pf.deltas(ou)["B"]),
        }
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.85), sharey=True, gridspec_kw={"wspace": 0.12})
    ylab = {"2v2": "2v2", "4v4": "4v4$^\\dagger$", "6v6": "6v6"}
    y = {s: i for i, s in enumerate(reversed(SCALES))}
    panels = (("wr", "Win rate of Strategy A on Pole B\n(lower = A no longer an all-purpose policy)", (0, 1.0)),
              ("dB", "$\\Delta_B$: Strategy B's advantage on Pole B\n(higher = B worth selecting)", (-0.3, 0.7)))
    for ax, (key, title, xlim) in zip(axes, panels):
        if key == "dB":
            ax.axvline(0, color=INK, lw=0.8, zorder=1)
        for s in SCALES:
            a, b = rows[s][key]
            ax.plot([a[0], b[0]], [y[s]] * 2, color="#C9C9C9", lw=2.2, zorder=2, solid_capstyle="round")
            ax.errorbar(a[0], y[s], xerr=_err(a), fmt="o", color=BASE["Specialists (no roles)"], ms=6,
                        mec="white", mew=0.6, ecolor=BASE["Specialists (no roles)"], zorder=3, **EKW)
            ax.errorbar(b[0], y[s], xerr=_err(b), fmt="o", color=OURS, ms=6, mec="white", mew=0.6,
                        ecolor=OURS, zorder=4, **EKW)
        ax.set_xlim(*xlim)
        ax.set_title(title, fontsize=7.8, pad=4, linespacing=1.2)
        ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=2.5)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(list(y.values()))
    axes[0].set_yticklabels([ylab[s] for s in y])
    axes[0].set_ylim(-0.6, len(SCALES) - 0.4)
    fig.legend([Line2D([], [], color=BASE["Specialists (no roles)"], marker="o", ls="none", ms=6),
                Line2D([], [], color=OURS, marker="o", ls="none", ms=6)],
               ["No roles", "Ours (roles)"], loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2,
               frameon=False, handletextpad=0.3, columnspacing=1.6)
    export(fig, "fig2_mechanism")
    return {s: {k: [list(v[0]), list(v[1])] for k, v in r.items()} for s, r in rows.items()}


# --------------------------------------------------------------- Figure 3: deployed win rate
def fig3_winrate(cells: dict) -> None:
    methods = ("Generalist",) + METHODS
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=True, gridspec_kw={"wspace": 0.08})
    w = 0.15
    for ax, pole, title in zip(axes, ("A", "B"), ("Against Pole A (Strategy A deployed)",
                                                    "Against Pole B (Strategy B deployed)")):
        for k, method in enumerate(methods):
            strat = "pi_G" if method == "Generalist" else f"pi_{pole}"
            st = [pf.winrates(cells[(s, method)])[(strat, pole)] for s in SCALES]
            xs = np.arange(len(SCALES)) + (k - (len(methods) - 1) / 2) * w
            ax.bar(xs, [v[0] for v in st], w * 0.9, color=COLOR[method], lw=0, zorder=2,
                   yerr=pf._yerr(st), error_kw=dict(ecolor=INK, **{**EKW, "capsize": 1.2, "elinewidth": 0.6}))
        ax.set_xticks(np.arange(len(SCALES)))
        ax.set_xticklabels(SCALES)
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=8, pad=4)
        _grid_y(ax)
    axes[0].set_ylabel("Win rate")
    axes[1].spines["left"].set_visible(False)
    axes[1].tick_params(axis="y", length=0)
    fig.legend([Patch(color=COLOR[m]) for m in methods], [LABEL[m] for m in methods], loc="upper center",
               bbox_to_anchor=(0.5, 1.12), ncol=len(methods), frameon=False, handlelength=1.0,
               handleheight=0.8, columnspacing=1.4)
    export(fig, "fig3_deployed_winrate")


# ---------------------------------------------------------------------- Figure 4: robustness
def fig4_robustness() -> dict:
    stats = {}
    for s in SCALES:
        for c in pf.NOISE_CONDS:
            name = f"standardized_{s}_noise_{pf.NOISE_FILE[c]}_specialist_crossover_eval_rows.csv"
            stats[(s, c)] = pf.deltas(pf.load_cells(name, "spec"))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0), sharey=True, gridspec_kw={"wspace": 0.08})
    x = np.arange(len(pf.NOISE_CONDS))
    for ax, s in zip(axes, SCALES):
        ax.axvspan(-0.5, 0.5, color=BAND, lw=0, zorder=0)
        ax.axhline(0, color=INK, lw=0.8, zorder=1)
        for key, col, mk, dx in (("A", C_A, "o", -0.12), ("B", C_B, "s", 0.12)):
            st = [stats[(s, c)][key] for c in pf.NOISE_CONDS]
            ax.errorbar(x + dx, [v[0] for v in st], yerr=pf._yerr(st), fmt=mk, color=col, ms=5,
                        mec="white", mew=0.5, ecolor=col, zorder=3, **EKW)
        ax.set_xticks(x)
        ax.set_xticklabels(["Nominal", "Localiz.", "Motion", "Delay"])
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_ylim(-0.2, 0.7)
        ax.set_title(s, fontsize=8.5, fontweight="bold", pad=4)
        _grid_y(ax)
    axes[0].set_ylabel(r"Specialization $\Delta$")
    for ax in axes[1:]:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.legend([Line2D([], [], color=C_A, marker="o", ls="none", ms=5),
                Line2D([], [], color=C_B, marker="s", ls="none", ms=5)],
               [r"$\Delta_A$ (Strategy A on Pole A)", r"$\Delta_B$ (Strategy B on Pole B)"],
               loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, frameon=False, columnspacing=1.8)
    export(fig, "fig4_robustness")
    return {f"{s}|{c}": v for (s, c), v in stats.items()}


# --------------------------------------------------------------- Figure 5: 4v4 trajectories
def _paths(ax, cell: dict, strat: str) -> None:
    x, y = cell["blue_x"][: pf.TRAJ_STEPS], cell["blue_y"][: pf.TRAJ_STEPS]
    roles = cell["roles"][0]
    for i in range(x.shape[1]):
        if strat == "pi_B":
            col, lw, z = C_B, 1.0, 3
        elif np.isfinite(roles[i]) and roles[i] < 0.5:
            col, lw, z = C_A_DARK, 1.5, 4
        else:
            col, lw, z = C_A, 1.0, 3
        ax.plot(x[:, i], y[:, i], color=col, lw=lw, alpha=0.9, zorder=z, solid_capstyle="round")
        ax.plot(x[0, i], y[0, i], marker="o", color="#111111", ms=2.4, zorder=5, ls="none")
        ax.plot(x[-1, i], y[-1, i], marker="o", color=col, mec="white", mew=0.4, ms=3.6, zorder=5, ls="none")


def fig5_trajectories() -> int:
    man, cells = pf._load_traj("4v4", require_occupancy=False)
    seed = int(man["trajectory_seed"])
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.25), gridspec_kw={"wspace": 0.08})
    order = (("A", "pi_A"), ("A", "pi_B"), ("B", "pi_A"), ("B", "pi_B"))
    for ax, (pole, strat) in zip(axes, order):
        cell = cells[(strat, pole, seed)]
        pf._field(ax, cell)
        _paths(ax, cell, strat)
        ax.set_title("Strategy A" if strat == "pi_A" else "Strategy B", fontsize=8, pad=3)
    fig.canvas.draw()
    for pair, text in (((0, 1), "Against Pole A"), ((2, 3), "Against Pole B")):
        b0, b1 = axes[pair[0]].get_position(), axes[pair[1]].get_position()
        fig.text((b0.x0 + b1.x1) / 2, b0.y1 + 0.13, text, ha="center", va="bottom", fontsize=8.5,
                 fontweight="bold")
        fig.add_artist(Line2D([b0.x0 + 0.01, b1.x1 - 0.01], [b0.y1 + 0.115] * 2, color=INK, lw=0.6))
    handles = [Line2D([], [], color=C_A, lw=1.0), Line2D([], [], color=C_A_DARK, lw=1.5),
               Line2D([], [], color=C_B, lw=1.0),
               Line2D([], [], color="#111111", marker="o", ls="none", ms=2.6),
               Line2D([], [], color="#111111", marker="s", ls="none", ms=5),
               Line2D([], [], color="#111111", marker="^", mfc="white", ls="none", ms=5.5)]
    fig.legend(handles, ["Strategy A: attack role", "Strategy A: defend role", "Strategy B", "Start",
                         "Own flag", "Opponent flag"], loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.06), handlelength=1.5, columnspacing=1.1, fontsize=7.2)
    export(fig, "fig5_trajectories_4v4")
    return seed


# -------------------------------------------------------------- Figure 6: win vs score margin
def fig6_margin(stats: dict, margin: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.33, 1.95), gridspec_kw={"wspace": 0.42})
    x = np.arange(len(SCALES))
    w = 0.34
    ours = "Ours (heuristic roles)"
    for ax, src, title, ylab, ylim in (
            (axes[0], {s: stats[(s, ours)] for s in SCALES}, "Win/loss", r"$\Delta$ (win rate)", (-0.05, 0.7)),
            (axes[1], margin, "Score margin", r"$\Delta$ (goals per game)", (-0.2, 1.8))):
        ax.axvspan(1.5, 2.5, color=BAND, lw=0, zorder=0)
        ax.set_ylabel(ylab)
        for key, col, dx in (("A", C_A, -w / 2), ("B", C_B, w / 2)):
            st = [src[s][key] for s in SCALES]
            ax.bar(x + dx, [v[0] for v in st], w * 0.92, color=col, lw=0, zorder=2, yerr=pf._yerr(st),
                   error_kw=dict(ecolor=INK, **{**EKW, "capsize": 1.4, "elinewidth": 0.6}))
        ax.axhline(0, color=INK, lw=0.8, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(SCALES)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=8, pad=4)
        _grid_y(ax)
    fig.legend([Patch(color=C_A), Patch(color=C_B)], [r"$\Delta_A$", r"$\Delta_B$"], loc="upper center",
               bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False, handlelength=1.0, handleheight=0.8)
    export(fig, "fig6_margin")


def main() -> int:
    pf.apply_style()
    cells = {k: pf.load_cells(n, kind) for k, (n, kind) in pf.ROWS.items()}
    stats = {(s, m): pf.deltas(cells[(s, m)]) for s in SCALES for m in METHODS}
    margin = {s: pf.deltas(pf.load_cells(pf.ROWS[(s, "Ours (heuristic roles)")][0], "spec", field="margin"))
              for s in SCALES}

    fig1_complementarity(stats)
    mech = fig2_mechanism(cells)
    fig3_winrate(cells)
    noise = fig4_robustness()
    seed = fig5_trajectories()
    fig6_margin(stats, margin)

    summary = {
        "source": "sealed n=128 rows; 95% percentile bootstrap over seeds (20000, rng 7)",
        "fig1_delta_win": {f"{s}|{m}": v for (s, m), v in stats.items()},
        "fig2_mechanism_no_roles_vs_ours": mech,
        "fig4_noise": noise,
        "fig5_trajectory_seed_4v4": seed,
        "fig6_margin_ours": margin,
    }
    (HERE / "figure_stats.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
