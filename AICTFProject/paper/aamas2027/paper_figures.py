"""All AAMAS 2027 Experimental Evaluation figures, from the sealed n=128 rows.

Figures carry no titles, annotations or captions; every explanatory sentence lives in the
LaTeX caption. Semantic colors are fixed across the paper: blue = Pole-A strategy / Delta_A,
vermillion = Pole-B strategy / Delta_B. Error bars are 95% paired percentile bootstrap
intervals over seeds (20000 resamples, rng seed 7), the procedure of the sealed evaluator.

Agent-movement figures read replays from experiments/export_paper_trajectories.py and refuse
to draw unless every replayed terminal matches its sealed row.

Run (from AICTFProject):
  ./.venv/Scripts/python.exe paper/aamas2027/paper_figures.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
TRAJ = ROOT / "artifacts" / "qualitative_capture" / "paper_suite_trajectories"
FIG = Path(__file__).resolve().parent / "figures"

C_A, C_B = "#0072B2", "#D55E00"
C_A_DARK = "#003F66"
INK, GRID, BAND = "#333333", "#E6E6E6", "#F3F3F3"
ONE_COL, TWO_COL = 3.33, 7.0
N_BOOT, BOOT_SEED = 20_000, 7

SCALES = ("2v2", "4v4", "6v6")
METHODS = ("Specialists (no roles)", "Share-Encoder", "Fully Shared+$z$", "Ours (heuristic roles)")
SHORT = {"Generalist": "Generalist", "Specialists (no roles)": "No roles", "Share-Encoder": "Share-Enc.",
         "Fully Shared+$z$": "Shared+$z$", "Ours (heuristic roles)": "Ours"}
METHOD_COLOR = {"Generalist": "#BBBBBB", "Specialists (no roles)": "#E69F00", "Share-Encoder": "#009E73",
                "Fully Shared+$z$": "#CC79A7", "Ours (heuristic roles)": "#222222"}
SCALE_MARKER = {"2v2": "o", "4v4": "s", "6v6": "^"}

ROWS = {
    ("2v2", "Specialists (no roles)"): ("standardized_2v2_diag_presplit_specialist_crossover_eval_rows.csv", "spec"),
    ("2v2", "Share-Encoder"): ("standardized_2v2_share_encoder_crossover_eval_rows.csv", "z"),
    ("2v2", "Fully Shared+$z$"): ("standardized_2v2_fully_shared_z_crossover_eval_rows.csv", "z"),
    ("2v2", "Ours (heuristic roles)"): ("standardized_2v2_split_k1_confirmatory_specialist_crossover_eval_rows.csv", "spec"),
    ("2v2", "Generalist"): ("standardized_2v2_generalist_crossover_eval_rows.csv", "gen"),
    ("4v4", "Specialists (no roles)"): ("confirmatory_b3_entity_repair_corrected_specialist_crossover_eval_rows.csv", "spec"),
    ("4v4", "Share-Encoder"): ("standardized_4v4_share_encoder_crossover_eval_rows.csv", "z"),
    ("4v4", "Fully Shared+$z$"): ("standardized_4v4_fully_shared_z_crossover_eval_rows.csv", "z"),
    ("4v4", "Ours (heuristic roles)"): ("defend_attack_split_policy_a_v1_confirmatory_v1_specialist_crossover_eval_rows.csv", "spec"),
    ("4v4", "Generalist"): ("standardized_4v4_generalist_crossover_eval_rows.csv", "gen"),
    ("6v6", "Specialists (no roles)"): ("standardized_6v6_norole_specialist_crossover_eval_rows.csv", "spec"),
    ("6v6", "Share-Encoder"): ("standardized_6v6_share_encoder_crossover_eval_rows.csv", "z"),
    ("6v6", "Fully Shared+$z$"): ("standardized_6v6_fully_shared_z_crossover_eval_rows.csv", "z"),
    ("6v6", "Ours (heuristic roles)"): ("standardized_6v6_split_k1_confirmatory_specialist_crossover_eval_rows.csv", "spec"),
    ("6v6", "Generalist"): ("standardized_6v6_generalist_crossover_eval_rows.csv", "gen"),
}
NOISE_CONDS = ("Nominal", "Localization", "Motion", "Delay")
NOISE_FILE = {"Nominal": "nominal", "Localization": "localization_medium", "Motion": "motion_medium",
              "Delay": "delay_medium"}


# ------------------------------------------------------------------------------ style / io
def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.size": 0,
        "ytick.major.size": 2.5,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def export(fig, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", FIG / f"{stem}.pdf")


def boot_ci(x: np.ndarray) -> tuple[float, float, float]:
    """Mean and 95% percentile bootstrap interval of the per-seed values x."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    means = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def load_cells(csv_name: str, kind: str, field: str = "win") -> dict[tuple[str, str], dict[int, float]]:
    """{(strategy, pole): {seed: value}} with strategy in pi_A / pi_B / pi_G."""
    zmap = {"0": "pi_A", "1": "pi_B"} if kind == "z" else {"0": "pi_G"}
    out: dict[tuple[str, str], dict[int, float]] = {}
    with (SD / csv_name).open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            strat = r["policy"] if kind == "spec" else zmap[str(r["z"])]
            out.setdefault((strat, r["pole"]), {})[int(r["seed"])] = float(r[field])
    return out


def _vec(cells: dict, strat: str, pole: str, seeds: list[int]) -> np.ndarray:
    return np.array([cells[(strat, pole)][s] for s in seeds], dtype=np.float64)


def deltas(cells: dict) -> dict:
    seeds = sorted(cells[("pi_A", "A")])
    da = _vec(cells, "pi_A", "A", seeds) - _vec(cells, "pi_B", "A", seeds)
    db = _vec(cells, "pi_B", "B", seeds) - _vec(cells, "pi_A", "B", seeds)
    if len(seeds) != 128:
        raise SystemExit(f"expected 128 seeds, got {len(seeds)}")
    return {"A": boot_ci(da), "B": boot_ci(db), "n": len(seeds)}


def winrates(cells: dict) -> dict:
    out = {}
    for (strat, pole), by in cells.items():
        out[(strat, pole)] = boot_ci(np.array([by[s] for s in sorted(by)]))
    return out


def _yerr(stats: list[tuple[float, float, float]]) -> np.ndarray:
    m = np.array([s[0] for s in stats])
    return np.vstack([m - np.array([s[1] for s in stats]), np.array([s[2] for s in stats]) - m])


def _zero_line(ax) -> None:
    ax.axhline(0.0, color=INK, lw=0.8, zorder=3)
    ax.spines["bottom"].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def _group_labels(ax, centers: list[float], labels: list[str], *, offset: float = -0.2) -> None:
    sec = ax.secondary_xaxis(offset)
    sec.set_xticks(centers)
    sec.set_xticklabels(labels, fontsize=8.5, fontweight="bold")
    sec.tick_params(length=0)
    sec.spines["bottom"].set_visible(False)


def delta_bars(ax, groups: list[str], sub: list[str], stats: dict, *, highlight: str | None = None,
               ylabel: str = r"$\Delta$ (win rate)") -> None:
    """Grouped Delta_A / Delta_B bars with CIs. stats[(group, sub)] = {"A": ci, "B": ci}."""
    w, gap = 0.36, 0.9
    xs, labels, centers = [], [], []
    x = 0.0
    for g in groups:
        first = x
        for s in sub:
            xs.append(x)
            labels.append(s)
            if highlight is not None and s == highlight:
                ax.axvspan(x - 0.5, x + 0.5, color=BAND, zorder=0, lw=0)
            x += 1.0
        centers.append((first + x - 1.0) / 2)
        x += gap
    sa = [stats[(g, s)]["A"] for g in groups for s in sub]
    sb = [stats[(g, s)]["B"] for g in groups for s in sub]
    xs = np.array(xs)
    ekw = dict(elinewidth=0.7, capsize=1.6, capthick=0.7, ecolor=INK)
    ax.bar(xs - w / 2, [s[0] for s in sa], w, color=C_A, lw=0, zorder=2, yerr=_yerr(sa), error_kw=ekw)
    ax.bar(xs + w / 2, [s[0] for s in sb], w, color=C_B, lw=0, zorder=2, yerr=_yerr(sb), error_kw=ekw)
    _zero_line(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, linespacing=0.95)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.set_ylabel(ylabel)
    if len(groups) > 1:
        _group_labels(ax, centers, groups)


def delta_legend(fig_or_ax, **kw) -> None:
    handles = [Patch(color=C_A), Patch(color=C_B)]
    fig_or_ax.legend(handles, [r"$\Delta_A$", r"$\Delta_B$"], frameon=False, ncol=2,
                     handlelength=1.0, handleheight=0.8, columnspacing=1.0, **kw)


# ------------------------------------------------------------------- figures from sealed rows
def fig_crossover(cells: dict) -> None:
    """Rows = two-strategy systems, columns = team size: win rate of each strategy on each pole."""
    fig, axes = plt.subplots(len(METHODS), 3, figsize=(ONE_COL * 1.45, 4.6), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.12, "hspace": 0.18})
    for i, method in enumerate(METHODS):
        for j, scale in enumerate(SCALES):
            ax = axes[i, j]
            wr = winrates(cells[(scale, method)])
            for strat, color, ls, mk in (("pi_A", C_A, "-", "o"), ("pi_B", C_B, "--", "s")):
                st = [wr[(strat, "A")], wr[(strat, "B")]]
                ax.errorbar([0, 1], [s[0] for s in st], yerr=_yerr(st), color=color, ls=ls, marker=mk,
                            ms=3.6, lw=1.2, elinewidth=0.7, capsize=1.6, zorder=3)
            if method == "Ours (heuristic roles)":
                ax.set_facecolor(BAND)
            ax.set_xlim(-0.3, 1.3)
            ax.set_ylim(0.0, 1.04)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Pole A", "Pole B"])
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
            ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel("Win rate")
    handles = [Line2D([], [], color=C_A, ls="-", marker="o", ms=3.6), Line2D([], [], color=C_B, ls="--", marker="s", ms=3.6)]
    fig.legend(handles, [r"Pole-A strategy", r"Pole-B strategy"], loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    export(fig, "fig_crossover")


def fig_complementarity(stats: dict) -> None:
    """Scatter of (Δ_A, Δ_B). Only the complementary quadrant Δ_A>0, Δ_B>0 is shaded."""
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.85))
    allv = [v for st in stats.values() for k in ("A", "B") for v in st[k]]
    lo, hi = min(-0.25, min(allv) - 0.03), max(allv) + 0.03
    # Desirable region only: both specialization directions positive.
    ax.fill_between([0.0, hi], 0.0, hi, color=BAND, lw=0, zorder=0, clip_on=True)
    ax.axhline(0, color=INK, lw=0.7, zorder=1)
    ax.axvline(0, color=INK, lw=0.7, zorder=1)
    for (scale, method), st in stats.items():
        (ma, la, ha), (mb, lb, hb) = st["A"], st["B"]
        col = METHOD_COLOR[method]
        ax.errorbar(ma, mb, xerr=[[ma - la], [ha - ma]], yerr=[[mb - lb], [hb - mb]], fmt="none",
                    ecolor=col, elinewidth=0.6, alpha=0.55, zorder=2)
        ax.plot(ma, mb, marker=SCALE_MARKER[scale], color=col, ms=5.5 if "Ours" in method else 4.5,
                mec="white", mew=0.5, ls="none", zorder=4 if "Ours" in method else 3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\Delta_A$  (A preferred on Pole A)")
    ax.set_ylabel(r"$\Delta_B$  (B preferred on Pole B)")
    # Quiet cue for the shaded meaning (caption still owns the prose).
    ax.text(0.97, 0.97, "ideal region\n(both positive)",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color="#666666",
            linespacing=1.15)
    mh = [Line2D([], [], color=METHOD_COLOR[m], marker="o", ls="none", ms=4.5) for m in METHODS]
    sh = [Line2D([], [], color=INK, marker=SCALE_MARKER[s], ls="none", ms=4.5, mfc="none") for s in SCALES]
    leg1 = ax.legend(mh, [SHORT[m] for m in METHODS], title="Method", loc="upper left",
                     bbox_to_anchor=(1.01, 1.0), frameon=False, handletextpad=0.3, borderaxespad=0,
                     title_fontsize=7.5)
    ax.add_artist(leg1)
    ax.legend(sh, list(SCALES), title="Team size", loc="lower left", bbox_to_anchor=(1.01, 0.0),
              frameon=False, handletextpad=0.3, borderaxespad=0, title_fontsize=7.5)
    export(fig, "fig_complementarity")


def fig_baselines(stats: dict) -> None:
    fig, ax = plt.subplots(figsize=(TWO_COL, 2.2))
    delta_bars(ax, list(SCALES), list(METHODS), stats, highlight="Ours (heuristic roles)")
    two_line = {"Specialists (no roles)": "No\nroles", "Share-Encoder": "Share-\nEnc.",
                "Fully Shared+$z$": "Shared\n+$z$", "Ours (heuristic roles)": "Ours"}
    ax.set_xticklabels([two_line[m] for _ in SCALES for m in METHODS])
    delta_legend(ax, loc="upper right", bbox_to_anchor=(1.0, 1.08))
    export(fig, "fig_baselines")


def fig_worst_pole(stats: dict) -> None:
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.1))
    x = np.arange(len(SCALES))
    for k, method in enumerate(METHODS):
        y = [min(stats[(s, method)]["A"][0], stats[(s, method)]["B"][0]) for s in SCALES]
        ours = "Ours" in method
        ax.plot(x + (k - 1.5) * 0.03, y, color=METHOD_COLOR[method], marker="o", ms=4.5 if ours else 3.8,
                lw=1.6 if ours else 1.1, zorder=4 if ours else 3)
    _zero_line(ax)
    ax.spines["bottom"].set_visible(True)
    ax.set_xticks(x)
    ax.set_xticklabels(SCALES)
    ax.set_xlim(-0.25, len(SCALES) - 0.75)
    ax.set_ylabel(r"$\min(\Delta_A,\Delta_B)$")
    ax.legend([Line2D([], [], color=METHOD_COLOR[m], marker="o", ms=3.8) for m in METHODS],
              [SHORT[m] for m in METHODS], loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
              borderaxespad=0)
    export(fig, "fig_worst_pole")


def fig_deployed_winrate(cells: dict) -> None:
    """Intended-strategy win rate: Pole-A strategy on Pole A; Pole-B strategy on Pole B."""
    methods = ("Generalist",) + METHODS
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.25), sharey=True, gridspec_kw={"wspace": 0.08})
    w = 0.16
    for ax, pole, title in zip(axes, ("A", "B"),
                               ("Pole A\n(intended Pole-A strategy)", "Pole B\n(intended Pole-B strategy)")):
        for k, method in enumerate(methods):
            st = []
            for scale in SCALES:
                wr = winrates(cells[(scale, method)])
                strat = "pi_G" if method == "Generalist" else ("pi_A" if pole == "A" else "pi_B")
                st.append(wr[(strat, pole)])
            xs = np.arange(len(SCALES)) + (k - (len(methods) - 1) / 2) * w
            ax.bar(xs, [s[0] for s in st], w * 0.92, color=METHOD_COLOR[method], lw=0, zorder=2,
                   yerr=_yerr(st), error_kw=dict(elinewidth=0.6, capsize=1.2, capthick=0.6, ecolor=INK))
        ax.set_xticks(np.arange(len(SCALES)))
        ax.set_xticklabels(SCALES)
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=8, pad=4, linespacing=1.15)
        ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Intended-strategy win rate")
    axes[1].spines["left"].set_visible(False)
    axes[1].tick_params(axis="y", length=0)
    fig.legend([Patch(color=METHOD_COLOR[m]) for m in methods], [SHORT[m] for m in methods],
               loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=len(methods), frameon=False,
               handlelength=1.0, handleheight=0.8, columnspacing=1.2)
    export(fig, "fig_deployed_winrate")


def fig_noise() -> dict:
    stats = {}
    for scale in SCALES:
        for cond in NOISE_CONDS:
            name = f"standardized_{scale}_noise_{NOISE_FILE[cond]}_specialist_crossover_eval_rows.csv"
            stats[(scale, cond)] = deltas(load_cells(name, "spec"))
    fig, ax = plt.subplots(figsize=(TWO_COL, 2.1))
    delta_bars(ax, list(SCALES), list(NOISE_CONDS), stats, highlight="Nominal")
    ax.set_xticklabels(["Nominal", "Local.", "Motion", "Delay"] * len(SCALES))
    delta_legend(ax, loc="upper right", bbox_to_anchor=(1.0, 1.08))
    export(fig, "fig_noise")
    return stats


def fig_margin(margin_stats: dict) -> None:
    fig, ax = plt.subplots(figsize=(ONE_COL, 1.9))
    delta_bars(ax, ["Ours"], list(SCALES), {("Ours", s): margin_stats[s] for s in SCALES},
               ylabel=r"$\Delta$ (score margin)")
    delta_legend(ax, loc="upper right", bbox_to_anchor=(1.0, 1.1))
    export(fig, "fig_margin")


# --------------------------------------------------------------- agent-movement figures
def _load_traj(scale: str, *, require_occupancy: bool = True) -> tuple[dict, dict[tuple[str, str, int], dict]]:
    d = TRAJ / scale
    mp = d / "manifest.json"
    if not mp.is_file():
        raise FileNotFoundError(mp)
    man = json.loads(mp.read_text(encoding="utf-8"))
    bad = [k for k, c in man["cells"].items() if c["fidelity"] != "MATCH"]
    if bad:
        raise SystemExit(f"REFUSING: {scale} replays not matching sealed rows: {bad[:4]}")
    traj_seed = int(man["trajectory_seed"])
    need = {(pol, p, traj_seed) for pol in ("pi_A", "pi_B") for p in ("A", "B")}
    have = {(c["policy"], c["pole"], int(c["seed"])) for c in man["cells"].values()}
    missing_traj = sorted(need - have)
    if missing_traj:
        raise FileNotFoundError(f"{scale}: missing trajectory-seed cells {missing_traj}")
    if require_occupancy:
        expected = 4 * (1 + len([s for s in man["occupancy_seeds"] if s != man["trajectory_seed"]]))
        if len(man["cells"]) != expected:
            raise FileNotFoundError(f"{scale}: {len(man['cells'])}/{expected} replays present")
    cells = {}
    for name, c in man["cells"].items():
        z = np.load(d / name)
        cells[(c["policy"], c["pole"], int(c["seed"]))] = {k: z[k] for k in z.files}
    return man, cells


def _field(ax, cell: dict) -> None:
    bx, by = (float(v) for v in cell["bounds"])
    ax.add_patch(Rectangle((-0.5, -0.5), bx + 1, by + 1, fill=False, ec="#999999", lw=0.6, zorder=1))
    ax.plot([bx / 2, bx / 2], [-0.5, by + 0.5], color="#BBBBBB", lw=0.6, ls=":", zorder=1)
    hb, hr = cell["flag_home_blue"], cell["flag_home_red"]
    ax.plot(*hb, marker="s", color="#111111", ms=5.5, zorder=6, ls="none")
    ax.plot(*hr, marker="^", color="#111111", ms=6, mfc="white", mew=1.0, zorder=6, ls="none")
    ax.set_xlim(-0.7, bx + 0.7)
    ax.set_ylim(-0.7, by + 0.7)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


TRAJ_STEPS = 80
POSTURE_SMOOTH = 5


def _paths(ax, cell: dict, color: str, *, defenders: bool, steps: int = TRAJ_STEPS) -> None:
    x, y = cell["blue_x"][:steps], cell["blue_y"][:steps]
    roles = cell["roles"][0]
    for i in range(x.shape[1]):
        is_def = defenders and np.isfinite(roles[i]) and roles[i] < 0.5
        ax.plot(x[:, i], y[:, i], color=C_A_DARK if is_def else color, lw=1.5 if is_def else 1.0,
                alpha=0.95 if is_def else 0.85, zorder=4 if is_def else 3, solid_capstyle="round")
        ax.plot(x[0, i], y[0, i], marker="o", color="#111111", ms=2.4, zorder=5, ls="none")
        ax.plot(x[-1, i], y[-1, i], marker="o", color=C_A_DARK if is_def else color, mec="white", mew=0.4,
                ms=3.6, zorder=5, ls="none")


def fig_trajectories(traj: dict) -> None:
    """Rows = team size; columns = four crossover deployments with explicit labels."""
    col_titles = (
        r"$\pi_A$ vs Pole A",
        r"$\pi_B$ vs Pole A",
        r"$\pi_A$ vs Pole B",
        r"$\pi_B$ vs Pole B",
    )
    fig, axes = plt.subplots(len(SCALES), 4, figsize=(TWO_COL, 5.9),
                             gridspec_kw={"wspace": 0.06, "hspace": 0.12})
    for i, scale in enumerate(SCALES):
        man, cells = traj[scale]
        seed = int(man["trajectory_seed"])
        for j, (pole, strat) in enumerate((("A", "pi_A"), ("A", "pi_B"), ("B", "pi_A"), ("B", "pi_B"))):
            ax = axes[i, j]
            cell = cells[(strat, pole, seed)]
            _field(ax, cell)
            _paths(ax, cell, C_A if strat == "pi_A" else C_B, defenders=strat == "pi_A")
            if j == 2:
                ax.axvline(-0.5, color=INK, lw=0.8, clip_on=False)
            if i == 0:
                ax.set_title(col_titles[j], fontsize=7.5, pad=3)
            if j == 0:
                ax.set_ylabel(scale, fontsize=9, fontweight="bold", labelpad=6)
    # Group headers spanning Pole-A / Pole-B deployments.
    fig.text(0.30, 0.955, "Against Pole A", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    fig.text(0.72, 0.955, "Against Pole B", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    handles = [Line2D([], [], color=C_A, lw=1.0), Line2D([], [], color=C_A_DARK, lw=1.5),
               Line2D([], [], color=C_B, lw=1.0),
               Line2D([], [], color="#111111", marker="s", ls="none", ms=5),
               Line2D([], [], color="#111111", marker="^", mfc="white", ls="none", ms=5.5)]
    fig.legend(handles, ["Pole-A strategy, attack role", "Pole-A strategy, defend role", "Pole-B strategy",
                         "Own flag", "Opponent flag"], loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.01), handlelength=1.6, columnspacing=1.1)
    export(fig, "fig_trajectories")


def _occupancy(cells: dict, strat: str, bins: tuple[int, int]) -> tuple[np.ndarray, tuple[float, float]]:
    pts_x, pts_y, bounds = [], [], None
    for (s, _pole, _seed), c in cells.items():
        if s != strat:
            continue
        bounds = tuple(float(v) for v in c["bounds"])
        live = c["blue_alive"].astype(bool) & ~c["blue_tagged"].astype(bool)
        pts_x.append(c["blue_x"][live])
        pts_y.append(c["blue_y"][live])
    h, _, _ = np.histogram2d(np.concatenate(pts_y), np.concatenate(pts_x), bins=bins,
                             range=[[-0.5, bounds[1] + 0.5], [-0.5, bounds[0] + 0.5]])
    return h / h.sum(), bounds


def fig_occupancy(traj: dict) -> None:
    """Rows = team size; columns = Pole-A strategy, Pole-B strategy, difference. Both poles pooled."""
    cmap_a = LinearSegmentedColormap.from_list("a", ["#FFFFFF", C_A])
    cmap_b = LinearSegmentedColormap.from_list("b", ["#FFFFFF", C_B])
    cmap_d = LinearSegmentedColormap.from_list("d", [C_B, "#FFFFFF", C_A])
    fig, axes = plt.subplots(len(SCALES), 3, figsize=(ONE_COL, ONE_COL * 1.02),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.04})
    for i, scale in enumerate(SCALES):
        _man, cells = traj[scale]
        any_cell = next(iter(cells.values()))
        bx, by = (float(v) for v in any_cell["bounds"])
        bins = (int(round(by)) + 1, int(round(bx)) + 1)
        ha, bounds = _occupancy(cells, "pi_A", bins)
        hb, _ = _occupancy(cells, "pi_B", bins)
        ra, rb = np.sqrt(ha), np.sqrt(hb)
        d = ha - hb
        rd = np.sign(d) * np.sqrt(np.abs(d))
        vmax, dmax = max(ra.max(), rb.max()), np.abs(rd).max()
        ext = (-0.5, bounds[0] + 0.5, -0.5, bounds[1] + 0.5)
        for ax, img, cmap, vmin, vm in ((axes[i, 0], ra, cmap_a, 0, vmax), (axes[i, 1], rb, cmap_b, 0, vmax),
                                        (axes[i, 2], rd, cmap_d, -dmax, dmax)):
            ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vm, interpolation="nearest",
                      zorder=0, aspect="equal")
            _field(ax, any_cell)
    export(fig, "fig_occupancy")


def _smooth(x: np.ndarray, k: int = POSTURE_SMOOTH) -> np.ndarray:
    """Centered moving average that ignores NaNs and keeps the series length."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    kern = np.ones(k)
    num = np.convolve(np.where(ok, x, 0.0), kern, mode="same")
    den = np.convolve(ok.astype(np.float64), kern, mode="same")
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def fig_posture(traj: dict) -> None:
    """Fraction of active blue agents in their own half over the episode, per strategy."""
    fig, axes = plt.subplots(1, len(SCALES), figsize=(TWO_COL, 1.9), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, scale in zip(axes, SCALES):
        _man, cells = traj[scale]
        for strat, color, ls in (("pi_A", C_A, "-"), ("pi_B", C_B, "--")):
            series = []
            for (s, _pole, _seed), c in cells.items():
                if s != strat:
                    continue
                mid = float(c["bounds"][0]) / 2
                own_left = float(c["flag_home_blue"][0]) < mid
                live = c["blue_alive"].astype(bool) & ~c["blue_tagged"].astype(bool)
                own = (c["blue_x"] < mid) if own_left else (c["blue_x"] > mid)
                frac = np.where(live.sum(1) > 0, (own & live).sum(1) / np.maximum(live.sum(1), 1), np.nan)
                series.append(frac)
            T = max(len(f) for f in series)
            mat = np.full((len(series), T), np.nan)
            for k, f in enumerate(series):
                mat[k, : len(f)] = f
            keep = np.sum(np.isfinite(mat), axis=0) >= max(4, len(series) // 2)
            t = np.arange(T)[keep]
            m = _smooth(np.nanmean(mat, axis=0))[keep]
            se = _smooth(np.nanstd(mat, axis=0) / np.sqrt(np.maximum(np.sum(np.isfinite(mat), axis=0), 1)))[keep]
            ax.fill_between(t, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.18, lw=0, zorder=2)
            ax.plot(t, m, color=color, ls=ls, lw=1.3, zorder=3)
        ax.set_ylim(0, 1)
        ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel("Decision step")
        ax.tick_params(axis="x", length=2.5)
    axes[0].set_ylabel("Fraction in own half")
    for ax in axes[1:]:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.legend([Line2D([], [], color=C_A), Line2D([], [], color=C_B, ls="--")],
               ["Pole-A strategy", "Pole-B strategy"], loc="upper center", bbox_to_anchor=(0.5, 1.1),
               ncol=2, frameon=False)
    export(fig, "fig_posture")


# ------------------------------------------------------------------------------------- main
def main() -> int:
    apply_style()
    cells = {key: load_cells(name, kind) for key, (name, kind) in ROWS.items()}
    stats = {(s, m): deltas(cells[(s, m)]) for s in SCALES for m in METHODS}
    margin = {s: deltas(load_cells(ROWS[(s, "Ours (heuristic roles)")][0], "spec", field="margin")) for s in SCALES}

    fig_crossover(cells)
    fig_complementarity(stats)
    fig_baselines(stats)
    fig_worst_pole(stats)
    fig_deployed_winrate(cells)
    noise = fig_noise()
    fig_margin(margin)

    summary = {
        "source": "sealed n=128 rows; 95% paired percentile bootstrap (20000, rng 7)",
        "delta_win": {f"{s}|{m}": stats[(s, m)] for s, m in stats},
        "delta_margin_ours": margin,
        "noise": {f"{s}|{c}": v for (s, c), v in noise.items()},
    }
    try:
        traj = {s: _load_traj(s, require_occupancy=False) for s in SCALES}
    except FileNotFoundError as exc:
        print(f"trajectory figure skipped: {exc}")
        traj = None
    if traj is not None:
        fig_trajectories(traj)
        summary["trajectory_seeds"] = {s: traj[s][0]["trajectory_seed"] for s in SCALES}
        try:
            traj_full = {s: _load_traj(s, require_occupancy=True) for s in SCALES}
        except FileNotFoundError as exc:
            print(f"occupancy/posture figures skipped: {exc}")
            traj_full = None
        if traj_full is not None:
            fig_occupancy(traj_full)
            fig_posture(traj_full)
            summary["occupancy_seeds"] = {s: traj_full[s][0]["occupancy_seeds"] for s in SCALES}
    (FIG / "paper_figures_stats.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
