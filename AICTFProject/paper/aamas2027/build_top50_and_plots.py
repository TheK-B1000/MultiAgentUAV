"""Build n=128 verification stats + a secondary n=50 seed-ID-prefix view.

PRIMARY science: sealed n=128 per arm (do not replace with a subset).

SECONDARY n=50 rule in this script (deterministic prefix, NOT outcome-ranked):
  sort seeds by ascending seed ID within each sealed matched block, take first 50.
  Example: 25700001..25700128 -> 25700001..25700050.

This is a sensitivity check, not a "top 50 by performance" view. Do not add
win-rate / Delta ranking here until a single selection rule is frozen in writing
(see paper/aamas2027/README.md).

Run:
  ./.venv/Scripts/python.exe paper/aamas2027/build_top50_and_plots.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
TOP_K = 50


def _load_rows(name: str) -> list[dict]:
    p = SD / name
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _cells_specialist(rows: list[dict]) -> dict[tuple[str, str], dict[int, float]]:
    by: dict[tuple[str, str], dict[int, float]] = {}
    for r in rows:
        by.setdefault((r["policy"], r["pole"]), {})[int(r["seed"])] = float(r["win"])
    return by


def _cells_z(rows: list[dict]) -> dict[tuple[str, str], dict[int, float]]:
    by: dict[tuple[str, str], dict[int, float]] = {}
    for r in rows:
        by.setdefault((str(r["z"]), r["pole"]), {})[int(r["seed"])] = float(r["win"])
    return by


def _delta_stats(by: dict, *, z_mode: bool, top_k: int | None) -> dict:
    if z_mode:
        a0, a1 = ("0", "A"), ("1", "A")
        b1, b0 = ("1", "B"), ("0", "B")
    else:
        a0, a1 = ("pi_A", "A"), ("pi_B", "A")
        b1, b0 = ("pi_B", "B"), ("pi_A", "B")
    seeds = sorted(by[a0])
    for k in (a0, a1, b1, b0):
        if sorted(by[k]) != seeds:
            raise SystemExit(f"seed mismatch on {k}")
    if top_k is not None:
        seeds = seeds[:top_k]
    da = np.array([by[a0][s] - by[a1][s] for s in seeds], dtype=np.float64)
    db = np.array([by[b1][s] - by[b0][s] for s in seeds], dtype=np.float64)
    return {
        "n": len(seeds),
        "seed_lo": int(seeds[0]),
        "seed_hi": int(seeds[-1]),
        "delta_A_mean": float(da.mean()),
        "delta_A_std": float(da.std(ddof=1)),
        "delta_B_mean": float(db.mean()),
        "delta_B_std": float(db.std(ddof=1)),
        "da": da,
        "db": db,
    }


def _fmt(mean: float, std: float) -> str:
    sign = "+" if mean >= 0 else ""
    return f"{sign}{mean:.3f} $\\pm$ {std:.3f}"


def _row_tex(team: str, method: str, st: dict) -> str:
    return (
        f"{team}\n& {method}\n"
        f"& {_fmt(st['delta_A_mean'], st['delta_A_std'])}\n"
        f"& {_fmt(st['delta_B_mean'], st['delta_B_std'])} \\\\"
    )


SPECS = [
    # (scale, method_label, csv, kind)
    ("2v2", "Ours (heuristic roles)", "standardized_2v2_split_k1_confirmatory_specialist_crossover_eval_rows.csv", "spec"),
    ("2v2", "Specialists (no roles)", "standardized_2v2_diag_presplit_specialist_crossover_eval_rows.csv", "spec"),
    ("2v2", "Ours (paired w/ no-role)", "standardized_2v2_diag_split_specialist_crossover_eval_rows.csv", "spec"),
    ("2v2", "Share-Encoder", "standardized_2v2_share_encoder_crossover_eval_rows.csv", "z"),
    ("2v2", "Fully Shared+$z$", "standardized_2v2_fully_shared_z_crossover_eval_rows.csv", "z"),
    ("4v4", "Ours (heuristic roles)", "defend_attack_split_policy_a_v1_confirmatory_v1_specialist_crossover_eval_rows.csv", "spec"),
    ("4v4", "Specialists (no roles)", "confirmatory_b3_entity_repair_corrected_specialist_crossover_eval_rows.csv", "spec"),
    ("4v4", "Share-Encoder", "standardized_4v4_share_encoder_crossover_eval_rows.csv", "z"),
    ("4v4", "Fully Shared+$z$", "standardized_4v4_fully_shared_z_crossover_eval_rows.csv", "z"),
    ("6v6", "Ours (heuristic roles)", "standardized_6v6_split_k1_confirmatory_specialist_crossover_eval_rows.csv", "spec"),
    ("6v6", "Specialists (no roles)", "standardized_6v6_norole_specialist_crossover_eval_rows.csv", "spec"),
    ("6v6", "Share-Encoder", "standardized_6v6_share_encoder_crossover_eval_rows.csv", "z"),
    ("6v6", "Fully Shared+$z$", "standardized_6v6_fully_shared_z_crossover_eval_rows.csv", "z"),
]

NOISE = [
    ("2v2", "Nominal", "standardized_2v2_noise_nominal_specialist_crossover_eval_rows.csv"),
    ("2v2", "Localization", "standardized_2v2_noise_localization_medium_specialist_crossover_eval_rows.csv"),
    ("2v2", "Motion", "standardized_2v2_noise_motion_medium_specialist_crossover_eval_rows.csv"),
    ("2v2", "Delay", "standardized_2v2_noise_delay_medium_specialist_crossover_eval_rows.csv"),
    ("4v4", "Nominal", "standardized_4v4_noise_nominal_specialist_crossover_eval_rows.csv"),
    ("4v4", "Localization", "standardized_4v4_noise_localization_medium_specialist_crossover_eval_rows.csv"),
    ("4v4", "Motion", "standardized_4v4_noise_motion_medium_specialist_crossover_eval_rows.csv"),
    ("4v4", "Delay", "standardized_4v4_noise_delay_medium_specialist_crossover_eval_rows.csv"),
    ("6v6", "Nominal", "standardized_6v6_noise_nominal_specialist_crossover_eval_rows.csv"),
    ("6v6", "Localization", "standardized_6v6_noise_localization_medium_specialist_crossover_eval_rows.csv"),
    ("6v6", "Motion", "standardized_6v6_noise_motion_medium_specialist_crossover_eval_rows.csv"),
    ("6v6", "Delay", "standardized_6v6_noise_delay_medium_specialist_crossover_eval_rows.csv"),
]


C_A, C_B = "#0072B2", "#D55E00"
INK, GRID, BAND = "#333333", "#E6E6E6", "#F3F3F3"
FIG_W = 7.16  # two-column width (in)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
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
        "savefig.facecolor": "white",
    })


def _ylim(*series) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(s, dtype=float) for s in series])
    lo = min(0.0, float(vals.min()))
    hi = max(0.0, float(vals.max()))
    pad = 0.06 * (hi - lo)
    return (lo - pad if lo < 0 else 0.0), hi + pad


def _panel(ax, labels, da, db, *, title, highlight=None, first=False, ylim=None):
    x = np.arange(len(labels))
    w = 0.34
    if highlight is not None:
        ax.axvspan(highlight - 0.5, highlight + 0.5, color=BAND, zorder=0, lw=0)
    ax.bar(x - w / 2, da, w, color=C_A, zorder=2, lw=0)
    ax.bar(x + w / 2, db, w, color=C_B, zorder=2, lw=0)
    ax.axhline(0.0, color=INK, lw=0.8, zorder=3)
    ax.spines["bottom"].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=1)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, linespacing=0.95)
    ax.set_xlim(-0.55, len(labels) - 0.45)
    if ylim is not None:
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold", pad=3, color=INK)
    if first:
        ax.set_ylabel(r"Mean $\Delta$")
    else:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)


def _figure(title: str, ncols: int, height: float = 2.25, width: float = FIG_W):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_A), plt.Rectangle((0, 0), 1, 1, color=C_B)]
    fig.legend(handles, [r"$\Delta_A$", r"$\Delta_B$"], loc="upper right",
               bbox_to_anchor=(0.995, 1.04), ncol=2, frameon=False,
               handlelength=1.0, handleheight=0.8, columnspacing=1.0, fontsize=8.5)
    fig.suptitle(title, x=0.01, y=1.04, ha="left", fontsize=10, fontweight="bold", color=INK)
    return fig, axes


def _export(fig, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _plot_all(full, top, noise_top, scales, methods) -> None:
    _style()
    ours = "Ours (heuristic roles)"

    # Specialization: n=128 vs first 50 seeds
    da128 = [full[(s, ours)]["delta_A_mean"] for s in scales]
    db128 = [full[(s, ours)]["delta_B_mean"] for s in scales]
    da50 = [top[(s, ours)]["delta_A_mean"] for s in scales]
    db50 = [top[(s, ours)]["delta_B_mean"] for s in scales]
    yl = _ylim(da128, db128, da50, db50)
    fig, axes = _figure("Specialization across team sizes", 2, height=2.1, width=5.0)
    _panel(axes[0], scales, da128, db128, title="All 128 seeds", first=True, ylim=yl)
    _panel(axes[1], scales, da50, db50, title="First 50 seeds", ylim=yl)
    _export(fig, "fig_specialization_n128_vs_top50")

    # Baselines
    short = {
        "Specialists (no roles)": "No\nroles",
        "Share-Encoder": "Share-\nEncoder",
        "Fully Shared+$z$": "Fully\nShared+$z$",
        ours: "Ours",
    }
    labels = [short[m] for m in methods]
    data = {s: ([top[(s, m)]["delta_A_mean"] for m in methods],
                [top[(s, m)]["delta_B_mean"] for m in methods]) for s in scales}
    yl = _ylim(*[v for pair in data.values() for v in pair])
    fig, axes = _figure("Baseline comparison", 3, height=2.4)
    for i, (ax, s) in enumerate(zip(axes, scales)):
        _panel(ax, labels, *data[s], title=s, highlight=methods.index(ours), first=(i == 0), ylim=yl)
    _export(fig, "fig_baselines_top50")

    # Deployment noise
    conds = ["Nominal", "Localization", "Motion", "Delay"]
    labels = ["Nominal", "Local.", "Motion", "Delay"]
    data = {s: ([noise_top[(s, c)]["delta_A_mean"] for c in conds],
                [noise_top[(s, c)]["delta_B_mean"] for c in conds]) for s in scales}
    yl = _ylim(*[v for pair in data.values() for v in pair])
    fig, axes = _figure("Robustness to deployment noise", 3)
    for i, (ax, s) in enumerate(zip(axes, scales)):
        _panel(ax, labels, *data[s], title=s, highlight=0, first=(i == 0), ylim=yl)
    _export(fig, "fig_noise_top50")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    catalog: dict = {"top_k": TOP_K, "rule": "first K seeds by ascending seed ID in the sealed block", "rows": [], "noise": []}

    full: dict[tuple[str, str], dict] = {}
    top: dict[tuple[str, str], dict] = {}
    for scale, method, csv_name, kind in SPECS:
        rows = _load_rows(csv_name)
        by = _cells_z(rows) if kind == "z" else _cells_specialist(rows)
        st128 = _delta_stats(by, z_mode=(kind == "z"), top_k=None)
        st50 = _delta_stats(by, z_mode=(kind == "z"), top_k=TOP_K)
        # drop arrays before JSON
        for st in (st128, st50):
            st.pop("da", None)
            st.pop("db", None)
        full[(scale, method)] = st128
        top[(scale, method)] = st50
        catalog["rows"].append({"scale": scale, "method": method, "csv": csv_name, "n128": st128, "n50": st50})

    noise_full: dict[tuple[str, str], dict] = {}
    noise_top: dict[tuple[str, str], dict] = {}
    for scale, cond, csv_name in NOISE:
        rows = _load_rows(csv_name)
        by = _cells_specialist(rows)
        st128 = _delta_stats(by, z_mode=False, top_k=None)
        st50 = _delta_stats(by, z_mode=False, top_k=TOP_K)
        for st in (st128, st50):
            st.pop("da", None)
            st.pop("db", None)
        noise_full[(scale, cond)] = st128
        noise_top[(scale, cond)] = st50
        catalog["noise"].append({"scale": scale, "condition": cond, "csv": csv_name, "n128": st128, "n50": st50})

    (OUT / "top50_seed_stats.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")

    # ---- LaTeX: main specialization top-50 ----
    main_methods = [
        ("2v2", "Ours (heuristic roles)"),
        ("4v4", "Ours (heuristic roles)"),
        ("6v6", "Ours (heuristic roles)"),
    ]
    lines = [
        "% Auto-generated by build_top50_and_plots.py -- do not edit by hand.",
        "% SECONDARY view only. Selection = first 50 seeds by ascending seed ID",
        "% (deterministic prefix). NOT ranked by win rate or Delta. Primary = n=128.",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Secondary sensitivity check (not a replacement for $n=128$):",
        r"first 50 seeds by ascending seed ID in each sealed confirmatory block.",
        r"Not ranked by win rate or $\Delta$. Mean $\pm$ sample std of per-seed",
        r"paired win differences.}",
        r"\label{tab:main_specialization_seedid50}",
        r"\footnotesize",
        r"\begin{tabular}{c|cc}",
        r"\hline",
        r"Team size & $\Delta_A$ & $\Delta_B$ \\",
        r"\hline",
    ]
    for scale, method in main_methods:
        st = top[(scale, method)]
        lines.append(f"{scale} & {_fmt(st['delta_A_mean'], st['delta_A_std'])} & {_fmt(st['delta_B_mean'], st['delta_B_std'])} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]

    # sharing top-50
    share_order = [
        ("Specialists (no roles)", "Share-Encoder", "Fully Shared+$z$", "Ours (heuristic roles)"),
        ("Specialists (no roles)", "Share-Encoder", "Fully Shared+$z$", "Ours (heuristic roles)"),
        ("Specialists (no roles)", "Share-Encoder", "Fully Shared+$z$", "Ours (heuristic roles)"),
    ]
    scales = ["2v2", "4v4", "6v6"]
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Secondary sensitivity check (seed-ID prefix $n=50$; not",
        r"outcome-ranked). Parameter-sharing and role-allocation baselines.",
        r"Mean $\pm$ sample std. Primary results remain the sealed $n=128$ tables.}",
        r"\label{tab:sharing_baselines_seedid50}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}clcc@{}}",
        r"\hline",
        r"Team & Method & $\Delta_A$ & $\Delta_B$ \\",
        r"\hline",
    ]
    for scale, methods in zip(scales, share_order):
        for i, method in enumerate(methods):
            st = top[(scale, method)]
            team = scale if i == 0 else ""
            lines.append(_row_tex(team, method, st))
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}", ""]

    # noise top-50 + full n=128 for 6v6 fill
    lines += [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Secondary sensitivity check (seed-ID prefix $n=50$; not",
        r"outcome-ranked). Deployment-noise specialization. Nominal included for",
        r"reference. Primary results remain the sealed $n=128$ noise table.}",
        r"\label{tab:noise_results_seedid50}",
        r"\footnotesize",
        r"\begin{tabular}{c|cc|cc|cc|cc}",
        r"\hline",
        r"& \multicolumn{2}{c|}{Nominal}",
        r"& \multicolumn{2}{c|}{Localization}",
        r"& \multicolumn{2}{c|}{Motion}",
        r"& \multicolumn{2}{c}{Delay} \\",
        r"Team & $\Delta_A$ & $\Delta_B$ & $\Delta_A$ & $\Delta_B$ & $\Delta_A$ & $\Delta_B$ & $\Delta_A$ & $\Delta_B$ \\",
        r"\hline",
    ]
    for scale in scales:
        cells = []
        for cond in ("Nominal", "Localization", "Motion", "Delay"):
            st = noise_top[(scale, cond)]
            cells.append(_fmt(st["delta_A_mean"], st["delta_A_std"]))
            cells.append(_fmt(st["delta_B_mean"], st["delta_B_std"]))
        lines.append(scale + " & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}", ""]

    # n=128 noise fill helper comment block
    lines += [
        "% --- n=128 noise means (for experiments.tex fill) ---",
    ]
    for scale in scales:
        for cond in ("Nominal", "Localization", "Motion", "Delay"):
            st = noise_full[(scale, cond)]
            lines.append(
                f"% {scale} {cond}: "
                f"dA={st['delta_A_mean']:+.4f}+/-{st['delta_A_std']:.4f} "
                f"dB={st['delta_B_mean']:+.4f}+/-{st['delta_B_std']:.4f}"
            )

    (OUT / "top50_tables.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    _plot_all(full, top, noise_top, scales, share_order[0])
    conds = ["Nominal", "Localization", "Motion", "Delay"]
    print("wrote", OUT / "top50_seed_stats.json")
    print("wrote", OUT / "top50_tables.tex")
    print("wrote figures under", FIG)
    print("\n6v6 noise n=128:")
    for cond in conds:
        st = noise_full[("6v6", cond)]
        print(f"  {cond}: dA={st['delta_A_mean']:+.4f}+/-{st['delta_A_std']:.4f}  dB={st['delta_B_mean']:+.4f}+/-{st['delta_B_std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
