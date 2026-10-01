"""TRUE top-50 by deployed-performance score (post-hoc descriptive subset).

PRIMARY science remains sealed n=128. This script only builds a secondary view.

Ranking rule (frozen here):
  score(seed) = outcome(strategy A on Pole A) + outcome(strategy B on Pole B)
  Rank highest to lowest; ties broken by ascending seed ID.
  Take the first 50 seeds of that ranking.

Noise: select top-50 from NOMINAL only; carry those seed IDs into
localization / motion / delay (paired).

Generalist: score(seed) = outcome(G on A) + outcome(G on B); report WR_A, WR_B.

Run:
  ./.venv/Scripts/python.exe paper/aamas2027/build_true_top50.py
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
PROV = OUT / "true_top50_provenance"
TOP_K = 50

RANK_RULE = (
    "score(seed)=outcome(strategy A on Pole A)+outcome(strategy B on Pole B); "
    "rank descending; tie-break ascending seed ID; take top 50. "
    "Post-hoc descriptive subset; primary results remain sealed n=128."
)

C_A, C_B = "#0072B2", "#D55E00"
INK, GRID, BAND = "#333333", "#E6E6E6", "#F3F3F3"
FIG_W = 7.16


def _load_rows(name: str) -> list[dict]:
    with (SD / name).open(newline="", encoding="utf-8") as f:
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


def _keys(kind: str) -> tuple:
    if kind == "spec":
        return ("pi_A", "A"), ("pi_B", "A"), ("pi_B", "B"), ("pi_A", "B")
    if kind == "z":
        return ("0", "A"), ("1", "A"), ("1", "B"), ("0", "B")
    if kind == "gen":
        return ("0", "A"), ("0", "B")
    raise ValueError(kind)


def _all_seeds(by: dict, kind: str) -> list[int]:
    if kind == "gen":
        a, b = _keys(kind)
        seeds = sorted(by[a])
        if sorted(by[b]) != seeds:
            raise SystemExit("generalist seed mismatch")
        return seeds
    a0, a1, b1, b0 = _keys(kind)
    seeds = sorted(by[a0])
    for k in (a0, a1, b1, b0):
        if sorted(by[k]) != seeds:
            raise SystemExit(f"seed mismatch on {k}")
    return seeds


def _deployed_score(by: dict, seed: int, kind: str) -> float:
    if kind == "gen":
        a, b = _keys(kind)
        return float(by[a][seed] + by[b][seed])
    a0, _, b1, _ = _keys(kind)
    return float(by[a0][seed] + by[b1][seed])


def select_top_seeds(by: dict, kind: str, k: int = TOP_K) -> list[int]:
    seeds = _all_seeds(by, kind)
    # sort: score desc, then seed id asc
    ranked = sorted(seeds, key=lambda s: (-_deployed_score(by, s, kind), s))
    chosen = ranked[:k]
    if len(chosen) != k or len(set(chosen)) != k:
        raise SystemExit(f"expected {k} unique seeds, got {len(chosen)} / {len(set(chosen))}")
    return chosen


def metrics_on_seeds(by: dict, kind: str, seeds: list[int]) -> dict:
    if len(seeds) != TOP_K or len(set(seeds)) != TOP_K:
        raise SystemExit("metrics_on_seeds: need exactly 50 unique seeds")
    if kind == "gen":
        a, b = _keys(kind)
        wa = np.array([by[a][s] for s in seeds], dtype=np.float64)
        wb = np.array([by[b][s] for s in seeds], dtype=np.float64)
        return {
            "n": TOP_K,
            "WR_A": float(wa.mean()),
            "WR_B": float(wb.mean()),
            "delta_A_mean": None,
            "delta_A_std": None,
            "delta_B_mean": None,
            "delta_B_std": None,
            "min_delta": None,
        }
    a0, a1, b1, b0 = _keys(kind)
    wa = np.array([by[a0][s] for s in seeds], dtype=np.float64)
    wb = np.array([by[b1][s] for s in seeds], dtype=np.float64)
    da = np.array([by[a0][s] - by[a1][s] for s in seeds], dtype=np.float64)
    db = np.array([by[b1][s] - by[b0][s] for s in seeds], dtype=np.float64)
    return {
        "n": TOP_K,
        "WR_A": float(wa.mean()),
        "WR_B": float(wb.mean()),
        "delta_A_mean": float(da.mean()),
        "delta_A_std": float(da.std(ddof=1)),
        "delta_B_mean": float(db.mean()),
        "delta_B_std": float(db.std(ddof=1)),
        "min_delta": float(min(da.mean(), db.mean())),
    }


def _fmt_d(mean: float, std: float) -> str:
    sign = "+" if mean >= 0 else ""
    return f"{sign}{mean:.3f} $\\pm$ {std:.3f}"


SPECS = [
    ("2v2", "Specialists (no roles)", "standardized_2v2_diag_presplit_specialist_crossover_eval_rows.csv", "spec"),
    ("2v2", "Share-Encoder", "standardized_2v2_share_encoder_crossover_eval_rows.csv", "z"),
    ("2v2", "Fully Shared+$z$", "standardized_2v2_fully_shared_z_crossover_eval_rows.csv", "z"),
    ("2v2", "Ours (heuristic roles)", "standardized_2v2_split_k1_confirmatory_specialist_crossover_eval_rows.csv", "spec"),
    ("2v2", "Generalist", "standardized_2v2_generalist_crossover_eval_rows.csv", "gen"),
    ("4v4", "Specialists (no roles)", "confirmatory_b3_entity_repair_corrected_specialist_crossover_eval_rows.csv", "spec"),
    ("4v4", "Share-Encoder", "standardized_4v4_share_encoder_crossover_eval_rows.csv", "z"),
    ("4v4", "Fully Shared+$z$", "standardized_4v4_fully_shared_z_crossover_eval_rows.csv", "z"),
    ("4v4", "Ours (heuristic roles)", "defend_attack_split_policy_a_v1_confirmatory_v1_specialist_crossover_eval_rows.csv", "spec"),
    ("4v4", "Generalist", "standardized_4v4_generalist_crossover_eval_rows.csv", "gen"),
    ("6v6", "Specialists (no roles)", "standardized_6v6_norole_specialist_crossover_eval_rows.csv", "spec"),
    ("6v6", "Share-Encoder", "standardized_6v6_share_encoder_crossover_eval_rows.csv", "z"),
    ("6v6", "Fully Shared+$z$", "standardized_6v6_fully_shared_z_crossover_eval_rows.csv", "z"),
    ("6v6", "Ours (heuristic roles)", "standardized_6v6_split_k1_confirmatory_specialist_crossover_eval_rows.csv", "spec"),
    ("6v6", "Generalist", "standardized_6v6_generalist_crossover_eval_rows.csv", "gen"),
]

NOISE = {
    "2v2": {
        "Nominal": "standardized_2v2_noise_nominal_specialist_crossover_eval_rows.csv",
        "Localization": "standardized_2v2_noise_localization_medium_specialist_crossover_eval_rows.csv",
        "Motion": "standardized_2v2_noise_motion_medium_specialist_crossover_eval_rows.csv",
        "Delay": "standardized_2v2_noise_delay_medium_specialist_crossover_eval_rows.csv",
    },
    "4v4": {
        "Nominal": "standardized_4v4_noise_nominal_specialist_crossover_eval_rows.csv",
        "Localization": "standardized_4v4_noise_localization_medium_specialist_crossover_eval_rows.csv",
        "Motion": "standardized_4v4_noise_motion_medium_specialist_crossover_eval_rows.csv",
        "Delay": "standardized_4v4_noise_delay_medium_specialist_crossover_eval_rows.csv",
    },
    "6v6": {
        "Nominal": "standardized_6v6_noise_nominal_specialist_crossover_eval_rows.csv",
        "Localization": "standardized_6v6_noise_localization_medium_specialist_crossover_eval_rows.csv",
        "Motion": "standardized_6v6_noise_motion_medium_specialist_crossover_eval_rows.csv",
        "Delay": "standardized_6v6_noise_delay_medium_specialist_crossover_eval_rows.csv",
    },
}


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
    pad = 0.06 * max(hi - lo, 0.1)
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
                             gridspec_kw={"wspace": 0.08} if ncols > 1 else None)
    if ncols == 1:
        axes = [axes]
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


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    PROV.mkdir(parents=True, exist_ok=True)
    _style()

    catalog: dict = {
        "record": "TRUE_TOP50_DEPLOYED_PERFORMANCE",
        "status": "SECONDARY_POSTHOC_DESCRIPTIVE",
        "primary_results": "sealed n=128 tables remain authoritative",
        "ranking_rule": RANK_RULE,
        "top_k": TOP_K,
        "rows": [],
        "noise": [],
        "generalist": [],
    }

    results: dict[tuple[str, str], dict] = {}
    for scale, method, csv_name, kind in SPECS:
        rows = _load_rows(csv_name)
        by = _cells_z(rows) if kind in ("z", "gen") else _cells_specialist(rows)
        chosen = select_top_seeds(by, kind)
        st = metrics_on_seeds(by, kind, chosen)
        # provenance: seed list + per-seed score + source
        scores = {str(s): _deployed_score(by, s, kind) for s in chosen}
        entry = {
            "scale": scale,
            "method": method,
            "kind": kind,
            "source_csv": csv_name,
            "source_csv_path": str((SD / csv_name).resolve()),
            "n_selected": len(chosen),
            "n_unique": len(set(chosen)),
            "selected_seed_ids": chosen,
            "deployed_scores": scores,
            "metrics": st,
        }
        if kind == "gen":
            catalog["generalist"].append(entry)
        else:
            catalog["rows"].append(entry)
            results[(scale, method)] = st
        stem = f"{scale}_{method}".replace(" ", "_").replace("$", "").replace("+", "plus").replace("(", "").replace(")", "")
        (PROV / f"{stem}_seeds.json").write_text(
            json.dumps({"selected_seed_ids": chosen, "scores": scores, "source_csv": csv_name, "rule": RANK_RULE}, indent=2) + "\n",
            encoding="utf-8",
        )

    # Noise: select from nominal, apply to all conditions
    noise_res: dict[tuple[str, str], dict] = {}
    for scale, conds in NOISE.items():
        by_nom = _cells_specialist(_load_rows(conds["Nominal"]))
        chosen = select_top_seeds(by_nom, "spec")
        catalog["noise"].append({
            "scale": scale,
            "selection_condition": "Nominal",
            "selected_seed_ids": chosen,
            "n_unique": len(set(chosen)),
            "carried_into": list(conds.keys()),
        })
        (PROV / f"{scale}_noise_nominal_selected_seeds.json").write_text(
            json.dumps({"selected_seed_ids": chosen, "rule": RANK_RULE + " Selection from Nominal only."}, indent=2) + "\n",
            encoding="utf-8",
        )
        for cond, csv_name in conds.items():
            by = _cells_specialist(_load_rows(csv_name))
            # verify all chosen seeds exist
            missing = [s for s in chosen if s not in by[("pi_A", "A")]]
            if missing:
                raise SystemExit(f"{scale} {cond}: missing seeds {missing[:5]}")
            st = metrics_on_seeds(by, "spec", chosen)
            noise_res[(scale, cond)] = st
            catalog["noise"][-1].setdefault("metrics", {})[cond] = st

    (OUT / "true_top50_stats.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")

    # ---- LaTeX ----
    scales = ["2v2", "4v4", "6v6"]
    methods = ["Specialists (no roles)", "Share-Encoder", "Fully Shared+$z$", "Ours (heuristic roles)"]
    cap_rule = (
        r"Post-hoc descriptive top-50 subset (not a replacement for sealed $n=128$). "
        r"Per method/team size, seeds are ranked by deployed score "
        r"$\mathrm{outcome}(A{\to}A)+\mathrm{outcome}(B{\to}B)$, ties by ascending seed ID; "
        r"the top 50 seeds are retained."
    )
    lines = [
        "% Auto-generated by build_true_top50.py -- do not edit by hand.",
        "% SECONDARY post-hoc top-50 by deployed performance. Primary = sealed n=128.",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Post-hoc top-50 by deployed performance: role-allocated system. "
        + cap_rule +
        r" Mean $\pm$ sample std of $\Delta$ on the selected seeds.}",
        r"\label{tab:true_top50_ours}",
        r"\footnotesize",
        r"\begin{tabular}{c|cc|c}",
        r"\hline",
        r"Team & $\Delta_A$ & $\Delta_B$ & $\min(\Delta_A,\Delta_B)$ \\",
        r"\hline",
    ]
    for scale in scales:
        st = results[(scale, "Ours (heuristic roles)")]
        lines.append(
            f"{scale} & {_fmt_d(st['delta_A_mean'], st['delta_A_std'])} & "
            f"{_fmt_d(st['delta_B_mean'], st['delta_B_std'])} & "
            f"{st['min_delta']:+.3f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]

    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Post-hoc top-50 by deployed performance: baselines. "
        + cap_rule +
        r" Each method selects its own top-50 seeds.}",
        r"\label{tab:true_top50_baselines}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}clccc@{}}",
        r"\hline",
        r"Team & Method & $\Delta_A$ & $\Delta_B$ & $\min(\Delta)$ \\",
        r"\hline",
    ]
    for scale in scales:
        for i, method in enumerate(methods):
            st = results[(scale, method)]
            team = scale if i == 0 else ""
            lines.append(
                f"{team} & {method} & {_fmt_d(st['delta_A_mean'], st['delta_A_std'])} & "
                f"{_fmt_d(st['delta_B_mean'], st['delta_B_std'])} & "
                f"{st['min_delta']:+.3f} \\\\"
            )
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}", ""]

    # Deployed WR top-50 including Generalist
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Post-hoc top-50 by deployed performance: deployed win rates. "
        + cap_rule +
        r" Generalist ranked by $\mathrm{outcome}(G{\to}A)+\mathrm{outcome}(G{\to}B)$.}",
        r"\label{tab:true_top50_deployed_wr}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\hline",
        r"& \multicolumn{2}{c}{2v2} & \multicolumn{2}{c}{4v4} & \multicolumn{2}{c}{6v6} \\",
        r"Method & WR$_A$ & WR$_B$ & WR$_A$ & WR$_B$ & WR$_A$ & WR$_B$ \\",
        r"\hline",
    ]
    wr_methods = ["Generalist"] + methods
    # index generalist from catalog
    gen_map = {(e["scale"], e["method"]): e["metrics"] for e in catalog["generalist"]}
    for method in wr_methods:
        cells = []
        for scale in scales:
            if method == "Generalist":
                st = gen_map[(scale, "Generalist")]
            else:
                st = results[(scale, method)]
            cells.append(f"{st['WR_A']:.3f}")
            cells.append(f"{st['WR_B']:.3f}")
        lines.append(method + " & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]

    # Noise
    lines += [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Post-hoc top-50 by deployed performance under noise. "
        r"Seeds are selected from the \emph{nominal} condition only "
        r"(same ranking rule), then carried into localization, motion, and delay "
        r"so comparisons remain paired. Not a replacement for sealed $n=128$.}",
        r"\label{tab:true_top50_noise}",
        r"\footnotesize",
        r"\begin{tabular}{c|cc|cc|cc|cc}",
        r"\hline",
        r"& \multicolumn{2}{c|}{Nominal}"
        r"& \multicolumn{2}{c|}{Localization}"
        r"& \multicolumn{2}{c|}{Motion}"
        r"& \multicolumn{2}{c}{Delay} \\",
        r"Team & $\Delta_A$ & $\Delta_B$ & $\Delta_A$ & $\Delta_B$ & $\Delta_A$ & $\Delta_B$ & $\Delta_A$ & $\Delta_B$ \\",
        r"\hline",
    ]
    for scale in scales:
        cells = []
        for cond in ("Nominal", "Localization", "Motion", "Delay"):
            st = noise_res[(scale, cond)]
            cells.append(_fmt_d(st["delta_A_mean"], st["delta_A_std"]))
            cells.append(_fmt_d(st["delta_B_mean"], st["delta_B_std"]))
        lines.append(scale + " & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}", ""]

    (OUT / "true_top50_tables.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- Plots (baselines + noise + ours specialization) ----
    short = {
        "Specialists (no roles)": "No\nroles",
        "Share-Encoder": "Share-\nEncoder",
        "Fully Shared+$z$": "Fully\nShared+$z$",
        "Ours (heuristic roles)": "Ours",
    }
    data = {s: ([results[(s, m)]["delta_A_mean"] for m in methods],
                [results[(s, m)]["delta_B_mean"] for m in methods]) for s in scales}
    yl = _ylim(*[v for pair in data.values() for v in pair])
    fig, axes = _figure("Baseline comparison (top-50 deployed)", 3, height=2.4)
    for i, (ax, s) in enumerate(zip(axes, scales)):
        _panel(ax, [short[m] for m in methods], *data[s],
               title=s, highlight=methods.index("Ours (heuristic roles)"), first=(i == 0), ylim=yl)
    _export(fig, "fig_baselines_true_top50")

    # specialization ours only: could show min_delta or both deltas across scales
    da = [results[(s, "Ours (heuristic roles)")]["delta_A_mean"] for s in scales]
    db = [results[(s, "Ours (heuristic roles)")]["delta_B_mean"] for s in scales]
    yl = _ylim(da, db)
    fig, axes = _figure("Specialization (top-50 deployed)", 1, height=2.1, width=3.6)
    _panel(axes[0], scales, da, db, title="Ours", first=True, ylim=yl)
    _export(fig, "fig_specialization_true_top50")

    conds = ["Nominal", "Localization", "Motion", "Delay"]
    labels = ["Nominal", "Local.", "Motion", "Delay"]
    data = {s: ([noise_res[(s, c)]["delta_A_mean"] for c in conds],
                [noise_res[(s, c)]["delta_B_mean"] for c in conds]) for s in scales}
    yl = _ylim(*[v for pair in data.values() for v in pair])
    fig, axes = _figure("Robustness (top-50 from nominal)", 3)
    for i, (ax, s) in enumerate(zip(axes, scales)):
        _panel(ax, labels, *data[s], title=s, highlight=0, first=(i == 0), ylim=yl)
    _export(fig, "fig_noise_true_top50")

    # verify all subsets
    for e in catalog["rows"] + catalog["generalist"]:
        assert e["n_unique"] == TOP_K, e
    for e in catalog["noise"]:
        assert e["n_unique"] == TOP_K, e

    print("wrote", OUT / "true_top50_stats.json")
    print("wrote", OUT / "true_top50_tables.tex")
    print("wrote provenance under", PROV)
    print("wrote figures: fig_*_true_top50.*")
    print("sample Ours 2v2:", results[("2v2", "Ours (heuristic roles)")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
