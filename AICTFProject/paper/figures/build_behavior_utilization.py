"""Behavior-utilization figure: concrete strategy proxies by policy x pole.

Answers the professor's question: do different situations evoke different *behaviors*,
not just different win rates?

Primary figure (fig_behavior_utilization):
  Sealed n=128 exploratory 4v4 specialist diagnostic
  (4V4_EXPLORATORY_SPECIALIST_BEHAVIOR_DIAGNOSTIC.json), recomputed cell means with the
  project-frozen bootstrap. Five panels map to the protocol's measurement categories:
    1. Defender allocation   -- team-mean own-half time (%)
    2. Attacker allocation   -- team-mean enemy-half time (%)
    3. Intercept coverage    -- distinct nearest-agent assignment when >=2 intruders (%)
    4. Carrier support       -- mean carrier--nearest-teammate distance (cells; lower=tighter)
    5. Flag pressure timing  -- first tick any blue agent presses the enemy flag

Companion preview (fig_behavior_roles_c2_preview):
  Single sealed C2 verification seed (16400001) role mix from qualitative tick logs.
  ILLUSTRATIVE ONLY (n=1) -- shows the role vocabulary the qualitative frame grid will use;
  not a statistical claim.

Run:  python paper/figures/build_behavior_utilization.py
"""
from __future__ import annotations

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

DIAG = ROOT / "artifacts/strategic_demand/sppo/4V4_EXPLORATORY_SPECIALIST_BEHAVIOR_DIAGNOSTIC.json"
INTER = ROOT / "artifacts/strategic_demand/sppo/4V4_EXPLORATORY_SPECIALIST_BEHAVIOR_INTERACTION_ANALYSIS.json"
TICK_DIR = ROOT / "artifacts/qualitative_capture/4v4_c2_verification"

POLICY_COLOR = {"pi_A": COLORS["A"], "pi_B": COLORS["B"]}
POLICY_LABEL = {
    "pi_A": r"$\pi_A$ (trained vs Pole A)",
    "pi_B": r"$\pi_B$ (trained vs Pole B)",
}

# Panel definitions: (title, ylabel, extractor, scale, higher_is, interaction_key)
# higher_is documents the reading direction for captions; not drawn as an arrow.
PANELS = [
    {
        "title": "Defender allocation",
        "ylabel": "Own-half time (%)",
        "key": "own_half",
        "scale": 100.0,
        "ylim": (0, 100),
        "interaction_key": "own_half_frac_team_mean",
        "extract": lambda e: float(np.mean(e["own_half_frac"])),
    },
    {
        "title": "Attacker allocation",
        "ylabel": "Enemy-half time (%)",
        "key": "attack",
        "scale": 100.0,
        "ylim": (0, 100),
        "interaction_key": "time_past_midfield_team_mean",
        "extract": lambda e: float(np.mean(e["time_past_midfield"])),
    },
    {
        "title": "Intercept coverage",
        "ylabel": "Distinct coverage (%)",
        "key": "intercept",
        "scale": 100.0,
        "ylim": (0, 100),
        "interaction_key": "distinct_response_frac",
        "extract": lambda e: (
            None if e["distinct_response_frac"] is None
            else float(e["distinct_response_frac"])
        ),
    },
    {
        "title": "Carrier support",
        "ylabel": "Carrier–teammate dist.",
        "key": "carrier",
        "scale": 1.0,
        "ylim": (0, 5),
        "interaction_key": "carrier_support_dist_mean",
        "extract": lambda e: (
            None if e["carrier_support_dist_mean"] is None
            else float(e["carrier_support_dist_mean"])
        ),
    },
    {
        "title": "Flag pressure",
        "ylabel": "First pressure tick",
        "key": "pressure",
        "scale": 1.0,
        "ylim": (0, 30),
        "interaction_key": "first_flag_pressure_tick",
        "extract": lambda e: (
            None if e["first_flag_pressure_tick"] is None
            else float(e["first_flag_pressure_tick"])
        ),
    },
]


def _mean_ci(vals: list[float], n_boot=20000, alpha=0.05, rng_seed=7):
    xs = np.asarray([v for v in vals if v is not None], dtype=float)
    if xs.size == 0:
        raise ValueError("empty metric cell")
    rng = np.random.default_rng(rng_seed)
    boots = xs[rng.integers(0, xs.size, size=(n_boot, xs.size))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    mean = float(xs.mean())
    return mean, max(0.0, mean - lo), max(0.0, hi - mean), int(xs.size)


def _load_diagnostic():
    raw = json.loads(DIAG.read_text(encoding="utf-8"))["raw_per_episode"]
    inter = json.loads(INTER.read_text(encoding="utf-8"))["results"]
    return raw, inter


def build_utilization_figure() -> dict:
    apply_style()
    raw, inter = _load_diagnostic()

    n = len(PANELS)
    fig, axes = plt.subplots(1, n, figsize=(TWO_COLUMN, 3.05))
    group_x = {"A": 0.0, "B": 1.0}
    mode_dx = {"pi_A": -0.19, "pi_B": 0.19}
    bar_w = 0.34

    for ax, panel in zip(axes, PANELS):
        for pole in ("A", "B"):
            for policy in ("pi_A", "pi_B"):
                vals = [panel["extract"](e) for e in raw[policy][pole]]
                mean, err_lo, err_hi, _n = _mean_ci(vals)
                mean *= panel["scale"]
                err_lo *= panel["scale"]
                err_hi *= panel["scale"]
                x = group_x[pole] + mode_dx[policy]
                ax.bar(
                    x, mean, yerr=[[err_lo], [err_hi]], width=bar_w,
                    color=POLICY_COLOR[policy], edgecolor="black", linewidth=0.6,
                    capsize=2.5, error_kw={"elinewidth": 0.7, "capthick": 0.7},
                )
                label = f"{mean:.0f}" if panel["scale"] == 100.0 else f"{mean:.1f}"
                ax.text(x, mean + err_hi + 0.04 * (panel["ylim"][1] - panel["ylim"][0]),
                        label, ha="center", va="bottom", fontsize=6.5)

        ax.set_xticks([group_x["A"], group_x["B"]])
        ax.set_xticklabels(["Pole A", "Pole B"], fontsize=7.5)
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(*panel["ylim"])
        ax.set_title(panel["title"], fontsize=8.5, fontweight="bold")
        ax.set_ylabel(panel["ylabel"], fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        cls = inter[panel["interaction_key"]]["classification"]
        # A = true policy x pole interaction; B = main effect only; C = null
        tag = {"A": "interaction", "B": "main effect", "C": "null"}[cls]
        ax.text(
            0.5, -0.22, f"({cls}: {tag})", transform=ax.transAxes,
            ha="center", va="top", fontsize=6.5, style="italic", color="#444444",
        )

    handles = [
        mpatches.Patch(facecolor=POLICY_COLOR[p], edgecolor="black", label=POLICY_LABEL[p])
        for p in ("pi_A", "pi_B")
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.08),
        ncol=2, frameon=False, fontsize=8,
    )
    caption = (
        "Concrete strategy proxies on the sealed exploratory 4v4 crossover seeds "
        "(n=128 matched episodes/cell). Desired situational specialization shows as "
        "different bar heights under Pole A vs Pole B for the same policy. "
        "Panel tags: (A) policy$\\times$pole interaction CI excludes 0; "
        "(B) main effect only; (C) null. "
        "Source: 4V4_EXPLORATORY_SPECIALIST_BEHAVIOR_DIAGNOSTIC.json."
    )
    fig.text(0.5, -0.06, caption, ha="center", va="top", fontsize=7.5, style="italic")
    fig.subplots_adjust(wspace=0.45, bottom=0.28, top=0.82, left=0.06, right=0.99)

    return save_figure(fig, "fig_behavior_utilization")


ROLE_ORDER = ("DEFENDER", "ATTACKER", "INTERCEPTOR", "FLAG_RETR", "ESCORT", "COUNTER", "2V1_WING")
ROLE_COLORS = {
    "DEFENDER": "#0072B2",
    "ATTACKER": "#D55E00",
    "INTERCEPTOR": "#009E73",
    "FLAG_RETR": "#CC79A7",
    "ESCORT": "#E69F00",
    "COUNTER": "#56B4E9",
    "2V1_WING": "#999999",
}
C2_CELLS = [
    ("piA2_poleA_seed16400001_tick_log.json", r"$\pi_{A2}$ vs A"),
    ("piA2_poleB_seed16400001_tick_log.json", r"$\pi_{A2}$ vs B"),
    ("piB2_poleA_seed16400001_tick_log.json", r"$\pi_{B2}$ vs A"),
    ("piB2_poleB_seed16400001_tick_log.json", r"$\pi_{B2}$ vs B"),
]


def _role_fractions(tick_path: Path) -> dict[str, float]:
    ticks = json.loads(tick_path.read_text(encoding="utf-8"))
    counts: dict[str, float] = {r: 0.0 for r in ROLE_ORDER}
    total = 0.0
    switches = 0
    prev = None
    pressure = 0
    carrier_dists = []
    for t in ticks:
        rc = t.get("red_role_counts") or {}
        key = tuple(sorted((k, int(v)) for k, v in rc.items()))
        if prev is not None and key != prev:
            switches += 1
        prev = key
        for k, v in rc.items():
            counts[k] = counts.get(k, 0.0) + float(v)
            total += float(v)
        if t.get("blue_flag_pressure"):
            pressure += 1
        d = t.get("carrier_nearest_teammate_dist")
        if d is not None:
            carrier_dists.append(float(d))
    fracs = {k: (counts.get(k, 0.0) / total if total else 0.0) for k in ROLE_ORDER}
    # Role persistence: fraction of consecutive tick pairs with unchanged role mix
    persist = 1.0 - (switches / max(1, len(ticks) - 1))
    meta = {
        "flag_pressure_frac": pressure / max(1, len(ticks)),
        "carrier_support_mean": float(np.mean(carrier_dists)) if carrier_dists else None,
        "role_persistence": persist,
        "n_ticks": len(ticks),
    }
    return fracs, meta


def build_c2_role_preview() -> dict:
    """Stacked role-mix + persistence for the single sealed C2 verification seed."""
    apply_style()
    fig, (ax_roles, ax_persist) = plt.subplots(
        1, 2, figsize=(TWO_COLUMN, 2.7),
        gridspec_kw={"width_ratios": [2.4, 1.0]},
    )

    xs = np.arange(len(C2_CELLS))
    bottoms = np.zeros(len(C2_CELLS))
    metas = []
    used_roles = []

    for i, (fname, _label) in enumerate(C2_CELLS):
        fracs, meta = _role_fractions(TICK_DIR / fname)
        metas.append(meta)
        for role in ROLE_ORDER:
            h = fracs[role] * 100.0
            if h <= 0:
                continue
            if role not in used_roles:
                used_roles.append(role)
            ax_roles.bar(
                xs[i], h, bottom=bottoms[i], width=0.62,
                color=ROLE_COLORS[role], edgecolor="black", linewidth=0.4,
            )
            bottoms[i] += h

    ax_roles.set_xticks(xs)
    ax_roles.set_xticklabels([lab for _, lab in C2_CELLS], fontsize=8)
    ax_roles.set_ylim(0, 100)
    ax_roles.set_ylabel("Role mix (% of agent-ticks)")
    ax_roles.set_title("C2 role utilization (single seed)", fontsize=9.5, fontweight="bold")
    ax_roles.spines["top"].set_visible(False)
    ax_roles.spines["right"].set_visible(False)
    ax_roles.legend(
        handles=[mpatches.Patch(facecolor=ROLE_COLORS[r], edgecolor="black", label=r)
                 for r in used_roles],
        loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=7,
    )

    persist = [m["role_persistence"] * 100.0 for m in metas]
    colors = [COLORS["A"], COLORS["A"], COLORS["B"], COLORS["B"]]
    ax_persist.bar(
        xs, persist, width=0.62, color=colors, edgecolor="black", linewidth=0.5,
    )
    for x, v in zip(xs, persist):
        ax_persist.text(x, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax_persist.set_xticks(xs)
    ax_persist.set_xticklabels([lab for _, lab in C2_CELLS], fontsize=8)
    ax_persist.set_ylim(0, 100)
    ax_persist.set_ylabel("Role-mix persistence (%)")
    ax_persist.set_title("Role persistence", fontsize=9.5, fontweight="bold")
    ax_persist.spines["top"].set_visible(False)
    ax_persist.spines["right"].set_visible(False)

    caption = (
        "ILLUSTRATIVE ONLY (n=1 sealed verification seed 16400001). "
        "Left: fraction of agent-ticks in each scripted-style role tag. "
        "Right: persistence = share of consecutive ticks with unchanged team role mix. "
        "Not a confidence-interval claim -- preview of the qualitative capture vocabulary."
    )
    fig.text(0.5, -0.08, caption, ha="center", va="top", fontsize=7.5, style="italic")
    fig.subplots_adjust(wspace=0.35, bottom=0.28, top=0.88, left=0.08, right=0.98)

    return save_figure(fig, "fig_behavior_roles_c2_preview")


def main() -> None:
    p1 = build_utilization_figure()
    p2 = build_c2_role_preview()
    print({"utilization": p1, "c2_role_preview": p2})


if __name__ == "__main__":
    main()
