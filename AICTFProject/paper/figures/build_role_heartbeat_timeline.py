"""Blue per-agent role swimlane + team offense/defense heartbeat.

DIAGNOSTIC / ILLUSTRATIVE — not a sealed gate figure.

Supports two CSV schemas:

1. Legacy getflag-preserve rows
   (seed,t,agent,carrying,near_*,macro_student|macro_driver)
2. Rich capture from experiments.capture_role_heartbeat_timeline
   (policy,pole,seed,t,agent,team,own_half,...,blue_role,red_role)

Role proxy (when blue_role is absent):
  CARRIER  carrying == 1
  OFFENSE  GET_FLAG, or near_enemy_flag while not on own half
  DEFENSE  GO_HOME, or own-half linger near own flag / non-offensive macro
  TRANSIT  everything else

Figures:
  fig_role_heartbeat_timeline      single-seed swimlane + heartbeat
  fig_role_heartbeat_multiseed     multi-seed heartbeat panel
  fig_role_heartbeat_piA_vs_piB    matched pi_A vs pi_B (+ red overlay)

Run:
  python paper/figures/build_role_heartbeat_timeline.py
  python paper/figures/build_role_heartbeat_timeline.py \\
      --csv artifacts/strategic_demand/sppo/role_heartbeat_timeline_agent_step_rows.csv \\
      --compare-ab --seed 16700001
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.figures.figure_style import TWO_COLUMN, apply_style, save_figure

DEFAULT_CSV = (
    ROOT
    / "artifacts/strategic_demand/sppo/b_getflag_preserve_manipulation_agent_step_rows.csv"
)
RICH_CSV = (
    ROOT
    / "artifacts/strategic_demand/sppo/role_heartbeat_timeline_agent_step_rows.csv"
)

ROLE_IDS = {
    "DEFENSE": 0,
    "TRANSIT": 1,
    "OFFENSE": 2,
    "CARRIER": 3,
}
ROLE_ORDER = ("DEFENSE", "TRANSIT", "OFFENSE", "CARRIER")
ROLE_COLORS = {
    "DEFENSE": "#0072B2",
    "TRANSIT": "#CCCCCC",
    "OFFENSE": "#D55E00",
    "CARRIER": "#009E73",
}

RED_ROLE_ORDER = (
    "DEFENDER",
    "ATTACKER",
    "INTERCEPTOR",
    "FLAG_RETR",
    "ESCORT",
    "COUNTER",
    "2V1_WING",
)
RED_ROLE_COLORS = {
    "DEFENDER": "#0072B2",
    "ATTACKER": "#D55E00",
    "INTERCEPTOR": "#009E73",
    "FLAG_RETR": "#CC79A7",
    "ESCORT": "#E69F00",
    "COUNTER": "#56B4E9",
    "2V1_WING": "#999999",
}
RED_ROLE_IDS = {r: i for i, r in enumerate(RED_ROLE_ORDER)}


def _as_int(v, default: int = 0) -> int:
    if v is None or v == "":
        return default
    return int(v)


def _macro_of(row: dict) -> str:
    return str(
        row.get("macro")
        or row.get("macro_student")
        or row.get("macro_driver")
        or ""
    )


def role_of(row: dict) -> str:
    """Resolve blue role: prefer captured blue_role, else macro+geometry proxy."""
    if row.get("blue_role"):
        return str(row["blue_role"])
    if _as_int(row.get("carrying")) == 1:
        return "CARRIER"
    macro = _macro_of(row)
    if macro == "GET_FLAG":
        return "OFFENSE"
    if macro == "GO_HOME":
        return "DEFENSE"
    near_own = _as_int(row.get("near_own_flag")) == 1
    near_enemy = _as_int(row.get("near_enemy_flag")) == 1
    own_half = row.get("own_half")
    if own_half is not None and own_half != "":
        on_home = _as_int(own_half) == 1
    else:
        # Legacy CSV has no own_half: treat near_own_flag as home-guard signal.
        on_home = near_own
    if on_home and (near_own or macro in ("GO_TO", "PLACE_MINE", "GRAB_MINE", "")):
        if not near_enemy:
            return "DEFENSE"
    if near_enemy and not on_home:
        return "OFFENSE"
    return "TRANSIT"


def load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        raise SystemExit(f"REFUSING: missing agent-step CSV: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _is_rich(rows: list[dict]) -> bool:
    return bool(rows) and ("policy" in rows[0] or "team" in rows[0])


def filter_blue(
    rows: list[dict],
    *,
    seed: int | None = None,
    policy: str | None = None,
    pole: str | None = None,
) -> list[dict]:
    out = []
    for r in rows:
        if r.get("team") not in (None, "", "blue"):
            continue
        if seed is not None and int(r["seed"]) != seed:
            continue
        if policy is not None and r.get("policy") != policy:
            continue
        if pole is not None and r.get("pole") != pole:
            continue
        out.append(r)
    return out


def filter_red(
    rows: list[dict],
    *,
    seed: int,
    policy: str,
    pole: str,
) -> list[dict]:
    return [
        r
        for r in rows
        if r.get("team") == "red"
        and int(r["seed"]) == seed
        and r.get("policy") == policy
        and r.get("pole") == pole
    ]


def episode_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    by_agent: dict[int, dict[int, str]] = defaultdict(dict)
    for r in rows:
        a = int(r["agent"])
        t = int(r["t"])
        by_agent[a][t] = role_of(r)
    if not by_agent:
        raise SystemExit("REFUSING: no blue rows for requested filter")
    agents = sorted(by_agent)
    t_max = max(max(ts) for ts in by_agent.values())
    T = t_max + 1
    mat = np.full((len(agents), T), ROLE_IDS["TRANSIT"], dtype=np.int16)
    for i, a in enumerate(agents):
        for t, role in by_agent[a].items():
            mat[i, t] = ROLE_IDS.get(role, ROLE_IDS["TRANSIT"])

    heart = np.zeros((3, T), dtype=np.float64)
    heart[0] = ((mat == ROLE_IDS["OFFENSE"]) | (mat == ROLE_IDS["CARRIER"])).sum(axis=0)
    heart[1] = (mat == ROLE_IDS["DEFENSE"]).sum(axis=0)
    heart[2] = (mat == ROLE_IDS["TRANSIT"]).sum(axis=0)
    return mat, heart, agents


def red_matrix(rows: list[dict]) -> tuple[np.ndarray, list[int]]:
    by_agent: dict[int, dict[int, str]] = defaultdict(dict)
    for r in rows:
        by_agent[int(r["agent"])][int(r["t"])] = str(r.get("red_role") or "ATTACKER")
    if not by_agent:
        return np.zeros((0, 0), dtype=np.int16), []
    agents = sorted(by_agent)
    t_max = max(max(ts) for ts in by_agent.values())
    T = t_max + 1
    mat = np.full((len(agents), T), RED_ROLE_IDS["ATTACKER"], dtype=np.int16)
    for i, a in enumerate(agents):
        for t, role in by_agent[a].items():
            mat[i, t] = RED_ROLE_IDS.get(role, RED_ROLE_IDS["ATTACKER"])
    return mat, agents


def available_seeds(rows: list[dict]) -> list[int]:
    return sorted({int(r["seed"]) for r in rows if r.get("team") in (None, "", "blue")})


def _swimlane(ax, mat: np.ndarray, agents: list[int], title: str, ylabel: str = "Blue agent") -> None:
    cmap = ListedColormap([ROLE_COLORS[r] for r in ROLE_ORDER])
    ax.imshow(
        mat,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-0.5,
        vmax=len(ROLE_ORDER) - 0.5,
        origin="upper",
    )
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels([f"agent {a}" for a in agents])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _red_swimlane(ax, mat: np.ndarray, agents: list[int], title: str) -> None:
    cmap = ListedColormap([RED_ROLE_COLORS[r] for r in RED_ROLE_ORDER])
    ax.imshow(
        mat,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-0.5,
        vmax=len(RED_ROLE_ORDER) - 0.5,
        origin="upper",
    )
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels([f"red {a}" for a in agents])
    ax.set_ylabel("Red BT")
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _heartbeat(ax, heart: np.ndarray, title: str, show_xlabel: bool = True) -> None:
    t = np.arange(heart.shape[1])
    ax.stackplot(
        t,
        heart[0],
        heart[1],
        heart[2],
        colors=[ROLE_COLORS["OFFENSE"], ROLE_COLORS["DEFENSE"], ROLE_COLORS["TRANSIT"]],
        linewidth=0.0,
    )
    ax.set_xlim(0, max(1, heart.shape[1] - 1))
    ax.set_ylim(0, max(1.0, float(heart.sum(axis=0).max())))
    ax.set_ylabel("# agents")
    if show_xlabel:
        ax.set_xlabel("tick t")
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_single_seed(rows: list[dict], seed: int, *, policy: str | None = None, pole: str | None = None) -> dict:
    apply_style()
    blue = filter_blue(rows, seed=seed, policy=policy, pole=pole)
    mat, heart, agents = episode_matrix(blue)
    red = []
    if policy is not None and pole is not None:
        red = filter_red(rows, seed=seed, policy=policy, pole=pole)
    red_mat, red_agents = red_matrix(red) if red else (np.zeros((0, 0)), [])

    n_panels = 3 if red_mat.size else 2
    height_ratios = [2.0, 1.4, 1.1] if red_mat.size else [2.2, 1.2]
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(TWO_COLUMN, 4.4 if red_mat.size else 3.6),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.22},
    )
    _swimlane(axes[0], mat, agents, "(a) Per-agent blue role proxy over time")
    if red_mat.size:
        _red_swimlane(axes[1], red_mat, red_agents, "(b) Per-agent red BT role")
        _heartbeat(axes[2], heart, "(c) Blue team heartbeat", show_xlabel=True)
        axes[0].tick_params(labelbottom=False)
        axes[1].tick_params(labelbottom=False)
    else:
        _heartbeat(axes[1], heart, "(b) Team heartbeat (role counts)", show_xlabel=True)
        axes[0].tick_params(labelbottom=False)

    handles = [mpatches.Patch(facecolor=ROLE_COLORS[r], edgecolor="black", label=r) for r in ROLE_ORDER]
    axes[0].legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=4,
        frameon=False,
        fontsize=7.5,
    )
    tag = f"seed {seed}"
    if policy:
        tag += f" | {policy}"
    if pole:
        tag += f" vs Pole {pole}"
    caption = (
        f"ILLUSTRATIVE ({tag}). Role proxy uses macro+carry+home geometry "
        "(near_own_flag / own_half). Not a sealed statistical claim."
    )
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=7.5, style="italic")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.12)
    return save_figure(fig, "fig_role_heartbeat_timeline")


def build_multi_seed_panel(rows: list[dict], seeds: list[int]) -> dict:
    apply_style()
    n = len(seeds)
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(TWO_COLUMN, 1.15 * n + 0.6),
        sharex=True,
        squeeze=False,
    )
    for i, seed in enumerate(seeds):
        blue = filter_blue(rows, seed=seed)
        # If rich CSV has multiple policies, default to first policy/pole pair for panel.
        if blue and blue[0].get("policy"):
            pol = blue[0]["policy"]
            pole = blue[0].get("pole") or "A"
            blue = filter_blue(rows, seed=seed, policy=pol, pole=pole)
        _, heart, _ = episode_matrix(blue)
        ax = axes[i, 0]
        t = np.arange(heart.shape[1])
        ax.stackplot(
            t,
            heart[0],
            heart[1],
            heart[2],
            colors=[ROLE_COLORS["OFFENSE"], ROLE_COLORS["DEFENSE"], ROLE_COLORS["TRANSIT"]],
            linewidth=0.0,
        )
        ax.set_ylim(0, 4.2)
        ax.set_ylabel(f"s{seed}\n#", fontsize=7.5)
        ax.set_yticks([0, 2, 4])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.set_title(
                "Team offense/defense heartbeat across seeds",
                fontsize=9.5,
                fontweight="bold",
            )
        if i == n - 1:
            ax.set_xlabel("tick t")

    handles = [
        mpatches.Patch(facecolor=ROLE_COLORS["OFFENSE"], edgecolor="black", label="offense+carrier"),
        mpatches.Patch(facecolor=ROLE_COLORS["DEFENSE"], edgecolor="black", label="defense"),
        mpatches.Patch(facecolor=ROLE_COLORS["TRANSIT"], edgecolor="black", label="transit"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
        fontsize=7.5,
    )
    caption = (
        "ILLUSTRATIVE multi-seed panel. "
        "Shows how team role mix pulses over an episode; not a gate."
    )
    fig.text(0.5, -0.01, caption, ha="center", va="top", fontsize=7.5, style="italic")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.12, hspace=0.25)
    return save_figure(fig, "fig_role_heartbeat_multiseed")


def build_piA_vs_piB(rows: list[dict], seed: int, pole: str = "B") -> dict:
    """Matched-seed swimlanes: pi_A | pi_B blue roles, plus red overlay under each."""
    apply_style()
    fig, axes = plt.subplots(
        4,
        2,
        figsize=(TWO_COLUMN, 6.2),
        sharex="col",
        gridspec_kw={"height_ratios": [1.6, 1.2, 1.0, 0.05], "hspace": 0.28, "wspace": 0.18},
    )
    # hide spacer row
    for c in range(2):
        axes[3, c].axis("off")

    for col, policy in enumerate(("pi_A", "pi_B")):
        blue = filter_blue(rows, seed=seed, policy=policy, pole=pole)
        red = filter_red(rows, seed=seed, policy=policy, pole=pole)
        mat, heart, agents = episode_matrix(blue)
        rmat, ragents = red_matrix(red)
        _swimlane(
            axes[0, col],
            mat,
            agents,
            rf"(a) ${policy[-1]}$ blue roles  |  Pole {pole}",
            ylabel="Blue" if col == 0 else "",
        )
        if rmat.size:
            _red_swimlane(
                axes[1, col],
                rmat,
                ragents,
                rf"(b) red BT vs ${policy[-1]}$",
            )
        else:
            axes[1, col].axis("off")
        _heartbeat(
            axes[2, col],
            heart,
            rf"(c) ${policy[-1]}$ heartbeat",
            show_xlabel=True,
        )
        axes[0, col].tick_params(labelbottom=False)
        axes[1, col].tick_params(labelbottom=False)

    blue_handles = [mpatches.Patch(facecolor=ROLE_COLORS[r], edgecolor="black", label=r) for r in ROLE_ORDER]
    red_handles = [
        mpatches.Patch(facecolor=RED_ROLE_COLORS[r], edgecolor="black", label=r)
        for r in RED_ROLE_ORDER
        if any(
            str(x.get("red_role")) == r
            for x in rows
            if x.get("team") == "red" and int(x["seed"]) == seed and x.get("pole") == pole
        )
    ]
    fig.legend(
        handles=blue_handles + red_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=False,
        fontsize=7.0,
    )
    caption = (
        f"ILLUSTRATIVE matched seed {seed} on Pole {pole}. "
        r"Left $\pi_A$, right $\pi_B$; red lanes are explicit BT roles. Not a gate."
    )
    fig.text(0.5, -0.01, caption, ha="center", va="top", fontsize=7.5, style="italic")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.10)
    return save_figure(fig, "fig_role_heartbeat_piA_vs_piB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-panel", type=int, default=4)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--pole", default="B")
    ap.add_argument(
        "--compare-ab",
        action="store_true",
        help="Build matched pi_A vs pi_B figure (requires rich capture CSV)",
    )
    args = ap.parse_args()

    csv_path = args.csv
    if csv_path is None:
        csv_path = RICH_CSV if RICH_CSV.is_file() else DEFAULT_CSV

    rows = load_rows(csv_path)
    seeds = available_seeds(rows)
    seed = int(args.seed) if args.seed is not None else seeds[0]
    if seed not in seeds:
        raise SystemExit(f"REFUSING: seed {seed} not in CSV (have {seeds[:8]}...)")

    policy = args.policy
    pole = str(args.pole or "B")
    if _is_rich(rows) and policy is None:
        # Prefer a single cell so swimlanes are never a mix of policies.
        policy = "pi_B"

    out1 = build_single_seed(rows, seed, policy=policy, pole=pole if _is_rich(rows) else None)
    print("wrote", out1)

    if args.compare_ab:
        if not _is_rich(rows):
            raise SystemExit("REFUSING: --compare-ab needs rich capture CSV with policy/pole/team")
        out_ab = build_piA_vs_piB(rows, seed, pole=pole)
        print("wrote", out_ab)
    else:
        panel_seeds = seeds[: max(1, int(args.n_panel))]
        out2 = build_multi_seed_panel(rows, panel_seeds)
        print("wrote", out2)
        print(f"panel_seeds={panel_seeds}")

    print(f"source={csv_path}")
    print(f"single_seed={seed} policy={policy} pole={pole}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
