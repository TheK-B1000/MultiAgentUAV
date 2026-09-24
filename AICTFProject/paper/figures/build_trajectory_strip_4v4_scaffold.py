"""Matched A' vs pi_B trajectory strip (4v4 scaffold PASS*).

Reads artifacts/qualitative_capture/4v4_scaffold_trajectory_strip/trajectories.json
(produced by experiments/export_matched_scaffold_trajectories_4v4.py).

Layout: two panels (Pole A / Pole B). Blue agent paths under A' and pi_B are
overlaid from the same sealed seed. Trails are drawn in light blue; A' forced
defenders use a slightly darker stroke. Stars mark first action divergence.

Scientific claim:
  SUPPORTS: imposing 2A/2D on pi_A changes visible team geometry vs pi_B under
            matched seeds (connects Delta' payoff separation to behavior).
  DOES NOT: claim a learned latent z; A' is a scaffolded controller.

Run:  ./.venv/Scripts/python.exe paper/figures/build_trajectory_strip_4v4_scaffold.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.figures.figure_style import COLORS, TWO_COLUMN, apply_style, save_figure

SRC = ROOT / "artifacts/qualitative_capture/4v4_scaffold_trajectory_strip/trajectories.json"
STEM = "fig_trajectory_strip_4v4_scaffold"
COMBINED = ROOT / "paper" / "plots" / "combined"

# Light-blue path trails (colorblind-safe companion to Wong blue/vermillion)
LIGHT_BLUE = "#A6CEE3"
LIGHT_BLUE_DIM = "#C8DDF0"
APOSE_DARK = "#4A90C0"  # forced defenders under A'
BMODE = COLORS["B"]


def _cell(blob: dict, arm: str, pole: str) -> dict:
    for c in blob["cells"]:
        if c["arm"] == arm and c["pole"] == pole:
            return c
    raise KeyError((arm, pole))


def _paths(cell: dict) -> tuple[np.ndarray, list[int]]:
    ticks = cell["ticks"]
    n = len(ticks[0]["blue_x"])
    arr = np.zeros((len(ticks), n, 2), dtype=float)
    for t, row in enumerate(ticks):
        for i in range(n):
            arr[t, i, 0] = row["blue_x"][i]
            arr[t, i, 1] = row["blue_y"][i]
    forced = cell.get("forced_ids") or [
        i for i, v in enumerate(ticks[0]["forced_defend"]) if v
    ]
    return arr, list(forced)


def _panel(ax, blob: dict, pole: str) -> None:
    a = _cell(blob, "A_prime", pole)
    b = _cell(blob, "pi_B", pole)
    pa, forced = _paths(a)
    pb, _ = _paths(b)
    mark = blob["divergence_by_pole"][pole].get("mark_tick")
    n = pa.shape[1]

    # Trajectories taken so far: light-blue trails for every blue agent
    for i in range(n):
        is_forced = i in forced
        ax.plot(
            pa[:, i, 0], pa[:, i, 1],
            color=APOSE_DARK if is_forced else LIGHT_BLUE,
            ls="-", lw=1.8 if is_forced else 1.2,
            alpha=0.95 if is_forced else 0.85,
            zorder=3,
            label=(
                r"$A'$ forced DEFEND" if is_forced and i == forced[0]
                else (r"$A'$ other agents" if (not is_forced and i == next(j for j in range(n) if j not in forced)) else None)
            ),
        )
        ax.plot(
            pb[:, i, 0], pb[:, i, 1],
            color=BMODE if i == 0 else LIGHT_BLUE_DIM,
            ls="--", lw=1.5 if i == 0 else 1.0,
            alpha=0.9 if i == 0 else 0.55,
            zorder=2,
            label=(r"$\pi_B$ agent 0" if i == 0 else (r"$\pi_B$ other agents" if i == 1 else None)),
        )

    # starts
    for i in range(n):
        ax.plot(pa[0, i, 0], pa[0, i, 1], marker="o", color="#222", ms=3.5, zorder=5)

    if mark is not None and mark < len(pa) and mark < len(pb):
        ax.plot(pa[mark, forced[0], 0], pa[mark, forced[0], 1],
                marker="*", color=APOSE_DARK, ms=11, zorder=6)
        ax.plot(pb[mark, 0, 0], pb[mark, 0, 1],
                marker="*", color=BMODE, ms=11, zorder=6)
        ax.annotate(
            f"first action diverge t={mark}",
            xy=(pa[mark, forced[0], 0], pa[mark, forced[0], 1]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=7, color="#222",
        )

    # home marker
    home = a["ticks"][0]["flag_home"]
    ax.plot(home[0], home[1], marker="s", color="#333", ms=6, zorder=4)
    ax.annotate("own flag home", xy=(home[0], home[1]), xytext=(4, -10),
                textcoords="offset points", fontsize=6.5, color="#333")

    ta, tb = a["terminal"], b["terminal"]
    ax.set_title(
        f"Pole {pole}  |  $A'$ {ta['blue']}–{ta['red']}   "
        f"$\\pi_B$ {tb['blue']}–{tb['red']}",
        fontsize=9, fontweight="bold",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False, fontsize=6.0, ncol=1)


def main() -> dict:
    if not SRC.exists():
        raise SystemExit(
            f"REFUSING: missing {SRC}. Run "
            "python experiments/export_matched_scaffold_trajectories_4v4.py --device cuda first."
        )
    blob = json.loads(SRC.read_text(encoding="utf-8"))
    if blob.get("fidelity") != "ALL_MATCH":
        raise SystemExit(f"REFUSING: fidelity={blob.get('fidelity')}")

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COLUMN, 3.4))
    _panel(axes[0], blob, "A")
    _panel(axes[1], blob, "B")

    seed = blob["seed"]
    caption = (
        f"Matched seed {seed}; light-blue trails = blue paths so far. "
        r"Solid=$A'=\pi_A+2$D (darker = forced defenders); dashed=$\pi_B$. "
        "CUDA fidelity MATCH vs sealed scaffold rows. "
        "READ AS: scaffolded structure changes geometry. "
        "DO NOT READ AS: a learned latent $z$ flip."
    )
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=7.0, style="italic")
    fig.subplots_adjust(wspace=0.22, bottom=0.18, top=0.88, left=0.06, right=0.99)

    paths = save_figure(fig, STEM)
    COMBINED.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        src = ROOT / "paper" / "generated" / f"{STEM}.{ext}"
        shutil.copy2(src, COMBINED / f"{STEM}.{ext}")
    print(paths)
    return paths


if __name__ == "__main__":
    main()
