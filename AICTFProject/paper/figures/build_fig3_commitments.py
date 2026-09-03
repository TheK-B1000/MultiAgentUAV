"""Figure 3: the semi-MDP commitment boundary and the causal-branching method.

Two panels, both drawn as schematic diagrams (patches/lines/text, not a data plot -- there is
no frozen numeric result behind this figure, only the environment's own real mechanics):

  (a) Two agents committing to macro-actions for different, agent-specific tick counts, so
      their decision boundaries fall asynchronously. Uses the REAL macro names (GO_TO,
      GRAB_MINE, GET_FLAG, PLACE_MINE) and REAL default hold lengths from
      gpu_env/_config.py (macro_commit_*_ticks): GO_TO=4, GRAB_MINE=3, GET_FLAG=4,
      PLACE_MINE=2. A per-tick row shows the real per-agent decision predicate
      d_t^i = [commit_ticks_left_i <= 0] (gpu_env/_core/_step.py) -- most policy outputs are
      ignored mid-commitment; only the ticks where the predicate is true are executable
      decisions.

  (b) At one such boundary, the three-arm causal branch: R_0 (incumbent continues) / R_A
      (pi_A's SEQUENCE intervention) / R_B (pi_B's identical intervention), matched
      continuation seed, to terminal payoff, to (A_hat_A, A_hat_B), to the frozen routing
      rule, to a causal weight w_k. This is CCP_S2_SPEC.json#CAUSAL_ESTIMAND +
      #ROUTING_RULE drawn as a diagram, using the same predicate from panel (a) as the gate
      on which agent's supervision the resulting w_k can ever reach
      (rl/causal_supervision.py's d_ti).

Every identifier drawn here is real and grep-checked against the codebase, not illustrative:
GO_TO/GRAB_MINE/GET_FLAG/PLACE_MINE (macro_actions.py), commit_ticks_left (gpu_env/_core/
_step.py), the tick constants (gpu_env/_config.py), R_0/pi_A/pi_B and the estimator
(CCP_S2_SPEC.json).

Run:  python paper/figures/build_fig3_commitments.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from paper.figures.figure_style import COLORS, LINESTYLES, TWO_COLUMN, apply_style, save_figure

# real default hold lengths, gpu_env/_config.py
TICKS = {"GO_TO": 4, "GRAB_MINE": 3, "GET_FLAG": 4, "PLACE_MINE": 2}

# (macro, start_tick) per agent -- durations are the real per-macro constants above, so the
# two agents' decision boundaries fall on different ticks purely from that asymmetry, exactly
# as the real environment produces it (no scripting of the asynchrony itself)
AGENT0 = [("GO_TO", 0), ("PLACE_MINE", TICKS["GO_TO"])]
AGENT1 = [("GRAB_MINE", 0), ("GET_FLAG", TICKS["GRAB_MINE"])]


def _boundaries(segments: list[tuple[str, int]]) -> list[int]:
    starts = [t for _, t in segments]
    last_macro, last_start = segments[-1]
    return starts + [last_start + TICKS[last_macro]]


def _panel_a(ax) -> None:
    row_y = {"Agent 0": 1.6, "Agent 1": 0.6}
    bar_h = 0.34
    for label, segments, color in (("Agent 0", AGENT0, COLORS["A"]),
                                   ("Agent 1", AGENT1, COLORS["B"])):
        y = row_y[label]
        for macro, start in segments:
            dur = TICKS[macro]
            ax.add_patch(Rectangle((start, y - bar_h / 2), dur, bar_h,
                                   facecolor=color, edgecolor="black", lw=0.7, alpha=0.85,
                                   zorder=2))
            ax.text(start + dur / 2, y, macro, ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold", zorder=3)
        bounds = _boundaries(segments)
        ax.plot(bounds, [y - bar_h / 2 - 0.10] * len(bounds), marker="^", ms=4,
                color="black", ls="none", zorder=4)
        # per-tick decision predicate row: accepted (filled) vs ignored-mid-commitment (open)
        total_end = bounds[-1]
        for t in range(total_end + 1):
            accepted = t in bounds
            ax.plot(t + 0.5, y - bar_h / 2 - 0.30,
                    marker=("o" if accepted else "x"),
                    ms=(4.2 if accepted else 3.2),
                    mfc=(color if accepted else "none"),
                    mec=("black" if accepted else "#999999"),
                    mew=0.7, zorder=4)
        ax.text(-0.35, y, label, ha="right", va="center", fontsize=9, fontweight="bold")

    max_t = max(_boundaries(AGENT0)[-1], _boundaries(AGENT1)[-1])
    for t in range(max_t + 1):
        ax.axvline(t, color="#dddddd", lw=0.5, zorder=0)
    ax.set_xlim(-1.7, max_t + 0.6)
    ax.set_ylim(0.0, 2.15)
    ax.set_xticks(range(max_t + 1))
    ax.set_xlabel("simulator tick")
    ax.set_yticks([])
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False)
    ax.text(max_t + 0.55, row_y["Agent 0"] - bar_h / 2 - 0.30,
            r"$d_t^i=[\mathtt{commit\_ticks\_left}_i\leq 0]$",
            ha="left", va="center", fontsize=8)
    ax.set_title("(a) asynchronous per-agent commitment boundaries", fontsize=9.5,
                 fontweight="bold", loc="left")


def _box(ax, xy, text, color, ls, w=1.5, h=0.5):
    b = FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                       boxstyle="round,pad=0.06", facecolor="white",
                       edgecolor=color, linewidth=1.3, linestyle=ls, zorder=3)
    ax.add_patch(b)
    ax.text(*xy, text, ha="center", va="center", fontsize=8.5, zorder=4)
    return xy


def _arrow(ax, p0, p1, color="black"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8,
                                 color=color, lw=1.0, zorder=2))


def _panel_b(ax) -> None:
    top = _box(ax, (5, 4.3), r"commitment boundary $s_t$", "black", "-", w=3.4, h=0.45)
    r0 = _box(ax, (1.6, 3.0), r"$R_0$" + "\nincumbent continues", COLORS["control"],
             LINESTYLES["control"])
    ra = _box(ax, (5.0, 3.0), r"$R_A$" + "\n" + r"$\pi_A$ SEQUENCE", COLORS["A"],
             LINESTYLES["A"])
    rb = _box(ax, (8.4, 3.0), r"$R_B$" + "\n" + r"$\pi_B$ SEQUENCE", COLORS["B"],
             LINESTYLES["B"])
    for node, color in ((r0, COLORS["control"]), (ra, COLORS["A"]), (rb, COLORS["B"])):
        _arrow(ax, (top[0], top[1] - 0.225), (node[0], node[1] + 0.25), color)

    pay0 = _box(ax, (1.6, 1.8), "terminal\npayoff", COLORS["control"], LINESTYLES["control"],
               w=1.3, h=0.45)
    paya = _box(ax, (5.0, 1.8), "terminal\npayoff", COLORS["A"], LINESTYLES["A"], w=1.3, h=0.45)
    payb = _box(ax, (8.4, 1.8), "terminal\npayoff", COLORS["B"], LINESTYLES["B"], w=1.3, h=0.45)
    for node, pay, color in ((r0, pay0, COLORS["control"]), (ra, paya, COLORS["A"]),
                             (rb, payb, COLORS["B"])):
        _arrow(ax, (node[0], node[1] - 0.25), (pay[0], pay[1] + 0.225), color)

    bottom = _box(ax, (5, 0.55),
                 r"$(\hat A_A,\hat A_B)\ \rightarrow\ $routing$\ \rightarrow\ w_k$",
                 "black", "-", w=4.6, h=0.55)
    for pay in (pay0, paya, payb):
        _arrow(ax, (pay[0], pay[1] - 0.225), (bottom[0] + (pay[0] - bottom[0]) * 0.08,
                                              bottom[1] + 0.275), "#888888")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(b) matched-seed causal branching at the boundary", fontsize=9.5,
                 fontweight="bold", loc="left")


def main() -> int:
    apply_style()
    fig = plt.figure(figsize=(TWO_COLUMN, 4.7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.6], hspace=0.65)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])
    _panel_a(axA)
    _panel_b(axB)

    # figure-level legend, fixed figure-fraction position in the gap between the two panels --
    # anchoring to axA's own (data-dependent) axes coordinates proved fragile (two earlier
    # attempts collided with the title, then with the tick labels)
    handles = [Line2D([], [], marker="o", mfc="black", mec="black", ls="none", ms=4.2,
                      label="decision accepted"),
              Line2D([], [], marker="x", mec="#999999", ls="none", ms=3.2,
                      label="ignored (committed)")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.565), ncol=2,
              frameon=False, handletextpad=0.4, columnspacing=1.5)

    paths = save_figure(fig, "fig3_commitments")
    print("Figure 3 written:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
