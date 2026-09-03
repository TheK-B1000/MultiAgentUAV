"""DARS-style grouped bar chart: multi-panel win-rate comparison with error bars.

Matches the PI's prior DARS paper figure exactly (e.g. "Win rate % in ASV defense for
various values of |N|"): one panel per condition (team size, in our case), a small set of
colored bars per panel (method comparison), error bars, and a value+error label printed
directly above each bar ("97.0% +/- 1.7"). This is a DIFFERENT visual language from
figure_style.py's CI-forest-plot style (used for Figure 2's crossover deltas) and serves a
different purpose: a forest plot shows whether a confidence interval clears zero; this shows
magnitude comparison against a 0-100% axis. Both are wanted for different results -- this
module does not replace figure_style.py, it complements it.

Shares figure_style's type/embedding/export discipline (Times New Roman, IEEE column widths,
vector PDF canonical + PNG preview) rather than duplicating it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt

from paper.figures.figure_style import ONE_COLUMN, TWO_COLUMN, apply_style, save_figure

# Method-comparison palette (green/blue/purple), distinct from figure_style.COLORS'
# pole-comparison palette (blue/vermillion) -- a different semantic axis (which METHOD,
# not which POLE), so it is deliberately its own palette rather than reusing that one.
METHOD_COLORS = {
    "ours": "#2CA02C",       # green
    "baseline_1": "#1F77B4", # blue
    "baseline_2": "#8E44AD", # purple
}
METHOD_ORDER = ("ours", "baseline_1", "baseline_2")


class Bar(TypedDict):
    label: str          # bar x-axis label, e.g. "Ours", "Jacob et al.", "Self-play"
    value: float         # percentage, 0-100
    error: float          # +/- error bar magnitude, same units as value
    color_key: str         # one of METHOD_COLORS' keys


class Panel(TypedDict):
    title: str           # panel title, e.g. "2v2 Win rate"
    ylabel: str           # e.g. "Win rate vs OP4 (%)"
    bars: Sequence[Bar]


def _panel(ax, panel: Panel) -> None:
    xs = range(len(panel["bars"]))
    for x, bar in zip(xs, panel["bars"]):
        color = METHOD_COLORS[bar["color_key"]]
        ax.bar(x, bar["value"], yerr=bar["error"], width=0.6, color=color,
              edgecolor="black", linewidth=0.6, capsize=3,
              error_kw={"elinewidth": 0.8, "capthick": 0.8})
        ax.text(x, bar["value"] + bar["error"] + 2.5, f"{bar['value']:.1f}% ± {bar['error']:.1f}",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([b["label"] for b in panel["bars"]])
    ax.set_ylim(0, 100)
    ax.set_ylabel(panel["ylabel"])
    ax.set_title(panel["title"], fontsize=9.5, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_grouped_bar_figure(panels: Sequence[Panel], name: str, caption: str | None = None):
    """One evenly-spaced row of panels, DARS-style. Width scales with panel count."""
    apply_style()
    n = len(panels)
    width = TWO_COLUMN if n > 1 else ONE_COLUMN
    fig, axes = plt.subplots(1, n, figsize=(width, 2.6), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, panel in zip(axes, panels):
        _panel(ax, panel)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)   # sharey=True gotcha from Figure 2 -- see its history
        ax.set_ylabel("")
    fig.subplots_adjust(wspace=0.15, bottom=0.28)
    if caption:
        fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=8.5,
                 style="italic", wrap=True)
    return save_figure(fig, name)


if __name__ == "__main__":
    # Smoke test with clearly-fake placeholder numbers, matching the reference figure's
    # shape (3 panels, 3 bars/panel) -- NOT real data. Verifies rendering + font embedding
    # only, exactly like figure_style.py's own smoke test.
    demo_panels: list[Panel] = [
        {"title": f"{n}v{n} Win rate", "ylabel": "Win rate vs OP4 (%)" if i == 0 else "",
         "bars": [
             {"label": "Ours", "value": v0, "error": e0, "color_key": "ours"},
             {"label": "Baseline 1", "value": v1, "error": e1, "color_key": "baseline_1"},
             {"label": "Baseline 2", "value": v2, "error": e2, "color_key": "baseline_2"},
         ]}
        for i, (n, v0, e0, v1, e1, v2, e2) in enumerate([
            (2, 97.0, 1.7, 90.0, 3.0, 86.0, 3.5),
            (4, 94.0, 2.4, 93.0, 2.6, 85.0, 3.6),
            (6, 95.0, 2.2, 94.0, 2.4, 84.0, 3.7),
        ])
    ]
    paths = build_grouped_bar_figure(demo_panels, "_smoke_test_dars_bars",
                                     caption="SMOKE TEST -- placeholder numbers, not real data.")
    print(paths)
