#!/usr/bin/env python3
"""
Offline regeneration of per-size win-rate PNGs with binomial SE error bars.

Does NOT load any models or run any evaluations. Reuses the W/L totals from
the existing 100-episode paper results (hard-coded below, read off the prior
PNGs / CSVs) and renders each figure in the same visual style as
plot_<N>v<N>_winrate.py so that Ours / Jacob et al. / Self-play bars now
carry ±1 binomial-SE caps and ``XX.X% ± Y.Y`` labels.

Usage:
    python plot/regen_winrate_figs.py
    # (overwrites figures/<N>v<N>_winrate_<OP>_100ep.png for N in 2,3,4,5 and OP in OP3,OP4)

If the numbers ever drift, update WINRATES below or re-run plot_<N>v<N>_winrate.py.
"""
from __future__ import annotations

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from eval_rollout import binomial_se  # noqa: E402

# Win-rate percentages from the current 100-episode PNGs (N = 100 per bar, draws=0).
# order: (Ours, Jacob et al., Self-play)
WINRATES: dict[tuple[int, str], tuple[float, float, float]] = {
    (2, "OP3"): (93.0, 91.0, 86.0),
    (3, "OP3"): (83.0, 83.0, 69.0),
    (4, "OP3"): (97.0, 98.0, 99.0),
    (5, "OP3"): (95.0, 96.0, 91.0),
    (2, "OP4"): (97.0, 90.0, 86.0),
    (3, "OP4"): (94.0, 93.0, 85.0),
    (4, "OP4"): (95.0, 94.0, 84.0),
    (5, "OP4"): (73.0, 78.0, 63.0),
}

LABELS = ["Ours", "Jacob et al.", "Self-play"]
COLORS = ["#2ecc71", "#3498db", "#9b59b6"]
N_EPISODES = 100


def _regen_one(team_size: int, opponent: str, win_rates: tuple[float, float, float], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wrs = list(win_rates)
    # Reconstruct wins from WR% (N=100 -> wins = round(WR)).
    wins = [int(round(wr * N_EPISODES / 100.0)) for wr in wrs]
    ses = [binomial_se(w, N_EPISODES) for w in wins]

    x = np.arange(len(LABELS))
    plt.rc("font", size=16)
    fig = plt.figure(figsize=(7.0, 5.2))
    bars = plt.bar(
        x, wrs, color=COLORS, edgecolor="black", linewidth=1.2,
        yerr=ses, capsize=6, error_kw={"elinewidth": 1.6, "ecolor": "black"},
    )
    plt.xticks(x, LABELS, fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel(f"Win rate vs {opponent} (%)", fontsize=20)
    plt.title(f"{team_size}v{team_size} Win rate", fontsize=22)
    plt.ylim(0, 115)
    for bar, wr, se in zip(bars, wrs, ses):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + se + 2.0,
            f"{wr:.1f}% \u00b1 {se:.1f}",
            ha="center", fontsize=16,
        )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}   ({'  '.join(f'{l}={w}%\u00b1{s:.1f}' for l, w, s in zip(LABELS, wrs, ses))})")


def main() -> None:
    figures_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    for (team_size, opponent), wrs in WINRATES.items():
        out_path = os.path.join(
            figures_dir, f"{team_size}v{team_size}_winrate_{opponent}_{N_EPISODES}ep.png"
        )
        _regen_one(team_size, opponent, wrs, out_path)


if __name__ == "__main__":
    main()
