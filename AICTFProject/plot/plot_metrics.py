from __future__ import annotations

"""
Utility script to turn CSV metrics from custom PPO training/eval into LaTeX-ready plots.

Expected CSV format
-------------------
This script is designed for the episode-level CSVs written by the training/eval
pipeline (e.g. files like `checkpoints/2v2/ppo_latent_fixed_op3_2v2_metrics.csv` or the
`eval_*.csv` files produced by the viewer).

The CSV must contain at least:
    - episode_id (int)
    - success (0/1)

Optionally, it may also contain:
    - collisions_per_episode
    - zone_coverage
    - opponent
    - strategy_switch_rate
    - strategy_occupancy_0, strategy_occupancy_1, ...
    - strategy_phase_neutral_occupancy_0, strategy_phase_blue_attack_occupancy_1, ...

The script produces high-DPI PDF (and PNG) figures that can be included in LaTeX
via \\includegraphics.
"""

import argparse
import csv
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _is_finite(value: float) -> bool:
    return math.isfinite(float(value))


def _moving_average(values: List[float], window: int) -> Tuple[List[int], List[float]]:
    """Return (indices, smoothed_values) using a simple moving average."""
    if window <= 1 or len(values) == 0:
        return list(range(len(values))), values[:]
    out: List[float] = []
    idx: List[int] = []
    cumsum = 0.0
    for i, v in enumerate(values):
        cumsum += float(v)
        if i >= window:
            cumsum -= float(values[i - window])
        if i >= window - 1:
            out.append(cumsum / float(window))
            idx.append(i)
    return idx, out


def _load_csv(path: Path) -> Dict[str, List[float]]:
    """Load a metrics CSV into column -> list[float]."""
    cols: Dict[str, List[float]] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k not in cols:
                    cols[k] = []
                if v is None or v == "":
                    cols[k].append(float("nan"))
                else:
                    try:
                        cols[k].append(float(v))
                    except ValueError:
                        # Non-numeric (e.g. opponent); store NaN so lengths match.
                        cols[k].append(float("nan"))
    return cols


def _default_label(path: Path) -> str:
    name = path.stem
    if name.endswith("_metrics"):
        name = name[: -len("_metrics")]
    return name


def plot_success(csv_paths: List[Path], out_dir: Path, window: int = 50) -> None:
    """Plot smoothed success rate vs episode_id for one or more runs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "font.family": "serif",
            "figure.figsize": (5.0, 3.0),
        }
    )

    fig, ax = plt.subplots()

    for csv_path in csv_paths:
        data = _load_csv(csv_path)
        if "episode_id" not in data or "success" not in data:
            print(f"[plot_metrics] Skipping {csv_path} (missing episode_id or success).")
            continue
        episodes = [int(e) for e in data["episode_id"]]
        success = data["success"]
        idx, smooth = _moving_average(success, window)
        if not smooth:
            continue
        x = [episodes[i] for i in idx]
        label = _default_label(csv_path)
        ax.plot(x, smooth, label=label)

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Success rate (moving avg, window={window})")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.set_title("Episode success over training/eval")

    out_pdf = out_dir / "success_vs_episode.pdf"
    out_png = out_dir / "success_vs_episode.png"
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_metrics] Wrote {out_pdf} and {out_png}")


def plot_collisions(csv_paths: List[Path], out_dir: Path, window: int = 50) -> None:
    """Optional: plot collisions per episode (if available)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    for csv_path in csv_paths:
        data = _load_csv(csv_path)
        if "episode_id" not in data or "collisions_per_episode" not in data:
            print(f"[plot_metrics] Skipping {csv_path} for collisions (missing columns).")
            continue
        episodes = [int(e) for e in data["episode_id"]]
        colls = data["collisions_per_episode"]
        idx, smooth = _moving_average(colls, window)
        if not smooth:
            continue
        x = [episodes[i] for i in idx]
        label = _default_label(csv_path)
        ax.plot(x, smooth, label=label)

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Collisions per episode (moving avg, window={window})")
    ax.legend()
    ax.set_title("Collisions over training/eval")

    out_pdf = out_dir / "collisions_vs_episode.pdf"
    out_png = out_dir / "collisions_vs_episode.png"
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_metrics] Wrote {out_pdf} and {out_png}")


def plot_strategy_switches(csv_paths: List[Path], out_dir: Path, window: int = 50) -> None:
    """Optional: plot latent strategy switch rate (if available)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    plotted = False

    for csv_path in csv_paths:
        data = _load_csv(csv_path)
        if "episode_id" not in data or "strategy_switch_rate" not in data:
            print(f"[plot_metrics] Skipping {csv_path} for strategy switches (missing columns).")
            continue
        episodes = [int(e) for e in data["episode_id"]]
        switch_rate = data["strategy_switch_rate"]
        idx, smooth = _moving_average(switch_rate, window)
        filtered = [(episodes[i], v) for i, v in zip(idx, smooth) if _is_finite(v)]
        if not filtered:
            continue
        x, y = zip(*filtered)
        label = _default_label(csv_path)
        ax.plot(list(x), list(y), label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Strategy switch rate (moving avg, window={window})")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.set_title("Latent strategy switching")

    out_pdf = out_dir / "strategy_switch_rate_vs_episode.pdf"
    out_png = out_dir / "strategy_switch_rate_vs_episode.png"
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_metrics] Wrote {out_pdf} and {out_png}")


def plot_strategy_occupancy(csv_paths: List[Path], out_dir: Path) -> None:
    """Optional: plot mean occupancy for each latent strategy id (if available)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    plotted = False

    for run_idx, csv_path in enumerate(csv_paths):
        data = _load_csv(csv_path)
        occ_keys = sorted(
            (k for k in data if k.startswith("strategy_occupancy_")),
            key=lambda key: (
                0,
                int(key.rsplit("_", 1)[-1]),
            )
            if key.rsplit("_", 1)[-1].isdigit()
            else (1, key),
        )
        if not occ_keys:
            print(f"[plot_metrics] Skipping {csv_path} for strategy occupancy (missing columns).")
            continue

        means: List[float] = []
        for key in occ_keys:
            vals = [float(v) for v in data[key] if _is_finite(v)]
            means.append(sum(vals) / float(len(vals)) if vals else float("nan"))
        finite_pairs = [(idx, val) for idx, val in enumerate(means) if _is_finite(val)]
        if not finite_pairs:
            continue

        x_base = [idx for idx, _ in finite_pairs]
        y = [val for _, val in finite_pairs]
        width = min(0.8 / max(1, len(csv_paths)), 0.35)
        offset = (run_idx - (len(csv_paths) - 1) / 2.0) * width
        x = [v + offset for v in x_base]
        ax.bar(x, y, width=width, label=_default_label(csv_path), alpha=0.9)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Strategy id")
    ax.set_ylabel("Mean occupancy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Latent strategy occupancy")
    ax.legend()

    out_pdf = out_dir / "strategy_occupancy.pdf"
    out_png = out_dir / "strategy_occupancy.png"
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_metrics] Wrote {out_pdf} and {out_png}")


def plot_strategy_phase_occupancy(csv_paths: List[Path], out_dir: Path) -> None:
    """Optional: plot strategy occupancy conditioned on coarse game phase."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^strategy_phase_(.+)_occupancy_(\d+)$")
    phase_values: Dict[str, Dict[int, List[float]]] = {}

    for csv_path in csv_paths:
        data = _load_csv(csv_path)
        matched = False
        for key, values in data.items():
            m = pattern.match(key)
            if not m:
                continue
            matched = True
            phase = m.group(1)
            z_idx = int(m.group(2))
            phase_values.setdefault(phase, {}).setdefault(z_idx, []).extend(
                float(v) for v in values if _is_finite(v)
            )
        if not matched:
            print(f"[plot_metrics] Skipping {csv_path} for phase occupancy (missing columns).")

    if not phase_values:
        return

    phases = sorted(phase_values)
    strategies = sorted({z for per_phase in phase_values.values() for z in per_phase})
    if not strategies:
        return

    fig, ax = plt.subplots(figsize=(max(5.0, 1.2 * len(phases)), 3.2))
    x = list(range(len(phases)))
    bottoms = [0.0 for _ in phases]
    for z_idx in strategies:
        vals: List[float] = []
        for phase in phases:
            raw = phase_values.get(phase, {}).get(z_idx, [])
            vals.append(sum(raw) / float(len(raw)) if raw else 0.0)
        ax.bar(x, vals, bottom=bottoms, label=f"z={z_idx}", alpha=0.9)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=25, ha="right")
    ax.set_ylabel("Mean occupancy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Latent strategy occupancy by game phase")
    ax.legend(title="Strategy")

    out_pdf = out_dir / "strategy_phase_occupancy.pdf"
    out_png = out_dir / "strategy_phase_occupancy.png"
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_metrics] Wrote {out_pdf} and {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot custom PPO CTF metrics CSVs for LaTeX-ready figures.")
    parser.add_argument(
        "csv",
        nargs="+",
        help="One or more metrics CSV files (e.g. checkpoints/2v2/ppo_latent_fixed_op3_2v2_metrics.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to write figures (default: figures/ under script directory).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Moving-average window in episodes (default: 50).",
    )
    parser.add_argument(
        "--no-collisions",
        action="store_true",
        help="If set, skip plotting collisions_per_episode.",
    )
    parser.add_argument(
        "--no-strategy",
        action="store_true",
        help="If set, skip plotting latent strategy switch/occupancy diagnostics.",
    )
    args = parser.parse_args()

    # Output under AICTFProject/figures (parent of plot/ when script is in plot/)
    project_root = Path(__file__).resolve().parent.parent
    default_out_dir = project_root / "figures"
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out_dir

    csv_paths = [Path(p).expanduser().resolve() for p in args.csv]

    plot_success(csv_paths, out_dir, window=args.window)
    if not args.no_collisions:
        plot_collisions(csv_paths, out_dir, window=args.window)
    if not args.no_strategy:
        plot_strategy_switches(csv_paths, out_dir, window=args.window)
        plot_strategy_occupancy(csv_paths, out_dir)
        plot_strategy_phase_occupancy(csv_paths, out_dir)


if __name__ == "__main__":
    main()
