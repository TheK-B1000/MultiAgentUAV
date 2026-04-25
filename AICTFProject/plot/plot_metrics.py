from __future__ import annotations

"""
Utility script to turn CSV metrics from custom PPO training/eval into LaTeX-ready plots.

Expected CSV format
-------------------
This script is designed for the episode-level CSVs written by the training/eval
pipeline (e.g. files like `checkpoints/2v2/ppo_custom_fixed_op3_2v2_metrics.csv` or the
`eval_*.csv` files produced by the viewer).

The CSV must contain at least:
    - episode_id (int)
    - success (0/1)

Optionally, it may also contain:
    - collisions_per_episode
    - zone_coverage
    - phase_name

The script produces high-DPI PDF (and PNG) figures that can be included in LaTeX
via \\includegraphics.
"""

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


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
                        # Non-numeric (e.g. phase_name); store NaN so lengths match.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot custom PPO CTF metrics CSVs for LaTeX-ready figures.")
    parser.add_argument(
        "csv",
        nargs="+",
        help="One or more metrics CSV files (e.g. checkpoints/2v2/ppo_custom_fixed_op3_2v2_metrics.csv).",
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
    args = parser.parse_args()

    # Output under AICTFProject/figures (parent of plot/ when script is in plot/)
    project_root = Path(__file__).resolve().parent.parent
    default_out_dir = project_root / "figures"
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out_dir

    csv_paths = [Path(p).expanduser().resolve() for p in args.csv]

    plot_success(csv_paths, out_dir, window=args.window)
    if not args.no_collisions:
        plot_collisions(csv_paths, out_dir, window=args.window)


if __name__ == "__main__":
    main()
