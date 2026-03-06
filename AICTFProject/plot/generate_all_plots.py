#!/usr/bin/env python3
"""
Generate all interesting plots from project data in one run.

Uses training CSVs (fast, no model loading) by default. Optionally runs eval-based
scripts (win rates, eval metrics) which require checkpoints and take longer.

Usage:
  python generate_all_plots.py                    # Training CSV plots only (~30s)
  python generate_all_plots.py --eval             # Also run eval scripts (~5-10 min)
  python generate_all_plots.py --csv-dir DIR      # Custom CSV directory
  python generate_all_plots.py --out-dir figures  # Output directory

Outputs:
  - figures/success_vs_episode.png   : Learning curve (success rate)
  - figures/collisions_vs_episode.png: Collisions over training
  - figures/offense_metrics_*.png     : Time to first score, inter-robot dist, zone coverage, win margin
  - figures/offense_*.png             : Bar charts comparing runs (if multiple CSVs)
  - figures/all_winrates.png          : 2v2/3v3/4v4 win rates (if --eval)
  - figures/eval_metrics_*.png        : Performance, coordination, robustness, etc. (if --eval)
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CSV_DIR = PROJECT_ROOT / "checkpoints_sb3"
DEFAULT_OUT_DIR = PROJECT_ROOT / "figures"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "csv"


def plot_success_by_phase(csv_paths: list[Path], out_dir: Path) -> None:
    """Plot success rate by curriculum phase (OP1, OP2, OP3, etc.) per run."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rc("font", size=12)

    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        phase_success: dict[str, list[float]] = defaultdict(list)
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                phase = str(row.get("phase_name", "unknown")).strip() or "unknown"
                try:
                    s = float(row.get("success", 0))
                except (ValueError, TypeError):
                    s = 0.0
                phase_success[phase].append(s)

        if not phase_success:
            continue

        phases = sorted(phase_success.keys())
        means = [np.mean(phase_success[p]) * 100 for p in phases]
        counts = [len(phase_success[p]) for p in phases]
        stds = [np.std(phase_success[p], ddof=1) * 100 if len(phase_success[p]) > 1 else 0 for p in phases]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(phases))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color="#3498db", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p}\n(n={c})" for p, c in zip(phases, counts)], fontsize=11)
        ax.set_ylabel("Success rate (%)", fontsize=14)
        ax.set_title(f"Success by curriculum phase: {csv_path.stem.replace('_metrics', '')}", fontsize=14)
        ax.set_ylim(0, 105)
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{v:.1f}%", ha="center", fontsize=11)
        plt.tight_layout()
        out_path = out_dir / f"success_by_phase_{csv_path.stem.replace('_metrics', '')}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  -> {out_path.name}")


def find_metrics_csvs(csv_dir: Path) -> list[Path]:
    """Find ppo_*_metrics.csv files."""
    if not csv_dir.exists():
        return []
    return sorted(csv_dir.glob("ppo_*_metrics.csv"))


def run_cmd(cmd: list[str], cwd: Path) -> bool:
    """Run command; return True if success."""
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            print(f"[WARN] {cmd[0]} failed: {result.stderr[:200] if result.stderr else 'unknown'}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"[WARN] {cmd[0]} timed out")
        return False
    except Exception as e:
        print(f"[WARN] {cmd[0]} error: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all project plots from training CSVs and optionally eval."
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default=None,
        help=f"Directory with ppo_*_metrics.csv (default: {DEFAULT_CSV_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Also run eval-based scripts (plot_all_winrates, plot_eval_metrics). Slower, requires checkpoints.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Moving average window for learning curves (default: 100).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Episodes for eval scripts when --eval (default: 50).",
    )
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir).expanduser().resolve() if args.csv_dir else DEFAULT_CSV_DIR
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else DEFAULT_OUT_DIR
    table_dir = DEFAULT_TABLE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    print(f"CSV dir: {csv_dir}")
    print(f"Output dir: {out_dir}")
    print()

    # 1. Find training CSVs
    csvs = find_metrics_csvs(csv_dir)
    if not csvs:
        print(f"[WARN] No ppo_*_metrics.csv found in {csv_dir}")
        print("  Skipping training-based plots. Run training first or set --csv-dir.")
    else:
        print(f"Found {len(csvs)} training CSV(s):")
        for c in csvs:
            print(f"  - {c.name}")

    # 2. Success & collisions learning curves (plot_metrics.py)
    if csvs:
        print("\n--- Success & Collisions learning curves ---")
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "plot_metrics.py"),
            *[str(c) for c in csvs],
            "--out-dir", str(out_dir),
            "--window", str(args.window),
        ]
        if run_cmd(cmd, SCRIPT_DIR):
            print("  -> success_vs_episode.png, collisions_vs_episode.png")

    # 3. Offense metrics (plot_offense_metrics.py)
    if csvs:
        print("\n--- Offense / coordination metrics ---")
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "plot_offense_metrics.py"),
            *[str(c) for c in csvs],
            "--out-dir", str(out_dir),
            "--table", str(table_dir / "offense_table.csv"),
            "--window", str(args.window),
        ]
        if run_cmd(cmd, SCRIPT_DIR):
            print("  -> offense_metrics_*.png, offense_*.png, offense_table.csv")

    # 3b. Success by curriculum phase
    if csvs:
        print("\n--- Success by phase (OP1, OP2, OP3, ...) ---")
        plot_success_by_phase(csvs, out_dir)

    # 4. Eval-based plots (optional)
    if args.eval:
        print("\n--- Win rates (2v2, 3v3, 4v4) ---")
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "plot_all_winrates.py"),
            "--episodes", str(args.episodes),
            "--out", str(out_dir / "all_winrates.png"),
        ]
        if run_cmd(cmd, SCRIPT_DIR):
            print("  -> all_winrates.png")

        print("\n--- Eval metrics (success, coordination, robustness, etc.) ---")
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "plot_eval_metrics.py"),
            "--episodes", str(args.episodes),
            "--out", str(out_dir / "eval_metrics.png"),
            "--table-out", str(table_dir / "eval_table.csv"),
        ]
        if run_cmd(cmd, SCRIPT_DIR):
            print("  -> eval_metrics_*.png, eval_table.csv")

    print(f"\nDone. Plots in: {out_dir}  |  Tables in: {table_dir}")
    print("\nPlot summary (figures/):")
    for f in sorted(out_dir.glob("*.png")):
        print(f"  - {f.name}")
    for f in sorted(out_dir.glob("*.pdf")):
        print(f"  - {f.name}")
    print("Tables (csv/):")
    for f in sorted(table_dir.glob("*.csv")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
