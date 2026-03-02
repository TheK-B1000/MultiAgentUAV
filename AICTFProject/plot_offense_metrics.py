#!/usr/bin/env python3
"""
Plot offense/coordination metrics from training CSVs (ppo_*_metrics.csv).

Metrics:
  - Time to first score: Lower → faster offense
  - Mean inter-robot distance: Higher → more spread; lower → tighter formation
  - Zone coverage: Higher → better spatial coverage
  - Win margin (blue_score − red_score): Captures dominance

Usage:
  python plot_offense_metrics.py checkpoints_sb3/ppo_league_4v4_metrics.csv
  python plot_offense_metrics.py csv1.csv csv2.csv --out-dir figures --table offense_table.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_csv(path: Path) -> dict[str, list]:
    """Load CSV into column name -> list of values (float or nan)."""
    cols: dict[str, list] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k not in cols:
                    cols[k] = []
                if v is None or v == "":
                    cols[k].append(np.nan)
                else:
                    try:
                        cols[k].append(float(v))
                    except ValueError:
                        cols[k].append(np.nan)
    return cols


def _safe_mean(vals: list[float], omit_nan: bool = True) -> float:
    a = np.array(vals, dtype=np.float64)
    if omit_nan:
        a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan
    return float(np.mean(a))


def _safe_std(vals: list[float], omit_nan: bool = True) -> float:
    a = np.array(vals, dtype=np.float64)
    if omit_nan:
        a = a[np.isfinite(a)]
    if len(a) < 2:
        return 0.0
    return float(np.std(a, ddof=1))


def compute_metrics(cols: dict[str, list]) -> dict[str, float]:
    """Compute per-run aggregates."""
    out: dict[str, float] = {}

    # Win margin = blue_score - red_score
    if "blue_score" in cols and "red_score" in cols:
        margins = [
            float(b) - float(r)
            for b, r in zip(cols["blue_score"], cols["red_score"])
            if np.isfinite(float(b) if isinstance(b, (int, float)) else np.nan)
            and np.isfinite(float(r) if isinstance(r, (int, float)) else np.nan)
        ]
        out["win_margin_mean"] = _safe_mean(margins)
        out["win_margin_std"] = _safe_std(margins)
    else:
        out["win_margin_mean"] = np.nan
        out["win_margin_std"] = 0.0

    # Time to first score (only wins have it typically)
    if "time_to_first_score" in cols:
        vals = [float(x) for x in cols["time_to_first_score"] if np.isfinite(float(x) if isinstance(x, (int, float)) else np.nan)]
        out["time_to_first_score_mean"] = _safe_mean(vals)
        out["time_to_first_score_std"] = _safe_std(vals)
    else:
        out["time_to_first_score_mean"] = np.nan
        out["time_to_first_score_std"] = 0.0

    # Mean inter-robot distance
    if "mean_inter_robot_dist" in cols:
        vals = [float(x) for x in cols["mean_inter_robot_dist"] if np.isfinite(float(x) if isinstance(x, (int, float)) else np.nan)]
        out["mean_inter_robot_dist_mean"] = _safe_mean(vals)
        out["mean_inter_robot_dist_std"] = _safe_std(vals)
    else:
        out["mean_inter_robot_dist_mean"] = np.nan
        out["mean_inter_robot_dist_std"] = 0.0

    # Zone coverage
    if "zone_coverage" in cols:
        vals = [float(x) for x in cols["zone_coverage"] if np.isfinite(float(x) if isinstance(x, (int, float)) else np.nan)]
        out["zone_coverage_mean"] = _safe_mean(vals) * 100.0
        out["zone_coverage_std"] = _safe_std(vals) * 100.0
    else:
        out["zone_coverage_mean"] = np.nan
        out["zone_coverage_std"] = 0.0

    # Success rate for context
    if "success" in cols:
        vals = [float(x) for x in cols["success"] if np.isfinite(float(x) if isinstance(x, (int, float)) else np.nan)]
        out["success_rate"] = _safe_mean(vals) * 100.0 if vals else np.nan
    else:
        out["success_rate"] = np.nan

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot offense metrics (time_to_first_score, inter_robot_dist, zone_coverage, win_margin) from training CSVs."
    )
    parser.add_argument("csv", nargs="+", help="Training metrics CSV(s), e.g. checkpoints_sb3/ppo_league_4v4_metrics.csv")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory for plots")
    parser.add_argument("--table", type=str, default=None, help="Write summary table to this CSV")
    parser.add_argument("--window", type=int, default=100, help="Moving average window for learning curves")
    parser.add_argument("--no-plot", action="store_true", help="Only print table, skip plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    results: list[tuple[str, dict]] = []

    for csv_path in args.csv:
        p = Path(csv_path).expanduser().resolve()
        if not p.exists():
            print(f"[WARN] Not found: {p}")
            continue
        label = p.stem.replace("_metrics", "")
        cols = _load_csv(p)
        metrics = compute_metrics(cols)
        results.append((label, metrics))
        print(f"\n--- {label} ---")
        print(f"  time_to_first_score:  {metrics['time_to_first_score_mean']:.2f} ± {metrics['time_to_first_score_std']:.2f} (lower = faster offense)")
        print(f"  mean_inter_robot_dist: {metrics['mean_inter_robot_dist_mean']:.3f} ± {metrics['mean_inter_robot_dist_std']:.3f} (higher = more spread)")
        print(f"  zone_coverage:        {metrics['zone_coverage_mean']:.2f} ± {metrics['zone_coverage_std']:.2f} % (higher = better coverage)")
        print(f"  win_margin:           {metrics['win_margin_mean']:.2f} ± {metrics['win_margin_std']:.2f} (blue - red)")
        print(f"  success_rate:         {metrics['success_rate']:.1f} %")

    # Write table
    if args.table and results:
        table_path = Path(args.table).expanduser().resolve()
        table_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "run", "time_to_first_score_mean", "time_to_first_score_std",
            "mean_inter_robot_dist_mean", "mean_inter_robot_dist_std",
            "zone_coverage_mean", "zone_coverage_std",
            "win_margin_mean", "win_margin_std", "success_rate",
        ]
        with table_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for label, m in results:
                w.writerow({
                    "run": label,
                    "time_to_first_score_mean": m["time_to_first_score_mean"],
                    "time_to_first_score_std": m["time_to_first_score_std"],
                    "mean_inter_robot_dist_mean": m["mean_inter_robot_dist_mean"],
                    "mean_inter_robot_dist_std": m["mean_inter_robot_dist_std"],
                    "zone_coverage_mean": m["zone_coverage_mean"],
                    "zone_coverage_std": m["zone_coverage_std"],
                    "win_margin_mean": m["win_margin_mean"],
                    "win_margin_std": m["win_margin_std"],
                    "success_rate": m["success_rate"],
                })
        print(f"\nTable saved: {table_path}")

    # Plots (one CSV => learning curves; multiple => bar comparison)
    if args.no_plot:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    plt.rc("font", size=12)

    if len(results) == 1:
        # Single CSV: learning curves (episode vs metric)
        label, _ = results[0]
        p = Path(args.csv[0]).expanduser().resolve()
        cols = _load_csv(p)
        ep_ids = cols.get("episode_id", [])
        if not ep_ids or len(ep_ids) < 2:
            print("Not enough episodes for learning curves")
            return

        ep = np.array([float(x) for x in ep_ids if np.isfinite(float(x) if isinstance(x, (int, float)) else np.nan)])
        n = len(ep)
        w = min(args.window, n // 2 or 1)

        def smooth(vals: list) -> tuple[np.ndarray, np.ndarray]:
            a = np.array([float(x) if np.isfinite(float(x) if isinstance(x, (int, float)) else np.nan) else np.nan for x in vals])
            valid = np.isfinite(a)
            out = np.full_like(a, np.nan)
            for i in range(w, n - w):
                window = a[i - w : i + w + 1]
                win = window[np.isfinite(window)]
                if len(win) > 0:
                    out[i] = np.mean(win)
            return ep, out

        out_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        # Time to first score
        if "time_to_first_score" in cols:
            x, y = smooth(cols["time_to_first_score"])
            axes[0, 0].plot(x, y, "b-", linewidth=1.5)
            axes[0, 0].set_ylabel("Time to first score")
            axes[0, 0].set_title("Faster offense (lower = better)")
            axes[0, 0].set_xlabel("Episode")

        # Mean inter-robot dist
        if "mean_inter_robot_dist" in cols:
            x, y = smooth(cols["mean_inter_robot_dist"])
            axes[0, 1].plot(x, y, "g-", linewidth=1.5)
            axes[0, 1].set_ylabel("Mean inter-robot distance")
            axes[0, 1].set_title("Formation spread (higher = more spread)")
            axes[0, 1].set_xlabel("Episode")

        # Zone coverage
        if "zone_coverage" in cols:
            x, y = smooth(cols["zone_coverage"])
            axes[1, 0].plot(x, np.array(y) * 100, "m-", linewidth=1.5)
            axes[1, 0].set_ylabel("Zone coverage (%)")
            axes[1, 0].set_title("Spatial coverage (higher = better)")
            axes[1, 0].set_xlabel("Episode")

        # Win margin
        if "blue_score" in cols and "red_score" in cols:
            margins = [float(b) - float(r) for b, r in zip(cols["blue_score"], cols["red_score"])]
            x, y = smooth(margins)
            axes[1, 1].plot(x, y, "c-", linewidth=1.5)
            axes[1, 1].set_ylabel("Win margin (blue - red)")
            axes[1, 1].set_title("Dominance")
            axes[1, 1].axhline(0, color="gray", linestyle="--")
            axes[1, 1].set_xlabel("Episode")

        plt.suptitle(f"Offense metrics: {label}", fontsize=14)
        plt.tight_layout()
        out_path = out_dir / f"offense_metrics_{label}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

    else:
        # Multiple CSVs: bar charts comparing runs
        out_dir.mkdir(parents=True, exist_ok=True)
        labels = [r[0] for r in results]
        x = np.arange(len(labels))
        width = 0.6

        metrics_to_plot = [
            ("time_to_first_score_mean", "time_to_first_score_std", "Time to first score (lower = faster offense)", "time_to_first_score"),
            ("mean_inter_robot_dist_mean", "mean_inter_robot_dist_std", "Mean inter-robot distance (higher = more spread)", "inter_robot_dist"),
            ("zone_coverage_mean", "zone_coverage_std", "Zone coverage (%)", "zone_coverage"),
            ("win_margin_mean", "win_margin_std", "Win margin (blue - red)", "win_margin"),
        ]
        for mean_key, std_key, ylabel, suffix in metrics_to_plot:
            means = [m.get(mean_key, np.nan) for _, m in results]
            stds = [m.get(std_key, 0) for _, m in results]
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#f39c12", "#1abc9c", "#95a5a6"]
            bar_colors = [colors[i % len(colors)] for i in range(len(labels))]
            bars = ax.bar(x, means, width, yerr=stds, capsize=5, color=bar_colors, edgecolor="black")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=10, rotation=30 if len(labels) > 4 else 0, ha="right")
            ax.set_ylabel(ylabel, fontsize=14)
            ax.set_title(ylabel, fontsize=16)
            for i, (bar, v) in enumerate(zip(bars, means)):
                if np.isfinite(v):
                    y_off = stds[i] if i < len(stds) else 0.1
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_off, f"{v:.2f}", ha="center", fontsize=10)
            plt.tight_layout()
            out_path = out_dir / f"offense_{suffix}.png"
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
