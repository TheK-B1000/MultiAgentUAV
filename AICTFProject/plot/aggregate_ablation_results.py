"""
Aggregates the leave-one-out ablation study's per-episode metrics CSVs
(rl/run_ablations.py output: ppo_ablate_<arm>[_seed<seed>]_<NvN>_metrics.csv)
into a per-seed and per-arm (mean +/- std across seeds) summary -- directly
answers Reviewer 3's ablation request. Safe to run against a run that is
still training (tolerates a partially-flushed last row), so it's useful while
Stage 2 is still in progress.

Usage:
    python plot/aggregate_ablation_results.py --checkpoint-dir checkpoints_sb3/2v2
    python plot/aggregate_ablation_results.py --checkpoint-dir checkpoints_sb3/2v2 --tail-window 500 --out csv/ablation_summary_2v2.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics
from dataclasses import dataclass
from typing import Optional

_RUN_TAG_RE = re.compile(r"^ppo_ablate_(?P<arm>.+?)(?:_seed(?P<seed>\d+))?_(?P<n>\d+)v\d+_metrics\.csv$")


@dataclass
class ArmSeedSummary:
    arm: str
    seed: Optional[int]
    setting: str
    n_episodes: int
    overall_success_rate: float
    tail_success_rate: Optional[float]
    tail_window: int
    reached_op3: bool
    max_opponent_switch_count: int


def parse_run_tag(filename: str) -> Optional[tuple[str, Optional[int], str]]:
    m = _RUN_TAG_RE.match(os.path.basename(filename))
    if not m:
        return None
    arm = m.group("arm")
    seed = int(m.group("seed")) if m.group("seed") else None
    n = m.group("n")
    return arm, seed, f"{n}v{n}"


def _safe_int(s) -> Optional[int]:
    if s is None or s == "":
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def summarize_csv(path: str, *, tail_window: int) -> Optional[ArmSeedSummary]:
    """Summarize one run's per-episode metrics CSV. Rows that fail to parse
    (e.g. a partially-flushed last line from a still-running training process)
    are skipped rather than raising, so this is safe to run mid-training."""
    parsed = parse_run_tag(path)
    if parsed is None:
        return None
    arm, seed, setting = parsed

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            if _safe_int(row.get("episode_id")) is None or _safe_int(row.get("success")) is None:
                continue
            rows.append(row)
    if not rows:
        return None

    successes = [v for v in (_safe_int(r.get("success")) for r in rows) if v is not None]
    n_episodes = len(rows)
    overall_sr = 100.0 * sum(successes) / max(1, len(successes))

    tail_rows = rows[-tail_window:] if tail_window > 0 else rows
    tail_successes = [v for v in (_safe_int(r.get("success")) for r in tail_rows) if v is not None]
    tail_sr = (100.0 * sum(tail_successes) / len(tail_successes)) if tail_successes else None

    reached_op3 = any(str(r.get("phase_name", "")).upper() == "OP3" for r in rows)
    switch_counts = [v for v in (_safe_int(r.get("opponent_switch_count")) for r in rows) if v is not None]
    max_switch = max(switch_counts) if switch_counts else 0

    return ArmSeedSummary(
        arm=arm,
        seed=seed,
        setting=setting,
        n_episodes=n_episodes,
        overall_success_rate=overall_sr,
        tail_success_rate=tail_sr,
        tail_window=min(tail_window, n_episodes) if tail_window > 0 else n_episodes,
        reached_op3=reached_op3,
        max_opponent_switch_count=max_switch,
    )


def aggregate_by_arm(summaries: list[ArmSeedSummary]) -> list[dict]:
    """Group per-seed summaries by arm, computing mean/std across seeds -- the
    3-seed rollup Reviewer 3's ablation request needs."""
    by_arm: dict[str, list[ArmSeedSummary]] = {}
    for s in summaries:
        by_arm.setdefault(s.arm, []).append(s)

    rows = []
    for arm, items in sorted(by_arm.items()):
        tail_values = [it.tail_success_rate for it in items if it.tail_success_rate is not None]
        overall_values = [it.overall_success_rate for it in items]
        rows.append({
            "arm": arm,
            "n_seeds": len(items),
            "seeds": ",".join(str(it.seed) if it.seed is not None else "42(default)" for it in items),
            "episodes_total": sum(it.n_episodes for it in items),
            "overall_success_rate_mean": statistics.fmean(overall_values) if overall_values else float("nan"),
            "overall_success_rate_std": statistics.pstdev(overall_values) if len(overall_values) > 1 else 0.0,
            "tail_success_rate_mean": statistics.fmean(tail_values) if tail_values else float("nan"),
            "tail_success_rate_std": statistics.pstdev(tail_values) if len(tail_values) > 1 else 0.0,
            "all_reached_op3": all(it.reached_op3 for it in items),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing ppo_ablate_*_metrics.csv files")
    parser.add_argument(
        "--tail-window", type=int, default=500,
        help="Episodes from the end of each run treated as 'converged' performance (0 = whole run)",
    )
    parser.add_argument("--out", default=None, help="Optional CSV path to write the per-arm summary table")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.checkpoint_dir, "ppo_ablate_*_metrics.csv")))
    if not files:
        print(f"[aggregate_ablation_results] no ppo_ablate_*_metrics.csv files found under {args.checkpoint_dir}")
        return 1

    summaries: list[ArmSeedSummary] = []
    for f in files:
        s = summarize_csv(f, tail_window=args.tail_window)
        if s is None:
            print(f"[WARN] could not parse/summarize {f}")
            continue
        summaries.append(s)

    if not summaries:
        print("[aggregate_ablation_results] nothing parseable found")
        return 1

    print("--- Per-seed detail ---")
    for s in sorted(summaries, key=lambda s: (s.arm, s.seed if s.seed is not None else -1)):
        seed_label = f"seed{s.seed}" if s.seed is not None else "seed42(default)"
        op3_flag = "yes" if s.reached_op3 else "NO"
        tail_str = f"{s.tail_success_rate:.1f}%" if s.tail_success_rate is not None else "n/a"
        print(
            f"  {s.arm:16s} {seed_label:16s} episodes={s.n_episodes:6d}  "
            f"overall_wr={s.overall_success_rate:5.1f}%  last{s.tail_window}_wr={tail_str}  "
            f"reached_OP3={op3_flag}"
        )

    arm_rows = aggregate_by_arm(summaries)
    print("\n--- Per-arm summary (mean +/- std across seeds) ---")
    for row in arm_rows:
        print(
            f"  {row['arm']:16s} n_seeds={row['n_seeds']}  seeds=[{row['seeds']}]  "
            f"overall_wr={row['overall_success_rate_mean']:.1f}%+/-{row['overall_success_rate_std']:.1f}  "
            f"tail_wr={row['tail_success_rate_mean']:.1f}%+/-{row['tail_success_rate_std']:.1f}  "
            f"all_reached_OP3={row['all_reached_op3']}"
        )

    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(arm_rows[0].keys()))
            writer.writeheader()
            writer.writerows(arm_rows)
        print(f"\n[aggregate_ablation_results] wrote per-arm summary -> {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
