"""Check whether a latent PPO metrics CSV is learning as expected.

Example:
    python tools/diagnose_run.py checkpoints/2v2/my_run_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def _last_mean(rows: list[dict[str, str]], key: str, window: int) -> float:
    vals = [_float(row, key) for row in rows[-window:]]
    return statistics.fmean(vals) if vals else 0.0


def _last_mean_optional(rows: list[dict[str, str]], key: str, window: int) -> float | None:
    vals: list[float] = []
    for row in rows[-window:]:
        value = row.get(key, "")
        if value == "":
            continue
        vals.append(float(value))
    return statistics.fmean(vals) if vals else None


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0 if numerator <= 0.0 else math.inf
    return numerator / denominator


def _infer_latent_k(rows: list[dict[str, str]]) -> int:
    occupancy_fields = [
        key
        for key in rows[0].keys()
        if key.startswith("strategy_occupancy_")
    ]
    return max(1, len(occupancy_fields))


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose latent PPO training metrics.")
    parser.add_argument("metrics_csv", type=Path)
    parser.add_argument("--window", type=int, default=10, help="Number of final updates to average.")
    parser.add_argument("--latent-k", type=int, default=None, help="Override latent strategy count.")
    parser.add_argument("--entropy-frac", type=float, default=0.95)
    parser.add_argument("--max-clip-fraction", type=float, default=0.6)
    parser.add_argument("--min-explained-variance", type=float, default=0.4)
    parser.add_argument("--min-win-rate-gain", type=float, default=0.1)
    parser.add_argument(
        "--max-z-switch-adv-std-ratio",
        type=float,
        default=1.5,
        help=(
            "Maximum final-window ratio of rollout_adv_std_at_z_switch "
            "to rollout_adv_std_not_z_switch."
        ),
    )
    args = parser.parse_args()

    with args.metrics_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"FAIL: no metrics rows in {args.metrics_csv}", file=sys.stderr)
        return 2

    window = max(1, min(int(args.window), len(rows)))
    latent_k = int(args.latent_k or _infer_latent_k(rows))
    entropy_limit = float(args.entropy_frac) * math.log(max(2, latent_k))

    first_rollout_wr = _float(rows[0], "rollout_win_rate")
    last_rollout_wr = _float(rows[-1], "rollout_win_rate")
    last_strategy_entropy = _last_mean(rows, "strategy_entropy", window)
    last_persist_loss = _last_mean(rows, "strategy_persist_loss", window)
    last_resample_count = _last_mean_optional(rows, "strategy_resample_count", window)
    last_rollout_episodes = _last_mean_optional(rows, "rollout_episodes", window)
    last_clip_fraction = _float(rows[-1], "clip_fraction")
    last_explained_variance = _float(rows[-1], "explained_variance")
    last_z_switch_adv_std = _last_mean_optional(rows, "rollout_adv_std_at_z_switch", window)
    last_not_z_switch_adv_std = _last_mean_optional(rows, "rollout_adv_std_not_z_switch", window)
    rollout_wr_gain = last_rollout_wr - first_rollout_wr

    checks = [
        (
            "strategy_entropy",
            last_strategy_entropy < entropy_limit,
            f"{last_strategy_entropy:.4f} < {entropy_limit:.4f}",
        ),
        (
            "clip_fraction",
            last_clip_fraction < float(args.max_clip_fraction),
            f"{last_clip_fraction:.4f} < {float(args.max_clip_fraction):.4f}",
        ),
        (
            "explained_variance",
            last_explained_variance > float(args.min_explained_variance),
            f"{last_explained_variance:.4f} > {float(args.min_explained_variance):.4f}",
        ),
        (
            "rollout_win_rate_gain",
            rollout_wr_gain > float(args.min_win_rate_gain),
            f"{rollout_wr_gain:.4f} > {float(args.min_win_rate_gain):.4f}",
        ),
    ]
    skipped_checks: list[tuple[str, str]] = []
    if (
        last_resample_count is not None
        and last_rollout_episodes is not None
        and last_resample_count <= last_rollout_episodes * 1.05 + 1.0
    ):
        skipped_checks.append(
            (
                "strategy_persist_loss",
                (
                    "no meaningful non-initial z refreshes "
                    f"({last_resample_count:.1f} resamples / {last_rollout_episodes:.1f} episodes)"
                ),
            )
        )
    else:
        checks.append(
            (
                "strategy_persist_loss",
                last_persist_loss > 0.0,
                f"{last_persist_loss:.4f} > 0",
            )
        )
    if last_z_switch_adv_std is None or last_not_z_switch_adv_std is None:
        skipped_checks.append(
            (
                "z_switch_adv_std_ratio",
                "missing rollout_adv_std_at_z_switch/rollout_adv_std_not_z_switch",
            )
        )
    else:
        z_switch_adv_std_ratio = _safe_ratio(last_z_switch_adv_std, last_not_z_switch_adv_std)
        checks.append(
            (
                "z_switch_adv_std_ratio",
                z_switch_adv_std_ratio <= float(args.max_z_switch_adv_std_ratio),
                (
                    f"{z_switch_adv_std_ratio:.4f} <= {float(args.max_z_switch_adv_std_ratio):.4f} "
                    f"({last_z_switch_adv_std:.4f} / {last_not_z_switch_adv_std:.4f})"
                ),
            )
        )

    print(f"metrics: {args.metrics_csv}")
    print(f"rows: {len(rows)} | final-window: {window} | latent_k: {latent_k}")
    print(f"first rollout WR: {first_rollout_wr:.4f}")
    print(f"last rollout WR:  {last_rollout_wr:.4f}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {name}: {detail}")
    for name, detail in skipped_checks:
        print(f"SKIP: {name}: {detail}")

    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
