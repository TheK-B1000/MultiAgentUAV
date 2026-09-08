#!/usr/bin/env python3
"""Aggregate Phase-2 multi-seed ΔG and CI95 across Stage-1 LRO rounds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v6i26_lro_core import write_json  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--round-dirs",
        nargs="+",
        required=True,
        help="Stage-1 output dirs each containing stage1_round_log.json or acceptance.json",
    )
    p.add_argument(
        "--phase2-seed1",
        default=None,
        help="Optional phase2_seed1_confirm.json (large-eval seed 1)",
    )
    p.add_argument(
        "--output",
        default="artifacts/v6i26_phase2_multi_seed_summary.json",
    )
    args = p.parse_args()

    deltas: list[float] = []
    rows: list[dict] = []
    for d in args.round_dirs:
        path = Path(d)
        acc = path / "acceptance.json"
        log = path / "stage1_round_log.json"
        data = {}
        log_data = {}
        if acc.is_file():
            data = json.loads(acc.read_text(encoding="utf-8"))
            if log.is_file():
                log_data = json.loads(log.read_text(encoding="utf-8"))
        elif log.is_file():
            log_data = json.loads(log.read_text(encoding="utf-8"))
            data = log_data.get("acceptance") or log_data
        if "delta_G_available" not in data and "delta_G" not in data:
            print(f"WARN: no delta_G in {path}")
            continue
        dg = float(data.get("delta_G_available", data.get("delta_G")))
        deltas.append(dg)
        behavior = (
            data.get("behavior_distinctness")
            or log_data.get("behavior_distinctness")
            or {}
        )
        branch_behavior_nonredundant = (
            behavior.get("branch_behavior_nonredundant")
            if isinstance(behavior, dict)
            else None
        )
        accepted = bool(data.get("accepted") or data.get("verdict") == "ACCEPT")
        rows.append(
            {
                "dir": str(path),
                "G_before": data.get("G_before"),
                "G_after": data.get("G_after"),
                "delta_G": dg,
                "accepted": accepted,
                "branch_behavior_nonredundant": branch_behavior_nonredundant,
                "branch_nearest_behavior_distance": (
                    behavior.get("branch_nearest_behavior_distance")
                    if isinstance(behavior, dict)
                    else None
                ),
                "strategy_accepted": bool(
                    accepted and branch_behavior_nonredundant is True
                ),
            }
        )

    arr = np.asarray(deltas, dtype=np.float64)
    summary: dict = {
        "n_seeds": int(arr.size),
        "deltas": deltas,
        "rows": rows,
        "mean_delta_G": float(arr.mean()) if arr.size else None,
        "std_delta_G": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }
    if arr.size >= 2:
        # Student-t style normal approx for small n (n=3): mean ± 1.96*se
        se = float(arr.std(ddof=1) / np.sqrt(arr.size))
        summary["CI95_low_normal"] = float(arr.mean() - 1.96 * se)
        summary["CI95_high_normal"] = float(arr.mean() + 1.96 * se)
        summary["CI95_gt_0_normal"] = bool(summary["CI95_low_normal"] > 0.0)
        # Also: all seeds positive?
        summary["all_seeds_delta_G_gt_0"] = bool((arr > 0).all())
    if args.phase2_seed1:
        p2 = Path(args.phase2_seed1)
        if p2.is_file():
            summary["seed1_large_eval"] = json.loads(p2.read_text(encoding="utf-8"))

    p2_pass = False
    if summary.get("seed1_large_eval", {}).get("CI95_delta_G_gt_0") and summary.get(
        "all_seeds_delta_G_gt_0"
    ):
        p2_pass = True
    elif summary.get("CI95_gt_0_normal") and summary.get("all_seeds_delta_G_gt_0"):
        p2_pass = True
    summary["phase2_verdict"] = "PHASE2_PASS" if p2_pass else "PHASE2_INCOMPLETE_OR_FAIL"
    summary["all_seed_rows_strategy_accepted"] = bool(
        rows and all(bool(row.get("strategy_accepted")) for row in rows)
    )
    seed1_strategy_ok = True
    if "seed1_large_eval" in summary:
        seed1 = summary["seed1_large_eval"]
        if "phase2_strategy_verdict" in seed1:
            seed1_strategy_ok = seed1.get("phase2_strategy_verdict") == "PHASE2_STRATEGY_PASS"
        else:
            distinct = seed1.get("strategy_distinctness") or {}
            seed1_strategy_ok = bool(
                seed1.get("CI95_delta_G_gt_0")
                and distinct.get("branch_behavior_nonredundant")
            )
    summary["phase2_strategy_verdict"] = (
        "PHASE2_STRATEGY_PASS"
        if p2_pass and summary["all_seed_rows_strategy_accepted"] and seed1_strategy_ok
        else "PHASE2_STRATEGY_INCOMPLETE_OR_FAIL"
    )
    write_json(Path(args.output), summary)
    print(json.dumps(summary, indent=2))
    return 0 if p2_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
