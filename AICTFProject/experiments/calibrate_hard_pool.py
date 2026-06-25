#!/usr/bin/env python3
"""Calibrate OP8/OP9/OP10 difficulty against a V6I7 300k checkpoint.

Runs forced-latent evaluation (z0..z3 × 2 maps × OP8/OP9/OP10) and prints
a win-rate matrix.  Does NOT train — purely evaluation.

Calibration targets (from V6I7 design spec):
  - At least 2 cells with 35–60% WR  (challenging but learnable)
  - At least 2 cells with 60–80% WR  (tractable)
  - No more than 1 cell > 90% WR     (not trivially solved)
  - No cell at 0% WR across all 4 latents for a single opponent (not impossible)

If targets are not met, recommended tuning knobs:
  - Too easy (>90% WR cells): increase OP8 speed, tighten OP9 orbit radius,
    or reduce OP10 escort_interpose distance.
  - Too hard (0% WR): lower c_prob for the difficult opponent, raise
    role_switch_prob to reduce determinism.

Usage
-----

Evaluate V6I7 300k checkpoint (update --checkpoint to your actual path):

    python experiments/calibrate_hard_pool.py \\
        --checkpoint checkpoints/v6i7/ckpt_300000.zip \\
        --episodes 200 \\
        --device cuda

Quick sanity check with 50 episodes per cell:

    python experiments/calibrate_hard_pool.py \\
        --checkpoint checkpoints/v6i7/ckpt_300000.zip \\
        --episodes 50

Output: calibrate_hard_pool_<timestamp>.csv (one row per opponent × latent)
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OPPONENTS = ("OP8", "OP9", "OP10")
LATENTS = (0, 1, 2, 3)
MAPS = ("map_b", "map_b_split_lane_v2")

# WR bands for calibration gate
BAND_LOW_LO, BAND_LOW_HI = 0.35, 0.60   # must have >=2 cells here
BAND_MID_LO, BAND_MID_HI = 0.60, 0.80   # must have >=2 cells here
BAND_TRIVIAL = 0.90                        # must have <=1 cell here
BAND_IMPOSSIBLE = 0.05                     # per-opponent: must not have all 4 latents here


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate hard opponent pool against a V6I7 checkpoint")
    p.add_argument("--checkpoint", required=True, help="Path to .zip checkpoint file")
    p.add_argument("--episodes", type=int, default=200, help="Episodes per (opponent, latent, map) cell")
    p.add_argument("--device", default="cpu", help="torch device string")
    p.add_argument("--out-dir", default=None, help="Output directory (default: experiments/calibration_runs/)")
    p.add_argument("--stochastic", action="store_true", help="Use stochastic rather than deterministic policy")
    p.add_argument("--opponents", nargs="+", default=list(OPPONENTS), help="Subset of opponents to evaluate")
    p.add_argument("--maps", nargs="+", default=list(MAPS), help="Subset of maps to evaluate")
    return p.parse_args()


def _check_calibration_targets(wr_matrix: Dict[Tuple[str, int], float]) -> None:
    """Print a summary and flag any calibration failures."""
    vals = list(wr_matrix.values())

    low_band = [v for v in vals if BAND_LOW_LO <= v < BAND_LOW_HI]
    mid_band = [v for v in vals if BAND_MID_LO <= v < BAND_MID_HI]
    trivial = [v for v in vals if v >= BAND_TRIVIAL]

    print("\n--- Calibration Gate ---")
    print(f"  Cells 35–60% WR : {len(low_band)}  (need >= 2)")
    print(f"  Cells 60–80% WR : {len(mid_band)}  (need >= 2)")
    print(f"  Cells >90% WR   : {len(trivial)}  (need <= 1)")

    ok = True
    if len(low_band) < 2:
        print("  FAIL: too few challenging cells (35–60%). Consider raising OP difficulty.")
        ok = False
    if len(mid_band) < 2:
        print("  FAIL: too few tractable cells (60–80%). Consider lowering OP difficulty.")
        ok = False
    if len(trivial) > 1:
        print("  FAIL: too many trivial cells (>90%). Consider raising OP difficulty.")
        ok = False

    # Per-opponent: check no opponent is impossible (all latents <=5%)
    from collections import defaultdict
    by_opp: Dict[str, List[float]] = defaultdict(list)
    for (opp, _z), wr in wr_matrix.items():
        by_opp[opp].append(wr)
    for opp, wrs in by_opp.items():
        if all(w <= BAND_IMPOSSIBLE for w in wrs):
            print(f"  FAIL: {opp} is impossible — all latents WR <= {BAND_IMPOSSIBLE:.0%}.")
            ok = False

    print(f"\n  Overall: {'PASS' if ok else 'FAIL'}")


def main() -> None:
    args = _parse_args()

    try:
        from plot.eval_rollout import run_eval_episodes, count_wld  # type: ignore[import]
    except ImportError as exc:
        print(f"ERROR: could not import eval infrastructure: {exc}")
        print("Run this script from the project root with the venv active.")
        sys.exit(1)

    out_dir = args.out_dir or os.path.join(SCRIPT_DIR, "calibration_runs")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(out_dir, f"calibrate_hard_pool_{timestamp}.csv")

    rows: List[Dict[str, Any]] = []
    wr_matrix: Dict[Tuple[str, int], float] = {}

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Episodes   : {args.episodes} per cell")
    print(f"Device     : {args.device}")
    print(f"Opponents  : {args.opponents}")
    print(f"Maps       : {args.maps}")
    print()

    for opponent in args.opponents:
        for latent_z in LATENTS:
            wr_across_maps: List[float] = []
            for map_name in args.maps:
                try:
                    episodes = run_eval_episodes(
                        model_path=args.checkpoint,
                        n_episodes=args.episodes,
                        device=args.device,
                        opponent=opponent,
                        map_layout=map_name,
                        forced_latent_z=latent_z,
                        deterministic=not args.stochastic,
                    )
                    wld = count_wld(episodes)
                    total = max(1, wld.get("win", 0) + wld.get("loss", 0) + wld.get("draw", 0))
                    wr = wld.get("win", 0) / total
                except Exception as exc:  # noqa: BLE001
                    print(f"  ERROR evaluating {opponent} z={latent_z} {map_name}: {exc}")
                    wr = float("nan")
                    wld = {}

                wr_across_maps.append(wr)
                rows.append({
                    "opponent": opponent,
                    "latent_z": latent_z,
                    "map": map_name,
                    "win_rate": f"{wr:.4f}",
                    "wins": wld.get("win", 0),
                    "losses": wld.get("loss", 0),
                    "draws": wld.get("draw", 0),
                    "episodes": args.episodes,
                })
                print(f"  {opponent} z={latent_z} {map_name}: WR={wr:.1%}")

            valid = [w for w in wr_across_maps if not (w != w)]  # filter nan
            mean_wr = sum(valid) / len(valid) if valid else float("nan")
            wr_matrix[(opponent, latent_z)] = mean_wr

    # Write CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["opponent", "latent_z", "map", "win_rate", "wins", "losses", "draws", "episodes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {out_csv}")

    # Print WR table
    print("\n--- Win Rate Matrix (mean across maps) ---")
    header = f"{'':12s}" + "".join(f"  z={z}" for z in LATENTS)
    print(header)
    for opponent in args.opponents:
        row_str = f"{opponent:<12s}"
        for latent_z in LATENTS:
            wr = wr_matrix.get((opponent, latent_z), float("nan"))
            row_str += f"  {wr:5.1%}" if wr == wr else "    nan"
        print(row_str)

    _check_calibration_targets(wr_matrix)


if __name__ == "__main__":
    main()
