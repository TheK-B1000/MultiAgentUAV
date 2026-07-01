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


# Episode list keyed by (opponent, latent_z, map_name).
CellEpisodes = Dict[Tuple[str, int, str], List[Dict[str, Any]]]


def _make_env(checkpoint: str, map_name: str, device: str, seed: int) -> Any:
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig  # type: ignore[import]
    from rl.custom_ppo.inference import read_custom_ppo_metadata  # type: ignore[import]

    meta = read_custom_ppo_metadata(checkpoint)
    agents = int(meta.get("n_blue", 2))
    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=agents,
        max_red_agents=agents,
        map_layout=map_name,
        max_decision_steps=400,
        aquaticus_profile=True,
        rules_profile="OURS",
        device=device,
        seed=seed,
    )
    return GPUCTFVecEnv(cfg)


def _cell_seed(base_seed: int, opp_idx: int, map_idx: int) -> int:
    return base_seed + 1000 * opp_idx + 100 * map_idx


def run_forced_z_cells(
    checkpoint: str,
    opponents: List[str],
    latents: Tuple[int, ...],
    maps: List[str],
    n_episodes: int,
    device: str,
    deterministic: bool = True,
    base_seed: int = 42,
    collect_behavior_mean: bool = False,
    progress_every: int = 0,
) -> CellEpisodes:
    """Run forced-z eval; return {(opponent, latent_z, map_name): [episode_dicts]}.

    All latents on the same (opponent, map) cell share the same env seed so the
    initial-state distribution is matched across z values.

    Delegates to :mod:`experiments.forced_z_eval.runner` (one policy load per
    opponent×map block).
    """
    from experiments.forced_z_eval.protocol import ForcedZProtocol
    from experiments.forced_z_eval.runner import run_forced_z_episodes

    protocol = ForcedZProtocol(
        checkpoint=checkpoint,
        opponents=tuple(opponents),
        maps=tuple(maps),
        latents=tuple(latents),
        episodes_per_cell=int(n_episodes),
        base_seed=int(base_seed),
        deterministic_actions=bool(deterministic),
        device=str(device),
        collect_behavior_mean=bool(collect_behavior_mean),
        progress_every=int(progress_every),
    )
    return run_forced_z_episodes(protocol)


def _wr(eps: List[Dict[str, Any]]) -> float:
    if not eps:
        return float("nan")
    return sum(int(e.get("success", 0)) for e in eps) / len(eps)


def _mean_margin(eps: List[Dict[str, Any]]) -> float:
    if not eps:
        return float("nan")
    return sum(int(e.get("win_margin", 0)) for e in eps) / len(eps)


def cells_to_mean_wr_matrix(
    cells: CellEpisodes,
    opponents: List[str],
    latents: Tuple[int, ...],
    maps: List[str],
) -> Dict[Tuple[str, int], float]:
    """Average WR across maps → {(opponent, latent_z): mean_wr}."""
    matrix: Dict[Tuple[str, int], float] = {}
    for opponent in opponents:
        for z in latents:
            vals = [_wr(cells[(opponent, z, m)]) for m in maps if (opponent, z, m) in cells]
            valid = [v for v in vals if v == v]
            matrix[(opponent, z)] = sum(valid) / len(valid) if valid else float("nan")
    return matrix


def main() -> None:
    args = _parse_args()

    try:
        import plot.eval_rollout  # noqa: F401  # verify import before starting
    except ImportError as exc:
        print(f"ERROR: could not import eval infrastructure: {exc}")
        print("Run this script from the project root with the venv active.")
        sys.exit(1)

    out_dir = args.out_dir or os.path.join(SCRIPT_DIR, "calibration_runs")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(out_dir, f"calibrate_hard_pool_{timestamp}.csv")

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Episodes   : {args.episodes} per cell")
    print(f"Device     : {args.device}")
    print(f"Opponents  : {args.opponents}")
    print(f"Maps       : {args.maps}")
    print()

    cells = run_forced_z_cells(
        checkpoint=args.checkpoint,
        opponents=args.opponents,
        latents=tuple(LATENTS),
        maps=args.maps,
        n_episodes=args.episodes,
        device=args.device,
        deterministic=not args.stochastic,
    )
    wr_matrix = cells_to_mean_wr_matrix(cells, args.opponents, tuple(LATENTS), args.maps)

    # Write CSV (one row per opponent × latent × map)
    rows: List[Dict[str, Any]] = [
        {
            "opponent": opp, "latent_z": z, "map": m,
            "win_rate": f"{_wr(eps):.4f}",
            "mean_margin": f"{_mean_margin(eps):.4f}",
            "episodes": len(eps),
        }
        for (opp, z, m), eps in sorted(cells.items())
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["opponent", "latent_z", "map", "win_rate", "mean_margin", "episodes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {out_csv}")

    # Print WR table
    print("\n--- Win Rate Matrix (mean across maps) ---")
    header = f"{'':12s}" + "".join(f"  z={z}" for z in LATENTS)
    print(header)
    for opponent in args.opponents:
        row_str = f"{opponent:<12s}"
        for z in LATENTS:
            wr = wr_matrix.get((opponent, z), float("nan"))
            row_str += f"  {wr:5.1%}" if wr == wr else "    nan"
        print(row_str)

    _check_calibration_targets(wr_matrix)


if __name__ == "__main__":
    main()
