#!/usr/bin/env python3
"""
Sequential paper-revision evaluation / training pipeline (ONE GPU job at a time).

Order:
  1. ROA-Star shared frozen OP3/OP4 eval (9 finals × 1000 ep)   [skip if CSV complete]
  2. Ablation 2v2 shared frozen OP3/OP4 eval (1000 ep)
  3. Checkpoint tournaments (ROA seed42 + SEA-GUARD ours)
  4. Cross-play payoff matrix (2v2)
  5. Exploitability (red BR vs ours / ROA / self-play)
  6. Optional: train ROA-Star pfsp_exploiter seeds 42–44

Usage (from AICTFProject):

  # Wait for an already-running ROA eval, then continue from step 2
  python plot/run_paper_eval_pipeline.py --wait-roastar --skip-roastar-run

  # Full pipeline from scratch
  python plot/run_paper_eval_pipeline.py --episodes 1000 --train-exploiter

  # Dry-run
  python plot/run_paper_eval_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from typing import List, Optional, Sequence

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from rl.run_ablations import resolve_python  # noqa: E402


def _run(cmd: List[str], *, dry_run: bool) -> int:
    print("\n[pipeline] >>> " + " ".join(cmd), flush=True)
    if dry_run:
        print("[pipeline] dry-run; not executing")
        return 0
    proc = subprocess.run(cmd, cwd=_PROJECT_DIR)
    if proc.returncode != 0:
        print(f"[pipeline] FAILED exit={proc.returncode}", flush=True)
    return int(proc.returncode)


def _roastar_complete(per_seed_csv: str, *, episodes: int, n_expected: int = 18) -> bool:
    """Expect 3 settings × 3 seeds × 2 opponents = 18 per-seed rows at full episode budget."""
    if not os.path.isfile(per_seed_csv):
        return False
    with open(per_seed_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ok = [
        r
        for r in rows
        if int(float(r.get("n_episodes", 0) or 0)) == int(episodes)
    ]
    return len(ok) >= int(n_expected)


def _wait_roastar(per_seed_csv: str, *, episodes: int, poll_s: float = 60.0) -> None:
    print(
        f"[pipeline] waiting for ROA shared eval completion "
        f"({per_seed_csv}, {episodes} ep × 18 jobs)...",
        flush=True,
    )
    while not _roastar_complete(per_seed_csv, episodes=episodes):
        time.sleep(poll_s)
        print(f"[pipeline] still waiting... {time.strftime('%H:%M:%S')}", flush=True)
    print("[pipeline] ROA shared eval looks complete.", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-roastar-run",
        action="store_true",
        help="Do not launch ROA eval (assume it is already running or done)",
    )
    parser.add_argument(
        "--wait-roastar",
        action="store_true",
        help="Block until ROA per-seed CSV has 18 complete rows before continuing",
    )
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-tournament", action="store_true")
    parser.add_argument("--skip-crossplay", action="store_true")
    parser.add_argument("--skip-exploitability", action="store_true")
    parser.add_argument(
        "--train-exploiter",
        action="store_true",
        help="After evals, train pfsp_exploiter for 2v2 seeds 42,43,44",
    )
    parser.add_argument("--python", default=None)
    args = parser.parse_args(argv)

    py = resolve_python(args.python)
    ep = int(args.episodes)
    roastar_per_seed = os.path.join("csv", "eval_roastar_shared_per_seed.csv")
    roastar_out = os.path.join("csv", "eval_roastar_shared.csv")

    # --- 1. ROA shared eval ---
    if not args.skip_roastar_run:
        if _roastar_complete(roastar_per_seed, episodes=ep):
            print("[pipeline] ROA shared eval already complete; skipping launch")
        else:
            rc = _run(
                [
                    py,
                    os.path.join("plot", "eval_roastar_matrix.py"),
                    "--episodes",
                    str(ep),
                    "--require-complete",
                    "--progress-every",
                    "50",
                    "--out",
                    roastar_out,
                    "--per-seed-out",
                    roastar_per_seed,
                    "--points-dir",
                    os.path.join("csv", "eval_roastar_shared_points"),
                    "--device",
                    args.device,
                ],
                dry_run=args.dry_run,
            )
            if rc != 0:
                return rc
    if args.wait_roastar and not args.dry_run:
        _wait_roastar(roastar_per_seed, episodes=ep)

    # --- 2. Ablation shared eval ---
    if not args.skip_ablations:
        rc = _run(
            [
                py,
                os.path.join("plot", "eval_ablations.py"),
                "--checkpoint-dir",
                os.path.join("checkpoints_sb3", "2v2"),
                "--episodes",
                str(ep),
                "--require-complete",
                "--progress-every",
                "50",
                "--out",
                os.path.join("csv", "eval_ablation_2v2.csv"),
                "--paper-out",
                os.path.join("csv", "eval_ablation_2v2_paper.csv"),
                "--per-seed-out",
                os.path.join("csv", "eval_ablation_2v2_per_seed.csv"),
                "--device",
                args.device,
            ],
            dry_run=args.dry_run,
        )
        if rc != 0:
            return rc

    # --- 3. Checkpoint tournaments ---
    if not args.skip_tournament:
        for tag in (
            "ppo_roastar_pfsp_2v2_seed42",
            "ppo_ablate_ours_2v2",
        ):
            rc = _run(
                [
                    py,
                    os.path.join("plot", "checkpoint_tournament.py"),
                    "--checkpoint-dir",
                    os.path.join("checkpoints_sb3", "2v2"),
                    "--run-tag",
                    tag,
                    "--agents",
                    "2",
                    "--val-episodes",
                    "50",
                    "--cross-episodes",
                    "20",
                    "--out",
                    os.path.join("csv", f"tournament_{tag}.csv"),
                    "--device",
                    args.device,
                ],
                dry_run=args.dry_run,
            )
            if rc != 0:
                print(f"[pipeline] tournament for {tag} failed; continuing")

    # --- 4. Cross-play ---
    if not args.skip_crossplay:
        rc = _run(
            [
                py,
                os.path.join("plot", "eval_crossplay.py"),
                "--checkpoint-dir",
                os.path.join("checkpoints_sb3", "2v2"),
                "--episodes",
                "100",
                "--seeds",
                "42",
                "--out",
                os.path.join("csv", "crossplay_2v2.csv"),
                "--device",
                args.device,
            ],
            dry_run=args.dry_run,
        )
        if rc != 0:
            return rc

    # --- 5. Exploitability ---
    if not args.skip_exploitability:
        rc = _run(
            [
                py,
                os.path.join("plot", "eval_exploitability.py"),
                "--checkpoint-dir",
                os.path.join("checkpoints_sb3", "2v2"),
                "--exploiter-steps",
                "300000",
                "--exploiter-seeds",
                "0,1,2",
                "--n-envs",
                "8",
                "--eval-episodes",
                "200",
                "--out",
                os.path.join("csv", "exploitability_2v2.csv"),
                "--curve-out",
                os.path.join("csv", "exploitability_curves_2v2.csv"),
                "--device",
                args.device,
            ],
            dry_run=args.dry_run,
        )
        if rc != 0:
            return rc

    # --- 6. Optional pfsp_exploiter training ---
    if args.train_exploiter:
        rc = _run(
            [
                py,
                os.path.join("rl", "run_roastar.py"),
                "--modes",
                "pfsp_exploiter",
                "--agents",
                "2",
                "--seeds",
                "42,43,44",
                "--total-steps",
                "1000000",
                "--n-envs",
                "32",
                "--n-steps",
                "512",
                "--checkpoint-dir",
                os.path.join("checkpoints_sb3", "2v2"),
                "--skip-finished",
                "--stop-on-fail",
            ],
            dry_run=args.dry_run,
        )
        if rc != 0:
            return rc

    print("\n[pipeline] all requested stages finished OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
