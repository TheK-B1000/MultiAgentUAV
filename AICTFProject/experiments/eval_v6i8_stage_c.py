#!/usr/bin/env python3
"""Stage C gate evaluation — analysis over canonical forced-z episodes."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.forced_z_eval.analysis.stage_c import build_stage_c_report, print_stage_c_report  # noqa: E402
from experiments.forced_z_eval.io import load_episode_results  # noqa: E402
from experiments.forced_z_eval.protocol import DEFAULT_BASE_SEED, DEFAULT_MAPS, DEFAULT_OPPONENTS  # noqa: E402
from experiments.forced_z_eval.subprocess_utils import run_with_process_tree_timeout  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I8 Stage C gate evaluation")
    p.add_argument("--checkpoints", nargs="+", help="Checkpoint(s) to simulate")
    p.add_argument("--from-run", nargs="+", help="Existing forced_z run dir(s) with episode_results.csv")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    p.add_argument("--maps", nargs="+", default=list(DEFAULT_MAPS))
    p.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--timeout-seconds", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.from_run:
        results = []
        for run_path in args.from_run:
            protocol, cells = load_episode_results(run_path)
            print(f"\n{'='*60}\nRun: {run_path}\n{'='*60}")
            report = build_stage_c_report(
                cells,
                opponents=list(protocol.opponents),
                maps=list(protocol.maps),
                latents=tuple(protocol.latents),
            )
            print("\n--- Stage C Gate ---")
            print_stage_c_report(report)
            results.append((run_path, report["passed"]))
        print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
        for label, passed in results:
            status = "PASS — promote to Stage D" if passed else "FAIL — do not promote"
            print(f"  {label}: {status}")
        return

    if not args.checkpoints:
        print("ERROR: provide --checkpoints or --from-run")
        sys.exit(1)

    runner = os.path.join(SCRIPT_DIR, "run_forced_z_eval.py")
    results = []
    for ckpt in args.checkpoints:
        cmd = [
            sys.executable,
            runner,
            "--checkpoint",
            ckpt,
            "--episodes",
            str(args.episodes),
            "--device",
            args.device,
            "--base-seed",
            str(args.base_seed),
            "--opponents",
            *args.opponents,
            "--maps",
            *args.maps,
        ]
        if args.out_dir:
            cmd.extend(["--out-dir", args.out_dir])
        if args.stochastic:
            cmd.append("--stochastic")
        print(f"\n{'='*60}\nCheckpoint: {os.path.basename(ckpt)}\n{'='*60}")
        try:
            proc = run_with_process_tree_timeout(cmd, cwd=PROJECT_ROOT, timeout_seconds=args.timeout_seconds)
            ok = proc.returncode == 0
        except subprocess.TimeoutExpired as exc:
            print(f"ERROR: forced-z eval timed out after {exc.timeout} seconds and the process tree was terminated")
            ok = False
        results.append((os.path.basename(ckpt), ok))
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for label, ok in results:
        status = "PASS — promote to Stage D" if ok else "FAIL — do not promote"
        print(f"  {label}: {status}")


if __name__ == "__main__":
    main()
