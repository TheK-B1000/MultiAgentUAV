#!/usr/bin/env python3
"""V6I9 matched-seed forced-z repertoire complementarity evaluation.

Thin wrapper around :mod:`experiments.run_forced_z_eval` so Stage C and
complementarity share one ``episode_results.csv``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.forced_z_eval.analysis.complementarity import (  # noqa: E402
    build_complementarity_report,
    print_complementarity_report,
)
from experiments.forced_z_eval.io import load_episode_results  # noqa: E402
from experiments.forced_z_eval.protocol import DEFAULT_BASE_SEED, DEFAULT_MAPS, DEFAULT_OPPONENTS  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I9 matched-seed forced-z complementarity eval")
    p.add_argument(
        "--checkpoint",
        default="checkpoints/2v2/final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip",
    )
    p.add_argument("--from-run", default=None, help="Analyze existing forced_z run directory")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    p.add_argument("--maps", nargs="+", default=list(DEFAULT_MAPS))
    p.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--oracle-metric", choices=("return", "win_margin", "success"), default="return")
    p.add_argument("--stochastic", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.from_run:
        protocol, cells = load_episode_results(args.from_run)
        report = build_complementarity_report(
            cells,
            opponents=list(protocol.opponents),
            maps=list(protocol.maps),
            latents=tuple(protocol.latents),
            metric=args.oracle_metric,
        )
        print_complementarity_report(report)
        return

    runner = os.path.join(SCRIPT_DIR, "run_forced_z_eval.py")
    cmd = [
        sys.executable,
        runner,
        "--checkpoint",
        args.checkpoint,
        "--episodes",
        str(args.episodes),
        "--device",
        args.device,
        "--base-seed",
        str(args.base_seed),
        "--oracle-metric",
        args.oracle_metric,
        "--opponents",
        *args.opponents,
        "--maps",
        *args.maps,
        "--progress-every",
        "25",
    ]
    if args.out_dir:
        cmd.extend(["--out-dir", args.out_dir])
    if args.stochastic:
        cmd.append("--stochastic")
    if args.analyze_only:
        print("ERROR: --analyze-only requires --from-run; use run_forced_z_eval.py --from-run ... --analyze-only")
        sys.exit(1)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
