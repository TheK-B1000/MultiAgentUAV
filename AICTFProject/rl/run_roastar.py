#!/usr/bin/env python3
"""
Sequential EGT baseline-ladder runner (one job at a time).

Queues fictitious play / double oracle / PFSP (and optionally PFSP+exploiter)
across seeds, always using the project .venv interpreter so Hermes/system Python
cannot miss SB3.

Usage (from AICTFProject):

  # Preview
  python rl/run_roastar.py --dry-run

  # PFSP only, seeds 42/43/44 (recommended first)
  python rl/run_roastar.py --modes pfsp --seeds 42,43,44 --total-steps 1000000

  # Match ablation rollout shape (32 x 512)
  python rl/run_roastar.py --modes pfsp --seeds 42,43,44 --n-envs 32 --n-steps 512

  # The two missing ladder rungs (6 jobs)
  python rl/run_roastar.py --modes fp,do --seeds 42,43,44 --n-envs 32 --n-steps 512

  # Full paper baseline matrix alias (fp, do, pfsp, pfsp_exploiter x 3 seeds)
  python rl/run_roastar.py --full --n-envs 32 --n-steps 512
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import List, Optional, Sequence, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from rl.run_ablations import resolve_python  # noqa: E402
from rl.train_ppo_roastar import LEAGUE_MODES  # noqa: E402

_TRAIN_SCRIPT = os.path.join(_SCRIPT_DIR, "train_ppo_roastar.py")


def _parse_csv_list(raw: Optional[str]) -> List[str]:
    if raw is None or not str(raw).strip():
        return []
    return [p.strip() for p in str(raw).split(",") if p.strip()]


def _parse_seeds(raw: Optional[str], default: Sequence[int] = (42, 43, 44)) -> List[int]:
    parts = _parse_csv_list(raw)
    if not parts:
        return list(default)
    return [int(p) for p in parts]


def _parse_modes(raw: Optional[str], default: Sequence[str] = ("pfsp",)) -> List[str]:
    parts = [p.lower().replace("-", "_") for p in _parse_csv_list(raw)]
    if not parts:
        parts = list(default)
    allowed = set(LEAGUE_MODES)
    bad = [p for p in parts if p not in allowed]
    if bad:
        raise SystemExit(f"Unknown mode(s): {bad}. Expected: {', '.join(sorted(allowed))}")
    return parts


def default_run_tag(mode: str, agents: int, seed: int) -> str:
    return f"ppo_roastar_{mode}_{agents}v{agents}_seed{seed}"


def final_path(checkpoint_dir: str, run_tag: str) -> str:
    return os.path.join(checkpoint_dir, f"final_{run_tag}.zip")


def build_command(
    *,
    python_exe: str,
    mode: str,
    agents: int,
    total_steps: int,
    seed: int,
    device: str,
    checkpoint_dir: str,
    run_tag: str,
    n_envs: Optional[int],
    n_steps: Optional[int],
    resume_latest: bool = False,
    extra_args: Optional[Sequence[str]] = None,
) -> List[str]:
    cmd = [
        python_exe,
        _TRAIN_SCRIPT,
        "--mode",
        mode,
        "--agents",
        str(agents),
        "--total-steps",
        str(total_steps),
        "--seed",
        str(seed),
        "--device",
        device,
        "--checkpoint-dir",
        checkpoint_dir,
        "--run-tag",
        run_tag,
    ]
    if n_envs is not None:
        cmd.extend(["--n-envs", str(int(n_envs))])
    if n_steps is not None:
        cmd.extend(["--n-steps", str(int(n_steps))])
    if resume_latest:
        cmd.append("--resume-latest")
    if extra_args:
        cmd.extend(list(extra_args))
    return cmd


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run ROA-Star PFSP / PFSP+exploiter baselines ONE JOB AT A TIME."
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="pfsp",
        help="Comma-separated: fp, do, pfsp, pfsp_exploiter (default: pfsp)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Alias: modes=fp,do,pfsp,pfsp_exploiter and seeds=42,43,44 (the full EGT ladder)",
    )
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_sb3/2v2")
    parser.add_argument("--n-envs", type=int, default=None, help="Forwarded to train_ppo_roastar.py")
    parser.add_argument("--n-steps", type=int, default=None, help="Forwarded to train_ppo_roastar.py")
    parser.add_argument("--python", type=str, default=None)
    parser.add_argument(
        "--skip-finished",
        action="store_true",
        help="Skip jobs that already have final_<run_tag>.zip",
    )
    parser.add_argument(
        "--stop-on-fail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop after first failure (default: true)",
    )
    parser.add_argument(
        "--resume-latest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forward --resume-latest to train_ppo_roastar when no final_* exists (default: true)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true", help="List queued jobs and exit")
    args, extra = parser.parse_known_args(argv)

    if args.full:
        modes = ["fp", "do", "pfsp", "pfsp_exploiter"]
        seeds = [42, 43, 44]
    else:
        modes = _parse_modes(args.modes)
        seeds = _parse_seeds(args.seeds)

    ckpt_dir = args.checkpoint_dir
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(_PROJECT_DIR, ckpt_dir)

    python_exe = resolve_python(args.python)
    jobs: List[Tuple[str, List[str]]] = []
    skipped = 0
    for mode in modes:
        for seed in seeds:
            run_tag = default_run_tag(mode, int(args.agents), int(seed))
            if args.skip_finished and os.path.isfile(final_path(ckpt_dir, run_tag)):
                print(f"[roastar] skip finished: final_{run_tag}.zip")
                skipped += 1
                continue
            cmd = build_command(
                python_exe=python_exe,
                mode=mode,
                agents=int(args.agents),
                total_steps=int(args.total_steps),
                seed=int(seed),
                device=str(args.device),
                checkpoint_dir=ckpt_dir,
                run_tag=run_tag,
                n_envs=args.n_envs,
                n_steps=args.n_steps,
                resume_latest=bool(args.resume_latest),
                extra_args=extra,
            )
            jobs.append((run_tag, cmd))

    print(f"[roastar] project={_PROJECT_DIR}")
    print(f"[roastar] python={python_exe}")
    if os.path.abspath(python_exe) != os.path.abspath(sys.executable):
        print(f"[roastar] note: launcher is {sys.executable}; training uses project venv above")
    print(f"[roastar] checkpoint_dir={ckpt_dir}")
    print(
        f"[roastar] {len(jobs)} job(s) queued (skipped={skipped}) | modes={modes} | "
        f"seeds={seeds} | n_envs={args.n_envs} | n_steps={args.n_steps} | stop_on_fail={args.stop_on_fail}"
    )
    print("[roastar] mode=SEQUENTIAL (one process at a time - do not launch a second sweep)")
    for i, (label, cmd) in enumerate(jobs, start=1):
        print(f"[roastar] ({i}/{len(jobs)}) {label}")
        print(f"          {' '.join(cmd)}")

    if args.list or args.dry_run:
        if args.dry_run:
            print("[roastar] dry-run only; not launching.")
        return 0

    if not jobs:
        print("[roastar] nothing to run.")
        return 0

    failures = 0
    for i, (label, cmd) in enumerate(jobs, start=1):
        print(f"\n[roastar] === starting job {i}/{len(jobs)}: {label} ===")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=_PROJECT_DIR)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            failures += 1
            print(f"[roastar] job {i} FAILED exit={proc.returncode} after {elapsed/3600.0:.2f}h — {label}")
            if args.stop_on_fail:
                print("[roastar] stopping queue (--stop-on-fail).")
                return 1
        else:
            print(f"[roastar] job {i} finished OK after {elapsed/3600.0:.2f}h — {label}")

    if failures:
        print(f"[roastar] done with {failures} failure(s)")
        return 1
    print("[roastar] all jobs finished OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
