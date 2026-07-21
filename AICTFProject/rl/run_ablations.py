#!/usr/bin/env python3
"""
Leave-one-out ablation matrix for SEA-GUARD / CTF PPO training.

Runs (sequentially by default):
  ours           curriculum + league + full reward shaping
  no_league      curriculum only (−league)
  no_curriculum  fixed OP3 (−curriculum, −league)
  no_shaping     curriculum + league, terminal/offense events only (−dense/PBRS/team)

Usage (from AICTFProject):

  python rl/run_ablations.py --dry-run
  python rl/run_ablations.py --agents 2 --total-steps 1000000
  python rl/run_ablations.py --only ours,no_shaping --seeds 42,43
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_TRAIN_SCRIPT = os.path.join(_SCRIPT_DIR, "train_ppo.py")


def _python_can_import_numpy(python_exe: str) -> bool:
    try:
        proc = subprocess.run(
            [python_exe, "-c", "import numpy"],
            cwd=_PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _candidate_project_venvs() -> List[str]:
    """Prefer the project's own venv over whatever launched this script (e.g. Hermes)."""
    candidates: List[str] = []
    env_override = os.environ.get("AICTFPROJECT_PYTHON") or os.environ.get("MULTIAGENTUAV_PYTHON")
    if env_override:
        candidates.append(env_override)
    win = os.path.join(_PROJECT_DIR, ".venv", "Scripts", "python.exe")
    unix = os.path.join(_PROJECT_DIR, ".venv", "bin", "python")
    candidates.extend([win, unix])
    parent_win = os.path.join(os.path.dirname(_PROJECT_DIR), ".venv", "Scripts", "python.exe")
    parent_unix = os.path.join(os.path.dirname(_PROJECT_DIR), ".venv", "bin", "python")
    candidates.extend([parent_win, parent_unix])
    return candidates


def resolve_python(explicit: Optional[str] = None) -> str:
    """
    Pick a Python that has training deps (numpy, etc.).

    Order: --python / env override → project .venv → sys.executable (if usable).
    """
    ordered: List[str] = []
    if explicit:
        ordered.append(explicit)
    ordered.extend(_candidate_project_venvs())
    ordered.append(sys.executable)

    seen = set()
    for cand in ordered:
        if not cand:
            continue
        path = os.path.abspath(cand)
        if path in seen:
            continue
        seen.add(path)
        if not os.path.isfile(path):
            continue
        if _python_can_import_numpy(path):
            return path

    raise SystemExit(
        "No usable Python found with numpy installed.\n"
        "Activate the project venv first, or pass --python:\n"
        f"  {_PROJECT_DIR}\\.venv\\Scripts\\python.exe rl\\run_ablations.py ...\n"
        "  (or set AICTFPROJECT_PYTHON to that interpreter)"
    )


@dataclass(frozen=True)
class AblationSpec:
    name: str
    mode: str
    reward_ablation: str = "full"
    fixed_opponent: Optional[str] = None
    description: str = ""


# Canonical paper leave-one-out matrix (reviewer-requested ingredients).
DEFAULT_ABLATIONS: List[AblationSpec] = [
    AblationSpec(
        name="ours",
        mode="CURRICULUM_LEAGUE",
        reward_ablation="full",
        description="Full method: curriculum + league + shaped reward",
    ),
    AblationSpec(
        name="no_league",
        mode="CURRICULUM_NO_LEAGUE",
        reward_ablation="full",
        description="-league (curriculum + shaped reward)",
    ),
    AblationSpec(
        name="no_curriculum",
        mode="FIXED_OPPONENT",
        reward_ablation="full",
        fixed_opponent="OP3",
        description="-curriculum (fixed OP3 + shaped reward)",
    ),
    AblationSpec(
        name="no_shaping",
        mode="CURRICULUM_LEAGUE",
        reward_ablation="no_shaping",
        description="-reward shaping (curriculum + league, sparse/terminal events only)",
    ),
]


def _agents_suffix(n_agents: int) -> str:
    n = max(1, int(n_agents))
    return f"{n}v{n}"


def _parse_csv_list(raw: Optional[str]) -> List[str]:
    if raw is None or not str(raw).strip():
        return []
    return [p.strip() for p in str(raw).split(",") if p.strip()]


def _parse_seeds(raw: Optional[str], default: Sequence[int] = (42,)) -> List[int]:
    parts = _parse_csv_list(raw)
    if not parts:
        return list(default)
    return [int(p) for p in parts]


def build_run_tag(spec: AblationSpec, n_agents: int, seed: int, seed_count: int) -> str:
    """Stable, unique tags: ppo_ablate_<name>[_seedN]_NvN."""
    suffix = _agents_suffix(n_agents)
    base = f"ppo_ablate_{spec.name}"
    if seed_count > 1:
        base = f"{base}_seed{seed}"
    return f"{base}_{suffix}"


def build_command(
    spec: AblationSpec,
    *,
    n_agents: int,
    total_steps: Optional[int],
    seed: int,
    seed_count: int,
    device: Optional[str],
    checkpoint_dir: Optional[str],
    python_exe: str,
    extra_args: Optional[Sequence[str]] = None,
) -> List[str]:
    cmd = [
        python_exe,
        _TRAIN_SCRIPT,
        "--mode",
        spec.mode,
        "--max-blue-agents",
        str(n_agents),
        "--reward-ablation",
        spec.reward_ablation,
        "--seed",
        str(seed),
        "--run-tag",
        build_run_tag(spec, n_agents, seed, seed_count),
    ]
    if spec.fixed_opponent:
        cmd.extend(["--fixed-opponent", spec.fixed_opponent])
    if total_steps is not None:
        cmd.extend(["--total-steps", str(int(total_steps))])
    if device:
        cmd.extend(["--device", str(device)])
    if checkpoint_dir:
        cmd.extend(["--checkpoint-dir", str(checkpoint_dir)])
    if extra_args:
        cmd.extend(list(extra_args))
    return cmd


def select_ablations(only: Optional[str]) -> List[AblationSpec]:
    if not only:
        return list(DEFAULT_ABLATIONS)
    wanted = {name.lower() for name in _parse_csv_list(only)}
    known = {a.name: a for a in DEFAULT_ABLATIONS}
    missing = sorted(wanted - set(known))
    if missing:
        raise SystemExit(
            f"Unknown ablation(s): {', '.join(missing)}. "
            f"Known: {', '.join(known)}"
        )
    return [known[name] for name in known if name in wanted]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run leave-one-out PPO ablations (curriculum / league / reward shaping)."
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=2,
        choices=[2, 4, 6, 8],
        help="Team size (default: 2 = 2v2)",
    )
    parser.add_argument("--total-steps", type=int, default=None, help="Forwarded to train_ppo.py")
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated seeds (default: 42). Multi-seed tags get _seedN.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated subset: ours,no_league,no_curriculum,no_shaping",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (forwarded)")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Forwarded checkpoint root")
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Python interpreter for train jobs (default: project .venv if it has numpy)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without launching training",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List ablation specs and exit",
    )
    args, extra = parser.parse_known_args(argv)

    ablations = select_ablations(args.only)
    seeds = _parse_seeds(args.seeds)

    if args.list:
        for spec in ablations:
            print(f"{spec.name:16s} mode={spec.mode:22s} reward={spec.reward_ablation:12s}  {spec.description}")
        return 0

    python_exe = resolve_python(args.python)
    jobs: List[List[str]] = []
    for spec in ablations:
        for seed in seeds:
            jobs.append(
                build_command(
                    spec,
                    n_agents=int(args.agents),
                    total_steps=args.total_steps,
                    seed=seed,
                    seed_count=len(seeds),
                    device=args.device,
                    checkpoint_dir=args.checkpoint_dir,
                    python_exe=python_exe,
                    extra_args=extra,
                )
            )

    print(f"[ablations] project={_PROJECT_DIR}")
    print(f"[ablations] python={python_exe}")
    if os.path.abspath(python_exe) != os.path.abspath(sys.executable):
        print(f"[ablations] note: launcher is {sys.executable}; training uses project venv above")
    print(f"[ablations] {len(jobs)} job(s) | agents={args.agents} | seeds={seeds}")
    for i, cmd in enumerate(jobs, start=1):
        pretty = " ".join(cmd)
        print(f"[ablations] ({i}/{len(jobs)}) {pretty}")

    if args.dry_run:
        print("[ablations] dry-run only; not launching.")
        return 0

    failures = 0
    for i, cmd in enumerate(jobs, start=1):
        print(f"\n[ablations] === starting job {i}/{len(jobs)} ===")
        proc = subprocess.run(cmd, cwd=_PROJECT_DIR)
        if proc.returncode != 0:
            failures += 1
            print(f"[ablations] job {i} failed with exit code {proc.returncode}")
        else:
            print(f"[ablations] job {i} finished OK")

    if failures:
        print(f"[ablations] done with {failures} failure(s)")
        return 1
    print("[ablations] all jobs finished OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
