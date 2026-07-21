#!/usr/bin/env python3
"""
Leave-one-out ablation matrix for SEA-GUARD / CTF PPO training.

Always runs jobs **one at a time** (never parallel). Prefer a single invocation
with all seeds so host RAM is not exhausted by concurrent sweeps.

Recommended full matrix (4 arms x 3 seeds = 12 sequential jobs):

  python rl/run_ablations.py --full --agents 2 --total-steps 1000000 --n-envs 4 --resume-oom

Usage:

  python rl/run_ablations.py --dry-run --full
  python rl/run_ablations.py --only ours --seeds 42,43,44 --n-envs 4 --resume-oom
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


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
    # Always embed seed when running a multi-seed matrix (or a non-default seed alone).
    if seed_count > 1 or int(seed) != 42:
        base = f"{base}_seed{seed}"
    return f"{base}_{suffix}"


def legacy_run_tag(spec: AblationSpec, n_agents: int) -> str:
    """Pre-multi-seed tag used by seed-42 single-sweep runs (no _seed42)."""
    return f"ppo_ablate_{spec.name}_{_agents_suffix(n_agents)}"


def _has_any_artifacts(checkpoint_dir: str, run_tag: str) -> bool:
    """True if metrics / oom / crash / snapshots already exist for this tag."""
    names = [
        f"{run_tag}_metrics.csv",
        f"oom_save_{run_tag}.zip",
        f"crash_save_{run_tag}.zip",
        f"final_{run_tag}.zip",
    ]
    for name in names:
        if os.path.isfile(os.path.join(checkpoint_dir, name)):
            return True
    # league / self-play snapshots
    try:
        for fn in os.listdir(checkpoint_dir):
            if fn.startswith(f"{run_tag}_") and fn.endswith(".zip"):
                return True
    except OSError:
        pass
    return False


def resolve_run_tag(
    spec: AblationSpec,
    n_agents: int,
    seed: int,
    seed_count: int,
    checkpoint_dir: str,
    *,
    resume_oom: bool,
) -> Tuple[str, Optional[str]]:
    """
    Pick run_tag + optional --load path.

    Seed 42 often has legacy artifacts without `_seed42` in the name; prefer those
    when resuming/continuing so metrics/checkpoints stay continuous.
    """
    primary = build_run_tag(spec, n_agents, seed, seed_count)
    candidates = [primary]
    if int(seed) == 42:
        leg = legacy_run_tag(spec, n_agents)
        # Prefer legacy first so partial seed-42 sweeps continue cleanly.
        candidates = [leg, primary]

    if resume_oom:
        for tag in candidates:
            oom = find_oom_checkpoint(checkpoint_dir, tag)
            if oom:
                return tag, oom

    if int(seed) == 42:
        leg = legacy_run_tag(spec, n_agents)
        if _has_any_artifacts(checkpoint_dir, leg):
            return leg, None
    return primary, None


def find_oom_checkpoint(checkpoint_dir: str, run_tag: str) -> Optional[str]:
    """Return oom_save_<run_tag>.zip if present (and prefer it over crash_save)."""
    for prefix in ("oom_save_", "crash_save_"):
        path = os.path.join(checkpoint_dir, f"{prefix}{run_tag}.zip")
        if os.path.isfile(path):
            return path
    return None


def find_final_checkpoint(checkpoint_dir: str, run_tag: str) -> Optional[str]:
    path = os.path.join(checkpoint_dir, f"final_{run_tag}.zip")
    return path if os.path.isfile(path) else None


def job_already_finished(
    spec: AblationSpec,
    n_agents: int,
    seed: int,
    seed_count: int,
    checkpoint_dir: str,
) -> Optional[str]:
    """Return final zip path if any alias tag is already complete."""
    tags = [build_run_tag(spec, n_agents, seed, seed_count)]
    if int(seed) == 42:
        leg = legacy_run_tag(spec, n_agents)
        if leg not in tags:
            tags.append(leg)
    for tag in tags:
        final = find_final_checkpoint(checkpoint_dir, tag)
        if final:
            return final
    return None


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
    n_envs: Optional[int] = None,
    n_steps: Optional[int] = None,
    load_path: Optional[str] = None,
    run_tag: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> List[str]:
    tag = run_tag or build_run_tag(spec, n_agents, seed, seed_count)
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
        tag,
    ]
    if spec.fixed_opponent:
        cmd.extend(["--fixed-opponent", spec.fixed_opponent])
    if total_steps is not None:
        cmd.extend(["--total-steps", str(int(total_steps))])
    if device:
        cmd.extend(["--device", str(device)])
    if checkpoint_dir:
        cmd.extend(["--checkpoint-dir", str(checkpoint_dir)])
    if n_envs is not None:
        cmd.extend(["--n-envs", str(int(n_envs))])
    if n_steps is not None:
        cmd.extend(["--n-steps", str(int(n_steps))])
    if load_path:
        cmd.extend(["--load", load_path])
    if extra_args:
        cmd.extend(list(extra_args))
    return cmd


def select_ablations(only: Optional[str]) -> List[AblationSpec]:
    if not only:
        return list(DEFAULT_ABLATIONS)
    wanted = _parse_csv_list(only)
    known = {a.name: a for a in DEFAULT_ABLATIONS}
    missing = sorted({name.lower() for name in wanted} - set(known))
    if missing:
        raise SystemExit(
            f"Unknown ablation(s): {', '.join(missing)}. "
            f"Known: {', '.join(known)}"
        )
    # Preserve the user-specified order (important when finishing non-league arms first).
    out: List[AblationSpec] = []
    seen = set()
    for name in wanted:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(known[key])
    return out


def _default_checkpoint_dir(n_agents: int, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    suffix = _agents_suffix(n_agents)
    if os.path.exists("/content/drive/MyDrive"):
        return os.path.join("/content/drive/MyDrive/CTF_models", suffix)
    return os.path.join(_PROJECT_DIR, "checkpoints_sb3", suffix)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run leave-one-out PPO ablations ONE JOB AT A TIME "
            "(curriculum / league / reward shaping). Never launches concurrent trains."
        )
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=2,
        choices=[2, 4, 6, 8],
        help="Team size (default: 2 = 2v2)",
    )
    parser.add_argument("--total-steps", type=int, default=1_000_000, help="Forwarded to train_ppo.py (default: 1e6)")
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated seeds (default: 42). Multi-seed tags get _seedN.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Paper matrix shortcut: all 4 arms x seeds 42,43,44 (overrides --seeds unless also set).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated subset: ours,no_league,no_curriculum,no_shaping",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Parallel envs per job (default: 4; safer host-RAM than train_ppo's 8). Use 2 if OOM persists.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Optional rollout length override (default: train_ppo 2048).",
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
        "--resume-oom",
        action="store_true",
        help="If oom_save_<run_tag>.zip exists, pass --load so the job finishes the remaining budget.",
    )
    parser.add_argument(
        "--skip-finished",
        action="store_true",
        help="Skip jobs that already have final_<run_tag>.zip.",
    )
    parser.add_argument(
        "--stop-on-fail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the queue after the first failed job (default: true). Use --no-stop-on-fail to continue.",
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

    if args.full and args.seeds == "42":
        seeds = [42, 43, 44]
    else:
        seeds = _parse_seeds(args.seeds)
    ablations = select_ablations(args.only)

    if args.list:
        for spec in ablations:
            print(f"{spec.name:16s} mode={spec.mode:22s} reward={spec.reward_ablation:12s}  {spec.description}")
        return 0

    ckpt_dir = _default_checkpoint_dir(int(args.agents), args.checkpoint_dir)
    python_exe = resolve_python(args.python)

    jobs: List[Tuple[str, List[str]]] = []
    skipped = 0
    for spec in ablations:
        for seed in seeds:
            if args.skip_finished:
                done = job_already_finished(spec, int(args.agents), seed, len(seeds), ckpt_dir)
                if done:
                    print(f"[ablations] skip finished: {os.path.basename(done)}")
                    skipped += 1
                    continue
            run_tag, load_path = resolve_run_tag(
                spec,
                int(args.agents),
                seed,
                len(seeds),
                ckpt_dir,
                resume_oom=bool(args.resume_oom),
            )
            cmd = build_command(
                spec,
                n_agents=int(args.agents),
                total_steps=args.total_steps,
                seed=seed,
                seed_count=len(seeds),
                device=args.device,
                checkpoint_dir=args.checkpoint_dir or ckpt_dir,
                python_exe=python_exe,
                n_envs=int(args.n_envs) if args.n_envs is not None else None,
                n_steps=args.n_steps,
                load_path=load_path,
                run_tag=run_tag,
                extra_args=extra,
            )
            label = run_tag + (f" [resume {os.path.basename(load_path)}]" if load_path else "")
            jobs.append((label, cmd))

    print(f"[ablations] project={_PROJECT_DIR}")
    print(f"[ablations] python={python_exe}")
    if os.path.abspath(python_exe) != os.path.abspath(sys.executable):
        print(f"[ablations] note: launcher is {sys.executable}; training uses project venv above")
    print(f"[ablations] checkpoint_dir={ckpt_dir}")
    print(
        f"[ablations] {len(jobs)} job(s) queued (skipped={skipped}) | agents={args.agents} | "
        f"seeds={seeds} | n_envs={args.n_envs} | stop_on_fail={args.stop_on_fail}"
    )
    print("[ablations] mode=SEQUENTIAL (one process at a time - do not launch a second sweep)")
    for i, (label, cmd) in enumerate(jobs, start=1):
        print(f"[ablations] ({i}/{len(jobs)}) {label}")
        print(f"            {' '.join(cmd)}")

    if args.dry_run:
        print("[ablations] dry-run only; not launching.")
        return 0

    if not jobs:
        print("[ablations] nothing to run.")
        return 0

    failures = 0
    for i, (label, cmd) in enumerate(jobs, start=1):
        print(f"\n[ablations] === starting job {i}/{len(jobs)}: {label} ===")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=_PROJECT_DIR)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            failures += 1
            print(
                f"[ablations] job {i} FAILED exit={proc.returncode} after {elapsed/3600.0:.2f}h — {label}"
            )
            if args.stop_on_fail:
                print(
                    "[ablations] stopping queue (--stop-on-fail). "
                    "Re-run with --resume-oom after freeing RAM, or pass --no-stop-on-fail to continue."
                )
                return 1
        else:
            print(f"[ablations] job {i} finished OK after {elapsed/3600.0:.2f}h — {label}")

    if failures:
        print(f"[ablations] done with {failures} failure(s)")
        return 1
    print("[ablations] all jobs finished OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
