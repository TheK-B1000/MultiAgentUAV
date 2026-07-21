"""
Sanity-checks SB3 PPO checkpoints (ablation arms today, adapted-ROA-Star
checkpoints once they exist) without running a full evaluation.

Two levels of check, both fast:
  1. metadata: read num_timesteps / total_timesteps target directly out of the
     .zip (no torch/SB3 load) -- useful for progress-tracking a run that's
     still writing snapshots on the GPU.
  2. deep (default): also PPO.load() the checkpoint and run a handful of
     predict() calls on a fresh reset, checking for load errors, shape
     mismatches, or NaN/Inf actions -- catches a corrupted or incompatible
     checkpoint before it gets used as an eval baseline or a frozen exploiter
     opponent.

Usage:
    python plot/validate_checkpoints.py --checkpoint-dir checkpoints_sb3/2v2 --agents 2
    python plot/validate_checkpoints.py --checkpoints a.zip b.zip --agents 2 --metadata-only
    python plot/validate_checkpoints.py --checkpoint-dir checkpoints_sb3/2v2 --agents 2 --pattern "ppo_ablate_*.zip"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import zipfile
from dataclasses import dataclass, field
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


@dataclass
class CheckpointReport:
    path: str
    ok: bool
    num_timesteps: Optional[int] = None
    total_timesteps_target: Optional[int] = None
    progress_pct: Optional[float] = None
    deep_checked: bool = False
    errors: list[str] = field(default_factory=list)


def read_metadata(path: str) -> tuple[Optional[int], Optional[int], list[str]]:
    """Read num_timesteps / _total_timesteps out of an SB3 .zip's `data` entry
    without importing torch/SB3 -- safe to run on a checkpoint that's actively
    being (re)written by a training process."""
    errors: list[str] = []
    try:
        with zipfile.ZipFile(path) as z:
            raw = z.read("data")
        data = json.loads(raw.decode("utf-8"))
        return data.get("num_timesteps"), data.get("_total_timesteps"), errors
    except zipfile.BadZipFile:
        errors.append("not a valid zip file (possibly mid-write)")
        return None, None, errors
    except KeyError:
        errors.append("zip has no 'data' entry (not an SB3 checkpoint?)")
        return None, None, errors
    except Exception as exc:  # noqa: BLE001 - report, don't crash the batch
        errors.append(f"metadata read failed: {exc}")
        return None, None, errors


def deep_check(path: str, *, n_agents: int, device: str) -> list[str]:
    """Load the checkpoint for real and run a few predict() calls, checking for
    load errors, shape mismatches, and NaN/Inf actions."""
    import numpy as np

    errors: list[str] = []
    try:
        from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
        from eval_rollout import ppo_load_custom_objects, _numpy_compat_shim
        from stable_baselines3 import PPO

        _numpy_compat_shim()
        cfg = GPUFieldConfig(
            n_envs=1,
            max_blue_agents=n_agents,
            max_red_agents=n_agents,
            device=device,
            aquaticus_profile=True,
            rules_profile="OURS",
        )
        env = GPUCTFVecEnv(cfg)
        try:
            model = PPO.load(path, device=device, custom_objects=ppo_load_custom_objects(env))
            model.policy.set_training_mode(False)
            obs = env.reset()
            for _ in range(3):
                actions, _ = model.predict(obs, deterministic=True)
                if not np.all(np.isfinite(actions)):
                    errors.append("predict() returned non-finite (NaN/Inf) actions")
                    break
                env.step_async(actions)
                obs, _, dones, _infos = env.step_wait()
        finally:
            env.close()
    except Exception as exc:  # noqa: BLE001 - report, don't crash the batch
        errors.append(f"deep check failed: {exc}")
    return errors


def validate_one(path: str, *, n_agents: int, device: str, metadata_only: bool) -> CheckpointReport:
    if not os.path.isfile(path):
        return CheckpointReport(path=path, ok=False, errors=["file not found"])

    num_timesteps, total_target, errors = read_metadata(path)
    progress_pct = None
    if num_timesteps is not None and total_target:
        progress_pct = 100.0 * float(num_timesteps) / float(total_target)

    deep_checked = False
    if not errors and not metadata_only:
        deep_errors = deep_check(path, n_agents=n_agents, device=device)
        errors.extend(deep_errors)
        deep_checked = True

    return CheckpointReport(
        path=path,
        ok=(len(errors) == 0),
        num_timesteps=num_timesteps,
        total_timesteps_target=total_target,
        progress_pct=progress_pct,
        deep_checked=deep_checked,
        errors=errors,
    )


def print_report(report: CheckpointReport) -> None:
    status = "OK" if report.ok else "FAIL"
    base = os.path.basename(report.path)
    progress = f"{report.progress_pct:.1f}%" if report.progress_pct is not None else "?"
    steps = f"{report.num_timesteps:,}" if report.num_timesteps is not None else "?"
    target = f"{report.total_timesteps_target:,}" if report.total_timesteps_target is not None else "?"
    depth = "deep" if report.deep_checked else "metadata-only"
    line = f"[{status}] {base}  steps={steps}/{target} ({progress})  [{depth}]"
    print(line)
    for err in report.errors:
        print(f"    - {err}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="*", default=None, help="Explicit list of .zip paths")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory to glob for checkpoints")
    parser.add_argument("--pattern", default="*.zip", help="Glob pattern within --checkpoint-dir")
    parser.add_argument("--agents", type=int, required=True, help="Team size the checkpoint(s) were trained with")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--metadata-only", action="store_true", help="Skip the deep PPO.load()+predict() check")
    args = parser.parse_args()

    paths: list[str] = list(args.checkpoints or [])
    if args.checkpoint_dir:
        paths.extend(sorted(glob.glob(os.path.join(args.checkpoint_dir, args.pattern))))
    if not paths:
        print("[validate_checkpoints] no checkpoints given (use --checkpoints and/or --checkpoint-dir)")
        return 1

    reports = [validate_one(p, n_agents=args.agents, device=args.device, metadata_only=args.metadata_only) for p in paths]
    for r in reports:
        print_report(r)

    n_fail = sum(1 for r in reports if not r.ok)
    print(f"\n[validate_checkpoints] {len(reports)} checked, {len(reports) - n_fail} ok, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
