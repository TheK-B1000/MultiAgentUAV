#!/usr/bin/env python3
"""V6I24 lean full-policy population diagnostic (Path C fallback).

Hypothesis
----------
Can K=4 fully independent actor-critic policies, cloned from the same
V6I21J-competent checkpoint and trained under *fixed* opponent×map cell
pressures (no PFSP, no rotation, no shared gradients), produce a real
functional repertoire?

Engineering
-----------
Four ordinary independent ``train_ppo`` runs + one population manifest.
No ``PopulationTrainer``, GPU swapping, pressure rotation, or distillation.

Usage (from AICTFProject/)
--------------------------
    uv run python experiments/run_v6i24_full_policy_population.py \\
        --checkpoint checkpoints/2v2/final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip \\
        --output-dir artifacts/v6i24_population_seed1 \\
        --seed 1 \\
        --max-probe 5

    # Then evaluate:
    uv run python experiments/run_v6i24_population_eval_gates.py \\
        --checkpoint-dir artifacts/v6i24_population_seed1/probe_05u \\
        --episodes-per-cell 32
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v6i24_population_config import (  # noqa: E402
    DEFAULT_CALIBRATION_REPORT,
    PROBE_UPDATES,
    STEPS_PER_UPDATE,
    build_member_pressures,
    pressures_manifest,
)

DEFAULT_ANCHOR = (
    PROJECT_ROOT
    / "checkpoints"
    / "2v2"
    / "final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip"
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I24 lean full-policy population diagnostic")
    p.add_argument(
        "--checkpoint",
        "--anchor",
        dest="checkpoint",
        default=str(DEFAULT_ANCHOR),
        help="V6I21J-competent source checkpoint (documented V6I9 generalist under v6i21J arena).",
    )
    p.add_argument("--output-dir", default="artifacts/v6i24_population")
    p.add_argument("--calibration-report", default=str(DEFAULT_CALIBRATION_REPORT))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument(
        "--max-probe",
        type=int,
        default=25,
        choices=list(PROBE_UPDATES),
        help="Stop after this probe budget (5, 10, or 25 updates per policy).",
    )
    p.add_argument(
        "--members",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of member ids (default: all 4).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Write pressures/manifest only; do not train.",
    )
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Train only; skip invoking eval gates after each probe.",
    )
    return p.parse_args()


def _find_final_zip(ckpt_dir: Path, run_tag: str) -> Path:
    preferred = ckpt_dir / f"final_{run_tag}_2v2.zip"
    if preferred.is_file():
        return preferred
    finals = sorted(ckpt_dir.glob("final_*.zip"))
    if finals:
        return finals[-1]
    raise FileNotFoundError(f"No final_*.zip in {ckpt_dir} after training {run_tag}")


def _train_member(
    *,
    member_id: int,
    label: str,
    cell_weights: tuple[tuple[str, str, float], ...],
    load_path: Path,
    additional_steps: int,
    load_weights_only: bool,
    seed: int,
    device: str,
    n_envs: int,
    n_steps: int,
    checkpoint_dir: Path,
    run_tag: str,
) -> Path:
    from rl.config.ppo_config import PPOConfig
    from rl.presets import PRESET_REGISTRY
    from rl.train_ppo import train_ppo

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    apply_fn = PRESET_REGISTRY["v6i24"]
    cfg = apply_fn(PPOConfig())
    cfg.seed = int(seed)
    cfg.device = str(device)
    cfg.n_envs = int(n_envs)
    cfg.n_steps = int(n_steps)
    cfg.load_path = str(load_path)
    cfg.load_weights_only = bool(load_weights_only)
    cfg.additional_timesteps = int(additional_steps)
    cfg.checkpoint_dir = str(checkpoint_dir)
    cfg.run_tag = run_tag
    cfg.fresh_metrics_csv = True
    cfg.training_cell_distribution = tuple(
        (str(o).upper(), str(m), float(w)) for o, m, w in cell_weights
    )
    # Keep opponent_pool tags aligned with cells for validation / banners.
    cfg.opponent_pool = tuple(sorted({str(o).upper() for o, _, _ in cell_weights}))
    cfg.opponent_pool_weights = ()
    cfg.freeze_return_norm_after_load = True

    print(
        f"[v6i24] train member={member_id} ({label}) "
        f"load={load_path.name} additional_steps={additional_steps} "
        f"load_weights_only={load_weights_only} seed={seed}",
        flush=True,
    )
    train_ppo(cfg)
    return _find_final_zip(checkpoint_dir, run_tag)


def _copy_member_artifact(src: Path, dest_dir: Path, member_id: int, label: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"member_{member_id}_{label}.zip"
    shutil.copy2(src, dest)
    return dest


def _maybe_run_eval(probe_dir: Path, *, seed: int, device: str) -> None:
    from experiments.run_v6i24_population_eval_gates import run_eval_gates

    run_eval_gates(
        checkpoint_dir=probe_dir,
        output_dir=probe_dir / "eval_gates",
        episodes_per_cell=32,
        seed=seed,
        device=device,
        confirm_episodes=False,
    )


def main() -> int:
    args = _parse_args()
    ckpt = Path(args.checkpoint)
    calib = Path(args.calibration_report)
    if not calib.is_file():
        print(f"ERROR: calibration report not found: {calib}")
        return 2

    if not args.dry_run and not ckpt.is_file():
        print(f"ERROR: V6I21J-competent anchor checkpoint not found: {ckpt}")
        print("Pass --checkpoint / --anchor to an existing .zip (V6I9 generalist preferred).")
        print("Do not clone from V6I22E or V6I23.")
        return 2
    if args.dry_run and not ckpt.is_file():
        print(f"[v6i24] WARNING: anchor missing ({ckpt}); dry-run will still write pressures.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pressures = build_member_pressures(report_path=calib)
    if args.members is not None:
        keep = set(int(m) for m in args.members)
        pressures = [p for p in pressures if p.member_id in keep]
    if not pressures:
        print("ERROR: no members selected")
        return 2

    manifest: dict[str, Any] = {
        "experiment": "v6i24_full_policy_population",
        "classification": "DIAGNOSTIC",
        "path": "C_fallback_independent_teachers",
        "parent_preset": "v6i21j_hardpool_balance_calibration",
        "source_checkpoint": str(ckpt.resolve()) if ckpt.is_file() else str(ckpt),
        "seed": int(args.seed),
        "device": str(args.device),
        "created_at_utc": _utc(),
        "max_probe_updates": int(args.max_probe),
        "steps_per_update": STEPS_PER_UPDATE,
        "freeze_return_norm_after_load": True,
        "pressures": pressures_manifest(pressures, source=str(calib.resolve())),
        "probes": {},
    }
    manifest_path = output_dir / "population_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[v6i24] wrote {manifest_path}")

    print("=" * 72)
    print("V6I24 Path C: lean independent full-policy population")
    print("=" * 72)
    print(f"Anchor:     {ckpt}")
    print(f"Output:     {output_dir}")
    print(f"Max probe:  {args.max_probe}u ({args.max_probe * STEPS_PER_UPDATE} steps/policy)")
    print("Pressures (fixed through 25u):")
    for p in pressures:
        top = sorted(p.cell_weights, key=lambda t: -t[2])[:3]
        top_s = ", ".join(f"{o}|{m}={w:.3f}" for o, m, w in top)
        print(f"  pi{p.member_id} {p.label}: {p.description}")
        print(f"       top cells: {top_s}")
    print()

    if args.dry_run:
        print("[v6i24] dry-run complete (no training)")
        return 0

    probes = [u for u in PROBE_UPDATES if u <= int(args.max_probe)]
    prev_u = 0
    member_ckpts: dict[int, Path] = {}

    for probe_u in probes:
        delta_u = probe_u - prev_u
        delta_steps = delta_u * STEPS_PER_UPDATE
        probe_dir = output_dir / f"probe_{probe_u:02d}u"
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe_meta: dict[str, Any] = {
            "probe_updates": probe_u,
            "delta_updates": delta_u,
            "delta_steps": delta_steps,
            "members": {},
            "started_at_utc": _utc(),
        }

        for pressure in pressures:
            mid = pressure.member_id
            label = pressure.label
            member_work = output_dir / "work" / f"member_{mid}_{label}" / f"{probe_u:02d}u"
            run_tag = f"v6i24_m{mid}_{label}_{probe_u:02d}u_seed{args.seed}"
            if prev_u == 0:
                load_path = ckpt
                load_weights_only = True
            else:
                load_path = member_ckpts[mid]
                load_weights_only = False
            member_seed = int(args.seed) + int(pressure.seed_offset) + probe_u
            trained = _train_member(
                member_id=mid,
                label=label,
                cell_weights=pressure.cell_weights,
                load_path=load_path,
                additional_steps=delta_steps,
                load_weights_only=load_weights_only,
                seed=member_seed,
                device=args.device,
                n_envs=args.n_envs,
                n_steps=args.n_steps,
                checkpoint_dir=member_work,
                run_tag=run_tag,
            )
            artifact = _copy_member_artifact(trained, probe_dir, mid, label)
            member_ckpts[mid] = artifact
            probe_meta["members"][str(mid)] = {
                "label": label,
                "run_tag": run_tag,
                "seed": member_seed,
                "checkpoint": str(artifact),
                "work_checkpoint": str(trained),
                "cell_weights": [
                    {"opponent": o, "map": m, "weight": w} for o, m, w in pressure.cell_weights
                ],
            }

        probe_meta["finished_at_utc"] = _utc()
        (probe_dir / "probe_meta.json").write_text(json.dumps(probe_meta, indent=2), encoding="utf-8")
        manifest["probes"][f"{probe_u}u"] = probe_meta
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if not args.skip_eval:
            print(f"[v6i24] running eval gates at {probe_u}u ...", flush=True)
            try:
                _maybe_run_eval(probe_dir, seed=int(args.seed), device=str(args.device))
            except Exception as exc:
                print(f"[v6i24] WARNING: eval gates failed at {probe_u}u: {exc}")

        prev_u = probe_u

    print("=" * 72)
    print(f"V6I24 probes complete through {args.max_probe}u")
    print(f"Manifest: {manifest_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
