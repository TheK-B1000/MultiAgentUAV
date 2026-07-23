#!/usr/bin/env python3
"""V6I24 lean full-policy population diagnostic (Path C fallback).

Hypothesis
----------
Can K=4 fully independent actor-critic policies, cloned from the same
competent *shared core* and trained under fixed opponent×map cell pressures,
produce a real functional repertoire?

Checkpoint modes
----------------
* ``auto`` / ``full``: load a clean V6I9/V6I21J anchor as-is (preferred).
* ``shared-core``: extract shared trunk/CNN/action_head/critic (+ z-embed for
  frozen-z=0 competence) from a donor such as V6I23; discard adapters,
  per-z heads, router, and optimizer state.
* ``fresh``: identical random init for all four members.

Usage (from AICTFProject/)
--------------------------
    $anchor = "artifacts\\v6i23_population_birth_5u_seed1\\final_v6i23_population_birth_5u_seed1_2v2.zip"
    uv run python experiments/run_v6i24_full_policy_population.py `
      --checkpoint $anchor `
      --checkpoint-mode shared-core `
      --output-dir artifacts\\v6i24_population_seed1 `
      --max-probe 5 --seed 1
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
    DEFAULT_MAPS,
    DEFAULT_OPPONENTS,
    PROBE_UPDATES,
    STEPS_PER_UPDATE,
    build_member_pressures,
    pressures_manifest,
)
from experiments.v6i24_shared_core import (  # noqa: E402
    find_newest_competent_zip,
    materialize_shared_core_member_checkpoint,
)

DEFAULT_ANCHOR = (
    PROJECT_ROOT
    / "checkpoints"
    / "2v2"
    / "final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip"
)
DEFAULT_V6I23_DONOR = (
    PROJECT_ROOT
    / "artifacts"
    / "v6i23_population_birth_5u_seed1"
    / "final_v6i23_population_birth_5u_seed1_2v2.zip"
)

INIT_WR_SPREAD_MAX = 0.05
INIT_JSD_MAX = 1e-3
INIT_MEAN_WR_MIN = 0.35


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I24 lean full-policy population diagnostic")
    p.add_argument(
        "--checkpoint",
        "--anchor",
        dest="checkpoint",
        default=str(DEFAULT_ANCHOR),
        help="Source checkpoint (.zip). Prefer V6I9/V6I21J; shared-core may use V6I23/V6I22E.",
    )
    p.add_argument(
        "--checkpoint-mode",
        choices=("auto", "full", "shared-core", "fresh"),
        default="auto",
        help=(
            "auto: full if clean V6I9 path exists else shared-core from donor; "
            "full: load checkpoint as-is; shared-core: extract shared trunk; "
            "fresh: identical random init."
        ),
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
    p.add_argument(
        "--skip-init-gate",
        action="store_true",
        help="Skip update-0 competence / identity gate (not recommended).",
    )
    p.add_argument(
        "--init-episodes-per-cell",
        type=int,
        default=8,
        help="Episodes/cell for update-0 competence smoke (default 8).",
    )
    return p.parse_args()


def _looks_like_clean_v6i9_anchor(path: Path) -> bool:
    name = path.name.lower()
    return "v6i9" in name and "generalist" in name


def _resolve_checkpoint_mode_and_source(
    *,
    requested_mode: str,
    checkpoint: Path,
) -> tuple[str, Path | None, str]:
    """Return (mode, source_path_or_none, message)."""
    mode = str(requested_mode).strip().lower()
    if mode == "fresh":
        return "fresh", None, "Using identical fresh random initialization for all members."

    if mode == "full":
        if not checkpoint.is_file():
            raise FileNotFoundError(f"full mode requires an existing checkpoint: {checkpoint}")
        return "full", checkpoint, f"Loading checkpoint as-is: {checkpoint}"

    if mode == "shared-core":
        src = checkpoint if checkpoint.is_file() else None
        if src is None:
            src = DEFAULT_V6I23_DONOR if DEFAULT_V6I23_DONOR.is_file() else None
        if src is None:
            src = find_newest_competent_zip(
                [PROJECT_ROOT / "artifacts", PROJECT_ROOT / "checkpoints"]
            )
        if src is None:
            raise FileNotFoundError(
                "shared-core mode needs a donor zip (V6I23/V6I22E/…) or pass --checkpoint-mode fresh"
            )
        msg = (
            "No clean V6I9/V6I21J anchor was found.\n"
            "Using shared-core extraction from the supplied competent checkpoint.\n"
            "Latent adapters, routers, and per-z heads will not be loaded.\n"
            f"Donor: {src}"
        )
        if _looks_like_clean_v6i9_anchor(src):
            msg = (
                f"Shared-core extraction from {src} "
                "(adapters/per-z heads/router discarded if present)."
            )
        return "shared-core", src, msg

    # auto
    if checkpoint.is_file() and _looks_like_clean_v6i9_anchor(checkpoint):
        return "full", checkpoint, f"Auto: clean V6I9/V6I21J anchor found: {checkpoint}"
    if checkpoint.is_file():
        return (
            "shared-core",
            checkpoint,
            "No clean V6I9/V6I21J anchor was found.\n"
            "Using shared-core extraction from the supplied competent checkpoint.\n"
            "Latent adapters, routers, and per-z heads will not be loaded.\n"
            f"Donor: {checkpoint}",
        )
    donor = DEFAULT_V6I23_DONOR if DEFAULT_V6I23_DONOR.is_file() else None
    if donor is None:
        donor = find_newest_competent_zip(
            [PROJECT_ROOT / "artifacts", PROJECT_ROOT / "checkpoints"]
        )
    if donor is not None:
        return (
            "shared-core",
            donor,
            "No clean V6I9/V6I21J anchor was found.\n"
            "Using shared-core extraction from the supplied competent checkpoint.\n"
            "Latent adapters, routers, and per-z heads will not be loaded.\n"
            f"Donor: {donor}",
        )
    return (
        "fresh",
        None,
        "No trained checkpoint found anywhere under artifacts/checkpoints; "
        "falling back to --checkpoint-mode fresh.",
    )


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
    cfg = PRESET_REGISTRY["v6i24"](PPOConfig())
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


def _state_dict_max_abs_diff(path_a: Path, path_b: Path) -> float:
    import torch

    a = torch.load(path_a, map_location="cpu", weights_only=False)
    b = torch.load(path_b, map_location="cpu", weights_only=False)
    sa = a["model_state_dict"]
    sb = b["model_state_dict"]
    keys = sorted(set(sa) & set(sb))
    max_diff = 0.0
    for k in keys:
        if hasattr(sa[k], "float"):
            d = float((sa[k].float() - sb[k].float()).abs().max().item())
            if d > max_diff:
                max_diff = d
    return max_diff


def run_init_competence_gate(
    init_dir: Path,
    *,
    seed: int,
    device: str,
    episodes_per_cell: int,
) -> dict[str, Any]:
    """Update-0 gate: identical members + non-collapsed hardpool competence."""
    from experiments.run_v6i24_population_eval_gates import (
        collect_payoff_and_features,
        counterfactual_action_jsd,
        find_member_checkpoints,
        _load_policies,
        _make_env,
    )

    members = find_member_checkpoints(init_dir)
    if len(members) < 2:
        raise RuntimeError(f"init gate needs >=2 member zips in {init_dir}")

    # Exact weight identity (all members are copies of one shared-core init).
    ref = members[0][2]
    weight_spreads = []
    for _, _, path in members[1:]:
        weight_spreads.append(_state_dict_max_abs_diff(ref, path))
    max_weight_diff = float(max(weight_spreads) if weight_spreads else 0.0)

    env0 = _make_env(ref, list(DEFAULT_MAPS)[0], seed, device, 240)
    try:
        policies = _load_policies(members, env0.observation_space, env0.action_space, device)
    finally:
        env0.close()

    # Smoke cells: keep init gate cheap.
    smoke_opponents = ["OP8", "OP11", "OP12"]
    smoke_maps = list(DEFAULT_MAPS)
    collected = collect_payoff_and_features(
        policies,
        opponents=smoke_opponents,
        maps=smoke_maps,
        episodes_per_cell=int(episodes_per_cell),
        base_seed=seed,
        device=device,
        max_decision_steps=240,
    )
    wr = collected["winrate_matrix"]
    member_mean_wr = wr.mean(axis=1)
    wr_spread = float(member_mean_wr.max() - member_mean_wr.min())
    mean_wr = float(member_mean_wr.mean())

    jsd = counterfactual_action_jsd(
        policies,
        opponents=smoke_opponents,
        maps=smoke_maps,
        steps_per_cell=64,
        base_seed=seed + 123,
        device=device,
        max_decision_steps=240,
    )
    cell_jsds = [
        c["pair_jsd_mean"]
        for c in jsd.get("cells", [])
        if c.get("pair_jsd_mean") == c.get("pair_jsd_mean")
    ]
    mean_jsd = float(sum(cell_jsds) / len(cell_jsds)) if cell_jsds else float("nan")

    gate_weights = max_weight_diff <= 1e-6
    gate_wr_spread = wr_spread <= INIT_WR_SPREAD_MAX
    gate_jsd = mean_jsd == mean_jsd and mean_jsd <= INIT_JSD_MAX
    gate_competence = mean_wr >= INIT_MEAN_WR_MIN
    passed = gate_weights and gate_wr_spread and gate_jsd and gate_competence

    result = {
        "max_weight_diff": max_weight_diff,
        "member_mean_wr": member_mean_wr.tolist(),
        "wr_spread": wr_spread,
        "mean_wr": mean_wr,
        "mean_pairwise_jsd": mean_jsd,
        "gate_identical_weights": gate_weights,
        "gate_wr_spread": gate_wr_spread,
        "gate_jsd_near_zero": gate_jsd,
        "gate_stage_c_competence": gate_competence,
        "passed": passed,
        "thresholds": {
            "wr_spread_max": INIT_WR_SPREAD_MAX,
            "jsd_max": INIT_JSD_MAX,
            "mean_wr_min": INIT_MEAN_WR_MIN,
        },
        "action_jsd": jsd,
    }
    out = init_dir / "init_competence_gate.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("[v6i24 init-gate] weight max|diff|=", max_weight_diff)
    print(f"[v6i24 init-gate] mean WR={mean_wr:.3f} spread={wr_spread:.3f}")
    print(f"[v6i24 init-gate] mean action JSD={mean_jsd:.6f}")
    print(f"[v6i24 init-gate] PASS={passed}")
    return result


def _prepare_member_inits(
    *,
    mode: str,
    source: Path | None,
    pressures: list[Any],
    output_dir: Path,
    seed: int,
) -> tuple[dict[int, Path], dict[str, Any]]:
    init_dir = output_dir / "init"
    init_dir.mkdir(parents=True, exist_ok=True)
    template = init_dir / "_shared_core_template.zip"
    report: dict[str, Any] = {"mode": mode}

    if mode == "full":
        assert source is not None
        shutil.copy2(source, template)
        report["note"] = "full checkpoint copied as template (no shared-core filter)"
    else:
        mat = materialize_shared_core_member_checkpoint(
            source_checkpoint=source,
            output_path=template,
            seed=int(seed),
            mode=mode,
        )
        report.update(mat.report)

    member_inits: dict[int, Path] = {}
    for pressure in pressures:
        dest = _copy_member_artifact(template, init_dir, pressure.member_id, pressure.label)
        member_inits[pressure.member_id] = dest
    report["template"] = str(template)
    report["members"] = {str(k): str(v) for k, v in member_inits.items()}
    (init_dir / "shared_core_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return member_inits, report


def main() -> int:
    args = _parse_args()
    ckpt = Path(args.checkpoint)
    calib = Path(args.calibration_report)
    if not calib.is_file():
        print(f"ERROR: calibration report not found: {calib}")
        return 2

    try:
        mode, source, mode_msg = _resolve_checkpoint_mode_and_source(
            requested_mode=args.checkpoint_mode,
            checkpoint=ckpt,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2

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
        "checkpoint_mode": mode,
        "source_checkpoint": str(source.resolve()) if source is not None else None,
        "requested_checkpoint": str(ckpt),
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

    print("=" * 72)
    print("V6I24 Path C: lean independent full-policy population")
    print("=" * 72)
    print(mode_msg)
    print(f"Mode:       {mode}")
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

    print("[v6i24] materializing identical member inits ...", flush=True)
    member_inits, init_report = _prepare_member_inits(
        mode=mode,
        source=source,
        pressures=pressures,
        output_dir=output_dir,
        seed=int(args.seed),
    )
    manifest["init"] = init_report
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.skip_init_gate:
        print("[v6i24] running update-0 competence / identity gate ...", flush=True)
        try:
            gate = run_init_competence_gate(
                output_dir / "init",
                seed=int(args.seed),
                device=str(args.device),
                episodes_per_cell=int(args.init_episodes_per_cell),
            )
        except Exception as exc:
            print(f"ERROR: init competence gate failed: {exc}")
            return 1
        manifest["init_competence_gate"] = gate
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if not gate.get("passed"):
            print("ERROR: update-0 competence gate FAILED; refusing to train.")
            print("Check init/init_competence_gate.json")
            return 3

    probes = [u for u in PROBE_UPDATES if u <= int(args.max_probe)]
    prev_u = 0
    member_ckpts: dict[int, Path] = dict(member_inits)

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
                load_path = member_inits[mid]
                load_weights_only = True  # fresh optimizer; shared-core/full init has no opt state
            else:
                load_path = member_ckpts[mid]
                load_weights_only = False
            # Independent RNG streams per member; weights start identical.
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
                "init_checkpoint": str(member_inits[mid]),
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
