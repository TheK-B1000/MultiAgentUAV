#!/usr/bin/env python3
"""Compare the control vs treatment arms of the router-credit A/B experiment.

Reads both arm directories produced by ``run_ab_router_credit.py`` and:

1. Verifies the launch contract the operator requires:
     * ``initial_router_hash`` treatment == control
     * ``initial_frozen_actor_hash`` treatment == control
     * fresh optimizer in both arms
     * the *actual* advantage sources differ as intended
2. Builds the FULL per-update trajectory for each success signal (not just the
   final row) so centering / separation / MI can be read as a curve.
3. Emits a treatment success verdict.

This is analysis-only: it never launches training and has no torch/env
dependency. Point it at ``artifacts/ab_router_credit`` after both arms finish::

    python experiments/compare_ab_router_credit.py \\
      --ab-dir artifacts/ab_router_credit \\
      --out artifacts/ab_router_credit/comparison.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

# Signals tracked as per-update trajectories. Each maps to a telemetry key
# emitted by run_ab_router_credit.py.
_TRAJECTORY_SIGNALS = [
    "latent_arc_raw_advantage_mean",
    "latent_arc_positive_fraction",
    "latent_arc_raw_adv_z_spread",
    "latent_arc_raw_adv_mean_z0",
    "latent_arc_raw_adv_mean_z1",
    "latent_arc_raw_adv_mean_z2",
    "latent_arc_raw_adv_mean_z3",
    "router_selected_z_occupancy_z0",
    "router_selected_z_occupancy_z1",
    "router_selected_z_occupancy_z2",
    "router_selected_z_occupancy_z3",
    "router_selected_z_occupancy_max",
    "router_selected_z_unique_count",
    "q_phi_grad_norm",
    "strategy_entropy",
    "latent_mi_z_opponent_nats",
    "latent_mi_z_phase_nats",
    "latent_mi_z_outcome_nats",
    "latent_mi_z_flag_state_nats",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare A/B router-credit arms")
    p.add_argument("--ab-dir", default="artifacts/ab_router_credit")
    p.add_argument("--control-dir", default=None, help="Override control arm dir")
    p.add_argument("--treatment-dir", default=None, help="Override treatment arm dir")
    p.add_argument("--out", default=None, help="Comparison JSON output path")
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_arm(arm_dir: Path) -> dict[str, Any]:
    """Load an arm from summary.json, falling back to update_*.json + run_meta."""
    if not arm_dir.is_dir():
        raise FileNotFoundError(f"Arm directory not found: {arm_dir}")
    summary_path = arm_dir / "summary.json"
    if summary_path.is_file():
        summary = _load_json(summary_path)
        updates = summary.get("updates", [])
        meta = {k: v for k, v in summary.items() if k != "updates"}
        return {"meta": meta, "updates": updates, "complete": True}
    # Fallback: assemble from per-update files (run in progress / interrupted).
    meta = _load_json(arm_dir / "run_meta.json") if (arm_dir / "run_meta.json").is_file() else {}
    updates = [
        _load_json(p) for p in sorted(arm_dir.glob("update_*.json"))
    ]
    return {"meta": meta, "updates": updates, "complete": False}


def _trajectory(updates: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for rec in updates:
        tel = rec.get("telemetry", {})
        val = tel.get(key)
        try:
            f = float(val)
        except (TypeError, ValueError):
            f = float("nan")
        out.append(f)
    return out


def _finite_last(values: list[float]) -> float:
    for v in reversed(values):
        if math.isfinite(v):
            return v
    return float("nan")


def _z_spread_trajectory(updates: list[dict[str, Any]]) -> list[float]:
    """Per-update per-z spread, preferring the direct key and falling back to
    computing (max - min) from the per-z mean keys for artifacts collected
    before ``latent_arc_raw_adv_z_spread`` existed."""
    out: list[float] = []
    for rec in updates:
        tel = rec.get("telemetry", {})
        direct = tel.get("latent_arc_raw_adv_z_spread")
        try:
            f = float(direct)
        except (TypeError, ValueError):
            f = float("nan")
        if not math.isfinite(f):
            z_means = []
            for zi in range(4):
                v = tel.get(f"latent_arc_raw_adv_mean_z{zi}")
                cnt = tel.get(f"latent_arc_count_z{zi}", 0.0) or 0.0
                try:
                    vf = float(v)
                except (TypeError, ValueError):
                    vf = float("nan")
                if math.isfinite(vf) and float(cnt) > 0:
                    z_means.append(vf)
            f = float(max(z_means) - min(z_means)) if len(z_means) >= 2 else float("nan")
        out.append(f)
    return out


def _launch_contract(control_meta: dict, treat_meta: dict) -> dict[str, Any]:
    c_router = control_meta.get("initial_router_hash")
    t_router = treat_meta.get("initial_router_hash")
    c_frozen = control_meta.get("initial_frozen_actor_hash")
    t_frozen = treat_meta.get("initial_frozen_actor_hash")
    c_src = control_meta.get("advantage_source_cfg")
    t_src = treat_meta.get("advantage_source_cfg")
    checks = {
        "initial_router_hash_match": c_router is not None and c_router == t_router,
        "initial_frozen_actor_hash_match": c_frozen is not None and c_frozen == t_frozen,
        "control_fresh_optimizer": bool(control_meta.get("fresh_optimizer", False)),
        "treatment_fresh_optimizer": bool(treat_meta.get("fresh_optimizer", False)),
        "advantage_sources_differ": (c_src is not None and t_src is not None and c_src != t_src),
        "same_checkpoint": control_meta.get("checkpoint_sha256")
        == treat_meta.get("checkpoint_sha256"),
    }
    checks["all_passed"] = all(bool(v) for v in checks.values())
    return {
        "checks": checks,
        "control_advantage_source": c_src,
        "treatment_advantage_source": t_src,
        "control_initial_router_hash": c_router,
        "treatment_initial_router_hash": t_router,
    }


def _frozen_integrity(arm: dict) -> dict[str, Any]:
    meta = arm["meta"]
    report = meta.get("frozen_hash_report", {})
    return {
        "frozen_actor_z_unchanged": report.get("frozen_actor_z_unchanged"),
        "router_moved": report.get("router_moved"),
    }


def _treatment_verdict(treat: dict) -> dict[str, Any]:
    updates = treat["updates"]
    if not updates:
        return {"evaluable": False, "reason": "no treatment updates found"}

    raw_mean = _trajectory(updates, "latent_arc_raw_advantage_mean")
    pos_frac = _trajectory(updates, "latent_arc_positive_fraction")
    z_spread = _z_spread_trajectory(updates)
    q_phi = _trajectory(updates, "q_phi_grad_norm")
    occ_max = _trajectory(updates, "router_selected_z_occupancy_max")
    unique = _trajectory(updates, "router_selected_z_unique_count")

    last_raw = _finite_last(raw_mean)
    last_pos = _finite_last(pos_frac)
    last_spread = _finite_last(z_spread)
    last_qphi = _finite_last(q_phi)
    last_occ_max = _finite_last(occ_max)
    last_unique = _finite_last(unique)

    signals = {
        # Centering: raw advantage mean stays near zero.
        "raw_advantage_centered": math.isfinite(last_raw) and abs(last_raw) < 0.25,
        # Two-sided: positive fraction stays away from 0 and 1.
        "positive_fraction_two_sided": math.isfinite(last_pos) and 0.35 <= last_pos <= 0.65,
        # Router gradients remain active.
        "router_gradients_active": math.isfinite(last_qphi) and last_qphi > 0.0,
        # THE routing-teachability signal: per-z means actually separate.
        "per_z_means_separate": math.isfinite(last_spread) and last_spread > 0.05,
        # Router still uses more than one strategy at decision points.
        "router_not_collapsed": (
            math.isfinite(last_occ_max) and last_occ_max < 0.9 and last_unique >= 2
        ),
    }
    signals["routing_teachable"] = signals["per_z_means_separate"]
    signals["mechanism_healthy"] = (
        signals["raw_advantage_centered"]
        and signals["positive_fraction_two_sided"]
        and signals["router_gradients_active"]
    )
    return {
        "evaluable": True,
        "signals": signals,
        "final_raw_advantage_mean": last_raw,
        "final_positive_fraction": last_pos,
        "final_per_z_spread": last_spread,
        "final_q_phi_grad_norm": last_qphi,
        "final_selected_z_occupancy_max": last_occ_max,
        "final_selected_z_unique_count": last_unique,
        "note": (
            "per_z_means_separate is the gate that distinguishes 'grades repaired' "
            "from 'routing learned'. A centered, two-sided signal with z_spread ~ 0 "
            "gives every z the same expected credit and cannot teach routing. "
            "router_not_collapsed guards the failure where the router selects a "
            "single z (occupancy_max -> 1.0), which makes any routing gate vacuous."
        ),
    }


def main() -> None:
    args = _parse_args()
    ab_dir = Path(args.ab_dir)
    control_dir = Path(args.control_dir) if args.control_dir else ab_dir / "control"
    treatment_dir = Path(args.treatment_dir) if args.treatment_dir else ab_dir / "treatment"

    control = _load_arm(control_dir)
    treatment = _load_arm(treatment_dir)

    launch = _launch_contract(control["meta"], treatment["meta"])

    trajectories = {
        "control": {k: _trajectory(control["updates"], k) for k in _TRAJECTORY_SIGNALS},
        "treatment": {k: _trajectory(treatment["updates"], k) for k in _TRAJECTORY_SIGNALS},
    }
    # Backfill z_spread with the per-z fallback for older artifacts.
    trajectories["control"]["latent_arc_raw_adv_z_spread"] = _z_spread_trajectory(control["updates"])
    trajectories["treatment"]["latent_arc_raw_adv_z_spread"] = _z_spread_trajectory(treatment["updates"])

    report = {
        "control_dir": str(control_dir),
        "treatment_dir": str(treatment_dir),
        "control_complete": control["complete"],
        "treatment_complete": treatment["complete"],
        "control_router_type": control["meta"].get("router_type"),
        "treatment_router_type": treatment["meta"].get("router_type"),
        "launch_contract": launch,
        "control_frozen_integrity": _frozen_integrity(control),
        "treatment_frozen_integrity": _frozen_integrity(treatment),
        "treatment_verdict": _treatment_verdict(treatment),
        "trajectories": trajectories,
        "bias_sign_note": (
            "Telemetry reports ADVANTAGE bias (A = R - baseline), not critic bias. "
            "The critic's +2.705 value over-estimation maps to an ~ -2.705 advantage "
            "offset under A = R - V. latent_arc_raw_advantage_mean is the advantage "
            "quantity; a value near zero means the offset has been removed."
        ),
    }

    out_path = Path(args.out) if args.out else ab_dir / "comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 72)
    print("[ab-compare] Launch contract:")
    for check, ok in launch["checks"].items():
        print(f"    {'PASS' if ok else 'FAIL'}  {check}")
    print(f"    control  advantage source: {launch['control_advantage_source']}")
    print(f"    treatment advantage source: {launch['treatment_advantage_source']}")
    print("[ab-compare] Router types:")
    print(f"    control  : {report['control_router_type']}")
    print(f"    treatment: {report['treatment_router_type']}")
    verdict = report["treatment_verdict"]
    if verdict.get("evaluable"):
        print("[ab-compare] Treatment success signals:")
        for sig, ok in verdict["signals"].items():
            print(f"    {'YES ' if ok else 'no  '} {sig}")
        print(f"    final raw_adv_mean = {verdict['final_raw_advantage_mean']:.4f}")
        print(f"    final positive_fraction = {verdict['final_positive_fraction']:.4f}")
        print(f"    final per_z_spread = {verdict['final_per_z_spread']:.4f}")
        print(f"    final selected_z_occupancy_max = {verdict['final_selected_z_occupancy_max']:.4f}")
        print(f"    final selected_z_unique_count = {verdict['final_selected_z_unique_count']:.0f}")
    else:
        print(f"[ab-compare] Treatment not evaluable: {verdict.get('reason')}")
    print(f"[ab-compare] Wrote {out_path}")


if __name__ == "__main__":
    main()
