#!/usr/bin/env python3
"""Validate env-reuse optimization: fresh env per z vs reused env across z.

Scientifically narrow comparison:
  - Same immutable policy object (loaded once)
  - Same seeds, opponent, map, horizon, telemetry, deterministic actions
  - Only difference: fresh env per z vs reused env with hard resets between z

Phase 1: fresh_per_z vs reuse_block
Phase 2 (if phase 1 passes): latent order z0..z3 vs z3..z0 on reuse_block

Episode workload per phase (default 10 eps):
  2 modes × 4 latents × 10 episodes = 80 episodes (phase 1)
  2 orders × 4 latents × 10 episodes = 80 episodes (phase 2, optional)

Usage::

    uv run python experiments/validate_forced_z_env_reuse.py \\
        --checkpoint checkpoints/2v2/final_v6i9-...zip \\
        --episodes 10 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.forced_z_eval.equivalence import (  # noqa: E402
    STRICT_FLOAT_ATOL,
    STRICT_FLOAT_RTOL,
    annotate_expected_seeds,
    compare_forced_z_cells,
    decision_tree_hint,
)
from experiments.forced_z_eval.protocol import DEFAULT_BASE_SEED, DEFAULT_LATENTS, ForcedZProtocol, audit_protocol_note
from experiments.forced_z_eval.runner import load_shared_policy, run_forced_z_episodes  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate forced-z env reuse equivalence")
    p.add_argument(
        "--checkpoint",
        default="checkpoints/2v2/final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip",
    )
    p.add_argument("--opponent", default="OP8")
    p.add_argument("--map", default="map_b")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    p.add_argument("--skip-order-probe", action="store_true")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        import plot.eval_rollout  # noqa: F401
    except ImportError as exc:
        print(f"ERROR: {exc}")
        return 1

    protocol = ForcedZProtocol(
        checkpoint=str(args.checkpoint),
        opponents=(str(args.opponent),),
        maps=(str(args.map),),
        latents=DEFAULT_LATENTS,
        episodes_per_cell=int(args.episodes),
        base_seed=int(args.base_seed),
        deterministic_actions=True,
        device=str(args.device),
        collect_behavior_mean=True,
        progress_every=0,
    )
    cell_seed = protocol.cell_seed(0, 0)
    total_phase1 = 2 * len(DEFAULT_LATENTS) * int(args.episodes)

    print("=" * 60)
    print("Forced-z env reuse equivalence check")
    print("=" * 60)
    print(audit_protocol_note())
    print(f"Tolerance  : atol={STRICT_FLOAT_ATOL}, rtol={STRICT_FLOAT_RTOL}")
    print(f"Checkpoint : {protocol.checkpoint}")
    print(f"Grid       : {args.opponent} × {args.map} × z0..z3 × {args.episodes} eps")
    print(f"Phase 1 workload: {total_phase1} episodes (2 modes × 4 z × {args.episodes})")
    print(f"Policy     : single load, shared immutable object across both modes")
    print()

    print("--- Loading shared policy once ---")
    shared_model = load_shared_policy(protocol, map_name=str(args.map), cell_seed=cell_seed)
    print("Policy loaded.")
    print()

    print("--- Phase 1a: fresh env per z (reference) ---")
    fresh = run_forced_z_episodes(protocol, env_mode="fresh_per_z", shared_model=shared_model, quiet=True)
    print("--- Phase 1b: reuse env across z (optimized) ---")
    reused = run_forced_z_episodes(protocol, env_mode="reuse_block", shared_model=shared_model, quiet=True)

    annotate_expected_seeds(fresh, protocol)
    annotate_expected_seeds(reused, protocol)

    phase1 = compare_forced_z_cells(
        fresh,
        reused,
        opponents=list(protocol.opponents),
        maps=list(protocol.maps),
        latents=tuple(protocol.latents),
        comparison="fresh_vs_reuse",
    )
    print()
    print("=== Phase 1: fresh_per_z vs reuse_block ===")
    print(phase1.summary())

    phase2 = None
    if phase1.passed and not args.skip_order_probe:
        total_phase2 = 2 * len(DEFAULT_LATENTS) * int(args.episodes)
        print()
        print(f"--- Phase 2: order-independence probe ({total_phase2} episodes) ---")
        order_fwd = tuple(DEFAULT_LATENTS)
        order_rev = tuple(reversed(DEFAULT_LATENTS))
        fwd = run_forced_z_episodes(
            protocol, env_mode="reuse_block", shared_model=shared_model, latent_order=order_fwd, quiet=True
        )
        rev = run_forced_z_episodes(
            protocol, env_mode="reuse_block", shared_model=shared_model, latent_order=order_rev, quiet=True
        )
        annotate_expected_seeds(fwd, protocol)
        annotate_expected_seeds(rev, protocol)
        phase2 = compare_forced_z_cells(
            fwd,
            rev,
            opponents=list(protocol.opponents),
            maps=list(protocol.maps),
            latents=tuple(protocol.latents),
            comparison="order_fwd_vs_rev",
        )
        print()
        print("=== Phase 2: z0..z3 vs z3..z0 (reuse_block only) ===")
        print(phase2.summary())

    passed = phase1.passed and (phase2.passed if phase2 is not None else True)

    out_dir = Path(
        args.out_dir
        or (SCRIPT_DIR / "forced_z_runs" / f"equivalence_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "passed": passed,
        "reuse_block_approved": passed,
        "batched_eval_blocked_until_pass": not passed,
        "tolerance": {"atol": STRICT_FLOAT_ATOL, "rtol": STRICT_FLOAT_RTOL},
        "protocol": protocol.to_manifest(),
        "phase1": {
            "passed": phase1.passed,
            "episodes_compared": phase1.episodes_compared,
            "workload_episodes": total_phase1,
            "mismatch_count": len(phase1.mismatches),
            "decision": decision_tree_hint(phase1),
        },
        "phase2_order_probe": None
        if phase2 is None
        else {
            "passed": phase2.passed,
            "episodes_compared": phase2.episodes_compared,
            "mismatch_count": len(phase2.mismatches),
            "decision": decision_tree_hint(phase2),
        },
        "mismatches_phase1": [
            {
                "opponent": m.opponent,
                "map": m.map_name,
                "latent_z": m.latent_z,
                "episode_seed": m.episode_seed,
                "field": m.field,
                "fresh": m.left,
                "reused": m.right,
            }
            for m in phase1.mismatches
        ],
    }
    if phase2 is not None:
        payload["mismatches_phase2"] = [
            {
                "opponent": m.opponent,
                "map": m.map_name,
                "latent_z": m.latent_z,
                "episode_seed": m.episode_seed,
                "field": m.field,
                "fwd": m.left,
                "rev": m.right,
            }
            for m in phase2.mismatches
        ]
    (out_dir / "env_reuse_equivalence.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote: {out_dir / 'env_reuse_equivalence.json'}")
    if passed:
        print("\nVERDICT: reuse_block APPROVED — run unified 2,400-episode eval once via run_forced_z_eval.py")
    else:
        print("\nVERDICT: reuse_block NOT APPROVED — fix resets before unified eval or batching")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
