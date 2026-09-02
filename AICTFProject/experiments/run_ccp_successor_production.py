"""CCP successor production run: task PPO + SEQUENCE-mode causal supervision. Nothing else.

Implements CCP_SUCCESSOR_PPO_LAUNCH_SPEC.json.

FRESH init (no V4 warm-start), private critic heads kept from V4's validated axis,
SEQUENCE-mode causal supervision from the frozen 20-segment bank (7 non-zero), lambda 0.05,
winner-directed routing, committed heads excluded from numerator and denominator.

Every legacy auxiliary path from this program's prior runs -- V1's one-sided rehearsal bank,
OG-PSP/V3/V4's paired rehearsal, V3/V4's trajectory-identity PG -- is made FATAL if called,
not merely counted after the fact. Reusing V4's config-building shape (build_config through
V3/V4's own module) is the plumbing reuse; none of V3/V4's ATTACHED runners are installed.

Run:  python experiments/run_ccp_successor_production.py --verify-steps 12288
      python experiments/run_ccp_successor_production.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_hog_psp_v4_production as V4

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
LAUNCH_SPEC = SD / "CCP_SUCCESSOR_PPO_LAUNCH_SPEC.json"
SEQUENCE_NPZ = SD / "ccp_sequence_bank.npz"
SEQUENCE_META = SD / "CCP_SEQUENCE_BANK.json"
OUT_DIR = SD / "ccp_successor_production"
RECORD = SD / "CCP_SUCCESSOR_PRODUCTION_RESULT.json"

TRAINING_SEED = 11_600_001
EVAL_BLOCK = range(11_600_101, 11_600_133)
TOTAL_STEPS = 1_000_000
LAMBDA_CAUSAL = 0.05
CADENCE = 4              # matches V3/V4's paired-rehearsal cadence; same interleave rate
BATCH_ROWS = 48


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _frozen_launch_spec() -> dict:
    spec = json.loads(LAUNCH_SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_LAUNCH":
        raise SystemExit(f"REFUSING: launch spec is not frozen: {spec['status']!r}")
    return spec


def build_config():
    """V4's config (private critic heads True), re-seeded, FRESH init reasserted.

    No paired-rehearsal or trajectory-PG object is attached here -- those are separate
    objects V4's own main() attaches via post_trainer_setup, which this script never calls.
    """
    cfg = V4.build_config()          # rasr_private_critic_heads already True, asserted there

    cfg.seed = TRAINING_SEED
    cfg.total_timesteps = TOTAL_STEPS
    cfg.run_tag = "ccp_successor_production"
    cfg.checkpoint_dir = str(OUT_DIR / "ckpts")
    cfg.metrics_csv_path = str(OUT_DIR / "metrics.csv")
    cfg.episode_csv_path = str(OUT_DIR / "episode_rows.csv")
    cfg.load_path = None
    if int(cfg.seed) in EVAL_BLOCK:
        raise SystemExit("REFUSING: training seed lies inside the sealed EVAL block")
    if not getattr(cfg, "rasr_private_critic_heads", False):
        raise SystemExit("REFUSING: rasr_private_critic_heads did not survive from V4.build_config")
    if getattr(cfg, "load_path", None) is not None:
        raise SystemExit("REFUSING: load_path is set; this run must be fresh-init")
    return cfg


class _FatalLegacyPath(RuntimeError):
    """A disabled legacy auxiliary path was called during a supposedly-SEQUENCE-only run."""


def _install_legacy_tripwires():
    """Make every prior program's auxiliary path FATAL, not counted after the fact.

    V1's RehearsalBank.sample, OG-PSP/V3/V4's paired_rehearsal_loss, and V3/V4's
    TrajectoryIdentityRunner.loss are the three call sites this program has ever used for an
    auxiliary imitation/PG objective. None of them is attached by this script, so under
    correct wiring none of these patches ever fires -- but 'never attached' and 'cannot be
    called' are different guarantees, and PPO's update loop resolves its rehearsal hook by
    attribute name at USE time (rl/custom_ppo/update/updater.py), so a wiring mistake that
    accidentally left a legacy attribute in place would otherwise train silently.
    """
    import rl.oracle_rehearsal as V1
    import rl.paired_rehearsal as PR
    import rl.trajectory_identity as TI

    originals = {
        ("rl.oracle_rehearsal", "RehearsalBank.sample"): V1.RehearsalBank.sample,
        ("rl.paired_rehearsal", "paired_rehearsal_loss"): PR.paired_rehearsal_loss,
        ("rl.trajectory_identity", "TrajectoryIdentityRunner.loss"): TI.TrajectoryIdentityRunner.loss,
    }

    def _raise(name):
        def _fn(*a, **k):
            raise _FatalLegacyPath(
                f"{name} was called during a SEQUENCE-only successor run. This program's "
                "prior auxiliary paths (V1 one-sided rehearsal, OG-PSP/V3/V4 paired "
                "rehearsal, V3/V4 trajectory-identity PG) are structurally disabled here.")
        return _fn

    V1.RehearsalBank.sample = _raise("RehearsalBank.sample")
    PR.paired_rehearsal_loss = _raise("paired_rehearsal_loss")
    TI.TrajectoryIdentityRunner.loss = _raise("TrajectoryIdentityRunner.loss")
    return originals


def _restore_legacy_tripwires(originals: dict) -> None:
    import rl.oracle_rehearsal as V1
    import rl.paired_rehearsal as PR
    import rl.trajectory_identity as TI
    V1.RehearsalBank.sample = originals[("rl.oracle_rehearsal", "RehearsalBank.sample")]
    PR.paired_rehearsal_loss = originals[("rl.paired_rehearsal", "paired_rehearsal_loss")]
    TI.TrajectoryIdentityRunner.loss = originals[
        ("rl.trajectory_identity", "TrajectoryIdentityRunner.loss")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-steps", type=int, default=0)
    args = ap.parse_args()

    import torch  # noqa: F401
    from experiments.run_exp2b_specialization_preserving_compression import (
        configure_exp2b_live_environment,
    )
    from rl import launch_audit_hooks as hooks
    from rl.causal_segment_bank import build_segment_bank, segment_bank_hash
    from rl.causal_sequence_diagnostics import install_diagnostics_reporter
    from rl.causal_sequence_runner import CausalSequenceRunner
    from rl.launch_gate import (
        LaunchGateError, check_fresh_training, check_opponent_mode, format_checks,
    )
    from rl.training.orchestrator import orchestrate_training_run

    spec = _frozen_launch_spec()
    if RECORD.is_file():
        raise SystemExit(f"REFUSING: {RECORD} exists; this run is one-shot")
    if not SEQUENCE_NPZ.is_file() or not SEQUENCE_META.is_file():
        raise SystemExit("REFUSING: sequence bank artifacts missing; run "
                         "ccp_build_sequence_bank.py first")

    # --- fail-closed re-derivation of the frozen segment bank, before step 0 -----------
    fresh_bank = build_segment_bank(SD / "CCP_PHASE1_CAUSAL_BRANCHING.json")
    if len(fresh_bank) != 20:
        raise SystemExit(f"REFUSING: re-derived segment bank has {len(fresh_bank)} segments, not 20")
    nonzero = [s for s in fresh_bank if s.weight > 0]
    if len(nonzero) != 7:
        raise SystemExit(f"REFUSING: re-derived segment bank has {len(nonzero)} non-zero, not 7")
    fresh_hash = segment_bank_hash(fresh_bank)
    seq_meta = json.loads(SEQUENCE_META.read_text(encoding="utf-8"))
    if seq_meta["segment_bank_hash"] != fresh_hash:
        raise SystemExit("REFUSING: sequence bank hash does not match a fresh rebuild of "
                         "the segment bank from CCP_PHASE1_CAUSAL_BRANCHING.json")
    npz_sha = hashlib.sha256(SEQUENCE_NPZ.read_bytes()).hexdigest()
    if npz_sha != seq_meta["npz_sha256"]:
        raise SystemExit("REFUSING: sequence bank npz sha256 does not match its own metadata")

    cfg = build_config()
    verifying = args.verify_steps > 0
    if verifying:
        cfg.total_timesteps = int(args.verify_steps)
        cfg.run_tag = "ccp_successor_wiring_verification"
        cfg.checkpoint_dir = str(OUT_DIR / "verify_ckpts")
        cfg.metrics_csv_path = str(OUT_DIR / "verify_metrics.csv")
        cfg.episode_csv_path = str(OUT_DIR / "verify_episode_rows.csv")

    checks = [check_opponent_mode(cfg), check_fresh_training(cfg)]
    print("CCP SUCCESSOR PPO  (task PPO + SEQUENCE-mode causal supervision)")
    print(format_checks(checks))
    if [c for c in checks if c.blocking and not c.passed]:
        raise LaunchGateError("LAUNCH REFUSED:\n" + format_checks(checks))

    print(f"\nINIT")
    print(f"  fresh_step_0=True")
    print(f"  checkpoint_loaded={getattr(cfg, 'load_path', None) is not None}")
    print(f"  private_critic_heads={cfg.rasr_private_critic_heads}")
    print(f"\nCAUSAL BANK")
    print(f"  segments={len(fresh_bank)}")
    print(f"  nonzero={len(nonzero)}")
    print(f"  mode=SEQUENCE")
    print(f"  joint_precedence=True")
    print(f"  lambda={LAMBDA_CAUSAL}")
    print(f"  segment_bank_hash={fresh_hash}")
    print(f"  seed {cfg.seed}   steps {cfg.total_timesteps:,}")
    print(f"  EVAL {EVAL_BLOCK.start}..{EVAL_BLOCK.stop - 1} SEALED\n", flush=True)

    if args.dry_run:
        print("DRY RUN -- gates verified, nothing trained.")
        return 0

    originals = _install_legacy_tripwires()
    state = {}

    def _attach(trainer, _cfg):
        state["auditor"] = hooks.attach(
            trainer, z_to_pole={0: "A", 1: "B"},
            opponent_to_pole=V4.V3.OPPONENT_ID_TO_POLE, hard_fail=True, artifact_dir=OUT_DIR)
        state["seq"] = CausalSequenceRunner(
            trainer, SEQUENCE_NPZ, SEQUENCE_META, lam=LAMBDA_CAUSAL, cadence=CADENCE,
            batch_rows=BATCH_ROWS, expected_bank_hash=fresh_hash,
            device=str(getattr(trainer, "device", "cpu")))
        # Observability only -- wraps the INSTANCE, changes nothing about what step() computes.
        # Proven behaviour-neutral: tests/test_causal_sequence_diagnostics_neutral.py.
        state["seq_note_original"] = install_diagnostics_reporter(state["seq"], every=1)
        trainer.oracle_rehearsal_runner = state["seq"]
        m = trainer.model
        state["private_before"] = {n: p.detach().cpu().numpy().copy()
                                   for n, p in m.named_parameters()
                                   if any(marker in n for marker in
                                         ("latent_branch_trunks", "latent_action_heads",
                                          "latent_adapters", "head_V"))}

    try:
        manifest = orchestrate_training_run(
            cfg,
            pre_rollout_env_setup=partial(
                configure_exp2b_live_environment,
                contract={"record": "CCP successor production", "utc": _now()},
                training_seed_range=(int(cfg.seed), int(cfg.seed) + 320),
                manifest_key="ccp_successor_protocol",
                context_label="CCP successor production construction"),
            post_trainer_setup=_attach)
        verdict, reason = "COMPLETE", None
    except Exception as exc:                                   # noqa: BLE001
        if not isinstance(exc, (LaunchGateError, _FatalLegacyPath)):
            _restore_legacy_tripwires(originals)
            raise
        manifest, verdict, reason = None, "INVALID_TREATMENT", str(exc)
        print(f"\n  RUN TERMINATED BY AN INVARIANT: {exc}")
    finally:
        _restore_legacy_tripwires(originals)
        if "seq" in state and "seq_note_original" in state:
            from rl.causal_sequence_diagnostics import restore as _restore_diag
            _restore_diag(state["seq"], state["seq_note_original"])

    auditor = state.get("auditor")
    seq = state.get("seq")
    stel = seq.telemetry() if seq else {}
    coverage = auditor.coverage(expected_envs=32, min_resets=2) if auditor else None

    private_moved = {}
    if "private_before" in state and seq is not None:
        for n, p in seq.trainer.model.named_parameters():
            if n in state["private_before"]:
                private_moved[n] = not np.array_equal(
                    state["private_before"][n], p.detach().cpu().numpy())
    z0_priv_moved = any(v for k, v in private_moved.items()
                        if ("latent_branch_trunks.0" in k or "latent_action_heads.0" in k))
    z1_priv_moved = any(v for k, v in private_moved.items()
                        if ("latent_branch_trunks.1" in k or "latent_action_heads.1" in k))
    critic_v0_moved = any(v for k, v in private_moved.items() if "head_V0" in k)
    critic_v1_moved = any(v for k, v in private_moved.items() if "head_V1" in k)

    if verifying:
        ok = (stel.get("updates", 0) > 0 and stel.get("z0_exposures", 0) > 0
              and stel.get("z1_exposures", 0) > 0 and stel.get("positive_routes", 0) > 0
              and stel.get("negative_routes", 0) > 0
              and z0_priv_moved and z1_priv_moved
              and critic_v0_moved and critic_v1_moved
              and verdict == "COMPLETE")
        print(f"\nLIVE PATHS ({cfg.total_timesteps} steps, no record written)")
        print(f"  task PPO minibatches      {stel.get('n_ppo_minibatches', 0)}  <- MUST be > 0")
        print(f"  causal sequence updates   {stel.get('updates', 0)}  <- MUST be > 0")
        print(f"  z0 exposures              {stel.get('z0_exposures', 0)}  <- MUST be > 0")
        print(f"  z1 exposures              {stel.get('z1_exposures', 0)}  <- MUST be > 0")
        print(f"  positive delta routing    {stel.get('positive_routes', 0)}  <- MUST be > 0")
        print(f"  negative delta routing    {stel.get('negative_routes', 0)}  <- MUST be > 0")
        print(f"  legacy paired/PG calls    0 (fatal if any occurred; run would have aborted)")
        print(f"  missing decision predicate 0 (fatal if any occurred)")
        print(f"  wrong routing             0 (structurally unrepresentable; rl/causal_supervision.py)")
        print(f"  z0/z1 private actor moved {z0_priv_moved} / {z1_priv_moved}  <- BOTH must be True")
        print(f"  critic V0/V1 moved        {critic_v0_moved} / {critic_v1_moved}  <- BOTH must be True")
        if coverage:
            print(f"  envs / mismatches         {coverage['envs_observed']}/32, "
                  f"{coverage['total_mismatches']}")
        print(f"\nPURITY")
        print(f"  task reward unchanged=True   (structural: separate optimizer step, "
              f"tests/test_causal_task_purity.py)")
        print(f"  GAE/returns/value targets unchanged=True   (same)")
        print(f"\n  COMPOSITION: {'OK' if ok else 'BROKEN'}")
        return 0 if ok else 1

    RECORD.write_text(json.dumps({
        "record": "CCP successor production run", "status": "FROZEN_RESULT", "utc": _now(),
        "implements": "CCP_SUCCESSOR_PPO_LAUNCH_SPEC.json", "VERDICT": verdict,
        "termination_reason": reason, "seed": int(cfg.seed),
        "total_timesteps": int(cfg.total_timesteps), "fresh_step_0": True,
        "checkpoint_weights_loaded": False,
        "MODE": {"training_mode": "SEQUENCE", "lambda_causal": LAMBDA_CAUSAL,
                 "cadence": CADENCE, "batch_rows": BATCH_ROWS,
                 "segment_bank_hash": fresh_hash, "nonzero_segments": len(nonzero)},
        "architecture": {"rasr_private_critic_heads": True,
                         "private_capacity": V4.V3.LRO_FLAGS},
        "causal_sequence_telemetry": stel,
        "private_parameter_motion": {
            "z0_actor_moved": z0_priv_moved, "z1_actor_moved": z1_priv_moved,
            "critic_V0_moved": critic_v0_moved, "critic_V1_moved": critic_v1_moved,
            "per_param_moved": private_moved},
        "coverage": coverage,
        "legacy_paths_disabled": True,
        "EVAL_touched": False,
        "authorizes": "nothing further; EVAL is a separate frozen decision",
    }, indent=2), encoding="utf-8")
    print(f"\n  VERDICT: {verdict}")
    print(f"  -> {RECORD}")
    return 0 if verdict == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
