"""CCP-S2 production: matched CONTROL / TREATMENT warm-start fine-tuning.

Implements CCP_S2_SPEC.json#WARM_START_PROTOCOL + #CAUSAL_OBJECTIVE + #TRAINING_HORIZON.

    CONTROL    warm-start weights + task PPO only
    TREATMENT  same warm-start weights + same task PPO + lambda_causal * L_causal

Both arms load the SAME frozen incumbent weights with a FRESH optimizer
(cfg.load_weights_only=True -> optimizer_state_reused=false), the same training seed, the same
500k horizon, the same environment/opponent configuration. The ONLY difference is whether the
causal path is attached, and that asymmetry is enforced in both directions: TREATMENT attaches
CausalSequenceRunner, while CONTROL makes causal_supervision_loss itself FATAL, so "the control
arm had no causal supervision" is a structural guarantee rather than an assumption about wiring.

Warm start, not fresh init: rl/launch_gate.py's check_fresh_training REFUSES a load_path by
design and is therefore deliberately NOT used here. Its role is taken by an explicit warm-start
gate that requires load_path to be set AND to sha256-match CCP_SUCCESSOR_MODEL_FROZEN.json's
terminal checkpoint -- the same incumbent whose deployment distribution the causal bank was
measured on.

The training seed is NOT hardcoded. CCP_S2_SPEC.json leaves S2_training_seed explicitly "not
assigned", on the standing rule that seed/namespace choice is a PI decision requiring
mechanical preflight, never something a runner invents. This script reads it from a frozen
CCP_S2_TRAINING_SEED_ASSIGNMENT.json and refuses to launch without one.

Run:  python experiments/run_ccp_s2_production.py --arm control --verify-steps 12288
      python experiments/run_ccp_s2_production.py --arm control
      python experiments/run_ccp_s2_production.py --arm treatment
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
SPEC = SD / "CCP_S2_SPEC.json"
SEED_RECORD = SD / "CCP_S2_TRAINING_SEED_ASSIGNMENT.json"
INCUMBENT_FROZEN = SD / "CCP_SUCCESSOR_MODEL_FROZEN.json"
ROUTING = SD / "CCP_S2_CAUSAL_BANK_ROUTING.json"
STAGE_B = SD / "CCP_S2_CAUSAL_BANK_STAGE_B.json"
BANK_NPZ = SD / "ccp_s2_causal_bank.npz"
BANK_META = SD / "CCP_S2_CAUSAL_BANK_ARRAY.json"
OUT_ROOT = SD / "ccp_s2_production"

TOTAL_STEPS = 500_000                 # CCP_S2_SPEC.json#TRAINING_HORIZON, frozen, no extension
LAMBDA_CAUSAL = 0.05                  # frozen in FROZEN_MACHINERY_UNCHANGED
CADENCE = 4                           # inherited from the predecessor's validated interleave
BATCH_ROWS = 48                       # inherited; affects TREATMENT only, CONTROL has no path
EVAL_BLOCK = range(11_701_001, 11_701_065)      # CCP_S2_SEED_ASSIGNMENT.json, SEALED
COLLECTION_BLOCK = range(11_700_001, 11_700_321)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def training_seed() -> int:
    """Read the PI-assigned training seed. Never invented, never a CLI argument."""
    if not SEED_RECORD.is_file():
        raise SystemExit(
            f"REFUSING: {SEED_RECORD.name} does not exist.\n"
            "  CCP_S2_SPEC.json leaves S2_training_seed explicitly \"not assigned\", on this\n"
            "  program's standing rule that seed/namespace choice is a PI decision requiring\n"
            "  mechanical preflight -- not something this runner may invent. Freeze a training\n"
            "  seed (disjoint from the 11700001..11700320 collection block and the sealed\n"
            "  11701001..11701064 EVAL block) in that record before launching either arm.")
    rec = json.loads(SEED_RECORD.read_text(encoding="utf-8"))
    if rec.get("status") != "FROZEN_APPEND_ONLY_AMENDMENT":
        raise SystemExit(f"REFUSING: training seed record not frozen: {rec.get('status')!r}")
    seed = int(rec["training_seed"])
    if seed in EVAL_BLOCK:
        raise SystemExit("REFUSING: training seed lies inside the sealed EVAL block")
    if seed in COLLECTION_BLOCK:
        raise SystemExit("REFUSING: training seed lies inside the collection state-source block")
    return seed


def incumbent_checkpoint() -> Path:
    frozen = json.loads(INCUMBENT_FROZEN.read_text(encoding="utf-8"))
    ck = ROOT / frozen["TERMINAL_CHECKPOINT"]["path"]
    got = _sha(ck)
    if got != frozen["TERMINAL_CHECKPOINT"]["sha256"]:
        raise SystemExit(f"REFUSING: incumbent checkpoint sha mismatch\n  {got}\n  "
                         f"{frozen['TERMINAL_CHECKPOINT']['sha256']}")
    return ck


def build_config(arm: str, seed: int, ckpt: Path):
    """Identical for both arms except run_tag and output paths."""
    cfg = V4.build_config()
    cfg.seed = seed
    cfg.total_timesteps = TOTAL_STEPS
    cfg.run_tag = f"ccp_s2_{arm}"
    out = OUT_ROOT / arm
    cfg.checkpoint_dir = str(out / "ckpts")
    cfg.metrics_csv_path = str(out / "metrics.csv")
    cfg.episode_csv_path = str(out / "episode_rows.csv")
    cfg.load_path = str(ckpt)              # WARM START -- weights, not a resume
    cfg.load_weights_only = True           # optimizer_state_reused = false
    if not getattr(cfg, "rasr_private_critic_heads", False):
        raise SystemExit("REFUSING: rasr_private_critic_heads did not survive V4.build_config")
    return cfg


def check_warm_start(cfg, ckpt: Path):
    from rl.launch_gate import Check
    if not getattr(cfg, "load_path", None):
        return Check("warm_start", False, "load_path is not set; S2 is warm-start by protocol")
    if Path(cfg.load_path) != ckpt:
        return Check("warm_start", False, f"load_path {cfg.load_path} is not the frozen incumbent")
    if not getattr(cfg, "load_weights_only", False):
        return Check("warm_start", False,
                     "load_weights_only is False; WARM_START_PROTOCOL requires a fresh optimizer")
    return Check("warm_start", True,
                 f"frozen incumbent weights, fresh optimizer ({ckpt.name})")


class _FatalDisabledPath(RuntimeError):
    """A path that must not run in this arm was called."""


def _install_tripwires(arm: str) -> dict:
    """Legacy auxiliary paths are fatal in BOTH arms; the causal path is fatal in CONTROL.

    'Never attached' and 'cannot be called' are different guarantees. PPO resolves its
    rehearsal hook by attribute name at use time, so a wiring mistake could otherwise let an
    auxiliary objective train silently -- which in CONTROL would silently destroy the very
    contrast the experiment exists to measure.
    """
    import rl.causal_supervision as CS
    import rl.oracle_rehearsal as V1
    import rl.paired_rehearsal as PR
    import rl.trajectory_identity as TI

    originals = {
        "V1": V1.RehearsalBank.sample,
        "PR": PR.paired_rehearsal_loss,
        "TI": TI.TrajectoryIdentityRunner.loss,
        "CS": CS.causal_supervision_loss,
    }

    def _raise(name, why):
        def _fn(*a, **k):
            raise _FatalDisabledPath(f"{name} was called during a CCP-S2 {arm.upper()} run: {why}")
        return _fn

    legacy = "this program's prior auxiliary paths are structurally disabled in CCP-S2"
    V1.RehearsalBank.sample = _raise("RehearsalBank.sample", legacy)
    PR.paired_rehearsal_loss = _raise("paired_rehearsal_loss", legacy)
    TI.TrajectoryIdentityRunner.loss = _raise("TrajectoryIdentityRunner.loss", legacy)
    if arm == "control":
        CS.causal_supervision_loss = _raise(
            "causal_supervision_loss",
            "CONTROL is task PPO only; any causal gradient here would void the contrast")
    return originals


def _restore_tripwires(originals: dict) -> None:
    import rl.causal_supervision as CS
    import rl.oracle_rehearsal as V1
    import rl.paired_rehearsal as PR
    import rl.trajectory_identity as TI
    V1.RehearsalBank.sample = originals["V1"]
    PR.paired_rehearsal_loss = originals["PR"]
    TI.TrajectoryIdentityRunner.loss = originals["TI"]
    CS.causal_supervision_loss = originals["CS"]


def write_launch_manifest(arm: str, cfg, ckpt: Path, seed: int, bank_meta: dict,
                          fresh_hash: str, path: Path) -> dict:
    manifest = {
        "record": f"CCP-S2 {arm} launch manifest", "status": "FROZEN_BEFORE_LAUNCH",
        "utc": _now(), "arm": arm.upper(),
        "implements": "CCP_S2_SPEC.json#WARM_START_PROTOCOL + #CAUSAL_OBJECTIVE + "
                      "#TRAINING_HORIZON",
        "training_seed": seed,
        "total_timesteps": TOTAL_STEPS,
        "horizon_rules": ["no early stopping", "no extension after telemetry",
                          "no checkpoint selection", "terminal checkpoint only",
                          "no mid-run bank refresh"],
        "warm_start": {
            "checkpoint": str(ckpt.relative_to(ROOT)),
            "checkpoint_sha256": _sha(ckpt),
            "checkpoint_weights_loaded": True,
            "optimizer_state_reused": False,
            "same_weights_both_arms": True,
        },
        "stage_a_routing": {
            "artifact": "CCP_S2_CAUSAL_BANK_ROUTING.json", "sha256": _sha(ROUTING)},
        "stage_b_rollout": {
            "artifact": "CCP_S2_CAUSAL_BANK_STAGE_B.json", "sha256": _sha(STAGE_B)},
        "causal_bank_array": {
            "artifact": "CCP_S2_CAUSAL_BANK_ARRAY.json", "sha256": _sha(BANK_META),
            "npz_sha256": bank_meta["npz_sha256"], "segment_bank_hash": fresh_hash,
            "rows": bank_meta["n_rows"]},
        "config_diff_control_vs_treatment": {
            "identical": ["warm-start weights", "training seed", "total_timesteps",
                          "optimizer configuration", "rollout horizon",
                          "environment/opponent assignment", "task reward, returns, GAE",
                          "private critic heads", "number and timing of task PPO minibatches"],
            "differs": ("TREATMENT attaches CausalSequenceRunner "
                        f"(lambda={LAMBDA_CAUSAL}, cadence={CADENCE}, "
                        f"batch_rows={BATCH_ROWS}); CONTROL attaches nothing and makes "
                        "causal_supervision_loss fatal if called"),
            "only_difference_is_L_causal": True,
        },
        "documented_treatment_asymmetry_not_corrected": {
            "note": "logged prominently BEFORE launch so it cannot become a post-hoc "
                    "explanation. It did not alter this run: no reweighting, no added "
                    "anchors, no Stage A/B change.",
            "N_causal_anchors": bank_meta["nonzero_segments_rolled_out"],
            "N_usable_supervision_targets":
                bank_meta["N_usable_commitment_level_supervision_targets"],
            "N_behavior_changing_targets":
                bank_meta["N_supervision_targets_with_teacher_disagreement"],
            "behavior_changing_by_latent": {"z0": 177, "z1": 51},
            "implication_if_outcome_is_asymmetric":
                "a pre-existing 3.5x z0/z1 imbalance in behaviour-changing supervision is "
                "already on record as a candidate explanation for a delta_A > 0, "
                "delta_B <= 0 outcome -- to be investigated, never assumed",
        },
        "outputs": {
            "checkpoint_dir": str(Path(cfg.checkpoint_dir).relative_to(ROOT)),
            "metrics_csv": str(Path(cfg.metrics_csv_path).relative_to(ROOT)),
            "episode_csv": str(Path(cfg.episode_csv_path).relative_to(ROOT)),
        },
        "EVAL_block_sealed": f"{EVAL_BLOCK.start}..{EVAL_BLOCK.stop - 1}, untouched by training",
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("control", "treatment"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-steps", type=int, default=0)
    args = ap.parse_args()
    arm = args.arm

    import torch  # noqa: F401
    from experiments.run_exp2b_specialization_preserving_compression import (
        configure_exp2b_live_environment,
    )
    from experiments.ccp_s2_build_training_array import bank_hash
    from rl import launch_audit_hooks as hooks
    from rl.causal_sequence_runner import CausalSequenceRunner
    from rl.launch_gate import LaunchGateError, check_opponent_mode, format_checks
    from rl.training.orchestrator import orchestrate_training_run

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: S2 spec not frozen: {spec['status']!r}")

    out_dir = OUT_ROOT / arm
    record = SD / f"CCP_S2_{arm.upper()}_RESULT.json"
    verifying = args.verify_steps > 0
    if record.is_file() and not verifying:
        raise SystemExit(f"REFUSING: {record} exists; this arm is one-shot")

    seed = training_seed()
    ckpt = incumbent_checkpoint()

    # fail-closed: re-derive the bank hash from the frozen Stage A/B artifacts
    stage_b = json.loads(STAGE_B.read_text(encoding="utf-8"))
    if stage_b["status"] != "FROZEN_RESULT":
        raise SystemExit(f"REFUSING: Stage B not frozen: {stage_b['status']!r}")
    fresh_hash = bank_hash(stage_b)
    bank_meta = json.loads(BANK_META.read_text(encoding="utf-8"))
    if bank_meta["segment_bank_hash"] != fresh_hash:
        raise SystemExit("REFUSING: causal bank array hash does not match a fresh rebuild "
                         "from the frozen Stage A/Stage B records")
    if _sha(BANK_NPZ) != bank_meta["npz_sha256"]:
        raise SystemExit("REFUSING: causal bank npz sha256 does not match its own metadata")

    cfg = build_config(arm, seed, ckpt)
    if verifying:
        cfg.total_timesteps = int(args.verify_steps)
        cfg.run_tag = f"ccp_s2_{arm}_wiring_verification"
        cfg.checkpoint_dir = str(out_dir / "verify_ckpts")
        cfg.metrics_csv_path = str(out_dir / "verify_metrics.csv")
        cfg.episode_csv_path = str(out_dir / "verify_episode_rows.csv")

    checks = [check_opponent_mode(cfg), check_warm_start(cfg, ckpt)]
    print(f"CCP-S2 {arm.upper()}  (warm-start fine-tune, "
          f"{'task PPO + L_causal' if arm == 'treatment' else 'task PPO only'})")
    print(format_checks(checks))
    if [c for c in checks if c.blocking and not c.passed]:
        raise LaunchGateError("LAUNCH REFUSED:\n" + format_checks(checks))

    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / ("LAUNCH_MANIFEST_VERIFY.json" if verifying else "LAUNCH_MANIFEST.json")
    manifest = write_launch_manifest(arm, cfg, ckpt, seed, bank_meta, fresh_hash, man_path)

    print(f"\nWARM START")
    print(f"  checkpoint      {ckpt.name}")
    print(f"  sha256          {manifest['warm_start']['checkpoint_sha256'][:16]}...")
    print(f"  weights_loaded  True     optimizer_reused False")
    print(f"\nARM")
    print(f"  seed {seed}   steps {cfg.total_timesteps:,}")
    if arm == "treatment":
        print(f"  lambda_causal   {LAMBDA_CAUSAL}   cadence {CADENCE}   batch_rows {BATCH_ROWS}")
        print(f"  anchors         {bank_meta['nonzero_segments_rolled_out']} of "
              f"{bank_meta['total_segments_in_causal_bank']} active units")
        print(f"  bank rows       {bank_meta['n_rows']} "
              f"(z0 {bank_meta['rows_by_latent']['z0']} / z1 {bank_meta['rows_by_latent']['z1']})")
        print(f"  bank_hash       {fresh_hash[:16]}...")
    else:
        print(f"  causal path     ABSENT and FATAL if called")
    print(f"  EVAL {EVAL_BLOCK.start}..{EVAL_BLOCK.stop - 1} SEALED")
    print(f"  -> {man_path}\n", flush=True)

    if args.dry_run:
        print("DRY RUN -- gates verified, manifest written, nothing trained.")
        return 0

    originals = _install_tripwires(arm)
    state = {}

    def _attach(trainer, _cfg):
        state["auditor"] = hooks.attach(
            trainer, z_to_pole={0: "A", 1: "B"},
            opponent_to_pole=V4.V3.OPPONENT_ID_TO_POLE, hard_fail=True, artifact_dir=out_dir)
        if arm == "treatment":
            state["seq"] = CausalSequenceRunner(
                trainer, BANK_NPZ, BANK_META, lam=LAMBDA_CAUSAL, cadence=CADENCE,
                batch_rows=BATCH_ROWS, expected_bank_hash=fresh_hash,
                device=str(getattr(trainer, "device", "cpu")))
            trainer.oracle_rehearsal_runner = state["seq"]
        m = trainer.model
        state["before"] = {n: p.detach().cpu().numpy().copy()
                           for n, p in m.named_parameters()
                           if any(k in n for k in ("latent_branch_trunks", "latent_action_heads",
                                                   "latent_adapters", "head_V"))}

    try:
        orchestrate_training_run(
            cfg,
            pre_rollout_env_setup=partial(
                configure_exp2b_live_environment,
                contract={"record": f"CCP-S2 {arm}", "utc": _now()},
                training_seed_range=(int(cfg.seed), int(cfg.seed) + 320),
                manifest_key="ccp_s2_protocol",
                context_label=f"CCP-S2 {arm} construction"),
            post_trainer_setup=_attach)
        verdict, reason = "COMPLETE", None
    except Exception as exc:                                   # noqa: BLE001
        if not isinstance(exc, (LaunchGateError, _FatalDisabledPath)):
            _restore_tripwires(originals)
            raise
        verdict, reason = "INVALID_TREATMENT", str(exc)
        print(f"\n  RUN TERMINATED BY AN INVARIANT: {exc}")
    finally:
        _restore_tripwires(originals)

    seq = state.get("seq")
    stel = seq.telemetry() if seq else {}
    auditor = state.get("auditor")
    coverage = auditor.coverage(expected_envs=32, min_resets=2) if auditor else None

    moved = {}
    trainer_model = seq.trainer.model if seq else None
    if trainer_model is None and "before" in state:
        trainer_model = state.get("model")
    if "before" in state and trainer_model is not None:
        for n, p in trainer_model.named_parameters():
            if n in state["before"]:
                moved[n] = not np.array_equal(state["before"][n], p.detach().cpu().numpy())
    z0_moved = any(v for k, v in moved.items()
                   if "latent_branch_trunks.0" in k or "latent_action_heads.0" in k)
    z1_moved = any(v for k, v in moved.items()
                   if "latent_branch_trunks.1" in k or "latent_action_heads.1" in k)

    if verifying:
        ok = verdict == "COMPLETE"
        if arm == "treatment":
            ok = ok and stel.get("updates", 0) > 0 and stel.get("z0_exposures", 0) > 0 \
                 and stel.get("z1_exposures", 0) > 0
        print(f"\nLIVE PATHS ({cfg.total_timesteps} steps, no record written)")
        print(f"  task PPO minibatches    {stel.get('n_ppo_minibatches', 'n/a (control)')}")
        print(f"  causal updates          {stel.get('updates', 0 if arm == 'treatment' else 'ABSENT by design')}")
        if arm == "treatment":
            print(f"  z0 / z1 exposures       {stel.get('z0_exposures', 0)} / {stel.get('z1_exposures', 0)}"
                  f"   <- BOTH must be > 0")
        print(f"  legacy paths            0 (fatal if any fired)")
        if arm == "control":
            print(f"  causal_supervision_loss 0 calls (fatal if any fired)")
        print(f"  z0/z1 actor moved       {z0_moved} / {z1_moved}")
        if coverage:
            print(f"  envs / mismatches       {coverage['envs_observed']}/32, "
                  f"{coverage['total_mismatches']}")
        print(f"\n  COMPOSITION: {'OK' if ok else 'BROKEN'}")
        return 0 if ok else 1

    record.write_text(json.dumps({
        "record": f"CCP-S2 {arm} production run", "status": "FROZEN_RESULT", "utc": _now(),
        "implements": "CCP_S2_SPEC.json#WARM_START_PROTOCOL", "VERDICT": verdict,
        "arm": arm.upper(), "termination_reason": reason,
        "seed": int(cfg.seed), "total_timesteps": int(cfg.total_timesteps),
        "launch_manifest": json.loads(man_path.read_text(encoding="utf-8")),
        "causal_telemetry": stel if arm == "treatment" else "ABSENT by design",
        "private_parameter_motion": {"z0_actor_moved": z0_moved, "z1_actor_moved": z1_moved,
                                     "per_param_moved": moved},
        "coverage": coverage,
        "EVAL_touched": False,
        "authorizes": "nothing further; EVAL is a separate frozen decision",
    }, indent=2), encoding="utf-8")
    print(f"\n  VERDICT: {verdict}")
    print(f"  -> {record}")
    return 0 if verdict == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
