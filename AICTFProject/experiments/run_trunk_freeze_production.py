"""Trunk-freeze production: matched TRUNK_FROZEN_CONTROL / TRUNK_FROZEN_TREATMENT.

Implements TRUNK_FREEZE_SPEC.json.

    TRUNK_FROZEN_CONTROL    original incumbent + shared trunk frozen + task PPO ONLY
    TRUNK_FROZEN_TREATMENT  same, PLUS the existing causal loss (lambda_causal=0.05)

Both arms warm-start from the ORIGINAL incumbent (not any CCP-S2/RSCFT terminal checkpoint),
same reason as RSCFT: that is the checkpoint where delta_A = +0.0938 still existed.

The one experimental variable is the trunk freeze (rl.trunk_freeze.apply(), 17 real + 2
vestigial shared parameters set requires_grad=False, verified against the model's actual
parameter names at attach time -- not assumed). CONTROL additionally makes the causal loss
itself fatal if called, the same asymmetric-tripwire pattern every matched-arm pair in this
program has used, so "CONTROL had no causal path" is structural rather than an assumption
about wiring.

Carries forward every guard earned across CCP-S2 and RSCFT: additional_timesteps (not
absolute total_timesteps -- the horizon bug that would silently train zero steps), the
five-part anti-null check on parameter motion, and the fail-closed warm-start gate.

Run:  python experiments/run_trunk_freeze_production.py --arm control --verify-steps 12288
      python experiments/run_trunk_freeze_production.py --arm control
      python experiments/run_trunk_freeze_production.py --arm treatment
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
SPEC = SD / "TRUNK_FREEZE_SPEC.json"
INCUMBENT_FROZEN = SD / "CCP_SUCCESSOR_MODEL_FROZEN.json"
STAGE_B = SD / "CCP_S2_CAUSAL_BANK_STAGE_B.json"
BANK_NPZ = SD / "ccp_s2_causal_bank.npz"
BANK_META = SD / "CCP_S2_CAUSAL_BANK_ARRAY.json"
OUT_ROOT = SD / "trunk_freeze_production"

TOTAL_STEPS = 500_000
LAMBDA_CAUSAL = 0.05
CADENCE = 4
BATCH_ROWS = 48
TRAINING_SEED = 11_705_001
EVAL_BLOCK = range(11_706_001, 11_706_065)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def spec() -> dict:
    s = json.loads(SPEC.read_text(encoding="utf-8"))
    if s["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: trunk freeze spec not frozen: {s['status']!r}")
    seed = int(s["SEEDS"]["training_seed"])
    if seed != TRAINING_SEED:
        raise SystemExit(f"REFUSING: spec seed {seed} != module constant {TRAINING_SEED}")
    return s


def incumbent_checkpoint() -> Path:
    frozen = json.loads(INCUMBENT_FROZEN.read_text(encoding="utf-8"))
    ck = ROOT / frozen["TERMINAL_CHECKPOINT"]["path"]
    got = _sha(ck)
    if got != frozen["TERMINAL_CHECKPOINT"]["sha256"]:
        raise SystemExit("REFUSING: incumbent checkpoint sha mismatch")
    return ck


def build_config(arm: str, ckpt: Path):
    cfg = V4.build_config()
    cfg.seed = TRAINING_SEED
    cfg.additional_timesteps = TOTAL_STEPS      # ADDITIONAL, never absolute
    cfg.total_timesteps = TOTAL_STEPS
    cfg.run_tag = f"trunk_freeze_{arm}"
    out = OUT_ROOT / arm
    cfg.checkpoint_dir = str(out / "ckpts")
    cfg.metrics_csv_path = str(out / "metrics.csv")
    cfg.episode_csv_path = str(out / "episode_rows.csv")
    cfg.load_path = str(ckpt)
    cfg.load_weights_only = True
    if not getattr(cfg, "rasr_private_critic_heads", False):
        raise SystemExit("REFUSING: rasr_private_critic_heads did not survive V4.build_config")
    return cfg


class _FatalDisabledPath(RuntimeError):
    pass


def _install_tripwires(arm: str) -> dict:
    import rl.causal_supervision as CS
    import rl.oracle_rehearsal as V1
    import rl.paired_rehearsal as PR
    import rl.retention_stabilizer as RS
    import rl.trajectory_identity as TI

    originals = {"V1": V1.RehearsalBank.sample, "PR": PR.paired_rehearsal_loss,
                "TI": TI.TrajectoryIdentityRunner.loss, "CS": CS.causal_supervision_loss,
                "RS_kl": RS.retention_kl, "RS_ema": RS.EMATeacher.update}

    def _raise(name, why):
        def _fn(*a, **k):
            raise _FatalDisabledPath(f"{name} called during trunk-freeze {arm.upper()}: {why}")
        return _fn

    legacy = "this program's prior auxiliary paths are structurally disabled here"
    V1.RehearsalBank.sample = _raise("RehearsalBank.sample", legacy)
    PR.paired_rehearsal_loss = _raise("paired_rehearsal_loss", legacy)
    TI.TrajectoryIdentityRunner.loss = _raise("TrajectoryIdentityRunner.loss", legacy)
    RS.retention_kl = _raise("retention_kl", "RSCFT is closed; this experiment tests trunk "
                             "freezing, not retention")
    RS.EMATeacher.update = _raise("EMATeacher.update", "same")
    if arm == "control":
        CS.causal_supervision_loss = _raise(
            "causal_supervision_loss",
            "CONTROL isolates whether freezing ALONE preserves delta_A; any causal gradient "
            "here would void that isolation")
    return originals


def _restore_tripwires(originals: dict) -> None:
    import rl.causal_supervision as CS
    import rl.oracle_rehearsal as V1
    import rl.paired_rehearsal as PR
    import rl.retention_stabilizer as RS
    import rl.trajectory_identity as TI
    V1.RehearsalBank.sample = originals["V1"]
    PR.paired_rehearsal_loss = originals["PR"]
    TI.TrajectoryIdentityRunner.loss = originals["TI"]
    CS.causal_supervision_loss = originals["CS"]
    RS.retention_kl = originals["RS_kl"]
    RS.EMATeacher.update = originals["RS_ema"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("control", "treatment"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-steps", type=int, default=0)
    args = ap.parse_args()
    arm = args.arm

    import torch  # noqa: F401
    from experiments.ccp_s2_build_training_array import bank_hash
    from experiments.run_exp2b_specialization_preserving_compression import (
        configure_exp2b_live_environment,
    )
    from rl import launch_audit_hooks as hooks
    from rl.causal_sequence_runner import CausalSequenceRunner
    from rl.launch_gate import Check, LaunchGateError, check_opponent_mode, format_checks
    import rl.trunk_freeze as TF

    s = spec()
    ckpt = incumbent_checkpoint()
    out_dir = OUT_ROOT / arm
    record = SD / f"TRUNK_FREEZE_{arm.upper()}_RESULT.json"
    verifying = args.verify_steps > 0
    if record.is_file() and not verifying:
        raise SystemExit(f"REFUSING: {record} exists; this arm is one-shot")

    stage_b = json.loads(STAGE_B.read_text(encoding="utf-8"))
    fresh_hash = bank_hash(stage_b)
    bank_meta = json.loads(BANK_META.read_text(encoding="utf-8"))
    if bank_meta["segment_bank_hash"] != fresh_hash:
        raise SystemExit("REFUSING: causal bank array hash does not match a fresh rebuild")
    if _sha(BANK_NPZ) != bank_meta["npz_sha256"]:
        raise SystemExit("REFUSING: causal bank npz sha256 mismatch")

    cfg = build_config(arm, ckpt)
    if verifying:
        cfg.additional_timesteps = int(args.verify_steps)
        cfg.total_timesteps = int(args.verify_steps)
        cfg.run_tag = f"trunk_freeze_{arm}_wiring_verification"
        cfg.checkpoint_dir = str(out_dir / "verify_ckpts")
        cfg.metrics_csv_path = str(out_dir / "verify_metrics.csv")
        cfg.episode_csv_path = str(out_dir / "verify_episode_rows.csv")

    warm = Check("warm_start", bool(getattr(cfg, "load_path", None)) and
                 bool(getattr(cfg, "load_weights_only", False)),
                 f"original incumbent weights, fresh optimizer ({ckpt.name})")
    checks = [check_opponent_mode(cfg), warm]
    print(f"TRUNK FREEZE {arm.upper()}  (frozen shared trunk + "
          f"{'task PPO only, causal FATAL' if arm=='control' else 'task PPO + causal loss'})")
    print(format_checks(checks))
    if [c for c in checks if c.blocking and not c.passed]:
        raise LaunchGateError("LAUNCH REFUSED:\n" + format_checks(checks))

    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / ("LAUNCH_MANIFEST_VERIFY.json" if verifying else "LAUNCH_MANIFEST.json")
    man_path.write_text(json.dumps({
        "record": f"trunk-freeze {arm} launch manifest", "status": "FROZEN_BEFORE_LAUNCH",
        "utc": _now(), "arm": arm.upper(), "implements": "TRUNK_FREEZE_SPEC.json",
        "training_seed": TRAINING_SEED, "additional_timesteps": int(cfg.additional_timesteps),
        "warm_start": {"checkpoint": str(ckpt.relative_to(ROOT)), "sha256": _sha(ckpt),
                       "optimizer_state_reused": False},
        "trunk_freeze": "17 real + 2 vestigial shared params frozen, 20 private params "
                        "trainable, per TRUNK_FREEZE_SPEC.json's empirically-verified partition",
        "causal": ({"lambda_causal": LAMBDA_CAUSAL, "cadence": CADENCE, "batch_rows": BATCH_ROWS,
                    "segment_bank_hash": fresh_hash} if arm == "treatment" else
                   "ABSENT -- causal_supervision_loss is fatal if called"),
        "outputs": {"checkpoint_dir": str(Path(cfg.checkpoint_dir).relative_to(ROOT))},
        "sealed_eval_block": s["SEEDS"]["sealed_eval_block"],
    }, indent=2), encoding="utf-8")

    print(f"\n  seed {TRAINING_SEED}   +{cfg.additional_timesteps:,} steps   warm start {ckpt.name}")
    print(f"  trunk frozen: 19 params (17 real + 2 vestigial), 20 private trainable")
    if arm == "treatment":
        print(f"  causal   lambda={LAMBDA_CAUSAL} cadence={CADENCE} rows={BATCH_ROWS}")
    else:
        print(f"  causal   ABSENT and FATAL if called")
    print(f"  EVAL {s['SEEDS']['sealed_eval_block']} SEALED")
    print(f"  -> {man_path}\n", flush=True)

    if args.dry_run:
        print("DRY RUN -- gates verified, manifest written, nothing trained.")
        return 0

    originals = _install_tripwires(arm)
    state = {}

    def _attach(trainer, _cfg):
        m = trainer.model
        freeze_report = TF.apply(m)          # REFUSES if param names don't match exactly
        state["freeze_report"] = freeze_report
        state["auditor"] = hooks.attach(
            trainer, z_to_pole={0: "A", 1: "B"},
            opponent_to_pole=V4.V3.OPPONENT_ID_TO_POLE, hard_fail=True, artifact_dir=out_dir)
        if arm == "treatment":
            state["seq"] = CausalSequenceRunner(
                trainer, BANK_NPZ, BANK_META, lam=LAMBDA_CAUSAL, cadence=CADENCE,
                batch_rows=BATCH_ROWS, expected_bank_hash=fresh_hash,
                device=str(getattr(trainer, "device", "cpu")))
            trainer.oracle_rehearsal_runner = state["seq"]
        state["trainer"] = trainer
        state["before_all"] = TF.snapshot(m)
        state["before"] = {n: p.detach().cpu().numpy().copy()
                           for n, p in m.named_parameters()
                           if any(k in n for k in ("latent_branch_trunks", "latent_action_heads",
                                                   "latent_adapters", "head_V"))}
        state["global_step_at_attach"] = int(getattr(trainer, "global_step", 0))

    try:
        orchestrate_training_run = __import__(
            "rl.training.orchestrator", fromlist=["orchestrate_training_run"]
        ).orchestrate_training_run
        orchestrate_training_run(
            cfg,
            pre_rollout_env_setup=partial(
                configure_exp2b_live_environment,
                contract={"record": f"trunk-freeze {arm}", "utc": _now()},
                training_seed_range=(int(cfg.seed), int(cfg.seed) + 320),
                manifest_key="trunk_freeze_protocol",
                context_label=f"trunk-freeze {arm} construction"),
            post_trainer_setup=_attach)
        verdict, reason = "COMPLETE", None
    except Exception as exc:                                   # noqa: BLE001
        if not isinstance(exc, (LaunchGateError, _FatalDisabledPath,
                                __import__("rl.trunk_freeze", fromlist=["TrunkFreezeError"]).TrunkFreezeError)):
            _restore_tripwires(originals)
            raise
        verdict, reason = "INVALID_TREATMENT", str(exc)
        print(f"\n  RUN TERMINATED BY AN INVARIANT: {exc}")
    finally:
        _restore_tripwires(originals)

    trainer_obj = state.get("trainer")
    trainer_model = getattr(trainer_obj, "model", None) if trainer_obj is not None else None
    steps_advanced = None
    if trainer_obj is not None and "global_step_at_attach" in state:
        steps_advanced = int(getattr(trainer_obj, "global_step", 0)) - state["global_step_at_attach"]
        expected = int(getattr(cfg, "additional_timesteps", 0) or 0)
        if verdict == "COMPLETE" and steps_advanced < expected * 0.9:
            raise SystemExit(f"REFUSING: run advanced only {steps_advanced:,} of {expected:,} "
                             "requested steps")

    frozen_check = None
    if trainer_model is not None and "before_all" in state:
        frozen_check = TF.verify_frozen_after_step(state["before_all"], trainer_model)
        print(f"\n  TRUNK-FREEZE VERIFICATION (post-training, real weight diff)")
        print(f"    frozen params that moved: {frozen_check['moved_frozen']}  <- MUST be empty")
        print(f"    trainable params moved: {len(frozen_check['moved_trainable'])}/20  "
              f"<- MUST be > 0")

    moved = {}
    if "before" in state:
        for n, p in trainer_model.named_parameters():
            if n in state["before"]:
                moved[n] = not np.array_equal(state["before"][n], p.detach().cpu().numpy())
        if not moved:
            raise SystemExit("REFUSING: parameter-motion check matched zero parameters")
    z0_moved = any(v for k, v in moved.items()
                   if "latent_branch_trunks.0" in k or "latent_action_heads.0" in k)
    z1_moved = any(v for k, v in moved.items()
                   if "latent_branch_trunks.1" in k or "latent_action_heads.1" in k)
    seq = state.get("seq")
    stel = seq.telemetry() if seq else "ABSENT by design"
    auditor = state.get("auditor")
    coverage = auditor.coverage(expected_envs=32, min_resets=2) if auditor else None

    if verifying:
        ok = (verdict == "COMPLETE" and z0_moved and z1_moved
              and frozen_check is not None and not frozen_check["moved_frozen"]
              and len(frozen_check["moved_trainable"]) > 0)
        if arm == "treatment":
            ok = ok and isinstance(stel, dict) and stel.get("updates", 0) > 0
        print(f"\nLIVE PATHS ({cfg.additional_timesteps} additional steps, no record written)")
        print(f"  z0/z1 actor moved       {z0_moved} / {z1_moved}")
        if arm == "treatment":
            print(f"  causal updates          {stel.get('updates', 0)}  <- MUST be > 0")
        else:
            print(f"  causal                  ABSENT (fatal if called; run would have aborted)")
        if coverage:
            print(f"  envs / mismatches       {coverage['envs_observed']}/32, "
                  f"{coverage['total_mismatches']}")
        print(f"\n  COMPOSITION: {'OK' if ok else 'BROKEN'}")
        return 0 if ok else 1

    record.write_text(json.dumps({
        "record": f"trunk-freeze {arm} production run", "status": "FROZEN_RESULT", "utc": _now(),
        "implements": "TRUNK_FREEZE_SPEC.json", "VERDICT": verdict, "arm": arm.upper(),
        "termination_reason": reason, "seed": int(cfg.seed),
        "additional_timesteps": int(cfg.additional_timesteps), "steps_advanced": steps_advanced,
        "launch_manifest": json.loads(man_path.read_text(encoding="utf-8")),
        "trunk_freeze_verification": frozen_check, "freeze_report": state.get("freeze_report"),
        "causal_telemetry": stel,
        "private_parameter_motion": {"z0_actor_moved": z0_moved, "z1_actor_moved": z1_moved,
                                     "per_param_moved": moved},
        "coverage": coverage, "EVAL_touched": False,
        "authorizes": "nothing further; EVAL is a separate frozen decision",
    }, indent=2), encoding="utf-8")
    print(f"\n  VERDICT: {verdict}")
    print(f"  -> {record}")
    return 0 if verdict == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
