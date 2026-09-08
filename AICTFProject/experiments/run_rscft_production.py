"""RSCFT production: matched RETENTION_CONTROL / RETENTION_TREATMENT fine-tuning.

Implements RSCFT_SPEC.json.

    RETENTION_CONTROL    original incumbent + PPO + CCP-S2 causal loss
    RETENTION_TREATMENT  the same, PLUS the EMA-teacher retention objective

Note what the control is: the PREVIOUS CAUSAL TREATMENT, not plain PPO. Causal supervision
alone is already known to fail (CCP_S2_EVAL_INTEGRITY.json). Re-running plain PPO as the
control would re-answer an answered question instead of isolating retention, the one new
variable.

Both arms warm-start from the ORIGINAL 1,003,520-step incumbent -- NOT from either CCP-S2
terminal checkpoint -- because that is the checkpoint where delta_A = +0.0938 still existed.
The question is whether that specialization can be preserved while causal corrections operate;
starting where it has already been erased would test nothing.

Asymmetry is enforced in both directions, as RSCFT_SPEC.json's preflight requires: TREATMENT
attaches the RetentionRunner and the updater's cadence assertion aborts if it stops firing;
CONTROL makes retention_kl and EMATeacher.update FATAL if called, so "retention is unreachable"
is structural rather than an assumption about wiring.

Carries forward every guard the CCP-S2 runner earned, including the one that mattered most:
the horizon is additional_timesteps, not an absolute total_timesteps, because learn() runs
until global_step REACHES its target and the warm start already sits at ~1.0M -- an absolute
500k budget trains for zero steps and still exits COMPLETE.

Run:  python experiments/run_rscft_production.py --arm control --verify-steps 12288
      python experiments/run_rscft_production.py --arm control
      python experiments/run_rscft_production.py --arm treatment
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
SPEC = SD / "RSCFT_SPEC.json"
INCUMBENT_FROZEN = SD / "CCP_SUCCESSOR_MODEL_FROZEN.json"
ROUTING = SD / "CCP_S2_CAUSAL_BANK_ROUTING.json"
STAGE_B = SD / "CCP_S2_CAUSAL_BANK_STAGE_B.json"
BANK_NPZ = SD / "ccp_s2_causal_bank.npz"
BANK_META = SD / "CCP_S2_CAUSAL_BANK_ARRAY.json"
OUT_ROOT = SD / "rscft_production"

TOTAL_STEPS = 500_000
LAMBDA_CAUSAL = 0.05
CADENCE = 4
BATCH_ROWS = 48
LAMBDA_RET = 0.01          # RSCFT_SPEC.json: preregistered conservative choice, not copied
EMA_DECAY = 0.995          # same
RETENTION_CADENCE = 1      # "after each actor optimizer update"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def spec() -> dict:
    s = json.loads(SPEC.read_text(encoding="utf-8"))
    if s["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: RSCFT spec not frozen: {s['status']!r}")
    return s


def training_seed(s: dict) -> int:
    seed = int(s["SEEDS"]["training_seed"])
    ev_lo, ev_hi = (int(x) for x in s["SEEDS"]["sealed_eval_block"].split(".."))
    if ev_lo <= seed <= ev_hi:
        raise SystemExit("REFUSING: training seed lies inside the sealed EVAL block")
    if 11_701_001 <= seed <= 11_701_064:
        raise SystemExit("REFUSING: training seed lies inside CCP-S2's SPENT EVAL block")
    return seed


def incumbent_checkpoint() -> Path:
    frozen = json.loads(INCUMBENT_FROZEN.read_text(encoding="utf-8"))
    ck = ROOT / frozen["TERMINAL_CHECKPOINT"]["path"]
    got = _sha(ck)
    if got != frozen["TERMINAL_CHECKPOINT"]["sha256"]:
        raise SystemExit("REFUSING: incumbent checkpoint sha mismatch")
    return ck


def build_config(arm: str, seed: int, ckpt: Path):
    """Identical for both arms except run_tag and output paths."""
    cfg = V4.build_config()
    cfg.seed = seed
    cfg.additional_timesteps = TOTAL_STEPS      # ADDITIONAL, never absolute -- see docstring
    cfg.total_timesteps = TOTAL_STEPS
    cfg.run_tag = f"rscft_{arm}"
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
    """A path that must not run in this arm was called."""


def _install_tripwires(arm: str) -> dict:
    """Legacy auxiliary paths fatal in BOTH arms; retention fatal in CONTROL only.

    The causal path is deliberately NOT tripwired here -- unlike CCP-S2, BOTH RSCFT arms run
    it. Retention is the only differing mechanism, so it is the only one guarded.
    """
    import rl.oracle_rehearsal as V1
    import rl.paired_rehearsal as PR
    import rl.retention_stabilizer as RS
    import rl.trajectory_identity as TI

    originals = {
        "V1": V1.RehearsalBank.sample,
        "PR": PR.paired_rehearsal_loss,
        "TI": TI.TrajectoryIdentityRunner.loss,
        "RS_kl": RS.retention_kl,
        "RS_ema": RS.EMATeacher.update,
    }

    def _raise(name, why):
        def _fn(*a, **k):
            raise _FatalDisabledPath(f"{name} was called during an RSCFT {arm.upper()} run: {why}")
        return _fn

    legacy = "this program's prior auxiliary paths are structurally disabled in RSCFT"
    V1.RehearsalBank.sample = _raise("RehearsalBank.sample", legacy)
    PR.paired_rehearsal_loss = _raise("paired_rehearsal_loss", legacy)
    TI.TrajectoryIdentityRunner.loss = _raise("TrajectoryIdentityRunner.loss", legacy)
    if arm == "control":
        why = ("CONTROL is PPO + causal loss ONLY; any retention gradient here would erase "
               "the single variable this successor exists to isolate")
        RS.retention_kl = _raise("retention_kl", why)
        RS.EMATeacher.update = _raise("EMATeacher.update", why)
    return originals


def _restore_tripwires(originals: dict) -> None:
    import rl.oracle_rehearsal as V1
    import rl.paired_rehearsal as PR
    import rl.retention_stabilizer as RS
    import rl.trajectory_identity as TI
    V1.RehearsalBank.sample = originals["V1"]
    PR.paired_rehearsal_loss = originals["PR"]
    TI.TrajectoryIdentityRunner.loss = originals["TI"]
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
    from rl.launch_gate import LaunchGateError, Check, check_opponent_mode, format_checks
    from rl.retention_stabilizer import RetentionRunner
    from rl.training.orchestrator import orchestrate_training_run

    s = spec()
    seed = training_seed(s)
    ckpt = incumbent_checkpoint()
    out_dir = OUT_ROOT / arm
    record = SD / f"RSCFT_{arm.upper()}_RESULT.json"
    verifying = args.verify_steps > 0
    if record.is_file() and not verifying:
        raise SystemExit(f"REFUSING: {record} exists; this arm is one-shot")

    stage_b = json.loads(STAGE_B.read_text(encoding="utf-8"))
    fresh_hash = bank_hash(stage_b)
    bank_meta = json.loads(BANK_META.read_text(encoding="utf-8"))
    if bank_meta["segment_bank_hash"] != fresh_hash:
        raise SystemExit("REFUSING: causal bank array hash does not match a fresh rebuild")
    if _sha(BANK_NPZ) != bank_meta["npz_sha256"]:
        raise SystemExit("REFUSING: causal bank npz sha256 does not match its metadata")

    cfg = build_config(arm, seed, ckpt)
    if verifying:
        cfg.additional_timesteps = int(args.verify_steps)
        cfg.total_timesteps = int(args.verify_steps)
        cfg.run_tag = f"rscft_{arm}_wiring_verification"
        cfg.checkpoint_dir = str(out_dir / "verify_ckpts")
        cfg.metrics_csv_path = str(out_dir / "verify_metrics.csv")
        cfg.episode_csv_path = str(out_dir / "verify_episode_rows.csv")

    warm = Check("warm_start", bool(getattr(cfg, "load_path", None)) and
                 bool(getattr(cfg, "load_weights_only", False)),
                 f"original incumbent weights, fresh optimizer ({ckpt.name})")
    checks = [check_opponent_mode(cfg), warm]
    print(f"RSCFT {arm.upper()}  (PPO + L_causal"
          f"{' + L_retention' if arm == 'treatment' else ', retention ABSENT and FATAL'})")
    print(format_checks(checks))
    if [c for c in checks if c.blocking and not c.passed]:
        raise LaunchGateError("LAUNCH REFUSED:\n" + format_checks(checks))

    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / ("LAUNCH_MANIFEST_VERIFY.json" if verifying else "LAUNCH_MANIFEST.json")
    man_path.write_text(json.dumps({
        "record": f"RSCFT {arm} launch manifest", "status": "FROZEN_BEFORE_LAUNCH",
        "utc": _now(), "arm": arm.upper(), "implements": "RSCFT_SPEC.json",
        "training_seed": seed, "additional_timesteps": int(cfg.additional_timesteps),
        "warm_start": {"checkpoint": str(ckpt.relative_to(ROOT)), "sha256": _sha(ckpt),
                       "is_original_incumbent_not_an_S2_terminal": True,
                       "optimizer_state_reused": False},
        "causal": {"lambda_causal": LAMBDA_CAUSAL, "cadence": CADENCE,
                   "batch_rows": BATCH_ROWS, "segment_bank_hash": fresh_hash,
                   "present_in_both_arms": True},
        "retention": ({"lambda_ret": LAMBDA_RET, "ema_decay": EMA_DECAY,
                       "cadence": RETENTION_CADENCE,
                       "hyperparameters_are": "preregistered conservative choices by this "
                                              "program, NOT copied from any published source"}
                      if arm == "treatment" else
                      "ABSENT -- retention_kl and EMATeacher.update are fatal if called"),
        "outputs": {"checkpoint_dir": str(Path(cfg.checkpoint_dir).relative_to(ROOT))},
        "sealed_eval_block": s["SEEDS"]["sealed_eval_block"],
    }, indent=2), encoding="utf-8")

    print(f"\n  seed {seed}   +{cfg.additional_timesteps:,} steps   warm start {ckpt.name}")
    print(f"  causal   lambda={LAMBDA_CAUSAL} cadence={CADENCE} rows={BATCH_ROWS} (both arms)")
    if arm == "treatment":
        print(f"  retention lambda={LAMBDA_RET} decay={EMA_DECAY} cadence={RETENTION_CADENCE}")
    else:
        print(f"  retention ABSENT and FATAL if called")
    print(f"  EVAL {s['SEEDS']['sealed_eval_block']} SEALED")
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
        state["seq"] = CausalSequenceRunner(
            trainer, BANK_NPZ, BANK_META, lam=LAMBDA_CAUSAL, cadence=CADENCE,
            batch_rows=BATCH_ROWS, expected_bank_hash=fresh_hash,
            device=str(getattr(trainer, "device", "cpu")))
        trainer.oracle_rehearsal_runner = state["seq"]
        if arm == "treatment":
            state["ret"] = RetentionRunner(trainer, lam=LAMBDA_RET, decay=EMA_DECAY,
                                           cadence=RETENTION_CADENCE)
            trainer.retention_runner = state["ret"]
        state["trainer"] = trainer
        m = trainer.model
        state["before"] = {n: p.detach().cpu().numpy().copy()
                           for n, p in m.named_parameters()
                           if any(k in n for k in ("latent_branch_trunks", "latent_action_heads",
                                                   "latent_adapters", "head_V"))}
        state["before_all"] = {n: p.detach().cpu().numpy().copy()
                               for n, p in m.named_parameters()}
        state["global_step_at_attach"] = int(getattr(trainer, "global_step", 0))

    try:
        orchestrate_training_run(
            cfg,
            pre_rollout_env_setup=partial(
                configure_exp2b_live_environment,
                contract={"record": f"RSCFT {arm}", "utc": _now()},
                training_seed_range=(int(cfg.seed), int(cfg.seed) + 320),
                manifest_key="rscft_protocol",
                context_label=f"RSCFT {arm} construction"),
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

    trainer_obj = state.get("trainer")
    trainer_model = getattr(trainer_obj, "model", None) if trainer_obj is not None else None
    steps_advanced = None
    if trainer_obj is not None and "global_step_at_attach" in state:
        steps_advanced = int(getattr(trainer_obj, "global_step", 0)) - state["global_step_at_attach"]
        expected = int(getattr(cfg, "additional_timesteps", 0) or 0)
        if verdict == "COMPLETE" and steps_advanced < expected * 0.9:
            raise SystemExit(
                f"REFUSING: run advanced only {steps_advanced:,} steps but {expected:,} were "
                "requested. A zero- or short-trained warm-start run exits cleanly and would "
                "reach EVAL as a fabricated null result.")
    if "before" in state and trainer_model is None:
        raise SystemExit("REFUSING: parameter-motion baseline captured but no model reachable")

    moved = {}
    if "before" in state:
        for n, p in trainer_model.named_parameters():
            if n in state["before"]:
                moved[n] = not np.array_equal(state["before"][n], p.detach().cpu().numpy())
        if not moved:
            raise SystemExit("REFUSING: parameter-motion check matched zero parameters")
    n_all_moved = sum(1 for n, p in trainer_model.named_parameters()
                      if n in state.get("before_all", {})
                      and not np.array_equal(state["before_all"][n],
                                             p.detach().cpu().numpy())) if trainer_model else 0
    if verdict == "COMPLETE" and "before_all" in state and n_all_moved == 0:
        raise SystemExit("REFUSING: not a single model parameter changed across the whole run")

    z0_moved = any(v for k, v in moved.items()
                   if "latent_branch_trunks.0" in k or "latent_action_heads.0" in k)
    z1_moved = any(v for k, v in moved.items()
                   if "latent_branch_trunks.1" in k or "latent_action_heads.1" in k)
    seq, ret = state.get("seq"), state.get("ret")
    stel = seq.telemetry() if seq else {}
    rtel = ret.telemetry() if ret else "ABSENT by design"
    auditor = state.get("auditor")
    coverage = auditor.coverage(expected_envs=32, min_resets=2) if auditor else None

    if verifying:
        ok = (verdict == "COMPLETE" and stel.get("updates", 0) > 0
              and stel.get("z0_exposures", 0) > 0 and stel.get("z1_exposures", 0) > 0
              and z0_moved and z1_moved and n_all_moved > 0)
        if arm == "treatment":
            ok = ok and isinstance(rtel, dict) and rtel["retention_updates"] > 0 \
                 and rtel["ema_updates"] > 0
        print(f"\nLIVE PATHS ({cfg.additional_timesteps} additional steps, no record written)")
        print(f"  causal updates          {stel.get('updates', 0)}  <- MUST be > 0 (both arms)")
        print(f"  z0 / z1 exposures       {stel.get('z0_exposures',0)} / {stel.get('z1_exposures',0)}")
        if arm == "treatment":
            print(f"  retention updates       {rtel['retention_updates']}  <- MUST be > 0")
            print(f"  EMA updates             {rtel['ema_updates']}  <- MUST be > 0")
            print(f"  retention KL (last)     {rtel['last_kl_mean']:.6e}")
            print(f"  eligible heads (last)   {rtel['last_eligible_heads']}")
        else:
            print(f"  retention               ABSENT (fatal if called; run would have aborted)")
        print(f"  z0/z1 actor moved       {z0_moved} / {z1_moved}")
        print(f"  ALL params moved        {n_all_moved}  <- positive control, MUST be > 0")
        if coverage:
            print(f"  envs / mismatches       {coverage['envs_observed']}/32, "
                  f"{coverage['total_mismatches']}")
        print(f"\n  COMPOSITION: {'OK' if ok else 'BROKEN'}")
        return 0 if ok else 1

    record.write_text(json.dumps({
        "record": f"RSCFT {arm} production run", "status": "FROZEN_RESULT", "utc": _now(),
        "implements": "RSCFT_SPEC.json", "VERDICT": verdict, "arm": arm.upper(),
        "termination_reason": reason, "seed": int(cfg.seed),
        "additional_timesteps": int(cfg.additional_timesteps),
        "steps_advanced": steps_advanced,
        "launch_manifest": json.loads(man_path.read_text(encoding="utf-8")),
        "causal_telemetry": stel, "retention_telemetry": rtel,
        "private_parameter_motion": {"z0_actor_moved": z0_moved, "z1_actor_moved": z1_moved,
                                     "all_params_moved": n_all_moved,
                                     "per_param_moved": moved},
        "coverage": coverage, "EVAL_touched": False,
        "authorizes": "nothing further; EVAL is a separate frozen decision",
    }, indent=2), encoding="utf-8")
    print(f"\n  VERDICT: {verdict}")
    print(f"  -> {record}")
    return 0 if verdict == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
