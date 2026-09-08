"""Z-separation production: matched UNSEPARATED_CONTROL / SEPARATED_TREATMENT.

Implements Z_SEPARATION_SPEC.json.

    UNSEPARATED_CONTROL  original incumbent + full network trainable + plain PPO,
                         latent_actor_z_separation_coef=0.0 (structurally inert -- this codebase's
                         own default), no causal loss, no retention -- the cleanest
                         "just keep training, nothing added" reference this program has had
    SEPARATED_TREATMENT  same, PLUS latent_actor_z_separation_coef=0.02 -- the ONE changed
                         variable. Mechanism already exists in this codebase
                         (rl/custom_ppo/update/separation_objectives.py), simply never turned on
                         for this architecture lineage before.

Both arms warm-start from the ORIGINAL incumbent. NO causal bank, NO trunk freeze, NO retention
-- this experiment isolates the separation loss alone, matching the CONTROL convention CCP-S2/
trunk-freeze already established (a no-causal-loss arm as the clean baseline).

250,000 additional steps (disclosed deviation from the 500k standard -- see spec).

Run:  python experiments/run_z_separation_production.py --arm control --verify-steps 12288
      python experiments/run_z_separation_production.py --arm control
      python experiments/run_z_separation_production.py --arm treatment
"""
from __future__ import annotations

import argparse
import csv
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
SPEC = SD / "Z_SEPARATION_SPEC.json"
INCUMBENT_FROZEN = SD / "CCP_SUCCESSOR_MODEL_FROZEN.json"
OUT_ROOT = SD / "z_separation_production"

TOTAL_STEPS = 250_000
SEPARATION_COEF = 0.02
TRAINING_SEED = 11_910_001
EVAL_BLOCK = range(11_911_001, 11_911_065)

ARM_TO_ROLE = {"control": "UNSEPARATED_CONTROL", "treatment": "SEPARATED_TREATMENT"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def spec() -> dict:
    s = json.loads(SPEC.read_text(encoding="utf-8"))
    if s["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: z-separation spec not frozen: {s['status']!r}")
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
    cfg.run_tag = f"z_separation_{arm}"
    out = OUT_ROOT / arm
    cfg.checkpoint_dir = str(out / "ckpts")
    cfg.metrics_csv_path = str(out / "metrics.csv")
    cfg.episode_csv_path = str(out / "episode_rows.csv")
    cfg.load_path = str(ckpt)
    cfg.load_weights_only = True
    if float(getattr(cfg, "latent_actor_z_separation_coef", 0.0) or 0.0) != 0.0:
        raise SystemExit("REFUSING: base config already has a nonzero separation coef -- "
                         "the 'ONE_CHANGED_VARIABLE' assumption would be violated")
    if arm == "treatment":
        cfg.latent_actor_z_separation_coef = SEPARATION_COEF
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
            raise _FatalDisabledPath(f"{name} called during z-separation {arm.upper()}: {why}")
        return _fn

    legacy = "this experiment tests the separation loss alone; no other auxiliary path applies"
    V1.RehearsalBank.sample = _raise("RehearsalBank.sample", legacy)
    PR.paired_rehearsal_loss = _raise("paired_rehearsal_loss", legacy)
    TI.TrajectoryIdentityRunner.loss = _raise("TrajectoryIdentityRunner.loss", legacy)
    CS.causal_supervision_loss = _raise("causal_supervision_loss", legacy)
    RS.retention_kl = _raise("retention_kl", legacy)
    RS.EMATeacher.update = _raise("EMATeacher.update", legacy)
    if hasattr(RS, "AnchorRetentionRunner"):
        original_anchor_init = RS.AnchorRetentionRunner.__init__

        def _anchor_raise(*a, **k):
            raise _FatalDisabledPath("AnchorRetentionRunner attached during z-separation")
        RS.AnchorRetentionRunner.__init__ = _anchor_raise
        originals["RS_anchor_init"] = original_anchor_init
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
    if "RS_anchor_init" in originals:
        RS.AnchorRetentionRunner.__init__ = originals["RS_anchor_init"]


def _mean_jsd_from_metrics(csv_path: Path) -> float | None:
    if not csv_path.is_file():
        return None
    vals = []
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            v = row.get("z_sep_JSD")
            if v not in (None, "", "nan"):
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
    return float(np.mean(vals)) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("control", "treatment"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-steps", type=int, default=0)
    args = ap.parse_args()
    arm = args.arm
    role = ARM_TO_ROLE[arm]

    import torch  # noqa: F401
    from experiments.run_exp2b_specialization_preserving_compression import (
        configure_exp2b_live_environment,
    )
    from rl import launch_audit_hooks as hooks
    from rl.launch_gate import Check, LaunchGateError, check_opponent_mode, format_checks

    s = spec()
    ckpt = incumbent_checkpoint()
    out_dir = OUT_ROOT / arm
    record = SD / f"ZSEP_{arm.upper()}_RESULT.json"
    verifying = args.verify_steps > 0
    if record.is_file() and not verifying:
        raise SystemExit(f"REFUSING: {record} exists; this arm is one-shot")

    cfg = build_config(arm, ckpt)
    if verifying:
        cfg.additional_timesteps = int(args.verify_steps)
        cfg.total_timesteps = int(args.verify_steps)
        cfg.run_tag = f"z_separation_{arm}_wiring_verification"
        cfg.checkpoint_dir = str(out_dir / "verify_ckpts")
        cfg.metrics_csv_path = str(out_dir / "verify_metrics.csv")
        cfg.episode_csv_path = str(out_dir / "verify_episode_rows.csv")

    warm = Check("warm_start", bool(getattr(cfg, "load_path", None)) and
                 bool(getattr(cfg, "load_weights_only", False)),
                 f"original incumbent weights, fresh optimizer ({ckpt.name})")
    checks = [check_opponent_mode(cfg), warm]
    print(f"Z-SEPARATION {role}  (full network trainable, "
          f"separation_coef={cfg.latent_actor_z_separation_coef})")
    print(format_checks(checks))
    if [c for c in checks if c.blocking and not c.passed]:
        raise LaunchGateError("LAUNCH REFUSED:\n" + format_checks(checks))

    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / ("LAUNCH_MANIFEST_VERIFY.json" if verifying else "LAUNCH_MANIFEST.json")
    man_path.write_text(json.dumps({
        "record": f"z-separation {arm} launch manifest", "status": "FROZEN_BEFORE_LAUNCH",
        "utc": _now(), "arm": role, "implements": "Z_SEPARATION_SPEC.json",
        "training_seed": TRAINING_SEED, "additional_timesteps": int(cfg.additional_timesteps),
        "warm_start": {"checkpoint": str(ckpt.relative_to(ROOT)), "sha256": _sha(ckpt),
                       "optimizer_state_reused": False},
        "separation_coef": float(cfg.latent_actor_z_separation_coef),
        "causal": "ABSENT -- causal_supervision_loss is fatal if called",
        "retention": "ABSENT -- retention_kl/EMATeacher.update/AnchorRetentionRunner all fatal",
        "outputs": {"checkpoint_dir": str(Path(cfg.checkpoint_dir).relative_to(ROOT))},
        "sealed_eval_block": s["SEEDS"]["sealed_eval_block"],
    }, indent=2), encoding="utf-8")

    print(f"\n  seed {TRAINING_SEED}   +{cfg.additional_timesteps:,} steps   warm start {ckpt.name}")
    print(f"  full network trainable (no freeze)   separation_coef={cfg.latent_actor_z_separation_coef}")
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
        state["trainer"] = trainer
        m = trainer.model
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
                contract={"record": f"z-separation {arm}", "utc": _now()},
                training_seed_range=(int(cfg.seed), int(cfg.seed) + 320),
                manifest_key="z_separation_protocol",
                context_label=f"z-separation {arm} construction"),
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
            raise SystemExit(f"REFUSING: run advanced only {steps_advanced:,} of {expected:,} "
                             "requested steps")

    moved = {}
    if "before" in state and trainer_model is not None:
        for n, p in trainer_model.named_parameters():
            if n in state["before"]:
                moved[n] = not np.array_equal(state["before"][n], p.detach().cpu().numpy())
        if not moved:
            raise SystemExit("REFUSING: parameter-motion check matched zero parameters")
    z0_moved = any(v for k, v in moved.items()
                   if "latent_branch_trunks.0" in k or "latent_action_heads.0" in k)
    z1_moved = any(v for k, v in moved.items()
                   if "latent_branch_trunks.1" in k or "latent_action_heads.1" in k)
    auditor = state.get("auditor")
    coverage = auditor.coverage(expected_envs=32, min_resets=2) if auditor else None
    mean_jsd = _mean_jsd_from_metrics(Path(cfg.metrics_csv_path))

    if verifying:
        ok = verdict == "COMPLETE" and z0_moved and z1_moved
        print(f"\nLIVE PATHS ({cfg.additional_timesteps} additional steps, no record written)")
        print(f"  z0/z1 actor moved       {z0_moved} / {z1_moved}")
        print(f"  mean z_sep_JSD          {mean_jsd}")
        if coverage:
            print(f"  envs / mismatches       {coverage['envs_observed']}/32, "
                  f"{coverage['total_mismatches']}")
        print(f"\n  COMPOSITION: {'OK' if ok else 'BROKEN'}")
        return 0 if ok else 1

    record.write_text(json.dumps({
        "record": f"z-separation {arm} production run", "status": "FROZEN_RESULT", "utc": _now(),
        "implements": "Z_SEPARATION_SPEC.json", "VERDICT": verdict, "arm": role,
        "termination_reason": reason, "seed": int(cfg.seed),
        "additional_timesteps": int(cfg.additional_timesteps), "steps_advanced": steps_advanced,
        "launch_manifest": json.loads(man_path.read_text(encoding="utf-8")),
        "mean_z_sep_JSD_training_time": mean_jsd,
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
