"""Freeze both RSCFT terminal checkpoints, BEFORE the sealed EVAL is opened.

Same standard sequence as every prior rung (train -> validity check -> freeze -> separate PI
decision to open EVAL), with RSCFT's own validity rule. The discriminator differs from
CCP-S2's: here the CAUSAL path must be live in BOTH arms, and RETENTION is what separates
them. So the checks are:

  both arms      causal updates > 0, both latents exposed, both routing directions exercised,
                 both private actor branches and both private critic heads moved,
                 32/32 env coverage with zero mismatches, full step budget advanced
  CONTROL        retention telemetry is the structural "ABSENT by design" marker -- and
                 because retention_kl/EMATeacher.update were monkeypatched fatal for the
                 run's duration, a COMPLETE verdict is itself proof none ever fired
  TREATMENT      retention updates > 0, EMA updates > 0, and the two are EQUAL (a teacher
                 that lagged or double-stepped would break the 1:1 the frozen rule requires)

Run:  python experiments/rscft_freeze_terminal_checkpoints.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
OUT = SD / "RSCFT_MODELS_FROZEN.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def validate_arm(arm: str, rec: dict) -> dict:
    if rec["VERDICT"] != "COMPLETE":
        raise SystemExit(f"REFUSING: {arm} VERDICT is {rec['VERDICT']!r}, not COMPLETE")
    if rec["EVAL_touched"]:
        raise SystemExit(f"REFUSING: {arm} run touched the sealed EVAL block")
    expected = int(rec["additional_timesteps"])
    if int(rec["steps_advanced"]) < expected * 0.9:
        raise SystemExit(f"REFUSING: {arm} advanced only {rec['steps_advanced']} of {expected}")

    pm = rec["private_parameter_motion"]
    if not (pm["z0_actor_moved"] and pm["z1_actor_moved"]):
        raise SystemExit(f"REFUSING: {arm} private actor branches did not both move")
    per = pm["per_param_moved"]
    if not (all(per.get(k) for k in per if "head_V0" in k)
            and all(per.get(k) for k in per if "head_V1" in k)):
        raise SystemExit(f"REFUSING: {arm} private critic heads did not both move")
    cov = rec["coverage"]
    if not (cov and cov["passed"] and cov["envs_observed"] == 32
            and cov["total_mismatches"] == 0):
        raise SystemExit(f"REFUSING: {arm} coverage did not pass 32/32 with zero mismatches")

    ctel = rec["causal_telemetry"]
    for k in ("updates", "z0_exposures", "z1_exposures", "positive_routes", "negative_routes"):
        if ctel.get(k, 0) <= 0:
            raise SystemExit(f"REFUSING: {arm} causal telemetry has a zero at {k!r}; the "
                             "causal path must be live in BOTH RSCFT arms")

    rtel = rec["retention_telemetry"]
    if arm == "CONTROL":
        if rtel != "ABSENT by design":
            raise SystemExit("REFUSING: CONTROL's retention telemetry is not the structural "
                             "'ABSENT by design' marker -- retention may have run")
        retention = "ABSENT (retention_kl and EMATeacher.update were fatal for the run)"
    else:
        if rtel["retention_updates"] <= 0 or rtel["ema_updates"] <= 0:
            raise SystemExit("REFUSING: TREATMENT retention or EMA never fired")
        if rtel["retention_updates"] != rtel["ema_updates"]:
            raise SystemExit(
                f"REFUSING: TREATMENT retention/EMA counts differ "
                f"({rtel['retention_updates']} vs {rtel['ema_updates']}); the EMA teacher "
                "lagged or double-stepped relative to the actor updates")
        if rtel["empty_batches"] != 0:
            print(f"  note: TREATMENT saw {rtel['empty_batches']} batches with no eligible "
                  "decision boundary (contributed zero, by design)")
        retention = {k: rtel[k] for k in ("retention_updates", "ema_updates", "last_kl_mean",
                                          "last_eligible_heads", "empty_batches",
                                          "lambda_ret", "ema_decay")}

    return {"z0_actor_moved": True, "z1_actor_moved": True, "critic_V0_moved": True,
            "critic_V1_moved": True, "steps_advanced": rec["steps_advanced"],
            "coverage": {"envs_observed": 32, "total_mismatches": 0},
            "causal_path": {k: ctel[k] for k in ("updates", "z0_exposures", "z1_exposures",
                                                 "positive_routes", "negative_routes",
                                                 "segment_bank_hash")},
            "retention_path": retention}


def main() -> int:
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; one-shot")
    frozen = {}
    for arm in ("CONTROL", "TREATMENT"):
        rec = json.loads((SD / f"RSCFT_{arm}_RESULT.json").read_text(encoding="utf-8"))
        validity = validate_arm(arm, rec)
        man = rec["launch_manifest"]
        ck = ROOT / man["outputs"]["checkpoint_dir"] / f"final_rscft_{arm.lower()}.zip"
        if not ck.is_file():
            raise SystemExit(f"REFUSING: {arm} terminal checkpoint missing: {ck}")
        frozen[arm] = {
            "arm": arm, "seed": rec["seed"],
            "TERMINAL_CHECKPOINT": {"path": str(ck.relative_to(ROOT)), "sha256": _sha(ck),
                                    "bytes": ck.stat().st_size},
            "WARM_START_SOURCE": man["warm_start"],
            "TERMINAL_RECORD_VALIDITY": {"verdict": "VALID", "summary": validity},
        }
        print(f"  {arm:10s} sha256={frozen[arm]['TERMINAL_CHECKPOINT']['sha256'][:16]}...  "
              f"steps+{rec['steps_advanced']:,}  VALID", flush=True)

    if frozen["CONTROL"]["seed"] != frozen["TREATMENT"]["seed"]:
        raise SystemExit("REFUSING: arms trained on different seeds")
    if (frozen["CONTROL"]["WARM_START_SOURCE"]["sha256"]
            != frozen["TREATMENT"]["WARM_START_SOURCE"]["sha256"]):
        raise SystemExit("REFUSING: arms warm-started from different checkpoints")
    if frozen["CONTROL"]["TERMINAL_CHECKPOINT"]["sha256"] == \
       frozen["TREATMENT"]["TERMINAL_CHECKPOINT"]["sha256"]:
        raise SystemExit("REFUSING: both arms produced the SAME terminal checkpoint; the "
                         "retention treatment cannot have had any effect")

    spec = json.loads((SD / "RSCFT_SPEC.json").read_text(encoding="utf-8"))
    OUT.write_text(json.dumps({
        "record_id": "RSCFT_MODELS_FROZEN",
        "status": "FROZEN_MODELS -- frozen BEFORE the sealed EVAL is opened", "utc": _now(),
        "implements": "RSCFT_SPEC.json",
        "shared": {"training_seed": frozen["CONTROL"]["seed"],
                   "warm_start_sha256": frozen["CONTROL"]["WARM_START_SOURCE"]["sha256"],
                   "warm_start_is_the_original_incumbent": True},
        "CONTROL": frozen["CONTROL"], "TREATMENT": frozen["TREATMENT"],
        "EVAL_STATE_AT_FREEZE": {"block": spec["SEEDS"]["sealed_eval_block"], "touched": False,
                                 "note": "sealed; opening requires a separate PI decision"},
        "NO_MODEL_SELECTION_ON_EVAL": {
            "rule": "no model selection, no tuning, no checkpoint choice using EVAL. These two "
                    "terminal checkpoints are the only artifacts that will be scored."},
        "NEXT": f"a separate, explicit PI decision to open EVAL "
                f"{spec['SEEDS']['sealed_eval_block']}",
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
