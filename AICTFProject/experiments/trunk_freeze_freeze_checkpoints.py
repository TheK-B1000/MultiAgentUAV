"""Freeze both trunk-freeze terminal checkpoints, BEFORE the sealed EVAL is opened.

Same standard sequence as every prior rung. Validity rule mirrors RSCFT's (causal path
checked in the direction appropriate to each arm), PLUS a check unique to this experiment:
the trunk-freeze verification itself must show zero frozen-parameter motion across the WHOLE
run, not just the short wiring check.

Run:  python experiments/trunk_freeze_freeze_checkpoints.py
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
OUT = SD / "TRUNK_FREEZE_MODELS_FROZEN.json"


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

    tf = rec["trunk_freeze_verification"]
    if tf is None or tf["moved_frozen"]:
        raise SystemExit(f"REFUSING: {arm} shows frozen-parameter motion: "
                         f"{tf['moved_frozen'] if tf else 'no verification recorded'}")
    if len(tf["moved_trainable"]) == 0:
        raise SystemExit(f"REFUSING: {arm} shows zero trainable parameter motion -- the "
                         "positive control failed")

    pm = rec["private_parameter_motion"]
    if not (pm["z0_actor_moved"] and pm["z1_actor_moved"]):
        raise SystemExit(f"REFUSING: {arm} private actor branches did not both move")
    cov = rec["coverage"]
    if not (cov and cov["passed"] and cov["envs_observed"] == 32
            and cov["total_mismatches"] == 0):
        raise SystemExit(f"REFUSING: {arm} coverage did not pass 32/32 with zero mismatches")

    ctel = rec["causal_telemetry"]
    if arm == "CONTROL":
        if ctel != "ABSENT by design":
            raise SystemExit("REFUSING: CONTROL's causal_telemetry is not the structural "
                             "'ABSENT by design' marker")
        causal = "ABSENT (causal_supervision_loss was fatal for the run)"
    else:
        for k in ("updates", "z0_exposures", "z1_exposures", "positive_routes", "negative_routes"):
            if ctel.get(k, 0) <= 0:
                raise SystemExit(f"REFUSING: {arm} causal telemetry has a zero at {k!r}")
        causal = {k: ctel[k] for k in ("updates", "z0_exposures", "z1_exposures",
                                       "positive_routes", "negative_routes")}

    return {"steps_advanced": rec["steps_advanced"],
            "trunk_freeze": {"n_frozen_verified_static": tf["moved_frozen"] == [],
                             "n_trainable_moved": len(tf["moved_trainable"])},
            "z0_actor_moved": True, "z1_actor_moved": True,
            "coverage": {"envs_observed": 32, "total_mismatches": 0}, "causal_path": causal}


def main() -> int:
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; one-shot")
    frozen = {}
    for arm in ("CONTROL", "TREATMENT"):
        rec = json.loads((SD / f"TRUNK_FREEZE_{arm}_RESULT.json").read_text(encoding="utf-8"))
        validity = validate_arm(arm, rec)
        man = rec["launch_manifest"]
        ck = ROOT / man["outputs"]["checkpoint_dir"] / f"final_trunk_freeze_{arm.lower()}.zip"
        if not ck.is_file():
            raise SystemExit(f"REFUSING: {arm} terminal checkpoint missing: {ck}")
        frozen[arm] = {"arm": arm, "seed": rec["seed"],
                       "TERMINAL_CHECKPOINT": {"path": str(ck.relative_to(ROOT)),
                                               "sha256": _sha(ck), "bytes": ck.stat().st_size},
                       "WARM_START_SOURCE": man["warm_start"],
                       "TERMINAL_RECORD_VALIDITY": {"verdict": "VALID", "summary": validity}}
        print(f"  {arm:10s} sha256={frozen[arm]['TERMINAL_CHECKPOINT']['sha256'][:16]}...  "
              f"steps+{rec['steps_advanced']:,}  frozen_params_static={validity['trunk_freeze']}"
              f"  VALID", flush=True)

    if frozen["CONTROL"]["seed"] != frozen["TREATMENT"]["seed"]:
        raise SystemExit("REFUSING: arms trained on different seeds")
    if (frozen["CONTROL"]["WARM_START_SOURCE"]["sha256"]
            != frozen["TREATMENT"]["WARM_START_SOURCE"]["sha256"]):
        raise SystemExit("REFUSING: arms warm-started from different checkpoints")
    if frozen["CONTROL"]["TERMINAL_CHECKPOINT"]["sha256"] == \
       frozen["TREATMENT"]["TERMINAL_CHECKPOINT"]["sha256"]:
        raise SystemExit("REFUSING: both arms produced the SAME terminal checkpoint")

    spec = json.loads((SD / "TRUNK_FREEZE_SPEC.json").read_text(encoding="utf-8"))
    OUT.write_text(json.dumps({
        "record_id": "TRUNK_FREEZE_MODELS_FROZEN",
        "status": "FROZEN_MODELS -- frozen BEFORE the sealed EVAL is opened", "utc": _now(),
        "implements": "TRUNK_FREEZE_SPEC.json",
        "shared": {"training_seed": frozen["CONTROL"]["seed"],
                   "warm_start_sha256": frozen["CONTROL"]["WARM_START_SOURCE"]["sha256"],
                   "warm_start_is_the_original_incumbent": True},
        "CONTROL": frozen["CONTROL"], "TREATMENT": frozen["TREATMENT"],
        "EVAL_STATE_AT_FREEZE": {"block": spec["SEEDS"]["sealed_eval_block"], "touched": False,
                                 "note": "sealed; opening requires a separate PI decision"},
        "NO_MODEL_SELECTION_ON_EVAL": {
            "rule": "no model selection, no tuning, no checkpoint choice using EVAL. These "
                    "two terminal checkpoints are the only artifacts that will be scored."},
        "NEXT": f"a separate, explicit PI decision to open EVAL "
                f"{spec['SEEDS']['sealed_eval_block']}",
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
