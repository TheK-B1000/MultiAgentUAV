"""Freeze both BCS terminal checkpoints, BEFORE the sealed EVAL is opened.

Same standard sequence as every prior rung. Validity rule mirrors trunk-freeze's, PLUS a check
unique to this experiment: TREATMENT's balance_check must show exact per-run z0/z1 exposure
equality (not just nonzero causal telemetry) -- that exact balance is the whole point of this
arm and must be verified from the real production run, not assumed from the wiring check alone.

Run:  python experiments/bcs_freeze_terminal_checkpoints.py
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
OUT = SD / "BCS_MODELS_FROZEN.json"


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
    cov = rec["coverage"]
    if not (cov and cov["passed"] and cov["envs_observed"] == 32
            and cov["total_mismatches"] == 0):
        raise SystemExit(f"REFUSING: {arm} coverage did not pass 32/32 with zero mismatches")

    ctel = rec["causal_telemetry"]
    if ctel is None:
        raise SystemExit(f"REFUSING: {arm} has no causal telemetry at all")
    for k in ("updates", "z0_exposures", "z1_exposures", "positive_routes", "negative_routes"):
        if ctel.get(k, 0) <= 0:
            raise SystemExit(f"REFUSING: {arm} causal telemetry has a zero at {k!r}")

    if arm == "TREATMENT":
        bc = rec.get("balance_check")
        if not bc:
            raise SystemExit("REFUSING: TREATMENT has no balance_check recorded")
        if bc["abs_diff"] != 0:
            raise SystemExit(f"REFUSING: TREATMENT balance_check shows a nonzero exposure "
                             f"difference: {bc}")
        if bc["z0_exposures"] != ctel["z0_exposures"] or bc["z1_exposures"] != ctel["z1_exposures"]:
            raise SystemExit("REFUSING: TREATMENT balance_check does not match causal_telemetry")
    else:
        z0e, z1e = ctel["z0_exposures"], ctel["z1_exposures"]
        ratio = z0e / max(1, z0e + z1e)
        if not (0.50 < ratio < 0.75):
            raise SystemExit(f"REFUSING: CONTROL z0 exposure ratio {ratio:.3f} is outside the "
                             "historically-expected imbalanced range (0.50, 0.75) -- either the "
                             "unbalanced sampler changed behavior or something else did")

    return {"steps_advanced": rec["steps_advanced"],
            "z0_actor_moved": True, "z1_actor_moved": True,
            "coverage": {"envs_observed": 32, "total_mismatches": 0},
            "causal_telemetry_summary": {k: ctel[k] for k in
                                         ("updates", "z0_exposures", "z1_exposures",
                                          "positive_routes", "negative_routes")},
            "balance_check": rec.get("balance_check")}


def main() -> int:
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; one-shot")
    frozen = {}
    for arm_lower, arm in (("control", "CONTROL"), ("treatment", "TREATMENT")):
        rec = json.loads((SD / f"BCS_{arm}_RESULT.json").read_text(encoding="utf-8"))
        validity = validate_arm(arm, rec)
        man = rec["launch_manifest"]
        ck_dir = ROOT / man["outputs"]["checkpoint_dir"]
        ck = ck_dir / f"final_balanced_causal_sampling_{arm_lower}.zip"
        if not ck.is_file():
            alt = list(ck_dir.glob("final_*.zip"))
            if len(alt) != 1:
                raise SystemExit(f"REFUSING: cannot locate unique terminal checkpoint under "
                                 f"{ck_dir}: {[p.name for p in alt]}")
            ck = alt[0]
        frozen[arm] = {"arm": arm, "seed": rec["seed"],
                       "TERMINAL_CHECKPOINT": {"path": str(ck.relative_to(ROOT)),
                                               "sha256": _sha(ck), "bytes": ck.stat().st_size},
                       "WARM_START_SOURCE": man["warm_start"],
                       "TERMINAL_RECORD_VALIDITY": {"verdict": "VALID", "summary": validity}}
        print(f"  {arm:10s} sha256={frozen[arm]['TERMINAL_CHECKPOINT']['sha256'][:16]}...  "
              f"steps+{rec['steps_advanced']:,}  "
              f"z0/z1_exposures={validity['causal_telemetry_summary']['z0_exposures']}/"
              f"{validity['causal_telemetry_summary']['z1_exposures']}  VALID", flush=True)

    if frozen["CONTROL"]["seed"] != frozen["TREATMENT"]["seed"]:
        raise SystemExit("REFUSING: arms trained on different seeds")
    if (frozen["CONTROL"]["WARM_START_SOURCE"]["checkpoint_sha256"]
            if "checkpoint_sha256" in frozen["CONTROL"]["WARM_START_SOURCE"]
            else frozen["CONTROL"]["WARM_START_SOURCE"]["sha256"]) != (
            frozen["TREATMENT"]["WARM_START_SOURCE"]["checkpoint_sha256"]
            if "checkpoint_sha256" in frozen["TREATMENT"]["WARM_START_SOURCE"]
            else frozen["TREATMENT"]["WARM_START_SOURCE"]["sha256"]):
        raise SystemExit("REFUSING: arms warm-started from different checkpoints")
    if frozen["CONTROL"]["TERMINAL_CHECKPOINT"]["sha256"] == \
       frozen["TREATMENT"]["TERMINAL_CHECKPOINT"]["sha256"]:
        raise SystemExit("REFUSING: both arms produced the SAME terminal checkpoint")

    spec = json.loads((SD / "BALANCED_CAUSAL_SAMPLING_SPEC.json").read_text(encoding="utf-8"))
    OUT.write_text(json.dumps({
        "record_id": "BCS_MODELS_FROZEN",
        "status": "FROZEN_MODELS -- frozen BEFORE the sealed EVAL is opened", "utc": _now(),
        "implements": "BALANCED_CAUSAL_SAMPLING_SPEC.json",
        "shared": {"training_seed": frozen["CONTROL"]["seed"],
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
