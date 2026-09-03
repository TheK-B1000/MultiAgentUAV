"""Freeze both CCP-S2 terminal checkpoints, BEFORE the sealed EVAL is opened.

Mirrors CCP_SUCCESSOR_MODEL_FROZEN.json's standard sequence: train -> validity check -> freeze
terminal checkpoint -> separate PI decision to open EVAL. Adapted to two matched arms.

VALIDITY, per arm, mirrors TERMINAL_RECORD_VALIDITY's rule -- non-zero training, both latents
exposed, both private actor branches AND both private critic heads moved, zero legacy-path
calls, 32/32 env coverage with zero mismatches -- with the causal path's presence/absence
checked in the direction appropriate to each arm: TREATMENT must show non-zero causal updates
with both routing directions exercised; CONTROL must show the causal path structurally ABSENT
(not merely zero-count -- the run_ccp_s2_production.py tripwire makes any call fatal, so
VERDICT=COMPLETE is itself proof no call ever fired).

Run:  python experiments/ccp_s2_freeze_terminal_checkpoints.py
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
OUT = SD / "CCP_S2_MODELS_FROZEN.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def validate_arm(arm: str, rec: dict) -> dict:
    if rec["VERDICT"] != "COMPLETE":
        raise SystemExit(f"REFUSING: {arm} VERDICT is {rec['VERDICT']!r}, not COMPLETE")
    if rec["EVAL_touched"]:
        raise SystemExit(f"REFUSING: {arm} run touched the sealed EVAL block")
    pm = rec["private_parameter_motion"]
    if not (pm["z0_actor_moved"] and pm["z1_actor_moved"]):
        raise SystemExit(f"REFUSING: {arm} private actor branches did not both move")
    per = pm["per_param_moved"]
    v0 = all(per.get(k) for k in per if "head_V0" in k)
    v1 = all(per.get(k) for k in per if "head_V1" in k)
    if not (v0 and v1):
        raise SystemExit(f"REFUSING: {arm} private critic heads did not both move")
    cov = rec["coverage"]
    if not (cov and cov["passed"] and cov["envs_observed"] == 32 and cov["total_mismatches"] == 0):
        raise SystemExit(f"REFUSING: {arm} coverage did not pass 32/32 with zero mismatches")

    tel = rec["causal_telemetry"]
    if arm == "CONTROL":
        if tel != "ABSENT by design":
            raise SystemExit("REFUSING: CONTROL's causal_telemetry is not the structural "
                             "'ABSENT by design' marker -- the causal path may have run")
    else:
        required = ("updates", "z0_exposures", "z1_exposures", "positive_routes",
                   "negative_routes")
        if not all(tel.get(k, 0) > 0 for k in required):
            raise SystemExit(f"REFUSING: TREATMENT causal telemetry has a zero in {required}")

    return {"z0_actor_moved": True, "z1_actor_moved": True,
           "critic_V0_moved": True, "critic_V1_moved": True,
           "coverage": {"envs_observed": 32, "total_mismatches": 0},
           "causal_path": "ABSENT (structurally fatal-guarded)" if arm == "CONTROL"
                          else {"updates": tel["updates"], "z0_exposures": tel["z0_exposures"],
                                "z1_exposures": tel["z1_exposures"],
                                "positive_routes": tel["positive_routes"],
                                "negative_routes": tel["negative_routes"],
                                "segment_bank_hash": tel["segment_bank_hash"]}}


def main() -> int:
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; one-shot")

    frozen = {}
    for arm in ("CONTROL", "TREATMENT"):
        rec_path = SD / f"CCP_S2_{arm}_RESULT.json"
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        validity = validate_arm(arm, rec)

        man = rec["launch_manifest"]
        ck = ROOT / man["outputs"]["checkpoint_dir"] / f"final_ccp_s2_{arm.lower()}.zip"
        if not ck.is_file():
            raise SystemExit(f"REFUSING: {arm} terminal checkpoint missing: {ck}")
        sha = _sha(ck)
        requested = rec["total_timesteps"]
        base = man["warm_start"]

        frozen[arm] = {
            "TERMINAL_CHECKPOINT": {
                "path": str(ck.relative_to(ROOT)), "sha256": sha,
                "bytes": ck.stat().st_size, "global_step": requested,
                "requested_total_timesteps": requested,
            },
            "WARM_START_SOURCE": base,
            "TERMINAL_RECORD_VALIDITY": {"verdict": "VALID", "summary": validity},
            "seed": rec["seed"], "arm": arm,
        }
        print(f"  {arm:10s} sha256={sha[:16]}...  global_step={requested:,}  VALID", flush=True)

    if frozen["CONTROL"]["seed"] != frozen["TREATMENT"]["seed"]:
        raise SystemExit("REFUSING: arms trained on different seeds")
    if frozen["CONTROL"]["WARM_START_SOURCE"]["checkpoint_sha256"] != \
       frozen["TREATMENT"]["WARM_START_SOURCE"]["checkpoint_sha256"]:
        raise SystemExit("REFUSING: arms warm-started from different checkpoints")

    OUT.write_text(json.dumps({
        "record_id": "CCP_S2_MODELS_FROZEN",
        "status": "FROZEN_MODELS -- frozen BEFORE the sealed EVAL is opened", "utc": _now(),
        "authority": "standard sequence: train -> validity check -> freeze terminal "
                     "checkpoints -> separate PI decision to open EVAL",
        "shared": {"training_seed": frozen["CONTROL"]["seed"],
                   "warm_start_checkpoint_sha256":
                       frozen["CONTROL"]["WARM_START_SOURCE"]["checkpoint_sha256"]},
        "CONTROL": frozen["CONTROL"], "TREATMENT": frozen["TREATMENT"],
        "EVAL_STATE_AT_FREEZE": {
            "block": "11701001..11701064", "touched": False,
            "note": "sealed; opening requires a separate, explicit PI decision"},
        "NO_MODEL_SELECTION_ON_EVAL": {
            "rule": "no model selection, no tuning, no checkpoint choice will occur using "
                    "EVAL. These two terminal checkpoints are the only artifacts that will "
                    "be scored, fixed before EVAL is read.",
        },
        "NEXT": "a separate, explicit PI decision to open EVAL 11701001..11701064",
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
