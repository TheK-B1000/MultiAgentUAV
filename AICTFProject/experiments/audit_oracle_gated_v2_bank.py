"""One-shot V2 rehearsal bank assembly audit.

Counts eligible (Delta != 0) states from:
  - legacy FIT stratified shards 10700001..10700096
  - new V2 collection shards 11000001..11000320

PASS iff total eligible >= 1500 (frozen minimum). Does not open EVAL.

Run:  python experiments/audit_oracle_gated_v2_bank.py
"""
from __future__ import annotations

import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand"
SPEC = SD / "sppo" / "ORACLE_GATED_K2_V2_RUN_SPEC.json"
PROTOCOL = SD / "sppo" / "ORACLE_GATED_K2_V2_COLLECTION_PROTOCOL.json"
LEGACY = SD / "stratified_regime_data"
V2_DATA = SD / "sppo" / "oracle_gated_k2_v2_bank_data"
V2_COMPLETE = V2_DATA / "COLLECTION_COMPLETE.json"
OUT = SD / "sppo" / "ORACLE_GATED_K2_V2_BANK_ASSEMBLY.json"

FIT_LO, FIT_HI = 10_700_001, 10_700_096
V2_LO, V2_HI = 11_000_001, 11_000_320
MIN_ELIGIBLE = 1500


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _count_eligible(shards_dir: Path, lo: int, hi: int) -> dict:
    total, eligible, a_pref, b_pref = 0, 0, 0, 0
    seeds = 0
    for seed in range(lo, hi + 1):
        path = shards_dir / f"seed_{seed}.npz"
        if not path.is_file():
            continue
        seeds += 1
        with np.load(path, allow_pickle=False) as z:
            d = ((z["branch_pi_B_blue"].astype(np.int64) - z["branch_pi_B_red"].astype(np.int64))
                 - (z["branch_pi_A_blue"].astype(np.int64) - z["branch_pi_A_red"].astype(np.int64)))
            total += len(d)
            mask = d != 0
            eligible += int(mask.sum())
            a_pref += int((d < 0).sum())
            b_pref += int((d > 0).sum())
    return {
        "seeds_present": seeds,
        "branch_states": total,
        "eligible": eligible,
        "A_preferred": a_pref,
        "B_preferred": b_pref,
        "tied": total - eligible,
    }


def main() -> int:
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; one-shot audit")
    if not V2_COMPLETE.is_file():
        raise SystemExit(f"REFUSING: {V2_COMPLETE} missing; finish collection first")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if int(spec["REHEARSAL_BANK"]["minimum_total_eligible"]) != MIN_ELIGIBLE:
        raise SystemExit("REFUSING: minimum eligible drifted in run spec")

    legacy = _count_eligible(LEGACY / "seed_shards", FIT_LO, FIT_HI)
    v2 = _count_eligible(V2_DATA / "seed_shards", V2_LO, V2_HI)
    total_eligible = legacy["eligible"] + v2["eligible"]
    passed = total_eligible >= MIN_ELIGIBLE

    record = {
        "record": "Oracle-gated K=2 V2 bank assembly audit",
        "status": "FROZEN_RESULT",
        "one_shot": True,
        "utc": _now(),
        "implements": [str(SPEC.relative_to(ROOT)), str(PROTOCOL.relative_to(ROOT))],
        "minimum_total_eligible": MIN_ELIGIBLE,
        "legacy_FIT": {"seed_range": [FIT_LO, FIT_HI], **legacy},
        "new_V2_collection": {"seed_range": [V2_LO, V2_HI], **v2},
        "combined": {
            "eligible": total_eligible,
            "A_preferred": legacy["A_preferred"] + v2["A_preferred"],
            "B_preferred": legacy["B_preferred"] + v2["B_preferred"],
        },
        "VERDICT": "PASS" if passed else "BLOCKED",
        "consequence": (
            "V2 training authorized" if passed else
            "V2 training BLOCKED; new prospective collection decision required"
        ),
    }
    OUT.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"V2 BANK ASSEMBLY  {_now()}")
    print(f"  legacy FIT eligible {legacy['eligible']}")
    print(f"  new V2 eligible    {v2['eligible']}")
    print(f"  total eligible     {total_eligible}  (min {MIN_ELIGIBLE})")
    print(f"  VERDICT: {record['VERDICT']}")
    print(f"  -> {OUT}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
