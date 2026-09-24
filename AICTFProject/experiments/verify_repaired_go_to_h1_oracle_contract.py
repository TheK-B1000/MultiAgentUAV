"""Rule-12 gate for REPAIRED_GO_TO_H1_PROJECTED vs sealed oracle ORIGINAL.

ORIGINAL must be unaffected by macro_commit_go_to_ticks. If any overlapping
4v4 ORIGINAL seed disagrees with PROJECTED_TEACHER_ORACLE, refuse interpretation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
ORACLE_ROWS = SD / "projected_teacher_oracle_rows.csv"
REPAIR_ROWS = SD / "repaired_go_to_h1_projected_rows.csv"
OUT_OK = SD / "REPAIRED_GO_TO_H1_PROJECTED_SEALED_READING.json"
OUT_BAD = SD / "REPAIRED_GO_TO_H1_PROJECTED_INTEGRITY_REQUIRED.json"


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _key(r: dict) -> tuple:
    return (r["scale"], r["strategy"], r["pole"], r["arm"], int(r["seed"]))


def main() -> int:
    if not REPAIR_ROWS.is_file():
        raise SystemExit(f"REFUSING: missing {REPAIR_ROWS.name}")
    if not ORACLE_ROWS.is_file():
        raise SystemExit(f"REFUSING: missing {ORACLE_ROWS.name}")

    repair = _load(REPAIR_ROWS)
    oracle = {_key(r): r for r in _load(ORACLE_ROWS)
              if r["scale"] == "4v4" and r["arm"] == "ORIGINAL"}

    mismatches = []
    compared = 0
    for r in repair:
        if r["scale"] != "4v4" or r["arm"] != "ORIGINAL":
            continue
        k = _key(r)
        o = oracle.get(k)
        if o is None:
            mismatches.append({"key": list(k), "reason": "missing_in_oracle"})
            continue
        compared += 1
        if int(r["win"]) != int(o["win"]) or int(r["blue"]) != int(o["blue"]) or int(r["red"]) != int(o["red"]):
            mismatches.append({
                "key": list(k),
                "repair": {"win": int(r["win"]), "blue": int(r["blue"]), "red": int(r["red"])},
                "oracle": {"win": int(o["win"]), "blue": int(o["blue"]), "red": int(o["red"])},
            })

    # Load latest repair RESULT for gate numbers if present.
    results = sorted(SD.glob("REPAIRED_GO_TO_H1_PROJECTED_*_RESULT.json"))
    result = json.loads(results[-1].read_text(encoding="utf-8")) if results else None
    cell = (result or {}).get("results", {}).get("4v4", {})

    if mismatches:
        OUT_OK.unlink(missing_ok=True)
        rec = {
            "record": "REPAIRED_GO_TO_H1_PROJECTED integrity failure",
            "status": "INTEGRITY_REQUIRED",
            "utc": _now(),
            "rule12_original_contract": {
                "passed": False,
                "compared": compared,
                "mismatches": len(mismatches),
                "examples": mismatches[:12],
            },
            "action": "STOP interpretation; audit harness. Do not claim GO_TO repair.",
        }
        OUT_BAD.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"CONTRACT FAIL -- {len(mismatches)} mismatches / {compared} compared")
        print(f"  -> {OUT_BAD}")
        return 2

    OUT_BAD.unlink(missing_ok=True)
    gate_o = cell.get("gate_ORIGINAL")
    gate_p = cell.get("gate_PROJECTED")
    d_p = cell.get("delta_PROJECTED", {})
    rec = {
        "record": "REPAIRED_GO_TO_H1_PROJECTED sealed reading",
        "status": "FROZEN_RESULT",
        "utc": _now(),
        "implements": ["REPAIRED_GO_TO_H1_PROJECTED_SPEC.json"],
        "study_class": "MECHANISTIC_FOLLOW_UP",
        "not": "INDEPENDENT_CONFIRMATION",
        "rule12_original_contract": {
            "passed": True,
            "compared": compared,
            "mismatches": 0,
            "equality": "REPAIR.ORIGINAL == ORACLE.ORIGINAL (4v4, all strategies/poles)",
        },
        "primary_gate": "LCB95(delta_A)>0 AND LCB95(delta_B)>0 under PROJECTED h_GO_TO=1",
        "gate_ORIGINAL": gate_o,
        "gate_PROJECTED": gate_p,
        "delta_PROJECTED": d_p,
        "baseline_h4": "Compare against sealed PROJECTED_TEACHER_ORACLE PROJECTED deltas (same seeds).",
        "result_points_to": results[-1].name if results else None,
    }
    OUT_OK.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"CONTRACT PASS -- compared={compared}")
    print(f"  gate_ORIGINAL={gate_o}  gate_PROJECTED={gate_p}")
    if d_p:
        da, db = d_p.get("delta_A"), d_p.get("delta_B")
        if da and db:
            print(f"  PROJECTED h=1  delta_A={da['mean']:+.4f} "
                  f"[{da['lcb95']:+.4f},{da['ucb95']:+.4f}]"
                  f"  delta_B={db['mean']:+.4f} "
                  f"[{db['lcb95']:+.4f},{db['ucb95']:+.4f}]")
    print(f"  -> {OUT_OK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
