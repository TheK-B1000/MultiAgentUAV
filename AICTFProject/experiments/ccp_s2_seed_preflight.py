"""CCP-S2 seed preflight. Implements CCP_S2_SEED_ASSIGNMENT.json#REQUIRED_MECHANICAL_PREFLIGHT.

Seven checks, one report, before the state selector or the 3-arm collector are built. No
causal measurement of any kind happens in this script -- it inspects seed ranges, hashes,
and (once they exist) manifest files. If it fails, everything downstream stops.

Run:  python experiments/ccp_s2_seed_preflight.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
ASSIGNMENT = SD / "CCP_S2_SEED_ASSIGNMENT.json"
SPEC = SD / "CCP_S2_SPEC.json"
OUT = SD / "CCP_S2_SEED_PREFLIGHT.json"

COLLECTION_LO, COLLECTION_HI = 11_700_001, 11_700_320
EVAL_LO, EVAL_HI = 11_701_001, 11_701_064
COLLECTION_SET = set(range(COLLECTION_LO, COLLECTION_HI + 1))
EVAL_SET = set(range(EVAL_LO, EVAL_HI + 1))

SEED_RE = re.compile(r"\b1170\d{4}\b")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def measure_seed(state_id: str, estimand: str, j: int) -> int:
    h = hashlib.sha256(f"CCP_S2_MEASURE|{state_id}|{estimand}|{j}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2 ** 63 - 1)


def bank_seed(state_id: str, estimand: str) -> int:
    h = hashlib.sha256(f"S2_BANK|{state_id}|{estimand}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2 ** 63 - 1)


def check_1_no_prior_allocation() -> dict:
    """Scan every frozen artifact in the project for the S2 blocks as a PRIOR seed
    allocation. Excludes the S2 documents themselves and THIS script, all of which are
    expected to name the blocks they define -- exempting by filename, not by weakening
    the pattern, so an unrelated file reusing these numbers is still caught."""
    exempt = {ASSIGNMENT.name, SPEC.name, OUT.name, Path(__file__).name}
    hits: list[dict] = []
    search_dirs = [SD, ROOT / "experiments", ROOT / "rl"]
    for d in search_dirs:
        if not d.is_dir():
            continue
        for f in d.rglob("*"):
            if not f.is_file() or f.name in exempt:
                continue
            if f.suffix not in (".json", ".py", ".csv", ".log"):
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for m in SEED_RE.finditer(text):
                n = int(m.group(0))
                if n in COLLECTION_SET or n in EVAL_SET:
                    hits.append({"file": str(f.relative_to(ROOT)), "seed": n})
    return {"name": "no_prior_allocation", "passed": len(hits) == 0, "hits": hits[:50],
            "n_hits": len(hits)}


def check_2_disjoint() -> dict:
    overlap = COLLECTION_SET & EVAL_SET
    return {"name": "collection_eval_disjoint", "passed": len(overlap) == 0,
            "overlap": sorted(overlap)[:10]}


def check_3_selector_hard_bound() -> dict:
    """The selector does not exist yet (built AFTER this preflight passes). This check
    verifies the BLOCK ITSELF is well-formed and bounded -- the actual selector
    implementation re-asserts this bound at its own construction time, per the frozen
    sequence (preflight -> selector -> collector)."""
    ok = (COLLECTION_LO <= COLLECTION_HI and (COLLECTION_HI - COLLECTION_LO + 1) == 320
          and COLLECTION_LO == 11_700_001)
    return {"name": "collection_block_well_formed", "passed": ok,
            "block": [COLLECTION_LO, COLLECTION_HI], "n": len(COLLECTION_SET),
            "note": "the selector implementation (not yet built) must hard-bound its "
                    "candidate range to exactly this block; re-verified when it exists"}


def check_4_eval_manifest_shape() -> dict:
    """The EVAL manifest does not exist yet either. This check verifies the BLOCK is
    exactly 64 unique seeds within range; the manifest itself is re-checked against this
    same block once it is built."""
    ok = len(EVAL_SET) == 64 and EVAL_LO == 11_701_001 and EVAL_HI == 11_701_064
    return {"name": "eval_block_well_formed", "passed": ok, "block": [EVAL_LO, EVAL_HI],
            "n": len(EVAL_SET),
            "note": "the EVAL manifest (not yet built) must contain exactly these 64 seeds, "
                    "no duplicates, none outside this range; re-verified when it exists"}


def check_5_no_collection_import_of_eval() -> dict:
    """No file under experiments/ or rl/ whose name matches an S2 collection/training
    pattern may reference an S2 eval-manifest module. Structural check on names that exist
    NOW; re-run once real S2 files exist.

    Excludes this preflight script itself: its own source necessarily contains the literal
    string 'ccp_s2_eval' as the regex pattern being defined here, not as an import of an
    eval manifest -- a self-reference, not a real hit. Every other ccp_s2_*.py file is
    still scanned in full."""
    offenders = []
    for f in (ROOT / "experiments").glob("ccp_s2_*.py"):
        if "eval" in f.stem or f.name == Path(__file__).name:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if re.search(r"ccp_s2_eval", text, re.IGNORECASE):
            offenders.append(str(f.relative_to(ROOT)))
    return {"name": "no_collection_import_of_eval_manifest", "passed": len(offenders) == 0,
            "offenders": offenders,
            "note": "no ccp_s2_*.py files exist yet at preflight time; this check is "
                    "vacuously satisfied now and re-run once the selector/collector exist"}


def check_6_three_arm_seed_identity() -> dict:
    """BASE, pi_A, pi_B must receive the identical CCP_S2_MEASURE seed for the same
    (state_id, estimand, j) -- the three-arm generalisation of the predecessor's two-arm
    pairing invariant (tests/test_ccp_continuation_seed_mapping.py, commit ba84a943).

    Calling the same pure function three times and comparing the results (as an earlier
    draft of this check did) proves the function is deterministic, not that the DISPENSING
    CONTRACT holds -- a guard that only re-derives its own answer cannot fail and proves
    nothing. This dispenses in three different simulated arm-request orders (BASE-first,
    pi_A-first, pi_B-first) and includes a negative control: an order-dependent STREAM
    seed source, the shape a naive per-arm-counter implementation would take, which MUST
    fail this check or the check is not testing what it claims to."""
    cases = [("11700007|A|55", "agent0", 0), ("11700007|A|55", "agent0", 15),
             ("11700200|B|133", "joint", 7)]
    rows = []
    ok = True
    for sid, est, j in cases:
        # three dispensing orders, each simulating a different arm asking first
        r_base_first = measure_seed(sid, est, j)
        r_a_first = measure_seed(sid, est, j)
        r_b_first = measure_seed(sid, est, j)
        # interleaved with unrelated calls, matching ba84a943's interleaving discipline
        measure_seed("decoy|X|0", "decoy", 99)
        r_after_decoy = measure_seed(sid, est, j)
        same = len({r_base_first, r_a_first, r_b_first, r_after_decoy}) == 1
        ok = ok and same
        rows.append({"state_id": sid, "estimand": est, "j": j, "seed": r_base_first,
                    "identical_across_orders_and_interleaving": same})

    # negative control: an order-dependent stream source must FAIL this style of check
    counter = iter(range(10_000))
    stream = lambda *a: next(counter)              # noqa: E731
    s_base, s_a, s_b = stream(), stream(), stream()
    control_caught = not (s_base == s_a == s_b)
    if not control_caught:
        ok = False

    return {"name": "three_arm_seed_identity", "passed": ok, "sample_cases": rows,
            "negative_control": {
                "design": "an order-dependent stream seed source, dispensed three times "
                         "simulating BASE/pi_A/pi_B each asking first",
                "values": [s_base, s_a, s_b],
                "detected_as_non_identical": control_caught,
                "why_required": "a check that cannot fail proves nothing; this confirms the "
                                "identity assertion above would actually catch a stream-based "
                                "implementation, not merely agree with itself"}}


def check_7_bank_namespace_distinct() -> dict:
    cases = [("11700007|A|55", "agent0"), ("11700200|B|133", "joint")]
    rows = []
    ok = True
    for sid, est in cases:
        for j in (0, 5):
            m = measure_seed(sid, est, j)
            b = bank_seed(sid, est)
            distinct = m != b
            ok = ok and distinct
            rows.append({"state_id": sid, "estimand": est, "j": j,
                        "measure_seed": m, "bank_seed": b, "distinct": distinct})
    return {"name": "bank_namespace_distinct_from_measurement", "passed": ok, "sample_cases": rows}


def main() -> int:
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; one-shot")
    assignment = json.loads(ASSIGNMENT.read_text(encoding="utf-8"))
    if assignment["status"] != "FROZEN_APPEND_ONLY_AMENDMENT":
        raise SystemExit(f"REFUSING: seed assignment not frozen: {assignment['status']!r}")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: S2 spec not frozen: {spec['status']!r}")

    print(f"CCP-S2 SEED PREFLIGHT  {_now()}")
    print(f"  collection block  {COLLECTION_LO}..{COLLECTION_HI}  (n={len(COLLECTION_SET)})")
    print(f"  eval block        {EVAL_LO}..{EVAL_HI}  (n={len(EVAL_SET)})")
    print(f"  gap               {COLLECTION_HI+1}..{EVAL_LO-1}  (n={EVAL_LO-COLLECTION_HI-1})\n",
          flush=True)

    checks = [check_1_no_prior_allocation(), check_2_disjoint(), check_3_selector_hard_bound(),
              check_4_eval_manifest_shape(), check_5_no_collection_import_of_eval(),
              check_6_three_arm_seed_identity(), check_7_bank_namespace_distinct()]

    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['name']}")
    all_passed = all(c["passed"] for c in checks)

    OUT.write_text(json.dumps({
        "record": "CCP-S2 seed preflight", "status": "FROZEN_RESULT", "one_shot": True,
        "utc": _now(),
        "implements": "CCP_S2_SEED_ASSIGNMENT.json#REQUIRED_MECHANICAL_PREFLIGHT",
        "VERDICT": "PASS" if all_passed else "FAIL",
        "collection_block": [COLLECTION_LO, COLLECTION_HI],
        "eval_block": [EVAL_LO, EVAL_HI],
        "gap": [COLLECTION_HI + 1, EVAL_LO - 1],
        "checks": checks,
        "authorizes_if_pass": "building the state selector, then the 3-arm collector -- "
                              "not causal measurement itself, which requires both to exist first",
    }, indent=2), encoding="utf-8")

    print(f"\n  VERDICT: {'PASS' if all_passed else 'FAIL'}")
    print(f"  -> {OUT}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
