"""DEFEND_ATTACK_SPLIT_POLICY_A_V1_CONFIRMATORY_V1_SPEC.json.

Contracts-only runner for CONTRACTS_before_seed_spend (C0-C3). Spends no
seed, writes no checkpoint. This is a CONFIRMATORY replication of an
already-sealed exploratory result on fresh matched seeds -- no training, no
architecture change, no new code path beyond what
DEFEND_ATTACK_SPLIT_POLICY_A_V1's own exploratory pass already exercised
and contracted (C0-C13 in
run_defend_attack_split_policy_contracts_4v4.py). This script checks only
what is NEW to the confirmatory pass: the three frozen checkpoints are
exactly the ones the exploratory result scored, and the proposed seed
block is genuinely fresh (no overlap with the exploratory or training
seeds).

  contracts   The only stage this script implements: C0-C3 below.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEM = "DEFEND_ATTACK_SPLIT_POLICY_A_V1_CONFIRMATORY_V1"
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC_PATH = SD / f"{STEM}_SPEC.json"
CONTRACT_PATH = SD / f"{STEM}_CONTRACT_RESULT.json"

EXPLORATORY_BLOCK = (21_700_001, 21_700_064)
TRAINING_SEED = 21_600_001
CONFIRMATORY_BLOCK = (21_800_001, 21_800_128)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def run_contracts() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    # ---- C0: spec frozen -----------------------------------------------
    spec = _load_json(SPEC_PATH)
    add(
        "C0_SPEC_FROZEN",
        spec.get("status") == "FROZEN_BEFORE_SEED_SPEND",
        f"spec status = {spec.get('status')!r}",
    )

    # ---- C1: all three checkpoints equal their pins ----------------------
    assets = spec.get("FROZEN_ASSETS_locked", {})
    all_ok = True
    details = []
    for key in ("pi_A_frozen_attack", "pi_D_trained_defend", "pi_B_comparator"):
        entry = assets.get(key, {})
        p = ROOT / str(entry.get("path", ""))
        exists = p.is_file()
        actual = _sha256(p) if exists else ""
        pin = str(entry.get("sha256", ""))
        ok = exists and actual == pin
        all_ok = all_ok and ok
        details.append(f"{key}: exists={exists} matches_pin={actual == pin}")
    add("C1_ALL_THREE_CHECKPOINTS_EQUAL_PINS", all_ok, "; ".join(details))

    # ---- C2: fresh seed block, no overlap with prior reservations -------
    lo, hi = CONFIRMATORY_BLOCK
    overlaps_exploratory = not (hi < EXPLORATORY_BLOCK[0] or lo > EXPLORATORY_BLOCK[1])
    overlaps_training = lo <= TRAINING_SEED <= hi
    fresh = not overlaps_exploratory and not overlaps_training
    add(
        "C2_SEED_BLOCK_IS_FRESH_NO_OVERLAP",
        fresh,
        f"proposed=[{lo},{hi}] exploratory=[{EXPLORATORY_BLOCK[0]},{EXPLORATORY_BLOCK[1]}] "
        f"training={TRAINING_SEED} overlaps_exploratory={overlaps_exploratory} "
        f"overlaps_training={overlaps_training}",
    )
    from experiments import seed_registry as sr

    reg_ok, reg_msg = sr.check_block(lo, hi, "sealed_confirmatory", experiment_id=f"{STEM}_EVAL")
    add("C2b_SEED_REGISTRY_CONFIRMS_BLOCK_IS_FREE", reg_ok, reg_msg)

    # ---- C3: read-only fence (no training/checkpoint-writing call) ------
    src = Path(__file__).read_text(encoding="utf-8")
    forbidden = [".le" + "arn(", "optimizer" + ".step(", "torch.sa" + "ve("]
    hits = [t for t in forbidden if t in src]
    add(
        "C3_CONTRACTS_ARE_READ_ONLY",
        not hits,
        "no training/checkpoint-writing call in this script's own source"
        if not hits else f"found {hits}",
    )

    decision = "CONTRACTS_PASS" if all(c["pass"] for c in checks) else "CONTRACTS_FAIL"
    result = {
        "record_id": f"{STEM}_CONTRACT_RESULT",
        "implements": SPEC_PATH.name,
        "utc": _now(),
        "DECISION": decision,
        "spec_sha256": _sha256(SPEC_PATH),
        "script_sha256": _sha256(Path(__file__)),
        "checks": checks,
    }
    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"\n  DEFEND_ATTACK_SPLIT_POLICY_A_CONFIRMATORY CONTRACTS: {decision}  "
        f"({sum(not c['pass'] for c in checks)}/{len(checks)} failed)",
        flush=True,
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("contracts",), default="contracts")
    ap.parse_args()
    result = run_contracts()
    return 0 if result["DECISION"] == "CONTRACTS_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
