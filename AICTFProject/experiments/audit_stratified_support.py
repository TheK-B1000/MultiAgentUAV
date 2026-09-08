"""One-shot support-validity audit for the 16-cell stratified collection.

Scores the frozen floor: every one of the 16 pole x regime x horizon cells must
contain branch states drawn from at least 32 DISTINCT seeds.

Deliberately a SEPARATE program from the collector. The collector must not be able
to observe whether the floor is met, and this audit must not be able to influence
what is collected. It refuses to run until COLLECTION_COMPLETE exists, so the floor
is never checked mid-run -- checking during collection is the first step toward
"collect until the cells pass", which the protocol prohibits by name.

INVALID here means the same thing it meant for RASR: the required evidence
distribution did not materialize, so the protocol refuses to judge. It is NOT a
scientific failure of anything, and it is NOT a licence to collect more seeds.

Run:  python experiments/audit_stratified_support.py
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand"
PROTOCOL = SD / "rasrppo" / "STRATIFIED_REGIME_COLLECTION_PROTOCOL_DESIGN.json"
DATA = SD / "stratified_regime_data"
COMPLETE = DATA / "COLLECTION_COMPLETE.json"
OUT = DATA / "SUPPORT_VALIDITY.json"

SUPPORT_FLOOR = 32
CELLS = [f"{p}_r{r}_{b}" for p in "AB" for r in range(4) for b in ("not_late", "late")]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    if not COMPLETE.is_file():
        raise SystemExit(
            "REFUSING: COLLECTION_COMPLETE.json does not exist. The support floor is "
            "scored once, after the full frozen budget is collected -- never mid-run.")
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; this audit is one-shot")

    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if spec["SUPPORT_FLOOR"]["rule"].find("32") < 0:
        raise SystemExit("REFUSING: frozen floor is not 32; spec drift")

    states = defaultdict(int)
    seeds = defaultdict(set)
    n_states = 0
    seen = set()
    for path in sorted(glob.glob(str(DATA / "seed_shards" / "*.npz"))):
        seed = int(Path(path).stem.split("seed_")[-1])
        seen.add(seed)
        z = np.load(path, allow_pickle=True)
        for cell in z["branch_cell"]:
            c = str(cell)
            states[c] += 1
            seeds[c].add(seed)
            n_states += 1

    complete = json.loads(COMPLETE.read_text(encoding="utf-8"))
    print(f"STRATIFIED SUPPORT VALIDITY AUDIT  {_now()}")
    print(f"  seeds {len(seen)}   branch states {n_states}\n")
    print(f"  {'cell':18s} {'states':>7s} {'seeds':>6s}  {'>=32':>6s}")

    cells_out, invalid = {}, []
    for c in CELLS:
        n_seed = len(seeds[c])
        ok = n_seed >= SUPPORT_FLOOR
        if not ok:
            invalid.append(c)
        cells_out[c] = {"n_states": states[c], "n_distinct_seeds": n_seed, "valid": ok}
        print(f"  {c:18s} {states[c]:7d} {n_seed:6d}  {'PASS' if ok else 'FAIL':>6s}")

    verdict = "VALID" if not invalid else "INVALID"
    rec = {
        "record": "16-cell stratified collection support validity",
        "status": "FROZEN_RESULT",
        "one_shot": True,
        "utc": _now(),
        "protocol": "STRATIFIED_REGIME_COLLECTION_PROTOCOL_DESIGN.json, frozen c318bcea",
        "collection": {"seeds": len(seen), "branch_states": n_states,
                       "collection_complete_utc": complete.get("utc")},
        "minimum_distinct_seeds_per_cell": SUPPORT_FLOOR,
        "scope": "FULL block. Never per split -- a 32-seed split cannot satisfy a 32-seed floor.",
        "cells": cells_out,
        "invalid_cells": invalid,
        "VERDICT": verdict,
        "consequence": (
            "Calibration of tau, rho and o_max on the CALIB split may proceed."
            if verdict == "VALID" else
            "STOP without data growth. Collecting more seeds until the cells pass is "
            "prohibited by the frozen protocol. INVALID means the evidence "
            "distribution did not materialize, not that anything failed scientifically."),
    }
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\n  VERDICT: {verdict}")
    if invalid:
        print(f"  cells below floor: {invalid}")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
