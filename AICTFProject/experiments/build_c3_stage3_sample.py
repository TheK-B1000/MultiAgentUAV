"""Build the deterministic 210-anchor Stage-3 sample manifest.

Amendment: artifacts/c3_discovery/C3_STAGE3_SAMPLING_AMENDMENT.json (74b857c)
Contract:  artifacts/c3_discovery/C3_DISCOVERY_PREREG_FROZEN.json

Consumes the COMPLETE Stage-1 anchor census and emits the sampled-anchor
manifest that the sampled Stage-3 runner is required to match. Nothing here
touches a scientific cell: T_trace, H_response, delta, U, minimum_fork_rate,
the pressure predicate, the earliest-fork rule and the exhaustive legal-response
enumeration are all unchanged. Only which anchors get evaluated changes.

FAIL-CLOSED, in both directions
-------------------------------
Refuses to run if the amendment is missing or its hash does not match, and
refuses to run if ANY Stage-3 outcome already exists. The second check is the
important one: the sample must be drawn before outcomes are visible, or the
draw could be re-rolled until it flattered a result.

DETERMINISM
-----------
The seed is not chosen. It is derived from the frozen contract's own bytes:

    sampling_seed = int(sha256(C3_DISCOVERY_PREREG_FROZEN.json)[:8], 16)

so the draw is a function of a document that predates it. Within each stratum
anchors are sorted by anchor_key before selection, so the result does not depend
on Stage-1 row ordering.

Run:  python experiments/build_c3_stage3_sample.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from rl.analysis.c3_discovery_artifacts import (  # noqa: E402
    STAGE3_RESULTS_NAME,
    anchor_key_from_row,
    load_stage1_bundle,
)

OUT_DIR = PROJECT_ROOT / "artifacts" / "c3_discovery"
CONTRACT = OUT_DIR / "C3_DISCOVERY_PREREG_FROZEN.json"
AMENDMENT = OUT_DIR / "C3_STAGE3_SAMPLING_AMENDMENT.json"
SAMPLE_MANIFEST = OUT_DIR / "C3_STAGE3_SAMPLE_MANIFEST.json"

EXPECTED_POLICIES = (3_200_001, 3_200_002, 3_200_003)
EXPECTED_OPPONENTS = ("OP6", "OP7", "OP8", "OP9", "OP10", "OP11", "OP12")
EXPECTED_EPISODES_PER_CELL = 30


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stratum(row: dict) -> tuple[int, str]:
    return (int(row["train_seed"]), str(row["opponent"]))


def main() -> int:
    if not AMENDMENT.exists():
        raise SystemExit(f"REFUSED: sampling amendment missing at {AMENDMENT}")
    amend = json.loads(AMENDMENT.read_text(encoding="utf-8"))

    contract_hash = _sha256(CONTRACT)
    declared = str(amend["sampling_seed"]["c3_contract_sha256"])
    if declared != contract_hash:
        raise SystemExit(
            "REFUSED: contract hash drift.\n"
            f"  amendment declares {declared}\n  on disk            {contract_hash}"
        )

    seed = int(amend["sampling_seed"]["value"])
    derived = int(contract_hash[:8], 16)
    if seed != derived:
        raise SystemExit(
            f"REFUSED: sampling seed {seed} is not the value derived from the "
            f"contract hash ({derived}). The seed must not be chosen."
        )

    stage3_path = OUT_DIR / STAGE3_RESULTS_NAME
    if stage3_path.exists() and stage3_path.stat().st_size > 0:
        raise SystemExit(
            f"REFUSED: {STAGE3_RESULTS_NAME} already has content. The sample must "
            "be drawn BEFORE any Stage-3 outcome exists, or the draw could be "
            "re-rolled until it flattered a result."
        )

    try:
        anchors, manifest = load_stage1_bundle(OUT_DIR)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"REFUSED: {exc}\n"
            "The sampling frame is the COMPLETE 630-episode Stage-1 census. Wait "
            "for Stage 1 to persist before drawing the sample."
        ) from None
    if not anchors:
        raise SystemExit("REFUSED: Stage-1 anchor census is empty")

    # ---- census integrity ------------------------------------------------
    by_stratum: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in anchors:
        by_stratum[_stratum(row)].append(row)

    episodes = {(int(r["train_seed"]), str(r["opponent"]), int(r["eval_seed"]))
                for r in anchors}
    cells_present = sorted(by_stratum)
    expected_cells = [(p, o) for p in EXPECTED_POLICIES for o in EXPECTED_OPPONENTS]
    missing_cells = [c for c in expected_cells if c not in by_stratum]

    integrity = {
        "n_anchors_total": len(anchors),
        "n_strata_present": len(cells_present),
        "n_strata_expected": len(expected_cells),
        "missing_strata": [f"{p}|{o}" for p, o in missing_cells],
        "n_distinct_episodes_with_anchors": len(episodes),
        "expected_episodes": len(EXPECTED_POLICIES) * len(EXPECTED_OPPONENTS)
        * EXPECTED_EPISODES_PER_CELL,
        "stage1_manifest_keys": sorted(manifest)[:20],
    }
    if missing_cells:
        print("WARNING: strata absent from the census:", integrity["missing_strata"])

    # ---- weights from the COMPLETE census --------------------------------
    N = len(anchors)
    N_h = {c: len(by_stratum[c]) for c in cells_present}
    W_h = {c: N_h[c] / N for c in cells_present}

    # ---- allocation with the frozen low-count fallback -------------------
    per_target = int(amend["sampling_design"]["per_stratum_target"])
    n_target = int(amend["sampling_design"]["n_anchors_target"])

    alloc = {c: min(per_target, N_h[c]) for c in cells_present}
    deficit = n_target - sum(alloc.values())
    # Round-robin over strata with unused anchors, ordered by (policy, opponent).
    order = sorted(cells_present)
    guard = 0
    while deficit > 0 and guard < 100_000:
        progressed = False
        for c in order:
            if deficit <= 0:
                break
            if alloc[c] < N_h[c]:
                alloc[c] += 1
                deficit -= 1
                progressed = True
        guard += 1
        if not progressed:
            break  # every anchor in every stratum is already allocated

    # ---- deterministic draw ---------------------------------------------
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    per_stratum_selected: dict[str, list[str]] = {}
    for c in order:
        keys = sorted(anchor_key_from_row(r) for r in by_stratum[c])
        k = alloc[c]
        if k >= len(keys):
            chosen = list(keys)
        else:
            idx = rng.choice(len(keys), size=k, replace=False)
            chosen = [keys[i] for i in sorted(int(x) for x in idx)]
        per_stratum_selected[f"{c[0]}|{c[1]}"] = chosen
        selected.extend(chosen)

    doc = {
        "record": "C3 Stage-3 sampled-anchor manifest",
        "status": "FROZEN",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "amendment": "artifacts/c3_discovery/C3_STAGE3_SAMPLING_AMENDMENT.json",
        "amendment_commit": "74b857c",
        "amendment_sha256": _sha256(AMENDMENT),
        "c3_contract_sha256": contract_hash,
        "sampling_seed": seed,
        "sampling_seed_derivation": "int(sha256(C3_DISCOVERY_PREREG_FROZEN.json)[:8], 16)",
        "drawn_before_any_stage3_outcome": True,
        "census_integrity": integrity,
        "N_total": N,
        "N_h": {f"{p}|{o}": N_h[(p, o)] for p, o in order},
        "W_h": {f"{p}|{o}": round(W_h[(p, o)], 8) for p, o in order},
        "n_h_sampled": {f"{p}|{o}": alloc[(p, o)] for p, o in order},
        "n_sampled_total": len(selected),
        "n_target": n_target,
        "low_count_fallback_applied": bool(
            any(N_h[c] < per_target for c in cells_present)
        ),
        "strata_below_target": [
            f"{p}|{o}" for p, o in order if N_h[(p, o)] < per_target
        ],
        "selected_anchor_ids": selected,
        "selected_by_stratum": per_stratum_selected,
        "estimator": {
            "point": "p_hat = sum_h W_h * p_hat_h",
            "weights_from": "the COMPLETE Stage-1 census, not the sample",
            "estimand": "fork rate of the natural anchor population",
        },
        "decision_rule": "C3_PASS iff LCB95(weighted natural fork_rate) > 0.20",
        "unchanged_scientific_cells": amend["scientific_cells_unchanged"],
    }
    SAMPLE_MANIFEST.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print("=" * 74)
    print("C3 STAGE-3 SAMPLE MANIFEST")
    print("=" * 74)
    print(f"  census anchors      : {N:,} across {len(cells_present)} strata")
    print(f"  episodes w/ anchors : {len(episodes)} of {integrity['expected_episodes']}")
    print(f"  sampling seed       : {seed}  (derived, not chosen)")
    print(f"  sampled             : {len(selected)} / target {n_target}")
    if doc["strata_below_target"]:
        print(f"  strata below target : {doc['strata_below_target']}")
    print(f"  manifest sha256     : {hashlib.sha256(SAMPLE_MANIFEST.read_bytes()).hexdigest()[:32]}...")
    print(f"  wrote               : {SAMPLE_MANIFEST.relative_to(PROJECT_ROOT)}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
