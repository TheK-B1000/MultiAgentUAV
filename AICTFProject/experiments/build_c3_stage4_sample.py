"""Build the deterministic Stage-4 leg-2 sampled-anchor manifest.

Frozen cells: artifacts/c3_discovery/C3_STAGE4_CONFIRMATION_FROZEN.json (bebb626)

Draws 210 anchors from the COMPLETE fresh census on 9810000+, stratified 10 per
policy x opponent cell, using the seed the frozen cells derive from the contract
hash. Separate from the Stage-3 sampler so each stage's draw is independently
auditable rather than sharing a mode flag.

FAIL-CLOSED
-----------
Refuses if the frozen cells are missing or unfrozen, if the derived seed does
not match the frozen value, if the fresh census is absent or covers the wrong
policies/seed block, or if ANY Stage-4 counterfactual outcome already exists --
the draw must precede outcomes or it could be re-rolled until it flattered one.

Run:  python experiments/build_c3_stage4_sample.py
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

DISCOVERY_DIR = PROJECT_ROOT / "artifacts" / "c3_discovery"
STAGE4_DIR = PROJECT_ROOT / "artifacts" / "c3_stage4"
CONTRACT = DISCOVERY_DIR / "C3_DISCOVERY_PREREG_FROZEN.json"
FROZEN = DISCOVERY_DIR / "C3_STAGE4_CONFIRMATION_FROZEN.json"
MANIFEST_OUT = STAGE4_DIR / "C3_STAGE4_SAMPLE_MANIFEST.json"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if not FROZEN.exists():
        raise SystemExit(f"REFUSED: Stage-4 freeze missing at {FROZEN}")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise SystemExit("REFUSED: Stage-4 cells are not FROZEN")

    cf = frozen["leg_2_fresh_counterfactual"]["sampling"]
    per_target = 10
    n_target = 210

    contract_bytes = CONTRACT.read_bytes()
    seed = int(hashlib.sha256(contract_bytes + b"stage4").hexdigest()[:8], 16)
    declared_derivation = str(cf["seed_derivation"])
    if "stage4" not in declared_derivation:
        raise SystemExit(f"REFUSED: unexpected seed derivation {declared_derivation!r}")

    results_path = STAGE4_DIR / STAGE3_RESULTS_NAME
    if results_path.exists() and results_path.stat().st_size > 0:
        raise SystemExit(
            f"REFUSED: {results_path.name} already has content. The draw must "
            "precede any counterfactual outcome."
        )
    if MANIFEST_OUT.exists():
        raise SystemExit(f"REFUSED: a Stage-4 draw already exists at {MANIFEST_OUT}")

    try:
        anchors, manifest = load_stage1_bundle(STAGE4_DIR)
    except FileNotFoundError as exc:
        raise SystemExit(f"REFUSED: {exc}") from None

    want_pol = sorted(int(p) for p in frozen["policies"])
    got_pol = sorted(int(s) for s in manifest.get("seeds", []))
    if got_pol != want_pol:
        raise SystemExit(f"REFUSED: census policies {got_pol} != frozen {want_pol}")
    if int(manifest.get("discovery_seed_base", -1)) != int(frozen["seeds"]["base"]):
        raise SystemExit("REFUSED: census seed base does not match the frozen Stage-4 base")

    by_stratum: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in anchors:
        by_stratum[(int(row["train_seed"]), str(row["opponent"]))].append(row)

    expected = [(p, o) for p in want_pol for o in frozen["opponents"]]
    missing = [c for c in expected if c not in by_stratum]
    if missing:
        raise SystemExit(
            f"REFUSED: {len(missing)} stratum/strata absent from the fresh census: "
            f"{[f'{p}|{o}' for p, o in missing][:5]}"
        )

    N = len(anchors)
    order = sorted(by_stratum)
    N_h = {c: len(by_stratum[c]) for c in order}

    alloc = {c: min(per_target, N_h[c]) for c in order}
    deficit = n_target - sum(alloc.values())
    while deficit > 0:
        progressed = False
        for c in order:
            if deficit <= 0:
                break
            if alloc[c] < N_h[c]:
                alloc[c] += 1
                deficit -= 1
                progressed = True
        if not progressed:
            break

    rng = np.random.default_rng(seed)
    selected: list[str] = []
    by_cell: dict[str, list[str]] = {}
    for c in order:
        keys = sorted(anchor_key_from_row(r) for r in by_stratum[c])
        k = alloc[c]
        chosen = list(keys) if k >= len(keys) else [
            keys[i] for i in sorted(int(x) for x in rng.choice(len(keys), size=k, replace=False))
        ]
        by_cell[f"{c[0]}|{c[1]}"] = chosen
        selected.extend(chosen)

    # Per-policy weights: the frozen replication rule is >=2/3 POLICIES, so
    # leg 2 renormalises within each policy rather than pooling as Stage 3 did.
    n_by_policy: dict[int, int] = defaultdict(int)
    for c in order:
        n_by_policy[c[0]] += N_h[c]

    doc = {
        "record": "C3 Stage-4 leg-2 sampled-anchor manifest",
        "status": "FROZEN",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_cells": "artifacts/c3_discovery/C3_STAGE4_CONFIRMATION_FROZEN.json",
        "frozen_cells_sha256": _sha256(FROZEN),
        "c3_contract_sha256": _sha256(CONTRACT),
        "sampling_seed": seed,
        "sampling_seed_derivation": "int(sha256(C3_DISCOVERY_PREREG_FROZEN.json + b'stage4')[:8], 16)",
        "drawn_before_any_stage4_outcome": True,
        "seed_block": frozen["seeds"]["range"],
        "N_total": N,
        "N_by_policy": {str(k): v for k, v in sorted(n_by_policy.items())},
        "N_h": {f"{p}|{o}": N_h[(p, o)] for p, o in order},
        "W_h": {f"{p}|{o}": round(N_h[(p, o)] / N, 8) for p, o in order},
        "W_h_within_policy": {
            f"{p}|{o}": round(N_h[(p, o)] / n_by_policy[p], 8) for p, o in order
        },
        "n_h_sampled": {f"{p}|{o}": alloc[(p, o)] for p, o in order},
        "n_sampled_total": len(selected),
        "selected_anchor_ids": selected,
        "selected_by_stratum": by_cell,
        "estimator": "per policy: p_hat = sum over that policy's strata of (N_h / N_policy) * p_h",
        "decision_rule": "policy LEG2_PASS iff LCB95(weighted fork rate) > 0.20",
    }
    STAGE4_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print("=" * 74)
    print("C3 STAGE-4 LEG-2 SAMPLE MANIFEST")
    print("=" * 74)
    print(f"  fresh census   : {N:,} anchors across {len(order)} strata")
    print(f"  sampling seed  : {seed}  (derived from contract + 'stage4')")
    print(f"  sampled        : {len(selected)} / {n_target}")
    print(f"  per-policy N   : {dict(sorted(n_by_policy.items()))}")
    print(f"  wrote          : {MANIFEST_OUT.relative_to(PROJECT_ROOT)}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
