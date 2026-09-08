"""CCP-S2 compute-budget amendment, stage 1: derive the 64-state manifest.

Implements CCP_S2_COMPUTE_BUDGET_AMENDMENT.json#REDUCTION against the already-frozen 128-state
CCP_S2_STATE_MANIFEST.json (c366f317). This is NOT a new selection: it performs no rollout, no
new hash domain, and touches no outcome/advantage field anywhere. It is a deterministic
RANK-ORDER TRUNCATION -- for each of the 18 (pole, free_set, phase) strata, the states already
selected are already sorted by the frozen selection rule
(sha256('CCP_S2_SELECT|<seed>|<prefix_len>') ascending); this script keeps the first
N_amended <= N_original of that same ordering and moves the rest to a DESCOPED list, which is
recorded here (not deleted, not silently dropped).

Run:  python experiments/ccp_s2_amend_manifest.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
AMENDMENT = SD / "CCP_S2_COMPUTE_BUDGET_AMENDMENT.json"
MANIFEST = SD / "CCP_S2_STATE_MANIFEST.json"
OUT = SD / "CCP_S2_STATE_MANIFEST_AMENDMENT.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; one-shot")
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if amendment["status"] != "FROZEN_APPEND_ONLY_AMENDMENT":
        raise SystemExit(f"REFUSING: compute-budget amendment not frozen: {amendment['status']!r}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["status"] != "FROZEN_SELECTION":
        raise SystemExit(f"REFUSING: original state manifest not frozen: {manifest['status']!r}")

    quota = amendment["REDUCTION"]["new_stratification_quota_per_pole"]
    print(f"CCP-S2 MANIFEST AMENDMENT (rank-order truncation)  {_now()}")
    print(f"  source: CCP_S2_STATE_MANIFEST.json (128 states, FROZEN_SELECTION)")
    print(f"  new quota/pole: {quota}\n", flush=True)

    states = manifest["states"]
    # only immutable, outcome-blind fields are ever read: seed, pole, prefix_len, free_set,
    # phase, rank, actions -- never touches anything about a branching outcome (there is none
    # in this manifest to touch; that is exactly why it is safe to derive from)
    kept, descoped = [], []
    cell_counts = {}
    for pole in ("A", "B"):
        for fs in ("agent0_only", "agent1_only", "both_free"):
            for ph in ("early", "mid", "late"):
                need = quota[fs][ph]
                pool = sorted((s for s in states
                              if s["pole"] == pole and s["free_set"] == fs and s["phase"] == ph),
                             key=lambda s: s["rank"])
                if len(pool) < need:
                    raise SystemExit(f"REFUSING: amended quota {need} exceeds original pool "
                                     f"{len(pool)} at {pole}|{fs}|{ph} -- the amendment cannot "
                                     "ask for more than the original selection already froze")
                kept.extend(pool[:need])
                descoped.extend(s["state_id"] for s in pool[need:])
                cell_counts[f"{pole}|{fs}|{ph}"] = len(pool[:need])

    per_pole_counts = {p: sum(1 for s in kept if s["pole"] == p) for p in ("A", "B")}
    print(f"  kept {len(kept)} states {per_pole_counts}, descoped {len(descoped)}", flush=True)
    if len(kept) != 64 or any(v != 32 for v in per_pole_counts.values()):
        raise SystemExit(f"REFUSING: expected 64 states (32/pole), got {len(kept)} {per_pole_counts}")

    OUT.write_text(json.dumps({
        "record": "CCP-S2 state manifest amendment", "status": "FROZEN_SELECTION", "utc": _now(),
        "implements": "CCP_S2_COMPUTE_BUDGET_AMENDMENT.json#REDUCTION",
        "derived_from": "CCP_S2_STATE_MANIFEST.json (c366f317), rank-order truncation only, "
                        "zero new rollouts, zero outcome fields consulted",
        "selection_rule": "identical to the original: sha256('CCP_S2_SELECT|<seed>|<prefix_len>') "
                          "ascending within each stratum, truncated at the amended quota",
        "quota_per_pole": quota,
        "n_kept_total": len(kept), "per_pole_counts": per_pole_counts,
        "cell_counts": cell_counts,
        "n_descoped": len(descoped), "descoped_state_ids": descoped,
        "descoped_note": "these states remain valid, frozen, outcome-blind selections under the "
                         "ORIGINAL plan; they are simply outside the amended compute budget. Not "
                         "deleted from the original manifest, not reused for any other purpose.",
        "states": kept,
    }, indent=2), encoding="utf-8")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
