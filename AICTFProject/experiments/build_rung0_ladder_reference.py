"""Fix the sharing ladder's Rung-0 reference: per-seed outcomes on the 128-seed matched set.

Implements LADDER_MATCHED_EVALUATION_AMENDMENT.json#MATCHED_EVALUATION_RULE. Reads the two
Rung-0 row files (block 1 sealed 11960001..064, block 2 stability 11970001..064), verifies
them (sha256, 64 disjoint seeds each, all four cells complete), and freezes:

    per seed s:  delta_A(s) = win(z0,A,s) - win(z1,A,s),  delta_B(s) = win(z1,B,s) - win(z0,B,s)
    pooled n=128 delta_A / delta_B with the paired percentile bootstrap

Every later rung's within-seed difference d(s) = delta_Rk(s) - delta_R0(s) is computed against
these frozen per-seed values. Rung 0 is never re-run.

Run:  python experiments/build_rung0_ladder_reference.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.eval_hog_psp_v3 import _mean_ci

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
AMEND = SD / "LADDER_MATCHED_EVALUATION_AMENDMENT.json"
BLOCKS = {
    "block1_sealed": (SD / "rung0_crossover_eval_rows.csv", range(11_960_001, 11_960_065)),
    "block2_stability": (SD / "rung0_stability_rerun_rows.csv", range(11_970_001, 11_970_065)),
}
OUT = SD / "RUNG0_LADDER_REFERENCE.json"
CELLS = (("z0", "A"), ("z1", "A"), ("z0", "B"), ("z1", "B"))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT.name} exists; the reference is frozen once")
    amend = json.loads(AMEND.read_text(encoding="utf-8"))
    if amend["status"] != "FROZEN_APPEND_ONLY_AMENDMENT":
        raise SystemExit("REFUSING: matched-evaluation amendment not frozen")

    wins: dict[tuple[str, str], dict[int, int]] = {c: {} for c in CELLS}
    sources = {}
    for name, (path, seeds) in BLOCKS.items():
        if not path.is_file():
            raise SystemExit(f"REFUSING: {path.name} missing")
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        seen = {c: set() for c in CELLS}
        for r in rows:
            c = (r["z"], r["pole"])
            s = int(r["seed"])
            if s not in seeds:
                raise SystemExit(f"REFUSING: {path.name} has seed {s} outside its frozen block")
            if s in wins[c]:
                raise SystemExit(f"REFUSING: seed {s} appears twice for cell {c}")
            wins[c][s] = int(r["win"])
            seen[c].add(s)
        for c in CELLS:
            if seen[c] != set(seeds):
                raise SystemExit(f"REFUSING: {path.name} cell {c} does not cover all 64 seeds exactly")
        sources[name] = {"path": str(path.relative_to(ROOT)), "sha256": _sha(path),
                         "seeds": [seeds[0], seeds[-1]], "rows": len(rows)}

    all_seeds = sorted(set().union(*(set(r) for _, r in BLOCKS.values())))
    if len(all_seeds) != 128:
        raise SystemExit(f"REFUSING: expected 128 disjoint seeds, got {len(all_seeds)}")

    dA = {s: wins[("z0", "A")][s] - wins[("z1", "A")][s] for s in all_seeds}
    dB = {s: wins[("z1", "B")][s] - wins[("z0", "B")][s] for s in all_seeds}
    pooled_A = _mean_ci(np.array([dA[s] for s in all_seeds], dtype=np.float64))
    pooled_B = _mean_ci(np.array([dB[s] for s in all_seeds], dtype=np.float64))
    for d in (pooled_A, pooled_B):
        d["passes"] = bool(d["mean"] > 0 and d["lcb95"] > 0)

    cell_rates = {f"{z}_pole{p}": float(np.mean([wins[(z, p)][s] for s in all_seeds])) for z, p in CELLS}

    OUT.write_text(json.dumps({
        "record_id": "RUNG0_LADDER_REFERENCE", "status": "FROZEN_REFERENCE", "utc": _now(),
        "implements": "LADDER_MATCHED_EVALUATION_AMENDMENT.json#MATCHED_EVALUATION_RULE",
        "what_this_is": "Rung 0 (pi_A/pi_B dispatched by z, bit-exact) per-seed outcomes on the 128-seed matched set. Every later rung's within-seed difference is computed against these values. Rung 0 is never re-run.",
        "sources": sources,
        "matched_seed_set": {"n_per_cell": 128, "seeds": all_seeds},
        "cell_win_rates_n128": cell_rates,
        "POOLED_N128": {"delta_A": pooled_A, "delta_B": pooled_B,
                        "note": "Rung 0's own gate at n=128, reported for context; the ladder is read through within-seed D, which does not require this to pass"},
        "per_seed": {"delta_A": {str(s): dA[s] for s in all_seeds},
                     "delta_B": {str(s): dB[s] for s in all_seeds}},
        "bootstrap": {"procedure": "paired percentile bootstrap over seeds",
                      "samples": 20000, "alpha": 0.05, "rng_seed": 7},
    }, indent=2), encoding="utf-8")

    print(f"RUNG 0 LADDER REFERENCE  {_now()}")
    for k, v in sources.items():
        print(f"  {k}: {v['rows']} rows, seeds {v['seeds']}, sha {v['sha256'][:12]}...")
    print(f"  matched set: {len(all_seeds)} seeds/cell")
    print(f"  cell win rates: {cell_rates}")
    print(f"  pooled delta_A {pooled_A['mean']:+.4f} [{pooled_A['lcb95']:+.4f}, {pooled_A['ucb95']:+.4f}] "
          f"{'PASS' if pooled_A['passes'] else 'FAIL'}")
    print(f"  pooled delta_B {pooled_B['mean']:+.4f} [{pooled_B['lcb95']:+.4f}, {pooled_B['ucb95']:+.4f}] "
          f"{'PASS' if pooled_B['passes'] else 'FAIL'}")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
