"""Phase 0 -- Gate 0A: cell-level replication sanity check.

Frozen requirement (PHASE0_ACTION_CONDITIONED_SCORER_PROTOCOL.json ::
GATE_0A_cell_level_replication, restated in PHASE0_DATA_BUDGET_FROZEN.json
GATE_0A_allocation): does the fresh training-only sample reproduce the KNOWN
SAPPO crossover, before any critic is fitted?

    requires  WR(pi_A, A) > WR(pi_B, A)
              WR(pi_B, B) > WR(pi_A, B)

This is a sanity check on the DATASET, not a scorer validation, and it is
NOT the inferential gate -- no bootstrap/LCB is specified for 0A in the
frozen protocol (that machinery belongs to Gate 0B). This script reports the
point-estimate win rates over all 1024 plain episodes (all 256 seeds, both
splits -- 0A is not restricted to the train split) and, as a diagnostic only,
paired-by-seed sign counts. The diagnostic does not gate the verdict.

if_fails: STOP. Do not fit a critic on a dataset where the target structure
is absent. This script refuses to write any Q_psi-fitting artifact; it only
ever writes gate0a.json.

Run:  python experiments/phase0_gate0a.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts/strategic_demand"
COLL = SD / "phase0_scorer_data/full_collection_rebuild_per_branch"
OUT = SD / "phase0_scorer_data" / "gate0a.json"

SEED_BASE, N_SEEDS = 6_500_001, 256
CELLS = [("pi_A", "A"), ("pi_B", "A"), ("pi_A", "B"), ("pi_B", "B")]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} already exists; Gate 0A already frozen")

    manifest = json.loads((COLL / "collection_manifest.json").read_text())
    completed = manifest["completed_seeds"]
    if len(completed) != N_SEEDS or sorted(completed) != list(range(SEED_BASE, SEED_BASE + N_SEEDS)):
        raise SystemExit(f"REFUSING: manifest shows {len(completed)}/{N_SEEDS}; collection incomplete")

    files = sorted((COLL / "seed_summaries").glob("*.json"))
    if len(files) != N_SEEDS:
        raise SystemExit(f"REFUSING: {len(files)} seed_summaries on disk, expected {N_SEEDS}")

    wins, n = Counter(), Counter()
    by_seed: dict[int, dict] = {}
    seeds_seen = set()
    for p in files:
        rows = json.loads(p.read_text())
        if len(rows) != 4:
            raise SystemExit(f"REFUSING: {p} has {len(rows)} rows, expected 4")
        seed = rows[0]["seed"]
        if seed in seeds_seen:
            raise SystemExit(f"REFUSING: duplicate seed {seed} in {p}")
        seeds_seen.add(seed)
        by_seed[seed] = {}
        for r in rows:
            cell = (r["policy"], r["pole"])
            wins[cell] += r["win"]
            n[cell] += 1
            by_seed[seed][cell] = r["win"]

    missing = set(range(SEED_BASE, SEED_BASE + N_SEEDS)) - seeds_seen
    if missing:
        raise SystemExit(f"REFUSING: missing seeds {sorted(missing)}")

    wr = {f"{p}|{k}": wins[(p, k)] / n[(p, k)] for p, k in CELLS}
    req_A = wr["pi_A|A"] > wr["pi_B|A"]
    req_B = wr["pi_B|B"] > wr["pi_A|B"]
    verdict = "PASS" if (req_A and req_B) else "FAIL"

    # diagnostic only, paired by seed -- NOT part of the frozen requirement
    diag = {"pole_A": Counter(), "pole_B": Counter()}
    for seed, c in by_seed.items():
        dA = c[("pi_A", "A")] - c[("pi_B", "A")]
        dB = c[("pi_B", "B")] - c[("pi_A", "B")]
        diag["pole_A"][("pi_A_wins" if dA > 0 else "pi_B_wins" if dA < 0 else "tie")] += 1
        diag["pole_B"][("pi_B_wins" if dB > 0 else "pi_A_wins" if dB < 0 else "tie")] += 1

    rec = {
        "record": "PHASE0 Gate 0A -- cell-level replication",
        "utc": _now(),
        "protocol_ref": "PHASE0_ACTION_CONDITIONED_SCORER_PROTOCOL.json::GATE_0A_cell_level_replication",
        "block": f"{SEED_BASE}..{SEED_BASE + N_SEEDS - 1}",
        "n_episodes": sum(n.values()),
        "win_counts": {f"{p}|{k}": wins[(p, k)] for p, k in CELLS},
        "n_per_cell": {f"{p}|{k}": n[(p, k)] for p, k in CELLS},
        "win_rates": wr,
        "requirement": {
            "WR(pi_A,A) > WR(pi_B,A)": {"lhs": wr["pi_A|A"], "rhs": wr["pi_B|A"], "holds": req_A},
            "WR(pi_B,B) > WR(pi_A,B)": {"lhs": wr["pi_B|B"], "rhs": wr["pi_A|B"], "holds": req_B},
        },
        "VERDICT": verdict,
        "if_fail_rule": "STOP. Do not fit a critic on a dataset where the target structure is absent.",
        "diagnostic_only_not_gating": {
            "note": "paired by seed; no bootstrap/LCB specified for 0A in the frozen protocol",
            "pole_A": dict(diag["pole_A"]),
            "pole_B": dict(diag["pole_B"]),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    print(f"PHASE 0 GATE 0A  {_now()}")
    for p, k in CELLS:
        print(f"  WR({p},{k}) = {wins[(p,k)]:3d}/{n[(p,k)]} = {wr[f'{p}|{k}']:.4f}")
    print(f"\n  WR(pi_A,A) > WR(pi_B,A): {wr['pi_A|A']:.4f} > {wr['pi_B|A']:.4f} -> {req_A}")
    print(f"  WR(pi_B,B) > WR(pi_A,B): {wr['pi_B|B']:.4f} > {wr['pi_A|B']:.4f} -> {req_B}")
    print(f"\n  VERDICT: {verdict}")
    print(f"  -> {OUT}")
    if verdict == "FAIL":
        print("  STOP. Do not fit a critic on a dataset where the target structure is absent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
