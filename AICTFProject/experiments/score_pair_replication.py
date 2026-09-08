"""Score the frozen discovered-pair replication (Confirmation B).

Implements artifacts/summer_2026/DISCOVERED_PAIR_REPLICATION_FROZEN.json.

Primary gate is the OP7/OP8 axis, both directions, Wald LCB95 > 0 at n=60.
The OP7/OP11 crossover is REPORTED but not required and cannot rescue a failed
primary gate.

This protocol evaluates two policies only, so it cannot compute a repertoire
value: V_fixed is defined over the full five-policy set and the other three are
not evaluated on block 9200000. That is structural, not an omission, and the
scorer states it rather than inventing a two-policy substitute.

Run:  python experiments/score_pair_replication.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts/summer_2026/DISCOVERED_PAIR_REPLICATION_FROZEN.json"
SUMMARY = ROOT / "artifacts/vgc_diversity/crossplay/pair_replication_summary.json"
OUT = ROOT / "artifacts/summer_2026/CONFIRMATION_B_RESULT.json"

A_ID = "vgc_d1_seed3600001"      # D1
B_ID = "g0_v5_long_seed3200001"  # D7


def wald_lcb(p_a: float, p_b: float, n: int) -> tuple[float, float]:
    d = p_a - p_b
    se = math.sqrt(p_a * (1 - p_a) / n + p_b * (1 - p_b) / n)
    return d, d - 1.96 * se


def main() -> int:
    if not SUMMARY.is_file():
        print(f"BLOCKED: missing {SUMMARY}", file=sys.stderr)
        return 2
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    cells = {s["policy_id"]: {op: float(v["win_rate"])
                              for op, v in s["per_opponent"].items()}
             for s in summary["summaries"]}

    inv = {
        "eval_seed_base_is_9200000": summary.get("eval_seed_base") == 9200000,
        "not_default_block": summary.get("eval_seed_block_is_default") is False,
        "episodes_per_cell_is_60": summary.get("episodes_per_cell") == 60,
        "two_policies": len(summary.get("summaries", [])) == 2,
        "both_policies_present": A_ID in cells and B_ID in cells,
        "all_14_cells": all(len(v) == 7 for v in cells.values()),
    }
    if not all(inv.values()):
        OUT.write_text(json.dumps({"gate": "CONFIRMATION_B", "verdict": "BLOCKED",
                                   "reason": "provenance invariant failed",
                                   "invariants": inv}, indent=2), encoding="utf-8")
        print("BLOCKED: provenance invariant failed", inv, file=sys.stderr)
        return 3

    n = int(summary["episodes_per_cell"])
    d7, l7 = wald_lcb(cells[B_ID]["OP7"], cells[A_ID]["OP7"], n)   # D7 > D1 on OP7
    d8, l8 = wald_lcb(cells[A_ID]["OP8"], cells[B_ID]["OP8"], n)   # D1 > D7 on OP8
    d11, l11 = wald_lcb(cells[A_ID]["OP11"], cells[B_ID]["OP11"], n)

    primary = "PASS" if (l7 > 0 and l8 > 0) else "FAIL"
    out = {
        "gate": "CONFIRMATION_B_DISCOVERED_PAIR_REPLICATION",
        "protocol": str(FROZEN.relative_to(ROOT)).replace("\\", "/"),
        "eval_seed_base": summary["eval_seed_base"],
        "episodes_per_cell": n,
        "git_commit_of_evaluation": summary.get("git_commit"),
        "provenance_invariants": inv,
        "OP7_D7_over_D1": {"wr_d7": cells[B_ID]["OP7"], "wr_d1": cells[A_ID]["OP7"],
                           "diff": round(d7, 4), "LCB95": round(l7, 4),
                           "verdict": "PASS" if l7 > 0 else "FAIL"},
        "OP8_D1_over_D7": {"wr_d1": cells[A_ID]["OP8"], "wr_d7": cells[B_ID]["OP8"],
                           "diff": round(d8, 4), "LCB95": round(l8, 4),
                           "verdict": "PASS" if l8 > 0 else "FAIL"},
        "OP11_secondary_not_required": {
            "diff": round(d11, 4), "LCB95": round(l11, 4),
            "verdict": "PASS" if l11 > 0 else "FAIL",
            "note": "reported only; cannot rescue a failed primary gate"},
        "PRIMARY_GATE_OP7_OP8_CROSSOVER_REPLICATES": primary,
        "per_cell_win_rates": cells,
        "cannot_compute_repertoire_value": (
            "this protocol evaluates two policies, so V_fixed over the full "
            "five-policy set is undefined on block 9200000. Repertoire value is "
            "Confirmation A's gate 3, not this one."),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in
                      ("PRIMARY_GATE_OP7_OP8_CROSSOVER_REPLICATES",
                       "OP7_D7_over_D1", "OP8_D1_over_D7")}, indent=2))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
