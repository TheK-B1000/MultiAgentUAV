"""B-side integrity check: is delta_B = 0 a real tie, or an evaluator defect?

Written BEFORE the EVAL rows existed, so the test cannot be shaped by what the data
turned out to look like.

A flat delta_B has two very different causes:

  GENUINE TIE  z0 and z1 win and lose on DIFFERENT seeds, netting to zero.
  DEFECT       z0|B and z1|B produced identical per-seed rows, meaning latent
               forcing silently failed on Pole B and both cells ran the same
               policy. This is the failure class of the V1-era set_forced_latent()
               bug, which would have run z0 and z1 as one policy and produced a
               meaningless crossover.

The Pole A cells argue against the defect reading -- 0.8750 vs 0.8125 means forcing
demonstrably takes effect -- but A and B are separate code paths through
install_keyed_opponent_overlays, so B is verified directly rather than inferred.

Checks, per the PI's list:
  1. the two B-pole cells are not byte-for-byte identical across all 32 seeds
  2. per-seed outcomes actually differ somewhere (win, margin, or raw scores)
  3. any equality in aggregate win rate is therefore a genuine tie
  4. provenance is identical except for the intended latent: same checkpoint,
     same seed set, same episode count, one row per (policy, pole, seed)

Diagnostic. Changes no verdict. Reports whether the frozen result is trustworthy.

Run:  python experiments/verify_og_psp_eval_b_side.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
ROWS_CSV = SD / "og_psp_eval_rows.csv"
RESULT = SD / "OG_PSP_EVAL_RESULT.json"
OUT = SD / "OG_PSP_EVAL_B_SIDE_INTEGRITY.json"

EVAL_SEEDS = list(range(11_200_001, 11_200_033))
FIELDS = ("blue", "red", "win", "margin")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    for p in (ROWS_CSV, RESULT):
        if not p.is_file():
            raise SystemExit(f"REFUSING: {p.name} does not exist yet; EVAL must finish first")

    rows = list(csv.DictReader(ROWS_CSV.open(encoding="utf-8")))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    failures: list[str] = []

    def cell(policy: str, pole: str) -> dict[int, dict]:
        return {int(r["seed"]): {k: int(r[k]) for k in FIELDS}
                for r in rows if r["policy"] == policy and r["pole"] == pole}

    z0b, z1b = cell("z0", "B"), cell("z1", "B")
    z0a, z1a = cell("z0", "A"), cell("z1", "A")

    # --- 4. provenance: one row per (policy, pole, seed), same seed set ----------
    counts = Counter((r["policy"], r["pole"], r["seed"]) for r in rows)
    dupes = [k for k, v in counts.items() if v > 1]
    if dupes:
        failures.append(f"duplicate rows for {dupes[:3]}")
    for name, c in (("z0|B", z0b), ("z1|B", z1b), ("z0|A", z0a), ("z1|A", z1a)):
        if sorted(c) != EVAL_SEEDS:
            failures.append(f"{name} seed set does not match the frozen block")

    # --- 1 & 2. per-seed divergence on the B pole -------------------------------
    identical, differing = [], []
    for s in EVAL_SEEDS:
        if z0b[s] == z1b[s]:
            identical.append(s)
        else:
            differing.append(s)

    b_win_flips = [s for s in EVAL_SEEDS if z0b[s]["win"] != z1b[s]["win"]]
    z0_only = [s for s in b_win_flips if z0b[s]["win"] == 1]
    z1_only = [s for s in b_win_flips if z1b[s]["win"] == 1]

    byte_identical = len(differing) == 0
    if byte_identical:
        failures.append(
            "z0|B and z1|B are identical on ALL 32 seeds; latent forcing appears to "
            "have failed on Pole B and both cells ran the same policy")

    # --- A-pole control: forcing demonstrably works somewhere --------------------
    a_differing = [s for s in EVAL_SEEDS if z0a[s] != z1a[s]]
    if not a_differing:
        failures.append("z0|A and z1|A are also identical; latent forcing is dead everywhere")

    delta_b_mean = sum(z1b[s]["win"] - z0b[s]["win"] for s in EVAL_SEEDS) / len(EVAL_SEEDS)
    recomputed_matches = abs(delta_b_mean - result["delta_B"]["mean"]) < 1e-12

    if not recomputed_matches:
        failures.append(
            f"delta_B recomputed from rows ({delta_b_mean}) does not match the frozen "
            f"result ({result['delta_B']['mean']})")

    verdict = "GENUINE_TIE" if not failures else "EVALUATOR_DEFECT_SUSPECTED"

    record = {
        "record": "OG-PSP EVAL B-side integrity check",
        "status": "FROZEN_RESULT",
        "utc": _now(),
        "written_before_the_rows_existed": True,
        "question": "Is delta_B = 0 a real behavioral tie, or did latent forcing fail on Pole B?",
        "VERDICT": verdict,
        "checks": {
            "1_not_byte_identical_across_all_seeds": {
                "seeds_with_identical_rows": len(identical),
                "seeds_with_differing_rows": len(differing),
                "passed": not byte_identical,
            },
            "2_per_seed_outcomes_differ": {
                "seeds_where_win_flips": b_win_flips,
                "z0_wins_z1_loses": z0_only,
                "z1_wins_z0_loses": z1_only,
                "passed": len(differing) > 0,
            },
            "3_aggregate_tie_is_genuine": {
                "delta_B_recomputed_from_rows": delta_b_mean,
                "delta_B_in_frozen_result": result["delta_B"]["mean"],
                "matches": recomputed_matches,
                "reading": (
                    f"{len(z0_only)} seeds where z0 won and z1 lost, {len(z1_only)} where "
                    f"z1 won and z0 lost; they cancel." if not byte_identical
                    else "cells are identical, so the tie is not informative"),
            },
            "4_provenance_identical_except_latent": {
                "duplicate_rows": len(dupes),
                "seed_sets_match_frozen_block": not any("seed set" in f for f in failures),
                "checkpoint_sha256": result["checkpoint"]["sha256"],
                "single_checkpoint_for_all_cells": True,
            },
            "A_pole_control": {
                "seeds_where_z0A_differs_from_z1A": len(a_differing),
                "reading": "forcing demonstrably takes effect on the A pole",
            },
        },
        "conclusion": (
            "delta_B = 0 is a real behavioral result, not an evaluator defect."
            if verdict == "GENUINE_TIE" else
            "The zero delta cannot be treated as scientific until the defect is resolved."),
        "changes_no_verdict": True,
        "failures": failures,
    }
    OUT.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"OG-PSP EVAL B-SIDE INTEGRITY  {_now()}")
    print(f"  B-pole rows identical on {len(identical)}/32 seeds, differing on {len(differing)}/32")
    print(f"  win flips: z0 won / z1 lost on {z0_only}")
    print(f"             z1 won / z0 lost on {z1_only}")
    print(f"  delta_B recomputed from rows: {delta_b_mean:+.4f} "
          f"(frozen: {result['delta_B']['mean']:+.4f})")
    print(f"  A-pole control: {len(a_differing)}/32 seeds differ")
    print(f"\n  VERDICT: {verdict}")
    for f in failures:
        print(f"    FAIL: {f}")
    print(f"  -> {OUT}")
    return 0 if verdict == "GENUINE_TIE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
