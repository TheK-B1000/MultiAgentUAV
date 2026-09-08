"""SAC-RFT EVAL integrity audit: are the CONTROL-Pole-A and TREATMENT-Pole-B reversals real?

Triggered by SAC_RFT_EVAL_INTEGRITY_REQUIRED.json (['A_C', 'B_T']). Unlike trunk-freeze's
audit (a single reversed key, A_C only) or RSCFT's (both arms reversed together on the SAME
pole, A), this trigger is a CROSS pattern: CONTROL fails on Pole A while passing Pole B;
TREATMENT fails on Pole B while passing Pole A. A shared evaluator defect (transposed latent,
wrong opponent key, stale checkpoint) would be expected to act consistently within an arm or
within a pole -- it would not plausibly flip exactly the off-diagonal cells while leaving both
on-diagonal cells (A_T, B_C) positive. That specific asymmetry is what this audit checks for,
same as trunk-freeze's audit checked whether a single-arm reversal could be a shared defect.

Written before sac_rft_eval_rows.csv is read in any interpretive way.

Does not write SAC_RFT_EVAL_RESULT.json. Classifies GENUINE_* vs EVALUATOR_DEFECT_SUSPECTED.

Run:  python experiments/verify_sac_rft_eval_integrity.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
ROWS = SD / "sac_rft_eval_rows.csv"
FLAG = SD / "SAC_RFT_EVAL_INTEGRITY_REQUIRED.json"
FROZEN = SD / "SAC_RFT_MODELS_FROZEN.json"
INCUMBENT_EVAL = SD / "CCP_SUCCESSOR_EVAL_RESULT.json"
OUT = SD / "SAC_RFT_EVAL_INTEGRITY.json"

EVAL_SEEDS = list(range(11_804_001, 11_804_065))
FIELDS = ("blue", "red", "win", "margin")
ARMS = ("CONTROL", "TREATMENT")
N_BOOT, ALPHA, BOOT_SEED = 20_000, 0.05, 7


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ci(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    boot = v[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"mean": float(v.mean()), "lcb95": float(lo), "ucb95": float(hi)}


def main() -> int:
    for p in (ROWS, FLAG, FROZEN, INCUMBENT_EVAL):
        if not p.is_file():
            raise SystemExit(f"REFUSING: {p.name} missing")
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; one-shot")

    rows = list(csv.DictReader(ROWS.open(encoding="utf-8")))
    flag = json.loads(FLAG.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    incumbent = json.loads(INCUMBENT_EVAL.read_text(encoding="utf-8"))
    failures: list[str] = []

    def cell(arm: str, z: str, pole: str) -> dict[int, dict]:
        return {int(r["seed"]): {k: int(r[k]) for k in FIELDS}
                for r in rows if r["arm"] == arm and r["z"] == z and r["pole"] == pole}

    cells = {(arm, z, pole): cell(arm, z, pole)
             for arm in ARMS for z in ("z0", "z1") for pole in ("A", "B")}

    counts = Counter((r["arm"], r["z"], r["pole"], r["seed"]) for r in rows)
    dupes = [k for k, v in counts.items() if v > 1]
    if dupes:
        failures.append(f"duplicate rows: {dupes[:3]}")
    expected_cells = {(a, z, p) for a in ARMS for z in ("z0", "z1") for p in ("A", "B")}
    present_cells = {(r["arm"], r["z"], r["pole"]) for r in rows}
    missing = sorted(expected_cells - present_cells)
    if missing:
        failures.append(f"missing evaluation cells: {missing}")
    for key, c in cells.items():
        if sorted(c) != EVAL_SEEDS:
            failures.append(f"{key} seed set does not match the frozen 64-seed block")

    def split(c0: dict, c1: dict):
        diff_any = [s for s in EVAL_SEEDS if c0[s] != c1[s]]
        diff_win = [s for s in EVAL_SEEDS if c0[s]["win"] != c1[s]["win"]]
        diff_margin = [s for s in EVAL_SEEDS if c0[s]["margin"] != c1[s]["margin"]]
        first_only = [s for s in EVAL_SEEDS if c0[s]["win"] == 1 and c1[s]["win"] == 0]
        second_only = [s for s in EVAL_SEEDS if c1[s]["win"] == 1 and c0[s]["win"] == 0]
        return diff_any, diff_win, diff_margin, first_only, second_only

    liveness = {}
    for arm in ARMS:
        for pole in ("A", "B"):
            key = f"{arm}_{pole}"
            anyd, wind, mard, z0only, z1only = split(cells[(arm, "z0", pole)],
                                                      cells[(arm, "z1", pole)])
            liveness[key] = {"differing_any": len(anyd), "differing_win": len(wind),
                             "differing_margin": len(mard),
                             "z0_won_z1_lost": z0only, "z1_won_z0_lost": z1only}
            if not anyd:
                failures.append(f"z0 and z1 are IDENTICAL on all 64 seeds for {arm}/{pole}")

    ct_liveness = {}
    for z in ("z0", "z1"):
        for pole in ("A", "B"):
            key = f"{z}_{pole}"
            anyd, wind, _, conly, tonly = split(cells[("CONTROL", z, pole)],
                                                cells[("TREATMENT", z, pole)])
            ct_liveness[key] = {"differing_any": len(anyd), "differing_win": len(wind),
                               "control_won_treatment_lost": conly,
                               "treatment_won_control_lost": tonly}
            if not anyd:
                failures.append(f"CONTROL and TREATMENT IDENTICAL for {z}/{pole}")

    wins = lambda c: np.array([c[s]["win"] for s in EVAL_SEEDS], dtype=np.float64)
    delta = {}
    for arm in ARMS:
        delta[f"A_{arm[0]}"] = _ci(wins(cells[(arm, "z0", "A")]) - wins(cells[(arm, "z1", "A")]))
        delta[f"B_{arm[0]}"] = _ci(wins(cells[(arm, "z1", "B")]) - wins(cells[(arm, "z0", "B")]))
    gamma_a = _ci((wins(cells[("TREATMENT", "z0", "A")]) - wins(cells[("TREATMENT", "z1", "A")]))
                  - (wins(cells[("CONTROL", "z0", "A")]) - wins(cells[("CONTROL", "z1", "A")])))
    gamma_b = _ci((wins(cells[("TREATMENT", "z1", "B")]) - wins(cells[("TREATMENT", "z0", "B")]))
                  - (wins(cells[("CONTROL", "z1", "B")]) - wins(cells[("CONTROL", "z0", "B")])))

    pe = flag.get("point_estimates") or {}
    for k, d in delta.items():
        if k in pe and abs(d["mean"] - float(pe[k])) > 1e-9:
            failures.append(f"{k} recomputed {d['mean']} != flag {pe[k]}")
    triggered = list(flag.get("triggered_by") or [])
    for k in triggered:
        if delta[k]["mean"] > 0.0:
            failures.append(f"triggered cell {k} is not <= 0 after recompute")
    for k, d in delta.items():
        if k not in triggered and d["mean"] <= 0.0:
            failures.append(f"{k} is <= 0 on recompute but was NOT in the original flag's "
                            f"triggered_by {triggered} -- the flag under-reported")

    swap = {}
    for arm in ARMS:
        swapped_a = _ci(wins(cells[(arm, "z1", "A")]) - wins(cells[(arm, "z0", "A")]))
        swapped_b = _ci(wins(cells[(arm, "z0", "B")]) - wins(cells[(arm, "z1", "B")]))
        swap[arm] = {
            "A_if_z0_z1_transposed": {**swapped_a, "would_pass": swapped_a["lcb95"] > 0},
            "B_if_z0_z1_transposed": {**swapped_b, "would_pass": swapped_b["lcb95"] > 0},
        }

    inc_sha = incumbent["checkpoint"]["sha256"]
    # SAC_RFT_MODELS_FROZEN.json does not carry warm_start sha directly on each arm the same
    # way trunk-freeze's does; read it from the raw production result instead.
    warm_shas = []
    for arm in ARMS:
        rec = frozen[arm]
        result_path = ROOT / rec["TERMINAL_RECORD_VALIDITY"]["result_record"]
        raw = json.loads(result_path.read_text(encoding="utf-8"))
        warm_shas.append(raw["launch_manifest"]["warm_start"]["sha256"])
    same_warm_start = all(s == inc_sha for s in warm_shas)
    inc_delta_a = incumbent["delta_A"]["mean"]
    inc_delta_b = incumbent["delta_B"]["mean"]
    if not same_warm_start:
        failures.append("incumbent EVAL checkpoint sha does not match SAC-RFT warm-start")

    agreement = {
        "A": {"control": delta["A_C"]["mean"], "treatment": delta["A_T"]["mean"],
             "same_direction": bool((delta["A_C"]["mean"] < 0) == (delta["A_T"]["mean"] < 0)),
             "gap": delta["A_T"]["mean"] - delta["A_C"]["mean"]},
        "B": {"control": delta["B_C"]["mean"], "treatment": delta["B_T"]["mean"],
             "same_direction": bool((delta["B_C"]["mean"] < 0) == (delta["B_T"]["mean"] < 0)),
             "gap": delta["B_T"]["mean"] - delta["B_C"]["mean"]},
    }

    a_c_rev = delta["A_C"]["mean"] <= 0
    b_t_rev = delta["B_T"]["mean"] <= 0
    a_t_pos = delta["A_T"]["mean"] > 0
    b_c_pos = delta["B_C"]["mean"] > 0
    cross_pattern = a_c_rev and b_t_rev and a_t_pos and b_c_pos

    off_diagonal_check = {
        "on_diagonal_both_positive": {"A_T": bool(a_t_pos), "B_C": bool(b_c_pos)},
        "off_diagonal_both_reversed": {"A_C": bool(a_c_rev), "B_T": bool(b_t_rev)},
        "pattern_is_exactly_cross_shaped": cross_pattern,
        "why_this_argues_against_a_shared_defect": "a defect tied to an ARM (e.g. TREATMENT's "
            "checkpoint loaded wrong) would hit both of that arm's poles, not just one; a "
            "defect tied to a POLE (e.g. Pole B's opponent key wrong) would hit both arms on "
            "that pole, not just one. This pattern requires the defect to be simultaneously "
            "arm-AND-pole-specific in exactly the two off-diagonal cells while sparing both "
            "on-diagonal cells -- a coincidence a code-path bug would not produce, since "
            "CONTROL and TREATMENT share the identical run_cell() code path and only the "
            "loaded checkpoint differs between them.",
    }

    if failures:
        verdict = "EVALUATOR_DEFECT_SUSPECTED"
    elif cross_pattern:
        verdict = "GENUINE_CROSS_REVERSAL_OFF_DIAGONAL"
    elif a_c_rev or b_t_rev:
        verdict = "GENUINE_REVERSAL"
    else:
        verdict = "GENUINE_ROWS_NO_REVERSAL"

    record = {
        "record": "SAC-RFT EVAL integrity audit", "status": "FROZEN_RESULT", "utc": _now(),
        "question": "Are the flagged reversals (CONTROL Pole A, TREATMENT Pole B) real "
                    "behavioural results, or an evaluator defect -- and specifically, can a "
                    "shared defect explain this off-diagonal cross pattern while sparing both "
                    "on-diagonal cells (TREATMENT Pole A, CONTROL Pole B)?",
        "triggered_by": triggered,
        "VERDICT": verdict,
        "checks": {
            "1_liveness_z0_vs_z1_per_arm": liveness,
            "1b_liveness_control_vs_treatment": ct_liveness,
            "2_off_diagonal_cross_pattern": off_diagonal_check,
            "3_deltas_and_gammas_recomputed": {
                "delta_A_CONTROL": delta["A_C"], "delta_B_CONTROL": delta["B_C"],
                "delta_A_TREATMENT": delta["A_T"], "delta_B_TREATMENT": delta["B_T"],
                "Gamma_A": gamma_a, "Gamma_B": gamma_b,
                "matches_flag_point_estimates": not any(
                    k in pe and abs(delta[k]["mean"] - float(pe[k])) > 1e-9 for k in delta),
            },
            "4_provenance": {"duplicate_rows": len(dupes), "missing_cells": missing,
                             "seed_block": [EVAL_SEEDS[0], EVAL_SEEDS[-1]],
                             "total_rows": len(rows)},
            "5_latent_swap_test": swap,
            "5b_cross_instrument_vs_incumbent_own_sealed_eval": {
                "same_warm_start_checkpoint": same_warm_start,
                "incumbent_delta_A": inc_delta_a, "incumbent_delta_B": inc_delta_b,
                "pole_A_was_positive_at_incumbent": bool(inc_delta_a > 0),
                "pole_B_was_negative_at_incumbent": bool(inc_delta_b < 0),
            },
            "6_control_treatment_agreement": agreement,
        },
        "conclusion": (
            "Rows recompute exactly to the flagged point estimates, latent forcing is live in "
            "every cell, no duplicate or missing rows, warm-start checkpoints match. The "
            "reversal pattern is off-diagonal (CONTROL fails Pole A, TREATMENT fails Pole B) "
            "while both on-diagonal cells are positive -- see check 2 for why a shared "
            "evaluator defect could not selectively produce this shape. Treated as a genuine "
            "behavioural result: neither retention mechanism (EMA or frozen anchor) preserves "
            "both poles simultaneously in this run; each preserves a different single pole."
            if verdict.startswith("GENUINE") else
            "The result cannot be treated as scientific until the defect is resolved."),
        "changes_no_verdict": True,
        "does_not_write_SAC_RFT_EVAL_RESULT": True,
        "failures": failures,
    }
    OUT.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"SAC-RFT EVAL INTEGRITY AUDIT  {_now()}")
    print(f"  delta_A_C {delta['A_C']['mean']:+.4f}  delta_B_C {delta['B_C']['mean']:+.4f}")
    print(f"  delta_A_T {delta['A_T']['mean']:+.4f}  delta_B_T {delta['B_T']['mean']:+.4f}")
    print(f"  VERDICT: {verdict}")
    for f in failures:
        print(f"    FAIL: {f}")
    print(f"  -> {OUT}")
    return 0 if verdict.startswith("GENUINE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
