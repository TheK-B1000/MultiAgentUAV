"""RSCFT EVAL integrity audit: are the flagged Pole-A reversals real, or an evaluator defect?

Triggered by RSCFT_EVAL_INTEGRITY_REQUIRED.json (A_C, A_T). Does not write
RSCFT_EVAL_RESULT.json. Classifies GENUINE_* vs EVALUATOR_DEFECT_SUSPECTED.

Run:  python experiments/verify_rscft_eval_integrity.py
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
ROWS = SD / "rscft_eval_rows.csv"
FLAG = SD / "RSCFT_EVAL_INTEGRITY_REQUIRED.json"
FROZEN = SD / "RSCFT_MODELS_FROZEN.json"
INCUMBENT_EVAL = SD / "CCP_SUCCESSOR_EVAL_RESULT.json"
OUT = SD / "RSCFT_EVAL_INTEGRITY.json"

EVAL_SEEDS = list(range(11_704_001, 11_704_065))
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

    swap = {}
    for arm in ARMS:
        swapped_a = _ci(wins(cells[(arm, "z1", "A")]) - wins(cells[(arm, "z0", "A")]))
        swapped_b = _ci(wins(cells[(arm, "z0", "B")]) - wins(cells[(arm, "z1", "B")]))
        swap[arm] = {
            "A_if_z0_z1_transposed": {**swapped_a, "would_pass": swapped_a["lcb95"] > 0},
            "B_if_z0_z1_transposed": {**swapped_b, "would_pass": swapped_b["lcb95"] > 0},
        }

    inc_sha = incumbent["checkpoint"]["sha256"]
    warm = [frozen[arm]["WARM_START_SOURCE"]["sha256"] for arm in ARMS]
    same_warm_start = all(s == inc_sha for s in warm)
    inc_delta_a = incumbent["delta_A"]["mean"]
    inc_delta_b = incumbent["delta_B"]["mean"]
    if not same_warm_start:
        failures.append("incumbent EVAL checkpoint sha does not match RSCFT warm-start")

    agreement = {
        "A": {"control": delta["A_C"]["mean"], "treatment": delta["A_T"]["mean"],
             "same_direction": bool((delta["A_C"]["mean"] < 0) == (delta["A_T"]["mean"] < 0)),
             "gap": delta["A_T"]["mean"] - delta["A_C"]["mean"]},
        "B": {"control": delta["B_C"]["mean"], "treatment": delta["B_T"]["mean"],
             "same_direction": bool((delta["B_C"]["mean"] < 0) == (delta["B_T"]["mean"] < 0)),
             "gap": delta["B_T"]["mean"] - delta["B_C"]["mean"]},
    }

    pole_a_reversal = delta["A_C"]["mean"] <= 0 and delta["A_T"]["mean"] <= 0
    if pole_a_reversal and not failures:
        verdict = "GENUINE_REVERSAL_POLE_A"
    elif not failures:
        verdict = "GENUINE_ROWS_NO_POLE_A_REVERSAL"
    else:
        verdict = "EVALUATOR_DEFECT_SUSPECTED"

    record = {
        "record": "RSCFT EVAL integrity audit", "status": "FROZEN_RESULT", "utc": _now(),
        "question": "Are the flagged Pole-A reversals (A_C, A_T) real behavioural results, "
                    "or an evaluator defect such as a transposed latent?",
        "triggered_by": triggered,
        "VERDICT": verdict,
        "checks": {
            "1_liveness_z0_vs_z1_per_arm": liveness,
            "1b_liveness_control_vs_treatment": ct_liveness,
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
            "Pole-A reversals recompute from raw rows, latent forcing was live, CONTROL and "
            "TREATMENT differ, and the pattern is consistent with a real behavioural result "
            "rather than an evaluator defect."
            if verdict.startswith("GENUINE") else
            "The result cannot be treated as scientific until the defect is resolved."),
        "changes_no_verdict": True,
        "does_not_write_RSCFT_EVAL_RESULT": True,
        "failures": failures,
    }
    OUT.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"RSCFT EVAL INTEGRITY AUDIT  {_now()}")
    print(f"  delta_A_C {delta['A_C']['mean']:+.4f}  delta_B_C {delta['B_C']['mean']:+.4f}")
    print(f"  delta_A_T {delta['A_T']['mean']:+.4f}  delta_B_T {delta['B_T']['mean']:+.4f}")
    print(f"  VERDICT: {verdict}")
    for f in failures:
        print(f"    FAIL: {f}")
    print(f"  -> {OUT}")
    return 0 if verdict.startswith("GENUINE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
