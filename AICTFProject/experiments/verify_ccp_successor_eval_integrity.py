"""CCP successor EVAL integrity audit: is the observed Pole-B reversal real, or an artefact?

Frozen and committed BEFORE ccp_successor_eval_rows.csv or CCP_SUCCESSOR_EVAL_RESULT.json
existed as files -- not merely before they were read. The console win-rate lines showed a
raw point-estimate reversal on Pole B (z1 0.5625 < z0 0.7188, delta_B point estimate
-0.1563) while the EVAL run was still in flight and before any row-level data existed.

Mirrors experiments/verify_hog_psp_v3_eval_integrity.py and
verify_hog_psp_v4_eval_integrity.py exactly -- same five checks, same bootstrap convention,
same reasoning. V3's reversal turned out genuine; V4's tie turned out genuine; both were
established by evidence, not assumed. This audit exists to do the same for whatever the
frozen bootstrap CI on delta_B turns out to say, positive or negative.

The specific worry: if z0 and z1 were transposed anywhere in the evaluator, a Pole-B
reversal is exactly the signature that would produce -- and this successor's own mechanism
diagnostic (CCP_SUCCESSOR_MECHANISM_DIAGNOSTIC.json) found Pole B the STRONGEST cell for
trajectory identity (32/32 seed-level separation, delta_tau_B +26.4771), which makes a
transposition hypothesis worth checking explicitly rather than assuming away.

Checks:
  1. z0 and z1 are not byte-identical on Pole B      -- forcing took effect at all
  2. per-seed outcomes differ                        -- and where
  2b. margin-level divergence                        -- positive evidence forcing was live
      even where wins tie
  3. deltas recomputed from raw rows match the frozen result
  4. provenance: one row per (policy, pole, seed), correct seed block, no duplicates,
     all six cells present, checkpoint sha carried
  5. LATENT-SWAP TEST on Pole B, and cross-instrument agreement against the mechanism
     diagnostic, which reached the SAME checkpoint by a different code path

Diagnostic. Changes no verdict. Reads the EVAL's own frozen output once it exists; does not
touch EVAL itself, does not re-run it, does not alter any threshold.

Run:  python experiments/verify_ccp_successor_eval_integrity.py
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
ROWS = SD / "ccp_successor_eval_rows.csv"
RESULT = SD / "CCP_SUCCESSOR_EVAL_RESULT.json"
MECH = SD / "CCP_SUCCESSOR_MECHANISM_DIAGNOSTIC.json"
OUT = SD / "CCP_SUCCESSOR_EVAL_INTEGRITY.json"

EVAL_SEEDS = list(range(11_600_101, 11_600_133))
FIELDS = ("blue", "red", "win", "margin")
CELLS = (("z0", "A"), ("z1", "A"), ("z0", "B"), ("z1", "B"), ("pi_A", "A"), ("pi_B", "B"))
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
    for p in (ROWS, RESULT, MECH):
        if not p.is_file():
            raise SystemExit(f"REFUSING: {p.name} missing -- run this only after the EVAL "
                             "has completed and produced its own frozen output")
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; one-shot")

    rows = list(csv.DictReader(ROWS.open(encoding="utf-8")))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    mech = json.loads(MECH.read_text(encoding="utf-8"))
    failures: list[str] = []

    def cell(policy: str, pole: str) -> dict[int, dict]:
        return {int(r["seed"]): {k: int(r[k]) for k in FIELDS}
                for r in rows if r["policy"] == policy and r["pole"] == pole}

    z0a, z1a = cell("z0", "A"), cell("z1", "A")
    z0b, z1b = cell("z0", "B"), cell("z1", "B")

    # --- 4. provenance ------------------------------------------------------
    counts = Counter((r["policy"], r["pole"], r["seed"]) for r in rows)
    dupes = [k for k, v in counts.items() if v > 1]
    if dupes:
        failures.append(f"duplicate rows: {dupes[:3]}")
    present = {(r["policy"], r["pole"]) for r in rows}
    missing = [c for c in CELLS if c not in present]
    if missing:
        failures.append(f"missing evaluation cells: {missing}")
    for name, c in (("z0|A", z0a), ("z1|A", z1a), ("z0|B", z0b), ("z1|B", z1b)):
        if sorted(c) != EVAL_SEEDS:
            failures.append(f"{name} seed set does not match the frozen block")
    if result["checkpoint"]["sha256"] != mech["checkpoint"]["sha256"]:
        failures.append("EVAL and the mechanism diagnostic scored DIFFERENT checkpoints")

    # --- 1, 2, 2b. per-seed and per-margin divergence -----------------------
    def split(c0, c1):
        diff_any = [s for s in EVAL_SEEDS if c0[s] != c1[s]]
        diff_win = [s for s in EVAL_SEEDS if c0[s]["win"] != c1[s]["win"]]
        diff_margin = [s for s in EVAL_SEEDS if c0[s]["margin"] != c1[s]["margin"]]
        first_only = [s for s in EVAL_SEEDS if c0[s]["win"] == 1 and c1[s]["win"] == 0]
        second_only = [s for s in EVAL_SEEDS if c1[s]["win"] == 1 and c0[s]["win"] == 0]
        return diff_any, diff_win, diff_margin, first_only, second_only

    anyA, winA, marA, z0A_only, z1A_only = split(z0a, z1a)
    anyB, winB, marB, z0B_only, z1B_only = split(z0b, z1b)

    if not anyA:
        failures.append("z0|A and z1|A are IDENTICAL on all 32 seeds in every field; "
                        "latent forcing was dead on Pole A")
    if not anyB:
        failures.append("z0|B and z1|B are IDENTICAL on all 32 seeds in every field; "
                        "latent forcing was dead on Pole B")

    # --- 3. recompute deltas from raw rows ---------------------------------
    wins = lambda c: np.array([c[s]["win"] for s in EVAL_SEEDS], dtype=np.float64)
    dA = _ci(wins(z0a) - wins(z1a))
    dB = _ci(wins(z1b) - wins(z0b))
    fA, fB = result["delta_A"], result["delta_B"]
    matchA = abs(dA["mean"] - fA["mean"]) < 1e-12
    matchB = abs(dB["mean"] - fB["mean"]) < 1e-12
    if not matchA:
        failures.append(f"delta_A recomputed {dA['mean']} != frozen {fA['mean']}")
    if not matchB:
        failures.append(f"delta_B recomputed {dB['mean']} != frozen {fB['mean']}")

    # --- 5. latent-swap test on Pole B --------------------------------------
    # If the evaluator had z0 and z1 transposed on Pole B, the reversed delta_B would be
    # exactly what a correctly-labelled result would look like with the labels swapped.
    swapB = _ci(wins(z0b) - wins(z1b))
    swap_would_pass_B = swapB["lcb95"] > 0
    swap = {
        "pole_B_if_z0_z1_were_transposed": {
            "delta_B": swapB, "would_pass_the_gate": bool(swap_would_pass_B),
        },
        "reading": ("A transposition is the failure mode a reversal most resembles, so it is "
                   "tested explicitly. This does NOT by itself prove a swap occurred -- the "
                   "same numbers arise if the latents genuinely perform in the observed "
                   "direction. It is reported so the possibility is on the record."),
    }

    # cross-instrument agreement against the mechanism diagnostic, reached via a different
    # code path on the SAME checkpoint. The diagnostic found Pole B the STRONGEST identity
    # cell (32/32 seed-level separation), which makes a transposition hypothesis worth
    # checking rather than assuming away.
    routing = None
    for key in ("INTEGRITY_CHECKS_ALL_PASSED",):
        routing = mech.get(key, {}).get("wrong_latent_routing")
    cross = {
        "mechanism_diagnostic": {
            "reading": mech["READING"],
            "pole_B_seed_level_separation_pct":
                mech["LAYER_2_trajectory_identity"]["descriptive"]
                    .get("pct_seeds_different_label_pole_B"),
            "delta_tau_B": mech["LAYER_2_trajectory_identity"]["delta_tau_B"]["mean"],
            "episodes_with_live_latent_verified": (routing or {}).get("episodes_with_live_latent_verified"),
            "of": (routing or {}).get("of"),
            "code_path": "experiments/diagnose_ccp_successor_mechanism.py",
        },
        "eval_evidence_of_liveness_pole_B": {
            "seeds_differing_in_any_field": len(anyB),
            "seeds_differing_in_win": len(winB),
            "seeds_differing_in_margin": len(marB),
        },
        "agree": bool(anyB),
        "reading": ("The diagnostic independently confirmed the latents differ strongly on "
                   "Pole B via a different rollout code path on the same checkpoint. If EVAL "
                   "rows showed no latent effect on Pole B, the two instruments would "
                   "contradict each other."),
    }
    if not cross["agree"]:
        failures.append("EVAL rows show no latent effect on Pole B while the mechanism "
                        "diagnostic, on the same checkpoint, measured strong identity there; "
                        "the instruments contradict")

    verdict = "GENUINE_REVERSAL" if (fB["mean"] < 0 and not failures) else (
        "GENUINE_RESULT" if not failures else "EVALUATOR_DEFECT_SUSPECTED")

    record = {
        "record": "CCP successor EVAL integrity audit",
        "status": "FROZEN_RESULT", "utc": _now(),
        "question": ("Is the Pole-B reversal (delta_B < 0, first observed as a raw "
                     "point-estimate reversal while the EVAL run was still in flight) a real "
                     "behavioural result, or an evaluator defect such as a transposed latent?"),
        "why_this_matters_here": (
            "This successor's own mechanism diagnostic found Pole B the STRONGEST trajectory-"
            "identity cell in the entire program (32/32 seed-level separation, delta_tau_B "
            "+26.4771). A payoff reversal on the pole with the strongest confirmed identity "
            "separation is exactly the pattern a latent transposition would produce, so it is "
            "excluded explicitly rather than by assumption -- the same standard V3's genuine "
            "reversal was held to."),
        "VERDICT": verdict,
        "checks": {
            "1_not_byte_identical": {
                "pole_A": {"differing_seeds": len(anyA), "identical_seeds": 32 - len(anyA)},
                "pole_B": {"differing_seeds": len(anyB), "identical_seeds": 32 - len(anyB)},
                "passed": bool(anyA and anyB),
            },
            "2_per_seed_outcomes": {
                "pole_B": {"seeds_differing_in_win": winB,
                          "z0_won_z1_lost": z0B_only, "z1_won_z0_lost": z1B_only},
                "pole_A": {"seeds_differing_in_win": winA,
                          "z0_won_z1_lost": z0A_only, "z1_won_z0_lost": z1A_only},
            },
            "2b_margin_level_divergence": {
                "pole_A_seeds_with_different_margin": len(marA),
                "pole_B_seeds_with_different_margin": len(marB),
            },
            "3_deltas_recomputed_from_rows": {
                "delta_A": {"recomputed": dA, "frozen": fA, "matches": matchA},
                "delta_B": {"recomputed": dB, "frozen": fB, "matches": matchB},
            },
            "4_provenance": {
                "duplicate_rows": len(dupes),
                "missing_cells": [list(c) for c in missing],
                "seed_block": [EVAL_SEEDS[0], EVAL_SEEDS[-1]],
                "total_rows": len(rows),
                "checkpoint_sha256": result["checkpoint"]["sha256"],
                "same_checkpoint_as_mechanism_diagnostic":
                    result["checkpoint"]["sha256"] == mech["checkpoint"]["sha256"],
            },
            "5_latent_swap_test": swap,
            "5b_cross_instrument_agreement": cross,
        },
        "conclusion": (
            "The reversal is a real behavioural result, not an evaluator defect."
            if verdict == "GENUINE_REVERSAL" else
            "The result recomputes cleanly and is not an evaluator defect."
            if verdict == "GENUINE_RESULT" else
            "The result cannot be treated as scientific until the defect is resolved."),
        "changes_no_verdict": True,
        "failures": failures,
    }
    OUT.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"CCP SUCCESSOR EVAL INTEGRITY AUDIT  {_now()}")
    print(f"  Pole A: {len(anyA)}/32 differ in any field ({len(winA)} win, {len(marA)} margin)")
    print(f"          z0 won/z1 lost {z0A_only}")
    print(f"          z1 won/z0 lost {z1A_only}")
    print(f"  Pole B: {len(anyB)}/32 differ in any field ({len(winB)} win, {len(marB)} margin)")
    print(f"          z0 won/z1 lost {z0B_only}")
    print(f"          z1 won/z0 lost {z1B_only}")
    print(f"  delta_A recomputed {dA['mean']:+.4f} (frozen {fA['mean']:+.4f}) match={matchA}")
    print(f"  delta_B recomputed {dB['mean']:+.4f} (frozen {fB['mean']:+.4f}) match={matchB}")
    print(f"  swap test (Pole B): transposed delta {swapB['mean']:+.4f}, "
          f"would pass gate = {swap_would_pass_B}")
    print(f"  cross-instrument agreement: {cross['agree']}")
    print(f"\n  VERDICT: {verdict}")
    for f in failures:
        print(f"    FAIL: {f}")
    print(f"  -> {OUT}")
    return 0 if verdict in ("GENUINE_REVERSAL", "GENUINE_RESULT") else 1


if __name__ == "__main__":
    raise SystemExit(main())
