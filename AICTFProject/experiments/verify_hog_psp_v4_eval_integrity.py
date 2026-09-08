"""V4 EVAL integrity audit: are the DOUBLE exact ties real, or an evaluator defect?

OG-PSP's flat delta_B got a row-level audit before it was treated as scientific, and V3's
reversed delta_B got one too. V4 reports delta_A = delta_B = exactly 0.0000, with all four
latent cells tying their partner to the digit. A tie on BOTH poles is the single most
defect-shaped result this evaluator can produce, because it is exactly what a DEAD LATENT
FORCING would produce: if the scorer never actually forced z, z0 and z1 would run the same
policy, roll identical episodes, and tie by construction on every seed.

That worry has real teeth here. V4's own mechanism diagnostic, on the SAME checkpoint,
found the two latents behaviourally distinct on CALIB (JSD 0.0819 nats, delta_tau_A +0.98
[+0.53, +1.46]) with the live latent re-asserted from strategy_info() on 128 of 128
episodes. Two latents that measurably differ producing bit-for-bit identical payoff rows
would be a contradiction, not a finding.

Checks:
  1. z0 and z1 are not byte-identical         -- forcing took effect at all, per pole
  2. per-seed outcomes differ                 -- and where
  2b. MARGIN-level divergence                 -- win vectors can tie while margins differ;
                                                 that alone proves forcing was live
  3. deltas recomputed from raw rows match the frozen result
  4. provenance: one row per (policy, pole, seed), correct seed block, no duplicates,
     all six cells present, checkpoint sha carried
  5. cross-instrument agreement: does the EVAL evidence of latent liveness agree with the
     mechanism diagnostic, which reached the same checkpoint by a different code path?

Diagnostic. Changes no verdict. EVAL is already spent; this reads its rows only.

Run:  python experiments/verify_hog_psp_v4_eval_integrity.py
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
ROWS = SD / "hog_psp_v4_eval_rows.csv"
RESULT = SD / "HOG_PSP_V4_EVAL_RESULT.json"
MECH = SD / "HOG_PSP_V4_MECHANISM_DIAGNOSTIC.json"
OUT = SD / "HOG_PSP_V4_EVAL_INTEGRITY.json"

EVAL_SEEDS = list(range(11_400_101, 11_400_133))
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
            raise SystemExit(f"REFUSING: {p.name} missing")
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

    # --- 5. cross-instrument agreement -------------------------------------
    # The mechanism diagnostic reached the same checkpoint through a different code path
    # and found the latents behaviourally distinct with routing verified 128/128. If EVAL
    # rows were bit-identical, the two instruments would contradict each other.
    routing = mech["INTEGRITY_CHECKS_ALL_PASSED"]["wrong_latent_routing"]
    cross = {
        "mechanism_diagnostic": {
            "reading": mech["READING"],
            "jsd_nats": mech["LAYER_1_state_level"]["z0_z1_jsd_nats"],
            "delta_tau_A": mech["LAYER_2_trajectory_identity"]["delta_tau_A"]["mean"],
            "episodes_with_live_latent_verified": routing["episodes_with_live_latent_verified"],
            "of": routing["of"],
            "code_path": "experiments/diagnose_hog_psp_v4_mechanism.py (commit f8301faa)",
        },
        "eval_evidence_of_liveness": {
            "pole_A_seeds_differing_in_any_field": len(anyA),
            "pole_B_seeds_differing_in_any_field": len(anyB),
        },
        "agree": bool(anyA and anyB),
        "reading": (
            "Two instruments, two code paths, one checkpoint. The diagnostic says the "
            "latents differ behaviourally; EVAL rows must therefore differ somewhere. If "
            "they did not, the tie would be an artefact of dead forcing rather than a "
            "result, and this audit would say so."
        ),
    }
    if not cross["agree"]:
        failures.append("EVAL rows show no latent effect while the mechanism diagnostic, on "
                        "the same checkpoint, measured one; the instruments contradict")

    verdict = "GENUINE_TIE" if not failures else "EVALUATOR_DEFECT_SUSPECTED"

    record = {
        "record": "H-OG-PSP V4 EVAL integrity audit",
        "status": "FROZEN_RESULT", "utc": _now(),
        "question": ("delta_A and delta_B are both exactly 0.0000. Is that a real payoff "
                     "tie, or a dead latent forcing in the evaluator?"),
        "why_this_matters_here": (
            "A tie on BOTH poles is the most defect-shaped result this evaluator can return. "
            "If z were never actually forced, z0 and z1 would run the same policy and tie by "
            "construction on every seed. That possibility is excluded by row-level evidence "
            "rather than by assumption."
        ),
        "VERDICT": verdict,
        "checks": {
            "1_not_byte_identical": {
                "pole_A": {"seeds_differing_in_any_field": len(anyA),
                           "identical_seeds": 32 - len(anyA)},
                "pole_B": {"seeds_differing_in_any_field": len(anyB),
                           "identical_seeds": 32 - len(anyB)},
                "passed": bool(anyA and anyB),
            },
            "2_per_seed_outcomes": {
                "pole_A": {"seeds_differing_in_win": winA,
                           "z0_won_z1_lost": z0A_only, "z1_won_z0_lost": z1A_only},
                "pole_B": {"seeds_differing_in_win": winB,
                           "z0_won_z1_lost": z0B_only, "z1_won_z0_lost": z1B_only},
                "note": ("A genuine tie means these cancel, not that they are empty. Empty "
                         "lists on both poles would instead indicate dead forcing."),
            },
            "2b_margin_level_divergence": {
                "pole_A_seeds_with_different_margin": len(marA),
                "pole_B_seeds_with_different_margin": len(marB),
                "why": ("Win vectors can tie while margins differ. Margin divergence is "
                        "positive evidence that the two latents really played differently, "
                        "independent of who won."),
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
            "5_cross_instrument_agreement": cross,
        },
        "conclusion": (
            "The double tie is a real payoff result, not a dead latent forcing."
            if verdict == "GENUINE_TIE" else
            "The tie cannot be treated as scientific until the defect is resolved."),
        "changes_no_verdict": True,
        "what_this_does_not_establish": (
            "That V4's latents are equivalent policies. They are measurably different "
            "behaviourally; this audit only establishes that their PAYOFFS tied and that "
            "the evaluator was working when it said so."),
        "failures": failures,
    }
    OUT.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"V4 EVAL INTEGRITY AUDIT  {_now()}")
    print(f"  Pole A: {len(anyA)}/32 seeds differ in some field "
          f"({len(winA)} in win, {len(marA)} in margin)")
    print(f"          z0 won/z1 lost {z0A_only}")
    print(f"          z1 won/z0 lost {z1A_only}")
    print(f"  Pole B: {len(anyB)}/32 seeds differ in some field "
          f"({len(winB)} in win, {len(marB)} in margin)")
    print(f"          z0 won/z1 lost {z0B_only}")
    print(f"          z1 won/z0 lost {z1B_only}")
    print(f"  delta_A recomputed {dA['mean']:+.4f} (frozen {fA['mean']:+.4f}) match={matchA}")
    print(f"  delta_B recomputed {dB['mean']:+.4f} (frozen {fB['mean']:+.4f}) match={matchB}")
    print(f"  cross-instrument agreement: {cross['agree']}")
    print(f"\n  VERDICT: {verdict}")
    for f in failures:
        print(f"    FAIL: {f}")
    print(f"  -> {OUT}")
    return 0 if verdict == "GENUINE_TIE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
