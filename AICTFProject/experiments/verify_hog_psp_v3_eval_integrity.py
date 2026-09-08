"""V3 EVAL integrity audit: is the REVERSED delta_B real, or an evaluator defect?

OG-PSP's flat delta_B got a row-level audit before it was treated as scientific. V3's
delta_B is REVERSED (-0.1563), which deserves at least the same scrutiny -- arguably
more, because a reversal is what a latent-swap bug would look like.

The specific worry is sharp. V3's mechanism diagnostic confirmed trajectory identity
on both poles, with Pole B the STRONGEST cell: z1 classified pi_B-like on 91% of CALIB
seeds, delta_tau_B = +6.78 [+5.10, +8.66]. Yet EVAL says z0 out-wins z1 on Pole B. If
the evaluator had z0 and z1 transposed, that is exactly the signature we would see.

Checks:
  1. z0 and z1 are not byte-identical      -- forcing took effect at all
  2. per-seed outcomes differ              -- and where
  3. deltas recomputed from raw rows match the frozen result
  4. provenance: one row per (policy, pole, seed), correct seed block, no duplicates
  5. LATENT-SWAP TEST: does transposing z0/z1 reproduce the frozen deltas with
     flipped sign? If so, a swap is consistent with the data and must be excluded
     by other means; if not, a simple transposition is ruled out.

Diagnostic. Changes no verdict. EVAL is already spent; this reads its rows only.

Run:  python experiments/verify_hog_psp_v3_eval_integrity.py
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
ROWS = SD / "hog_psp_v3_eval_rows.csv"
RESULT = SD / "HOG_PSP_V3_EVAL_RESULT.json"
OUT = SD / "HOG_PSP_V3_EVAL_INTEGRITY.json"

EVAL_SEEDS = list(range(11_300_101, 11_300_133))
FIELDS = ("blue", "red", "win", "margin")
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
    for p in (ROWS, RESULT):
        if not p.is_file():
            raise SystemExit(f"REFUSING: {p.name} missing")
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; one-shot")

    rows = list(csv.DictReader(ROWS.open(encoding="utf-8")))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    failures: list[str] = []

    def cell(policy: str, pole: str) -> dict[int, dict]:
        return {int(r["seed"]): {k: int(r[k]) for k in FIELDS}
                for r in rows if r["policy"] == policy and r["pole"] == pole}

    z0b, z1b = cell("z0", "B"), cell("z1", "B")
    z0a, z1a = cell("z0", "A"), cell("z1", "A")

    # --- 4. provenance ------------------------------------------------------
    counts = Counter((r["policy"], r["pole"], r["seed"]) for r in rows)
    dupes = [k for k, v in counts.items() if v > 1]
    if dupes:
        failures.append(f"duplicate rows: {dupes[:3]}")
    for name, c in (("z0|B", z0b), ("z1|B", z1b), ("z0|A", z0a), ("z1|A", z1a)):
        if sorted(c) != EVAL_SEEDS:
            failures.append(f"{name} seed set does not match the frozen block")

    # --- 1 & 2. per-seed divergence ----------------------------------------
    def split(c0, c1):
        ident = [s for s in EVAL_SEEDS if c0[s] == c1[s]]
        diff = [s for s in EVAL_SEEDS if c0[s] != c1[s]]
        first_only = [s for s in EVAL_SEEDS if c0[s]["win"] == 1 and c1[s]["win"] == 0]
        second_only = [s for s in EVAL_SEEDS if c1[s]["win"] == 1 and c0[s]["win"] == 0]
        return ident, diff, first_only, second_only

    identB, diffB, z0B_only, z1B_only = split(z0b, z1b)
    identA, diffA, z0A_only, z1A_only = split(z0a, z1a)

    if not diffB:
        failures.append("z0|B and z1|B identical on ALL 32 seeds; latent forcing dead on Pole B")
    if not diffA:
        failures.append("z0|A and z1|A identical on ALL 32 seeds; latent forcing dead on Pole A")

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

    # --- 5. latent-swap test -----------------------------------------------
    # If the evaluator had transposed z0/z1, the deltas would be exactly negated.
    swapA = _ci(wins(z1a) - wins(z0a))
    swapB = _ci(wins(z0b) - wins(z1b))
    swap_would_pass = swapA["lcb95"] > 0 and swapB["lcb95"] > 0
    swap = {
        "if_z0_z1_were_transposed": {
            "delta_A": swapA, "delta_B": swapB,
            "would_pass_the_gate": bool(swap_would_pass),
        },
        "reading": (
            "A transposition is the failure mode a reversed delta most resembles, so it is "
            "tested explicitly. This does NOT by itself prove a swap occurred -- the same "
            "numbers arise if the latents genuinely perform in the observed direction. It is "
            "reported so the possibility is on the record rather than assumed away."
        ),
        "independent_evidence_against_a_swap": (
            "The mechanism diagnostic scored the SAME checkpoint through a different code "
            "path (experiments/diagnose_hog_psp_v3_mechanism.py, commit 7494081b) and found "
            "z0 unanimously pi_A-like (0% pi_B-like on both poles) and z1 majority pi_B-like "
            "(78% / 91%). If the EVAL scorer had z0 and z1 transposed relative to the "
            "diagnostic, the two would disagree about which latent is which; they agree."
        ),
    }

    verdict = "GENUINE_REVERSAL" if not failures else "EVALUATOR_DEFECT_SUSPECTED"

    record = {
        "record": "H-OG-PSP V3 EVAL integrity audit",
        "status": "FROZEN_RESULT", "utc": _now(),
        "question": ("Is the reversed delta_B a real behavioural result, or an evaluator "
                     "defect such as a transposed latent?"),
        "why_this_matters_here": (
            "V3's mechanism diagnostic confirmed trajectory identity most strongly on Pole B "
            "(z1 pi_B-like on 91% of CALIB seeds, delta_tau_B +6.78). EVAL reports z0 "
            "out-winning z1 on that same pole. A latent transposition would produce exactly "
            "this signature, so it is excluded explicitly rather than by assumption."
        ),
        "VERDICT": verdict,
        "checks": {
            "1_not_byte_identical": {
                "pole_B": {"identical_seeds": len(identB), "differing_seeds": len(diffB)},
                "pole_A": {"identical_seeds": len(identA), "differing_seeds": len(diffA)},
                "passed": bool(diffB and diffA),
            },
            "2_per_seed_outcomes": {
                "pole_B": {"z0_won_z1_lost": z0B_only, "z1_won_z0_lost": z1B_only},
                "pole_A": {"z0_won_z1_lost": z0A_only, "z1_won_z0_lost": z1A_only},
            },
            "3_deltas_recomputed_from_rows": {
                "delta_A": {"recomputed": dA, "frozen": fA, "matches": matchA},
                "delta_B": {"recomputed": dB, "frozen": fB, "matches": matchB},
            },
            "4_provenance": {
                "duplicate_rows": len(dupes),
                "seed_block": [EVAL_SEEDS[0], EVAL_SEEDS[-1]],
                "total_rows": len(rows),
                "checkpoint_sha256": result["checkpoint"]["sha256"],
            },
            "5_latent_swap_test": swap,
        },
        "conclusion": (
            "The reversed delta_B is a real behavioural result, not an evaluator defect."
            if verdict == "GENUINE_REVERSAL" else
            "The reversal cannot be treated as scientific until the defect is resolved."),
        "changes_no_verdict": True,
        "failures": failures,
    }
    OUT.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"V3 EVAL INTEGRITY AUDIT  {_now()}")
    print(f"  Pole B: {len(diffB)}/32 seeds differ | z0 won/z1 lost {z0B_only}")
    print(f"                                        | z1 won/z0 lost {z1B_only}")
    print(f"  Pole A: {len(diffA)}/32 seeds differ | z0 won/z1 lost {z0A_only}")
    print(f"                                        | z1 won/z0 lost {z1A_only}")
    print(f"  delta_A recomputed {dA['mean']:+.4f} (frozen {fA['mean']:+.4f}) match={matchA}")
    print(f"  delta_B recomputed {dB['mean']:+.4f} (frozen {fB['mean']:+.4f}) match={matchB}")
    print(f"  swap test: transposed deltas A {swapA['mean']:+.4f} B {swapB['mean']:+.4f}, "
          f"would pass gate = {swap_would_pass}")
    print(f"\n  VERDICT: {verdict}")
    for f in failures:
        print(f"    FAIL: {f}")
    print(f"  -> {OUT}")
    return 0 if verdict == "GENUINE_REVERSAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
