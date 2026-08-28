"""Stage 1 of calibration: MEASURE the world. Choose nothing.

Reads the CALIB split only and reports, per pole x regime x horizon cell, how much
of the terminal-margin target is actually resolvable:

    Delta(s) = M(pi_B | s) - M(pi_A | s)          deterministic, exact

    Delta > 0   -> B preferred
    Delta < 0   -> A preferred
    Delta == 0  -> preference not established        (a measurement limit,
                                                      NEVER a claim of equivalence)

    r_c = #(Delta != 0) / #branch states in cell     achievable resolvable mass

This file deliberately does NOT choose tau, rho or o_max, does not write
ABSTENTION_THRESHOLDS.json, and contains no default or fallback value for any of
them -- not even a commented-out one. A separate, preregistered threshold-selection
step consumes the frozen evidence written here.

That separation is the point: one program measures, another decides. If the same
program did both, its measurements could drift toward producing agreeable
thresholds, and nobody would be able to prove otherwise afterwards.

Run:  python experiments/calibrate_abstention_evidence.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.launch_gate import (                                    # noqa: E402
    SUPPORT_FLOOR,
    Check,
    LaunchGateError,
    check_collection_complete,
    check_final_untouched,
    check_seed_block,
    check_support_floor,
    format_checks,
)

SD = ROOT / "artifacts" / "strategic_demand"
DATA = SD / "stratified_regime_data"
AUDIT = DATA / "SUPPORT_VALIDITY.json"
PROTOCOL = SD / "sppo" / "ABSTAINING_SUPERVISION_PROTOCOL_DESIGN.json"
OUT = SD / "sppo" / "CALIBRATION_EVIDENCE.json"

CALIB_LO, CALIB_HI = 10_700_097, 10_700_128
CALIB_SEEDS = list(range(CALIB_LO, CALIB_HI + 1))
FINAL_RANGE = range(10_600_001, 10_600_193)
CELLS = [f"{p}_r{r}_{b}" for p in "AB" for r in range(4) for b in ("not_late", "late")]

# Project-wide convention since Gate 0B. Seed is the resampling unit, always.
N_BOOT, ALPHA, RNG_SEED = 20_000, 0.05, 7


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ci(draws: np.ndarray) -> dict:
    lo, hi = np.percentile(draws, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"mean": float(draws.mean()), "lcb95": float(lo), "ucb95": float(hi)}


def _seed_bootstrap(num: dict[int, float], den: dict[int, float]) -> dict | None:
    """Resample SEEDS, carrying every branch state of a sampled seed together.

    Treating thousands of branch states as thousands of independent samples would
    understate uncertainty by roughly the per-seed cluster size.
    """
    seeds = sorted(den)
    if not seeds:
        return None
    a = np.array([num.get(s, 0.0) for s in seeds], dtype=np.float64)
    b = np.array([den[s] for s in seeds], dtype=np.float64)
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, len(seeds), size=(N_BOOT, len(seeds)))
    d = b[idx].sum(axis=1)
    draws = np.divide(a[idx].sum(axis=1), d, out=np.full(N_BOOT, np.nan), where=d > 0)
    draws = draws[~np.isnan(draws)]
    return _ci(draws) if len(draws) else None


def preflight(data_dir: Path = DATA, audit_path: Path = AUDIT) -> list[Check]:
    """Refuse unless the evidence was genuinely earned."""
    checks = [
        check_collection_complete(data_dir),
        check_seed_block(data_dir),
        check_support_floor(audit_path),
        check_final_untouched(data_dir),
        _check_calib_present(data_dir),
    ]
    failed = [c for c in checks if c.blocking and not c.passed]
    if failed:
        raise LaunchGateError("CALIBRATION REFUSED\n" + format_checks(checks))
    return checks


def _check_calib_present(data_dir: Path) -> Check:
    shards = Path(data_dir) / "seed_shards"
    present = [s for s in CALIB_SEEDS if (shards / f"seed_{s}.npz").is_file()]
    if len(present) != len(CALIB_SEEDS):
        return Check("calib_split_complete", False,
                     f"CALIB split incomplete: {len(present)}/{len(CALIB_SEEDS)} seeds")
    return Check("calib_split_complete", True,
                 f"CALIB {CALIB_LO}..{CALIB_HI} complete ({len(present)} seeds)")


def _minimum_support_declared() -> int | None:
    """The per-cell seed minimum for calibration, if the PI has frozen one.

    Returns None when it has not been. This file will NOT invent one -- an
    adequacy bar chosen after seeing the counts is chosen to be met.
    """
    if not PROTOCOL.is_file():
        return None
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for block in spec.values():
        if isinstance(block, dict):
            value = block.get("minimum_calib_seeds_per_cell")
            if isinstance(value, int):
                return value
    return None


def collect_calib(data_dir: Path = DATA) -> dict:
    """Per-cell, per-seed label counts from the CALIB split only."""
    per_cell: dict[str, dict[int, dict[str, int]]] = defaultdict(lambda: defaultdict(
        lambda: {"b_preferred": 0, "a_preferred": 0, "not_established": 0}))
    shards = Path(data_dir) / "seed_shards"
    for seed in CALIB_SEEDS:
        path = shards / f"seed_{seed}.npz"
        if seed in FINAL_RANGE:                      # belt and braces
            raise LaunchGateError(f"REFUSING: {seed} is a FINAL seed")
        with np.load(path, allow_pickle=False) as z:
            cells = [str(c) for c in z["branch_cell"]]
            delta = ((z["branch_pi_B_blue"].astype(np.int64) - z["branch_pi_B_red"].astype(np.int64))
                     - (z["branch_pi_A_blue"].astype(np.int64) - z["branch_pi_A_red"].astype(np.int64)))
        for cell, d in zip(cells, delta):
            bucket = per_cell[cell][seed]
            key = "b_preferred" if d > 0 else ("a_preferred" if d < 0 else "not_established")
            bucket[key] += 1
    return per_cell


def build_evidence(per_cell: dict, minimum: int | None) -> dict:
    cells_out: dict[str, dict] = {}
    for cell in CELLS:
        by_seed = per_cell.get(cell, {})
        if not by_seed:
            cells_out[cell] = {"n_branch_states": 0, "n_seeds": 0,
                               "labels": {"b_preferred": 0, "a_preferred": 0, "not_established": 0},
                               "resolvable_mass": None,
                               "note": "no branch states in CALIB for this cell"}
            continue
        labels = {k: sum(v[k] for v in by_seed.values())
                  for k in ("b_preferred", "a_preferred", "not_established")}
        total = sum(labels.values())
        resolved_num = {s: v["b_preferred"] + v["a_preferred"] for s, v in by_seed.items()}
        den = {s: sum(v.values()) for s, v in by_seed.items()}
        cells_out[cell] = {
            "n_branch_states": total,
            "n_seeds": len(by_seed),
            "labels": labels,
            "resolvable_mass": _seed_bootstrap(resolved_num, den),
            "tie_rate_point": (labels["not_established"] / total) if total else None,
        }

    counted = {c: v for c, v in cells_out.items() if v["n_branch_states"] > 0}
    empty = [c for c, v in cells_out.items() if v["n_branch_states"] == 0]
    if minimum is None:
        verdict = "MINIMUM_SUPPORT_NOT_FROZEN"
        consequence = (
            "Evidence measured, but calibration MAY NOT PROCEED. The minimum per-cell "
            "CALIB seed support has never been frozen, and this program will not "
            "invent one -- an adequacy bar chosen after seeing these counts is a bar "
            "chosen to be met. A PI ruling must freeze "
            "'minimum_calib_seeds_per_cell' in the abstaining protocol first.")
        under = None
    else:
        under = sorted(c for c, v in counted.items() if v["n_seeds"] < minimum) + empty
        if under:
            verdict = "CALIBRATION_CANNOT_PROCEED"
            consequence = (f"{len(under)} cell(s) below the frozen minimum of {minimum} "
                           f"CALIB seeds: {under}. Stop; do not select thresholds and do "
                           "not grow the data.")
        else:
            verdict = "EVIDENCE_COMPLETE"
            consequence = ("All 16 cells meet the frozen minimum. The separate "
                           "threshold-selection step may consume this evidence.")

    return {
        "record": "Abstention calibration -- STAGE 1 EVIDENCE (measurement only)",
        "status": "FROZEN_EVIDENCE",
        "utc": _now(),
        "stage": "1 of 2. This program measures; a separate preregistered program decides.",
        "what_this_file_does_not_contain": (
            "tau, rho and o_max. No value, no default, no fallback. Selecting them "
            "here would let the measurement drift toward agreeable thresholds."),
        "split": {"name": "CALIB", "seeds": [CALIB_LO, CALIB_HI, len(CALIB_SEEDS)],
                  "FIT_used": False, "EVAL_used": False, "FINAL_touched": False},
        "quantity": "Delta(s) = M(pi_B|s) - M(pi_A|s), terminal win margin",
        "determinism": ("Branch continuations are deterministic from an exactly restored "
                        "state, so Delta is exact and a tie is a real tie, not a noisy draw. "
                        "No significance test is applied to an exact equality."),
        "label_semantics": {
            "b_preferred": "Delta > 0",
            "a_preferred": "Delta < 0",
            "not_established": ("Delta == 0 -- preference NOT established. A measurement "
                                "limit of a coarse target, never a claim that A and B are "
                                "equivalent.")},
        "bootstrap": {"unit": "seed (clustered)", "n_boot": N_BOOT, "alpha": ALPHA,
                      "rng_seed": RNG_SEED,
                      "why": ("Branch states within a seed are not independent; resampling "
                              "them individually would understate uncertainty by about the "
                              "per-seed cluster size.")},
        "minimum_calib_seeds_per_cell": minimum,
        "cells": cells_out,
        "cells_with_no_branch_states": empty,
        "cells_below_minimum": under,
        "VERDICT": verdict,
        "consequence": consequence,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    try:
        checks = preflight(Path(args.data_dir))
    except LaunchGateError as exc:
        print(str(exc))
        return 1
    print("CALIBRATION EVIDENCE (stage 1 of 2)")
    print(format_checks(checks))

    out = Path(args.out)
    if out.is_file():
        print(f"\nREFUSING: {out} exists; stage 1 is one-shot")
        return 1

    minimum = _minimum_support_declared()
    evidence = build_evidence(collect_calib(Path(args.data_dir)), minimum)

    print(f"\n  {'cell':18s} {'states':>7s} {'seeds':>6s} {'B':>6s} {'A':>6s} {'none':>6s} {'resolvable':>22s}")
    for cell in CELLS:
        v = evidence["cells"][cell]
        rm = v.get("resolvable_mass")
        band = (f"{rm['mean']:.3f} [{rm['lcb95']:.3f},{rm['ucb95']:.3f}]" if rm else "--")
        lab = v["labels"]
        print(f"  {cell:18s} {v['n_branch_states']:7d} {v['n_seeds']:6d} "
              f"{lab['b_preferred']:6d} {lab['a_preferred']:6d} {lab['not_established']:6d} {band:>22s}")

    out.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"\n  VERDICT: {evidence['VERDICT']}")
    print(f"  {evidence['consequence']}")
    print(f"  -> {out}")
    return 0 if evidence["VERDICT"] == "EVIDENCE_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
