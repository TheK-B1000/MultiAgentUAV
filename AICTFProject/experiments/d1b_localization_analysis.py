"""D1B -- pre-specified localization follow-up. Implements D1_SPEC_FROZEN.json.

POSTMORTEM FOLLOW-UP, NOT independent confirmation, NOT a new gate: own_flag_
stolen was discovered FROM D0 on this same data. Reads only the EXISTING
D0_pole_b_decision_rows.csv -- no new seeds, no new rollouts.

Recomputes D0's Q2 (worst-quartile Q_psi correct-rate) on three populations,
with the quartile cutoff recomputed inside each seed-bootstrap replicate exactly
as in D0:

  (a) all rows           -- D0's original Q2, for reference
  (b) own_flag_stolen EXCLUDED
  (c) own_flag_home only -- the complementary stratum

Run:  python experiments/d1b_localization_analysis.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPPO = ROOT / "artifacts/strategic_demand/sppo"
ROWS = SPPO / "D0_pole_b_decision_rows.csv"
SPEC = SPPO / "D1_SPEC_FROZEN.json"
OUT = SPPO / "D1B_LOCALIZATION_ANALYSIS.json"

N_BOOT, ALPHA, RNG_SEED = 20_000, 0.05, 7
QPSI_MARGIN = 0.04


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ci(draws: np.ndarray) -> dict:
    lo, hi = np.percentile(draws, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"mean": float(draws.mean()), "lcb95": float(lo), "ucb95": float(hi)}


def worst_quartile_bootstrap(mB_by_seed: dict, dq_by_seed: dict, seeds: list) -> dict:
    """Identical procedure to D0's Q2: quartile cutoff recomputed per replicate."""
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, len(seeds), size=(N_BOOT, len(seeds)))
    rate, mean_mB = np.full(N_BOOT, np.nan), np.full(N_BOOT, np.nan)
    for b in range(N_BOOT):
        picked = [seeds[j] for j in idx[b]]
        mB = np.concatenate([mB_by_seed[s] for s in picked if len(mB_by_seed[s])])
        dq = np.concatenate([dq_by_seed[s] for s in picked if len(dq_by_seed[s])])
        if len(mB) == 0:
            continue
        cut = float(np.percentile(mB, 25))
        worst = mB <= cut
        if worst.any():
            rate[b] = float((dq[worst] > QPSI_MARGIN).mean())
            mean_mB[b] = float(mB[worst].mean())
    return {
        "qpsi_correct_rate_worst_quartile": _ci(rate[~np.isnan(rate)]),
        "mean_margin_B_worst_quartile": _ci(mean_mB[~np.isnan(mean_mB)]),
        "n_seeds": len(seeds),
        "n_bootstrap_replicates_with_data": int((~np.isnan(rate)).sum()),
    }


def main() -> int:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    bs = spec["D1B_prespecified_localization_analysis"]["bootstrap"]
    if (bs["n_boot"], bs["alpha"], bs["rng_seed"]) != (N_BOOT, ALPHA, RNG_SEED):
        raise SystemExit("REFUSING: bootstrap params drifted from the frozen D1B spec")

    rows = list(csv.DictReader(open(ROWS, newline="", encoding="utf-8")))
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[int(r["seed"])].append(r)
    seeds = sorted(by_seed)
    if len(seeds) != 192:
        raise SystemExit(f"REFUSING: {len(seeds)} seeds in D0 rows, expected 192")

    def arrays(pred):
        mB, dq = {}, {}
        for s in seeds:
            sel = [r for r in by_seed[s] if pred(r)]
            mB[s] = np.array([float(r["margin_B_bits"]) for r in sel])
            dq[s] = np.array([float(r["delta_B_hat_qpsi"]) for r in sel])
        return mB, dq

    # NOTE: D0 tagged own_flag_home as a clean binary (home vs stolen), so
    # "stolen excluded" and "home only" are the IDENTICAL row set here -- not a
    # bug, a fact about the tag. Reported once under both names for traceability
    # against the frozen spec's wording. own_flag_stolen_only is the complement,
    # added because isolating it under THIS quartile-conditional procedure (vs
    # D0's flat category breakdown) is a genuinely different, informative cut.
    populations = {
        "all_rows_D0_reference": lambda r: True,
        "own_flag_stolen_excluded_EQUALS_own_flag_home_only": lambda r: r["own_flag_home"] == "1",
        "own_flag_stolen_only": lambda r: r["own_flag_home"] == "0",
    }

    result = {}
    print(f"D1B LOCALIZATION ANALYSIS  {_now()}\n")
    for name, pred in populations.items():
        mB, dq = arrays(pred)
        active_seeds = [s for s in seeds if len(mB[s]) > 0]
        r = worst_quartile_bootstrap(mB, dq, active_seeds)
        result[name] = r
        c = r["qpsi_correct_rate_worst_quartile"]
        print(f"  {name:26s} n_seeds={len(active_seeds):3d}  "
              f"qpsi_correct {c['mean']:.3f} [{c['lcb95']:.3f}, {c['ucb95']:.3f}]")

    rec = {
        "record": "D1B pre-specified localization follow-up",
        "status": "DIAGNOSTIC_ONLY -- POSTMORTEM FOLLOW-UP, not independent confirmation, not a gate",
        "utc": _now(),
        "data_source": "D0_pole_b_decision_rows.csv (existing, no new seeds/rollouts)",
        "bootstrap": {"unit": "seed", "n_boot": N_BOOT, "alpha": ALPHA, "rng_seed": RNG_SEED,
                     "quartile_cutoff": "recomputed inside every replicate, identical to D0 Q2"},
        "populations": result,
    }
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
