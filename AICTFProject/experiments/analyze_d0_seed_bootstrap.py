"""D0 seed-level bootstrap analysis -- implements D0_REPORTING_SPEC_FROZEN.json.

WRITTEN AND COMMITTED BEFORE D0's 192-seed rows existed. The spec was frozen
while the diagnostic was still running so the analysis is fixed rather than
chosen after seeing the outcome.

Bootstrap is at the SEED level. Decision points within an episode are heavily
clustered -- the 8-seed smoke produced 1,539 rows whose effective sample size is
nearer 8 than 1,539 -- so resampling rows would badly understate uncertainty.
Each replicate resamples the 192 seeds with replacement, carries every decision
point belonging to each sampled seed, and recomputes the quantity inside the
replicate. The worst-quartile cutoff is RECOMPUTED PER REPLICATE, so the
interval reflects both which seeds were drawn and where the failure-region
boundary lands.

MECHANISM DIAGNOSTIC ONLY. No PASS/FAIL, no threshold, no gate. Cannot alter
the frozen SPPPO_V1_NOT_CONFIRMED verdict.

Run:  python experiments/analyze_d0_seed_bootstrap.py
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

SPPO = ROOT / "artifacts" / "strategic_demand" / "sppo"
ROWS = SPPO / "D0_pole_b_decision_rows.csv"
SPEC = SPPO / "D0_REPORTING_SPEC_FROZEN.json"
OUT = SPPO / "D0_SEED_BOOTSTRAP.json"

N_BOOT, ALPHA, RNG_SEED = 20_000, 0.05, 7
QPSI_MARGIN = 0.04
EXPECTED_SEEDS = 192


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ci(draws: np.ndarray) -> dict:
    lo, hi = np.percentile(draws, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"mean": float(draws.mean()), "lcb95": float(lo), "ucb95": float(hi)}


def main() -> int:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["bootstrap"]["n_boot"] != N_BOOT or spec["bootstrap"]["rng_seed"] != RNG_SEED:
        raise SystemExit("REFUSING: analysis drifted from the frozen reporting spec")

    rows = list(csv.DictReader(open(ROWS, newline="", encoding="utf-8")))
    by_seed: dict[int, list] = defaultdict(list)
    for r in rows:
        by_seed[int(r["seed"])].append(r)
    seeds = sorted(by_seed)
    if len(seeds) != EXPECTED_SEEDS:
        raise SystemExit(f"REFUSING: {len(seeds)} seeds present, expected {EXPECTED_SEEDS}")

    # per-seed arrays, so a replicate is a concatenation of whole episodes
    mB_by_seed = {s: np.array([float(r["margin_B_bits"]) for r in by_seed[s]]) for s in seeds}
    dq_by_seed = {s: np.array([float(r["delta_B_hat_qpsi"]) for r in by_seed[s]]) for s in seeds}
    cat_by_seed = {}
    for s in seeds:
        rs = by_seed[s]
        cat_by_seed[s] = {
            "carrying": np.array([r["blue_carrying"] == "1" for r in rs]),
            "own_flag_home": np.array([r["own_flag_home"] == "1" for r in rs]),
            "early": np.array([r["tertile"] == "early" for r in rs]),
            "mid": np.array([r["tertile"] == "mid" for r in rs]),
            "late": np.array([r["tertile"] == "late" for r in rs]),
        }

    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, len(seeds), size=(N_BOOT, len(seeds)))

    q1_frac_neg, q1_mean_mB = np.empty(N_BOOT), np.empty(N_BOOT)
    q2_rate, q2_cut, q2_mean_dq = np.empty(N_BOOT), np.empty(N_BOOT), np.empty(N_BOOT)
    cat_keys = ["carrying", "not_carrying", "own_flag_home", "own_flag_stolen",
                "early", "mid", "late"]
    cat_mB = {k: np.full(N_BOOT, np.nan) for k in cat_keys}
    cat_ok = {k: np.full(N_BOOT, np.nan) for k in cat_keys}

    for b in range(N_BOOT):
        picked = [seeds[j] for j in idx[b]]
        mB = np.concatenate([mB_by_seed[s] for s in picked])
        dq = np.concatenate([dq_by_seed[s] for s in picked])
        ok = dq > QPSI_MARGIN

        q1_frac_neg[b] = float((mB < 0).mean())
        q1_mean_mB[b] = float(mB.mean())

        cut = float(np.percentile(mB, 25))          # recomputed INSIDE the replicate
        worst = mB <= cut
        q2_cut[b] = cut
        q2_rate[b] = float(ok[worst].mean()) if worst.any() else np.nan
        q2_mean_dq[b] = float(dq[worst].mean()) if worst.any() else np.nan

        for key in cat_keys:
            base = key.replace("not_", "").replace("_stolen", "_home")
            m = np.concatenate([cat_by_seed[s][base] for s in picked])
            if key.startswith("not_") or key.endswith("_stolen"):
                m = ~m
            if m.any():
                cat_mB[key][b] = float(mB[m].mean())
                cat_ok[key][b] = float(ok[m].mean())

    rec = {
        "record": "D0 seed-level bootstrap (implements D0_REPORTING_SPEC_FROZEN.json)",
        "status": "MECHANISM_DIAGNOSTIC_ONLY -- no gate, no threshold, no verdict change",
        "utc": _now(),
        "spec_frozen_before_data": True,
        "n_seeds": len(seeds), "n_decision_points": len(rows),
        "bootstrap": {"unit": "seed", "n_boot": N_BOOT, "alpha": ALPHA, "rng_seed": RNG_SEED,
                      "quartile_recomputed_per_replicate": True},
        "Q1_failure_is_z0_closer_to_piB": {
            "fraction_margin_B_negative": _ci(q1_frac_neg),
            "mean_margin_B_bits": _ci(q1_mean_mB),
            "reading": "margin_B_bits = KL(pi_B||z0) - KL(pi_B||z1); negative = z0 closer to pi_B",
        },
        "Q2_qpsi_ranking_in_the_failure_region": {
            "qpsi_correct_rate_worst_quartile": _ci(q2_rate[~np.isnan(q2_rate)]),
            "mean_delta_B_hat_worst_quartile": _ci(q2_mean_dq[~np.isnan(q2_mean_dq)]),
            "worst_quartile_cutoff": _ci(q2_cut),
            "qpsi_margin_threshold": QPSI_MARGIN,
        },
        "Q3_clustering_by_category": {
            k: {"mean_margin_B_bits": _ci(cat_mB[k][~np.isnan(cat_mB[k])]),
                "qpsi_correct_rate": _ci(cat_ok[k][~np.isnan(cat_ok[k])])}
            for k in cat_keys if not np.isnan(cat_mB[k]).all()
        },
        "WORDING_DISCIPLINE": spec["WORDING_DISCIPLINE"],
    }
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    q1, q2 = rec["Q1_failure_is_z0_closer_to_piB"], rec["Q2_qpsi_ranking_in_the_failure_region"]
    f = lambda d: f"{d['mean']:+.4f} [{d['lcb95']:+.4f}, {d['ucb95']:+.4f}]"
    print(f"D0 SEED-LEVEL BOOTSTRAP  {_now()}")
    print(f"  seeds {len(seeds)}   decision points {len(rows)}   n_boot {N_BOOT}\n")
    print("Q1  fraction margin_B < 0     :", f(q1["fraction_margin_B_negative"]))
    print("Q1  mean margin_B_bits        :", f(q1["mean_margin_B_bits"]))
    print("Q2  qpsi correct-rate (worst) :", f(q2["qpsi_correct_rate_worst_quartile"]))
    print("Q2  mean delta_B_hat (worst)  :", f(q2["mean_delta_B_hat_worst_quartile"]))
    print("\nQ3 by category (mean margin_B / qpsi correct-rate):")
    for k, v in rec["Q3_clustering_by_category"].items():
        print(f"   {k:18s} {f(v['mean_margin_B_bits'])}   {f(v['qpsi_correct_rate'])}")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
