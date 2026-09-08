"""V3_STRATEGIC_DEMAND confirmation under OBSERVABILITY_V2, both poles, n=192.

This is THE confirmation. Four prospective tests, all required:

    A payoff        delta_G(A)  >= 0.15  AND  LCB95 > 0
    A concealment   LCB95(p_C^A) > 0.50
    B payoff        delta_B(OP7) >= 0.15 AND  LCB95 > 0
    B concealment   LCB95(p_C^B) > 0.50

Conjunction licenses V3_STRATEGIC_DEMAND = VALIDATED. If one component fails,
the whole claim fails exactly as preregistered.

Why BOTH poles run prospectively
--------------------------------
OBSERVABILITY_V2 was defined AFTER the OP7 episodes were observed, so the
existing reading of OP7 (p_C 0.875) is retrospective with respect to the V2
definition. The validation must not rest on one pole's concealment being
prospective and the other's post-hoc. The earlier OP7 confirmation is not
discarded -- it becomes prior independent replication on a separate block.

There is also a substantive reason. The claim is uncertainty between conflicting
strategic contexts. If A conceals but B announces itself early, a capable policy
could infer A by elimination -- clear B signal means BREACH, no B signal means
A, so GUARD. "Planning is required" needs BOTH sides of the fork to resist
trivial early identification.

Nothing scientific is redefined here. run_episode (treatment semantics),
paired_ci (payoff bootstrap) and observability_v2.assay (concealment) are all
imported unchanged.

Run:  python experiments/sds_confirmation_v2.py --device cuda
      python experiments/sds_confirmation_v2.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.m1_payoff_assay import paired_ci                    # noqa: E402
from experiments.observability_v2 import assay                       # noqa: E402
from experiments.sds_genome import SDSGenome, canonical_parent       # noqa: E402
from experiments.strategic_demand_searcher import (                  # noqa: E402
    BREACH, GUARD, MAP, MAX_STEPS, run_episode,
)

SD = ROOT / "artifacts/strategic_demand"
CANDIDATE = SD / "CANDIDATE_A2_SDS2_INIT_3_FROZEN.json"
AMENDMENT = SD / "CONFIRMATION_N_AMENDMENT_192.json"
OUT_DIR = SD / "confirmation_v2"

PAYOFF_FLOOR = 0.15          # frozen, both poles
P_C_FLOOR = 0.50             # frozen, both poles
BOOTSTRAP_RNG = 7
DISQUALIFIED = {2500001, 2600001}
SPENT = {5000001}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_frozen():
    if not AMENDMENT.is_file():
        raise SystemExit("REFUSING TO RUN: n=192 amendment record missing")
    amd = json.loads(AMENDMENT.read_text(encoding="utf-8"))["amendment"]
    n = int(amd["to"])
    seed_base = int(str(amd["block"]).split("..")[0])
    cand = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    a_genome = SDSGenome.from_dict(cand["candidate_genome"])
    return a_genome, canonical_parent("OP7"), seed_base, n


def guard_rails(seed_base: int, n: int) -> None:
    lo, hi = seed_base, seed_base + n - 1
    for bad in DISQUALIFIED | SPENT:
        if lo <= bad <= hi:
            raise SystemExit(f"REFUSING TO RUN: block {lo}..{hi} touches "
                             f"disqualified/spent seed {bad}")
    existing = OUT_DIR / "summary.json"
    if existing.is_file():
        raise SystemExit(f"REFUSING TO RUN: {existing} exists. The block is spent "
                         "once; no extension, no re-run.")


def pole(genome: SDSGenome, *, label: str, direction: str, seed_base: int,
         n: int, device: str) -> dict:
    """One pole: payoff (paired bootstrap) + concealment (Observability V2).

    Concealment is measured on the BREACH episodes: commitment is a BREACH
    behaviour, so 'did BLUE have to commit before intent was readable' is only
    meaningful in the arm where BLUE actually commits.
    """
    rows, d, breach_eps = [], [], []
    for i in range(n):
        seed = seed_base + i
        g = run_episode(style=GUARD, genome=genome, seed=seed, device=device)
        b = run_episode(style=BREACH, genome=genome, seed=seed, device=device)
        for style, ep in ((GUARD, g), (BREACH, b)):
            rows.append({"pole": label, "genome_id": genome.genome_id,
                         "episode_seed": seed, "blue_style": style,
                         "blue_score": ep["blue_score"], "red_score": ep["red_score"],
                         "win": ep["win"], "draw": ep["draw"], "steps": ep["steps"],
                         "t_intent": ep["t_intent"], "t_commit": ep["t_commit"],
                         "zero_zero": ep["zero_zero"],
                         "total_score": ep["total_score"]})
        d.append((g["win"] - b["win"]) if direction == "guard_minus_breach"
                 else (b["win"] - g["win"]))
        breach_eps.append({"t_intent": b["t_intent"], "t_commit": b["t_commit"]})
        if (i + 1) % 8 == 0:
            print(f"  [{label}] {i+1}/{n} paired  running delta="
                  f"{np.mean(d):+.3f}", flush=True)

    d = np.asarray(d, dtype=float)
    mean, lo, hi = paired_ci(d, np.random.default_rng(BOOTSTRAP_RNG))
    obs = assay(breach_eps, horizon=MAX_STEPS, rng_seed=BOOTSTRAP_RNG,
                floor=P_C_FLOOR)

    payoff_pass = bool(mean >= PAYOFF_FLOOR and lo > 0.0)
    conceal_pass = bool(obs["lcb95"] > P_C_FLOOR)
    return {
        "pole": label, "genome_id": genome.genome_id, "direction": direction,
        "n_paired": n, "seed_range": f"{seed_base}..{seed_base + n - 1}",
        "guard_wr": float(np.mean([r["win"] for r in rows
                                   if r["blue_style"] == GUARD])),
        "breach_wr": float(np.mean([r["win"] for r in rows
                                    if r["blue_style"] == BREACH])),
        "payoff": {"delta": mean, "lcb95": lo, "ucb95": hi,
                   "floor": PAYOFF_FLOOR,
                   "clears_floor": bool(mean >= PAYOFF_FLOOR),
                   "lcb_positive": bool(lo > 0.0), "passes": payoff_pass},
        "concealment": {"p_C": obs["p_C"], "lcb95": obs["lcb95"],
                        "ucb95": obs["ucb95"], "counts": obs["counts"],
                        "floor": P_C_FLOOR, "passes": conceal_pass,
                        "telemetry": obs["telemetry"]},
        "frac_0_0": float(np.mean([r["zero_zero"] for r in rows])),
        "mean_total_score": float(np.mean([r["total_score"] for r in rows])),
        "_rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    A_genome, B_genome, seed_base, n = load_frozen()
    guard_rails(seed_base, n)

    print(f"V3_STRATEGIC_DEMAND CONFIRMATION (Observability V2)  {_now()}")
    print(f"  A pole   {A_genome.genome_id}  base={A_genome.base_opponent} "
          f"overlay={A_genome.overlay}")
    print(f"  B pole   {B_genome.genome_id}  base={B_genome.base_opponent}")
    print(f"  seeds    {seed_base}..{seed_base + n - 1}  ({n} paired)")
    print(f"  gates    payoff >= {PAYOFF_FLOOR} with LCB95>0; "
          f"concealment LCB95(p_C) > {P_C_FLOOR}; all four required")
    print(f"  episodes {n * 2 * 2} total, NO EXTENSION")
    if a.dry_run:
        print("\nDRY RUN -- wiring validated, no episode run, block untouched.")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n--- A pole (GUARD should pay, and A should conceal) ---", flush=True)
    A = pole(A_genome, label="A_SDS2_INIT_3", direction="guard_minus_breach",
             seed_base=seed_base, n=n, device=a.device)
    print("\n--- B pole (BREACH should pay, and B should conceal) ---", flush=True)
    B = pole(B_genome, label="B_OP7", direction="breach_minus_guard",
             seed_base=seed_base, n=n, device=a.device)

    rows = A.pop("_rows") + B.pop("_rows")
    four = {
        "A_payoff": A["payoff"]["passes"],
        "A_concealment": A["concealment"]["passes"],
        "B_payoff": B["payoff"]["passes"],
        "B_concealment": B["concealment"]["passes"],
    }
    validated = all(four.values())
    summary = {
        "record": "V3_STRATEGIC_DEMAND confirmation, Observability V2, both poles",
        "utc": _now(),
        "protocol": ["artifacts/strategic_demand/CANDIDATE_A2_SDS2_INIT_3_FROZEN.json",
                     "artifacts/strategic_demand/OBSERVABILITY_V2_FROZEN.json",
                     "artifacts/strategic_demand/CONFIRMATION_N_AMENDMENT_192.json"],
        "seed_range": f"{seed_base}..{seed_base + n - 1}", "n_paired": n,
        "total_episodes": len(rows),
        "no_extension": "192 from the start; if it fails at 192, it fails",
        "A_pole": A, "B_pole": B,
        "four_prospective_tests": four,
        "verdict": ("V3_STRATEGIC_DEMAND_VALIDATED" if validated
                    else "V3_STRATEGIC_DEMAND_NOT_VALIDATED"),
        "thresholds_not_moved": "0.15 payoff floor, LCB95>0, p_C floor 0.50 all "
                                "unchanged from their freezes",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    with open(OUT_DIR / "episode_rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 70)
    for lbl, P in (("A " + A["genome_id"], A), ("B " + B["genome_id"], B)):
        print(f"{lbl:<24} payoff {P['payoff']['delta']:+.4f} "
              f"LCB {P['payoff']['lcb95']:+.4f} "
              f"{'PASS' if P['payoff']['passes'] else 'FAIL'}   "
              f"p_C {P['concealment']['p_C']:.4f} "
              f"LCB {P['concealment']['lcb95']:.4f} "
              f"{'PASS' if P['concealment']['passes'] else 'FAIL'}")
    print("=" * 70)
    print(f"VERDICT: {summary['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
