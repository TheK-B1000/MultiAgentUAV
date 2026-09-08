#!/usr/bin/env python3
"""Signed assigned-repertoire gain (Delta_assigned) + replication power study.

WHY A NEW STATISTIC
-------------------
The original repertoire statistic is non-negative by construction:

    V_selective = mean_c  max_f pay(f, c)
    V_fixed     = max_f   mean_c pay(f, c)
    Delta_pool  = V_selective - V_fixed  >= 0   always

(For any family f, mean_c pay(f,c) <= mean_c max_f' pay(f',c) = V_selective, so
the max over f is also <= V_selective.) A percentile bootstrap on a statistic
pinned at a boundary null piles probability mass exactly at 0, so "LCB95 > 0"
demands >97.5% of replicates be strictly positive. It also implicitly selects
which policy handles which context AFTER seeing outcomes.

Because the specialist assignment is predeclared -- piR handles C_RUSH, piS
handles C_SPLIT -- the confirmatory statistic can instead be signed:

    V_assigned      = (R_R + S_S) / 2
    V_fixed         = max( (R_R + R_S)/2 , (S_R + S_S)/2 )
    Delta_assigned  = V_assigned - V_fixed
                    = 0.5 * min( R_R - S_R , S_S - R_S )

using pay(policy, context) with R_R = piR on C_RUSH, R_S = piR on C_SPLIT,
S_R = piS on C_RUSH, S_S = piS on C_SPLIT.

Delta_assigned CAN be negative -- it goes negative as soon as either predeclared
assignment fails. This is not a relaxation: it is the JOINT form of gates 1 and
2. Because it equals half the MINIMUM of the two crossover margins, requiring
LCB95(Delta_assigned) > 0 demands both directions hold SIMULTANEOUSLY in >=97.5%
of bootstrap replicates, which is strictly stronger than each marginal CI
clearing zero on its own.

Both statistics are reported: Delta_pool for continuity with the failed 1M gate,
Delta_assigned as the confirmatory gate for the replication.

MODES
-----
  --mode observed   Delta_assigned on existing rows. POST-HOC DIAGNOSTIC ONLY;
                    it can never convert the failed experiment into a pass.
  --mode power      Simulate the replication design (n fresh training seeds per
                    family, n fresh paired eval seeds per context) by resampling
                    the observed data, and estimate the probability that all
                    three payoff gates clear.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CONTEXTS = ["C_RUSH", "C_SPLIT"]
FAMILIES = ["piR", "piS"]
# Predeclared assignment: family -> the context it is the designated specialist for.
ASSIGNMENT = {"piR": "C_RUSH", "piS": "C_SPLIT"}


def load_arrays(rows_path: Path, step: int):
    """-> {family: (train_seeds, eval_seeds_per_ctx, arr)}, arr (n_tseed, n_ctx, n_eval).

    Each context uses its OWN evaluation seed block (C_RUSH 1_010_001+,
    C_SPLIT 1_020_001+), so seeds are intersected WITHIN a context, never
    across them. Pairing that matters is across FAMILIES within a context:
    both families see the same seed block there, so resampling eval index
    positions per context and applying them to both families preserves it.
    """
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    with open(rows_path, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["checkpoint_step"]) != step:
                continue
            raw[r["family"]][r["context"]][int(r["train_seed"])][int(r["episode_seed"])] = \
                float(r["win_margin"])

    for fam in FAMILIES:
        if not raw[fam]:
            return None
        for c in CONTEXTS:
            if not raw[fam][c]:
                return None

    # Eval seeds per context, shared across families.
    eseeds_by_ctx = {}
    for c in CONTEXTS:
        sets = [set(raw[fam][c][t]) for fam in FAMILIES for t in sorted(raw[fam][c])]
        eseeds_by_ctx[c] = sorted(set.intersection(*sets))
    counts = {c: len(v) for c, v in eseeds_by_ctx.items()}
    if len(set(counts.values())) != 1 or 0 in counts.values():
        print(f"[abort] unequal/empty eval-seed counts per context: {counts}",
              file=sys.stderr)
        return None

    out = {}
    for fam in FAMILIES:
        tseeds = sorted(raw[fam][CONTEXTS[0]])
        arr = np.array([[[raw[fam][c][t][e] for e in eseeds_by_ctx[c]]
                         for c in CONTEXTS] for t in tseeds], dtype=float)
        out[fam] = (tseeds, eseeds_by_ctx, arr)
    return out


def payoffs(data, sel_t: dict, sel_e: np.ndarray) -> dict:
    """pay[(family, context_index)] under a given resample."""
    pay = {}
    for fam in FAMILIES:
        arr = data[fam][2]
        sub = arr[np.ix_(sel_t[fam], range(arr.shape[1]), sel_e)]
        pay[fam] = sub.mean(axis=(0, 2))  # (n_ctx,)
    return pay


def statistics_from(pay: dict) -> dict:
    ri, si = CONTEXTS.index("C_RUSH"), CONTEXTS.index("C_SPLIT")
    R_R, R_S = pay["piR"][ri], pay["piR"][si]
    S_R, S_S = pay["piS"][ri], pay["piS"][si]

    gate1 = R_R - S_R          # piR beats piS on C_RUSH
    gate2 = S_S - R_S          # piS beats piR on C_SPLIT
    d_assigned = 0.5 * min(gate1, gate2)

    v_sel = 0.5 * (max(R_R, S_R) + max(R_S, S_S))
    v_fix = max(0.5 * (R_R + R_S), 0.5 * (S_R + S_S))
    d_pool = v_sel - v_fix

    return {"gate1": gate1, "gate2": gate2,
            "delta_assigned": d_assigned, "delta_pool": d_pool,
            "R_R": R_R, "R_S": R_S, "S_R": S_R, "S_S": S_S}


def hierarchical_bootstrap(data, n_boot: int, rng, alpha: float,
                           n_t_override: int | None = None,
                           n_e_override: int | None = None) -> dict:
    """Resample training seeds within family AND eval seeds within context."""
    n_t = {f: len(data[f][0]) for f in FAMILIES}
    n_e = data[FAMILIES[0]][2].shape[2]
    draw_t = {f: (n_t_override or n_t[f]) for f in FAMILIES}
    draw_e = n_e_override or n_e

    point = statistics_from(payoffs(
        data, {f: list(range(n_t[f])) for f in FAMILIES}, np.arange(n_e)))

    keys = ["gate1", "gate2", "delta_assigned", "delta_pool"]
    boots = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        sel_t = {f: rng.integers(0, n_t[f], draw_t[f]) for f in FAMILIES}
        sel_e = rng.integers(0, n_e, draw_e)   # shared -> pairing preserved
        st = statistics_from(payoffs(data, sel_t, sel_e))
        for k in keys:
            boots[k][b] = st[k]

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = {}
    for k in keys:
        lo, hi = np.percentile(boots[k], [lo_q, hi_q])
        out[k] = {"point": point[k], "lo": float(lo), "hi": float(hi),
                  "pass": bool(lo > 0)}
    out["_payoffs"] = {k: point[k] for k in ("R_R", "R_S", "S_R", "S_S")}
    return out


def run_observed(data, args, rng) -> dict:
    print("=" * 78)
    print(f"Delta_assigned on EXISTING rows @ step {args.step:,}")
    print("POST-HOC DIAGNOSTIC ONLY -- cannot convert the failed experiment")
    print("into a pass, and is not the preregistered replication result.")
    print("=" * 78)

    res = hierarchical_bootstrap(data, args.n_boot, rng, args.alpha)
    p = res["_payoffs"]
    print("\nFamily payoff matrix")
    print(f"  {'':8s}{'C_RUSH':>12s}{'C_SPLIT':>12s}")
    print(f"  {'piR':8s}{p['R_R']:>12.4f}{p['R_S']:>12.4f}")
    print(f"  {'piS':8s}{p['S_R']:>12.4f}{p['S_S']:>12.4f}")

    print("\nStatistics (hierarchical bootstrap: training seeds + paired eval seeds)")
    labels = {
        "gate1": "gate1  R_R - S_R   (piR > piS on C_RUSH)",
        "gate2": "gate2  S_S - R_S   (piS > piR on C_SPLIT)",
        "delta_assigned": "Delta_assigned = 0.5*min(gate1, gate2)  [SIGNED]",
        "delta_pool": "Delta_pool     (legacy, >= 0 by construction)",
    }
    for k, lab in labels.items():
        r = res[k]
        print(f"  {lab:<48s} {r['point']:+.4f}  "
              f"CI95=[{r['lo']:+.4f}, {r['hi']:+.4f}]  "
              f"[{'PASS' if r['pass'] else 'FAIL'}]")

    print("\n  Note: Delta_assigned is half the MINIMUM of the two crossover")
    print("  margins, so its LCB > 0 requires both directions to hold jointly in")
    print("  >=97.5% of replicates -- strictly stronger than the two marginal CIs.")
    return res


def run_power(data, args, rng) -> None:
    print("\n" + "=" * 78)
    print("REPLICATION POWER STUDY")
    print(f"design: {args.sim_seeds} fresh training seeds per family, "
          f"{args.sim_evals} fresh paired eval seeds per context")
    print("=" * 78)

    n_t = {f: len(data[f][0]) for f in FAMILIES}
    n_e = data[FAMILIES[0]][2].shape[2]
    print(f"observed basis: {n_t} training seeds/family, {n_e} eval seeds/context")

    # Between-training-seed spread on each family's ASSIGNED context, which is
    # what the gates actually depend on.
    print("\nper-training-seed means on the assigned context")
    seed_sd = {}
    for fam in FAMILIES:
        ci = CONTEXTS.index(ASSIGNMENT[fam])
        per_seed = data[fam][2][:, ci, :].mean(axis=1)
        seed_sd[fam] = float(per_seed.std(ddof=1)) if len(per_seed) > 1 else float("nan")
        vals = ", ".join(f"{v:+.3f}" for v in per_seed)
        print(f"  {fam} on {ASSIGNMENT[fam]:8s}: [{vals}]  sd={seed_sd[fam]:.4f} (df={len(per_seed)-1})")

    print("\nNONPARAMETRIC simulation (resample training seeds from the observed set)")
    print("  CAVEAT: with only 3 observed seeds per family this cannot generate a")
    print("  seed worse than the worst observed, so it OVERSTATES power.")
    passes = {"gate1": 0, "gate2": 0, "delta_assigned": 0, "all3": 0}
    for _ in range(args.n_sim):
        sim_t = {f: rng.integers(0, n_t[f], args.sim_seeds) for f in FAMILIES}
        sim_e = rng.integers(0, n_e, args.sim_evals)
        sim = {f: (list(range(args.sim_seeds)), list(range(args.sim_evals)),
                   data[f][2][np.ix_(sim_t[f], range(2), sim_e)]) for f in FAMILIES}
        res = hierarchical_bootstrap(sim, args.sim_boot, rng, args.alpha)
        g1, g2 = res["gate1"]["pass"], res["gate2"]["pass"]
        da = res["delta_assigned"]["pass"]
        passes["gate1"] += int(g1)
        passes["gate2"] += int(g2)
        passes["delta_assigned"] += int(da)
        passes["all3"] += int(g1 and g2 and da)
    for k, v in passes.items():
        print(f"    {k:16s} {v / args.n_sim:6.1%}")

    print("\nPARAMETRIC simulation (fresh seeds ~ Normal(mean, observed seed sd))")
    print("  CAVEAT: seed sd is estimated from 3 values (df=2); it is itself very")
    print("  uncertain, so this estimate is wide. The two simulations bracket the")
    print("  plausible range rather than pinning it down.")
    eval_resid = {}
    seed_mean = {}
    for fam in FAMILIES:
        arr = data[fam][2]
        seed_mean[fam] = arr.mean(axis=(0, 2))
        eval_resid[fam] = arr.mean(axis=0) - seed_mean[fam][:, None]

    p_all = 0
    for _ in range(args.n_sim):
        pay = {}
        for fam in FAMILIES:
            ci_assigned = CONTEXTS.index(ASSIGNMENT[fam])
            sd = seed_sd[fam] if np.isfinite(seed_sd[fam]) else 0.0
            offs = rng.normal(0.0, sd, args.sim_seeds).mean()
            pick = rng.integers(0, eval_resid[fam].shape[1], args.sim_evals)
            noise = eval_resid[fam][:, pick].mean(axis=1)
            vec = seed_mean[fam] + noise
            vec[ci_assigned] += offs
            pay[fam] = vec
        st = statistics_from(pay)
        p_all += int(st["gate1"] > 0 and st["gate2"] > 0 and st["delta_assigned"] > 0)
    print(f"    point-estimate-positive rate (no CI): {p_all / args.n_sim:6.1%}")
    print("    (an upper bound on power: requiring CIs to clear zero is stricter)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", default="artifacts/k2v2_specialist_cross_eval/episode_rows.csv")
    p.add_argument("--step", type=int, default=300_000)
    p.add_argument("--mode", choices=["observed", "power", "both"], default="both")
    p.add_argument("--n-boot", type=int, default=20000)
    p.add_argument("--n-sim", type=int, default=300)
    p.add_argument("--sim-boot", type=int, default=800)
    p.add_argument("--sim-seeds", type=int, default=6)
    p.add_argument("--sim-evals", type=int, default=64)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows_path = Path(args.rows)
    if not rows_path.is_absolute():
        rows_path = PROJECT_ROOT / rows_path
    data = load_arrays(rows_path, args.step)
    if data is None:
        print(f"[abort] no complete rows at step {args.step:,}", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    if args.mode in ("observed", "both"):
        run_observed(data, args, rng)
    if args.mode in ("power", "both"):
        run_power(data, args, rng)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
