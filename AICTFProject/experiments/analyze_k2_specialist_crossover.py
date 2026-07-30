#!/usr/bin/env python3
"""K=2 LRO steps 6-7: formal crossover + repertoire-gain gate.

Consumes ``episode_rows.csv`` from run_k2_specialist_cross_eval.py.

Formal gate (1M checkpoints only; earlier checkpoints are trajectory
context and must NOT be selected post hoc):

  1. Payoff(piR, C_RUSH)  > Payoff(piS, C_RUSH)   paired 95% CI clears 0
  2. Payoff(piS, C_SPLIT) > Payoff(piR, C_SPLIT)  paired 95% CI clears 0
  3. LCB95(delta_pool) > 0, where
       V_selective = [max_f payoff(f, C_RUSH) + max_f payoff(f, C_SPLIT)] / 2
       V_fixed     = max_f mean_c payoff(f, c)
       delta_pool  = V_selective - V_fixed

delta_pool uses a HIERARCHICAL clustered bootstrap resampling, per
replicate: (a) training seeds with replacement within each family, and
(b) paired evaluation seeds with replacement within each context.

Also reports every individual (piR seed x piS seed) pairing so a single
strong run cannot mask family-level collapse.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

CONTEXTS = ["C_RUSH", "C_SPLIT"]
FAMILIES = ["piR", "piS"]
OWN_CONTEXT = {"piR": "C_RUSH", "piS": "C_SPLIT"}


def load(path: Path, step: int):
    """-> payoff[family][context][train_seed][episode_seed] = win_margin"""
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["checkpoint_step"]) != step:
                continue
            data[r["family"]][r["context"]][int(r["train_seed"])][int(r["episode_seed"])] = float(r["win_margin"])
    return data


def family_mean_by_epseed(data, family, context):
    """Mean across training seeds, per evaluation seed (keeps pairing)."""
    per_seed = data[family][context]
    ep_seeds = sorted(set.intersection(*[set(v) for v in per_seed.values()]))
    tseeds = sorted(per_seed)
    arr = np.array([[per_seed[ts][es] for es in ep_seeds] for ts in tseeds], dtype=float)
    return ep_seeds, tseeds, arr  # arr shape (n_tseed, n_epseed)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", default="artifacts/k2v2_specialist_cross_eval/episode_rows.csv")
    p.add_argument("--step", type=int, default=1_000_000)
    p.add_argument(
        "--formal-step",
        type=int,
        default=1_000_000,
        help="Only this checkpoint is a formal gate; others are labeled DIAGNOSTIC.",
    )
    p.add_argument("--n-boot", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    rows_path = Path(args.rows)
    data = load(rows_path, args.step)
    for fam in FAMILIES:
        for ctx in CONTEXTS:
            if not data[fam][ctx]:
                print(f"[abort] no rows for {fam}/{ctx} at step {args.step}")
                return 1

    formal = int(args.step) == int(args.formal_step)
    label = "FORMAL GATE" if formal else "DIAGNOSTIC (trajectory; not a formal gate)"

    rng = np.random.default_rng(args.seed)
    lo_q, hi_q = 100 * args.alpha / 2, 100 * (1 - args.alpha / 2)

    print(f"=== K=2 specialist cross-evaluation @ step {args.step:,} [{label}] ===\n")

    # ---- family payoff matrix -------------------------------------------
    cache = {}
    print("Family payoff matrix (mean win_margin, averaged over 3 training seeds)")
    print(f"{'':10s} " + "".join(f"{c:>12s}" for c in CONTEXTS))
    for fam in FAMILIES:
        cells = []
        for ctx in CONTEXTS:
            es, ts, arr = family_mean_by_epseed(data, fam, ctx)
            cache[(fam, ctx)] = (es, ts, arr)
            cells.append(arr.mean())
        print(f"{fam:10s} " + "".join(f"{v:>12.4f}" for v in cells))
    print()

    print("Per training-seed detail (mean win_margin)")
    print(f"{'':16s} " + "".join(f"{c:>12s}" for c in CONTEXTS))
    for fam in FAMILIES:
        for i, ts in enumerate(cache[(fam, CONTEXTS[0])][1]):
            vals = [cache[(fam, c)][2][i].mean() for c in CONTEXTS]
            print(f"  {fam}/s{ts:<9d} " + "".join(f"{v:>12.4f}" for v in vals))
    print()

    # ---- gates 1 & 2: paired crossover CIs ------------------------------
    print("Crossover gates (paired on evaluation seed, family means)")
    gate_pass = {}
    for ctx in CONTEXTS:
        winner = "piR" if ctx == "C_RUSH" else "piS"
        loser = "piS" if winner == "piR" else "piR"
        es_w, _, arr_w = cache[(winner, ctx)]
        es_l, _, arr_l = cache[(loser, ctx)]
        shared = sorted(set(es_w) & set(es_l))
        iw = [es_w.index(e) for e in shared]
        il = [es_l.index(e) for e in shared]
        d = arr_w.mean(axis=0)[iw] - arr_l.mean(axis=0)[il]
        idx = rng.integers(0, len(d), size=(args.n_boot, len(d)))
        boot = d[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [lo_q, hi_q])
        ok = lo > 0
        gate_pass[ctx] = ok
        status = "PASS" if ok else "FAIL"
        if not formal:
            status = f"DIAGNOSTIC/{status}"
        print(f"  {ctx:8s}: {winner} - {loser} = {d.mean():+.4f}  "
              f"CI95=[{lo:+.4f}, {hi:+.4f}]  n={len(d)}  [{status}]")
    print()

    # ---- per-pairing breakdown -----------------------------------------
    print("Every (piR seed x piS seed) pairing -- both directions must hold")
    tsR = cache[("piR", "C_RUSH")][1]
    tsS = cache[("piS", "C_RUSH")][1]
    n_pair_ok = 0
    for i, r_seed in enumerate(tsR):
        for j, s_seed in enumerate(tsS):
            r_on_rush = cache[("piR", "C_RUSH")][2][i].mean()
            s_on_rush = cache[("piS", "C_RUSH")][2][j].mean()
            s_on_split = cache[("piS", "C_SPLIT")][2][j].mean()
            r_on_split = cache[("piR", "C_SPLIT")][2][i].mean()
            d1 = r_on_rush - s_on_rush
            d2 = s_on_split - r_on_split
            ok = d1 > 0 and d2 > 0
            n_pair_ok += int(ok)
            print(f"  piR s{r_seed} x piS s{s_seed}: "
                  f"C_RUSH(R-S)={d1:+.4f}  C_SPLIT(S-R)={d2:+.4f}  [{'ok' if ok else 'XX'}]")
    print(f"  pairings with BOTH directions positive: {n_pair_ok}/{len(tsR)*len(tsS)}\n")

    # ---- gate 3: delta_pool, hierarchical clustered bootstrap ----------
    def delta_pool_from(sel_t: dict, sel_e: dict) -> float:
        pay = {}
        for fam in FAMILIES:
            for ctx in CONTEXTS:
                es, ts, arr = cache[(fam, ctx)]
                ti = sel_t[fam]
                ei = sel_e[ctx]
                pay[(fam, ctx)] = arr[np.ix_(ti, ei)].mean()
        v_sel = np.mean([max(pay[(f, c)] for f in FAMILIES) for c in CONTEXTS])
        v_fix = max(np.mean([pay[(f, c)] for c in CONTEXTS]) for f in FAMILIES)
        return v_sel - v_fix

    n_t = {f: len(cache[(f, CONTEXTS[0])][1]) for f in FAMILIES}
    n_e = {c: len(cache[(FAMILIES[0], c)][0]) for c in CONTEXTS}
    point = delta_pool_from({f: list(range(n_t[f])) for f in FAMILIES},
                            {c: list(range(n_e[c])) for c in CONTEXTS})
    boot = np.empty(args.n_boot)
    for b in range(args.n_boot):
        sel_t = {f: rng.integers(0, n_t[f], n_t[f]).tolist() for f in FAMILIES}
        sel_e = {c: rng.integers(0, n_e[c], n_e[c]).tolist() for c in CONTEXTS}
        boot[b] = delta_pool_from(sel_t, sel_e)
    lo, hi = np.percentile(boot, [lo_q, hi_q])
    gate3 = lo > 0
    print("Repertoire gain (hierarchical clustered bootstrap: training seeds + eval seeds)")
    status3 = "PASS" if gate3 else "FAIL"
    if not formal:
        status3 = f"DIAGNOSTIC/{status3}"
    print(f"  delta_pool = {point:+.4f}   CI95=[{lo:+.4f}, {hi:+.4f}]   LCB={lo:+.4f}  "
          f"[{status3}]")
    if not formal:
        print("  NOTE: delta_pool at this step is DIAGNOSTIC only; formal gate is 1M.")
    print()

    # ---- verdict --------------------------------------------------------
    g1, g2 = gate_pass["C_RUSH"], gate_pass["C_SPLIT"]
    print("=" * 66)
    tag = "" if formal else " (diagnostic)"
    print(f"gate1 piR>piS on C_RUSH : {'PASS' if g1 else 'FAIL'}{tag}")
    print(f"gate2 piS>piR on C_SPLIT: {'PASS' if g2 else 'FAIL'}{tag}")
    print(f"gate3 LCB(delta_pool)>0 : {'PASS' if gate3 else 'FAIL'}{tag}")
    if not formal:
        verdict = (
            f"TRAJECTORY DIAGNOSTIC @ {args.step:,}: "
            f"crossover={'both' if (g1 and g2) else ('rush-only' if g1 else ('split-only' if g2 else 'neither'))}; "
            f"delta_pool_LCB={lo:+.4f}. Does NOT pass or fail the formal K=2 proof."
        )
        print(verdict)
        return 0
    if g1 and g2 and gate3:
        verdict = "K=2 INDEPENDENT-SPECIALIST PROOF: PASS -> proceed to latent birth"
    elif g1 != g2:
        verdict = ("K=2 PROOF FAIL at specialist stage: only ONE crossover direction "
                   "holds -> one specialist generalized or failed to specialize")
    elif not g1 and not g2:
        verdict = ("K=2 PROOF FAIL: neither direction holds -> near-identical "
                   "generalists; contexts produced no learned complementarity")
    else:
        verdict = ("K=2 PROOF FAIL: both crossover directions hold but repertoire "
                   "gain LCB does not clear zero")
    print(verdict)
    return 0 if (g1 and g2 and gate3) else 2


if __name__ == "__main__":
    raise SystemExit(main())
