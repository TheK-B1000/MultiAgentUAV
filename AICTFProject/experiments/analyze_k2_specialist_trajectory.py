#!/usr/bin/env python3
"""K=2 LRO trajectory analysis across training checkpoints.

The 1M formal gate already FAILED. This script does NOT re-litigate it. Every
number here below 1M is a DIAGNOSTIC: the checkpoint set was fixed in advance
(200k / 300k / 500k bracketing the predeclared 250k/500k points), and no
earlier checkpoint may be selected post hoc to claim a pass.

It answers one question:

    Did complementarity ever exist during training, or was piR the better
    generalist from the beginning?

Per checkpoint it reports:
  1. the family payoff matrix
  2. both paired crossover confidence intervals
  3. delta_pool, explicitly labeled diagnostic
  4. per-training-seed results
  5. piR's C_SPLIT transfer curve
  6. piS's C_SPLIT learning / collapse curve

and then locates the crossover onset -- the earliest checkpoint at which piR
begins dominating C_SPLIT.

Interpretation is predeclared (see docstring of the crossover analyzer):

  * a real earlier crossover  -> diagnostic discovery only; justifies a NEW
    preregistered short-budget replication with fresh seeds. It does not
    retroactively pass this experiment.
  * piR dominant by 200k      -> this context pair never induced learned
    complementarity; the scripted payoff crossover was not predictive.
  * one seed only             -> seed-sensitive specialization, not a family.

Dense in-training curves are read from the training episode CSVs, which are
far finer-grained than the four evaluated checkpoints.
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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_k2_specialist_crossover import (  # noqa: E402
    CONTEXTS,
    FAMILIES,
    family_mean_by_epseed,
    load,
)

FORMAL_STEP = 1_000_000
TRAIN_DIRS = {
    "piR": [("s901001", "artifacts/k2v2_piR_train_s901001"),
            ("s901002", "artifacts/k2v2_piR_train_s901002"),
            ("s901003", "artifacts/k2v2_piR_train_s901003")],
    "piS": [("s902001", "artifacts/k2v2_piS_train_s902001"),
            ("s902002", "artifacts/k2v2_piS_train_s902002"),
            ("s902003", "artifacts/k2v2_piS_train_s902003")],
}


def available_steps(rows_path: Path) -> list[int]:
    steps = set()
    with open(rows_path, newline="") as f:
        for r in csv.DictReader(f):
            steps.add(int(r["checkpoint_step"]))
    return sorted(steps)


def paired_ci(d: np.ndarray, rng, n_boot: int, alpha: float) -> tuple[float, float, float]:
    """Bootstrap CI of the mean of paired differences."""
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)


def delta_pool(cache, rng, n_boot: int, alpha: float) -> tuple[float, float, float]:
    """Hierarchical clustered bootstrap over training seeds and eval seeds."""
    fams = list(FAMILIES)

    def compute(sel_t, sel_e):
        pay = {}
        for fam in fams:
            for ctx in CONTEXTS:
                _es, _ts, arr = cache[(fam, ctx)]
                pay[(fam, ctx)] = arr[np.ix_(sel_t[fam], sel_e[ctx])].mean()
        v_sel = np.mean([max(pay[(f, c)] for f in fams) for c in CONTEXTS])
        v_fix = max(np.mean([pay[(f, c)] for c in CONTEXTS]) for f in fams)
        return v_sel - v_fix

    n_t = {f: len(cache[(f, CONTEXTS[0])][1]) for f in fams}
    n_e = {c: len(cache[(fams[0], c)][0]) for c in CONTEXTS}
    point = compute({f: list(range(n_t[f])) for f in fams},
                    {c: list(range(n_e[c])) for c in CONTEXTS})
    boot = np.empty(n_boot)
    for b in range(n_boot):
        boot[b] = compute(
            {f: rng.integers(0, n_t[f], n_t[f]).tolist() for f in fams},
            {c: rng.integers(0, n_e[c], n_e[c]).tolist() for c in CONTEXTS},
        )
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def analyze_step(data, rng, n_boot: int, alpha: float) -> dict:
    cache = {}
    for fam in FAMILIES:
        for ctx in CONTEXTS:
            cache[(fam, ctx)] = family_mean_by_epseed(data, fam, ctx)

    out = {"cache": cache, "payoff": {}, "per_seed": {}}
    for fam in FAMILIES:
        for ctx in CONTEXTS:
            out["payoff"][(fam, ctx)] = float(cache[(fam, ctx)][2].mean())
        tseeds = cache[(fam, CONTEXTS[0])][1]
        for i, ts in enumerate(tseeds):
            out["per_seed"][(fam, ts)] = {
                ctx: float(cache[(fam, ctx)][2][i].mean()) for ctx in CONTEXTS
            }

    out["gates"] = {}
    for ctx in CONTEXTS:
        winner = "piR" if ctx == "C_RUSH" else "piS"
        loser = "piS" if winner == "piR" else "piR"
        es_w, _, arr_w = cache[(winner, ctx)]
        es_l, _, arr_l = cache[(loser, ctx)]
        shared = sorted(set(es_w) & set(es_l))
        iw = [es_w.index(e) for e in shared]
        il = [es_l.index(e) for e in shared]
        d = arr_w.mean(axis=0)[iw] - arr_l.mean(axis=0)[il]
        mean, lo, hi = paired_ci(d, rng, n_boot, alpha)
        out["gates"][ctx] = {"winner": winner, "loser": loser, "mean": mean,
                             "lo": lo, "hi": hi, "pass": lo > 0, "n": len(d)}

    # piR-minus-piS on C_SPLIT: positive means piR is DOMINATING its
    # non-training context, which is the failure signature.
    es_r, _, arr_r = cache[("piR", "C_SPLIT")]
    es_s, _, arr_s = cache[("piS", "C_SPLIT")]
    shared = sorted(set(es_r) & set(es_s))
    d_rs = (arr_r.mean(axis=0)[[es_r.index(e) for e in shared]]
            - arr_s.mean(axis=0)[[es_s.index(e) for e in shared]])
    mean, lo, hi = paired_ci(d_rs, rng, n_boot, alpha)
    out["piR_dominates_split"] = {"mean": mean, "lo": lo, "hi": hi, "significant": lo > 0}

    point, lo, hi = delta_pool(cache, rng, n_boot, alpha)
    out["delta_pool"] = {"point": point, "lo": lo, "hi": hi, "pass": lo > 0}

    n_pair_ok = 0
    pairs = []
    tsR = cache[("piR", "C_RUSH")][1]
    tsS = cache[("piS", "C_RUSH")][1]
    for i, r_seed in enumerate(tsR):
        for j, s_seed in enumerate(tsS):
            d1 = cache[("piR", "C_RUSH")][2][i].mean() - cache[("piS", "C_RUSH")][2][j].mean()
            d2 = cache[("piS", "C_SPLIT")][2][j].mean() - cache[("piR", "C_SPLIT")][2][i].mean()
            ok = bool(d1 > 0 and d2 > 0)
            n_pair_ok += int(ok)
            pairs.append((r_seed, s_seed, float(d1), float(d2), ok))
    out["pairs"] = pairs
    out["n_pair_ok"] = n_pair_ok
    return out


def training_curve(dir_rel: str, n_bins: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Binned mean win_margin vs timesteps from a training episode CSV."""
    path = PROJECT_ROOT / dir_rel / "episodes.csv"
    if not path.exists():
        return None
    ts, wm = [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                ts.append(float(r["timesteps"]))
                wm.append(float(r["win_margin"]))
            except (KeyError, TypeError, ValueError):
                continue
    if not ts:
        return None
    ts_a, wm_a = np.asarray(ts), np.asarray(wm)
    edges = np.linspace(0, max(ts_a.max(), 1.0), n_bins + 1)
    idx = np.clip(np.digitize(ts_a, edges) - 1, 0, n_bins - 1)
    centers, means = [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() >= 5:
            centers.append(0.5 * (edges[b] + edges[b + 1]))
            means.append(wm_a[sel].mean())
    return np.asarray(centers), np.asarray(means)


def sparkline(vals: np.ndarray, lo: float, hi: float) -> str:
    """ASCII sparkline. Deliberately not Unicode blocks: the Windows console
    this project runs on is cp1252 and raises UnicodeEncodeError on them."""
    ramp = "._-=+*#@"
    if hi - lo < 1e-9:
        return ramp[0] * len(vals)
    scaled = np.clip((vals - lo) / (hi - lo), 0, 1)
    return "".join(ramp[int(round(v * (len(ramp) - 1)))] for v in scaled)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", default="artifacts/k2v2_specialist_cross_eval/episode_rows.csv")
    p.add_argument("--steps", type=int, nargs="+", default=None,
                   help="Default: every checkpoint present in the rows file.")
    p.add_argument("--n-boot", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--train-bins", type=int, default=25)
    p.add_argument("--out-csv", default="artifacts/k2v2_specialist_cross_eval/trajectory_table.csv")
    args = p.parse_args()

    rows_path = PROJECT_ROOT / args.rows if not Path(args.rows).is_absolute() else Path(args.rows)
    if not rows_path.exists():
        print(f"[abort] rows file not found: {rows_path}", file=sys.stderr)
        return 1

    steps = args.steps if args.steps else available_steps(rows_path)
    rng = np.random.default_rng(args.seed)

    results = {}
    for step in steps:
        data = load(rows_path, step)
        incomplete = [f"{f}/{c}" for f in FAMILIES for c in CONTEXTS if not data[f][c]]
        if incomplete:
            print(f"[skip] step {step:,}: missing cells {incomplete}")
            continue
        results[step] = analyze_step(data, rng, args.n_boot, args.alpha)

    if not results:
        print("[abort] no complete checkpoints to analyze", file=sys.stderr)
        return 1

    done = sorted(results)
    print("=" * 78)
    print("K=2 SPECIALIST TRAJECTORY ANALYSIS")
    print("The 1,000,000-step gate is the ONLY formal result. Every earlier")
    print("checkpoint below is a DIAGNOSTIC and cannot be selected post hoc.")
    print("=" * 78)

    # ---- per-checkpoint blocks -----------------------------------------
    for step in done:
        r = results[step]
        tag = "FORMAL GATE" if step == FORMAL_STEP else "DIAGNOSTIC (trajectory)"
        print(f"\n{'-' * 78}\n### step {step:,}  [{tag}]\n{'-' * 78}")

        print("\n1. Family payoff matrix (mean win_margin over 3 training seeds)")
        print(f"   {'':8s}" + "".join(f"{c:>12s}" for c in CONTEXTS))
        for fam in FAMILIES:
            print(f"   {fam:8s}" + "".join(f"{r['payoff'][(fam, c)]:>12.4f}" for c in CONTEXTS))

        print("\n2. Crossover gates (paired on evaluation seed)")
        for ctx in CONTEXTS:
            g = r["gates"][ctx]
            print(f"   {ctx:8s} {g['winner']}-{g['loser']} = {g['mean']:+.4f}  "
                  f"CI95=[{g['lo']:+.4f}, {g['hi']:+.4f}]  n={g['n']}  "
                  f"[{'PASS' if g['pass'] else 'FAIL'}]")

        dp = r["delta_pool"]
        lbl = "formal" if step == FORMAL_STEP else "DIAGNOSTIC -- not a formal gate"
        print(f"\n3. delta_pool ({lbl})")
        print(f"   delta_pool = {dp['point']:+.4f}  CI95=[{dp['lo']:+.4f}, {dp['hi']:+.4f}]  "
              f"LCB={dp['lo']:+.4f}  [{'PASS' if dp['pass'] else 'FAIL'}]")

        print("\n4. Per-training-seed detail (mean win_margin)")
        print(f"   {'':14s}" + "".join(f"{c:>12s}" for c in CONTEXTS))
        for fam in FAMILIES:
            for (f2, ts), vals in r["per_seed"].items():
                if f2 == fam:
                    print(f"   {fam}/s{ts:<7d}" + "".join(f"{vals[c]:>12.4f}" for c in CONTEXTS))
        print(f"   pairings with BOTH crossover directions positive: "
              f"{r['n_pair_ok']}/{len(r['pairs'])}")

    # ---- 5. piR C_SPLIT transfer curve ----------------------------------
    print(f"\n{'=' * 78}\n5. piR C_SPLIT TRANSFER CURVE")
    print("   piR was never trained on C_SPLIT. Rising values = transfer.")
    print(f"{'=' * 78}")
    print(f"   {'step':>10s} {'piR@C_SPLIT':>12s} {'piS@C_SPLIT':>12s} "
          f"{'piR-piS':>10s} {'CI95':>22s} {'piR dominates?':>15s}")
    for step in done:
        r = results[step]
        d = r["piR_dominates_split"]
        ci = f"[{d['lo']:+.3f}, {d['hi']:+.3f}]"
        if d["significant"]:
            verdict = "YES (sig)"
        else:
            verdict = "yes" if d["mean"] > 0 else "no"
        print(f"   {step:>10,} {r['payoff'][('piR', 'C_SPLIT')]:>12.4f} "
              f"{r['payoff'][('piS', 'C_SPLIT')]:>12.4f} {d['mean']:>+10.4f} "
              f"{ci:>22s} {verdict:>15s}")

    vals = np.array([results[s]["payoff"][("piR", "C_SPLIT")] for s in done])
    print(f"\n   piR@C_SPLIT {sparkline(vals, vals.min(), vals.max())}  "
          f"({vals.min():.3f} .. {vals.max():.3f})")

    # ---- 6. piS C_SPLIT learning / collapse curve ------------------------
    print(f"\n{'=' * 78}\n6. piS C_SPLIT LEARNING / COLLAPSE CURVE")
    print("   C_SPLIT is piS's OWN training context; falling values = collapse.")
    print(f"{'=' * 78}")
    piS_seeds = [ts for (fam, ts) in results[done[0]]["per_seed"] if fam == "piS"]
    header = "".join(f"{'piS/s' + str(s):>13s}" for s in piS_seeds)
    print(f"   {'step':>10s}{header}{'family':>12s}")
    for step in done:
        r = results[step]
        cells = [r["per_seed"][("piS", s)]["C_SPLIT"] for s in piS_seeds]
        print(f"   {step:>10,}" + "".join(f"{v:>13.4f}" for v in cells)
              + f"{r['payoff'][('piS', 'C_SPLIT')]:>12.4f}")

    vals = np.array([results[s]["payoff"][("piS", "C_SPLIT")] for s in done])
    print(f"\n   piS@C_SPLIT {sparkline(vals, vals.min(), vals.max())}  "
          f"({vals.min():.3f} .. {vals.max():.3f})")

    print("\n   Dense in-training curves (own training opponent, binned win_margin):")
    for fam in ("piR", "piS"):
        for label, d in TRAIN_DIRS[fam]:
            cur = training_curve(d, args.train_bins)
            if cur is None:
                print(f"     {fam}/{label}: [no training episodes.csv]")
                continue
            centers, means = cur
            print(f"     {fam}/{label}: {sparkline(means, means.min(), means.max())}  "
                  f"start={means[0]:+.3f} peak={means.max():+.3f} "
                  f"@{centers[int(means.argmax())]/1000:.0f}k end={means[-1]:+.3f}")

    # ---- crossover onset -------------------------------------------------
    print(f"\n{'=' * 78}\nCROSSOVER ONSET: when did piR begin dominating C_SPLIT?\n{'=' * 78}")
    both = [s for s in done if results[s]["gates"]["C_RUSH"]["pass"]
            and results[s]["gates"]["C_SPLIT"]["pass"]]
    gain = [s for s in both if results[s]["delta_pool"]["pass"]]
    first_dom = next((s for s in done if results[s]["piR_dominates_split"]["mean"] > 0), None)
    first_sig = next((s for s in done if results[s]["piR_dominates_split"]["significant"]), None)

    print(f"   checkpoints with BOTH crossover directions : "
          f"{[f'{s:,}' for s in both] if both else 'NONE'}")
    print(f"   ... and LCB(delta_pool) > 0                : "
          f"{[f'{s:,}' for s in gain] if gain else 'NONE'}")
    print(f"   earliest piR > piS on C_SPLIT (point)      : "
          f"{f'{first_dom:,}' if first_dom else 'never'}")
    print(f"   earliest piR > piS on C_SPLIT (CI clears 0): "
          f"{f'{first_sig:,}' if first_sig else 'never'}")

    print()
    if gain:
        print(f"   => Complementarity DID exist at {[f'{s:,}' for s in gain]}.")
        print("      This is a DIAGNOSTIC DISCOVERY, not a rescued pass. It justifies a")
        print("      NEW preregistered replication: fixed budget chosen in advance, 3 fresh")
        print("      piR seeds, 3 fresh piS seeds, fresh evaluation seeds, same frozen")
        print("      contexts. Only that experiment can establish that a shorter budget")
        print("      reliably preserves complementarity.")
    elif first_dom is not None and first_dom == done[0]:
        print(f"   => piR already dominates C_SPLIT at the earliest checkpoint "
              f"({done[0]:,}).")
        print("      This context pair never induced learned complementarity. The scripted")
        print("      payoff crossover was not predictive of PPO policy specialization.")
        print("      Stop trying to make OP9 the SPLIT niche; promote piR to incumbent G0")
        print("      and select the next context from where the LEARNED incumbent fails.")
    else:
        n_seed_ok = {s: results[s]["n_pair_ok"] for s in done}
        print("   => No checkpoint shows a family-level two-direction crossover with")
        print("      repertoire gain. Per-checkpoint pairings passing both directions:")
        print(f"      {n_seed_ok}")
        print("      Treat as failure of the context pair, not of the evaluation.")

    # ---- machine-readable table -----------------------------------------
    out_csv = PROJECT_ROOT / args.out_csv if not Path(args.out_csv).is_absolute() else Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["checkpoint_step", "role", "piR_C_RUSH", "piR_C_SPLIT", "piS_C_RUSH",
                    "piS_C_SPLIT", "gate1_mean", "gate1_lo", "gate1_hi", "gate1_pass",
                    "gate2_mean", "gate2_lo", "gate2_hi", "gate2_pass",
                    "delta_pool", "delta_pool_lo", "delta_pool_hi", "delta_pool_pass",
                    "piR_minus_piS_on_split", "piR_split_lo", "piR_split_hi",
                    "n_pairings_both_directions"])
        for step in done:
            r = results[step]
            g1, g2, dp = r["gates"]["C_RUSH"], r["gates"]["C_SPLIT"], r["delta_pool"]
            ds = r["piR_dominates_split"]
            w.writerow([
                step, "formal" if step == FORMAL_STEP else "diagnostic",
                f"{r['payoff'][('piR', 'C_RUSH')]:.6f}", f"{r['payoff'][('piR', 'C_SPLIT')]:.6f}",
                f"{r['payoff'][('piS', 'C_RUSH')]:.6f}", f"{r['payoff'][('piS', 'C_SPLIT')]:.6f}",
                f"{g1['mean']:.6f}", f"{g1['lo']:.6f}", f"{g1['hi']:.6f}", int(g1["pass"]),
                f"{g2['mean']:.6f}", f"{g2['lo']:.6f}", f"{g2['hi']:.6f}", int(g2["pass"]),
                f"{dp['point']:.6f}", f"{dp['lo']:.6f}", f"{dp['hi']:.6f}", int(dp["pass"]),
                f"{ds['mean']:.6f}", f"{ds['lo']:.6f}", f"{ds['hi']:.6f}", r["n_pair_ok"],
            ])
    print(f"\n[done] trajectory table -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
