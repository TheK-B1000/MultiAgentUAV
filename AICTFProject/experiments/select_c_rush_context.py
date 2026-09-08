#!/usr/bin/env python3
"""Apply the PREDECLARED C_RUSH context-selection rule mechanically.

Locked selection rule (declared before seeing confirmation results,
2026-07-29) -- candidates are compared in this order:

  1. RUSH uniquely best at 16 seeds.
  2. All paired RUSH-vs-other CIs clear zero.
  3. Non-degenerate (NOT all styles' mean margins same-signed).
  4. Pooled best-other LCB clears zero.
  5. Larger LCB for (RUSH - runner-up).
  6. Lower saturation and tie rate as tie-breaker.

Criteria 1-4 are hard gates; 5 then 6 rank the survivors. Reports every
number so the choice is auditable rather than eyeballed.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RUSH = "BLUE_RUSH"


def analyze(artifact_dir: Path, *, n_boot: int, seed: int, alpha: float) -> dict:
    rows = list(csv.DictReader(open(artifact_dir / "episode_results.csv", newline="")))
    by_style: dict[str, dict[int, float]] = defaultdict(dict)
    ctx = None
    for r in rows:
        ctx = (r["red_style"], r["map"])
        by_style[r["blue_style"]][int(r["episode_seed"])] = float(r["win_margin"])

    styles = sorted(by_style)
    means = {s: float(np.mean(list(by_style[s].values()))) for s in styles}
    best = max(means, key=lambda s: means[s])
    ordered = sorted(styles, key=lambda s: -means[s])
    runner_up = ordered[1]

    rng = np.random.default_rng(seed)
    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)

    def paired_ci(a: str, b: str):
        shared = sorted(set(by_style[a]) & set(by_style[b]))
        d = np.array([by_style[a][k] - by_style[b][k] for k in shared], dtype=float)
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        boot = d[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [lo_q, hi_q])
        return float(d.mean()), float(lo), float(hi), len(d)

    pairwise = {s: paired_ci(RUSH, s) for s in styles if s != RUSH}

    # Pooled vs others: per-episode mean of RUSH-minus-each-other difference.
    others = [s for s in styles if s != RUSH]
    shared = sorted(set.intersection(*[set(by_style[s]) for s in styles]))
    pooled_d = np.array(
        [np.mean([by_style[RUSH][k] - by_style[o][k] for o in others]) for k in shared],
        dtype=float,
    )
    idx = rng.integers(0, len(pooled_d), size=(n_boot, len(pooled_d)))
    pooled_boot = pooled_d[idx].mean(axis=1)
    pooled_lo, pooled_hi = np.percentile(pooled_boot, [lo_q, hi_q])

    all_means = np.array([means[s] for s in styles])
    degenerate = bool(np.all(all_means > 0) or np.all(all_means < 0))
    n_eps = len(next(iter(by_style.values())))
    saturation = float(np.mean([v > 0 for v in by_style[best].values()]))
    tie_rate = float(np.mean([v == 0 for s in styles for v in by_style[s].values()]))

    return {
        "ctx": f"{ctx[0]}|{ctx[1]}",
        "means": means, "best": best, "runner_up": runner_up,
        "pairwise": pairwise,
        "pooled": (float(pooled_d.mean()), float(pooled_lo), float(pooled_hi)),
        "degenerate": degenerate, "saturation": saturation, "tie_rate": tie_rate,
        "n_eps": n_eps,
        "g1_rush_uniquely_best": best == RUSH and means[RUSH] > means[runner_up],
        "g2_all_cis_clear": all(lo > 0 for (_, lo, _, _) in pairwise.values()),
        "g3_non_degenerate": not degenerate,
        "g4_pooled_lcb_clears": pooled_lo > 0,
        "runner_up_lcb": pairwise[runner_up][1] if runner_up in pairwise else float("-inf"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact_dirs", nargs="+")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    results = []
    for d in args.artifact_dirs:
        path = Path(d)
        if not (path / "episode_results.csv").exists():
            print(f"[skip] {d}: no episode_results.csv", file=sys.stderr)
            continue
        results.append(analyze(path, n_boot=args.n_boot, seed=args.seed, alpha=args.alpha))

    for r in results:
        print(f"=== {r['ctx']}  (n={r['n_eps']}) ===")
        for s in sorted(r["means"], key=lambda s: -r["means"][s]):
            print(f"    {s:12s} mean_margin={r['means'][s]:+.4f}")
        print(f"    best={r['best']}  runner_up={r['runner_up']}")
        for s, (m, lo, hi, n) in sorted(r["pairwise"].items(), key=lambda kv: kv[1][0]):
            print(f"    RUSH - {s:12s} mean={m:+.4f} CI95=[{lo:+.4f},{hi:+.4f}]")
        pm, plo, phi = r["pooled"]
        print(f"    RUSH - pooled_others mean={pm:+.4f} CI95=[{plo:+.4f},{phi:+.4f}]")
        print(f"    degenerate={r['degenerate']}  saturation={r['saturation']:.3f}  tie_rate={r['tie_rate']:.3f}")
        print(f"    GATES: g1_uniquely_best={r['g1_rush_uniquely_best']} "
              f"g2_all_cis_clear={r['g2_all_cis_clear']} "
              f"g3_non_degenerate={r['g3_non_degenerate']} "
              f"g4_pooled_lcb={r['g4_pooled_lcb_clears']}")
        passes = all([r["g1_rush_uniquely_best"], r["g2_all_cis_clear"],
                       r["g3_non_degenerate"], r["g4_pooled_lcb_clears"]])
        print(f"    -> HARD GATES {'PASS' if passes else 'FAIL'}")
        print()

    survivors = [r for r in results if all([
        r["g1_rush_uniquely_best"], r["g2_all_cis_clear"],
        r["g3_non_degenerate"], r["g4_pooled_lcb_clears"]])]
    print("=" * 62)
    if not survivors:
        print("NO CANDIDATE PASSES ALL HARD GATES (criteria 1-4).")
        return 2
    # Criterion 5: larger runner-up LCB; criterion 6: lower saturation, then tie rate.
    survivors.sort(key=lambda r: (-r["runner_up_lcb"], r["saturation"], r["tie_rate"]))
    win = survivors[0]
    print(f"SELECTED C_RUSH = {win['ctx']}")
    print(f"  criterion 5 (RUSH-runner_up LCB) = {win['runner_up_lcb']:+.4f}")
    print(f"  criterion 6 (saturation, tie_rate) = {win['saturation']:.3f}, {win['tie_rate']:.3f}")
    if len(survivors) > 1:
        print("  ranked survivors:")
        for r in survivors:
            print(f"    {r['ctx']:52s} lcb={r['runner_up_lcb']:+.4f} "
                  f"sat={r['saturation']:.3f} tie={r['tie_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
