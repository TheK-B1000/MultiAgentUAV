#!/usr/bin/env python3
"""Paired bootstrap CIs for a scripted-style payoff-matrix artifact.

Reads an ``episode_results.csv`` produced by
run_scripted_style_payoff_matrix.py and reports, for each (red, map)
context, the paired per-episode margin difference between the best blue
style and every alternative, with a bootstrap CI.

Pairing is on ``episode_seed`` -- the matrix tool's matched-seed contract
guarantees ``episode_seed = f(red, map, episode_index)`` independent of
blue style, so the same seed is directly comparable across styles.

Used for the K=2 LRO proof's step-1/step-2 context confirmations
(each requires "best style uniquely best, all paired CIs clear zero").
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact_dir", help="Directory containing episode_results.csv")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    csv_path = Path(args.artifact_dir) / "episode_results.csv"
    if not csv_path.exists():
        print(f"missing {csv_path}", file=sys.stderr)
        return 1
    rows = load_rows(csv_path)

    # margins[(red, map)][style][seed] = win_margin
    margins: dict[tuple[str, str], dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        ctx = (r["red_style"], r["map"])
        margins[ctx][r["blue_style"]][int(r["episode_seed"])] = float(r["win_margin"])

    rng = np.random.default_rng(args.seed)
    lo_q, hi_q = 100 * (args.alpha / 2), 100 * (1 - args.alpha / 2)
    overall_pass = True

    for ctx in sorted(margins):
        red, mp = ctx
        by_style = margins[ctx]
        styles = sorted(by_style)
        means = {s: float(np.mean(list(by_style[s].values()))) for s in styles}
        best = max(means, key=lambda s: means[s])
        print(f"=== {red} | {mp} ===")
        for s in sorted(styles, key=lambda s: -means[s]):
            n = len(by_style[s])
            wins = sum(1 for v in by_style[s].values() if v > 0)
            print(f"  {s:12s} mean_margin={means[s]:+.4f}  WR={wins}/{n}")
        print(f"  best = {best}")

        ctx_pass = True
        for s in styles:
            if s == best:
                continue
            shared = sorted(set(by_style[best]) & set(by_style[s]))
            if not shared:
                print(f"  [WARN] no shared seeds for {best} vs {s}")
                ctx_pass = False
                continue
            d = np.array([by_style[best][k] - by_style[s][k] for k in shared], dtype=float)
            idx = rng.integers(0, len(d), size=(args.n_boot, len(d)))
            boot = d[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [lo_q, hi_q])
            clears = lo > 0
            ctx_pass = ctx_pass and clears
            flag = "PASS" if clears else "FAIL"
            print(f"  {best} - {s:12s} mean={d.mean():+.4f}  CI95=[{lo:+.4f}, {hi:+.4f}]  n={len(d)}  [{flag}]")
        verdict = "ALL PAIRED CIs CLEAR ZERO" if ctx_pass else "NOT ALL CIs CLEAR ZERO"
        print(f"  -> {verdict}")
        print()
        overall_pass = overall_pass and ctx_pass

    print("=" * 60)
    print("OVERALL:", "PASS" if overall_pass else "FAIL")
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
