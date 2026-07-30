#!/usr/bin/env python3
"""Print the πR / πS C_SPLIT transfer curves across checkpoint steps.

Consumes episode_rows.csv from the K=2 cross-eval harness. Diagnostic only.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", default="artifacts/k2v2_specialist_cross_eval/episode_rows.csv")
    p.add_argument("--steps", type=int, nargs="+", default=[200_000, 300_000, 500_000, 1_000_000])
    args = p.parse_args()

    # data[step][family][context][train_seed] -> list of win_margins
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    with open(args.rows, newline="") as f:
        for r in csv.DictReader(f):
            step = int(r["checkpoint_step"])
            if step not in args.steps:
                continue
            data[step][r["family"]][r["context"]][int(r["train_seed"])].append(float(r["win_margin"]))

    print("C_SPLIT transfer / learning curves (mean win_margin)")
    print(f"{'step':>10s}  {'πR_fam':>8s}  {'πS_fam':>8s}  "
          f"{'πR901001':>9s} {'πR901002':>9s} {'πR901003':>9s}  "
          f"{'πS902001':>9s} {'πS902002':>9s} {'πS902003':>9s}")
    for step in args.steps:
        if step not in data:
            print(f"{step:>10,d}  (no rows)")
            continue
        def fam_mean(fam):
            seeds = data[step][fam]["C_SPLIT"]
            if not seeds:
                return float("nan")
            return float(np.mean([np.mean(v) for v in seeds.values()]))
        def seed_mean(fam, seed):
            vals = data[step][fam]["C_SPLIT"].get(seed)
            return float(np.mean(vals)) if vals else float("nan")
        print(
            f"{step:>10,d}  {fam_mean('piR'):8.4f}  {fam_mean('piS'):8.4f}  "
            f"{seed_mean('piR',901001):9.4f} {seed_mean('piR',901002):9.4f} {seed_mean('piR',901003):9.4f}  "
            f"{seed_mean('piS',902001):9.4f} {seed_mean('piS',902002):9.4f} {seed_mean('piS',902003):9.4f}"
        )

    print("\nC_RUSH (for contrast)")
    print(f"{'step':>10s}  {'πR_fam':>8s}  {'πS_fam':>8s}")
    for step in args.steps:
        if step not in data:
            print(f"{step:>10,d}  (no rows)")
            continue
        def fam_mean(fam):
            seeds = data[step][fam]["C_RUSH"]
            if not seeds:
                return float("nan")
            return float(np.mean([np.mean(v) for v in seeds.values()]))
        print(f"{step:>10,d}  {fam_mean('piR'):8.4f}  {fam_mean('piS'):8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
