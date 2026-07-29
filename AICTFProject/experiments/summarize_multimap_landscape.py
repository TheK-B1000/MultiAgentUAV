#!/usr/bin/env python3
"""Summarize multi-map landscape: best blue per (red, map) context."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


STYLES = ("RUSH", "SPLIT", "TURTLE", "ESCORT")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    by: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        style = r["blue_style"].replace("BLUE_", "")
        by[(r["red_style"], r["map"], style)].append(r)

    contexts = sorted({(r["red_style"], r["map"]) for r in rows})
    table = []
    print(f"{'context':<48} {'best':<8} {'mean':>6} {'gap':>6} {'WR':>5}  ranking")
    for red, mp in contexts:
        means = {}
        wrs = {}
        ttfs = {}
        for s in STYLES:
            eps = by[(red, mp, s)]
            if not eps:
                continue
            means[s] = mean(float(e["win_margin"]) for e in eps)
            wrs[s] = mean(float(e["success"]) for e in eps)
            tvals = [
                float(e["time_to_first_score"])
                for e in eps
                if e.get("time_to_first_score") not in ("", None)
            ]
            ttfs[s] = mean(tvals) if tvals else float("nan")
        ranked = sorted(means.items(), key=lambda kv: -kv[1])
        best, bm = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else bm
        gap = bm - second
        ranking = " > ".join(f"{s}:{m:+.2f}" for s, m in ranked)
        row = {
            "red": red,
            "map": mp,
            "best": best,
            "best_mean": bm,
            "gap": gap,
            "best_wr": wrs.get(best, float("nan")),
            "best_ttfs": ttfs.get(best, float("nan")),
            "means": means,
            "wrs": wrs,
            "ttfs": ttfs,
            "saturated": all(m >= 1.5 for m in means.values()),
            "near_tie": abs(gap) < 0.25,
        }
        table.append(row)
        print(
            f"{red}|{mp:<22} {best:<8} {bm:+6.2f} {gap:+6.2f} "
            f"{wrs.get(best, 0):5.2f}  {ranking}"
        )

    # Candidate contexts per style (unique best, gap>=0.5, not saturated/near-tie)
    print("\nCandidate contexts (unique best, gap>=0.5, not saturated/near-tie):")
    by_style: dict[str, list] = defaultdict(list)
    for row in table:
        if row["near_tie"] or row["saturated"] or row["gap"] < 0.5:
            continue
        by_style[row["best"]].append(row)
    for s in STYLES:
        cands = sorted(by_style[s], key=lambda r: -r["gap"])
        if not cands:
            print(f"  {s}: (none)")
            continue
        for c in cands:
            print(
                f"  {s}: {c['red']}|{c['map']}  gap={c['gap']:+.2f} "
                f"mean={c['best_mean']:+.2f} WR={c['best_wr']:.2f}"
            )

    if args.out_json:
        args.out_json.write_text(json.dumps(table, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
