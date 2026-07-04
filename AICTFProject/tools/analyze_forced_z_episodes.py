#!/usr/bin/env python3
"""Summarize forced-z episode_results.csv: margin/timing/WR spread by z."""
from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path


def analyze(path: Path, label: str) -> None:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        print(f"{label}: NO DATA")
        return

    by_z: dict[int, list[dict]] = defaultdict(list)
    by_cell_z: dict[tuple[str, int], list[int]] = defaultdict(list)
    for r in rows:
        z = int(r["latent_z"])
        cell = f"{r['opponent']}|{r['map']}"
        rec = {
            "wm": float(r.get("win_margin") or 0),
            "ret": float(r.get("return") or 0),
            "tfs": float(r.get("time_to_first_score") or 0),
            "steps": float(r.get("steps") or 0),
            "succ": int(r.get("success") or 0),
            "intercept": float(r.get("behavior_n_intercept_near_enemy_carrier") or 0),
            "escort": float(r.get("behavior_carrier_escort_count") or 0),
        }
        by_z[z].append(rec)
        by_cell_z[(cell, z)].append(rec["succ"])

    print(f"=== {label} (n={len(rows)}) ===")
    for z in sorted(by_z):
        xs = by_z[z]
        wr = sum(x["succ"] for x in xs) / len(xs)
        print(
            f"  z{z}: WR={wr:.0%} n={len(xs)} "
            f"margin={st.mean(x['wm'] for x in xs):.2f}±{st.pstdev(x['wm'] for x in xs):.2f} "
            f"tfs={st.mean(x['tfs'] for x in xs):.1f} steps={st.mean(x['steps'] for x in xs):.0f} "
            f"return={st.mean(x['ret'] for x in xs):.2f} "
            f"intercept={st.mean(x['intercept'] for x in xs):.3f} escort={st.mean(x['escort'] for x in xs):.3f}"
        )

    cells = sorted({c for c, _ in by_cell_z})
    wr_spreads = []
    for c in cells:
        wrs = [
            sum(by_cell_z.get((c, z), [])) / max(1, len(by_cell_z.get((c, z), [])))
            for z in range(4)
        ]
        wr_spreads.append(max(wrs) - min(wrs))
    z_margins = {z: st.mean(x["wm"] for x in by_z[z]) for z in by_z}
    z_tfs = {z: st.mean(x["tfs"] for x in by_z[z]) for z in by_z}
    z_steps = {z: st.mean(x["steps"] for x in by_z[z]) for z in by_z}
    print(
        f"  WR spread across z per cell: mean={st.mean(wr_spreads):.3f} max={max(wr_spreads):.3f}"
    )
    print(
        f"  margin spread across z: {max(z_margins.values()) - min(z_margins.values()):.3f} "
        f"({', '.join(f'z{z}={z_margins[z]:.2f}' for z in sorted(z_margins))})"
    )
    print(
        f"  tfs spread across z: {max(z_tfs.values()) - min(z_tfs.values()):.1f} "
        f"steps spread: {max(z_steps.values()) - min(z_steps.values()):.0f}"
    )
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="+", help="episode_results.csv paths")
    p.add_argument("--label", nargs="*", default=[], help="optional labels")
    args = p.parse_args()
    for i, csv_path in enumerate(args.csv):
        label = args.label[i] if i < len(args.label) else Path(csv_path).parent.parent.name
        analyze(Path(csv_path), label)


if __name__ == "__main__":
    main()
