#!/usr/bin/env python3
"""Aggregate eval results across seeds for Phase 4A vs no_latent multi-seed validation.

Reads ``op4_zero_shot_comparison.csv`` and groups rows by (base_method, opponent),
where base_method strips the ``_seed<N>`` marker and treats tags without a marker
as seed 0. Reports per-cell mean / std / standard error / 95% CI half-width over
the available seeds, and a paired comparison row showing
``phase4a_mean - no_latent_mean`` with combined SE so you can read off whether
the gap is statistically meaningful.

Naming convention assumed:

    plan_faithful_no_latent_hardpool_1m_4v4               <- seed 0 (no marker)
    plan_faithful_no_latent_seed7_hardpool_1m_4v4         <- seed 7
    plan_faithful_no_latent_seed42_hardpool_1m_4v4        <- seed 42
    plan_faithful_latent_phase4a_rescue_hardpool_1m_4v4               <- seed 0
    plan_faithful_latent_phase4a_rescue_seed7_hardpool_1m_4v4         <- seed 7
    plan_faithful_latent_phase4a_rescue_seed42_hardpool_1m_4v4        <- seed 42

Outputs (default to <comparison-dir>/):

    seed_aggregate_table.csv      one row per (method, opponent) with mean/SE/CI/N
    seed_paired_comparison.csv    per-opponent phase4a_mean - no_latent_mean ± SE

Usage
-----

    python experiments/build_seed_aggregate_table.py
    python experiments/build_seed_aggregate_table.py \
        --comparison-csv checkpoints/4v4/eval_op4_zero_shot/op4_zero_shot_comparison.csv

Independent of pandas; uses only stdlib + csv + math.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict
from typing import Iterable

SEED_PATTERN = re.compile(r"_seed(\d+)_")


def _strip_seed(run_tag: str) -> tuple[str, int]:
    """Return (base_tag_without_seed_marker, seed). Tags without _seed<N>_ are seed 0."""
    m = SEED_PATTERN.search(run_tag)
    if m is None:
        return run_tag, 0
    seed = int(m.group(1))
    base = run_tag[: m.start()] + run_tag[m.end() - 1 :]
    return base, seed


def _mean_std_se_ci(values: list[float]) -> tuple[float, float, float, float, int]:
    """Return (mean, std_sample, se, ci95_halfwidth, n). For n<2, std/se = 0."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0)
    mean = sum(values) / n
    if n < 2:
        return (mean, 0.0, 0.0, 0.0, n)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    se = std / math.sqrt(n)
    # 1.96 * SE is the asymptotic 95% half-width; for very small N this is liberal,
    # but with N=3 a t-critical of ~4.30 would be more honest. We report 1.96*SE for
    # simplicity and let the user reason about it.
    ci95 = 1.96 * se
    return (mean, std, se, ci95, n)


def _read_comparison_rows(path: str) -> list[dict[str, str]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"comparison CSV not found: {path}")
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [dict(r) for r in reader]


def _filter_methods(rows: list[dict[str, str]], methods: Iterable[str]) -> list[dict[str, str]]:
    keep_bases = set(methods)
    out: list[dict[str, str]] = []
    for r in rows:
        base, _seed = _strip_seed(str(r.get("run_tag", "")))
        if base in keep_bases:
            r = dict(r)
            r["_base_run_tag"] = base
            r["_seed"] = str(_seed)
            out.append(r)
    return out


def _write_csv(path: str, rows: list[dict[str, object]], fields: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--comparison-csv",
        type=str,
        default=os.path.join("checkpoints", "4v4", "eval_op4_zero_shot", "op4_zero_shot_comparison.csv"),
        help="Path to op4_zero_shot_comparison.csv produced by eval_op4_zero_shot.py.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "plan_faithful_no_latent_hardpool_1m_4v4",
            "plan_faithful_latent_phase4a_rescue_hardpool_1m_4v4",
        ],
        help="Base run-tags (after stripping _seed<N>_) to aggregate.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Where to write seed_aggregate_table.csv / seed_paired_comparison.csv. "
        "Default = same dir as --comparison-csv.",
    )
    args = parser.parse_args()

    comp_path = os.path.abspath(args.comparison_csv)
    out_dir = os.path.abspath(args.out_dir or os.path.dirname(comp_path))
    os.makedirs(out_dir, exist_ok=True)

    rows = _read_comparison_rows(comp_path)
    rows = _filter_methods(rows, args.methods)
    if not rows:
        print(f"[seed-aggregate] no rows match methods={args.methods} in {comp_path}", file=sys.stderr)
        sys.exit(2)

    # Group: (base_method, opponent) -> list[(seed, win_rate, mean_steps, mean_return)]
    cells: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for r in rows:
        base = r["_base_run_tag"]
        opp = str(r.get("opponent", ""))
        try:
            wr = float(r.get("win_rate", 0.0))
        except (TypeError, ValueError):
            continue
        cells[(base, opp)].append(
            {
                "seed": float(r["_seed"]),
                "win_rate": wr,
                "mean_steps": float(r.get("mean_steps", 0.0)),
                "mean_return": float(r.get("mean_return", 0.0)),
            }
        )

    methods_seen = sorted({k[0] for k in cells.keys()})
    opponents_seen = sorted({k[1] for k in cells.keys()})

    aggregate_rows: list[dict[str, object]] = []
    for method in methods_seen:
        for opp in opponents_seen:
            cell = cells.get((method, opp), [])
            if not cell:
                continue
            wrs = [c["win_rate"] for c in cell]
            mean, std, se, ci95, n = _mean_std_se_ci(wrs)
            seeds_used = sorted({int(c["seed"]) for c in cell})
            aggregate_rows.append(
                {
                    "method": method,
                    "opponent": opp,
                    "n_seeds": n,
                    "seeds": ";".join(str(s) for s in seeds_used),
                    "win_rate_mean": round(mean, 4),
                    "win_rate_std": round(std, 4),
                    "win_rate_se": round(se, 4),
                    "win_rate_ci95_half": round(ci95, 4),
                    "win_rate_lo95": round(mean - ci95, 4),
                    "win_rate_hi95": round(mean + ci95, 4),
                    "mean_steps_mean": round(sum(c["mean_steps"] for c in cell) / n, 2),
                    "mean_return_mean": round(sum(c["mean_return"] for c in cell) / n, 4),
                }
            )

    agg_path = os.path.join(out_dir, "seed_aggregate_table.csv")
    _write_csv(
        agg_path,
        aggregate_rows,
        [
            "method",
            "opponent",
            "n_seeds",
            "seeds",
            "win_rate_mean",
            "win_rate_std",
            "win_rate_se",
            "win_rate_ci95_half",
            "win_rate_lo95",
            "win_rate_hi95",
            "mean_steps_mean",
            "mean_return_mean",
        ],
    )

    # Paired comparison (phase4a - no_latent) per opponent, with combined SE.
    phase4a = "plan_faithful_latent_phase4a_rescue_hardpool_1m_4v4"
    no_latent = "plan_faithful_no_latent_hardpool_1m_4v4"
    paired_rows: list[dict[str, object]] = []
    if phase4a in methods_seen and no_latent in methods_seen:
        for opp in opponents_seen:
            a = cells.get((phase4a, opp), [])
            b = cells.get((no_latent, opp), [])
            if not a or not b:
                continue
            a_wrs = [c["win_rate"] for c in a]
            b_wrs = [c["win_rate"] for c in b]
            a_mean, a_std, a_se, _, a_n = _mean_std_se_ci(a_wrs)
            b_mean, b_std, b_se, _, b_n = _mean_std_se_ci(b_wrs)
            diff = a_mean - b_mean
            combined_se = math.sqrt(a_se * a_se + b_se * b_se)
            ci95_half = 1.96 * combined_se
            wins = "phase4a" if diff > ci95_half else ("no_latent" if diff < -ci95_half else "tie")
            paired_rows.append(
                {
                    "opponent": opp,
                    "phase4a_mean": round(a_mean, 4),
                    "phase4a_n": a_n,
                    "no_latent_mean": round(b_mean, 4),
                    "no_latent_n": b_n,
                    "diff_mean": round(diff, 4),
                    "diff_se": round(combined_se, 4),
                    "diff_lo95": round(diff - ci95_half, 4),
                    "diff_hi95": round(diff + ci95_half, 4),
                    "verdict": wins,
                }
            )

    paired_path = os.path.join(out_dir, "seed_paired_comparison.csv")
    _write_csv(
        paired_path,
        paired_rows,
        [
            "opponent",
            "phase4a_mean",
            "phase4a_n",
            "no_latent_mean",
            "no_latent_n",
            "diff_mean",
            "diff_se",
            "diff_lo95",
            "diff_hi95",
            "verdict",
        ],
    )

    # Pretty-print to terminal.
    print(f"\n[seed-aggregate] aggregate written: {agg_path}")
    print(f"[seed-aggregate] paired comparison written: {paired_path}")

    print("\n[seed-aggregate] CELL SUMMARY (mean ± 1.96·SE)")
    header = f"  {'method':<58}  {'opp':<12}  {'N':>2}  {'mean':>6}  {'95% CI':>16}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in aggregate_rows:
        ci = f"[{100.0 * float(r['win_rate_lo95']):>5.1f}, {100.0 * float(r['win_rate_hi95']):>5.1f}]"
        print(
            f"  {str(r['method']):<58}  {str(r['opponent']):<12}  {r['n_seeds']:>2}  "
            f"{100.0 * float(r['win_rate_mean']):>5.1f}%  {ci:>16}"
        )

    if paired_rows:
        print("\n[seed-aggregate] PAIRED COMPARISON (phase4a − no_latent)")
        ph = f"  {'opp':<12}  {'phase4a':>8}  {'no_latent':>10}  {'diff':>7}  {'95% CI':>18}  verdict"
        print(ph)
        print("  " + "-" * (len(ph) - 2))
        for r in paired_rows:
            ci = f"[{100.0 * float(r['diff_lo95']):>+5.1f}, {100.0 * float(r['diff_hi95']):>+5.1f}]"
            print(
                f"  {str(r['opponent']):<12}  {100.0 * float(r['phase4a_mean']):>7.1f}%  "
                f"{100.0 * float(r['no_latent_mean']):>9.1f}%  "
                f"{100.0 * float(r['diff_mean']):>+6.1f}%  {ci:>18}  {r['verdict']}"
            )


if __name__ == "__main__":
    main()
