#!/usr/bin/env python3
"""CONFIRMATORY behavior gate B_distinct for the 300k replication.

Separate from analyze_k2_specialist_behavior.py on purpose: that script is
DESCRIPTIVE (D_policy, separation ratio, argmax disagreement, pairwise
matrices, healthy-seed sensitivity) and must not decide branch birth. This
script computes the single preregistered confirmatory statistic.

WHY D_policy WAS REPLACED
-------------------------
    D_policy = mean(between) - mean(within)

is not a meaningful strategy-distinction gate at this sample size. Two
independently trained networks are always distinguishable given enough matched
observations, so LCB95(D_policy) > 0 tests "are these networks different at
all", not "are they meaningfully different". Demonstrated empirically: the 1M
discovery checkpoint PASSES that gate (D_policy = +0.0058, LCB > 0) despite
complete payoff collapse into a single dominant generalist, on only 3 episodes
per reference.

THE CONFIRMATORY STATISTIC
--------------------------
    B_distinct = median( JSD_between_families ) - Q_0.95( JSD_within_families )

    gate: LCB95(B_distinct) > 0

In plain language: the TYPICAL piR-vs-piS difference must exceed at least 95%
of the differences seen between independently trained seeds of the SAME family.
This carries real weight without inventing an arbitrary JSD or ratio threshold
-- the bar is set by the observed seed-to-seed variation itself.

CONSTRUCTION
------------
Observation bank: balanced across C_RUSH and C_SPLIT, balanced across
piR-generated and piS-generated states, byte-identical observations for every
compared policy, identical legal-action masking. Each unordered pair's value is
symmetrized over its own two endpoints' state distributions and then averaged
with equal weight across the two contexts, so the formal statistic is one
balanced pooled number. Context-specific values are reported as diagnostics.

Within-family pool : all non-self piR-piR pairs + all non-self piS-piS pairs
Between-family pool: all piR-piS pairs

Bootstrap (hierarchical): resample training seeds with replacement within each
family; resample episodes with replacement within each observation source;
exclude degenerate self-pairs created by duplicate seed draws.

UNSTABLE SEEDS ARE NOT EXCLUDED from the formal analysis. If a collapsed seed
inflates within-family variation enough to fail this gate, that means the
proposed family is not behaviorally coherent enough to serve as a stable latent
branch. That is a real finding, not a nuisance to be corrected.
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

METRICS = ["jsd_all_bits", "jsd_macro_bits", "argmax_disagreement", "macro_disagreement"]
WITHIN_QUANTILE = 0.95


def family_of(key: str) -> str:
    return key.split("/")[0]


def load(path: Path, step: int, metric: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["checkpoint_step"]) != step:
                continue
            rows.append({
                "context": r["context"],
                "obs_source": r["obs_source"],
                "episode_index": int(r["episode_index"]),
                "a": r["policy_a"], "b": r["policy_b"],
                "v": float(r[metric]),
            })
    return rows


def pair_values(rows: list[dict], balanced: bool = True) -> dict:
    """-> {(a,b): value}. Symmetrized over endpoints, balanced over contexts.

    Symmetrization uses only the two endpoints' own on-policy states, which is
    what makes each pair's value comparable regardless of which family it spans.
    Context balancing gives C_RUSH and C_SPLIT equal weight rather than letting
    whichever context produced more steps dominate.
    """
    by = defaultdict(list)
    for r in rows:
        if r["obs_source"] in (r["a"], r["b"]):
            by[(r["a"], r["b"], r["context"], r["obs_source"])].append(r["v"])

    per_ctx = defaultdict(dict)
    for (a, b, ctx, src), vals in by.items():
        per_ctx[(a, b, ctx)][src] = float(np.mean(vals))

    tmp = defaultdict(list)
    for (a, b, ctx), srcmap in per_ctx.items():
        tmp[(a, b)].append((ctx, float(np.mean(list(srcmap.values())))))

    out = {}
    for pair, items in tmp.items():
        if balanced:
            byc = defaultdict(list)
            for ctx, v in items:
                byc[ctx].append(v)
            out[pair] = float(np.mean([np.mean(v) for v in byc.values()]))
        else:
            out[pair] = float(np.mean([v for _c, v in items]))
    return out


def b_distinct(pairs: dict, drawn: dict | None = None) -> tuple[float, float, float]:
    """-> (B_distinct, median_between, Q95_within).

    ``drawn`` optionally supplies a resampled multiset of seeds per family;
    degenerate self-pairs are skipped.

    The family names are read from ``drawn`` rather than hardcoded, so the same
    statistic serves the O1 gate (families G0 / O1) as well as the k2 families
    it was written for. For any two-family ``drawn`` this iterates exactly the
    pairs the hardcoded ("piR", "piS") version did, in the same order; the k2
    numbers are unchanged, which
    ``tests/test_b_distinct_family_generalization.py`` asserts against the
    recorded k2 audit CSV.
    """
    def look(a, b):
        return pairs.get((a, b), pairs.get((b, a)))

    within, between = [], []
    if drawn is None:
        for (a, b), v in pairs.items():
            (between if family_of(a) != family_of(b) else within).append(v)
    else:
        families = list(drawn)
        if len(families) != 2:
            raise ValueError(
                f"B_distinct compares exactly two families; got {families!r}"
            )
        for fam in families:
            d = drawn[fam]
            for i in range(len(d)):
                for j in range(i + 1, len(d)):
                    if d[i] == d[j]:
                        continue
                    v = look(d[i], d[j])
                    if v is not None:
                        within.append(v)
        left, right = families
        for a in drawn[left]:
            for b in drawn[right]:
                v = look(a, b)
                if v is not None:
                    between.append(v)

    if not within or not between:
        nan = float("nan")
        return nan, nan, nan
    med_b = float(np.median(between))
    q_w = float(np.quantile(within, WITHIN_QUANTILE))
    return med_b - q_w, med_b, q_w


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audit-dir", default="artifacts/k2v2_specialist_behavior_audit")
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--metric", default="jsd_all_bits", choices=METRICS)
    p.add_argument("--n-boot", type=int, default=4000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--descriptive", action="store_true",
                   help="Also print the descriptive-only D_policy and ratio.")
    args = p.parse_args()

    base = Path(args.audit_dir)
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    div = base / "divergence_episodes.csv"
    if not div.exists():
        print(f"[abort] missing {div}", file=sys.stderr)
        return 1

    rows = load(div, args.step, args.metric)
    if not rows:
        print(f"[abort] no rows at step {args.step:,}", file=sys.stderr)
        return 1

    pairs = pair_values(rows, balanced=True)
    keys = sorted({k for pr in pairs for k in pr})
    fams = {f: [k for k in keys if family_of(k) == f] for f in ("piR", "piS")}
    point, med_b, q_w = b_distinct(pairs)

    print("=" * 78)
    print(f"CONFIRMATORY BEHAVIOR GATE  B_distinct  @ step {args.step:,}")
    print(f"metric = {args.metric}, balanced pooled over contexts")
    print("=" * 78)
    print(f"  policies: piR={len(fams['piR'])} seeds, piS={len(fams['piS'])} seeds")
    print(f"  pairs   : within={sum(len(v)*(len(v)-1)//2 for v in fams.values())}, "
          f"between={len(fams['piR'])*len(fams['piS'])}")
    print(f"\n  median(JSD between families) = {med_b:.6f}")
    print(f"  Q{WITHIN_QUANTILE:.2f}(JSD within families)   = {q_w:.6f}")
    print(f"  B_distinct                   = {point:+.6f}")

    # hierarchical bootstrap: episodes within obs source, then seeds within family
    rng = np.random.default_rng(args.seed)
    by_src_ep = defaultdict(list)
    for r in rows:
        by_src_ep[(r["obs_source"], r["episode_index"])].append(r)
    srcs = sorted({s for (s, _e) in by_src_ep})
    eps_by_src = {s: sorted({e for (s2, e) in by_src_ep if s2 == s}) for s in srcs}

    boots = np.empty(args.n_boot)
    for i in range(args.n_boot):
        resampled = []
        for s in srcs:
            eps = eps_by_src[s]
            for j in rng.integers(0, len(eps), len(eps)):
                resampled.extend(by_src_ep[(s, eps[j])])
        pm = pair_values(resampled, balanced=True)
        drawn = {f: [v[k] for k in rng.integers(0, len(v), len(v))]
                 for f, v in fams.items()}
        boots[i] = b_distinct(pm, drawn)[0]

    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [100 * args.alpha / 2, 100 * (1 - args.alpha / 2)])
    ok = lo > 0
    print(f"  CI95                         = [{lo:+.6f}, {hi:+.6f}]")
    print(f"  LCB95                        = {lo:+.6f}")
    print(f"\n  GATE B_distinct: {'PASS' if ok else 'FAIL'}  (requires LCB95 > 0)")

    print("\n  Context-specific diagnostics (NOT the gate):")
    for ctx in sorted({r["context"] for r in rows}):
        sub = [r for r in rows if r["context"] == ctx]
        pv = pair_values(sub, balanced=False)
        bd, mb, qw = b_distinct(pv)
        print(f"    {ctx:9s} median_between={mb:.6f}  Q95_within={qw:.6f}  "
              f"B_distinct={bd:+.6f}")

    if args.descriptive:
        wr = [v for (a, b), v in pairs.items() if family_of(a) == family_of(b) == "piR"]
        ws = [v for (a, b), v in pairs.items() if family_of(a) == family_of(b) == "piS"]
        bt = [v for (a, b), v in pairs.items() if family_of(a) != family_of(b)]
        within_mean = float(np.mean([np.mean(wr), np.mean(ws)]))
        print("\n  DESCRIPTIVE ONLY -- does not decide branch birth:")
        print(f"    D_policy         = {float(np.mean(bt)) - within_mean:+.6f}")
        print(f"    separation_ratio = {float(np.mean(bt)) / within_mean:.4f}")

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
