#!/usr/bin/env python3
"""K=2 behavior audit analysis: pairwise divergence matrices, bootstrap CIs,
separation ratios, and the s902002 sensitivity slice.

Consumes ``divergence_episodes.csv`` and ``tactical_episodes.csv`` from
audit_k2_specialist_behavior.py. DIAGNOSTIC ONLY -- nothing here is a gate and
nothing here can overturn the 1M payoff verdict. The audit explains WHY the
formal proof failed; it does not re-decide it.

Two separations are kept strictly apart, because they answer different
questions and can disagree:

  COUNTERFACTUAL POLICY SEPARATION (divergence_episodes.csv)
      masked-logit JSD / KL, argmax disagreement, at byte-identical
      observations. "Given exactly the same information and the same legal
      actions, do these networks choose differently?"

  ON-POLICY BEHAVIORAL SEPARATION (tactical_episodes.csv)
      lane occupancy, agent separation, carrier-return route, home-defense
      time, screening/interposition, capture timing. "Do their full
      trajectories differ when each policy drives the environment?"

Joint interpretation:

  low counterfactual + similar trajectories -> same policy region; piR simply
                                               executes better
  high counterfactual + similar payoff      -> distinct behavior with no
                                               complementary value
  early separation that vanishes by 1M      -> specialization emerged and was
                                               erased by continued optimization
  no separation at any checkpoint           -> both families converged toward
                                               the same solution class

Confidence intervals are bootstrapped over EPISODES, resampled within each
(checkpoint, context, observation source) stratum so the pairing across
policies is preserved -- the same episode contributes to every pair at once.

s902002 collapsed during training. It is RETAINED in the headline full-family
result. Because a degenerate policy can inflate within_piS and thereby deflate
the separation ratio, a clearly labeled sensitivity slice over s902001 /
s902003 is also reported. That slice is diagnostic and never a gate.
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

PI_R = ["piR/s901001", "piR/s901002", "piR/s901003"]
PI_S = ["piS/s902001", "piS/s902002", "piS/s902003"]
COLLAPSED = "piS/s902002"
METRICS = ["jsd_all_bits", "jsd_macro_bits", "argmax_disagreement", "macro_disagreement"]
SEPARATION_RATIO_THRESHOLD = 1.5
# Minimum relative change in within_piS before the s902002 sensitivity slice
# claims the collapsed seed is distorting the denominator.
INFLATION_REL_THRESHOLD = 0.10


def family_of(key: str) -> str:
    return key.split("/")[0]


def load_divergence(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rec = {
                "step": int(r["checkpoint_step"]),
                "context": r["context"],
                "obs_source": r["obs_source"],
                "episode_index": int(r["episode_index"]),
                "a": r["policy_a"],
                "b": r["policy_b"],
                "pair_type": r["pair_type"],
            }
            for m in METRICS:
                rec[m] = float(r[m])
            rows.append(rec)
    return rows


def symmetrized_pair_means(rows: list[dict], metric: str, keys: set[str] | None = None) -> dict:
    """Mean of `metric` per unordered pair, symmetrized over observation source.

    JSD(a,b) = 0.5 * [ mean on a's states + mean on b's states ]. Sources that
    are neither a nor b still carry information, so they are folded into a
    third "other" term only when present; the headline uses the a/b symmetric
    form, which is what the audit docstring defines.
    """
    by_pair_src = defaultdict(list)
    for r in rows:
        if keys is not None and (r["a"] not in keys or r["b"] not in keys):
            continue
        by_pair_src[(r["a"], r["b"], r["obs_source"])].append(r[metric])

    out = {}
    pairs = {(a, b) for (a, b, _s) in by_pair_src}
    for (a, b) in pairs:
        on_a = by_pair_src.get((a, b, a))
        on_b = by_pair_src.get((a, b, b))
        parts = [np.mean(x) for x in (on_a, on_b) if x]
        if parts:
            out[(a, b)] = float(np.mean(parts))
    return out


def separation_ratio(pair_means: dict) -> tuple[float, float, float, float, float]:
    """-> (ratio, between, within_piR, within_piS, d_policy) from pair means.

    ``d_policy`` is the PREREGISTERED confirmatory statistic:

        D_policy = between - mean(within_piR, within_piS)

    It is a signed difference, not a ratio. Its null value is exactly zero and
    it can go negative, so a percentile LCB is not clipped at a boundary the way
    the ratio (>= 0, null 1.0) and the old Delta_pool were. The ratio is kept
    for readability only.
    """
    wr, ws, bt = [], [], []
    for (a, b), v in pair_means.items():
        fa, fb = family_of(a), family_of(b)
        if fa != fb:
            bt.append(v)
        elif fa == "piR":
            wr.append(v)
        else:
            ws.append(v)
    nan = float("nan")
    if not (wr and ws and bt):
        return nan, nan, nan, nan, nan
    within = float(np.mean([np.mean(wr), np.mean(ws)]))
    between = float(np.mean(bt))
    ratio = between / within if within > 1e-12 else float("inf")
    return ratio, between, float(np.mean(wr)), float(np.mean(ws)), between - within


def bootstrap_cell(rows: list[dict], metric: str, keys: set[str] | None,
                   n_boot: int, rng, alpha: float) -> dict:
    """Bootstrap over episodes within each (obs_source) stratum.

    Resampling whole episodes (not individual pair rows) keeps every pair
    measured on the same resampled states, which is what makes the ratio's CI
    meaningful.
    """
    sub = [r for r in rows
           if keys is None or (r["a"] in keys and r["b"] in keys)]
    if not sub:
        return {}

    by_src_ep = defaultdict(list)
    for r in sub:
        by_src_ep[(r["obs_source"], r["episode_index"])].append(r)
    srcs = sorted({s for (s, _e) in by_src_ep})
    eps_by_src = {s: sorted({e for (s2, e) in by_src_ep if s2 == s}) for s in srcs}

    point = separation_ratio(symmetrized_pair_means(sub, metric, keys))

    boots = np.empty((n_boot, 5))
    for i in range(n_boot):
        resampled = []
        for s in srcs:
            eps = eps_by_src[s]
            pick = rng.integers(0, len(eps), len(eps))
            for j in pick:
                resampled.extend(by_src_ep[(s, eps[j])])
        boots[i] = separation_ratio(symmetrized_pair_means(resampled, metric, keys))

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    names = ["ratio", "between", "within_piR", "within_piS", "d_policy"]
    out = {}
    for k, name in enumerate(names):
        col = boots[:, k]
        col = col[np.isfinite(col)]
        if col.size == 0:
            out[name] = (point[k], float("nan"), float("nan"))
        else:
            lo, hi = np.percentile(col, [lo_q, hi_q])
            out[name] = (point[k], float(lo), float(hi))
    return out


def print_matrix(pair_means: dict, title: str) -> None:
    keys = sorted({k for pair in pair_means for k in pair})
    if not keys:
        return
    print(f"\n  {title}")
    print("    " + "".join(f"{k.split('/')[1]:>11s}" for k in keys))
    for a in keys:
        cells = []
        for b in keys:
            if a == b:
                cells.append(f"{'-':>11s}")
            else:
                v = pair_means.get((a, b), pair_means.get((b, a)))
                cells.append(f"{v:>11.5f}" if v is not None else f"{'':>11s}")
        print(f"    {a.split('/')[1]:>9s}" + "".join(cells))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audit-dir", default="artifacts/k2v2_specialist_behavior_audit")
    p.add_argument("--metric", default="jsd_all_bits", choices=METRICS)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    base = Path(args.audit_dir)
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    div_path = base / "divergence_episodes.csv"
    tac_path = base / "tactical_episodes.csv"
    if not div_path.exists():
        print(f"[abort] missing {div_path}", file=sys.stderr)
        return 1

    rows = load_divergence(div_path)
    rng = np.random.default_rng(args.seed)
    steps = sorted({r["step"] for r in rows})
    contexts = sorted({r["context"] for r in rows})

    print("=" * 78)
    print("K=2 BEHAVIOR AUDIT -- COUNTERFACTUAL POLICY SEPARATION")
    print(f"metric = {args.metric}   (masked-logit, byte-identical observations)")
    print("DIAGNOSTIC ONLY. Does not modify the 1M payoff verdict (FAIL).")
    print("=" * 78)

    full_keys = set(PI_R + PI_S)
    sens_keys = set(PI_R + [k for k in PI_S if k != COLLAPSED])

    summary = []
    for step in steps:
        for ctx in contexts:
            cell = [r for r in rows if r["step"] == step and r["context"] == ctx]
            if not cell:
                continue
            print(f"\n{'-' * 78}\n### step {step:,}  {ctx}\n{'-' * 78}")

            pm = symmetrized_pair_means(cell, args.metric, full_keys)
            print_matrix({k: v for k, v in pm.items()
                          if family_of(k[0]) == "piR" and family_of(k[1]) == "piR"},
                         "piR within-family matrix")
            print_matrix({k: v for k, v in pm.items()
                          if family_of(k[0]) == "piS" and family_of(k[1]) == "piS"},
                         "piS within-family matrix")
            cross = {k: v for k, v in pm.items() if family_of(k[0]) != family_of(k[1])}
            if cross:
                print("\n  cross-family pairs (piR x piS)")
                for (a, b), v in sorted(cross.items()):
                    print(f"    {a:>12s} x {b:<12s} {v:>11.5f}")

            res = bootstrap_cell(cell, args.metric, full_keys, args.n_boot, rng, args.alpha)
            print("\n  FULL FAMILY (s902002 retained -- this is the headline result)")
            for name in ("within_piR", "within_piS", "between", "ratio"):
                pt, lo, hi = res[name]
                print(f"    {name:12s} {pt:>9.5f}  CI95=[{lo:>8.5f}, {hi:>8.5f}]")
            ratio_pt, ratio_lo, _ = res["ratio"]
            verdict = ("families distinguishable"
                       if ratio_lo >= SEPARATION_RATIO_THRESHOLD
                       else "NOT distinguishable -- same behavior region")
            print(f"    => separation_ratio {ratio_pt:.3f} (LCB {ratio_lo:.3f}): {verdict}")

            sens = bootstrap_cell(cell, args.metric, sens_keys, args.n_boot, rng, args.alpha)
            if sens:
                s_pt, s_lo, s_hi = sens["ratio"]
                w_pt, _, _ = sens["within_piS"]
                print("\n  SENSITIVITY SLICE (s902001 / s902003 only) -- LABELED DIAGNOSTIC,")
                print("  not a gate, does not replace the full-family result above.")
                full_w = res["within_piS"][0]
                # Relative change, not a bare '>': any two bootstrap means differ
                # by noise, so a strict inequality would claim inflation almost
                # every time. Require a materially larger within_piS.
                rel = (full_w - w_pt) / w_pt if w_pt > 1e-12 else 0.0
                print(f"    within_piS   {w_pt:>9.5f}  (full-family: {full_w:.5f}, "
                      f"{rel:+.1%})")
                print(f"    ratio        {s_pt:>9.5f}  CI95=[{s_lo:>8.5f}, {s_hi:>8.5f}]")
                if rel >= INFLATION_REL_THRESHOLD:
                    print(f"    -> s902002 inflates within_piS by {rel:.1%}; the full-family")
                    print("       ratio is CONSERVATIVE (understates separation of the")
                    print("       healthy seeds). Both numbers stand; neither is a gate.")
                elif rel <= -INFLATION_REL_THRESHOLD:
                    print(f"    -> s902002 DEFLATES within_piS by {-rel:.1%} (it is closer to")
                    print("       its siblings than they are to each other).")
                else:
                    print(f"    -> s902002 does not materially move within_piS ({rel:+.1%}, "
                          f"under {INFLATION_REL_THRESHOLD:.0%});")
                    print("       the full-family ratio is not being distorted by it.")

            summary.append((step, ctx, res["ratio"][0], res["ratio"][1],
                            sens["ratio"][0] if sens else float("nan")))

    print(f"\n{'=' * 78}\nSEPARATION-RATIO TRAJECTORY\n{'=' * 78}")
    print(f"{'step':>10s} {'context':>10s} {'ratio':>9s} {'LCB95':>9s} {'sens.ratio':>11s}")
    for step, ctx, pt, lo, sp in summary:
        print(f"{step:>10,} {ctx:>10s} {pt:>9.3f} {lo:>9.3f} {sp:>11.3f}")

    ever = [s for s in summary if s[3] >= SEPARATION_RATIO_THRESHOLD]
    print()
    if not ever:
        print("No checkpoint or context shows counterfactual family separation.")
        print("Combined with the payoff result, this supports: both families converged")
        print("toward the same solution class, and piR simply executes it better.")
    else:
        print("Counterfactual separation present at:")
        for s in ever:
            print(f"  step {s[0]:,} {s[1]} (ratio {s[2]:.3f}, LCB {s[3]:.3f})")
        print("Check whether it coincides with any payoff crossover in the trajectory")
        print("analysis. Separation without complementary payoff = distinct behavior,")
        print("no repertoire value.")

    # ---- on-policy behavioral separation --------------------------------
    if tac_path.exists():
        print(f"\n{'=' * 78}\nON-POLICY BEHAVIORAL SEPARATION (each policy drives the env)")
        print(f"{'=' * 78}")
        trows = []
        with open(tac_path, newline="") as f:
            for r in csv.DictReader(f):
                trows.append(r)
        fields = ["opposed_lane_frac", "mean_agent_sep", "mean_y_sep",
                  "home_half_occupancy", "home_defense_frac", "screen_frac",
                  "carry_path_efficiency", "first_pickup_step", "first_capture_step",
                  "capture_occurred"]
        for step in steps:
            for ctx in contexts:
                sel = [r for r in trows
                       if int(r["checkpoint_step"]) == step and r["context"] == ctx]
                if not sel:
                    continue
                print(f"\n  step {step:,} {ctx}")
                print(f"    {'metric':<24s}{'piR':>10s}{'piS':>10s}{'diff':>10s}")
                for fld in fields:
                    vals = {}
                    for fam in ("piR", "piS"):
                        xs = []
                        for r in sel:
                            if r["family"] != fam:
                                continue
                            try:
                                v = float(r[fld])
                            except (KeyError, ValueError):
                                continue
                            if np.isfinite(v):
                                xs.append(v)
                        vals[fam] = float(np.mean(xs)) if xs else float("nan")
                    print(f"    {fld:<24s}{vals['piR']:>10.4f}{vals['piS']:>10.4f}"
                          f"{vals['piR'] - vals['piS']:>+10.4f}")
    else:
        print(f"\n[note] {tac_path} not found; on-policy section skipped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
