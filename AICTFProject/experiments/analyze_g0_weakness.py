#!/usr/bin/env python3
"""Rank OP6-OP12 contexts by where the LEARNED incumbent G0 actually fails.

Consumes episode_rows.csv from run_g0_weakness_sweep.py.

Competence (predeclared, three-way — count opponents with family mean < 0):

    0–2 negative  → COMPETENT
    exactly 3     → AMBIGUOUS  (do NOT select C1; confirm map-wide competence
                                 with fresh seeds or train G0_map_a)
    4–7 negative  → INCOMPETENT (train a map_a incumbent first)

Only COMPETENT unlocks weakness selection. AMBIGUOUS / INCOMPETENT refuse C1.

WEAKNESS GATE (predeclared, strict; discovery only)

    all three G0 seeds have negative mean margin
    AND the family-level upper CI95 is strictly below zero  (UCB95 < 0)

Ranking among discovery qualifiers (mechanical):
    (1) lowest strongest-member margin W(c)=max_seed payoff
    (2) lowest family mean
    (3) lower saturation + tie rate

Discovery winners are NOT training targets. Confirm each candidate on a fresh
64-seed block before freezing C1 (see docs/g0-c1-confirmation-preregistration.md).

map_a results are never pooled with map_b. Behavior telemetry is descriptive
only and is not part of weakness selection.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

BEHAVIOR_FIELDS = [
    "team_spread", "num_attackers", "num_defenders", "carrier_escort_count",
    "avg_blue_to_enemy_flag", "avg_blue_to_own_flag",
    "intercept_pressure", "defense_pressure", "attack_defense_ratio",
]
SATURATION_MARGIN = 3
ROW_KEY = ("checkpoint_step", "g0_seed", "opponent", "map", "episode_seed")
N_OPPONENTS_EXPECTED = 7


def load(path: Path):
    """-> (data[opponent][g0_seed][episode_seed] = row, maps, duplicate_keys)"""
    data = defaultdict(lambda: defaultdict(dict))
    maps = set()
    seen = Counter()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            maps.add((r.get("map", "?"), r.get("resolved_map", "?")))
            seen[tuple(str(r.get(k, "")) for k in ROW_KEY)] += 1
            data[r["opponent"]][int(r["g0_seed"])][int(r["episode_seed"])] = r
    dups = {k: v for k, v in seen.items() if v > 1}
    return data, maps, dups


def paired_matrix(per_seed: dict):
    """-> (episode_seeds, g0_seeds, margins (n_seed, n_ep))"""
    seeds = sorted(per_seed)
    ep = sorted(set.intersection(*[set(v) for v in per_seed.values()]))
    arr = np.array([[float(per_seed[s][e]["win_margin"]) for e in ep] for s in seeds])
    return ep, seeds, arr


def hier_ci(arr: np.ndarray, rng, n_boot: int, alpha: float):
    """Resample G0 seeds and episodes — family-level uncertainty."""
    n_s, n_e = arr.shape
    b = np.empty(n_boot)
    for i in range(n_boot):
        si = rng.integers(0, n_s, n_s)
        ei = rng.integers(0, n_e, n_e)
        b[i] = arr[np.ix_(si, ei)].mean()
    lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)


def competence_verdict(n_negative_family_mean: int) -> str:
    """Three-way map-wide competence from count of opponents with family mean < 0."""
    if n_negative_family_mean <= 2:
        return "COMPETENT"
    if n_negative_family_mean == 3:
        return "AMBIGUOUS"
    return "INCOMPETENT"


def qualifies_weakness(per_seed_mean: np.ndarray, family_hi: float) -> dict:
    """Strict discovery weakness gate. UCB95 must be *strictly* < 0."""
    all_neg = bool((np.asarray(per_seed_mean, dtype=float) < 0).all())
    ucb_ok = bool(float(family_hi) < 0)  # exact 0 fails
    return {
        "all_seeds_negative": all_neg,
        "ucb95_strictly_negative": ucb_ok,
        "qualifies": bool(all_neg and ucb_ok),
    }


def select_c1(results: dict, order: list[str], competence: str) -> str | None:
    """Return selected opponent or None. Never selects under AMBIGUOUS/INCOMPETENT."""
    if competence != "COMPETENT":
        return None
    qualified = [
        o for o in order
        if qualifies_weakness(results[o]["per_seed_mean"], results[o]["hi"])["qualifies"]
    ]
    if not qualified:
        return None
    ranked = sorted(
        qualified,
        key=lambda o: (
            results[o]["W"],
            results[o]["family_mean"],
            results[o]["saturation"] + results[o]["tie"],
        ),
    )
    return ranked[0]


def analyze_results(results: dict, *, n_boot: int, alpha: float, rng) -> dict:
    """Pure summary from a filled results dict (tests / fixtures use this)."""
    order = sorted(results, key=lambda o: results[o]["W"])
    n_neg = sum(1 for o in results if results[o]["family_mean"] < 0)
    competence = competence_verdict(n_neg)

    # Pooled map-wide diagnostic: concatenate all (seed, episode) margins.
    pools = []
    for o in results:
        arr = results[o].get("arr")
        if arr is not None:
            pools.append(arr.reshape(-1))
    if pools:
        pooled = np.concatenate(pools)
        # Episode-only bootstrap for the pooled diagnostic (not a gate).
        idx = rng.integers(0, len(pooled), size=(n_boot, len(pooled)))
        boot = pooled[idx].mean(axis=1)
        lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
        pooled_mean = float(pooled.mean())
        pooled_lo, pooled_hi = float(np.percentile(boot, lo_q)), float(np.percentile(boot, hi_q))
    else:
        pooled_mean = pooled_lo = pooled_hi = float("nan")

    for o in order:
        results[o]["gate"] = qualifies_weakness(results[o]["per_seed_mean"], results[o]["hi"])

    selected = select_c1(results, order, competence)
    return {
        "order": order,
        "n_negative_opponents": n_neg,
        "competence": competence,
        "pooled_map_mean": pooled_mean,
        "pooled_map_lo": pooled_lo,
        "pooled_map_hi": pooled_hi,
        "selected_c1": selected,
        "qualified": [
            o for o in order if results[o]["gate"]["qualifies"]
        ],
    }


def build_results_from_data(data: dict, rng, n_boot: int, alpha: float) -> dict:
    results = {}
    for opp in sorted(data):
        ep, seeds, arr = paired_matrix(data[opp])
        per_seed_mean = arr.mean(axis=1)
        fam_mean, fam_lo, fam_hi = hier_ci(arr, rng, n_boot, alpha)
        flat = arr.reshape(-1)
        beh = {}
        for b in BEHAVIOR_FIELDS:
            vals = []
            for s in seeds:
                for e in ep:
                    v = data[opp][s][e].get(f"behavior_{b}", "")
                    if v not in ("", None):
                        try:
                            fv = float(v)
                        except ValueError:
                            continue
                        if np.isfinite(fv):
                            vals.append(fv)
            beh[b] = float(np.mean(vals)) if vals else float("nan")
        results[opp] = {
            "seeds": seeds,
            "per_seed_mean": per_seed_mean,
            "family_mean": fam_mean,
            "lo": fam_lo,
            "hi": fam_hi,
            "W": float(per_seed_mean.max()),
            "win": float(np.mean(flat > 0)),
            "tie": float(np.mean(flat == 0)),
            "loss": float(np.mean(flat < 0)),
            "saturation": float(np.mean(np.abs(flat) >= SATURATION_MARGIN)),
            "behavior": beh,
            "n_ep": len(ep),
            "arr": arr,
        }
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", default="artifacts/g0_weakness_sweep/episode_rows.csv")
    p.add_argument("--n-boot", type=int, default=20000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows_path = Path(args.rows)
    if not rows_path.is_absolute():
        rows_path = PROJECT_ROOT / rows_path
    if not rows_path.exists():
        print(f"[abort] rows not found: {rows_path}", file=sys.stderr)
        return 1

    data, maps, dups = load(rows_path)
    if len(maps) != 1:
        print(f"[abort] rows mix multiple maps {sorted(maps)}; map_a must not be pooled "
              f"with map_b", file=sys.stderr)
        return 1
    if dups:
        print(f"[abort] {len(dups)} duplicate row keys {ROW_KEY}; refusing to analyze "
              f"possibly double-counted data", file=sys.stderr)
        for k in list(dups)[:5]:
            print(f"   {k} x{dups[k]}", file=sys.stderr)
        return 1
    map_label, map_layout = next(iter(maps))

    rng = np.random.default_rng(args.seed)
    results = build_results_from_data(data, rng, args.n_boot, args.alpha)
    summary = analyze_results(results, n_boot=args.n_boot, alpha=args.alpha, rng=rng)
    order = summary["order"]

    print("=" * 84)
    print(f"G0 LEARNED-INCUMBENT WEAKNESS SWEEP -- {map_label} ({map_layout})")
    print("G0 = frozen 1M piR family, used as incumbent generalist.")
    print("Ranking: W(c) = max over G0 seeds of payoff; LOWER = better weakness.")
    print("=" * 84)

    print(f"\n{'opponent':<32s}{'W=max seed':>11s}{'family':>9s}"
          f"{'CI95':>20s}{'win':>7s}{'tie':>7s}{'sat':>7s}")
    for opp in order:
        r = results[opp]
        ci = f"[{r['lo']:+.3f},{r['hi']:+.3f}]"
        print(f"{opp:<32s}{r['W']:>+11.4f}{r['family_mean']:>+9.4f}{ci:>20s}"
              f"{r['win']:>7.1%}{r['tie']:>7.1%}{r['saturation']:>7.1%}")

    print("\nPer-G0-seed mean margin (no best-seed selection; all three shown)")
    seed_hdr = "".join(f"{'s' + str(s):>12s}" for s in results[order[0]]["seeds"])
    print(f"{'opponent':<32s}{seed_hdr}{'all<0?':>9s}")
    for opp in order:
        r = results[opp]
        cells = "".join(f"{v:>+12.4f}" for v in r["per_seed_mean"])
        allneg = bool((r["per_seed_mean"] < 0).all())
        print(f"{opp:<32s}{cells}{('YES' if allneg else 'no'):>9s}")

    print("\nTrajectory fingerprints (mean behavior telemetry; DESCRIPTIVE ONLY)")
    keys = ["team_spread", "num_attackers", "num_defenders", "avg_blue_to_enemy_flag",
            "avg_blue_to_own_flag", "intercept_pressure", "defense_pressure"]
    print(f"{'opponent':<32s}" + "".join(f"{k[:11]:>12s}" for k in keys))
    for opp in order:
        b = results[opp]["behavior"]
        print(f"{opp:<32s}" + "".join(f"{b[k]:>12.3f}" for k in keys))

    # ---- competence (three-way) -----------------------------------------
    print(f"\n{'=' * 84}")
    print("MAP-WIDE COMPETENCE (count opponents with family mean < 0)")
    print("  0–2 → COMPETENT | exactly 3 → AMBIGUOUS | 4–7 → INCOMPETENT")
    print("=" * 84)
    print(f"  n_negative_family_mean = {summary['n_negative_opponents']} / {len(results)}")
    print(f"  verdict                = {summary['competence']}")
    print(f"  pooled map-wide mean   = {summary['pooled_map_mean']:+.4f}  "
          f"CI95=[{summary['pooled_map_lo']:+.4f}, {summary['pooled_map_hi']:+.4f}]  "
          f"(DIAGNOSTIC only; not a gate)")
    print("  Note: a single easy opponent must not hide several losses — competence")
    print("  is decided by the negative-opponent count, not the pooled mean.")

    if summary["competence"] == "INCOMPETENT":
        print("\n  → Broadly weak map_a incumbent. Do NOT select C1.")
        print("    Train a proper G0_map_a (OP6–OP12 mixture, multi-seed, no latent).")
    elif summary["competence"] == "AMBIGUOUS":
        print("\n  → Exactly three failing opponents — too broad for an isolated niche.")
        print("    Do NOT select a response-oracle context yet.")
        print("    Confirm map-wide competence with fresh seeds, or train G0_map_a.")
    else:
        print("\n  → Competent map_a incumbent. Isolated failures may become C1 candidates.")

    # ---- strict weakness gate -------------------------------------------
    print(f"\n{'=' * 84}")
    print("WEAKNESS GATE: all_seeds_negative AND UCB95 < 0 (strict)")
    print("=" * 84)
    for opp in order:
        g = results[opp]["gate"]
        mark = "QUALIFIES" if g["qualifies"] else "-"
        print(f"  {opp:<32s} all_seeds_negative={str(g['all_seeds_negative']):<5s} "
              f"UCB95<0={str(g['ucb95_strictly_negative']):<5s} {mark}")

    print()
    selected = summary["selected_c1"]
    if summary["competence"] != "COMPETENT":
        print(f"C1 SELECTION BLOCKED by competence={summary['competence']}.")
        print("Discovery qualifiers (if any) are informational only:")
        for o in summary["qualified"][:5]:
            r = results[o]
            print(f"  {o:<32s} W={r['W']:+.4f} family={r['family_mean']:+.4f}")
    elif selected:
        ranked = [
            o for o in order if results[o]["gate"]["qualifies"]
        ]
        ranked = sorted(
            ranked,
            key=lambda o: (results[o]["W"], results[o]["family_mean"],
                           results[o]["saturation"] + results[o]["tie"]),
        )
        if len(ranked) > 1:
            print(f"{len(ranked)} contexts qualify; ranked by strongest-member margin, "
                  f"then family mean, then saturation+tie:")
            for o in ranked:
                r = results[o]
                print(f"  {o:<32s} W={r['W']:+.4f} family={r['family_mean']:+.4f} "
                      f"sat+tie={r['saturation'] + r['tie']:.3f}")
        print(f"\nDISCOVERY CANDIDATE C1 = {selected}")
        print(f"  W(c) = {results[selected]['W']:+.4f}  "
              f"family = {results[selected]['family_mean']:+.4f} "
              f"CI95=[{results[selected]['lo']:+.4f}, {results[selected]['hi']:+.4f}]")
        print("  This 32-seed sweep is DISCOVERY only.")
        print("  Do NOT train O1 yet. Run the 64-seed confirmation block on every")
        print("  discovery qualifier (docs/g0-c1-confirmation-preregistration.md).")
    else:
        print("NO DISCOVERY CANDIDATE: competence is COMPETENT but no context clears")
        print("the strict weakness gate.")
        print("\nNearest by W(c) (lower = closer; NOT auto-selected):")
        for opp in order[:3]:
            r = results[opp]
            print(f"  {opp:<32s} W={r['W']:+.4f} family={r['family_mean']:+.4f} "
                  f"CI95=[{r['lo']:+.4f}, {r['hi']:+.4f}]")
        print("\nDo NOT relax the gate. Next: a separately declared OP6–OP12 variant")
        print("sweep in its own output directory, if still needed after confirmation.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
