#!/usr/bin/env python3
"""Rank OP6-OP12 contexts by where the LEARNED incumbent G0 actually fails.

Consumes episode_rows.csv from run_g0_weakness_sweep.py.

RANKING RULE (predeclared)

    W(c) = max over G0 seeds of payoff(G0_s, c)      lower is better

The MAXIMUM, not the mean: a context that only defeats one weak member of the
incumbent family is not a weakness of the family. Ranking by the strongest
seed forces the selected context to trouble the whole incumbent.

WEAKNESS GATE (predeclared, strict)

    all three G0 seeds have negative mean margin
    AND the family-level upper CI95 is below zero

If nothing meets that gate, SELECT NOTHING. Report the nearest weaknesses and
decide separately whether to engineer an allowed OP6-OP12 opponent or run a
separately declared map-specific search. Do not relax the gate to manufacture
a candidate -- that is exactly the error that produced the failed OP11/OP9
specialist attempt.

map_a results are never pooled with map_b.
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
# |margin| at or above this counts as a saturated (blowout) episode.
SATURATION_MARGIN = 3


ROW_KEY = ("checkpoint_step", "g0_seed", "opponent", "map", "episode_seed")


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


def boot_ci(x: np.ndarray, rng, n_boot: int, alpha: float):
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    b = x[idx].mean(axis=1)
    lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def hier_ci(arr: np.ndarray, rng, n_boot: int, alpha: float):
    """Resample G0 seeds and episodes -- family-level uncertainty."""
    n_s, n_e = arr.shape
    b = np.empty(n_boot)
    for i in range(n_boot):
        si = rng.integers(0, n_s, n_s)
        ei = rng.integers(0, n_e, n_e)
        b[i] = arr[np.ix_(si, ei)].mean()
    lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)


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
    print("=" * 84)
    print(f"G0 LEARNED-INCUMBENT WEAKNESS SWEEP -- {map_label} ({map_layout})")
    print("G0 = frozen 1M piR family, used as incumbent generalist.")
    print("Ranking: W(c) = max over G0 seeds of payoff; LOWER = better weakness.")
    print("=" * 84)

    results = {}
    for opp in sorted(data):
        ep, seeds, arr = paired_matrix(data[opp])
        per_seed_mean = arr.mean(axis=1)
        fam_mean, fam_lo, fam_hi = hier_ci(arr, rng, args.n_boot, args.alpha)

        flat = arr.reshape(-1)
        wins = float(np.mean(flat > 0))
        ties = float(np.mean(flat == 0))
        losses = float(np.mean(flat < 0))
        sat = float(np.mean(np.abs(flat) >= SATURATION_MARGIN))

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
            "seeds": seeds, "per_seed_mean": per_seed_mean,
            "family_mean": fam_mean, "lo": fam_lo, "hi": fam_hi,
            "W": float(per_seed_mean.max()),
            "win": wins, "tie": ties, "loss": losses, "saturation": sat,
            "behavior": beh, "n_ep": len(ep),
        }

    order = sorted(results, key=lambda o: results[o]["W"])

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

    print("\nTrajectory fingerprints (mean behavior telemetry)")
    keys = ["team_spread", "num_attackers", "num_defenders", "avg_blue_to_enemy_flag",
            "avg_blue_to_own_flag", "intercept_pressure", "defense_pressure"]
    print(f"{'opponent':<32s}" + "".join(f"{k[:11]:>12s}" for k in keys))
    for opp in order:
        b = results[opp]["behavior"]
        print(f"{opp:<32s}" + "".join(f"{b[k]:>12.3f}" for k in keys))

    # ---- strict weakness gate -------------------------------------------
    print(f"\n{'=' * 84}")
    print("WEAKNESS GATE: all three G0 seeds negative AND family upper CI95 < 0")
    print("=" * 84)
    qualified = [o for o in order
                 if (results[o]["per_seed_mean"] < 0).all() and results[o]["hi"] < 0]
    for opp in order:
        r = results[opp]
        a = bool((r["per_seed_mean"] < 0).all())
        b = bool(r["hi"] < 0)
        mark = "QUALIFIES" if (a and b) else "-"
        print(f"  {opp:<32s} all_seeds_negative={str(a):<5s} upper_CI<0={str(b):<5s} {mark}")

    print()
    if qualified:
        # Predeclared tiebreak: (1) lowest strongest-member margin, (2) lowest
        # family mean margin, (3) lower saturation and tie rate.
        ranked = sorted(qualified, key=lambda o: (results[o]["W"],
                                                  results[o]["family_mean"],
                                                  results[o]["saturation"] + results[o]["tie"]))
        if len(ranked) > 1:
            print(f"{len(ranked)} contexts qualify; ranked by strongest-member margin, "
                  f"then family mean, then saturation+tie:")
            for o in ranked:
                r = results[o]
                print(f"  {o:<32s} W={r['W']:+.4f} family={r['family_mean']:+.4f} "
                      f"sat+tie={r['saturation'] + r['tie']:.3f}")
        best = ranked[0]
        print(f"\nSELECTED WEAKNESS C1 = {best}")
        print(f"  W(c) = {results[best]['W']:+.4f}  family = {results[best]['family_mean']:+.4f} "
              f"CI95=[{results[best]['lo']:+.4f}, {results[best]['hi']:+.4f}]")
        print("  Next: freeze G0, train a fresh response-oracle family O1 against this")
        print("  context, then evaluate G0 and O1 across the frozen context pool and apply")
        print("  the retention rule. Birth a latent branch ONLY after positive repertoire gain.")
    else:
        print("BASE SWEEP CLOSED: NO ROBUST WEAKNESS.")
        print("No base OP6-OP12 context on this map defeats the whole G0 family.")
        print("The least-good opponent is deliberately NOT auto-selected.")
        print("\nNearest candidates by W(c) (lower = closer to a weakness):")
        for opp in order[:3]:
            r = results[opp]
            print(f"  {opp:<32s} W={r['W']:+.4f} family={r['family_mean']:+.4f} "
                  f"CI95=[{r['lo']:+.4f}, {r['hi']:+.4f}]")
        print("\nDo NOT relax the gate to manufacture a candidate -- selecting a context")
        print("that G0 already handles is what produced the failed OP11/OP9 attempt.")
        print("Next step, to be declared before running: a SEPARATE existing-variant")
        print("sweep within OP6-OP12 (e.g. OP11_ADAPTIVE_EXPLOITER, OP9_SPLIT_LANE_FEINT,")
        print("OP6_TURTLE, OP7_DEEP_FORTRESS, ...), in its own output directory.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
