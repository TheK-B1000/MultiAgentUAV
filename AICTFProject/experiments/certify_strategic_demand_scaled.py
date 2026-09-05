"""Strategic-demand certification at a given team size.

Implements SIZE_NORMALIZED_POLE_SEMANTICS_SPEC.json#CERTIFICATION_PROTOCOL.

Asks the same question the 2v2 certification asked, with the size-normalized probe and
pole gate: do the frozen poles demand OPPOSITE strategic responses?

    delta_A = WR(GUARD, pole A) - WR(BREACH, pole A)      GUARD should win on the wide zone
    delta_B = WR(BREACH, pole B) - WR(GUARD, pole B)      BREACH should win on the tight one

CERTIFIED iff LCB95(delta_A) > 0 AND LCB95(delta_B) > 0 -- the same rule and the same
bootstrap (n_boot=20000, alpha=0.05, rng_seed=7, seed as the resampling unit) used
everywhere else in this project. No threshold is adjusted for team size.

GUARD and BREACH are run on the SAME seed, so the comparison is paired per seed.

No policy is trained, loaded or updated: both blue styles are scripted.

Run:  python experiments/certify_strategic_demand_scaled.py --team-size 4 --n-seeds 64 --device cpu
      python experiments/certify_strategic_demand_scaled.py --team-size 4 --n-seeds 2 --probe
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "artifacts" / "strategic_demand" / "sppo"

# Certification seed blocks, disjoint per team size and from every prior block.
SEED_BASE = {4: 12_400_001, 6: 12_600_001}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mean_ci(vals, n_boot=20000, alpha=0.05, rng_seed=7):
    a = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boots = a[idx].mean(axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {"mean": float(a.mean()), "lcb95": lo, "ucb95": hi, "n": int(a.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-size", type=int, required=True, choices=(2, 4, 6))
    ap.add_argument("--n-seeds", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--probe", action="store_true",
                    help="timing probe only; writes no certification record")
    args = ap.parse_args()

    N = int(args.team_size)
    n_seeds = int(args.n_seeds)

    import experiments.strategic_demand_searcher as S
    from experiments.opponent_spec import expected_profile, pole_A_genome, pole_B_genome

    # Team size must reach the scripted-episode env: it reads this module global.
    S.AGENTS = N
    if int(S.AGENTS) != N:
        raise SystemExit("FAIL-CLOSED: could not set team size on the episode runner")

    gA, gB = pole_A_genome(N), pole_B_genome(N)
    pA, pB = expected_profile("OP6", gA), expected_profile("OP7", gB)

    # Fail closed if the size normalization did not reach the resolved profiles.
    for tag, prof in (("A", pA), ("B", pB)):
        got = int(getattr(prof, "min_alive_for_defender", -1))
        if got != N:
            raise SystemExit(f"FAIL-CLOSED: pole {tag} resolved min_alive_for_defender={got}, "
                             f"expected {N} under the size-normalized semantics")

    print("=" * 78)
    print(f"STRATEGIC DEMAND CERTIFICATION  {N}v{N}   device={args.device}")
    print("=" * 78)
    print(f"  utc            {_now()}")
    print(f"  seeds          {SEED_BASE[N] if N in SEED_BASE else 'PROBE'}"
          f"..+{n_seeds}   (paired: GUARD and BREACH share each seed)")
    print(f"  GUARD          {(N + 1) // 2} of {N} defend  (indices "
          f"{list(range(N - (N + 1) // 2, N))})")
    print(f"  pole A         OP6+overlay  min_alive={getattr(pA, 'min_alive_for_defender', None)}"
          f"  zone_frac={getattr(pA, 'defender_zone_frac', None)}"
          f"  threat_radius={getattr(pA, 'threat_radius', None)}")
    print(f"  pole B         OP7+overlay  min_alive={getattr(pB, 'min_alive_for_defender', None)}"
          f"  zone_frac={getattr(pB, 'defender_zone_frac', None)}"
          f"  threat_radius={getattr(pB, 'threat_radius', None)}")
    print(f"  gate           LCB95(delta_A) > 0 AND LCB95(delta_B) > 0")
    print("=" * 78, flush=True)

    base = SEED_BASE.get(N, 99_990_001)
    rows = []
    t0 = time.time()
    for i in range(n_seeds):
        seed = base + i
        r = {"seed": seed}
        for pole, genome in (("A", gA), ("B", gB)):
            for style_tag, style in (("guard", S.GUARD), ("breach", S.BREACH)):
                ep = S.run_episode(style=style, genome=genome, seed=seed, device=args.device)
                r[f"{pole}_{style_tag}"] = int(ep["win"])
        rows.append(r)
        if (i + 1) % 8 == 0 or i == 0:
            el = time.time() - t0
            print(f"  seed {i + 1}/{n_seeds}  elapsed {el:6.1f}s  "
                  f"({el / (i + 1):.2f}s/seed, 4 episodes each)", flush=True)

    dA = np.array([r["A_guard"] - r["A_breach"] for r in rows], dtype=float)
    dB = np.array([r["B_breach"] - r["B_guard"] for r in rows], dtype=float)
    ciA, ciB = _mean_ci(dA), _mean_ci(dB)
    certified = bool(ciA["lcb95"] > 0 and ciB["lcb95"] > 0)

    cells = {
        "poleA_guard_wr": float(np.mean([r["A_guard"] for r in rows])),
        "poleA_breach_wr": float(np.mean([r["A_breach"] for r in rows])),
        "poleB_guard_wr": float(np.mean([r["B_guard"] for r in rows])),
        "poleB_breach_wr": float(np.mean([r["B_breach"] for r in rows])),
    }

    print("\n  cell win rates")
    for k, v in cells.items():
        print(f"    {k:20s} {v:.4f}")
    print("\n  PRIMARY")
    print(f"    delta_A (GUARD-BREACH on A) {ciA['mean']:+.4f} "
          f"[{ciA['lcb95']:+.4f}, {ciA['ucb95']:+.4f}]")
    print(f"    delta_B (BREACH-GUARD on B) {ciB['mean']:+.4f} "
          f"[{ciB['lcb95']:+.4f}, {ciB['ucb95']:+.4f}]")
    print(f"\n  {N}v{N} STRATEGIC DEMAND: {'CERTIFIED' if certified else 'NOT CERTIFIED'}")

    if args.probe:
        print("\n  --probe: no record written.")
        return 0

    out = OUT_DIR / f"STRATEGIC_DEMAND_{N}v{N}_CERTIFICATION.json"
    if out.exists():
        raise SystemExit(f"REFUSING: {out.name} exists; certification is one-shot")
    out.write_text(json.dumps({
        "record": f"Strategic demand certification {N}v{N}",
        "status": "FROZEN_RESULT", "one_shot": True, "utc": _now(),
        "implements": "SIZE_NORMALIZED_POLE_SEMANTICS_SPEC.json#CERTIFICATION_PROTOCOL",
        "team_size": N, "device": args.device,
        "guard_defenders": (N + 1) // 2,
        "guard_defender_indices": list(range(N - (N + 1) // 2, N)),
        "seeds": {"base": base, "n": n_seeds, "paired": True},
        "poles": {
            "A": {"base": "OP6", "overlay": dict(gA.overlay or {}),
                  "min_alive_for_defender": getattr(pA, "min_alive_for_defender", None),
                  "defender_zone_frac": getattr(pA, "defender_zone_frac", None),
                  "threat_radius": getattr(pA, "threat_radius", None)},
            "B": {"base": "OP7", "overlay": dict(gB.overlay or {}),
                  "min_alive_for_defender": getattr(pB, "min_alive_for_defender", None),
                  "defender_zone_frac": getattr(pB, "defender_zone_frac", None),
                  "threat_radius": getattr(pB, "threat_radius", None)},
        },
        "cell_win_rates": cells,
        "PRIMARY": {"delta_A": ciA, "delta_B": ciB},
        "bootstrap": {"procedure": "paired percentile bootstrap over seeds",
                      "samples": 20000, "alpha": 0.05, "rng_seed": 7},
        "gate": "LCB95(delta_A) > 0 AND LCB95(delta_B) > 0",
        "VERDICT": "CERTIFIED" if certified else "NOT_CERTIFIED",
        "rows": rows,
        "total_episodes": 4 * n_seeds,
        "note_if_not_certified": (
            "SIZE_NORMALIZED_POLE_SEMANTICS_SPEC.json#NORMALIZATION_1_GUARD_PROBE records that "
            "multiple defenders CONVERGE on a single threat rather than covering distinct "
            "intruders. That stacking is a candidate explanation for a failure at N>2 and must "
            "be acknowledged before concluding the poles do not demand different behaviour."),
    }, indent=2), encoding="utf-8")
    print(f"  -> {out}")
    return 0 if certified else 1


if __name__ == "__main__":
    raise SystemExit(main())
