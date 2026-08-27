"""D1A -- Phase-0 scorer support audit. Implements D1_SPEC_FROZEN.json.

Measures how much representation Q_psi's OWN fitting data had of the state
categories D0 flagged (own_flag_stolen, own_flag_home, carrying, not_carrying),
by pole and by seed, for plain rows and branch rows separately.

NO NEW SEEDS. Reads only the 256 already-collected Phase 0 shards that Q_psi was
actually fit on. Tagging is QPsi.regime_from_vec(), verified against live core
ground truth (0/240 mismatches) before this spec was frozen.

Reports counts and proportions with seed-level bootstrap CIs. Invents no
retrospective off-support threshold.

Run:  python experiments/d1a_scorer_support_audit.py
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COLL = ROOT / "artifacts/strategic_demand/phase0_scorer_data/full_collection_rebuild_per_branch"
SPPO = ROOT / "artifacts/strategic_demand/sppo"
OUT = SPPO / "D1A_SCORER_SUPPORT_AUDIT.json"
SPEC = SPPO / "D1_SPEC_FROZEN.json"

N_BOOT, ALPHA, RNG_SEED = 20_000, 0.05, 7
REGIME_NAME = {0: "home_not_carrying", 1: "home_carrying",
              2: "stolen_not_carrying", 3: "stolen_carrying"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ci(draws: np.ndarray) -> dict:
    lo, hi = np.percentile(draws, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"mean": float(draws.mean()), "lcb95": float(lo), "ucb95": float(hi)}


def _bootstrap_proportion(hits_by_seed: dict, n_by_seed: dict, seeds: list) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, len(seeds), size=(N_BOOT, len(seeds)))
    draws = np.empty(N_BOOT)
    hits = np.array([hits_by_seed[s] for s in seeds])
    ns = np.array([n_by_seed[s] for s in seeds])
    for b in range(N_BOOT):
        picked = idx[b]
        draws[b] = hits[picked].sum() / max(1, ns[picked].sum())
    return _ci(draws)


def main() -> int:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_ANY_D1_QUANTITY_IS_COMPUTED":
        raise SystemExit("REFUSING: D1 spec is not in the expected frozen state")
    bs = spec["D1A_phase0_scorer_support_audit"]["bootstrap"]
    if (bs["n_boot"], bs["alpha"], bs["rng_seed"]) != (N_BOOT, ALPHA, RNG_SEED):
        raise SystemExit("REFUSING: bootstrap params drifted from the frozen D1A spec")

    from rl.scorer.qpsi import QPsi, QPsiConfig
    m = QPsi(QPsiConfig())

    shards = sorted(glob.glob(str(COLL / "seed_shards" / "*.npz")))
    if len(shards) != 256:
        raise SystemExit(f"REFUSING: found {len(shards)} shards, expected 256")

    # counts[population][pole][regime][seed] = n
    counts = {pop: {pole: defaultdict(lambda: defaultdict(int)) for pole in (0, 1)}
             for pop in ("plain", "branch")}
    totals = {pop: {pole: defaultdict(int) for pole in (0, 1)} for pop in ("plain", "branch")}
    seeds_seen = []

    for i, path in enumerate(shards):
        seed = int(Path(path).stem.split("seed_")[-1])
        seeds_seen.append(seed)
        z = np.load(path, allow_pickle=True)

        pv = torch.as_tensor(z["plain_obs_vec"][:, 0], dtype=torch.float32)
        pr = m.regime_from_vec(pv).numpy()
        pp = z["plain_pole"]
        for pole in (0, 1):
            mask = pp == pole
            for r in range(4):
                counts["plain"][pole][r][seed] += int(((pr == r) & mask).sum())
            totals["plain"][pole][seed] += int(mask.sum())

        bv = torch.as_tensor(z["branch_obs_vec"][:, 0], dtype=torch.float32)
        br = m.regime_from_vec(bv).numpy()
        bp = z["branch_pole"]
        for pole in (0, 1):
            mask = bp == pole
            for r in range(4):
                counts["branch"][pole][r][seed] += int(((br == r) & mask).sum())
            totals["branch"][pole][seed] += int(mask.sum())

        if (i + 1) % 64 == 0:
            print(f"  {i+1}/256 shards scanned", flush=True)

    result = {}
    for pop in ("plain", "branch"):
        result[pop] = {}
        for pole_i, pole_name in ((0, "A"), (1, "B")):
            result[pop][pole_name] = {}
            seeds = [s for s in seeds_seen if totals[pop][pole_i][s] > 0]
            n_by_seed = {s: totals[pop][pole_i][s] for s in seeds}
            n_total = sum(n_by_seed.values())
            for r in range(4):
                hits_by_seed = {s: counts[pop][pole_i][r][s] for s in seeds}
                n_hits = sum(hits_by_seed.values())
                ci = (_bootstrap_proportion(hits_by_seed, n_by_seed, seeds)
                     if seeds else {"mean": float("nan"), "lcb95": float("nan"), "ucb95": float("nan")})
                result[pop][pole_name][REGIME_NAME[r]] = {
                    "n": n_hits, "n_total_in_population": n_total,
                    "n_seeds_with_any_data": len(seeds),
                    "proportion": ci,
                }
            # marginal views
            for label, rset in (("own_flag_home", (0, 1)), ("own_flag_stolen", (2, 3)),
                               ("carrying", (1, 3)), ("not_carrying", (0, 2))):
                hits_by_seed = {s: sum(counts[pop][pole_i][r][s] for r in rset) for s in seeds}
                n_hits = sum(hits_by_seed.values())
                ci = (_bootstrap_proportion(hits_by_seed, n_by_seed, seeds)
                     if seeds else {"mean": float("nan"), "lcb95": float("nan"), "ucb95": float("nan")})
                result[pop][pole_name][f"marginal_{label}"] = {
                    "n": n_hits, "n_total_in_population": n_total, "proportion": ci,
                }

    rec = {
        "record": "D1A Phase-0 scorer support audit",
        "status": "DIAGNOSTIC_ONLY -- implements D1_SPEC_FROZEN.json",
        "utc": _now(),
        "n_shards_scanned": len(shards),
        "tagging": "QPsi.regime_from_vec, verified 0/240 mismatches vs live core ground truth",
        "bootstrap": {"unit": "seed", "n_boot": N_BOOT, "alpha": ALPHA, "rng_seed": RNG_SEED},
        "by_population": result,
    }
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    print(f"\nD1A RESULT  {_now()}")
    for pop in ("plain", "branch"):
        print(f"\n  population: {pop}")
        for pole in ("A", "B"):
            print(f"    pole {pole}:")
            for k in ("marginal_own_flag_stolen", "marginal_carrying"):
                v = result[pop][pole][k]
                p = v["proportion"]
                print(f"      {k:26s} n={v['n']:6d}/{v['n_total_in_population']:6d}  "
                      f"prop {p['mean']:.4f} [{p['lcb95']:.4f}, {p['ucb95']:.4f}]")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
