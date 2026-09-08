"""D2 -- evaluate recovery at each rung. Implements D2_SPEC_FROZEN.json +
D2_SPEC_AMENDMENT_2.json.

Metric is Gate 0B's OWN original convention, unchanged: for each rung's fitted
Q_psi, on INNER_VAL's own_flag_stolen pole-B subset,

    d_B(o) = V_hat(o, pi_B, B) - V_hat(o, pi_A, B)

querying the two FROZEN SAPPO TEACHERS directly via teacher_action_dists() --
no student model, closed or otherwise, is touched anywhere in this file.

Seed-level bootstrap (n_boot 20000, alpha 0.05, rng_seed 7), same convention as
Gate 0B / D0 / D1. Applies the frozen monotonicity/meaningful-margin rule
(D2_SPEC_AMENDMENT_1) rather than reading raw point estimates.

Run:  python experiments/d2_evaluate_recovery.py --device cuda
"""
from __future__ import annotations

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

from experiments.phase0_scorer_common import (            # noqa: E402
    INNER_VAL_SEEDS, load_split, teacher_action_dists,
)
from experiments.d2_fit_density_ablation import OUT_DIR as FIT_DIR  # noqa: E402
from rl.scorer.qpsi import QPsi, QPsiConfig                # noqa: E402

SPPO = ROOT / "artifacts/strategic_demand/sppo"
OUT = SPPO / "D2_RESULT.json"
QPSI_MARGIN = 0.04
N_BOOT, ALPHA, RNG_SEED = 20_000, 0.05, 7


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ci(draws: np.ndarray) -> dict:
    lo, hi = np.percentile(draws, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"mean": float(draws.mean()), "lcb95": float(lo), "ucb95": float(hi)}


def _bootstrap(hits_by_seed: dict, n_by_seed: dict, seeds: list) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, len(seeds), size=(N_BOOT, len(seeds)))
    hits = np.array([hits_by_seed[s] for s in seeds])
    ns = np.array([n_by_seed[s] for s in seeds])
    draws = np.array([hits[idx[b]].sum() / max(1, ns[idx[b]].sum()) for b in range(N_BOOT)])
    return _ci(draws)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu"

    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; D2 evaluation is one-shot")

    from rl.custom_ppo import load_custom_ppo_policy
    import experiments.r2_learned_crossover as R2
    import experiments.phase0_collect_scorer_data as P0

    probe = R2.build_env(device, INNER_VAL_SEEDS[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    teachers = {k: load_custom_ppo_policy(str(v), obs_space, act_space, device=device)
               for k, v in P0.TEACHERS.items()}

    # load_split does NOT propagate plain-row action masks, and
    # teacher_action_dists REQUIRES them -- querying unmasked distributions is the
    # exact Phase 0 defect that produced a sign-inverted diagnostic. So the plain
    # rows are read straight from the shards here, masks included, rather than
    # modifying the shared loader the frozen fit depends on.
    import glob as _glob
    from experiments.phase0_scorer_common import COLL as _COLL
    tagger = QPsi(QPsiConfig())
    P = {k: [] for k in ("grid", "vec", "amask", "mask", "pole", "seed")}
    for seed in INNER_VAL_SEEDS:
        z = np.load(_COLL / "seed_shards" / f"seed_{seed}.npz", allow_pickle=True)
        sel = z["plain_pole"] == 1
        if not sel.any():
            continue
        P["grid"].append(z["plain_obs_grid"][sel][:, 0])
        P["vec"].append(z["plain_obs_vec"][sel][:, 0])
        P["amask"].append(z["plain_obs_agent_mask"][sel][:, 0])
        P["mask"].append(z["plain_obs_mask"][sel][:, 0])
        P["pole"].append(z["plain_pole"][sel].astype(np.int64))
        P["seed"].append(np.full(int(sel.sum()), seed, dtype=np.int64))
    P = {k: np.concatenate(v) for k, v in P.items()}
    regime = tagger.regime_from_vec(torch.as_tensor(P["vec"], dtype=torch.float32)).numpy()
    keep = np.nonzero(regime >= 2)[0]        # already restricted to pole B above
    if len(keep) == 0:
        raise SystemExit("REFUSING: no INNER_VAL stolen pole-B states found")
    print(f"D2 EVALUATION  {_now()}")
    print(f"  INNER_VAL own_flag_stolen pole-B states: {len(keep)}"
          f" across {len(set(P['seed'][keep].tolist()))} seeds\n")

    t = lambda a, dt: torch.as_tensor(a, dtype=dt)
    grid, vecs = t(P["grid"][keep], torch.float32), t(P["vec"][keep], torch.float32)
    am, mk = t(P["amask"][keep], torch.float32), t(P["mask"][keep], torch.float32)
    pole = t(P["pole"][keep], torch.long)
    seeds_here = P["seed"][keep]

    results = {}
    for rung in (1, 2, 3):
        wpath = FIT_DIR / f"qpsi_rung{rung}.pt"
        ck = torch.load(wpath, map_location=device, weights_only=False)
        model = QPsi(QPsiConfig(**ck["config"])).to(device)
        model.load_state_dict(ck["state_dict"]); model.eval()

        dB = []
        with torch.no_grad():
            for i in range(0, len(keep), 512):
                s = slice(i, i + 512)
                g, v, a_, m, p = (grid[s].to(device), vecs[s].to(device), am[s].to(device),
                                  mk[s].to(device), pole[s].to(device))
                p1_A, p2_A = teacher_action_dists(teachers["pi_A"], g, v, a_, m)
                p1_B, p2_B = teacher_action_dists(teachers["pi_B"], g, v, a_, m)
                vA = model.expected_value(g, v, a_, p, p1_A, p2_A)
                vB = model.expected_value(g, v, a_, p, p1_B, p2_B)
                dB.append((vB - vA).cpu().numpy())
        dB = np.concatenate(dB)
        correct = (dB > QPSI_MARGIN).astype(np.float64)

        by_seed_hits, by_seed_n = defaultdict(float), defaultdict(int)
        for s, c in zip(seeds_here, correct):
            by_seed_hits[s] += c; by_seed_n[s] += 1
        seeds = sorted(by_seed_n)
        rate_ci = _bootstrap(by_seed_hits, by_seed_n, seeds)
        results[str(rung)] = {
            "n_states": len(keep), "n_seeds": len(seeds),
            "d_B_mean": float(dB.mean()),
            "qpsi_correct_rate": rate_ci,
        }
        print(f"  rung {rung}: correct-rate {rate_ci['mean']:.3f} "
              f"[{rate_ci['lcb95']:.3f}, {rate_ci['ucb95']:.3f}]  (d_B mean {dB.mean():+.4f})")

    r1, r2, r3 = (results[str(k)]["qpsi_correct_rate"] for k in (1, 2, 3))
    monotonic = r1["mean"] <= r2["mean"] <= r3["mean"]
    clears = r3["lcb95"] > r1["mean"]     # rung_3's LCB95 must clear rung_1's point estimate
    if monotonic and clears:
        verdict = "ABSOLUTE_DENSITY_DOMINANT"
    elif r3["ucb95"] < 0.5 or abs(r3["mean"] - r1["mean"]) < 0.05:
        verdict = "REPRESENTATION_CAPACITY_LIMITED"
    else:
        verdict = "INCONCLUSIVE"

    rec = {
        "record": "D2 recovery evaluation result", "status": "DIAGNOSTIC_ONLY", "utc": _now(),
        "metric": "d_B(o) = V_hat(o,pi_B,B) - V_hat(o,pi_A,B), teacher-vs-teacher, Gate 0B convention",
        "eval_set": "INNER_VAL own_flag_stolen pole-B subset -- reused development-side diagnostic resource, NOT held-out confirmation, NOT a gate",
        "bootstrap": {"unit": "seed", "n_boot": N_BOOT, "alpha": ALPHA, "rng_seed": RNG_SEED},
        "rungs": results,
        "monotonic_50_100_200": bool(monotonic),
        "rung3_LCB95_clears_rung1_point_estimate": bool(clears),
        "VERDICT": verdict,
    }
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\n  monotonic: {monotonic}   rung3 LCB95 clears rung1 point estimate: {clears}")
    print(f"  VERDICT: {verdict}")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
