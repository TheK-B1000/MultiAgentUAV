"""D4 -- Target Resolution Audit. Implements D4_SPEC_FROZEN.json.

Asks whether D3's 91% terminal-margin tie rate is SPECIFIC to own_flag_stolen or
GENERAL to terminal win margin at the state level.

Compares pole-B paired branch states across the four flag/carrying regimes,
reporting per-regime tie rate and, among non-tied states only, which teacher is
favoured. Seed-level bootstrap throughout.

The D2 supplement is EXCLUDED from the comparison -- it is 100% stolen-flag by
construction, so including it would inflate only the stolen cell and bias the
exact contrast this audit measures. Phase 0 original pairs are the only source
spanning all four regimes. The supplement is reported for the stolen cell alone,
clearly labelled, never pooled.

Existing stored outcomes only. No environment steps, no new seeds, no FINAL
access, no interference with the running RASR collector.

Run:  python experiments/d4_target_resolution_audit.py
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

SD = ROOT / "artifacts/strategic_demand"
SPPO = SD / "sppo"
PHASE0 = SD / "phase0_scorer_data/full_collection_rebuild_per_branch/seed_shards"
D2_SUP = SPPO / "d2_density/supplement_shards"
SPEC = SPPO / "D4_SPEC_FROZEN.json"
OUT = SPPO / "D4_RESULT.json"

N_BOOT, ALPHA, RNG_SEED = 20_000, 0.05, 7
POLE_B = 1
REGIME_NAME = {0: "home_not_carrying", 1: "home_carrying",
              2: "stolen_not_carrying", 3: "stolen_carrying"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ci(draws: np.ndarray) -> dict:
    lo, hi = np.percentile(draws, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"mean": float(draws.mean()), "lcb95": float(lo), "ucb95": float(hi)}


def _seed_bootstrap(num_by_seed: dict, den_by_seed: dict) -> dict:
    """Ratio statistic with SEED as the resampling unit."""
    seeds = sorted(den_by_seed)
    if not seeds:
        return {"mean": float("nan"), "lcb95": float("nan"), "ucb95": float("nan")}
    num = np.array([num_by_seed.get(s, 0.0) for s in seeds], dtype=np.float64)
    den = np.array([den_by_seed[s] for s in seeds], dtype=np.float64)
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, len(seeds), size=(N_BOOT, len(seeds)))
    d = den[idx].sum(axis=1)
    draws = np.divide(num[idx].sum(axis=1), d, out=np.full(N_BOOT, np.nan), where=d > 0)
    draws = draws[~np.isnan(draws)]
    return _ci(draws) if len(draws) else {"mean": float("nan"), "lcb95": float("nan"), "ucb95": float("nan")}


def _load(shard_paths):
    """Per-seed (regime, delta) for pole-B paired branch states."""
    rows: dict[int, list] = defaultdict(list)
    from rl.scorer.qpsi import QPsi, QPsiConfig
    tagger = QPsi(QPsiConfig())
    for path in shard_paths:
        seed = int(Path(path).stem.split("seed_")[-1])
        z = np.load(path, allow_pickle=True)
        if "branch_pole" not in z.files:
            continue
        pole = z["branch_pole"]
        vec = z["branch_obs_vec"]
        vec = vec[:, 0] if vec.ndim == 4 else vec
        regime = tagger.regime_from_vec(torch.as_tensor(vec, dtype=torch.float32)).numpy()
        sel = pole == POLE_B
        if not sel.any():
            continue
        mB = (z["branch_pi_B_blue"][sel].astype(np.int32)
              - z["branch_pi_B_red"][sel].astype(np.int32))
        mA = (z["branch_pi_A_blue"][sel].astype(np.int32)
              - z["branch_pi_A_red"][sel].astype(np.int32))
        for r, d in zip(regime[sel], (mB - mA)):
            rows[seed].append((int(r), float(d)))
    return rows


def _stratum(rows: dict, keep_regimes: set, label: str) -> dict:
    tie_num, tie_den = {}, {}
    piB_num, piA_num, nontied_den = {}, {}, {}
    n_states = n_tied = n_piB = n_piA = 0
    for seed, items in rows.items():
        sel = [(r, d) for r, d in items if r in keep_regimes]
        if not sel:
            continue
        tie_den[seed] = len(sel)
        tie_num[seed] = sum(1 for _, d in sel if d == 0.0)
        nt = [d for _, d in sel if d != 0.0]
        if nt:
            nontied_den[seed] = len(nt)
            piB_num[seed] = sum(1 for d in nt if d > 0)
            piA_num[seed] = sum(1 for d in nt if d < 0)
        n_states += len(sel); n_tied += tie_num[seed]
        n_piB += sum(1 for d in nt if d > 0); n_piA += sum(1 for d in nt if d < 0)
    if n_states == 0:
        return {"label": label, "n_states": 0}
    return {
        "label": label,
        "n_states": n_states, "n_seeds": len(tie_den),
        "n_tied": n_tied, "n_informative": n_states - n_tied,
        "tie_rate": _seed_bootstrap(tie_num, tie_den),
        "among_non_tied": {
            "n": n_states - n_tied,
            "frac_favouring_piB": _seed_bootstrap(piB_num, nontied_den) if nontied_den else None,
            "frac_favouring_piA": _seed_bootstrap(piA_num, nontied_den) if nontied_den else None,
            "raw_piB": n_piB, "raw_piA": n_piA,
        },
    }


def main() -> int:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_ANY_D4_QUANTITY_IS_COMPUTED":
        raise SystemExit("REFUSING: D4 spec is not in the expected pre-computation state")
    bs = spec["bootstrap"]
    if (bs["n_boot"], bs["alpha"], bs["rng_seed"]) != (N_BOOT, ALPHA, RNG_SEED):
        raise SystemExit("REFUSING: bootstrap params drifted from the frozen D4 spec")
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; D4 is one-shot")

    print(f"D4 TARGET RESOLUTION AUDIT  {_now()}")
    print("  pole-B paired branch states, Phase 0 original (all four regimes)\n")
    rows = _load(sorted(glob.glob(str(PHASE0 / "*.npz"))))

    four_way = {REGIME_NAME[r]: _stratum(rows, {r}, REGIME_NAME[r]) for r in range(4)}
    marginals = {
        "own_flag_stolen": _stratum(rows, {2, 3}, "own_flag_stolen"),
        "own_flag_home": _stratum(rows, {0, 1}, "own_flag_home"),
        "carrying": _stratum(rows, {1, 3}, "carrying"),
        "not_carrying": _stratum(rows, {0, 2}, "not_carrying"),
        "ALL_pole_B": _stratum(rows, {0, 1, 2, 3}, "ALL pole-B states"),
    }

    def show(d):
        if not d.get("n_states"):
            print(f"    {d['label']:22s} (no states)"); return
        t = d["tie_rate"]
        print(f"    {d['label']:22s} n={d['n_states']:5d} seeds={d['n_seeds']:3d}"
              f"  tie {t['mean']:.3f} [{t['lcb95']:.3f}, {t['ucb95']:.3f}]"
              f"  informative={d['n_informative']:4d}")

    print("  MARGINAL VIEWS")
    for k in ("ALL_pole_B", "own_flag_home", "own_flag_stolen", "not_carrying", "carrying"):
        show(marginals[k])
    print("\n  FOUR-WAY")
    for k in ("home_not_carrying", "home_carrying", "stolen_not_carrying", "stolen_carrying"):
        show(four_way[k])

    sup_rows = _load(sorted(glob.glob(str(D2_SUP / "*.npz"))))
    sup_stolen = _stratum(sup_rows, {2, 3}, "D2 supplement, stolen only (NOT pooled)")
    print("\n  SECONDARY, reported alone and never pooled:")
    show(sup_stolen)

    stolen, home = marginals["own_flag_stolen"]["tie_rate"], marginals["own_flag_home"]["tie_rate"]
    if stolen["lcb95"] > home["ucb95"]:
        verdict = "STOLEN_UNIQUELY_TARGET_DEGENERATE"
    elif home["lcb95"] > 0.5 and stolen["lcb95"] > 0.5:
        verdict = "TARGET_TOO_COARSE_IN_GENERAL"
    else:
        verdict = "MIXED"

    rec = {
        "record": "D4 Target Resolution Audit",
        "status": "DIAGNOSTIC_RESULT -- stored outcomes only, no environment steps",
        "utc": _now(),
        "scope": "pole B only, per the frozen spec",
        "bootstrap": {"unit": "seed", "n_boot": N_BOOT, "alpha": ALPHA, "rng_seed": RNG_SEED},
        "PRIMARY_phase0": {"marginals": marginals, "four_way": four_way},
        "SECONDARY_d2_supplement_stolen_only_never_pooled": sup_stolen,
        "VERDICT": verdict,
        "reading": spec["PERMITTED_READINGS"].get(
            {"STOLEN_UNIQUELY_TARGET_DEGENERATE": "stolen_uniquely_tied",
             "TARGET_TOO_COARSE_IN_GENERAL": "all_regimes_similarly_tied",
             "MIXED": "mixed"}[verdict]),
    }
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\n  VERDICT: {verdict}\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
