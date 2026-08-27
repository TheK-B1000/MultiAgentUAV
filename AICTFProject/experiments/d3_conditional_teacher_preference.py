"""D3 -- Conditional Teacher Preference Audit. Implements D3_SPEC_FROZEN.json.

Asks whether pi_B actually beats pi_A in own_flag_stolen pole-B states, using
paired branch outcomes already stored on disk. No environment steps, no new
seeds, no FINAL access, no interference with the running RASR collector.

    Delta_branch(s) = M(pi_B | s) - M(pi_A | s)

M is the terminal win margin (blue - red), the SAME target Q_psi was fit on. The
pairing is intrinsic: both teachers were branched from the identical restored
state with teacher-consistent continuation.

Sources are reported SEPARATELY, per the frozen spec:
  PRIMARY   Phase 0 original branch pairs  (collected before the finding existed)
  SECONDARY D2 supplement pairs            (collected after, targeting the regime)
  pooled    additional descriptive estimate only

Seed-level PAIRED bootstrap: resample seeds, carry all of a seed's states
together. Bootstrapping individual states as independent is prohibited.

Run:  python experiments/d3_conditional_teacher_preference.py
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
SPEC = SPPO / "D3_SPEC_FROZEN.json"
OUT = SPPO / "D3_RESULT.json"

N_BOOT, ALPHA, RNG_SEED = 20_000, 0.05, 7
POLE_B = 1


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ci(draws: np.ndarray) -> dict:
    lo, hi = np.percentile(draws, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"mean": float(draws.mean()), "lcb95": float(lo), "ucb95": float(hi)}


def _paired_seed_bootstrap(by_seed: dict[int, np.ndarray]) -> dict:
    """Resample SEEDS, carrying every state of a sampled seed together."""
    seeds = sorted(by_seed)
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, len(seeds), size=(N_BOOT, len(seeds)))
    draws = np.empty(N_BOOT)
    arrs = [by_seed[s] for s in seeds]
    for b in range(N_BOOT):
        draws[b] = np.concatenate([arrs[j] for j in idx[b]]).mean()
    return _ci(draws)


def _collect(shard_paths, tagger, source: str) -> dict[int, np.ndarray]:
    """Per-seed arrays of Delta_branch on pole-B own_flag_stolen states."""
    by_seed: dict[int, list] = defaultdict(list)
    for path in shard_paths:
        seed = int(Path(path).stem.split("seed_")[-1])
        z = np.load(path, allow_pickle=True)
        if "branch_pole" not in z.files:
            continue
        pole = z["branch_pole"]
        vec = z["branch_obs_vec"]
        vec = vec[:, 0] if vec.ndim == 4 else vec        # strip vec-env dim if present
        regime = tagger.regime_from_vec(torch.as_tensor(vec, dtype=torch.float32)).numpy()
        sel = (pole == POLE_B) & (regime >= 2)           # regime 2/3 == own flag stolen
        if not sel.any():
            continue
        mB = (z["branch_pi_B_blue"][sel].astype(np.int32)
              - z["branch_pi_B_red"][sel].astype(np.int32))
        mA = (z["branch_pi_A_blue"][sel].astype(np.int32)
              - z["branch_pi_A_red"][sel].astype(np.int32))
        by_seed[seed].extend((mB - mA).astype(np.float64).tolist())
    return {s: np.asarray(v) for s, v in by_seed.items() if len(v)}


def _summarise(by_seed: dict, label: str) -> dict:
    if not by_seed:
        return {"label": label, "n_states": 0, "n_seeds": 0, "note": "no qualifying states"}
    allv = np.concatenate([by_seed[s] for s in sorted(by_seed)])
    ci = _paired_seed_bootstrap(by_seed)
    return {
        "label": label,
        "n_states": int(len(allv)), "n_seeds": int(len(by_seed)),
        "delta_branch": ci,
        "raw_mean": float(allv.mean()),
        "frac_states_favouring_piB": float((allv > 0).mean()),
        "frac_states_favouring_piA": float((allv < 0).mean()),
        "frac_states_tied": float((allv == 0).mean()),
    }


def main() -> int:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_ANY_D3_QUANTITY_IS_COMPUTED":
        raise SystemExit("REFUSING: D3 spec is not in the expected pre-computation state")
    bs = spec["bootstrap"]
    if (bs["n_boot"], bs["alpha"], bs["rng_seed"]) != (N_BOOT, ALPHA, RNG_SEED):
        raise SystemExit("REFUSING: bootstrap params drifted from the frozen D3 spec")
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; D3 is one-shot")

    from rl.scorer.qpsi import QPsi, QPsiConfig
    tagger = QPsi(QPsiConfig())

    print(f"D3 CONDITIONAL TEACHER PREFERENCE AUDIT  {_now()}")
    print("  Delta_branch = M(pi_B|s) - M(pi_A|s), terminal win margin")
    print("  states: pole B AND own_flag_stolen, paired by construction\n")

    primary = _collect(sorted(glob.glob(str(PHASE0 / "*.npz"))), tagger, "phase0")
    secondary = _collect(sorted(glob.glob(str(D2_SUP / "*.npz"))), tagger, "d2_supplement")

    res = {
        "PRIMARY_phase0_original": _summarise(primary, "Phase 0 original branch pairs"),
        "SECONDARY_d2_supplement": _summarise(secondary, "D2 supplement (targeted, post-finding)"),
    }
    pooled_by_seed: dict[int, list] = defaultdict(list)
    for src in (primary, secondary):
        for s, v in src.items():
            pooled_by_seed[s].extend(v.tolist())
    res["POOLED_descriptive_only"] = _summarise(
        {s: np.asarray(v) for s, v in pooled_by_seed.items()}, "pooled (descriptive only)")

    for key in ("PRIMARY_phase0_original", "SECONDARY_d2_supplement", "POOLED_descriptive_only"):
        r = res[key]
        if not r.get("n_states"):
            print(f"  {key:32s} no qualifying states")
            continue
        d = r["delta_branch"]
        print(f"  {key:32s} n={r['n_states']:4d} states / {r['n_seeds']:3d} seeds")
        print(f"     Delta_branch {d['mean']:+.4f} [{d['lcb95']:+.4f}, {d['ucb95']:+.4f}]"
              f"   piB-favouring {r['frac_states_favouring_piB']:.3f}"
              f"  piA-favouring {r['frac_states_favouring_piA']:.3f}"
              f"  tied {r['frac_states_tied']:.3f}")

    p = res["PRIMARY_phase0_original"]["delta_branch"]
    if p["lcb95"] > 0:
        verdict, reading = "PI_B_BETTER_CONDITIONALLY", spec["PERMITTED_READINGS"]["delta_positive"]
    elif p["ucb95"] < 0:
        verdict, reading = "PI_A_BETTER_CONDITIONALLY", spec["PERMITTED_READINGS"]["delta_negative"]
    else:
        verdict, reading = "UNRESOLVED", spec["PERMITTED_READINGS"]["near_zero_or_wide"]

    rec = {
        "record": "D3 Conditional Teacher Preference Audit",
        "status": "DIAGNOSTIC_RESULT -- reads stored outcomes only, no environment steps",
        "utc": _now(),
        "quantity": "Delta_branch(s) = M(pi_B|s) - M(pi_A|s), terminal win margin",
        "states": "paired branch states, pole B AND own_flag_stolen",
        "bootstrap": {"unit": "seed (paired)", "n_boot": N_BOOT, "alpha": ALPHA, "rng_seed": RNG_SEED},
        "sources": res,
        "verdict_basis": "PRIMARY (Phase 0 original) only; the D2 supplement is a source-stratified check",
        "VERDICT": verdict,
        "reading": reading,
        "D2_preserved": "D2 is not overwritten under any outcome; see D3_SPEC_FROZEN.json::D2_IS_NOT_OVERWRITTEN",
    }
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\n  VERDICT (from PRIMARY): {verdict}")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
