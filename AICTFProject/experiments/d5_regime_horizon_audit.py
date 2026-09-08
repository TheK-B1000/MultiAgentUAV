"""D5 -- Regime x Horizon Tie-Rate Audit. Implements D5_SPEC_FROZEN.json.

Does the stolen-flag tie excess survive controlling for branch step?

  M_A == M_B may mean "strategically indifferent" OR "not enough remaining
  horizon for either policy to move the coarse terminal score."

Tertile boundaries were fixed REGIME-BLIND from the global pole-B branch_step
distribution and frozen in the spec (early <=49, mid 49-127, late >127) before
any regime-specific rate was computed.

Primary: Delta_tie(t) = P(tie | stolen, t) - P(tie | home, t), seed-level
bootstrap. Existing stored data only; no environment steps, no RASR interference.

Run:  python experiments/d5_regime_horizon_audit.py
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
SPEC = SPPO / "D5_SPEC_FROZEN.json"
OUT = SPPO / "D5_RESULT.json"

N_BOOT, ALPHA, RNG_SEED = 20_000, 0.05, 7
POLE_B = 1


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ci(d: np.ndarray) -> dict:
    lo, hi = np.percentile(d, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"mean": float(d.mean()), "lcb95": float(lo), "ucb95": float(hi)}


def _load(early_cut: float, late_cut: float):
    """rows[seed] = list of (tertile, is_stolen, is_carrying, tied)."""
    from rl.scorer.qpsi import QPsi, QPsiConfig
    tag = QPsi(QPsiConfig())
    rows: dict[int, list] = defaultdict(list)
    for path in sorted(glob.glob(str(PHASE0 / "*.npz"))):
        seed = int(Path(path).stem.split("seed_")[-1])
        z = np.load(path, allow_pickle=True)
        sel = z["branch_pole"] == POLE_B
        if not sel.any():
            continue
        vec = z["branch_obs_vec"]
        vec = vec[:, 0] if vec.ndim == 4 else vec
        regime = tag.regime_from_vec(torch.as_tensor(vec, dtype=torch.float32)).numpy()[sel]
        step = z["branch_step"][sel]
        mB = z["branch_pi_B_blue"][sel].astype(int) - z["branch_pi_B_red"][sel].astype(int)
        mA = z["branch_pi_A_blue"][sel].astype(int) - z["branch_pi_A_red"][sel].astype(int)
        for r, s, d in zip(regime, step, mB - mA):
            t = "early" if s <= early_cut else ("mid" if s <= late_cut else "late")
            rows[seed].append((t, int(r) >= 2, int(r) in (1, 3), d == 0))
    return rows


def _rate(rows, keep, label):
    """Seed-bootstrapped tie rate over states satisfying keep(row)."""
    num, den = {}, {}
    n = ties = 0
    for seed, items in rows.items():
        sel = [it for it in items if keep(it)]
        if not sel:
            continue
        den[seed] = len(sel)
        num[seed] = sum(1 for it in sel if it[3])
        n += len(sel); ties += num[seed]
    if not den:
        return {"label": label, "n_states": 0}
    seeds = sorted(den)
    a = np.array([num[s] for s in seeds], float)
    b = np.array([den[s] for s in seeds], float)
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, len(seeds), size=(N_BOOT, len(seeds)))
    dd = b[idx].sum(1)
    draws = np.divide(a[idx].sum(1), dd, out=np.full(N_BOOT, np.nan), where=dd > 0)
    draws = draws[~np.isnan(draws)]
    return {"label": label, "n_states": n, "n_seeds": len(seeds), "n_tied": ties,
            "tie_rate": _ci(draws)}


def _diff(rows, tertile):
    """Delta_tie(t) = P(tie|stolen,t) - P(tie|home,t), paired by seed."""
    sn, sd, hn, hd = {}, {}, {}, {}
    for seed, items in rows.items():
        st = [it for it in items if it[0] == tertile and it[1]]
        ho = [it for it in items if it[0] == tertile and not it[1]]
        if st:
            sd[seed] = len(st); sn[seed] = sum(1 for it in st if it[3])
        if ho:
            hd[seed] = len(ho); hn[seed] = sum(1 for it in ho if it[3])
    seeds = sorted(set(sd) | set(hd))
    if not seeds or not sd or not hd:
        return None
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, len(seeds), size=(N_BOOT, len(seeds)))
    arr = lambda m: np.array([m.get(s, 0.0) for s in seeds], float)
    sN, sD, hN, hD = arr(sn), arr(sd), arr(hn), arr(hd)
    draws = []
    for b in range(N_BOOT):
        j = idx[b]
        a1, b1 = sN[j].sum(), sD[j].sum()
        a2, b2 = hN[j].sum(), hD[j].sum()
        if b1 > 0 and b2 > 0:
            draws.append(a1 / b1 - a2 / b2)
    return _ci(np.array(draws)) if draws else None


def main() -> int:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_ANY_REGIME_SPECIFIC_TIE_RATE_IS_COMPUTED":
        raise SystemExit("REFUSING: D5 spec is not in the expected pre-computation state")
    tb = spec["TERTILE_BOUNDARIES_FROZEN_REGIME_BLIND"]
    early_cut, late_cut = 49, 127
    if abs(tb["q33"] - 48.7) > 0.5 or abs(tb["q66"] - 127.3) > 0.5:
        raise SystemExit("REFUSING: tertile boundaries drifted from the frozen spec")
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; D5 is one-shot")

    rows = _load(early_cut, late_cut)
    print(f"D5 REGIME x HORIZON TIE-RATE AUDIT  {_now()}")
    print(f"  frozen regime-blind tertiles: early<={early_cut}  mid<={late_cut}  late>{late_cut}\n")

    per_tertile, diffs = {}, {}
    print("  PRIMARY: tie rate by regime WITHIN each tertile")
    for t in ("early", "mid", "late"):
        st = _rate(rows, lambda it, t=t: it[0] == t and it[1], f"{t}/stolen")
        ho = _rate(rows, lambda it, t=t: it[0] == t and not it[1], f"{t}/home")
        per_tertile[t] = {"stolen": st, "home": ho}
        d = _diff(rows, t)
        diffs[t] = d
        f = lambda r: (f"n={r['n_states']:4d} tie {r['tie_rate']['mean']:.3f} "
                       f"[{r['tie_rate']['lcb95']:.3f},{r['tie_rate']['ucb95']:.3f}]"
                       if r.get("n_states") else "no states")
        print(f"    {t:6s} stolen  {f(st)}")
        print(f"    {t:6s} home    {f(ho)}")
        if d:
            print(f"    {t:6s} DELTA   {d['mean']:+.3f} [{d['lcb95']:+.3f}, {d['ucb95']:+.3f}]"
                  f"{'   overlaps 0' if d['lcb95'] <= 0 <= d['ucb95'] else '   excludes 0'}")
        print()

    ctx = {k: _rate(rows, f, k) for k, f in (
        ("carrying", lambda it: it[2]), ("not_carrying", lambda it: not it[2]))}
    print("  SECONDARY (contextual only):")
    for k, r in ctx.items():
        if r.get("n_states"):
            print(f"    {k:14s} n={r['n_states']:4d} tie {r['tie_rate']['mean']:.3f} "
                  f"[{r['tie_rate']['lcb95']:.3f},{r['tie_rate']['ucb95']:.3f}]")

    sig = [t for t in ("early", "mid", "late") if diffs[t] and diffs[t]["lcb95"] > 0]
    if len(sig) >= 2:
        verdict = "REGIME_EFFECT_SURVIVES_HORIZON_CONTROL"
    elif not sig:
        verdict = "HORIZON_EFFECT_DOMINANT"
    else:
        verdict = "REGIME_AND_HORIZON_BOTH_MATTER"

    rec = {
        "record": "D5 Regime x Horizon Tie-Rate Audit",
        "status": "DIAGNOSTIC_RESULT -- stored outcomes only",
        "utc": _now(),
        "frozen_tertiles": {"early": f"<= {early_cut}", "mid": f"<= {late_cut}", "late": f"> {late_cut}",
                            "derived": "regime-blind from the global pole-B branch_step distribution"},
        "bootstrap": {"unit": "seed", "n_boot": N_BOOT, "alpha": ALPHA, "rng_seed": RNG_SEED},
        "PRIMARY_per_tertile": per_tertile,
        "PRIMARY_within_tertile_difference": diffs,
        "SECONDARY_contextual": ctx,
        "tertiles_with_stolen_excess_excluding_zero": sig,
        "VERDICT": verdict,
        "reading": spec["PERMITTED_READINGS"].get(
            {"REGIME_EFFECT_SURVIVES_HORIZON_CONTROL": "regime_effect_survives",
             "HORIZON_EFFECT_DOMINANT": "horizon_effect_dominant",
             "REGIME_AND_HORIZON_BOTH_MATTER": "both_matter"}[verdict]),
    }
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\n  VERDICT: {verdict}\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
