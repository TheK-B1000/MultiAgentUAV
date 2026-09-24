r"""CURRENT PROJECTED vs CARRIER-EVASION-RESTORED, 4v4 pole A (2v2 control).

Everything identical between arms -- same commitment, same waypoint handling,
same seeds. The ONLY intervention: a carrying agent's teacher target is
preserved through the student-compatible pathway instead of being collapsed to
exact home by `_build_targets_from_action`'s `own_carrying` override.

Primary quantity is the SPECIALIZATION CONTRAST, not one cell's win rate:

    delta_A = V(GUARD@A) - V(BREACH@A)

computed per arm. The h=1 repair left GUARD@A fine (0.708 -> 0.750) while
BREACH@A jumped (0.292 -> 0.750), collapsing delta_A from +0.417 to 0.000.
If restoring carrier evasion moves delta_A back toward ORIGINAL, carrier
representation is the binding action-side defect.

Gated by experiments/carrier_restore.carrier_anchors (3 known-answer anchors)
and by the oracle harness anchor. Refuses to run if either fails.

    python -m experiments.eval_carrier_restore_test --scales 4 2 --promote
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
LABEL = "CARRIER_RESTORE_TEST"
ARMS = ("ORIGINAL", "PROJECTED", "CARRIER_RESTORED")
FIELDS = ["scale", "strategy", "pole", "arm", "seed", "blue", "red", "win"]


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _boot(d, n_boot=20000, alpha=0.05, seed=7):
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return {"mean": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    b = d[idx].mean(axis=1)
    lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": round(float(d.mean()), 6), "lcb95": round(float(lo), 6),
            "ucb95": round(float(hi), 6), "n": int(d.size)}


def run_episode(scale, strategy, arm, seed, device):
    import experiments.strategic_demand_searcher as S0
    from experiments.carrier_restore import CarrierRestore, adapt_no_carrier_override
    from experiments.eval_projected_teacher_oracle import _make_env
    from experiments.teacher_action_adapter import adapt

    style = S0.GUARD if strategy == "GUARD" else S0.BREACH
    env, core, S = _make_env(scale, "A", style, seed, device)
    try:
        if arm != "ORIGINAL":
            core.blue_scripted = False
        if arm == "CARRIER_RESTORED":
            # Suppress ONLY the carrying->home override, using the real engine
            # function (rule 11). Restored immediately after each call.
            orig_btfa = core._build_targets_from_action

            def patched(macro, target, side="blue", _o=orig_btfa):
                if side == "blue":
                    with CarrierRestore(core):
                        return _o(macro, target, side=side)
                return _o(macro, target, side=side)
            core._build_targets_from_action = patched

        W = core._macro_targets.detach().cpu().numpy()
        action = np.zeros((scale, 2), dtype=np.int64)
        have = False
        term = None
        for _t in range(S.MAX_STEPS):
            if arm == "ORIGINAL":
                env.step_async(env.action_space.sample() * 0)
            else:
                nc = (core.blue_commit_ticks_left[0] <= 0).detach().cpu().numpy()
                btx, bty = core._get_scripted_targets("blue")
                for i in range(scale):
                    if have and not bool(nc[i]):
                        continue
                    tx, ty = float(btx[0, i]), float(bty[0, i])
                    a = (adapt_no_carrier_override(core, tx, ty, i, W)
                         if arm == "CARRIER_RESTORED" else adapt(core, tx, ty, i, W))
                    if a.uses_waypoint:
                        action[i] = (a.macro, a.target_idx)
                    elif a.macro is not None:
                        action[i] = (a.macro, 0)
                    else:
                        action[i] = (0, 0)
                have = True
                env.step_async(action.reshape(-1))
            _o, _r, done, info = env.step_wait()
            if bool(np.asarray(done).any()):
                i0 = info[0] if isinstance(info, (list, tuple)) else info
                er = (i0 or {}).get("episode_result") or {}
                term = (int(er.get("blue_score", 0)), int(er.get("red_score", 0)))
                break
        if term is None:
            term = (int(core.blue_score[0]), int(core.red_score[0]))
        b, r = term
        return {"scale": f"{scale}v{scale}", "strategy": strategy, "pole": "A",
                "arm": arm, "seed": seed, "blue": b, "red": r, "win": int(b > r)}
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scales", type=int, nargs="+", default=[4, 2])
    ap.add_argument("--n-seeds", type=int, default=24)
    ap.add_argument("--promote", action="store_true")
    a = ap.parse_args()
    seeds = list(range(17600001, 17600001 + a.n_seeds))

    # ---- gates ----------------------------------------------------------
    import experiments.strategic_demand_searcher as S0
    from experiments.carrier_restore import carrier_anchors
    from experiments.eval_projected_teacher_oracle import _make_env, harness_anchor
    fails = []
    for sc in a.scales:
        env, core, _ = _make_env(sc, "A", S0.GUARD, seeds[0], a.device)
        try:
            fails += [f"{sc}v{sc}: {m}" for m in
                      carrier_anchors(core, core._macro_targets.detach().cpu().numpy(), 0)]
        finally:
            env.close()
        fails += harness_anchor(sc, a.device, tuple(seeds[:2]))
    if fails:
        raise SystemExit("ANCHORS FAILED -- refusing to run:\n  " + "\n  ".join(fails))
    print(f"{LABEL}  {_now()}")
    print(f"  carrier anchors + harness anchor: PASS")
    print(f"  arms {ARMS}, GUARD/BREACH @ poleA, scales {a.scales}, n={len(seeds)}\n",
          flush=True)

    ROWS = SD / f"{LABEL.lower()}_rows.csv"
    rows = []
    done = set()
    if ROWS.is_file():
        with ROWS.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
                done.add((r["scale"], r["strategy"], r["arm"], int(r["seed"])))
        print(f"  RESUME: {len(done)} episodes on disk", flush=True)
    else:
        with ROWS.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writeheader()

    from experiments.tqdm_loop import set_postfix, tqdm_iter
    cells = [(n, st, arm) for n in a.scales for st in ("GUARD", "BREACH") for arm in ARMS]
    t0 = time.time()
    bar = tqdm_iter(cells, desc=LABEL, unit="cell")
    for n, st, arm in bar:
        set_postfix(bar, f"{n}v{n} {st} {arm}")
        wins = []
        for seed in seeds:
            key = (f"{n}v{n}", st, arm, seed)
            if key in done:
                wins.append(int(next(r["win"] for r in rows if
                    (r["scale"], r["strategy"], r["arm"], int(r["seed"])) == key)))
                continue
            rec = run_episode(n, st, arm, seed, a.device)
            wins.append(rec["win"])
            with ROWS.open("a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=FIELDS).writerow(rec)
                fh.flush(); os.fsync(fh.fileno())
            rows.append({k: str(v) for k, v in rec.items()})
        print(f"  {n}v{n} {st:<7s} {arm:<17s} V={np.mean(wins):.4f}", flush=True)

    def V(n, st, arm):
        d = {int(r["seed"]): int(r["win"]) for r in rows if r["scale"] == f"{n}v{n}"
             and r["strategy"] == st and r["arm"] == arm}
        return np.array([d[s] for s in seeds if s in d], dtype=float)

    out = {}
    for n in a.scales:
        cell = {"V": {}, "delta_A": {}}
        for arm in ARMS:
            g, b = V(n, "GUARD", arm), V(n, "BREACH", arm)
            m = min(g.size, b.size)
            cell["V"][arm] = {"GUARD@A": round(float(g.mean()), 4) if g.size else None,
                              "BREACH@A": round(float(b.mean()), 4) if b.size else None}
            cell["delta_A"][arm] = _boot(g[:m] - b[:m]) if m else None
        out[f"{n}v{n}"] = cell
        print(f"\n  {n}v{n} pole A -- delta_A = V(GUARD@A) - V(BREACH@A)")
        for arm in ARMS:
            d = cell["delta_A"][arm]
            v = cell["V"][arm]
            print(f"    {arm:<18s} GUARD={v['GUARD@A']:.4f} BREACH={v['BREACH@A']:.4f}"
                  f"   delta_A={d['mean']:+.4f} [{d['lcb95']:+.4f},{d['ucb95']:+.4f}]")

    cfg = json.dumps({"scales": a.scales, "n_seeds": a.n_seeds, "device": a.device},
                     sort_keys=True)
    rid = f"{_now().replace(':','').replace('-','')}_{hashlib.sha256(cfg.encode()).hexdigest()[:8]}"
    rec = {"record": f"{LABEL} carrier-evasion restoration test",
           "status": "FROZEN_RESULT", "utc": _now(), "run_id": rid,
           "arm": "DIAGNOSTIC", "confirmatory": False,
           "study_class": "MECHANISTIC_FOLLOW_UP",
           "intervention": "suppress ONLY the own_carrying->exact-home override in "
                           "_build_targets_from_action; everything else identical",
           "config_signature": json.loads(cfg), "elapsed_s": round(time.time() - t0, 1),
           "results": out}
    p = SD / f"{LABEL}_{rid}_RESULT.json"
    p.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    if a.promote:
        (SD / f"{LABEL}_CANONICAL.json").write_text(json.dumps(
            {"record": f"CANONICAL {LABEL}", "points_to": p.name, "run_id": rid,
             "promoted_utc": _now()}, indent=2), encoding="utf-8")
    print(f"\n  -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
