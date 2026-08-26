"""Phase 0 — environment-reuse equivalence test.

The frozen collector rebuilds an environment for every counterfactual branch.
Measured cost projects to ~27 hours for the full frozen collection, dominated by
per-branch environment construction rather than by simulated steps.

This tests whether construction can be AMORTISED across the three branches of a
source trajectory without changing behaviour. It is an implementation
optimisation only: the scientific allocation, state selection, continuation
semantics, teacher queries, seed split and Gates 0A/0B are untouched.

    BASELINE   (frozen reference)        CANDIDATE   (optimised)
    build env                            build env ONCE
    replay prefix -> o_t                 replay prefix -> o_t   (branch 1)
    branch, continue                     rebuild-in-place, replay -> o_t (branch 2)
    close env                            rebuild-in-place, replay -> o_t (branch 3)
    (repeat per branch)

Adoption requires EXACT agreement. The hidden-state trap is the reason: reusing
an environment can leave behind RNG state, opponent-controller internals, macro
commitment counters, telemetry accumulators, event history or cached targets
that a casual observation comparison would not reveal. So the comparison covers
the restored observation, the action mask, engine-internal counters, the first
branch action, the terminal outcome and the return label -- for both teachers,
both poles, and early/mid/late branch points.

Any meaningful mismatch REJECTS reuse, and the frozen ~27h collector is used
instead. The dataset is never shrunk to make runtime prettier.

Run:  python experiments/phase0_env_reuse_equivalence.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.phase0_collect_scorer_data as P0                 # noqa: E402
import experiments.r2_learned_crossover as R2                       # noqa: E402

OUT = ROOT / "artifacts/strategic_demand/phase0_scorer_data"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _core_fingerprint(core) -> dict:
    """Engine-internal state a plain observation comparison would miss."""
    def g(name, idx=True):
        v = getattr(core, name, None)
        if v is None:
            return None
        try:
            t = v[0] if idx else v
            return [round(float(x), 6) for x in torch.as_tensor(t).reshape(-1).tolist()]
        except Exception:
            return None
    return {
        "blue_x": g("blue_x"), "blue_y": g("blue_y"),
        "red_x": g("red_x"), "red_y": g("red_y"),
        "blue_commit_ticks_left": g("blue_commit_ticks_left"),
        "blue_commit_macro": g("blue_commit_macro"),
        "blue_commit_target": g("blue_commit_target"),
        "blue_carrying": g("blue_carrying"), "red_carrying": g("red_carrying"),
        "blue_alive": g("blue_alive"), "red_alive": g("red_alive"),
        "blue_tagged": g("blue_tagged"), "red_tagged": g("red_tagged"),
        "blue_score": g("blue_score"), "red_score": g("red_score"),
        "step_count": g("step_count"),
    }


def _replay_to(env, core, pole, seed, prefix, t):
    obs = P0._prep(env, core, pole, seed)
    for i in range(t):
        env.step_async(prefix[i])
        obs, _r, done, _i = env.step_wait()
        if bool(np.asarray(done).any()):
            return None
    return obs


def _branch_from(env, core, model, obs, t, device):
    act, _ = model.predict(obs, deterministic=True)
    act = np.asarray(act).reshape(-1).astype(np.int64)
    first = act.copy()
    term = None
    for _ in range(R2.MAX_STEPS - t):
        env.step_async(act)
        obs, _r, done, info = env.step_wait()
        if bool(np.asarray(done).any()):
            term = P0._terminal(core, info)
            break
        act, _ = model.predict(obs, deterministic=True)
        act = np.asarray(act).reshape(-1).astype(np.int64)
    if term is None:
        term = (int(core.blue_score[0]), int(core.red_score[0]))
    blue, red = term
    return {"first_action": first.tolist(), "blue": blue, "red": red,
            "win": int(blue > red), "margin": blue - red}


def baseline(models, pole, seed, prefix, ts, device):
    """Frozen reference: a fresh env for every (branch point, teacher)."""
    out = {}
    for t in ts:
        for tag in ("pi_A", "pi_B"):
            env = R2.build_env(device, seed); core = env.core
            try:
                obs = _replay_to(env, core, pole, seed, prefix, t)
                if obs is None:
                    out[(t, tag)] = None; continue
                fp = _core_fingerprint(core)
                mask = np.asarray(obs["mask"]).copy()
                res = _branch_from(env, core, models[tag], obs, t, device)
                out[(t, tag)] = {"fingerprint": fp, "mask": mask.tolist(), **res}
            finally:
                env.close()
    return out


def candidate(models, pole, seed, prefix, ts, device):
    """Optimised: build the env once, re-prepare in place between branches."""
    out = {}
    env = R2.build_env(device, seed); core = env.core
    try:
        for t in ts:
            for tag in ("pi_A", "pi_B"):
                obs = _replay_to(env, core, pole, seed, prefix, t)
                if obs is None:
                    out[(t, tag)] = None; continue
                fp = _core_fingerprint(core)
                mask = np.asarray(obs["mask"]).copy()
                res = _branch_from(env, core, models[tag], obs, t, device)
                out[(t, tag)] = {"fingerprint": fp, "mask": mask.tolist(), **res}
    finally:
        env.close()
    return out


def compare(base, cand):
    diffs = []
    for k in sorted(set(base) | set(cand), key=lambda x: (x[0], x[1])):
        b, c = base.get(k), cand.get(k)
        if (b is None) != (c is None):
            diffs.append({"key": f"t={k[0]}|{k[1]}", "field": "presence"}); continue
        if b is None:
            continue
        for f in ("first_action", "blue", "red", "win", "margin", "mask"):
            if b[f] != c[f]:
                diffs.append({"key": f"t={k[0]}|{k[1]}", "field": f,
                              "baseline": b[f] if f != "mask" else "<mask>",
                              "candidate": c[f] if f != "mask" else "<mask>"})
        for f, bv in b["fingerprint"].items():
            if bv != c["fingerprint"].get(f):
                diffs.append({"key": f"t={k[0]}|{k[1]}", "field": f"core.{f}",
                              "baseline": bv, "candidate": c["fingerprint"].get(f)})
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()

    from rl.custom_ppo import load_custom_ppo_policy
    probe = R2.build_env(a.device, P0.SEED_BASE)
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    models = {k: load_custom_ppo_policy(str(v), obs_space, act_space, device=a.device)
              for k, v in P0.TEACHERS.items()}

    print(f"PHASE 0 ENV-REUSE EQUIVALENCE  {_now()}")
    rec = {"record": "Phase 0 env-reuse equivalence", "utc": _now(), "cases": []}
    all_diffs, t_base, t_cand = [], 0.0, 0.0

    for i in range(a.seeds):
        seed = P0.SEED_BASE + i
        for pole in ("A", "B"):
            src = P0.source_policy_for(seed, pole)
            r = P0.rollout(models[src], pole, seed, a.device, record_prefix=True)
            ts, notes = P0.select_tertile_points(r["decision_steps"], r["steps"])
            if not ts:
                continue
            print(f"  seed {seed} pole {pole} src {src} branches {ts} ...", flush=True)
            t0 = time.perf_counter(); B = baseline(models, pole, seed, r["prefix"], ts, a.device)
            t_base += time.perf_counter() - t0
            t1 = time.perf_counter(); C = candidate(models, pole, seed, r["prefix"], ts, a.device)
            t_cand += time.perf_counter() - t1
            d = compare(B, C)
            all_diffs += d
            rec["cases"].append({"seed": seed, "pole": pole, "source": src,
                                 "branch_points": ts, "tertile_notes": notes,
                                 "n_mismatches": len(d), "mismatches": d[:8]})
            print(f"      mismatches: {len(d)}")

    speedup = t_base / max(1e-9, t_cand)
    rec["timing"] = {"baseline_s": round(t_base, 1), "candidate_s": round(t_cand, 1),
                     "speedup": round(speedup, 2)}
    rec["total_mismatches"] = len(all_diffs)
    rec["VERDICT"] = "REUSE_EQUIVALENT" if not all_diffs else "REUSE_REJECTED"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "env_reuse_equivalence.json").write_text(json.dumps(rec, indent=2),
                                                    encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"  total mismatches : {len(all_diffs)}")
    print(f"  baseline {t_base:.1f}s vs candidate {t_cand:.1f}s  (speedup {speedup:.2f}x)")
    print(f"  VERDICT: {rec['VERDICT']}")
    if all_diffs:
        print("  reuse REJECTED -- run the frozen collector unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
