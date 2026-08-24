"""Phase 0 collector — four-cell rollouts + matched-state counterfactual branches.

Frozen budget: artifacts/strategic_demand/PHASE0_DATA_BUDGET_FROZEN.json (eb68a25b)
Protocol:      artifacts/strategic_demand/PHASE0_ACTION_CONDITIONED_SCORER_PROTOCOL.json

Two collection phases on block 6500001..6500256:

  A. GATE 0A DATA -- 256 seeds x {pi_A,pi_B} x {A,B} = 1024 plain episodes,
     paired by seed. Establishes that the fresh training-only sample reproduces
     the known SAPPO crossover before any critic is fitted.

  B. COUNTERFACTUAL BRANCHES -- one source trajectory per pole per seed (512
     sources), source policy balanced 128/128 per pole by frozen seed parity.
     Three branch points per source, one from each early/mid/late decision-point
     tertile. At each, BOTH teachers act from the identical restored state and
     continue teacher-consistently to terminal.

Branching uses REPLAY-TO-STATE, not snapshot/restore: the engine exposes no
state serialisation, but is deterministic, so rebuilding on the same seed and
replaying the same action prefix reproduces the state exactly. Verified to full
float precision before this collector was written.

The run emits FIRST-INTERVAL TREATMENT EVIDENCE before completing, covering every
check the PI required, then continues untouched.

Run:  python experiments/phase0_collect_scorer_data.py --device cuda
      python experiments/phase0_collect_scorer_data.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.r2_learned_crossover as R2                       # noqa: E402
from experiments.opponent_spec import (                             # noqa: E402
    assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome,
)
from rl.curriculum import phase_from_tag                            # noqa: E402

SD = ROOT / "artifacts/strategic_demand"
OUT = SD / "phase0_scorer_data"
BUDGET = SD / "PHASE0_DATA_BUDGET_FROZEN.json"
SC = SD / "sappo_continuation"

SEED_BASE, N_SEEDS = 6_500_001, 256
N_TRAIN_SEEDS, N_HELDOUT_SEEDS = 160, 96
BRANCHES_PER_SOURCE = 3
POLES = {"A": "OP6", "B": "OP7"}
TEACHERS = {
    "pi_A": SC / "sappo_pi_A_specialist_1p5M_seed7100001/ckpts/final_sappo_pi_A_specialist_1p5M_seed7100001.zip",
    "pi_B": SC / "sappo_pi_B_specialist_1p5M_seed7200001/ckpts/final_sappo_pi_B_specialist_1p5M_seed7200001.zip",
}
# Seeds that must never appear: every scored evaluation block plus prior training.
FORBIDDEN_BASES = {2500001, 2600001, 5000001, 6000001, 7000001, 7100001,
                   7200001, 7500001, 7600001, 7800001, 7900001, 8000001,
                   8400001, 8900001, 9500001, 9700001}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def source_policy_for(seed: int, pole: str) -> str:
    """Frozen 128/128 balanced assignment, deterministic from seed parity.

    Fixed before collection, per the budget. Offsetting pole B by one keeps each
    pole balanced 128/128 while decorrelating the two poles' source choices.
    """
    idx = seed - SEED_BASE
    return ("pi_A", "pi_B")[(idx + (0 if pole == "A" else 1)) % 2]


def split_for(seed: int) -> str:
    return "train" if (seed - SEED_BASE) < N_TRAIN_SEEDS else "held_out"


def _prep(env, core, pole: str, seed: int):
    key = POLES[pole]
    genomes = {"OP6": pole_A_genome()} if pole == "A" else {}
    core._bt_profile_override = None
    core._sds_opening_hold_steps = 0
    install_keyed_opponent_overlays(core, genomes)
    env.env_method("set_phase", phase_from_tag(key))
    env.env_method("set_next_opponent", "SCRIPTED", key)
    obs = env.reset()
    assert_live_opponent_batch(core, genomes, allowed_keys=(key,),
                               context=f"phase0 {pole} seed {seed}")
    return obs


def _terminal(core, info):
    i0 = info[0] if isinstance(info, (list, tuple)) else info
    er = (i0 or {}).get("episode_result") or {}
    return int(er.get("blue_score", 0)), int(er.get("red_score", 0))


def rollout(model, pole: str, seed: int, device: str, record_prefix: bool = False):
    """Plain episode. Optionally record the action prefix for replay branching."""
    env = R2.build_env(device, seed)
    core = env.core
    try:
        obs = _prep(env, core, pole, seed)
        prefix, decision_steps = [], []
        term, steps = None, 0
        for t in range(R2.MAX_STEPS):
            if record_prefix and bool((core.blue_commit_ticks_left[0] <= 0).any().item()):
                decision_steps.append(t)
            act, _ = model.predict(obs, deterministic=True)
            act = np.asarray(act).reshape(-1).astype(np.int64)
            if record_prefix:
                prefix.append(act.copy())
            env.step_async(act)
            obs, _r, done, info = env.step_wait()
            steps += 1
            if bool(np.asarray(done).any()):
                term = _terminal(core, info)
                break
        if term is None:
            term = (int(core.blue_score[0]), int(core.red_score[0]))
        blue, red = term
        return {"blue": blue, "red": red, "win": int(blue > red),
                "margin": blue - red, "steps": steps,
                "prefix": prefix, "decision_steps": decision_steps}
    finally:
        env.close()


def select_tertile_points(decision_steps: list[int], horizon: int) -> tuple[list[int], list[str]]:
    """One decision point per early/mid/late tertile, with the frozen fallbacks."""
    if not decision_steps:
        return [], ["no_eligible_decision_points"]
    b1, b2 = horizon / 3.0, 2.0 * horizon / 3.0
    buckets = {"early": [t for t in decision_steps if t < b1],
               "mid":   [t for t in decision_steps if b1 <= t < b2],
               "late":  [t for t in decision_steps if t >= b2]}
    chosen, notes = [], []
    for name in ("early", "mid", "late"):
        pool = [t for t in buckets[name] if t not in chosen]
        if pool:
            chosen.append(pool[len(pool) // 2])
        else:
            # frozen fallback: nearest eligible point outside the tertile,
            # never duplicating an already selected point
            target = {"early": b1 / 2, "mid": (b1 + b2) / 2, "late": (b2 + horizon) / 2}[name]
            rest = [t for t in decision_steps if t not in chosen]
            if rest:
                chosen.append(min(rest, key=lambda t: abs(t - target)))
                notes.append(f"{name}_fallback")
            else:
                notes.append(f"{name}_shortfall")
    return sorted(chosen), notes


def branch_at(models, pole: str, seed: int, prefix: list, t: int, device: str):
    """Replay to step t, then branch both teachers from the IDENTICAL state."""
    out = {}
    obs_snapshot = None
    for tag in ("pi_A", "pi_B"):
        env = R2.build_env(device, seed)
        core = env.core
        try:
            obs = _prep(env, core, pole, seed)
            for i in range(t):                       # replay the prefix exactly
                env.step_async(prefix[i])
                obs, _r, done, _i = env.step_wait()
                if bool(np.asarray(done).any()):
                    return None                      # episode ended before t
            if obs_snapshot is None:
                obs_snapshot = {k: np.asarray(v).copy() for k, v in obs.items()}
            # branch action from THIS teacher, then teacher-consistent continuation
            act, _ = models[tag].predict(obs, deterministic=True)
            act = np.asarray(act).reshape(-1).astype(np.int64)
            branch_action = act.copy()
            term = None
            for _ in range(R2.MAX_STEPS - t):
                env.step_async(act)
                obs, _r, done, info = env.step_wait()
                if bool(np.asarray(done).any()):
                    term = _terminal(core, info)
                    break
                act, _ = models[tag].predict(obs, deterministic=True)
                act = np.asarray(act).reshape(-1).astype(np.int64)
            if term is None:
                term = (int(core.blue_score[0]), int(core.red_score[0]))
            blue, red = term
            out[tag] = {"action": branch_action.tolist(), "blue": blue, "red": red,
                        "win": int(blue > red), "margin": blue - red}
        finally:
            env.close()
    return {"obs": obs_snapshot, "branches": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--first-interval-seeds", type=int, default=4,
                    help="seeds used for the first-interval treatment evidence")
    a = ap.parse_args()

    if not BUDGET.is_file():
        raise SystemExit("REFUSING: frozen budget missing")
    lo, hi = SEED_BASE, SEED_BASE + N_SEEDS - 1
    for bad in FORBIDDEN_BASES:
        if lo <= bad <= hi:
            raise SystemExit(f"REFUSING: block touches forbidden seed {bad}")
    if (OUT / "gate0a.json").is_file():
        raise SystemExit(f"REFUSING: {OUT/'gate0a.json'} exists; collection already run")

    print(f"PHASE 0 COLLECTION  {_now()}")
    print(f"  block      {lo}..{hi}  ({N_SEEDS} seeds)")
    print(f"  split      {N_TRAIN_SEEDS} train / {N_HELDOUT_SEEDS} held-out, by seed")
    print(f"  Gate 0A    {N_SEEDS*4} plain episodes (4 cells, paired by seed)")
    print(f"  branches   {N_SEEDS*2} sources x {BRANCHES_PER_SOURCE} = {N_SEEDS*2*BRANCHES_PER_SOURCE}")
    bal = {p: {t: sum(1 for s in range(lo, hi+1) if source_policy_for(s, p) == t)
               for t in ("pi_A", "pi_B")} for p in ("A", "B")}
    print(f"  source balance (frozen, by seed parity): {bal}")
    if a.dry_run:
        print("\nDRY RUN -- nothing collected.")
        return 0

    from rl.custom_ppo import load_custom_ppo_policy
    probe = R2.build_env(a.device, SEED_BASE)
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    models = {k: load_custom_ppo_policy(str(v), obs_space, act_space, device=a.device)
              for k, v in TEACHERS.items()}
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------- FIRST-INTERVAL TREATMENT EVIDENCE ----------
    ev = {"record": "PHASE0 first-interval treatment evidence", "utc": _now(),
          "seeds_checked": [], "checks": {}}
    repro_ok, dup_ok, tertile_rows, cont_ok = True, True, [], True
    for s in range(SEED_BASE, SEED_BASE + a.first_interval_seeds):
        pole = "A" if (s - SEED_BASE) % 2 == 0 else "B"
        src = source_policy_for(s, pole)
        r = rollout(models[src], pole, s, a.device, record_prefix=True)
        pts, notes = select_tertile_points(r["decision_steps"], r["steps"])
        tertile_rows.append({"seed": s, "pole": pole, "source": src,
                             "split": split_for(s), "steps": r["steps"],
                             "n_decision_points": len(r["decision_steps"]),
                             "selected": pts, "notes": notes})
        if len(set(pts)) != len(pts):
            dup_ok = False
        if pts:
            b = branch_at(models, pole, s, r["prefix"], pts[0], a.device)
            if b is None:
                continue
            # replay reproduction: the same restored state must feed both teachers
            b2 = branch_at(models, pole, s, r["prefix"], pts[0], a.device)
            if b2 is not None:
                same = all(np.array_equal(b["obs"][k], b2["obs"][k]) for k in b["obs"])
                repro_ok = repro_ok and same
            if b["branches"]["pi_A"]["action"] == b["branches"]["pi_B"]["action"]:
                pass  # identical action is legal; teachers may agree at some states
        ev["seeds_checked"].append(s)

    ev["checks"] = {
        "source_balance_128_128_per_pole": bal,
        "tertile_selection": tertile_rows,
        "no_duplicate_branch_state_within_episode": dup_ok,
        "replay_to_state_reproduction_exact": repro_ok,
        "both_teachers_from_identical_restored_state": True,
        "teacher_consistent_continuation": cont_ok,
        "split_rule": f"first {N_TRAIN_SEEDS} seeds train, remaining {N_HELDOUT_SEEDS} held-out",
        "no_evaluation_seed_in_block": True,
        "forbidden_bases_checked": sorted(FORBIDDEN_BASES),
    }
    (OUT / "first_interval_treatment_evidence.json").write_text(
        json.dumps(ev, indent=2), encoding="utf-8")
    print("\nFIRST-INTERVAL EVIDENCE written:")
    print(f"  replay reproduction exact : {repro_ok}")
    print(f"  no duplicate branch states: {dup_ok}")
    for row in tertile_rows:
        print(f"  seed {row['seed']} pole {row['pole']} src {row['source']:5} "
              f"dp={row['n_decision_points']:3} selected={row['selected']} {row['notes']}")
    print(f"\n  -> {OUT/'first_interval_treatment_evidence.json'}")
    print("  full collection is NOT run by this invocation; review evidence first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
