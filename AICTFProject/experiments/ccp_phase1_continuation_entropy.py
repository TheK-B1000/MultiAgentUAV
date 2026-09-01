"""CCP Phase 1 prerequisite: characterise the continuation distribution.

Phase 0 proved prefix replay is EXACT, which means M continuations from a reconstructed
state are M identical rollouts unless the environment generator is explicitly reseeded. This
smoke does two things:

  1. validates the reseeding machinery -- does reseeding actually produce distinct
     environmental continuations at all?
  2. characterises the continuation distribution -- how often does stochasticity alter the
     trajectory, and how often does it alter the terminal outcome?

What this smoke is NOT for (PI, 2026-09-01): defining an effective sample size from unique
trajectories. Independent seeds that happen to produce the same continuation are still two
legitimate samples; sparse randomness just means the distribution has large mass on one
continuation. Precision is governed by Var(D_j), not by trajectory uniqueness. M = 32 stands
unless the machinery turns out to be inoperative.

Seeds come from the frozen deterministic mapping r_j = H('CCP_PHASE1', state_id, j), never
drawn opportunistically, so pairing cannot depend on execution order.

Policy actions are deterministic argmax throughout, so policy stochasticity is outside the
expectation by construction.

Run:  python experiments/ccp_phase1_continuation_entropy.py --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "CCP_PHASE1_CAUSAL_BRANCHING_SPEC.json"
OUT = SD / "CCP_PHASE1_CONTINUATION_ENTROPY.json"

SEED, POLE = 11_500_021, "A"          # fresh; disjoint from Phase 0 and the CRN smoke
BOUNDARY_MINS = (40, 80, 120)         # prospectively fixed: first live boundary at/after each
M = 32


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def continuation_seed(state_id: str, j: int) -> int:
    """r_j = H('CCP_PHASE1', state_id, j) -- frozen, reproducible from the artifact alone."""
    h = hashlib.sha256(f"CCP_PHASE1|{state_id}|{j}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2 ** 63 - 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; one-shot")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: Phase 1 spec not frozen: {spec['status']!r}")
    if int(spec["M_AND_CONTINUATION_SAMPLING"]["M"]) != M:
        raise SystemExit("REFUSING: M disagrees with the frozen spec")

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome)
    import experiments.phase0_collect_scorer_data as P0
    import experiments.r2_learned_crossover as R2
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    frozen = json.loads((SD / "HOG_PSP_V4_MODEL_FROZEN.json").read_text(encoding="utf-8"))
    ck = ROOT / frozen["TERMINAL_CHECKPOINT"]["path"]

    probe = R2.build_env(device, SEED)
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    policy = load_custom_ppo_policy(str(ck), obs_space, act_space, device=device)
    (policy.model if hasattr(policy, "model") else policy).eval()

    def setup():
        env = R2.build_env(device, SEED)
        core = env.core
        core._bt_profile_override = None
        core._sds_opening_hold_steps = 0
        genomes = {"OP6": pole_A_genome()} if POLE == "A" else {}
        install_keyed_opponent_overlays(core, genomes)
        key = P0.POLES[POLE]
        env.env_method("set_phase", phase_from_tag(key))
        env.env_method("set_next_opponent", "SCRIPTED", key)
        obs = env.reset()
        obs["global_state"] = env.state()
        assert_live_opponent_batch(core, genomes, allowed_keys=(key,), context="ccp entropy")
        return env, obs, core

    def outcome(info) -> int:
        rec = info[0] if isinstance(info, (list, tuple)) and info else info
        b, r = int(np.any(rec.get("blue_score", 0))), int(np.any(rec.get("red_score", 0)))
        return 1 if rec.get("blue_score", 0) > rec.get("red_score", 0) else 0

    # ---- reference episode, recording live boundaries from the runtime predicate ----
    env, obs, core = setup()
    policy.fixed_latent_strategy = True
    policy.fixed_latent_strategy_id = 0
    policy.reset_strategy()
    actions, free0 = [], []
    try:
        for _ in range(R2.MAX_STEPS):
            free0.append(bool((core.blue_commit_ticks_left[0, 0] <= 0).item()))
            a, _ = policy.predict(obs, deterministic=True)
            actions.append(np.asarray(a).copy())
            env.step_async(a)
            obs, r, done, _ = env.step_wait()
            obs["global_state"] = env.state()
            if bool(np.asarray(done).any()):
                break
        T = len(actions)
    finally:
        env.close()

    boundaries = []
    for lo in BOUNDARY_MINS:
        b = next((s for s in range(lo, T) if free0[s]), None)
        if b is not None and b not in boundaries:
            boundaries.append(b)
    if len(boundaries) < 3:
        raise SystemExit(f"REFUSING: found {len(boundaries)} distinct live boundaries, need 3")

    print(f"CCP PHASE 1 CONTINUATION ENTROPY  {_now()}")
    print(f"  seed {SEED} pole {POLE}  T={T}  M={M}")
    print(f"  boundaries (first live at/after {BOUNDARY_MINS}): {boundaries}\n", flush=True)

    def rollout(boundary: int, rj: int | None):
        """Replay prefix, optionally reseed env RNG, then run the policy to termination."""
        env, obs, core = setup()
        try:
            for i in range(boundary):
                env.step_async(np.asarray(actions[i]))
                obs, r, done, _ = env.step_wait()
                obs["global_state"] = env.state()
                if bool(np.asarray(done).any()):
                    break
            if rj is not None:
                core._rng.manual_seed(int(rj))          # ONLY the environment RNG
            policy.fixed_latent_strategy = True
            policy.fixed_latent_strategy_id = 0
            policy.reset_strategy()
            h = hashlib.sha256()
            info = None
            steps = 0
            for _ in range(R2.MAX_STEPS):
                a, _ = policy.predict(obs, deterministic=True)
                env.step_async(a)
                obs, r, done, info = env.step_wait()
                obs["global_state"] = env.state()
                h.update(np.asarray(env.state()).tobytes())
                steps += 1
                if bool(np.asarray(done).any()):
                    break
            return h.hexdigest(), outcome(info), steps
        finally:
            env.close()

    results = {}
    for boundary in boundaries:
        state_id = f"{SEED}|{POLE}|{boundary}"
        traj, outs, lens = [], [], []
        for j in range(M):
            t, o, n = rollout(boundary, continuation_seed(state_id, j))
            traj.append(t); outs.append(o); lens.append(n)
        no_reseed_a, _, _ = rollout(boundary, None)
        no_reseed_b, _, _ = rollout(boundary, None)

        uniq_traj = len(set(traj))
        modal = max(set(traj), key=traj.count)
        results[state_id] = {
            "boundary_step": boundary,
            "reseeding_machinery_live": uniq_traj > 1,
            "unreseeded_rollouts_identical": no_reseed_a == no_reseed_b,
            "distinct_trajectories": uniq_traj,
            "modal_trajectory_share": traj.count(modal) / M,
            "fraction_altered_vs_modal": 1.0 - traj.count(modal) / M,
            "distinct_terminal_outcomes": len(set(outs)),
            "outcome_mean": float(np.mean(outs)),
            "outcome_counts": {"win": int(sum(outs)), "loss": int(M - sum(outs))},
            "episode_length_range": [int(min(lens)), int(max(lens))],
            "effectively_deterministic_outcome": len(set(outs)) == 1,
        }
        r = results[state_id]
        print(f"  boundary {boundary:3d}: distinct traj {uniq_traj:2d}/{M}  "
              f"modal share {r['modal_trajectory_share']:.3f}  "
              f"outcomes {r['outcome_counts']}  "
              f"reseed live {r['reseeding_machinery_live']}", flush=True)

    machinery_ok = all(v["reseeding_machinery_live"] for v in results.values())
    unreseeded_identical = all(v["unreseeded_rollouts_identical"] for v in results.values())
    verdict = ("RESEEDING_MACHINERY_VALIDATED" if machinery_ok and unreseeded_identical
               else "RESEEDING_MACHINERY_INOPERATIVE")

    OUT.write_text(json.dumps({
        "record": "CCP Phase 1 continuation entropy",
        "status": "FROZEN_RESULT", "one_shot": True, "utc": _now(),
        "implements": "CCP_PHASE1_CAUSAL_BRANCHING_SPEC.json#AMENDMENT_1",
        "VERDICT": verdict,
        "what_this_measures": ("whether reseeding produces distinct environmental continuations, "
                               "and how concentrated the continuation distribution is"),
        "what_this_does_NOT_define": ("an effective sample size. Independent seeds producing the "
                                      "same continuation are still legitimate samples; precision "
                                      "is governed by Var(D_j), not trajectory uniqueness. M "
                                      "stands unless the machinery is inoperative."),
        "M": M,
        "seed_mapping": "r_j = sha256('CCP_PHASE1|<state_id>|<j>')[:8] as int, mod 2^63-1",
        "only_env_rng_reseeded": True,
        "policy_actions": "deterministic argmax; policy stochasticity is outside the expectation",
        "reference_episode": {"seed": SEED, "pole": POLE, "length": T,
                              "boundary_rule": f"first live boundary at or after {BOUNDARY_MINS}"},
        "per_state": results,
        "unreseeded_control": ("two rollouts with NO reseed must be identical, confirming that "
                               "any variation observed is caused by the reseed and not by "
                               "ambient nondeterminism"),
        "EVAL_touched": False,
    }, indent=2), encoding="utf-8")
    print(f"\n  VERDICT: {verdict}")
    print(f"  -> {OUT}")
    return 0 if verdict == "RESEEDING_MACHINERY_VALIDATED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
