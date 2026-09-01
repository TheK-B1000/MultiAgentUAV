"""CCP Phase 1 prerequisite: do two diverged branches consume randomness identically?

Phase 0 established that (seed, action prefix) -> s_t is exact. That gives matched STATES.
It does not establish matched FUTURES: if stochastic code consumes random numbers at an
action-dependent rate, two branches forked from the same state would drift out of common
random number alignment, and paired continuation seeds would stop being paired.

The test is binary and byte-level, per the PI: at a prospectively fixed live boundary, force
two distinct legal interventions, advance both branches tick for tick, and compare
core._rng.get_state() after EVERY tick.

Note that _scripted_red.py:361 draws role_coin unconditionally on every step, so the
generator advances even when every stochastic knob is zero. Generator-state comparison is
therefore meaningful regardless of the config, and the config values are recorded alongside.

Verdicts:
  CRN_ALIGNED                        generator states byte-equal at every tick
  POST_RESET_DYNAMICS_DETERMINISTIC  aligned AND every stochastic knob is zero, so there is
                                     no continuation randomness to match in the first place
  CRN_DIVERGES                       generator states differ; paired seeds would need to be
                                     indexed independently of branch-dependent draw count

Phase 1 does not proceed under an ambiguous RNG story.

Run:  python experiments/ccp_phase1_crn_smoke.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
OUT = SD / "CCP_PHASE1_CRN_SMOKE.json"

SEED, POLE = 11_500_011, "A"          # fresh, disjoint from Phase 0's 11500001..11500008
BOUNDARY_RULE = "first step at or after 60 where agent 0 is free"
BOUNDARY_MIN = 60
N_MACROS = 5


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; one-shot")

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
        assert_live_opponent_batch(core, genomes, allowed_keys=(key,), context="ccp crn")
        return env, obs, core

    def rng_state(core):
        g = getattr(core, "_rng", None)
        if g is None:
            raise SystemExit("REFUSING: core exposes no _rng; the RNG story cannot be "
                             "established, and absence is not evidence of determinism")
        return g.get_state().clone()

    # ---- record, capturing per-step freedom straight from the runtime predicate ----
    env, obs, core = setup()
    policy.fixed_latent_strategy = True
    policy.fixed_latent_strategy_id = 0
    policy.reset_strategy()
    actions, free0, masks = [], [], []
    try:
        for _ in range(R2.MAX_STEPS):
            free0.append(bool((core.blue_commit_ticks_left[0, 0] <= 0).item()))
            masks.append(np.asarray(obs["mask"])[0].copy())
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

    boundary = next((s for s in range(BOUNDARY_MIN, T) if free0[s]), None)
    if boundary is None:
        raise SystemExit(f"REFUSING: no live boundary for agent 0 at or after step {BOUNDARY_MIN}")
    legal_macros = [int(m) for m in np.nonzero(masks[boundary][:N_MACROS])[0]]
    if len(legal_macros) < 2:
        raise SystemExit(f"REFUSING: only {len(legal_macros)} legal macro(s) at the boundary; "
                         "two distinct interventions are not available")
    print(f"CCP PHASE 1 CRN SMOKE  {_now()}")
    print(f"  seed {SEED} pole {POLE}  T={T}")
    print(f"  boundary (prospective rule: {BOUNDARY_RULE}): step {boundary}")
    print(f"  legal macros for agent 0 there: {legal_macros}", flush=True)

    def branch(macro: int):
        """Replay the prefix, intervene at the boundary, then record RNG state per tick."""
        env, obs, core = setup()
        try:
            states, knobs = [], None
            for i in range(T):
                a = np.asarray(actions[i]).copy()
                if i == boundary:
                    a[0] = macro
                env.step_async(a)
                obs, r, done, _ = env.step_wait()
                if i >= boundary:
                    states.append(rng_state(core))
                if knobs is None:
                    knobs = {
                        "rt_sensor_dropout_prob_max": float(core.rt_sensor_dropout_prob.max()),
                        "rt_sensor_noise_sigma_cells_max": float(core.rt_sensor_noise_sigma_cells.max()),
                        "red_deception_prob_max": float(core.red_deception_prob.max()),
                    }
                if bool(np.asarray(done).any()):
                    break
            return states, knobs
        finally:
            env.close()

    s1, knobs = branch(legal_macros[0])
    s2, _ = branch(legal_macros[1])
    n = min(len(s1), len(s2))
    first_div = next((i for i in range(n) if not torch.equal(s1[i], s2[i])), None)
    aligned = first_div is None and len(s1) == len(s2)
    all_knobs_zero = all(v == 0.0 for v in knobs.values())

    if not aligned:
        verdict = "CRN_DIVERGES"
    elif all_knobs_zero:
        verdict = "POST_RESET_DYNAMICS_DETERMINISTIC"
    else:
        verdict = "CRN_ALIGNED"

    print(f"  ticks compared: {n}  (branch lengths {len(s1)} / {len(s2)})")
    print(f"  first generator-state divergence: {first_div}")
    print(f"  stochastic knobs: {knobs}")
    print(f"\n  VERDICT: {verdict}")

    OUT.write_text(json.dumps({
        "record": "CCP Phase 1 CRN prerequisite smoke",
        "status": "FROZEN_RESULT", "one_shot": True, "utc": _now(),
        "program": "CAUSAL_CROSSOVER_PROGRAM_OPENING.json",
        "question": ("Do two branches forked from the same state consume randomness "
                     "identically, so paired continuation seeds stay paired?"),
        "VERDICT": verdict,
        "boundary": {"rule": BOUNDARY_RULE, "step": boundary, "seed": SEED, "pole": POLE,
                     "episode_length": T, "legal_macros": legal_macros,
                     "interventions": [legal_macros[0], legal_macros[1]]},
        "generator": {"ticks_compared": n, "branch_lengths": [len(s1), len(s2)],
                      "first_divergence": first_div, "byte_equal_throughout": bool(aligned),
                      "method": "torch.Generator.get_state() compared byte-for-byte after every tick"},
        "stochastic_config": knobs,
        "all_knobs_zero": bool(all_knobs_zero),
        "note": ("_scripted_red.py:361 draws role_coin unconditionally each step, so the "
                 "generator advances even when every knob is zero; generator-state comparison "
                 "is meaningful regardless of the config."),
        "consequence_for_M": (
            "If POST_RESET_DYNAMICS_DETERMINISTIC, continuation rollouts from a fixed state "
            "under a deterministic policy are identical, so Q(s,pi) is exact in {0,1} and M "
            "collapses to 1 by mathematical necessity rather than by budget choice. This is "
            "recorded prospectively, before any delta_Q is computed."),
        "EVAL_touched": False,
    }, indent=2), encoding="utf-8")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
