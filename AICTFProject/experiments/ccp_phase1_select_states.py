"""CCP Phase 1 stage 1: choose the 20 pilot states, deterministically, before any rollout.

Implements CCP_PHASE1_CAUSAL_BRANCHING_SPEC.json#AMENDMENT_2_M16_PILOT_AND_CLAIM_SCOPE.

Selection must be frozen and committed BEFORE collection, so it cannot be influenced by any
delta_Q. Within each stratum, candidates are ordered by a hash of IMMUTABLE identifiers
(seed, prefix length) -- never by teacher disagreement, commit success, trajectory appearance,
or anything downstream of the causal question.

Strata: 2 poles x 3 free-sets x 3 phases = 18 cells, all covered.

    per pole    agent0_only  early 1  mid 1  late 1
                agent1_only  early 1  mid 2  late 1
                both_free    early 1  mid 1  late 1     = 10 per pole, 20 total

Phase is defined prospectively by prefix length: early 0..79, mid 80..159, late 160..239.
The free set comes from the runtime predicate blue_commit_ticks_left[i] <= 0.

If a required stratum has no candidate this STOPS and records it. It never substitutes a
state from another cell.

The prefix is driven by the V4 policy under the POLE-MATCHED latent (z0 on Pole A, z1 on
Pole B), which is the condition the model would be deployed under and the one EVAL scored.
That choice is recorded here rather than left implicit.

Run:  python experiments/ccp_phase1_select_states.py --device cuda
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
OUT = SD / "CCP_PHASE1_PILOT_MANIFEST.json"

# fresh, disjoint from Phase 0 (11500001..08) and the smokes (11500011, 11500021)
CANDIDATE_SEEDS = list(range(11_500_101, 11_500_111))
POLES = ("A", "B")
PHASES = {"early": (0, 79), "mid": (80, 159), "late": (160, 239)}
QUOTA = {                      # per pole
    "agent0_only": {"early": 1, "mid": 1, "late": 1},
    "agent1_only": {"early": 1, "mid": 2, "late": 1},
    "both_free":   {"early": 1, "mid": 1, "late": 1},
}
POLE_LATENT = {"A": 0, "B": 1}          # pole-matched latent drives the prefix


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rank(seed: int, prefix_len: int) -> str:
    """Deterministic order over IMMUTABLE identifiers only."""
    return hashlib.sha256(f"CCP_PHASE1_SELECT|{seed}|{prefix_len}".encode()).hexdigest()


def _phase_of(prefix_len: int) -> str | None:
    for name, (lo, hi) in PHASES.items():
        if lo <= prefix_len <= hi:
            return name
    return None


def _free_set(f0: bool, f1: bool) -> str | None:
    if f0 and f1:
        return "both_free"
    if f0:
        return "agent0_only"
    if f1:
        return "agent1_only"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; selection is frozen once")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if "AMENDMENT_2_M16_PILOT_AND_CLAIM_SCOPE" not in spec:
        raise SystemExit("REFUSING: pilot amendment not present in the frozen spec")

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

    probe = R2.build_env(device, CANDIDATE_SEEDS[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    policy = load_custom_ppo_policy(str(ck), obs_space, act_space, device=device)
    (policy.model if hasattr(policy, "model") else policy).eval()

    print(f"CCP PHASE 1 STATE SELECTION  {_now()}")
    print(f"  candidate seeds {CANDIDATE_SEEDS[0]}..{CANDIDATE_SEEDS[-1]}  poles {POLES}")
    print(f"  prefix driven by V4 under the pole-matched latent\n", flush=True)

    candidates: list[dict] = []
    for pole in POLES:
        z = POLE_LATENT[pole]
        for seed in CANDIDATE_SEEDS:
            env = R2.build_env(device, seed)
            core = env.core
            try:
                core._bt_profile_override = None
                core._sds_opening_hold_steps = 0
                genomes = {"OP6": pole_A_genome()} if pole == "A" else {}
                install_keyed_opponent_overlays(core, genomes)
                key = P0.POLES[pole]
                env.env_method("set_phase", phase_from_tag(key))
                env.env_method("set_next_opponent", "SCRIPTED", key)
                obs = env.reset()
                obs["global_state"] = env.state()
                assert_live_opponent_batch(core, genomes, allowed_keys=(key,),
                                           context=f"select {pole} seed {seed}")
                policy.fixed_latent_strategy = True
                policy.fixed_latent_strategy_id = z
                policy.reset_strategy()
                actions = []
                for step in range(R2.MAX_STEPS):
                    f0 = bool((core.blue_commit_ticks_left[0, 0] <= 0).item())
                    f1 = bool((core.blue_commit_ticks_left[0, 1] <= 0).item())
                    fs, ph = _free_set(f0, f1), _phase_of(step)
                    if fs and ph:
                        candidates.append({
                            "state_id": f"{seed}|{pole}|{step}",
                            "seed": seed, "pole": pole, "prefix_len": step,
                            "free_set": fs, "phase": ph, "rank": _rank(seed, step)})
                    a, _ = policy.predict(obs, deterministic=True)
                    actions.append([int(x) for x in np.asarray(a).ravel()])
                    env.step_async(a)
                    obs, r, done, _ = env.step_wait()
                    obs["global_state"] = env.state()
                    if bool(np.asarray(done).any()):
                        break
                # actions are needed by the collector to replay each prefix
                for c in candidates:
                    if c["seed"] == seed and c["pole"] == pole and "actions" not in c:
                        c["actions"] = actions[: c["prefix_len"]]
            finally:
                env.close()
            print(f"  {pole} seed {seed}: {len(actions):3d} steps, "
                  f"{sum(1 for c in candidates if c['seed']==seed and c['pole']==pole)} candidates",
                  flush=True)

    selected, shortfalls = [], []
    for pole in POLES:
        for fs, per_phase in QUOTA.items():
            for ph, need in per_phase.items():
                pool = sorted((c for c in candidates
                               if c["pole"] == pole and c["free_set"] == fs and c["phase"] == ph),
                              key=lambda c: c["rank"])
                if len(pool) < need:
                    shortfalls.append({"pole": pole, "free_set": fs, "phase": ph,
                                       "needed": need, "available": len(pool)})
                selected.extend(pool[:need])

    if shortfalls:
        print("\n  STRATUM SHORTFALL -- stopping rather than substituting:")
        for s in shortfalls:
            print(f"    {s}")
        raise SystemExit("REFUSING: a required stratum had no candidate; recorded, not substituted")

    counts = {}
    for c in selected:
        k = f"{c['pole']}|{c['free_set']}|{c['phase']}"
        counts[k] = counts.get(k, 0) + 1
    print(f"\n  selected {len(selected)} states across {len(counts)} cells")

    OUT.write_text(json.dumps({
        "record": "CCP Phase 1 pilot state manifest",
        "status": "FROZEN_SELECTION", "utc": _now(),
        "implements": "CCP_PHASE1_CAUSAL_BRANCHING_SPEC.json#AMENDMENT_2_M16_PILOT_AND_CLAIM_SCOPE",
        "frozen_before_any_rollout": True,
        "selection_rule": "sha256('CCP_PHASE1_SELECT|<seed>|<prefix_len>') ascending within each stratum",
        "selected_on_immutable_identifiers_only": True,
        "never_selected_on": ["teacher disagreement", "commit success", "trajectory appearance",
                              "anything downstream of the causal question"],
        "prefix_driver": {"model": "HOG_PSP_V4 terminal checkpoint",
                          "sha256": frozen["TERMINAL_CHECKPOINT"]["sha256"],
                          "latent": "pole-matched: z0 on Pole A, z1 on Pole B",
                          "why": "the deployment condition, and the one EVAL scored"},
        "candidate_seeds": [CANDIDATE_SEEDS[0], CANDIDATE_SEEDS[-1]],
        "phase_definition": {k: list(v) for k, v in PHASES.items()},
        "quota_per_pole": QUOTA,
        "n_candidates": len(candidates),
        "n_selected": len(selected),
        "cell_counts": counts,
        "shortfalls": shortfalls,
        "states": selected,
        "EVAL_touched": False,
    }, indent=2), encoding="utf-8")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
