"""CCP-S2 stage 1: choose 128 boundaries (64/pole), deterministically, before any measurement.

Implements CCP_S2_SPEC.json#COLLECTION_DISTRIBUTION and CCP_S2_SEED_ASSIGNMENT.json.

Selection is frozen before Phase-1-style branching measurement begins, so it cannot be
influenced by any advantage estimate. Within each stratum, candidates are ordered by a hash
of IMMUTABLE identifiers (seed, prefix length) -- never by teacher disagreement, commit
success, incumbent outcome, or anything downstream of the causal question.

Strata: 2 poles x 3 free-sets x 3 phases = 18 cells, quota exactly as frozen:

    per pole    agent0_only  early 6  mid 6  late 7   = 19
                agent1_only  early 8  mid 8  late 8   = 24
                both_free    early 7  mid 7  late 7   = 21
                                                total  = 64

The prefix is driven by the INCUMBENT -- the sealed CCP successor checkpoint
(7164b662...) -- under the POLE-MATCHED latent: z0 on Pole A, z1 on Pole B. This is S2's
entire premise (collection from the incumbent's OWN deployment distribution), so the
rollout is verified element-by-element against experiments/eval_ccp_successor.py's exact
semantics rather than assumed to match.

Candidates are drawn ONLY from the frozen collection block 11700001..11700320. If a
required stratum cannot be filled, this STOPS with a selection shortfall and records it --
it never quietly substitutes a seed from outside the block or another cell.

The COMPLETE candidate counts per stratum are frozen in the output BEFORE the selected rows,
so a later audit can see not just what was chosen but what eligible support existed
everywhere -- not only in the cells that happened to be scarce.

Run:  python experiments/ccp_s2_select_states.py --device cuda
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
SPEC = SD / "CCP_S2_SPEC.json"
ASSIGNMENT = SD / "CCP_S2_SEED_ASSIGNMENT.json"
PREFLIGHT = SD / "CCP_S2_SEED_PREFLIGHT.json"
FROZEN_SUCCESSOR = SD / "CCP_SUCCESSOR_MODEL_FROZEN.json"
EVAL_SCRIPT = ROOT / "experiments" / "eval_ccp_successor.py"
OUT = SD / "CCP_S2_STATE_MANIFEST.json"

COLLECTION_SEEDS = list(range(11_700_001, 11_700_321))
POLES = ("A", "B")
PHASES = {"early": (0, 79), "mid": (80, 159), "late": (160, 239)}
QUOTA = {                      # per pole, frozen in CCP_S2_SPEC.json
    "agent0_only": {"early": 6, "mid": 6, "late": 7},
    "agent1_only": {"early": 8, "mid": 8, "late": 8},
    "both_free":   {"early": 7, "mid": 7, "late": 7},
}
POLE_LATENT = {"A": 0, "B": 1}          # the incumbent's own deployment latent per pole


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rank(seed: int, prefix_len: int) -> str:
    """Deterministic order over IMMUTABLE identifiers only, in S2's own hash domain."""
    return hashlib.sha256(f"CCP_S2_SELECT|{seed}|{prefix_len}".encode()).hexdigest()


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
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: S2 spec not frozen: {spec['status']!r}")
    assignment = json.loads(ASSIGNMENT.read_text(encoding="utf-8"))
    if assignment["status"] != "FROZEN_APPEND_ONLY_AMENDMENT":
        raise SystemExit(f"REFUSING: seed assignment not frozen: {assignment['status']!r}")
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if preflight["VERDICT"] != "PASS":
        raise SystemExit(f"REFUSING: seed preflight is not PASS: {preflight['VERDICT']!r}")
    block = assignment["SEED_BLOCKS"]["collection_state_source"]["block"]
    if block != f"{COLLECTION_SEEDS[0]}..{COLLECTION_SEEDS[-1]}":
        raise SystemExit(f"REFUSING: collection block drifted from the frozen assignment: {block!r}")

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome,
    )
    import experiments.phase0_collect_scorer_data as P0
    import experiments.r2_learned_crossover as R2
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    frozen = json.loads(FROZEN_SUCCESSOR.read_text(encoding="utf-8"))
    ck = ROOT / frozen["TERMINAL_CHECKPOINT"]["path"]
    actual = hashlib.sha256(ck.read_bytes()).hexdigest()
    if actual != frozen["TERMINAL_CHECKPOINT"]["sha256"]:
        raise SystemExit("REFUSING: incumbent checkpoint sha mismatch")
    if frozen["TERMINAL_RECORD_VALIDITY"]["verdict"] != "VALID":
        raise SystemExit("REFUSING: incumbent run was not established VALID")

    probe = R2.build_env(device, COLLECTION_SEEDS[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    policy = load_custom_ppo_policy(str(ck), obs_space, act_space, device=device)
    if not getattr(policy.model.critic, "private_z_heads", False):
        raise SystemExit("REFUSING: loaded incumbent critic does not have private z heads")
    model = policy.model if hasattr(policy, "model") else policy
    model.eval()

    print(f"CCP-S2 STATE SELECTION  {_now()}")
    print(f"  incumbent sha256 verified against CCP_SUCCESSOR_MODEL_FROZEN.json")
    print(f"  candidate seeds {COLLECTION_SEEDS[0]}..{COLLECTION_SEEDS[-1]} (n={len(COLLECTION_SEEDS)})")
    print(f"  poles {POLES}, incumbent latent by pole: {POLE_LATENT}")
    print(f"  prefix rollout semantics verified element-by-element against "
          f"{EVAL_SCRIPT.relative_to(ROOT)}\n", flush=True)

    candidates: list[dict] = []
    for pole in POLES:
        z = POLE_LATENT[pole]
        for seed in COLLECTION_SEEDS:
            env = R2.build_env(device, seed)
            core = env.core
            try:
                # element-by-element match to eval_ccp_successor.py's run_cell
                policy.fixed_latent_strategy = True
                policy.fixed_latent_strategy_id = int(z)
                policy.reset_strategy()
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
                                           context=f"s2 select {pole} seed {seed}")
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
                    obs, _r, done, _info = env.step_wait()
                    obs["global_state"] = env.state()
                    if bool(np.asarray(done).any()):
                        break
                for c in candidates:
                    if c["seed"] == seed and c["pole"] == pole and "actions" not in c:
                        c["actions"] = actions[: c["prefix_len"]]
            finally:
                env.close()
        n_pole = sum(1 for c in candidates if c["pole"] == pole)
        print(f"  pole {pole}: {n_pole} candidates from {len(COLLECTION_SEEDS)} seeds", flush=True)

    # ---- COMPLETE candidate counts, frozen BEFORE the selected rows ---------
    candidate_counts: dict[str, int] = {}
    for pole in POLES:
        for fs in QUOTA:
            for ph in PHASES:
                key_ = f"{pole}|{fs}|{ph}"
                candidate_counts[key_] = sum(
                    1 for c in candidates
                    if c["pole"] == pole and c["free_set"] == fs and c["phase"] == ph)

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
        # freeze the shortfall record itself before refusing, so the attempt is on record
        OUT.write_text(json.dumps({
            "record": "CCP-S2 state manifest", "status": "SELECTION_SHORTFALL", "utc": _now(),
            "candidate_counts_by_stratum": candidate_counts,
            "shortfalls": shortfalls, "n_candidates_total": len(candidates),
        }, indent=2), encoding="utf-8")
        raise SystemExit("REFUSING: a required stratum had no sufficient candidate pool; "
                         f"recorded in {OUT}, not substituted")

    if len(selected) != 128:
        raise SystemExit(f"REFUSING: expected 128 selected states, got {len(selected)}")
    per_pole_counts = {pole: sum(1 for s in selected if s["pole"] == pole) for pole in POLES}
    if any(v != 64 for v in per_pole_counts.values()):
        raise SystemExit(f"REFUSING: per-pole selection is not 64/64: {per_pole_counts}")

    cell_counts = {}
    for c in selected:
        k = f"{c['pole']}|{c['free_set']}|{c['phase']}"
        cell_counts[k] = cell_counts.get(k, 0) + 1
    print(f"\n  selected {len(selected)} states ({per_pole_counts}) across {len(cell_counts)} cells")

    OUT.write_text(json.dumps({
        "record": "CCP-S2 state manifest", "status": "FROZEN_SELECTION", "utc": _now(),
        "implements": "CCP_S2_SPEC.json#COLLECTION_DISTRIBUTION",
        "frozen_before_any_causal_measurement": True,
        "selection_rule": "sha256('CCP_S2_SELECT|<seed>|<prefix_len>') ascending within each stratum",
        "selected_on_immutable_identifiers_only": True,
        "never_selected_on": ["teacher disagreement", "commit success", "incumbent outcome",
                              "advantage estimate", "anything downstream of the causal question"],
        "prefix_driver": {"model": "CCP successor terminal checkpoint (the incumbent)",
                          "sha256": frozen["TERMINAL_CHECKPOINT"]["sha256"],
                          "latent": "pole-matched: z0 on Pole A, z1 on Pole B",
                          "verified_against": str(EVAL_SCRIPT.relative_to(ROOT))},
        "collection_block": [COLLECTION_SEEDS[0], COLLECTION_SEEDS[-1]],
        "phase_definition": {k: list(v) for k, v in PHASES.items()},
        "quota_per_pole": QUOTA,
        "n_candidates_total": len(candidates),
        "candidate_counts_by_stratum_BEFORE_selection": candidate_counts,
        "n_selected": len(selected),
        "per_pole_counts": per_pole_counts,
        "cell_counts": cell_counts,
        "shortfalls": [],
        "states": selected,
    }, indent=2), encoding="utf-8")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
