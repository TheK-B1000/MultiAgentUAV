"""CCP-S2 stage 2: collect the 3-arm incumbent-relative causal branching bank.

Implements CCP_S2_SPEC.json#CAUSAL_ESTIMAND / #COLLECTION_DISTRIBUTION against the frozen
CCP_S2_STATE_MANIFEST.json. Selection is closed; this stage is pure measurement.

Job hierarchy, asserted before rollout #1 rather than only totalled:

    128 states -> 212 state-estimand -> 636 arm cells -> 10176 jobs

    one-free states (43/pole)   x1 estimand x3 arms x16 = 2064 / pole
    both-free states (21/pole)  x3 estimands x3 arms x16 = 3024 / pole

Estimand and intervened-agent semantics mirror the sealed predecessor
(experiments/ccp_phase1_collect.py) exactly -- one-free states get the one estimand
matching the free agent, both-free states get agent0/agent1/joint.

Three-arm causal estimand (CCP_S2_SPEC.json#CAUSAL_ESTIMAND), replacing the predecessor's
pairwise pi_A-vs-pi_B ranking with an incumbent-relative advantage:

    R_0    incumbent latent policy continues normally -- the baseline. No teacher is
           consulted at all; both agents are driven by the incumbent under its own
           pole-matched latent (z0 on Pole A, z1 on Pole B) for the whole continuation.
    pi_A   pi_A supplies the intervened agent(s)' action at EVERY step from the boundary
           to termination (SEQUENCE takeover, the only mode this program uses). The
           incumbent continues to drive every non-intervened agent.
    pi_B   identical, with pi_B.

Continuation seeds: r_j = H('CCP_S2_MEASURE', state_id, estimand, j), independent of arm --
R_0, pi_A and pi_B all receive the IDENTICAL r_j for a given (state_id, estimand, j). This is
the three-arm generalisation of the predecessor's two-arm r(s,e,pi_A,j) == r(s,e,pi_B,j)
invariant, asserted structurally in validate() before any job runs.

The incumbent is queried at every prefix step so it arrives at s_t with the internal state
it would really have (it carries a latent and a temporal tracker). The SAPPO teachers are
verified STATELESS (CCP_PHASE1_CAUSAL_BRANCHING_SPEC.json), so they need no prefix warm-up
and are only ever queried once branching begins.

This file must never import or reference the EVAL manifest/block -- collection code paths
are structurally forbidden from seeing it (CCP_S2_SEED_ASSIGNMENT.json preflight check 5).

Usage:
  python experiments/ccp_s2_collect.py --plan-only
  python experiments/ccp_s2_collect.py --worker K --workers 8 --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
MANIFEST = SD / "CCP_S2_STATE_MANIFEST.json"
FROZEN_SUCCESSOR = SD / "CCP_SUCCESSOR_MODEL_FROZEN.json"
PHASE1_MANIFEST = SD / "CCP_PHASE1_PILOT_MANIFEST.json"           # SAPPO teacher checkpoints, reused
ROWS_DIR = SD / "ccp_s2_rows"
PLAN = SD / "CCP_S2_JOB_PLAN.json"

M = 16
ARMS = ("R_0", "pi_A", "pi_B")
POLE_LATENT = {"A": 0, "B": 1}
EXPECT = {"states": 128, "state_estimand": 212, "arm_cells": 636, "jobs": 10176}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def continuation_seed(state_id: str, estimand: str, j: int) -> int:
    """The frozen CCP_S2_MEASURE mapping. Arm identity does not enter it -- matches
    experiments/ccp_s2_seed_preflight.py's measure_seed() byte for byte."""
    h = hashlib.sha256(f"CCP_S2_MEASURE|{state_id}|{estimand}|{j}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2 ** 63 - 1)


def estimands_for(free_set: str) -> list[str]:
    if free_set == "agent0_only":
        return ["agent0"]
    if free_set == "agent1_only":
        return ["agent1"]
    if free_set == "both_free":
        return ["agent0", "agent1", "joint"]
    raise SystemExit(f"REFUSING: unknown free_set {free_set!r}")


def intervened_agents(estimand: str) -> tuple[int, ...]:
    return {"agent0": (0,), "agent1": (1,), "joint": (0, 1)}[estimand]


def build_jobs(manifest: dict) -> tuple[list[dict], dict]:
    states = manifest["states"]
    jobs, se, ac = [], set(), set()
    for st in states:
        for e in estimands_for(st["free_set"]):
            se.add((st["state_id"], e))
            for arm in ARMS:
                ac.add((st["state_id"], e, arm))
                for j in range(M):
                    jobs.append({
                        "job_id": f"{st['state_id']}|{e}|{arm}|{j}",
                        "state_id": st["state_id"], "seed": st["seed"], "pole": st["pole"],
                        "prefix_len": st["prefix_len"], "free_set": st["free_set"],
                        "phase": st["phase"], "estimand": e, "arm": arm, "j": j,
                        "r_j": continuation_seed(st["state_id"], e, j),
                    })
    counts = {"states": len(states), "state_estimand": len(se), "arm_cells": len(ac),
              "jobs": len(jobs)}
    return jobs, counts


def validate(jobs: list[dict], counts: dict) -> None:
    for k, want in EXPECT.items():
        if counts[k] != want:
            raise SystemExit(f"REFUSING: hierarchy mismatch at {k}: got {counts[k]}, expected {want}")
    if len({j["job_id"] for j in jobs}) != len(jobs):
        raise SystemExit("REFUSING: duplicate job ids")
    # three-arm CRN: identical r_j across R_0/pi_A/pi_B for every (state, estimand, j) cell,
    # and all three arms must actually be present -- a missing arm is as much a break in the
    # matched design as a mismatched seed
    by: dict = {}
    for j in jobs:
        by.setdefault((j["state_id"], j["estimand"], j["j"]), {})[j["arm"]] = j["r_j"]
    bad = [k for k, v in by.items()
           if set(v.keys()) != set(ARMS) or len(set(v.values())) != 1]
    if bad:
        raise SystemExit(f"REFUSING: {len(bad)} cells where the three arms do not share an "
                         "identical r_j (or an arm is missing); arms would not be matched")


def outcome(core, info) -> dict:
    """Identical extraction to the predecessor and to eval_ccp_successor.py."""
    terminal = None
    if info is not None:
        i0 = info[0] if isinstance(info, (list, tuple)) else info
        res = (i0 or {}).get("episode_result") or {}
        if res:
            terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
    if terminal is None:
        terminal = (int(core.blue_score[0]), int(core.red_score[0]))
    blue, red = terminal
    return {"blue": blue, "red": red, "win": int(blue > red), "margin": blue - red}


def setup_env(R2, P0, phase_from_tag, install_keyed_opponent_overlays, pole_A_genome,
              assert_live_opponent_batch, device: str, seed: int, pole: str):
    env = R2.build_env(device, seed)
    core = env.core
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
                               context=f"s2 collect {pole} seed {seed}")
    return env, obs, core


def replay_prefix(R2, incumbent, env, obs, actions: list, z: int, job_id: str):
    """Replay the recorded prefix under the incumbent, asserting live agreement at every
    step rather than trusting the stored actions blindly -- REFUSES on any divergence or on
    the episode ending before the boundary the manifest recorded."""
    incumbent.fixed_latent_strategy = True
    incumbent.fixed_latent_strategy_id = int(z)
    incumbent.reset_strategy()
    for i, want in enumerate(actions):
        a, _ = incumbent.predict(obs, deterministic=True)
        got = [int(x) for x in np.asarray(a).ravel()]
        if got != list(want):
            raise SystemExit(f"REFUSING: prefix divergence at step {i} of {job_id}: "
                             f"{got} != {list(want)}")
        env.step_async(a)
        obs, _r, done, _i = env.step_wait()
        obs["global_state"] = env.state()
        if bool(np.asarray(done).any()):
            raise SystemExit(f"REFUSING: episode ended inside the prefix for {job_id}")
    return obs


def run(job: dict, incumbent, teachers: dict, R2, device: str, states_by_id: dict,
        env_ctx: dict) -> dict:
    st = states_by_id[job["state_id"]]
    pole, z = job["pole"], POLE_LATENT[job["pole"]]
    env, obs, core = setup_env(R2, env_ctx["P0"], env_ctx["phase_from_tag"],
                               env_ctx["install_keyed_opponent_overlays"],
                               env_ctx["pole_A_genome"], env_ctx["assert_live_opponent_batch"],
                               device, job["seed"], pole)
    try:
        spec_pol = teachers.get(job["arm"])                       # None for R_0
        if spec_pol is not None:
            if getattr(spec_pol, "fixed_latent_strategy", None) is not None:
                spec_pol.fixed_latent_strategy = False
            spec_pol.reset_strategy()

        obs = replay_prefix(R2, incumbent, env, obs, st["actions"], z, job["job_id"])

        f0 = bool((core.blue_commit_ticks_left[0, 0] <= 0).item())
        f1 = bool((core.blue_commit_ticks_left[0, 1] <= 0).item())
        expect = {"agent0_only": (True, False), "agent1_only": (False, True),
                  "both_free": (True, True)}[job["free_set"]]
        if (f0, f1) != expect:
            raise SystemExit(f"REFUSING: free set at s_t is {(f0, f1)}, manifest says "
                             f"{job['free_set']} for {job['job_id']}")

        core._rng.manual_seed(int(job["r_j"]))                    # ONLY the env RNG
        targets = intervened_agents(job["estimand"]) if spec_pol is not None else ()
        info, steps = None, 0
        for _ in range(R2.MAX_STEPS):
            a_inc, _ = incumbent.predict(obs, deterministic=True)
            act = np.asarray(a_inc).ravel().copy()
            if spec_pol is not None:
                a_sp, _ = spec_pol.predict(obs, deterministic=True)
                sp = np.asarray(a_sp).ravel()
                for i in targets:
                    act[i * 2] = sp[i * 2]
                    act[i * 2 + 1] = sp[i * 2 + 1]
            env.step_async(act)
            obs, _r, done, info = env.step_wait()
            obs["global_state"] = env.state()
            steps += 1
            if bool(np.asarray(done).any()):
                break
        res = outcome(core, info)
        res.update({"job_id": job["job_id"], "state_id": job["state_id"],
                    "estimand": job["estimand"], "arm": job["arm"], "j": job["j"],
                    "r_j": job["r_j"], "continuation_steps": steps, "utc": _now()})
        return res
    finally:
        env.close()


def load_runtime(device: str):
    """Everything the worker and the smoke test both need: incumbent + teachers, sha-verified,
    plus the env-construction helpers. Factored so the smoke gate exercises the SAME loaded
    objects and code paths as the real collector, not a re-derivation of them."""
    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome)
    import experiments.phase0_collect_scorer_data as P0
    import experiments.r2_learned_crossover as R2
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy

    device = device if torch.cuda.is_available() or device == "cpu" else "cpu"

    frozen = json.loads(FROZEN_SUCCESSOR.read_text(encoding="utf-8"))
    ck = ROOT / frozen["TERMINAL_CHECKPOINT"]["path"]
    got = hashlib.sha256(ck.read_bytes()).hexdigest()
    if got != frozen["TERMINAL_CHECKPOINT"]["sha256"]:
        raise SystemExit("REFUSING: incumbent checkpoint sha mismatch")
    if frozen["TERMINAL_RECORD_VALIDITY"]["verdict"] != "VALID":
        raise SystemExit("REFUSING: incumbent run was not established VALID")

    p1 = json.loads(PHASE1_MANIFEST.read_text(encoding="utf-8"))
    teach = p1["TEACHER_POLICIES"]
    tpaths = {}
    for name in ("pi_A", "pi_B"):
        p = ROOT / teach[name]["path"]
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != teach[name]["sha256"]:
            raise SystemExit(f"REFUSING: {name} sha mismatch\n  {got}\n  {teach[name]['sha256']}")
        tpaths[name] = p

    probe = R2.build_env(device, 11_700_001)
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()

    incumbent = load_custom_ppo_policy(str(ck), obs_space, act_space, device=device)
    if not getattr(incumbent.model.critic, "private_z_heads", False):
        raise SystemExit("REFUSING: loaded incumbent critic does not have private z heads")
    (incumbent.model if hasattr(incumbent, "model") else incumbent).eval()

    teachers = {}
    for name, p in tpaths.items():
        pol = load_custom_ppo_policy(str(p), obs_space, act_space, device=device)
        (pol.model if hasattr(pol, "model") else pol).eval()
        teachers[name] = pol

    env_ctx = {"P0": P0, "phase_from_tag": phase_from_tag,
               "install_keyed_opponent_overlays": install_keyed_opponent_overlays,
               "pole_A_genome": pole_A_genome,
               "assert_live_opponent_batch": assert_live_opponent_batch}
    return device, incumbent, teachers, R2, env_ctx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: S2 spec not frozen: {spec['status']!r}")
    assignment = json.loads(ASSIGNMENT.read_text(encoding="utf-8"))
    if assignment["status"] != "FROZEN_APPEND_ONLY_AMENDMENT":
        raise SystemExit(f"REFUSING: seed assignment not frozen: {assignment['status']!r}")
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if preflight["VERDICT"] != "PASS":
        raise SystemExit(f"REFUSING: seed preflight is not PASS: {preflight['VERDICT']!r}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["status"] != "FROZEN_SELECTION":
        raise SystemExit(f"REFUSING: state manifest not frozen: {manifest['status']!r}")

    jobs, counts = build_jobs(manifest)
    validate(jobs, counts)

    if args.plan_only:
        one_free = sum(1 for s in manifest["states"] if s["free_set"] != "both_free")
        both = sum(1 for s in manifest["states"] if s["free_set"] == "both_free")
        print(f"CCP-S2 JOB PLAN  {_now()}\n")
        print(f"  hierarchy   {counts['states']} -> {counts['state_estimand']} -> "
              f"{counts['arm_cells']} -> {counts['jobs']}")
        print(f"  expected    128 -> 212 -> 636 -> 10176     "
              f"{'MATCH' if counts == EXPECT else 'MISMATCH'}")
        print(f"\n  one-free states {one_free}  x1 estimand x3 arms x{M} = {one_free*1*3*M}")
        print(f"  both-free states {both}  x3 estimands x3 arms x{M} = {both*3*3*M}")
        print(f"  total {counts['jobs']}")
        print(f"\n  three-arm CRN: r_j identical across R_0/pi_A/pi_B in all "
              f"{counts['arm_cells']//3*M} (state,estimand,j) cells  OK")
        print(f"  seed mapping depends on (state_id, estimand, j) only -- not arm")
        by_arm = {a: sum(1 for x in jobs if x["arm"] == a) for a in ARMS}
        by_est: dict = {}
        for x in jobs:
            by_est[x["estimand"]] = by_est.get(x["estimand"], 0) + 1
        print(f"\n  jobs by arm       {by_arm}")
        print(f"  jobs by estimand  {by_est}")
        shards = {k: sum(1 for i, _ in enumerate(jobs) if i % 8 == k) for k in range(8)}
        print(f"  8-worker shards   {shards}")
        PLAN.write_text(json.dumps({
            "record": "CCP-S2 job plan", "status": "FROZEN_PLAN", "utc": _now(),
            "hierarchy": counts, "expected": EXPECT, "matches": counts == EXPECT,
            "M": M, "arms": list(ARMS),
            "one_free_states": one_free, "both_free_states": both,
            "three_arm_seed_pairing_verified": True,
            "seed_mapping": ("r_j = sha256('CCP_S2_MEASURE|<state_id>|<estimand>|<j>')[:8], "
                             "independent of arm"),
            "jobs_by_arm": by_arm, "jobs_by_estimand": by_est,
            "worker_shards": shards,
        }, indent=2), encoding="utf-8")
        print(f"\n  -> {PLAN}")
        return 0

    # ------------------------------------------------------------------ run
    ROWS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ROWS_DIR / f"worker_{args.worker:02d}.jsonl"
    done_ids = set()
    if out_path.is_file():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["job_id"])
    mine = [j for i, j in enumerate(jobs) if i % args.workers == args.worker
            and j["job_id"] not in done_ids]
    print(f"[w{args.worker}] {len(mine)} jobs to run ({len(done_ids)} already done)", flush=True)
    if not mine:
        return 0

    device, incumbent, teachers, R2, env_ctx = load_runtime(args.device)
    states_by_id = {s["state_id"]: s for s in manifest["states"]}

    with out_path.open("a", encoding="utf-8") as fh:
        for n, job in enumerate(mine, 1):
            row = run(job, incumbent, teachers, R2, device, states_by_id, env_ctx)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            if n % 10 == 0 or n == len(mine):
                print(f"[w{args.worker}] {n}/{len(mine)}", flush=True)
    print(f"[w{args.worker}] done -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
