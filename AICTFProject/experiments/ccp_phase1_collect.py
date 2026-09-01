"""CCP Phase 1 stage 2: collect the paired causal branching bank.

Implements CCP_PHASE1_CAUSAL_BRANCHING_SPEC.json and its three amendments against the frozen
manifest CCP_PHASE1_PILOT_MANIFEST.json. Selection is closed; this stage is pure measurement.

Job hierarchy, asserted before rollout #1 rather than only totalled:

    20 states -> 32 state-estimand -> 64 mode cells -> 128 policy cells -> 2048 jobs

    14 one-free states  x 1 estimand  x 2 modes x 2 policies x 16 =  896
     6 both-free states x 3 estimands x 2 modes x 2 policies x 16 = 1152

Intervention semantics:

  one-free state   only the free agent is intervened on; the committed teammate stays on its
                   existing macro and is never overridden
  both-free state  three DISTINCT estimands: agent0 (agent1 stays V4), agent1 (agent0 stays
                   V4), and joint (both get the specialist)

  single_macro     the specialist supplies the action at the first continuation step only.
                   The env ignores actions from a committed agent, so V4 naturally resumes
                   the moment that agent's macro expires -- per agent, at its own duration.
                   No artificial synchronised "joint option end".
  full_takeover    the specialist supplies the intervened agent's action at EVERY step, so it
                   controls every later commitment boundary through termination.

Continuation seeds: the frozen mapping r_j = H('CCP_PHASE1', state_id, j) depends on neither
policy nor estimand nor mode, so r(s,e,m,pi_A,j) == r(s,e,m,pi_B,j) holds a fortiori. Asserted
anyway, because that is what makes A/B a matched pair.

V4 is queried at every prefix step so it arrives at s_t with the internal state it would
really have: it carries a latent and a temporal tracker. The SAPPO teachers were checked and
are STATELESS -- no recurrent modules, uses_latent_strategy False, no selector hidden state --
so they need no prefix warm-up and are queried only when they are actually controlling.

Usage:
  python experiments/ccp_phase1_collect.py --plan-only
  python experiments/ccp_phase1_collect.py --worker K --workers 8 --device cuda
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
SPEC = SD / "CCP_PHASE1_CAUSAL_BRANCHING_SPEC.json"
MANIFEST = SD / "CCP_PHASE1_PILOT_MANIFEST.json"
ROWS_DIR = SD / "ccp_phase1_rows"
PLAN = SD / "CCP_PHASE1_JOB_PLAN.json"

M = 16
MODES = ("single_macro", "full_takeover")
POLICIES = ("pi_A", "pi_B")
EXPECT = {"states": 20, "state_estimand": 32, "mode_cells": 64, "policy_cells": 128, "jobs": 2048}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def continuation_seed(state_id: str, j: int) -> int:
    """The frozen mapping. Policy identity does not enter it."""
    h = hashlib.sha256(f"CCP_PHASE1|{state_id}|{j}".encode()).digest()
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
    jobs, se, mc, pc = [], set(), set(), set()
    for st in states:
        for e in estimands_for(st["free_set"]):
            se.add((st["state_id"], e))
            for mode in MODES:
                mc.add((st["state_id"], e, mode))
                for pol in POLICIES:
                    pc.add((st["state_id"], e, mode, pol))
                    for j in range(M):
                        jobs.append({
                            "job_id": f"{st['state_id']}|{e}|{mode}|{pol}|{j}",
                            "state_id": st["state_id"], "seed": st["seed"], "pole": st["pole"],
                            "prefix_len": st["prefix_len"], "free_set": st["free_set"],
                            "phase": st["phase"], "estimand": e, "mode": mode,
                            "policy": pol, "j": j,
                            "r_j": continuation_seed(st["state_id"], j),
                        })
    counts = {"states": len(states), "state_estimand": len(se), "mode_cells": len(mc),
              "policy_cells": len(pc), "jobs": len(jobs)}
    return jobs, counts


def validate(jobs: list[dict], counts: dict) -> None:
    for k, want in EXPECT.items():
        if counts[k] != want:
            raise SystemExit(f"REFUSING: hierarchy mismatch at {k}: got {counts[k]}, expected {want}")
    if len({j["job_id"] for j in jobs}) != len(jobs):
        raise SystemExit("REFUSING: duplicate job ids")
    # A/B pairing: identical r_j across policies for every (state, estimand, mode, j)
    by = {}
    for j in jobs:
        by.setdefault((j["state_id"], j["estimand"], j["mode"], j["j"]), {})[j["policy"]] = j["r_j"]
    bad = [k for k, v in by.items() if v.get("pi_A") != v.get("pi_B")]
    if bad:
        raise SystemExit(f"REFUSING: {len(bad)} cells where pi_A and pi_B differ in r_j; "
                         "A/B would not be a matched pair")
    if len(by) != EXPECT["policy_cells"] // 2 * M // M * (EXPECT["mode_cells"]):
        pass  # structural count already covered by EXPECT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["status"] != "FROZEN_SELECTION":
        raise SystemExit(f"REFUSING: manifest not frozen: {manifest['status']!r}")
    jobs, counts = build_jobs(manifest)
    validate(jobs, counts)

    if args.plan_only:
        one_free = sum(1 for s in manifest["states"] if s["free_set"] != "both_free")
        both = sum(1 for s in manifest["states"] if s["free_set"] == "both_free")
        print(f"CCP PHASE 1 JOB PLAN  {_now()}\n")
        print(f"  hierarchy   {counts['states']} -> {counts['state_estimand']} -> "
              f"{counts['mode_cells']} -> {counts['policy_cells']} -> {counts['jobs']}")
        print(f"  expected    20 -> 32 -> 64 -> 128 -> 2048     "
              f"{'MATCH' if counts == EXPECT else 'MISMATCH'}")
        print(f"\n  one-free states {one_free}  x1 estimand  x2 modes x2 policies x{M} = "
              f"{one_free*1*2*2*M}")
        print(f"  both-free states {both}  x3 estimands x2 modes x2 policies x{M} = "
              f"{both*3*2*2*M}")
        print(f"  total {counts['jobs']}")
        print(f"\n  A/B seed pairing: r_j identical across pi_A/pi_B in all "
              f"{counts['mode_cells']*M} (state,estimand,mode,j) cells  OK")
        print(f"  seed mapping depends on (state_id, j) only -- not policy, estimand or mode")
        by_mode = {m: sum(1 for x in jobs if x["mode"] == m) for m in MODES}
        by_est = {}
        for x in jobs:
            by_est[x["estimand"]] = by_est.get(x["estimand"], 0) + 1
        print(f"\n  jobs by mode      {by_mode}")
        print(f"  jobs by estimand  {by_est}")
        shards = {k: sum(1 for i, _ in enumerate(jobs) if i % 8 == k) for k in range(8)}
        print(f"  8-worker shards   {shards}")
        PLAN.write_text(json.dumps({
            "record": "CCP Phase 1 job plan", "status": "FROZEN_PLAN", "utc": _now(),
            "hierarchy": counts, "expected": EXPECT, "matches": counts == EXPECT,
            "M": M, "modes": list(MODES), "policies": list(POLICIES),
            "one_free_states": one_free, "both_free_states": both,
            "ab_seed_pairing_verified": True,
            "seed_mapping": "r_j = sha256('CCP_PHASE1|<state_id>|<j>')[:8], independent of policy/estimand/mode",
            "jobs_by_mode": by_mode, "jobs_by_estimand": by_est,
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

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome)
    import experiments.phase0_collect_scorer_data as P0
    import experiments.r2_learned_crossover as R2
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    frozen = json.loads((SD / "HOG_PSP_V4_MODEL_FROZEN.json").read_text(encoding="utf-8"))
    v4_ck = ROOT / frozen["TERMINAL_CHECKPOINT"]["path"]

    # teacher checkpoints, sha re-verified before rollout #1 -- fatal on mismatch
    teach = manifest["TEACHER_POLICIES"]
    tpaths = {}
    for name in POLICIES:
        p = ROOT / teach[name]["path"]
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != teach[name]["sha256"]:
            raise SystemExit(f"REFUSING: {name} sha mismatch\n  {got}\n  {teach[name]['sha256']}")
        tpaths[name] = p

    probe = R2.build_env(device, manifest["states"][0]["seed"])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    v4 = load_custom_ppo_policy(str(v4_ck), obs_space, act_space, device=device)
    (v4.model if hasattr(v4, "model") else v4).eval()
    teachers = {}
    for name, p in tpaths.items():
        pol = load_custom_ppo_policy(str(p), obs_space, act_space, device=device)
        (pol.model if hasattr(pol, "model") else pol).eval()
        teachers[name] = pol
    POLE_LATENT = {"A": 0, "B": 1}
    states_by_id = {s["state_id"]: s for s in manifest["states"]}

    def outcome(core, info) -> dict:
        """EVAL's exact extraction: episode_result at done, core scores as fallback."""
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

    def run(job: dict) -> dict:
        st = states_by_id[job["state_id"]]
        pole, z = job["pole"], POLE_LATENT[job["pole"]]
        env = R2.build_env(device, job["seed"])
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
            assert_live_opponent_batch(core, genomes, allowed_keys=(key,), context=job["job_id"])

            v4.fixed_latent_strategy = True
            v4.fixed_latent_strategy_id = z
            v4.reset_strategy()
            spec_pol = teachers[job["policy"]]
            if getattr(spec_pol, "fixed_latent_strategy", None) is not None:
                spec_pol.fixed_latent_strategy = False
            spec_pol.reset_strategy()

            # --- prefix: V4 drives and is queried every step so it reaches s_t with the
            #     internal state it would really have (it carries a latent and a temporal
            #     tracker). The SAPPO teachers are verified STATELESS -- no recurrent modules,
            #     uses_latent_strategy False, no selector hidden state -- so stepping them
            #     through the prefix would change nothing and cost ~14 ms/step.
            #     Stored actions are asserted against V4's live output, not replayed blindly.
            for i, want in enumerate(st["actions"]):
                a, _ = v4.predict(obs, deterministic=True)
                got = [int(x) for x in np.asarray(a).ravel()]
                if got != list(want):
                    raise SystemExit(f"REFUSING: prefix divergence at step {i} of "
                                     f"{job['job_id']}: {got} != {list(want)}")
                env.step_async(a)
                obs, _r, done, _i = env.step_wait()
                obs["global_state"] = env.state()
                if bool(np.asarray(done).any()):
                    raise SystemExit(f"REFUSING: episode ended inside the prefix for {job['job_id']}")

            # --- the free set must be what the manifest recorded
            f0 = bool((core.blue_commit_ticks_left[0, 0] <= 0).item())
            f1 = bool((core.blue_commit_ticks_left[0, 1] <= 0).item())
            expect = {"agent0_only": (True, False), "agent1_only": (False, True),
                      "both_free": (True, True)}[job["free_set"]]
            if (f0, f1) != expect:
                raise SystemExit(f"REFUSING: free set at s_t is {(f0,f1)}, manifest says "
                                 f"{job['free_set']} for {job['job_id']}")

            core._rng.manual_seed(int(job["r_j"]))                 # ONLY the env RNG
            targets = intervened_agents(job["estimand"])
            first, info, steps = True, None, 0
            for _ in range(R2.MAX_STEPS):
                a_v4, _ = v4.predict(obs, deterministic=True)
                act = np.asarray(a_v4).ravel().copy()
                if job["mode"] == "full_takeover" or first:
                    a_sp, _ = spec_pol.predict(obs, deterministic=True)
                    sp = np.asarray(a_sp).ravel()
                    for i in targets:
                        act[i * 2] = sp[i * 2]
                        act[i * 2 + 1] = sp[i * 2 + 1]
                first = False
                env.step_async(act)
                obs, _r, done, info = env.step_wait()
                obs["global_state"] = env.state()
                steps += 1
                if bool(np.asarray(done).any()):
                    break
            res = outcome(core, info)
            res.update({"job_id": job["job_id"], "state_id": job["state_id"],
                        "estimand": job["estimand"], "mode": job["mode"],
                        "policy": job["policy"], "j": job["j"], "r_j": job["r_j"],
                        "continuation_steps": steps, "utc": _now()})
            return res
        finally:
            env.close()

    with out_path.open("a", encoding="utf-8") as fh:
        for n, job in enumerate(mine, 1):
            row = run(job)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            if n % 10 == 0 or n == len(mine):
                print(f"[w{args.worker}] {n}/{len(mine)}", flush=True)
    print(f"[w{args.worker}] done -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
