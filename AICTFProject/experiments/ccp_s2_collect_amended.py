"""CCP-S2 compute-budget amendment, stage 2: the amended collector.

Implements CCP_S2_COMPUTE_BUDGET_AMENDMENT.json#REDUCTION: 64 states (32/pole, rank-order
truncated in CCP_S2_STATE_MANIFEST_AMENDMENT.json) x M=8 instead of the original 128 states x
M=16. This file reuses experiments/ccp_s2_collect.py's actual measurement code -- load_runtime,
setup_env, replay_prefix, run, outcome, continuation_seed, estimands_for, intervened_agents,
ARMS, POLE_LATENT -- verbatim, unchanged, imported rather than reimplemented. The
COLLECTOR_VERIFIED gate already proved that code correct; only the job LIST (which states, how
many j per state) changes here, not the branching/replay logic itself.

Writes to the SAME ccp_s2_rows/ directory as the original collector, so any row already
completed under the original 10176-job plan that happens to also belong to this amended plan is
recognised as already-done by the existing per-worker resume logic and is not re-run. Rows that
belong to the original plan but not this one are simply inert lines in those files -- present,
harmless, unused. See CCP_S2_AMENDMENT_RECONCILIATION.json for the frozen accounting of which is
which; worker mode below refuses to start until that reconciliation is frozen.

Usage:
  python experiments/ccp_s2_collect_amended.py --plan-only
  python experiments/ccp_s2_collect_amended.py --worker K --workers 4 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.ccp_s2_collect as C

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
AMENDMENT = SD / "CCP_S2_COMPUTE_BUDGET_AMENDMENT.json"
MANIFEST = SD / "CCP_S2_STATE_MANIFEST_AMENDMENT.json"
RECONCILIATION = SD / "CCP_S2_AMENDMENT_RECONCILIATION.json"
ROWS_DIR = SD / "ccp_s2_rows"                 # SAME directory as the original collector
PLAN = SD / "CCP_S2_JOB_PLAN_AMENDMENT.json"

M = 8                                          # was 16
EXPECT = {"states": 64, "state_estimand": 104, "arm_cells": 312, "jobs": 2496}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_jobs(manifest: dict) -> tuple[list[dict], dict]:
    """Identical shape to C.build_jobs, only M and the source states differ."""
    states = manifest["states"]
    jobs, se, ac = [], set(), set()
    for st in states:
        for e in C.estimands_for(st["free_set"]):
            se.add((st["state_id"], e))
            for arm in C.ARMS:
                ac.add((st["state_id"], e, arm))
                for j in range(M):
                    jobs.append({
                        "job_id": f"{st['state_id']}|{e}|{arm}|{j}",
                        "state_id": st["state_id"], "seed": st["seed"], "pole": st["pole"],
                        "prefix_len": st["prefix_len"], "free_set": st["free_set"],
                        "phase": st["phase"], "estimand": e, "arm": arm, "j": j,
                        "r_j": C.continuation_seed(st["state_id"], e, j),
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
    by: dict = {}
    for j in jobs:
        by.setdefault((j["state_id"], j["estimand"], j["j"]), {})[j["arm"]] = j["r_j"]
    bad = [k for k, v in by.items()
           if set(v.keys()) != set(C.ARMS) or len(set(v.values())) != 1]
    if bad:
        raise SystemExit(f"REFUSING: {len(bad)} cells where the three arms do not share an "
                         "identical r_j (or an arm is missing)")
    # every j used here must be < the original M -- this amendment only STOPS using j>=8, it
    # must never invent a j the frozen seed mapping wasn't already defined for
    if any(j["j"] >= 16 for j in jobs):
        raise SystemExit("REFUSING: a job references j outside the frozen CCP_S2_MEASURE range")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if amendment["status"] != "FROZEN_APPEND_ONLY_AMENDMENT":
        raise SystemExit(f"REFUSING: compute-budget amendment not frozen: {amendment['status']!r}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["status"] != "FROZEN_SELECTION":
        raise SystemExit(f"REFUSING: amended manifest not frozen: {manifest['status']!r}")

    jobs, counts = build_jobs(manifest)
    validate(jobs, counts)

    if args.plan_only:
        print(f"CCP-S2 AMENDED JOB PLAN  {_now()}\n")
        print(f"  hierarchy   {counts['states']} -> {counts['state_estimand']} -> "
              f"{counts['arm_cells']} -> {counts['jobs']}")
        print(f"  expected    64 -> 104 -> 312 -> 2496     "
              f"{'MATCH' if counts == EXPECT else 'MISMATCH'}")
        print(f"  three-arm CRN: r_j identical across R_0/pi_A/pi_B in all "
              f"{counts['arm_cells']//3*M} (state,estimand,j) cells  OK")
        by_arm = {a: sum(1 for x in jobs if x["arm"] == a) for a in C.ARMS}
        by_est: dict = {}
        for x in jobs:
            by_est[x["estimand"]] = by_est.get(x["estimand"], 0) + 1
        print(f"  jobs by arm       {by_arm}")
        print(f"  jobs by estimand  {by_est}")
        shards = {k: sum(1 for i, _ in enumerate(jobs) if i % 4 == k) for k in range(4)}
        print(f"  4-worker shards   {shards}")
        PLAN.write_text(json.dumps({
            "record": "CCP-S2 amended job plan", "status": "FROZEN_PLAN", "utc": _now(),
            "implements": "CCP_S2_COMPUTE_BUDGET_AMENDMENT.json#REDUCTION",
            "hierarchy": counts, "expected": EXPECT, "matches": counts == EXPECT,
            "M": M, "arms": list(C.ARMS),
            "three_arm_seed_pairing_verified": True,
            "seed_mapping": ("r_j = sha256('CCP_S2_MEASURE|<state_id>|<estimand>|<j>')[:8], "
                             "unchanged formula, j restricted to 0..7"),
            "jobs_by_arm": by_arm, "jobs_by_estimand": by_est,
            "worker_shards": shards,
        }, indent=2), encoding="utf-8")
        print(f"\n  -> {PLAN}")
        return 0

    reconciliation = json.loads(RECONCILIATION.read_text(encoding="utf-8"))
    if reconciliation["status"] != "FROZEN_RESULT":
        raise SystemExit(f"REFUSING: amendment reconciliation not frozen: "
                         f"{reconciliation['status']!r}")

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
    print(f"[w{args.worker}] {len(mine)} amended jobs to run "
          f"({sum(1 for i, j in enumerate(jobs) if i % args.workers == args.worker) - len(mine)} "
          "already done)", flush=True)
    if not mine:
        return 0

    device, incumbent, teachers, R2, env_ctx = C.load_runtime(args.device)
    states_by_id = {s["state_id"]: s for s in manifest["states"]}

    with out_path.open("a", encoding="utf-8") as fh:
        for n, job in enumerate(mine, 1):
            row = C.run(job, incumbent, teachers, R2, device, states_by_id, env_ctx)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            if n % 10 == 0 or n == len(mine):
                print(f"[w{args.worker}] {n}/{len(mine)}", flush=True)
    print(f"[w{args.worker}] done -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
