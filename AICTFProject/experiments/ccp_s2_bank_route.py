"""CCP-S2 bank construction, stage A: routing and weights (pure computation, no GPU).

Implements CCP_S2_SPEC.json#CAUSAL_ESTIMAND (A_hat_A/A_hat_B, R = win per outcome()/
eval_ccp_successor.py's own V(z,pole) convention), #ROUTING_RULE, and the joint-precedence
clause of #BANK_CONSTRUCTION. Reads only already-collected measurement rows (ccp_s2_rows/
worker_*.jsonl, restricted to job_ids in the amended plan) and never touches the GPU or the
environment. It does NOT perform the canonical r_bank(x) trajectory rollout
(ANTI_TRAJECTORY_SHOPPING_RULE) -- that is a separate stage (ccp_s2_bank_rollout.py, not yet
built) that needs the runtime loaded and can only run once this routing is frozen.

Per #BANK_CONSTRUCTION ("runs_after_all_measurement_and_routing_is_frozen"), this script
REFUSES to freeze anything until all 104 amended state-estimand units have complete 3-arm x
M=8 measurement. Before that it only prints a dry-run status report -- safe to re-run at any
time, including while collection is still in flight, and writes nothing until collection is
done.

Run:  python experiments/ccp_s2_bank_route.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.ccp_s2_collect as C
import experiments.ccp_s2_collect_amended as A

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
AMENDMENT = SD / "CCP_S2_COMPUTE_BUDGET_AMENDMENT.json"
MANIFEST = SD / "CCP_S2_STATE_MANIFEST_AMENDMENT.json"
ROWS_DIR = SD / "ccp_s2_rows"
OUT = SD / "CCP_S2_CAUSAL_BANK_ROUTING.json"

M = A.M                                        # 8, the amended horizon


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_measured(amended_ids: set) -> dict:
    """(state_id, estimand) -> {arm: {j: win}}, restricted to amended-scope job_ids only."""
    by_unit: dict = defaultdict(lambda: defaultdict(dict))
    seen_ids = set()
    for p in sorted(ROWS_DIR.glob("worker_*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            jid = row["job_id"]
            if jid not in amended_ids:
                continue                                       # archived_unused, out of scope
            if jid in seen_ids:
                raise SystemExit(f"REFUSING: job_id {jid} written more than once")
            seen_ids.add(jid)
            key = (row["state_id"], row["estimand"])
            by_unit[key][row["arm"]][row["j"]] = row["win"]
    return by_unit


def route_unit(win_by_arm: dict) -> tuple[str | None, float, float, float, bool]:
    """-> (t_star or None, w, A_hat_A, A_hat_B, exact_tie)"""
    r0 = win_by_arm["R_0"]
    A_hat_A = sum(win_by_arm["pi_A"][j] - r0[j] for j in range(M)) / M
    A_hat_B = sum(win_by_arm["pi_B"][j] - r0[j] for j in range(M)) / M
    exact_tie = A_hat_A == A_hat_B and A_hat_A > 0
    if exact_tie:
        return None, 0.0, A_hat_A, A_hat_B, True
    if A_hat_A > max(0.0, A_hat_B):
        return "pi_A", A_hat_A, A_hat_A, A_hat_B, False
    if A_hat_B > max(0.0, A_hat_A):
        return "pi_B", A_hat_B, A_hat_A, A_hat_B, False
    return None, 0.0, A_hat_A, A_hat_B, False


def main() -> int:
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if amendment["status"] != "FROZEN_APPEND_ONLY_AMENDMENT":
        raise SystemExit(f"REFUSING: compute-budget amendment not frozen: {amendment['status']!r}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["status"] != "FROZEN_SELECTION":
        raise SystemExit(f"REFUSING: amended manifest not frozen: {manifest['status']!r}")
    states_by_id = {s["state_id"]: s for s in manifest["states"]}

    jobs, counts = A.build_jobs(manifest)
    A.validate(jobs, counts)
    amended_ids = {j["job_id"] for j in jobs}
    n_units_expected = counts["state_estimand"]                # 104

    by_unit = load_measured(amended_ids)

    complete, incomplete = {}, []
    for key, win_by_arm in by_unit.items():
        ok = (set(win_by_arm.keys()) == set(C.ARMS)
              and all(set(win_by_arm[arm].keys()) == set(range(M)) for arm in C.ARMS))
        if ok:
            complete[key] = win_by_arm
        else:
            incomplete.append(key)

    n_ready = len(complete)
    print(f"CCP-S2 BANK ROUTING  {_now()}")
    print(f"  units with complete 3-arm x M={M} measurement: {n_ready} / {n_units_expected}")
    if n_ready < n_units_expected:
        missing = n_units_expected - n_ready
        print(f"  {missing} units still incomplete -- DRY RUN ONLY, nothing will be frozen")
        print(f"  (re-run this script once collection finishes; it is idempotent and safe to "
              f"call any time)")

    # -------------------------------------------------------- route every complete unit
    routed = {}                                    # (state_id, estimand) -> record
    exact_tie_count = 0
    for key, win_by_arm in complete.items():
        state_id, estimand = key
        t_star, w, ahA, ahB, tie = route_unit(win_by_arm)
        if tie:
            exact_tie_count += 1
        routed[key] = {
            "state_id": state_id, "estimand": estimand,
            "free_set": states_by_id[state_id]["free_set"],
            "pole": states_by_id[state_id]["pole"],
            "A_hat_A": ahA, "A_hat_B": ahB, "t_star": t_star, "w": w, "exact_tie": tie,
        }

    # -------------------------------------------------- joint precedence (per BANK_CONSTRUCTION)
    # at a both_free state, a usable joint record (w>0) supersedes that state's individual
    # agent0/agent1 records; if no usable joint record exists, individual records may contribute
    by_state: dict = defaultdict(dict)
    for (state_id, estimand), rec in routed.items():
        by_state[state_id][estimand] = rec

    final_bank = {}
    superseded = []
    for state_id, ests in by_state.items():
        joint = ests.get("joint")
        if joint is not None and joint["w"] > 0:
            final_bank[(state_id, "joint")] = joint
            for e in ("agent0", "agent1"):
                if e in ests:
                    superseded.append((state_id, e))
        else:
            for e, rec in ests.items():
                if e == "joint" and joint is not None:
                    final_bank[(state_id, "joint")] = joint      # w<=0, kept for the record
                elif e != "joint":
                    final_bank[(state_id, e)] = rec

    # -------------------------------------------------------------------------- report
    def _stats(units: dict) -> dict:
        n = len(units)
        rA = sum(1 for r in units.values() if r["t_star"] == "pi_A")
        rB = sum(1 for r in units.values() if r["t_star"] == "pi_B")
        rZ = n - rA - rB
        posA = [r["A_hat_A"] for r in units.values() if r["A_hat_A"] > 0]
        posB = [r["A_hat_B"] for r in units.values() if r["A_hat_B"] > 0]
        return {
            "candidate_boundary_units": n,
            "routed_to_pi_A": rA, "routed_to_pi_B": rB, "weight_zero": rZ,
            "mean_positive_A_hat_A": (sum(posA) / len(posA)) if posA else None,
            "mean_positive_A_hat_B": (sum(posB) / len(posB)) if posB else None,
            "n_positive_A_hat_A": len(posA), "n_positive_A_hat_B": len(posB),
        }

    # note: not superseded by joint precedence == what actually enters L_causal
    active = {k: v for k, v in final_bank.items()
              if not (v["estimand"] != "joint" and (v["state_id"], v["estimand"]) in superseded)}
    overall = _stats(active)
    by_topology = {
        "individual_agent_boundaries (agent0_only + agent1_only states)":
            _stats({k: v for k, v in active.items() if v["free_set"] != "both_free"}),
        "joint_free_boundaries (both_free states)":
            _stats({k: v for k, v in active.items() if v["free_set"] == "both_free"}),
    }

    print(f"\n  --- overall (post joint-precedence, {len(active)} active units) ---")
    for k, v in overall.items():
        print(f"  {k:28s} {v}")
    print(f"\n  exact_tie_rate: {exact_tie_count}/{n_ready} = "
          f"{exact_tie_count/n_ready:.4f}" if n_ready else "  exact_tie_rate: n/a")
    print(f"  superseded-by-joint (individual units dropped): {len(superseded)}")
    for label, stats in by_topology.items():
        print(f"\n  --- {label} ---")
        for k, v in stats.items():
            print(f"  {k:28s} {v}")

    if n_ready < n_units_expected:
        print(f"\n  NOT FROZEN -- {n_units_expected - n_ready} units still pending collection")
        return 0

    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; one-shot, freeze once")

    OUT.write_text(json.dumps({
        "record": "CCP-S2 causal bank routing", "status": "FROZEN_RESULT", "one_shot": True,
        "utc": _now(),
        "implements": "CCP_S2_SPEC.json#CAUSAL_ESTIMAND + #ROUTING_RULE + "
                      "#BANK_CONSTRUCTION.joint_precedence_preserved",
        "note": "this record freezes ROUTING AND WEIGHTS only. The canonical r_bank(x) "
                "trajectory rollout (ANTI_TRAJECTORY_SHOPPING_RULE) that supplies (obs,action) "
                "training data is a separate, later, GPU-bound stage -- not run here.",
        "n_units_expected": n_units_expected, "n_units_ready": n_ready,
        "M": M, "R_metric": "win (binary), identical convention to eval_ccp_successor.py's V(z,pole)",
        "exact_tie_count": exact_tie_count,
        "exact_tie_rate": exact_tie_count / n_ready if n_ready else None,
        "superseded_by_joint_precedence": [f"{s}|{e}" for s, e in superseded],
        "overall": overall,
        "by_topology": {k: v for k, v in by_topology.items()},
        "units": [
            {**v, "unit": f"{v['state_id']}|{v['estimand']}"}
            for v in active.values()
        ],
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
