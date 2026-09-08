"""C5 discovery re-analysis and frozen candidate ranking.

Runs offline from the raw states the scan persisted, so it needs no GPU replay.
This is the AUTHORITATIVE C5 discovery analysis: the scan's inline result was
produced by the module as loaded before the candidate-key collision fix (7029942)
and is superseded.

Implements the ranking function frozen in C5_RANKING_AMENDMENT.json (deb3701):
    reversal_strength = min(lcb95_A, lcb95_B), descending
    tie-break 1: higher min(support_A, support_B)
    tie-break 2: lexicographic on (opponent_a, opponent_b, response_1, response_2)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_c4_opportunity_cost as M  # noqa: E402
C4_FROZEN = ROOT / "artifacts/c4_preregistration/C4_OPPORTUNITY_COST_FROZEN.json"
C5_FROZEN = ROOT / "artifacts/c5_preregistration/C5_OPPONENT_DEMAND_FROZEN.json"
C5_AMEND = ROOT / "artifacts/c5_preregistration/C5_RANKING_AMENDMENT.json"
OBS_FROZEN = ROOT / "artifacts/c5_preregistration/OBSERVABILITY_TEST_FROZEN.json"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_commit_of(p: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(p)],
            cwd=ROOT, capture_output=True, text=True, timeout=30)
        return (out.stdout or "").strip() or "unknown"
    except Exception:
        return "unknown"


def rank_key(cand: dict) -> tuple:
    """The frozen ranking function. Sorted ascending on this key gives the
    frozen descending-strength order, so index 0 is the selected candidate."""
    a = cand["side_a_R1_minus_R2"]
    b = cand["side_b_R2_minus_R1"]
    strength = min(float(a["lcb95"]), float(b["lcb95"]))
    support = min(int(cand["n_states_side_a"]), int(cand["n_states_side_b"]))
    return (
        -strength,                       # descending reversal strength
        -support,                        # tie-break 1: descending support
        str(cand["side_a"]), str(cand["side_b"]),   # tie-break 2: lexicographic
        str(cand["response_1"]), str(cand["response_2"]),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default=str(ROOT / "artifacts/c5_discovery/states.json"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/c5_discovery/C5_RESULT.json"))
    args = ap.parse_args()

    frozen = json.loads(C4_FROZEN.read_text(encoding="utf-8"))
    states_by_policy = json.loads(Path(args.states).read_text(encoding="utf-8"))

    # Opponent identity as an EVALUATION LABEL only, derived from episode_key
    # (which the scan writes as f"{opponent}:{seed}"). It is never a policy input.
    opponents = set()
    for rows in states_by_policy.values():
        for r in rows:
            opp = r["episode_key"].split(":")[0]
            r["classes"] = {"opponent": opp}
            opponents.add(opp)

    M.PARTITIONS = {"opponent": lambda c: None}
    res = M.analyze(states_by_policy, frozen)

    # Rank ONLY replicated candidates: replication in >=2/3 policies is the
    # discovery criterion, so unreplicated cells are not candidates at all.
    cands = []
    for key, policies in res["replicated"].items():
        detail = next(res["per_policy"][p][key] for p in policies)
        cands.append({"key": key, "replicated_in": sorted(policies), **detail})
    cands.sort(key=rank_key)

    verdict = "C5_PASS_DISCOVERY" if cands else "C5_NO_REVERSAL"

    out = {
        "record": "C5 discovery — opponent-conditional demand screen",
        "authoritative": True,
        "supersedes": "the scan's inline result, produced before the candidate-key "
                      "collision fix 7029942; see C5_RESULT_INLINE_SUPERSEDED.json",
        "verdict": verdict,
        "n_opponents": len(opponents),
        "opponents_observed": sorted(opponents),
        "n_opponent_pairs_examined": len(opponents) * (len(opponents) - 1) // 2,
        "n_candidates_clearing_discovery": len(cands),
        "selected_candidate": cands[0] if cands else None,
        "all_candidates_ranked": cands,
        "per_policy": res["per_policy"],
        "replicated": res["replicated"],
        "confirmation_block_spent": False,
        "provenance": {
            "c4_inherited_statistics_contract_sha256": sha256(C4_FROZEN),
            "c5_freeze_sha256": sha256(C5_FROZEN),
            "c5_ranking_amendment_sha256": sha256(C5_AMEND),
            "observability_freeze_sha256": sha256(OBS_FROZEN),
            "c5_freeze_commit": git_commit_of(C5_FROZEN),
            "c5_amendment_commit": git_commit_of(C5_AMEND),
            "observability_freeze_commit": git_commit_of(OBS_FROZEN),
            "analysis_runner_commit": git_commit_of(Path(__file__)),
            "seed_block": [9840000, 9840029],
            "device": "cuda",
            "device_note": "pinned to C4's device. CPU/CUDA trajectory divergence "
                           "from identical seeds is a binding invariant, so the "
                           "replay-equivalence the C5 freeze rests on requires it.",
            "freezes_predate_any_c5_result": True,
            "ordering_evidence": "C5 freeze, ranking amendment and observability "
                                 "freeze were all committed while artifacts/c5_discovery/ "
                                 "did not yet exist; verified inline before each commit.",
        },
        "scope_of_claim": json.loads(C5_AMEND.read_text(encoding="utf-8"))
            ["frozen_reporting_language"],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"verdict: {verdict}")
    print(f"opponents observed: {len(opponents)}  pairs examined: "
          f"{len(opponents) * (len(opponents) - 1) // 2}")
    print(f"candidates clearing discovery: {len(cands)}")
    if cands:
        c = cands[0]
        print(f"selected: {c['key']}")
        print(f"  replicated in {c['replicated_in']}")
        print(f"  side A ({c['side_a']}): delta={c['side_a_R1_minus_R2']['delta']:.4f} "
              f"lcb95={c['side_a_R1_minus_R2']['lcb95']:.4f} n={c['n_states_side_a']}")
        print(f"  side B ({c['side_b']}): delta={c['side_b_R2_minus_R1']['delta']:.4f} "
              f"lcb95={c['side_b_R2_minus_R1']['lcb95']:.4f} n={c['n_states_side_b']}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
