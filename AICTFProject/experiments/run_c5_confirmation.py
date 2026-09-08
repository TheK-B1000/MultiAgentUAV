"""C5 confirmation — tests the ONE candidate discovery selected, on unspent 9850000.

The freeze permits exactly one thing here: re-testing the single top-ranked
candidate from discovery under the identical criterion. It permits no search.
That is the whole point of a discovery/confirmation split -- 21 opponent pairs
were examined without correction on the discovery block, and the fresh block
buys back that multiplicity only if nothing is chosen after seeing it.

So this runner does not call analyze(). It never enumerates pairs. It reads the
candidate, collects the two opponents it names, and evaluates that one reversal.

FAIL CLOSED
    1. refuses unless discovery says C5_PASS_DISCOVERY with a selected_candidate
    2. refuses if a confirmation result already exists (no second attempt)
    3. refuses any seed base that is not the frozen confirmation block
    4. evaluates ONLY the frozen candidate; no alternative pair can be reported
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from experiments.run_c4_opportunity_cost import bootstrap_delta  # noqa: E402

C4_FROZEN = ROOT / "artifacts/c4_preregistration/C4_OPPORTUNITY_COST_FROZEN.json"
DISCOVERY = ROOT / "artifacts/c5_discovery/C5_RESULT.json"
OUT = ROOT / "artifacts/c5_confirmation/C5_CONFIRMATION.json"
SHARDS = ROOT / "artifacts/c5_confirmation/shards"
STATES = ROOT / "artifacts/c5_confirmation/states.json"
G0_SEEDS = ["3200001", "3200002", "3200003"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--skip-collect", action="store_true")
    args = ap.parse_args()

    if OUT.exists():
        print(f"REFUSED: {OUT.name} already exists. The confirmation block is "
              f"spent once. A second attempt would be a re-roll.", file=sys.stderr)
        return 2

    if not DISCOVERY.exists():
        print(f"REFUSED: no discovery result at {DISCOVERY}. Run the discovery "
              f"scan and analyze_c5_discovery.py first.", file=sys.stderr)
        return 2

    frozen = json.loads(C4_FROZEN.read_text(encoding="utf-8"))
    disc = json.loads(DISCOVERY.read_text(encoding="utf-8"))
    if disc.get("verdict") != "C5_PASS_DISCOVERY" or not disc.get("selected_candidate"):
        print(f"REFUSED: discovery verdict is {disc.get('verdict')!r} with no "
              f"selected candidate. Nothing to confirm.", file=sys.stderr)
        return 2

    cand = disc["selected_candidate"]
    opp_a, opp_b = str(cand["side_a"]), str(cand["side_b"])
    r1, r2 = str(cand["response_1"]), str(cand["response_2"])
    base = int(frozen["seeds"]["confirmation_block"][0])

    print("=" * 74)
    print("C5 CONFIRMATION — one candidate, one block, no search")
    print(f"  candidate : {opp_a} vs {opp_b}   R1={r1}  R2={r2}")
    print(f"  direction : R1 wins vs {opp_a}, R2 wins vs {opp_b}")
    print(f"  block     : {base}..{base + args.episodes - 1} (unspent)")
    print("=" * 74, flush=True)

    if not args.skip_collect:
        cmd = [str(ROOT / ".venv/Scripts/python.exe"), "-u",
               str(ROOT / "experiments/run_c5_parallel.py"),
               "--episodes", str(args.episodes), "--workers", str(args.workers),
               "--seed-base", str(base), "--only-opponents", f"{opp_a},{opp_b}",
               "--shard-dir", str(SHARDS), "--states-out", str(STATES)]
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        if rc != 0:
            print(f"ABORT: collection failed rc={rc}", file=sys.stderr)
            return 1

    states = json.loads(STATES.read_text(encoding="utf-8"))
    crit = frozen["reversal_criterion"]
    delta_bar = float(crit["delta"])
    min_sup = int(frozen["support_minima"]["min_states_per_response_pair_per_class"])
    rng = np.random.default_rng(int(frozen["statistics"]["seed"]))
    resamples = int(frozen["statistics"]["resamples"])

    per_policy, replicated = {}, []
    for pseed in G0_SEEDS:
        rows = states.get(pseed, [])
        side = {opp_a: [], opp_b: []}
        for r in rows:
            o = r["episode_key"].split(":")[0]
            if o in side and r1 in r["utilities"] and r2 in r["utilities"]:
                side[o].append(r)

        def pull(o, resp):
            return [(r["episode_key"], r["utilities"][resp]) for r in side[o]]

        na, nb = len(side[opp_a]), len(side[opp_b])
        if na < min_sup or nb < min_sup:
            per_policy[pseed] = {"support_a": na, "support_b": nb,
                                 "passed": False, "reason": "insufficient support"}
            continue
        da = bootstrap_delta(pull(opp_a, r1), pull(opp_a, r2),
                             rng=rng, resamples=resamples, lcb_pct=2.5)
        db = bootstrap_delta(pull(opp_b, r2), pull(opp_b, r1),
                             rng=rng, resamples=resamples, lcb_pct=2.5)
        ok = (da["delta"] is not None and db["delta"] is not None
              and da["delta"] >= delta_bar and da["lcb95"] > 0
              and db["delta"] >= delta_bar and db["lcb95"] > 0)
        per_policy[pseed] = {"support_a": na, "support_b": nb,
                             f"{opp_a}_R1_minus_R2": da,
                             f"{opp_b}_R2_minus_R1": db, "passed": bool(ok)}
        if ok:
            replicated.append(pseed)

    verdict = "C5_PASS" if len(replicated) >= 2 else "C5_CONFIRMATION_FAIL"
    out = {
        "record": "C5 confirmation",
        "verdict": verdict,
        "candidate": {"opponent_a": opp_a, "opponent_b": opp_b,
                      "response_1": r1, "response_2": r2},
        "candidate_source": "discovery selected_candidate; NOT reselected here",
        "confirmation_block": [base, base + args.episodes - 1],
        "n_policies_replicating": len(replicated),
        "replicated_in": replicated,
        "per_policy": per_policy,
        "criterion": {"delta": delta_bar, "lcb95_rule": "q_0.025 > 0 both directions",
                      "min_support_per_side": min_sup,
                      "replication": ">=2 of 3 policies"},
        "no_search_performed": True,
        "meaning": (
            "C5_PASS: opponent-conditional strategic demand EXISTS. It does NOT "
            "establish that the demand is routable from legal context -- that is "
            "the already-frozen observability test, which is mandatory before any "
            "oracle training."
            if verdict == "C5_PASS" else
            "C5_CONFIRMATION_FAIL: the discovery candidate did not survive a fresh "
            "block. Treat the discovery result as unreplicated. Do NOT test the "
            "second-ranked candidate -- the block is spent."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nVERDICT: {verdict}  ({len(replicated)}/3 policies)")
    for p in G0_SEEDS:
        d = per_policy.get(p, {})
        print(f"  {p}: passed={d.get('passed')}  n_a={d.get('support_a')} n_b={d.get('support_b')}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
