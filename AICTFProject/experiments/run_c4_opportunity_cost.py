"""C4 opportunity-cost scanner — literal implementation of c913c1e.

Searches natural map_a states for a preference REVERSAL between two legal team
responses across the two sides of one frozen binary partition:

    context side A:  U(R1) - U(R2) >= delta,  CI excludes 0
    context side B:  U(R2) - U(R1) >= delta,  CI excludes 0

C3 asked whether some response beats the natural continuation. C4 asks whether
the RANKING between two responses flips across contexts, which is the minimal
condition under which keeping two policies can pay.

Branching is EXHAUSTIVE (short_circuit=False). C3's existential short-circuit
stops at the first witness, which is correct for actionability and useless here:
a reversal test needs every response's utility, not one that clears a bar.

FAIL-CLOSED SAFEGUARDS
    1. a tested response pair must be legal on BOTH sides of the partition
    2. support reported separately per side
    3. no pooled rescue -- the frozen >=2/3 policy replication is per policy
    4. no ranking on point estimates; the CI rule must hold in both directions
    5. no fallback partitions; if all six fail the verdict is C4_NO_REVERSAL
    6. refuses to run if a result already exists

Run:  python experiments/run_c4_opportunity_cost.py
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

FROZEN = PROJECT_ROOT / "artifacts" / "c4_preregistration" / "C4_OPPORTUNITY_COST_FROZEN.json"
OUT = PROJECT_ROOT / "artifacts" / "c4_discovery"
RESULT = OUT / "C4_RESULT.json"
G0_SEEDS = (3_200_001, 3_200_002, 3_200_003)

# C7 Stage 2 scans BOTH arms under one identical protocol. "2v2" is the default,
# so every historical invocation is byte-for-byte unchanged.
ARMS = {
    "2v2": {"seeds": (3_200_001, 3_200_002, 3_200_003), "agents": 2,
            "dir": "g0_v5_long", "tag": "g0_v5_long_seed{}"},
    "4v4": {"seeds": (3_300_001, 3_300_002, 3_300_003), "agents": 4,
            "dir": "c7_stage0", "tag": "c7_4v4_seed{}"},
}

# The six frozen partitions, as (name, predicate on a legal_context dict).
PARTITIONS = {
    "score_stratum": lambda c: ("leading" if c["score_diff"] > 0
                                else "trailing" if c["score_diff"] < 0 else None),
    "carrier_present": lambda c: "carrying" if c["carrier_present"] else "not_carrying",
    "home_threatened": lambda c: "threatened" if c["home_threatened"] else "secure",
    "forward_commitment": lambda c: ("both_forward" if c["agents_forward"] >= 2
                                     else "none_forward" if c["agents_forward"] == 0 else None),
    "carrier_pressure": lambda c: ("pressured" if c["carrier_under_pressure"]
                                   else "unpressured" if c["carrier_present"] else None),
    "defender_tag_available": lambda c: "tag_ready" if c["defender_tag_available"] else "tag_cold",
}


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def bootstrap_delta(pairs_a, pairs_b, *, rng, resamples, lcb_pct):
    """Episode-clustered difference of means between two response utilities."""
    def by_ep(rows):
        g = defaultdict(list)
        for ep, v in rows:
            g[ep].append(float(v))
        return g

    ga, gb = by_ep(pairs_a), by_ep(pairs_b)
    ka, kb = list(ga), list(gb)
    if len(ka) < 2 or len(kb) < 2:
        return {"delta": None, "lcb95": None, "n_ep_a": len(ka), "n_ep_b": len(kb)}

    def m(g, keys):
        vals = [v for k in keys for v in g[k]]
        return statistics.fmean(vals) if vals else float("nan")

    point = m(ga, ka) - m(gb, kb)
    draws = np.empty(resamples)
    for i in range(resamples):
        a = [ka[j] for j in rng.integers(0, len(ka), len(ka))]
        b = [kb[j] for j in rng.integers(0, len(kb), len(kb))]
        draws[i] = m(ga, a) - m(gb, b)
    draws = draws[np.isfinite(draws)]
    lo, hi = np.percentile(draws, [lcb_pct, 100 - lcb_pct])
    return {"delta": round(float(point), 6), "lcb95": round(float(lo), 6),
            "ucb95": round(float(hi), 6), "n_ep_a": len(ka), "n_ep_b": len(kb)}


def _runner_commit_safe() -> str:
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
                           capture_output=True, text=True, timeout=30)
        return (r.stdout or "").strip() or "unknown"
    except Exception:
        return "unknown"


def collect_states(policy, *, opponent, seed, device, contract, max_states):
    """Natural states with >=2 legal responses, each with EXHAUSTIVE response utilities."""
    from experiments.eval_v6i9_map_awareness import (
        _adapt_obs_for_policy, _done, _predict, _reset_obs, _unpack_step,
    )
    from experiments.run_g0_v2_evaluation import (
        AGENTS, CANONICAL_MAP, EPISODE_HORIZON, V2_RULES, legal_context,
    )
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    from rl.analysis.counterfactual_actionability import (
        resolve_utility, run_counterfactual_branches,
    )
    from rl.analysis.legal_team_responses import count_legal_team_responses_batched
    from rl.evaluation.opponent_resolution import (
        get_opponent_key, set_opponent, validate_opponent_name,
    )

    requested = validate_opponent_name(opponent)
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=AGENTS, max_red_agents=AGENTS,
        map_set="train", map_layout=CANONICAL_MAP,
        max_decision_steps=EPISODE_HORIZON, aquaticus_profile=True,
        rules_profile="OURS", device=device, seed=int(seed),
        obstacle_obs_channel=True, tag_telemetry_enabled=True, **V2_RULES,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    # C3 passes the INFERENCE POLICY wrapper here, not the raw module: the
    # branching snapshot reads _prev_z, which CustomPPOInferencePolicy defines
    # in its constructor and a bare nn.Module never creates.
    model = policy
    inner = policy.model if hasattr(policy, "model") else policy
    was = getattr(inner, "training", False)
    if hasattr(inner, "eval"):
        inner.eval()
    utility_fn = resolve_utility(contract.utility_name)

    out = []
    try:
        set_opponent(env, requested)
        obs = _reset_obs(env.reset())
        if get_opponent_key(env) != requested:
            raise RuntimeError("opponent drift")
        core.drain_tag_events()
        for step_i in range(EPISODE_HORIZON + 8):
            ctx = legal_context(core)
            n_legal = int(count_legal_team_responses_batched(core)[0])
            if n_legal >= 2 and len(out) < max_states:
                bs = run_counterfactual_branches(
                    env, model, obs,
                    candidate_step=step_i,
                    response_horizon=contract.h_response,
                    utility_fn=utility_fn,
                    short_circuit=False,          # C4 needs EVERY response
                )
                util = {tuple(b.team_response): float(b.branch_utility) for b in bs.branches}
                if len(util) >= 2:
                    out.append({
                        "episode_key": f"{requested}:{seed}",
                        "step": step_i,
                        "classes": {k: f(ctx) for k, f in PARTITIONS.items()},
                        # Full legal_context vector, not just the six partition
                        # labels derived from it. The frozen observability test
                        # fits its legal-selector on exactly these fields, and
                        # recovering them later would cost a full re-rollout.
                        # Pure instrumentation: recorded, never acted on here.
                        "legal_context": {k: (float(v) if isinstance(v, (bool, int, float))
                                              else v) for k, v in ctx.items()},
                        "utilities": {str(k): v for k, v in util.items()},
                    })
            # Once max_states is reached no further append is possible -- the
            # guard above requires len(out) < max_states -- so the rest of the
            # episode only steps the env and is discarded. Breaking here yields
            # a byte-identical `out`, verified by hash against a full-episode run.
            if len(out) >= max_states:
                break
            action = _predict(policy, _adapt_obs_for_policy(obs, policy))
            obs, _, done, _infos = _unpack_step(env.step(action))
            if _done(done):
                break
    finally:
        if hasattr(inner, "train"):
            inner.train(was)
        env.close()
    return out


def analyze(states_by_policy: dict, frozen: dict) -> dict:
    crit = frozen["reversal_criterion"]
    delta = float(crit["delta"])
    min_sup = int(frozen["support_minima"]["min_states_per_response_pair_per_class"])
    resamples = int(frozen["statistics"]["resamples"])
    boot_seed = int(frozen["statistics"]["seed"])
    lcb_pct = 2.5
    rng = np.random.default_rng(boot_seed)

    per_policy: dict[str, dict] = {}
    for pseed, states in states_by_policy.items():
        found = {}
        for pname in PARTITIONS:
            sides = defaultdict(list)
            for s in states:
                side = s["classes"].get(pname)
                if side is not None:
                    sides[side].append(s)
            if len(sides) < 2:
                continue
            # A partition with exactly two values yields one side-pair, which is
            # C4's behaviour unchanged. C5's opponent partition has seven, so
            # every unordered side-pair is tested.
            side_pairs = list(itertools.combinations(sorted(sides.items()), 2))
            for (sa, rows_a), (sb, rows_b) in side_pairs:

                # SAFEGUARD 1: a pair must be legal on BOTH sides.
                resp_a = set().union(*[set(r["utilities"]) for r in rows_a]) if rows_a else set()
                resp_b = set().union(*[set(r["utilities"]) for r in rows_b]) if rows_b else set()
                common = sorted(resp_a & resp_b)

                for r1, r2 in itertools.combinations(common, 2):
                    a1 = [(r["episode_key"], r["utilities"][r1]) for r in rows_a if r1 in r["utilities"] and r2 in r["utilities"]]
                    a2 = [(r["episode_key"], r["utilities"][r2]) for r in rows_a if r1 in r["utilities"] and r2 in r["utilities"]]
                    b1 = [(r["episode_key"], r["utilities"][r1]) for r in rows_b if r1 in r["utilities"] and r2 in r["utilities"]]
                    b2 = [(r["episode_key"], r["utilities"][r2]) for r in rows_b if r1 in r["utilities"] and r2 in r["utilities"]]
                    # SAFEGUARD 2: support reported per side.
                    if len(a1) < min_sup or len(b1) < min_sup:
                        continue
                    # BOTH ORIENTATIONS. combinations() yields each unordered pair
                    # once, so testing only "R1 wins side A, R2 wins side B" would
                    # silently miss every mirror reversal and report NO_REVERSAL.
                    for x1, x2, u_a1, u_a2, u_b1, u_b2 in (
                        (r1, r2, a1, a2, b1, b2),
                        (r2, r1, a2, a1, b2, b1),
                    ):
                        da = bootstrap_delta(u_a1, u_a2, rng=rng, resamples=resamples, lcb_pct=lcb_pct)
                        db = bootstrap_delta(u_b2, u_b1, rng=rng, resamples=resamples, lcb_pct=lcb_pct)
                        if da["delta"] is None or db["delta"] is None:
                            continue
                        # SAFEGUARD 4: CI rule in BOTH directions, never point estimates.
                        ok = (da["delta"] >= delta and da["lcb95"] > 0
                              and db["delta"] >= delta and db["lcb95"] > 0)
                        if ok:
                            # The key MUST carry the side-pair. In C4 a binary
                            # partition name determined its two sides uniquely,
                            # so pname|R1|R2 was unambiguous. With 21 opponent
                            # side-pairs sharing one partition name, omitting
                            # (sa, sb) makes distinct opponent pairs collide:
                            # later pairs overwrite earlier ones, and cross-policy
                            # replication then compares DIFFERENT opponent pairs
                            # as if they were the same candidate.
                            found[f"{pname}|{sa}|{sb}|{x1}|{x2}"] = {
                                "partition": pname, "side_a": sa, "side_b": sb,
                                "response_1": x1, "response_2": x2,
                                "n_states_side_a": len(u_a1), "n_states_side_b": len(u_b1),
                                "side_a_R1_minus_R2": da, "side_b_R2_minus_R1": db,
                            }
        per_policy[str(pseed)] = found

    # SAFEGUARD 3: replication is per policy; no pooled rescue.
    counts = defaultdict(list)
    for pseed, found in per_policy.items():
        for key in found:
            counts[key].append(pseed)
    replicated = {k: v for k, v in counts.items() if len(v) >= 2}

    return {"per_policy": per_policy, "replicated": replicated,
            "n_replicated": len(replicated)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-states-per-episode", type=int, default=2)
    ap.add_argument("--out", default=str(RESULT))
    ap.add_argument("--arm", default="2v2", choices=tuple(ARMS))
    ap.add_argument("--opponent-set", default="historical",
                    choices=("historical", "srctf"))
    ap.add_argument("--seed-base", type=int, default=0,
                    help="override seed block; must equal the frozen confirmation block")
    ap.add_argument("--policies", default="", help="comma-separated subset of policy seeds")
    ap.add_argument("--opponents", default="", help="comma-separated subset of opponents")
    ap.add_argument("--partition-mode", choices=("state", "opponent"), default="state",
                    help="state = C4 six frozen partitions; opponent = C5 opponent pairs")
    ap.add_argument("--states-out", default="", help="persist raw states for re-analysis")
    args = ap.parse_args()

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise SystemExit("REFUSED: C4 contract is not FROZEN")
    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(f"REFUSED: {out_path.name} already exists")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("REFUSED: CUDA required")

    # Confirmation uses the unspent block; the frozen discovery block stays the
    # default so no caller silently lands on the wrong seeds.
    base = int(args.seed_base) if args.seed_base else int(frozen["seeds"]["discovery_block"][0])
    if args.seed_base:
        # A seed base is legal only if some frozen file already names it. This is
        # what stops a caller from quietly evaluating on unfrozen seeds.
        allowed = {int(frozen["seeds"]["discovery_block"][0]),
                   int(frozen["seeds"]["confirmation_block"][0])}
        obs = PROJECT_ROOT / "artifacts/c5_preregistration/OBSERVABILITY_TEST_FROZEN.json"
        if obs.exists():
            allowed.add(int(json.loads(obs.read_text(encoding="utf-8"))
                            ["seeds"]["evaluation_block"][0]))
        c7 = PROJECT_ROOT / "artifacts/c7_preregistration/C7_TEAM_SIZE_DEMAND_FROZEN.json"
        if c7.exists():
            _c7 = json.loads(c7.read_text(encoding="utf-8"))["seeds"]
            for _k in ("stage_1", "stage_2", "stage_3"):
                allowed.add(int(_c7[_k][0]))
        if int(args.seed_base) not in allowed:
            raise SystemExit(f"--seed-base {args.seed_base} is named by no frozen file "
                             f"(allowed: {sorted(allowed)}); refusing unfrozen seeds")
    from experiments.run_c3_decision_proximal_discovery import _load_runtime_contract
    from experiments.run_g0_v2_evaluation import resolve_cnn_channels
    # Opponent set is INJECTED from a canonical registry rather than hardcoded.
    # Order is frozen there because cell order defines the parallel merge order.
    from srctf.opponent_sets import get as _opponent_set
    OPPONENTS = _opponent_set(args.opponent_set)
    from rl.custom_ppo.checkpoints.loader import read_checkpoint_payload
    from rl.evaluation.checkpoint import load_policy

    contract = _load_runtime_contract()
    print("=" * 74)
    print("C4 OPPORTUNITY-COST SCAN")
    print(f"  contract sha256 {_sha256(FROZEN)[:16]}...  delta {frozen['reversal_criterion']['delta']}")
    print(f"  block {base}..{base + args.episodes - 1}  partitions {list(PARTITIONS)}")
    print("=" * 74)

    started = time.time()
    # Subsetting changes only WHICH cells this process computes, never how any
    # cell is computed: collect_states builds a fresh env seeded per (opponent,
    # seed), so a cell's states do not depend on which other cells ran.
    arm = ARMS[args.arm]
    if int(arm["agents"]) != 2:
        # BOTH sides must be rebound. Patching one produced a 4-agent policy
        # evaluated in a 2v2 env during C7 Stage 0.
        import experiments.run_g0_v2_evaluation as _E
        _E.AGENTS = int(arm["agents"])
    sel_pol = [int(x) for x in args.policies.split(",") if x.strip()] or list(arm["seeds"])
    states_by_policy = {}
    for pseed in sel_pol:
        tag = arm["tag"].format(pseed)
        ck = PROJECT_ROOT / "artifacts" / arm["dir"] / tag / "ckpts" / f"final_{tag}.zip"
        payload = read_checkpoint_payload(str(ck), map_location="cpu")
        pol = load_policy(str(ck), device=args.device,
                          num_cnn_channels=resolve_cnn_channels(payload, context=str(ck)))
        rows = []
        sel_opp = [x for x in args.opponents.split(",") if x.strip()] or list(OPPONENTS)
        for opp in sel_opp:
            for i in range(args.episodes):
                rows += collect_states(pol, opponent=opp, seed=base + i, device=args.device,
                                       contract=contract, max_states=args.max_states_per_episode)
            print(f"  policy {pseed} vs {opp}: {len(rows)} states", flush=True)
        states_by_policy[pseed] = rows

    if args.states_out:
        import hashlib as _hl
        from experiments.run_g0_v2_evaluation import CANONICAL_MAP as _MAP
        Path(args.states_out).parent.mkdir(parents=True, exist_ok=True)
        blob = json.dumps({str(k): v for k, v in states_by_policy.items()})
        Path(args.states_out).write_text(blob, encoding="utf-8")
        # Identity manifest: a shard must carry enough provenance that an
        # accidental mismatch cannot sneak into a merge. The merger fails closed
        # on any disagreement rather than silently pooling incompatible shards.
        ck_sha = {}
        for pseed in sel_pol:
            # ARM-AWARE. This was hardcoded to g0_v5_long while the policy-loading
            # path above was already arm-aware, so 4v4 cells completed their full
            # measurement, wrote their states, and then died here looking for a
            # 3300001 checkpoint under the 2v2 directory.
            tag = arm["tag"].format(pseed)
            ckp = PROJECT_ROOT / "artifacts" / arm["dir"] / tag / "ckpts" / f"final_{tag}.zip"
            ck_sha[str(pseed)] = _hl.sha256(ckp.read_bytes()).hexdigest()
        man = {
            "policies": [str(x) for x in sel_pol],
            "policy_checkpoint_sha256": ck_sha,
            "opponents": sel_opp,
            "episodes": int(args.episodes),
            "seed_block_base": int(base),
            "resolved_seeds": [int(base + i) for i in range(args.episodes)],
            "device": args.device,
            "max_states_per_episode": int(args.max_states_per_episode),
            "partition_mode": args.partition_mode,
            "map": str(_MAP),
            "ruleset": "OURS/V2_RULES",
            "frozen_contract_sha256": _sha256(FROZEN),
            "runtime_contract_utility": contract.utility_name,
            "runtime_contract_h_response": int(contract.h_response),
            "runner_commit": _runner_commit_safe(),
            "n_states": {str(k): len(v) for k, v in states_by_policy.items()},
            "states_file_sha256": _hl.sha256(blob.encode("utf-8")).hexdigest(),
        }
        Path(str(args.states_out) + ".manifest.json").write_text(
            json.dumps(man, indent=2), encoding="utf-8")
        print(f"  persisted raw states -> {args.states_out} (+ manifest)")

    if args.partition_mode == "opponent":
        # C5: opponent identity as an EVALUATION LABEL, never a policy input.
        for rows in states_by_policy.values():
            for r in rows:
                r["classes"] = {"opponent": r["episode_key"].split(":")[0]}
        globals()["PARTITIONS"] = {"opponent": lambda c: None}
    res = analyze(states_by_policy, frozen)
    verdict = "C4_PASS" if res["n_replicated"] > 0 else "C4_NO_REVERSAL"
    claims = frozen["outcomes"][verdict]
    doc = {
        "record": "C4 opportunity-cost scan",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract_sha256": _sha256(FROZEN),
        "device": args.device, "torch": torch.__version__, "cuda": torch.version.cuda,
        "discovery_block": [base, base + args.episodes - 1],
        "n_states": {str(k): len(v) for k, v in states_by_policy.items()},
        "verdict": verdict,
        "n_replicated_reversals": res["n_replicated"],
        "replicated": res["replicated"],
        "per_policy": res["per_policy"],
        "outcome_meaning": claims,
        "wall_seconds": round(time.time() - started, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"  states: {doc['n_states']}")
    print(f"  replicated reversals: {res['n_replicated']}")
    for k, v in list(res["replicated"].items())[:5]:
        print(f"    {k}  policies {v}")
    print(f"\n  VERDICT: {verdict}")
    print(f"  wrote {out_path}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
