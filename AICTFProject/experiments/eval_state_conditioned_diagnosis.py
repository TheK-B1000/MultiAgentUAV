r"""State-conditioned diagnosis: WHERE does B_worse replace B_better's decisions?

Governed by artifacts/strategic_demand/sppo/STATE_CONDITIONED_DIAGNOSIS_SPEC.json
(frozen before any episode). B_worse (Bfinal) drives real Pole-B rollouts;
B_better (B_t500k) and frozen A3 are queried COUNTERFACTUALLY at every visited
state (same technique validated in eval_pole_b_behavioral_diagnosis.py). Every
feature is PER AGENT, bucketed by frozen definitions, before any data exists.

    python -m experiments.eval_state_conditioned_diagnosis [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.eval_pole_b_behavioral_diagnosis as D  # reuse _build_env, _np, MACRO_NAMES

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
LABEL = "STATE_CONDITIONED_DIAGNOSIS"
SPEC = SD / f"{LABEL}_SPEC.json"
EXPERIMENT_ID = LABEL

SEED_LO, N_SEEDS = 18_900_001, 48
SCALE = ROOT / "artifacts" / "scale_4v4_specialists"
CORR = SCALE / "pi_B_specialist_4v4_b3_entity_repair_corrected" / "ckpts"

B_WORSE = ("Bfinal", CORR / "final_pi_B_specialist_4v4_b3_entity_repair_corrected.zip")
B_BETTER = ("B_t500k", CORR / "ckpt_pi_B_specialist_4v4_b3_entity_repair_corrected_500000.zip")
A3 = ("A3", SCALE / "pi_A_specialist_4v4_b3_entity_repair" / "ckpts" / "final_pi_A_specialist_4v4_b3_entity_repair.zip")

SHA_B_WORSE_PREFIX = "021342c84bbe"
SHA_B_BETTER_PREFIX = "d4c0d7ba2477"

PRESSURE_RADIUS = 3.0
DEFEND_RADIUS = 4.0
MACRO_BUCKET = {2: "GET_FLAG", 0: "GO_TO", 4: "GO_HOME"}  # else "OTHER"
MIN_BUCKET_COUNT = 30


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _per_agent_context(core) -> list[dict]:
    """Per-agent context features, frozen bucket definitions. Also returns
    team-level aggregates for the K4 cross-check against the validated
    team-level formula in eval_pole_b_behavioral_diagnosis.py."""
    bx, by = D._np(core.blue_x).reshape(-1), D._np(core.blue_y).reshape(-1)
    rx, ry = D._np(core.red_x).reshape(-1), D._np(core.red_y).reshape(-1)
    bf = D._np(core.blue_flag_pos).reshape(-1)
    rf = D._np(core.red_flag_pos).reshape(-1)
    carrying = D._np(core.blue_carrying).reshape(-1).astype(bool)

    ctx = []
    n_pressured_team = 0
    n_attackers_team = 0
    n_defenders_team = 0
    for i in range(4):
        n_near = int(np.sum(np.hypot(rx - bx[i], ry - by[i]) <= PRESSURE_RADIUS))
        pressured = n_near >= 2
        d_enemy = float(np.hypot(bx[i] - rf[0], by[i] - rf[1]))
        d_own = float(np.hypot(bx[i] - bf[0], by[i] - bf[1]))
        near_enemy = d_enemy < DEFEND_RADIUS
        near_own = d_own < DEFEND_RADIUS
        n_pressured_team += int(pressured)
        n_attackers_team += int(near_enemy)
        n_defenders_team += int(near_own)
        ctx.append({"carrying": bool(carrying[i]), "pressured": pressured,
                    "near_enemy_flag": near_enemy, "near_own_flag": near_own})
    return ctx, {"n_pressured": n_pressured_team, "n_attackers": n_attackers_team,
                 "n_defenders": n_defenders_team}


def _macro_of(action: np.ndarray, i: int) -> int:
    return int(np.asarray(action).reshape(-1)[2 * i])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from experiments.pole_attestation import assert_resolved_matches_certification, governing_certification, resolve_pole_genome
    from experiments.run_lock import RunLock
    from gpu_env._core._entity_obs import augment_obs_with_entities
    from experiments.tqdm_loop import set_postfix, tqdm_iter
    from rl.custom_ppo import load_custom_ppo_policy
    import experiments.r2_learned_crossover as R2

    if not SPEC.is_file():
        raise SystemExit(f"REFUSING: frozen spec missing: {SPEC}")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: spec not frozen: {spec.get('status')!r}")

    out_path = SD / f"{LABEL}_RESULT.json"
    rows_csv = SD / f"{LABEL.lower()}_agent_step_rows.csv"
    if not args.dry_run:
        import experiments.seed_registry as R
        doc = R.load()
        b = next((x for x in doc["blocks"] if x["experiment_id"] == EXPERIMENT_ID), None)
        if b is None or b["lo"] != SEED_LO or b["hi"] != SEED_LO + N_SEEDS - 1:
            raise SystemExit(f"FAIL-CLOSED (Rule 9): seed block for {EXPERIMENT_ID!r} not "
                             f"registered as {SEED_LO}..{SEED_LO + N_SEEDS - 1}")
        if out_path.is_file() or rows_csv.is_file():
            raise SystemExit(f"REFUSING: output for {LABEL} already exists; one-shot")

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    seeds = list(range(SEED_LO, SEED_LO + N_SEEDS))
    _v, cert_path = governing_certification(4)

    print(f"{LABEL}  {_now()}  device={device}")
    print(f"  spec       {SPEC.name} [{spec.get('status')}]")
    print(f"  B_worse    {B_WORSE[0]}  sha {_sha(B_WORSE[1])[:12]}...")
    print(f"  B_better   {B_BETTER[0]} sha {_sha(B_BETTER[1])[:12]}...")
    print(f"  A3 (tert.) sha {_sha(A3[1])[:12]}...")
    print(f"  seeds      {seeds[0]}..{seeds[-1]} (n={len(seeds)})")

    genome = resolve_pole_genome("B", 4, str(D.POLE_B_GENOME))
    attestation = assert_resolved_matches_certification("B", 4, cert_path, genome, is_smoke=False)
    print(f"  pole B     {attestation['live_genome_id']} MATCH={'PASS' if attestation['hashes_match'] else 'FAIL'}")

    env, _c, _o = D._build_env(device, seeds[0], "B", genome)
    obs_space, act_space = env.observation_space, env.action_space
    env.close()

    policies = {lbl: load_custom_ppo_policy(str(p), obs_space, act_space, device=device)
                for lbl, p in (B_WORSE, B_BETTER, A3)}
    for pol in policies.values():
        pol.model.eval()

    contracts = {
        "K1_checkpoints_match_screened_shas": (
            _sha(B_WORSE[1]).startswith(SHA_B_WORSE_PREFIX) and _sha(B_BETTER[1]).startswith(SHA_B_BETTER_PREFIX)),
        "K2_pole_attested": attestation["hashes_match"],
    }
    # K3: determinism of a counterfactual query.
    env, core, obs = D._build_env(device, seeds[0], "B", genome)
    try:
        obs = augment_obs_with_entities(obs, core, side="blue")
        a1, _ = policies[B_BETTER[0]].predict(obs, deterministic=True)
        a2, _ = policies[B_BETTER[0]].predict(obs, deterministic=True)
        contracts["K3_counterfactual_query_deterministic"] = bool(np.array_equal(np.asarray(a1), np.asarray(a2)))
        # K4: per-agent context sums must reproduce the validated team-level aggregate formula.
        ctx, team = _per_agent_context(core)
        contracts["K4_per_agent_sums_match_team_aggregate"] = (
            sum(c["pressured"] for c in ctx) == team["n_pressured"]
            and sum(c["near_enemy_flag"] for c in ctx) == team["n_attackers"]
            and sum(c["near_own_flag"] for c in ctx) == team["n_defenders"])
    finally:
        env.close()

    print("\n  known-answer contracts ...")
    for k, v in contracts.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    if not all(contracts.values()):
        raise SystemExit(f"FAIL-CLOSED: contracts failed {[k for k, v in contracts.items() if not v]}")

    if args.dry_run:
        print("\n  --dry-run: spec frozen, checkpoints match screened shas, pole attested, "
              "counterfactual query deterministic, per-agent/team aggregates agree. "
              "NO episodes run, NOTHING written.")
        return 0

    fields = ["seed", "t", "agent", "carrying", "pressured", "near_enemy_flag", "near_own_flag",
              "prev_macro_bucket", "macro_worse", "macro_better", "macro_A3", "disagree_worse_better"]

    disagreements: list[dict] = []
    episode_outcomes: dict[int, dict] = {}

    with RunLock(SD / f"{LABEL}.run.lock", run_id=LABEL):
        with rows_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            bar = tqdm_iter(seeds, desc=f"{LABEL}", unit="ep")
            for seed in bar:
                set_postfix(bar, f"seed={seed}")
                env, core, obs = D._build_env(device, seed, "B", genome)
                try:
                    for pol in policies.values():
                        pol.reset_strategy()
                    obs = augment_obs_with_entities(obs, core, side="blue")
                    prev_macro = [2, 2, 2, 2]  # GET_FLAG is the natural opening default; recorded as such
                    terminal = None
                    for t in range(R2.MAX_STEPS):
                        ctx, _team = _per_agent_context(core)
                        with torch.no_grad():
                            a_worse, _ = policies[B_WORSE[0]].predict(obs, deterministic=True)
                            a_better, _ = policies[B_BETTER[0]].predict(obs, deterministic=True)
                            a_a3, _ = policies[A3[0]].predict(obs, deterministic=True)
                        for i in range(4):
                            mw, mb, ma = _macro_of(a_worse, i), _macro_of(a_better, i), _macro_of(a_a3, i)
                            row = {"seed": seed, "t": t, "agent": i,
                                  "carrying": int(ctx[i]["carrying"]), "pressured": int(ctx[i]["pressured"]),
                                  "near_enemy_flag": int(ctx[i]["near_enemy_flag"]),
                                  "near_own_flag": int(ctx[i]["near_own_flag"]),
                                  "prev_macro_bucket": MACRO_BUCKET.get(prev_macro[i], "OTHER"),
                                  "macro_worse": D.MACRO_NAMES.get(mw, str(mw)),
                                  "macro_better": D.MACRO_NAMES.get(mb, str(mb)),
                                  "macro_A3": D.MACRO_NAMES.get(ma, str(ma)),
                                  "disagree_worse_better": int(mw != mb)}
                            w.writerow(row)
                            if mw != mb:
                                disagreements.append({**row, "seed": seed})
                            prev_macro[i] = mw
                        # B_worse DRIVES the real trajectory.
                        env.step_async(a_worse)
                        obs, _r, done, info = env.step_wait()
                        obs["global_state"] = env.state()
                        obs = augment_obs_with_entities(obs, core, side="blue")
                        if bool(np.asarray(done).any()):
                            i0 = info[0] if isinstance(info, (list, tuple)) else info
                            res = (i0 or {}).get("episode_result") or {}
                            terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                            break
                    if terminal is None:
                        terminal = (int(core.blue_score[0]), int(core.red_score[0]))
                    blue_s, red_s = terminal
                    episode_outcomes[seed] = {"won": int(blue_s > red_s), "blue_score": blue_s, "red_score": red_s}
                    fh.flush()
                finally:
                    env.close()

    # ---- bucket aggregation, per the fixed reporting rule -----------------------
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    all_by_bucket_count: dict[tuple, int] = Counter()
    for r in disagreements:
        key = (bool(r["carrying"]), bool(r["pressured"]), bool(r["near_enemy_flag"]),
               bool(r["near_own_flag"]), r["prev_macro_bucket"])
        buckets[key].append(r)

    # denominator per bucket needs the TOTAL agent-steps in that bucket (not just
    # disagreements) -- recount from the full CSV rather than re-rolling.
    total_by_bucket: dict[tuple, int] = Counter()
    with rows_csv.open("r", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (bool(int(r["carrying"])), bool(int(r["pressured"])), bool(int(r["near_enemy_flag"])),
                   bool(int(r["near_own_flag"])), r["prev_macro_bucket"])
            total_by_bucket[key] += 1

    overall_won = float(np.mean([o["won"] for o in episode_outcomes.values()]))
    overall_blue = float(np.mean([o["blue_score"] for o in episode_outcomes.values()]))

    reported = []
    for key, rows in buckets.items():
        n_disagree = len(rows)
        if n_disagree < MIN_BUCKET_COUNT:
            continue
        subs = Counter((r["macro_worse"], r["macro_better"]) for r in rows)
        dom_sub = subs.most_common(1)[0]
        a3_agrees_better = np.mean([r["macro_A3"] == r["macro_better"] for r in rows])
        seeds_touched = sorted(set(r["seed"] for r in rows))
        won_touched = float(np.mean([episode_outcomes[s]["won"] for s in seeds_touched]))
        blue_touched = float(np.mean([episode_outcomes[s]["blue_score"] for s in seeds_touched]))
        reported.append({
            "carrying": key[0], "pressured": key[1], "near_enemy_flag": key[2],
            "near_own_flag": key[3], "prev_macro_bucket": key[4],
            "n_disagree": n_disagree, "n_total_in_bucket": total_by_bucket[key],
            "disagreement_rate": n_disagree / max(1, total_by_bucket[key]),
            "dominant_substitution": f"{dom_sub[0][0]} -> {dom_sub[0][1]} ({dom_sub[1]}/{n_disagree})",
            "A3_agreement_with_B_better": float(a3_agrees_better),
            "episodes_touched": len(seeds_touched),
            "won_rate_in_touched_episodes": won_touched,
            "blue_score_in_touched_episodes": blue_touched,
        })
    reported.sort(key=lambda r: r["n_disagree"], reverse=True)

    print(f"\n  overall B_worse rollout: won={overall_won:.3f} blue_score={overall_blue:.3f}")
    print(f"  {len(reported)} bucket(s) meet the >= {MIN_BUCKET_COUNT} disagreement threshold:\n")
    for r in reported[:15]:
        print(f"    carrying={r['carrying']!s:5s} pressured={r['pressured']!s:5s} "
              f"near_enemy={r['near_enemy_flag']!s:5s} near_own={r['near_own_flag']!s:5s} "
              f"prev={r['prev_macro_bucket']:9s} n_disagree={r['n_disagree']:4d} "
              f"rate={r['disagreement_rate']:.3f}  {r['dominant_substitution']:28s} "
              f"A3~better={r['A3_agreement_with_B_better']:.3f} "
              f"won(touched)={r['won_rate_in_touched_episodes']:.3f}")

    out_path.write_text(json.dumps({
        "record": f"{LABEL} context-bucketed macro substitution analysis",
        "status": "COMPLETE_DIAGNOSTIC", "utc": _now(), "device": device,
        "arm": "DIAGNOSTIC", "confirmatory": False,
        "implements": f"{SPEC.name}",
        "question": spec.get("THE_QUESTION"),
        "B_worse": {"label": B_WORSE[0], "sha256": _sha(B_WORSE[1])},
        "B_better": {"label": B_BETTER[0], "sha256": _sha(B_BETTER[1])},
        "A3_tertiary": {"sha256": _sha(A3[1])},
        "seeds": {"block": [seeds[0], seeds[-1]], "n": len(seeds)},
        "contracts": contracts,
        "overall_rollout": {"won": overall_won, "blue_score": overall_blue},
        "min_bucket_count": MIN_BUCKET_COUNT,
        "reported_buckets_sorted_by_disagreement_count": reported,
        "total_agent_steps": sum(total_by_bucket.values()),
        "total_disagreements": len(disagreements),
        "NOT_A_CLAIM": spec.get("WHAT_THIS_EXPERIMENT_MAY_NOT_CLAIM"),
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {out_path}\n  -> {rows_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
