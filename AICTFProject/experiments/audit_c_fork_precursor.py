"""One-shot precursor audit: does C_fork fire before the known qualified forks?

Frozen by: artifacts/o3_preregistration/O3_CFORK_RECALL_AMENDMENT.json (ce7949f)

THE ONLY COMPONENT ALLOWED TO JOIN THE TWO WORLDS
-------------------------------------------------
    C3 fork labels ------> audit_c_fork_precursor.py
                                    |
                                    X   (no import path back)
                                    |
    O3 training <------- CForkDetector, natural state only

This module reads counterfactual labels. The O3 trainer must never import it.
Keeping the label dependency in exactly one file makes the firewall visible in
the architecture rather than maintained by discipline.

EVALUATION-ONLY, RUN ONCE
-------------------------
Its output may not retune the predicate. Iterating audit-and-adjust would let
fork labels supervise the detector through model selection even though they
never enter PPO. If coverage disappoints, that is a reportable limitation and an
interpretation constraint, per the frozen rule: below 0.50, an O3 FAILURE is
INCONCLUSIVE_FOR_SPECIALIZATION rather than evidence against specialization. A
positive O3 result is unaffected by coverage.

EXPECTED RESULT
---------------
Coverage should be essentially 1.0 by construction: a qualified fork requires
n_legal_team_responses >= 2 (its own R1) and lies inside a carry phase, and the
detector fires at the FIRST decision of that phase meeting both conditions.
Anything materially below 1.0 indicates a semantic or bookkeeping discrepancy --
carry-phase boundaries, legality computation, replay alignment, first-fire
bookkeeping -- and is worth investigating BEFORE O3, not shrugged off because it
cleared 0.50.

Run:  python experiments/audit_c_fork_precursor.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from rl.analysis.c_fork_detector import CForkDetector  # noqa: E402

STAGE4_DIR = PROJECT_ROOT / "artifacts" / "c3_stage4"
OUT = PROJECT_ROOT / "artifacts" / "o3_preregistration" / "C_FORK_PRECURSOR_AUDIT.json"
QUALIFIED = "QUALIFIED_COMMITMENT_FORK"
INTERPRETATION_THRESHOLD = 0.50


def replay_with_detector(policy, *, opponent: str, seed: int, device: str) -> list[dict]:
    """Replay one natural episode, recording every detector firing."""
    from experiments.eval_v6i9_map_awareness import (
        _adapt_obs_for_policy, _done, _predict, _reset_obs, _unpack_step,
    )
    from experiments.run_g0_v2_evaluation import (
        AGENTS, CANONICAL_MAP, EPISODE_HORIZON, V2_RULES,
    )
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
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
    model = policy.model if hasattr(policy, "model") else policy
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()

    det = CForkDetector()
    det.reset()
    try:
        set_opponent(env, requested)
        obs = _reset_obs(env.reset())
        if get_opponent_key(env) != requested:
            raise RuntimeError("opponent drift")
        core.drain_tag_events()
        for step_i in range(EPISODE_HORIZON + 8):
            det.step(core, step_i)
            action = _predict(policy, _adapt_obs_for_policy(obs, policy))
            obs, _, done, _infos = _unpack_step(env.step(action))
            if _done(done):
                break
    finally:
        if hasattr(model, "train"):
            model.train(was_training)
        env.close()
    return det.firings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--results", default=str(STAGE4_DIR / "C3_STAGE3_ANCHOR_RESULTS.jsonl"))
    args = ap.parse_args()

    if OUT.exists():
        raise SystemExit(
            f"REFUSED: {OUT.name} already exists. The audit is frozen as RUN ONCE; "
            "re-running after a predicate edit is exactly the loop the freeze forbids."
        )

    rows = [json.loads(l) for l in Path(args.results).read_text(encoding="utf-8").splitlines() if l.strip()]
    forks = [r for r in rows if r["episode_status"] == QUALIFIED and r.get("fork_step") is not None]
    if not forks:
        raise SystemExit("REFUSED: no qualified forks in the results file")

    from experiments.run_g0_v2_evaluation import resolve_cnn_channels
    from rl.custom_ppo.checkpoints.loader import read_checkpoint_payload
    from rl.evaluation.checkpoint import load_policy

    print("=" * 74)
    print("C_FORK PRECURSOR AUDIT — evaluation-only, run once")
    print(f"qualified forks to audit: {len(forks)}")
    print("=" * 74)

    policies: dict[int, object] = {}
    firings_cache: dict[tuple, list[dict]] = {}
    records = []
    started = time.time()

    for i, fk in enumerate(forks, 1):
        seed = int(fk["train_seed"])
        if seed not in policies:
            tag = f"g0_v5_long_seed{seed}"
            ck = PROJECT_ROOT / "artifacts" / "g0_v5_long" / tag / "ckpts" / f"final_{tag}.zip"
            payload = read_checkpoint_payload(str(ck), map_location="cpu")
            policies[seed] = load_policy(
                str(ck), device=args.device,
                num_cnn_channels=resolve_cnn_channels(payload, context=str(ck)))
        key = (seed, fk["opponent"], int(fk["eval_seed"]))
        if key not in firings_cache:
            firings_cache[key] = replay_with_detector(
                policies[seed], opponent=fk["opponent"],
                seed=int(fk["eval_seed"]), device=args.device)
        firings = firings_cache[key]
        fork_step = int(fk["fork_step"])
        before = [f for f in firings if f["step"] <= fork_step]
        rec = {
            "train_seed": seed, "opponent": fk["opponent"],
            "eval_seed": int(fk["eval_seed"]),
            "fork_step": fork_step, "pressure_step": int(fk["pressure_step"]),
            "n_firings_in_episode": len(firings),
            "precursor_before_or_at_fork": bool(before),
            "nearest_firing_step": max((f["step"] for f in before), default=None),
            "lead_time": (fork_step - max(f["step"] for f in before)) if before else None,
            "any_firing_after_fork_only": bool(firings and not before),
        }
        records.append(rec)
        if i % 25 == 0:
            print(f"  audited {i}/{len(forks)}", flush=True)

    covered = [r for r in records if r["precursor_before_or_at_fork"]]
    coverage = len(covered) / len(records)
    leads = [r["lead_time"] for r in covered if r["lead_time"] is not None]
    after_only = sum(1 for r in records if r["any_firing_after_fork_only"])
    none_found = sum(1 for r in records if r["n_firings_in_episode"] == 0)

    by_pol = defaultdict(lambda: [0, 0])
    by_opp = defaultdict(lambda: [0, 0])
    for r in records:
        by_pol[r["train_seed"]][0] += 1
        by_pol[r["train_seed"]][1] += int(r["precursor_before_or_at_fork"])
        by_opp[r["opponent"]][0] += 1
        by_opp[r["opponent"]][1] += int(r["precursor_before_or_at_fork"])

    doc = {
        "record": "C_fork precursor audit",
        "classification": "EVALUATION-ONLY DIAGNOSTIC — may not retune the predicate",
        "run_once": True,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_predicate": "artifacts/o3_preregistration/O3_CFORK_RECALL_AMENDMENT.json",
        "label_source": str(Path(args.results).relative_to(PROJECT_ROOT)),
        "qualified_forks_audited": len(records),
        "forks_with_precursor_before_or_at": len(covered),
        "coverage": round(coverage, 4),
        "precursor_after_fork_only": after_only,
        "no_precursor_found": none_found,
        "lead_time": {
            "mean": round(statistics.fmean(leads), 2) if leads else None,
            "median": statistics.median(leads) if leads else None,
            "min": min(leads) if leads else None,
            "max": max(leads) if leads else None,
        },
        "per_policy_coverage": {str(k): round(v[1] / v[0], 4) for k, v in sorted(by_pol.items())},
        "per_opponent_coverage": {k: round(v[1] / v[0], 4) for k, v in sorted(by_opp.items())},
        "interpretation_threshold": INTERPRETATION_THRESHOLD,
        "interpretation": (
            "coverage >= 0.50: an O3 failure is interpretable under the preregistered experiment"
            if coverage >= INTERPRETATION_THRESHOLD else
            "coverage < 0.50: an O3 FAILURE is INCONCLUSIVE_FOR_SPECIALIZATION"
        ),
        "positive_result_note": "A positive O3 crossover result stands regardless of coverage.",
        "expected_by_construction": (
            "Coverage should be essentially 1.0: a qualified fork requires "
            ">=2 legal team responses and lies inside a carry phase. Materially "
            "below 1.0 indicates a semantic or bookkeeping discrepancy worth "
            "investigating before O3, not a reason to edit the frozen predicate."
        ),
        "wall_seconds": round(time.time() - started, 1),
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"  forks audited            : {len(records)}")
    print(f"  precursor before/at fork : {len(covered)}   coverage {coverage:.4f}")
    print(f"  precursor AFTER fork only: {after_only}   (ideally 0)")
    print(f"  no precursor found       : {none_found}   (ideally 0)")
    if leads:
        print(f"  lead time (decisions)    : mean {statistics.fmean(leads):.1f}  "
              f"median {statistics.median(leads)}  min {min(leads)}  max {max(leads)}")
    print(f"  per-policy coverage      : {doc['per_policy_coverage']}")
    print(f"\n  {doc['interpretation']}")
    print(f"  wrote {OUT.relative_to(PROJECT_ROOT)}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
