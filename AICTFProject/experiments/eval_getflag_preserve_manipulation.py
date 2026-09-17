r"""Matched-state manipulation check for B_GETFLAG_PRESERVE.

Governed by artifacts/strategic_demand/sppo/B_GETFLAG_PRESERVE_SPEC.json
#EVALUATION_after_training_not_during.manipulation_check.

Bfinal DRIVES the Pole-B trajectory (same seeds as STATE_CONDITIONED_DIAGNOSIS
so visit distribution is paired). The NEW continuation checkpoint and frozen
B_better (B_t500k) are queried COUNTERFACTUALLY at every visited state.

Primary metric (frozen before training):
  among agent-steps with carrying=False AND B_better argmax=GET_FLAG,
  P(student argmax=GO_TO)

Compared to the sealed diagnosis baseline on the same metric from
state_conditioned_diagnosis_agent_step_rows.csv (Bfinal as student).

    python -m experiments.eval_getflag_preserve_manipulation [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.eval_pole_b_behavioral_diagnosis as D
import experiments.eval_state_conditioned_diagnosis as SCD

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
LABEL = "B_GETFLAG_PRESERVE_MANIPULATION"
SPEC = SD / "B_GETFLAG_PRESERVE_SPEC.json"
EXPERIMENT_ID = LABEL

SEED_LO, N_SEEDS = 18_900_001, 48
SCALE = ROOT / "artifacts" / "scale_4v4_specialists"
CORR = SCALE / "pi_B_specialist_4v4_b3_entity_repair_corrected" / "ckpts"
NEW = (
    ROOT / "artifacts" / "exploratory_scale_4v4_specialists"
    / "exploratory_pi_B_specialist_4v4_b3_entity_repair_getflag_preserve" / "ckpts"
    / "final_exploratory_pi_B_specialist_4v4_b3_entity_repair_getflag_preserve.zip"
)
DRIVER = ("Bfinal_driver", CORR / "final_pi_B_specialist_4v4_b3_entity_repair_corrected.zip")
STUDENT = ("NEW", NEW)
BETTER = ("B_t500k", CORR / "ckpt_pi_B_specialist_4v4_b3_entity_repair_corrected_500000.zip")
SHA_DRIVER = "021342c84bbe"
SHA_BETTER = "d4c0d7ba2477"
BASELINE_CSV = SD / "state_conditioned_diagnosis_agent_step_rows.csv"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _macro_of(action: np.ndarray, i: int) -> int:
    return int(np.asarray(action).reshape(-1)[2 * i])


def drift_rate_from_rows(rows) -> dict:
    """P(student=GO_TO | ¬carrying ∧ better=GET_FLAG)."""
    n = n_goto = 0
    for r in rows:
        carrying = r["carrying"] if isinstance(r["carrying"], int) else int(r["carrying"])
        better = r["macro_better"]
        student = r["macro_student"] if "macro_student" in r else r["macro_worse"]
        if carrying != 0:
            continue
        if better != "GET_FLAG":
            continue
        n += 1
        if student == "GO_TO":
            n_goto += 1
    return {
        "n_eligible": n,
        "n_goto": n_goto,
        "rate": (n_goto / n) if n else float("nan"),
    }


def sealed_baseline() -> dict:
    if not BASELINE_CSV.is_file():
        raise SystemExit(f"REFUSING: sealed diagnosis rows missing: {BASELINE_CSV}")
    with BASELINE_CSV.open(newline="", encoding="utf-8") as fh:
        return drift_rate_from_rows(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from experiments.pole_attestation import (
        assert_resolved_matches_certification,
        governing_certification,
        resolve_pole_genome,
    )
    from experiments.run_lock import RunLock
    from experiments.tqdm_loop import set_postfix, tqdm_iter
    from gpu_env._core._entity_obs import augment_obs_with_entities
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
            raise SystemExit(
                f"FAIL-CLOSED (Rule 9): seed block for {EXPERIMENT_ID!r} not "
                f"registered as {SEED_LO}..{SEED_LO + N_SEEDS - 1}"
            )
        if b.get("subdivides") != "STATE_CONDITIONED_DIAGNOSIS":
            raise SystemExit(
                "FAIL-CLOSED: manipulation seeds must subdivide STATE_CONDITIONED_DIAGNOSIS"
            )
        if out_path.is_file() or rows_csv.is_file():
            raise SystemExit(f"REFUSING: output for {LABEL} already exists; one-shot")

    for lbl, p in (DRIVER, STUDENT, BETTER):
        if not Path(p).is_file():
            raise SystemExit(f"REFUSING: checkpoint missing ({lbl}): {p}")

    baseline = sealed_baseline()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    seeds = list(range(SEED_LO, SEED_LO + N_SEEDS))
    _v, cert_path = governing_certification(4)

    print(f"{LABEL}  {_now()}  device={device}")
    print(f"  spec         {SPEC.name} [{spec.get('status')}]")
    print(f"  driver       {DRIVER[0]} sha {_sha(DRIVER[1])[:12]}...  (drives trajectory)")
    print(f"  student NEW  sha {_sha(STUDENT[1])[:12]}...  (counterfactual query)")
    print(f"  B_better     sha {_sha(BETTER[1])[:12]}...")
    print(f"  seeds        {seeds[0]}..{seeds[-1]} (n={len(seeds)}; subdivides STATE_CONDITIONED_DIAGNOSIS)")
    print(f"  sealed baseline P(GO_TO|¬carry,better=GET_FLAG) = "
          f"{baseline['rate']:.6f}  (n={baseline['n_eligible']})")

    genome = resolve_pole_genome("B", 4, str(D.POLE_B_GENOME))
    attestation = assert_resolved_matches_certification(
        "B", 4, cert_path, genome, is_smoke=False
    )
    print(f"  pole B       {attestation['live_genome_id']} "
          f"MATCH={'PASS' if attestation['hashes_match'] else 'FAIL'}")

    env, _c, _o = D._build_env(device, seeds[0], "B", genome)
    obs_space, act_space = env.observation_space, env.action_space
    env.close()

    policies = {
        lbl: load_custom_ppo_policy(str(p), obs_space, act_space, device=device)
        for lbl, p in (DRIVER, STUDENT, BETTER)
    }
    for pol in policies.values():
        pol.model.eval()

    contracts = {
        "K1_driver_is_sealed_Bfinal": _sha(DRIVER[1]).startswith(SHA_DRIVER),
        "K2_better_is_sealed_Bt500k": _sha(BETTER[1]).startswith(SHA_BETTER),
        "K3_pole_attested": bool(attestation["hashes_match"]),
        "K4_student_ckpt_exists": STUDENT[1].is_file(),
        "K5_baseline_csv_present": BASELINE_CSV.is_file(),
    }
    env, core, obs = D._build_env(device, seeds[0], "B", genome)
    try:
        from gpu_env._core._entity_obs import augment_obs_with_entities
        obs = augment_obs_with_entities(obs, core, side="blue")
        a1, _ = policies[BETTER[0]].predict(obs, deterministic=True)
        a2, _ = policies[BETTER[0]].predict(obs, deterministic=True)
        contracts["K6_counterfactual_query_deterministic"] = bool(
            np.array_equal(np.asarray(a1), np.asarray(a2))
        )
    finally:
        env.close()

    print("\n  known-answer contracts ...")
    for k, v in contracts.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    if not all(contracts.values()):
        raise SystemExit(
            f"FAIL-CLOSED: contracts failed {[k for k, v in contracts.items() if not v]}"
        )

    if args.dry_run:
        print("\n  --dry-run: contracts OK, baseline computed. NO episodes, NOTHING written.")
        return 0

    fields = [
        "seed", "t", "agent", "carrying", "pressured", "near_enemy_flag", "near_own_flag",
        "macro_driver", "macro_student", "macro_better",
    ]
    collected: list[dict] = []

    with RunLock(SD / f"{LABEL}.run.lock", run_id=LABEL):
        with rows_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            bar = tqdm_iter(seeds, desc=LABEL, unit="ep")
            for seed in bar:
                set_postfix(bar, f"seed={seed}")
                env, core, obs = D._build_env(device, seed, "B", genome)
                try:
                    for pol in policies.values():
                        pol.reset_strategy()
                    obs = augment_obs_with_entities(obs, core, side="blue")
                    for t in range(R2.MAX_STEPS):
                        ctx, _team = SCD._per_agent_context(core)
                        with torch.no_grad():
                            a_drive, _ = policies[DRIVER[0]].predict(obs, deterministic=True)
                            a_student, _ = policies[STUDENT[0]].predict(obs, deterministic=True)
                            a_better, _ = policies[BETTER[0]].predict(obs, deterministic=True)
                        for i in range(4):
                            md = _macro_of(a_drive, i)
                            ms = _macro_of(a_student, i)
                            mb = _macro_of(a_better, i)
                            row = {
                                "seed": seed, "t": t, "agent": i,
                                "carrying": int(ctx[i]["carrying"]),
                                "pressured": int(ctx[i]["pressured"]),
                                "near_enemy_flag": int(ctx[i]["near_enemy_flag"]),
                                "near_own_flag": int(ctx[i]["near_own_flag"]),
                                "macro_driver": D.MACRO_NAMES.get(md, str(md)),
                                "macro_student": D.MACRO_NAMES.get(ms, str(ms)),
                                "macro_better": D.MACRO_NAMES.get(mb, str(mb)),
                            }
                            w.writerow(row)
                            collected.append(row)
                        env.step_async(a_drive)
                        obs, _r, done, info = env.step_wait()
                        obs["global_state"] = env.state()
                        obs = augment_obs_with_entities(obs, core, side="blue")
                        if bool(np.asarray(done).any()):
                            break
                    fh.flush()
                finally:
                    env.close()

    new_metric = drift_rate_from_rows(collected)
    manipulated = bool(
        new_metric["n_eligible"] > 0
        and new_metric["rate"] < baseline["rate"]
    )
    delta = float(new_metric["rate"] - baseline["rate"])

    print(f"\n  BASELINE (Bfinal as student):  rate={baseline['rate']:.6f}  "
          f"n={baseline['n_eligible']}")
    print(f"  NEW      (continuation):       rate={new_metric['rate']:.6f}  "
          f"n={new_metric['n_eligible']}")
    print(f"  delta (new - baseline):        {delta:+.6f}")
    print(f"  manipulated (strictly lower):  {manipulated}")

    payload = {
        "record": f"{LABEL} matched-state GET_FLAG→GO_TO drift manipulation check",
        "status": "COMPLETE_DIAGNOSTIC",
        "utc": _now(),
        "device": device,
        "arm": "EXPLORATORY",
        "confirmatory": False,
        "implements": SPEC.name,
        "question": (
            "Did the intervention actually reduce the non-carrying "
            "GET_FLAG→GO_TO drift?"
        ),
        "driver": {"label": DRIVER[0], "sha256": _sha(DRIVER[1])},
        "student": {"label": STUDENT[0], "path": str(STUDENT[1]), "sha256": _sha(STUDENT[1])},
        "B_better": {"label": BETTER[0], "sha256": _sha(BETTER[1])},
        "seeds": {"block": [seeds[0], seeds[-1]], "n": len(seeds),
                  "subdivides": "STATE_CONDITIONED_DIAGNOSIS"},
        "contracts": contracts,
        "metric": (
            "among agent-steps with carrying=False AND B_better argmax=GET_FLAG, "
            "P(student argmax=GO_TO)"
        ),
        "baseline_Bfinal": baseline,
        "new_continuation": new_metric,
        "delta_new_minus_baseline": delta,
        "manipulated": manipulated,
        "kill_table_axis": "drift_manipulated",
        "NOT_A_CLAIM": [
            "any Δ_B result — that is the separate crossover eval",
            "causation of specialization — only whether the diagnosed drift moved",
            "confirmatory status",
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  -> {out_path}\n  -> {rows_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
