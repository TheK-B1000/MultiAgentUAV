r"""A-vs-B behavioural trajectory: does the GET_FLAG->GO_TO drift found in
POLE_B_BEHAVIORAL_DIAGNOSIS track performance checkpoint-by-checkpoint, and is
it B-specific, general to both runs, or a phase A passes through and exits?

Governed by artifacts/strategic_demand/sppo/AB_BEHAVIORAL_TRAJECTORY_SPEC.json
(frozen before any episode). Reuses the EXACT measurement code from
eval_pole_b_behavioral_diagnosis.py (_snapshot, _step_features, run_cell,
summarize) rather than reimplementing it -- including the score-reading fix
(info["episode_result"] at the done instant, never a live core read after the
vec-env's auto-reset).

Each policy is instrumented on ITS OWN pole at 100k-step granularity across
its own training trajectory: A3 on Pole A, B3 on Pole B, plus one cross-check
cell (A3's final checkpoint on Pole B) to anchor against the prior diagnosis.

    python -m experiments.eval_ab_behavioral_trajectory [--dry-run]
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

import experiments.eval_pole_b_behavioral_diagnosis as D  # reuse validated instrument

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
LABEL = "AB_BEHAVIORAL_TRAJECTORY"
SPEC = SD / f"{LABEL}_SPEC.json"
EXPERIMENT_ID = LABEL

SEED_LO, N_SEEDS = 18_700_001, 32
SCALE = ROOT / "artifacts" / "scale_4v4_specialists"

A_DIR = SCALE / "pi_A_specialist_4v4_b3_entity_repair" / "ckpts"
B_DIR = SCALE / "pi_B_specialist_4v4_b3_entity_repair_corrected" / "ckpts"
STEPS = [100_000, 200_000, 300_000, 400_000, 500_000, 600_000, 700_000, 800_000, 900_000]

A_TRAJECTORY = (
    [("A_t0", SCALE / "pi_A_specialist_4v4_b3" / "ckpts" / "final_pi_A_specialist_4v4_b3.zip")]
    + [(f"A_t{s // 1000}k", A_DIR / f"ckpt_pi_A_specialist_4v4_b3_entity_repair_{s}.zip") for s in STEPS]
    + [("A_tfinal", A_DIR / "final_pi_A_specialist_4v4_b3_entity_repair.zip")]
)
B_TRAJECTORY = (
    [("B_t0", SCALE / "pi_B_specialist_4v4_b3" / "ckpts" / "final_pi_B_specialist_4v4_b3.zip")]
    + [(f"B_t{s // 1000}k", B_DIR / f"ckpt_pi_B_specialist_4v4_b3_entity_repair_corrected_{s}.zip") for s in STEPS]
    + [("B_tfinal", B_DIR / "final_pi_B_specialist_4v4_b3_entity_repair_corrected.zip")]
)
CROSS_CHECK = ("A_tfinal_on_PoleB", A_DIR / "final_pi_A_specialist_4v4_b3_entity_repair.zip")

SEALED_A3_SHA_PREFIX = "94dde69d091a"
SEALED_B3_SHA_PREFIX = "021342c84bbe"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from experiments.pole_attestation import (
        assert_resolved_matches_certification, governing_certification, resolve_pole_genome,
    )
    from experiments.run_lock import RunLock
    from rl.custom_ppo import load_custom_ppo_policy

    if not SPEC.is_file():
        raise SystemExit(f"REFUSING: frozen spec missing: {SPEC}")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: spec not frozen: {spec.get('status')!r}")

    out_path = SD / f"{LABEL}_RESULT.json"
    rows_csv = SD / f"{LABEL.lower()}_episode_rows.csv"
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

    all_cells = [(lbl, p, "A") for lbl, p in A_TRAJECTORY] \
        + [(lbl, p, "B") for lbl, p in B_TRAJECTORY] \
        + [(CROSS_CHECK[0], CROSS_CHECK[1], "B")]

    print(f"{LABEL}  {_now()}  device={device}")
    print(f"  spec      {SPEC.name} [{spec.get('status')}]")
    print(f"  question  {spec.get('THE_QUESTION')[:100]}...")
    print(f"  seeds     {seeds[0]}..{seeds[-1]} (n={len(seeds)})")
    print(f"  cells     {len(all_cells)} ({len(A_TRAJECTORY)} A-side + {len(B_TRAJECTORY)} B-side + 1 cross-check)")

    genomes, attestations = {}, {}
    for pole in ("A", "B"):
        g = resolve_pole_genome(pole, 4, str(D.POLE_B_GENOME) if pole == "B" else None)
        attestations[pole] = assert_resolved_matches_certification(pole, 4, cert_path, g, is_smoke=False)
        genomes[pole] = g
        print(f"  pole {pole}    {attestations[pole]['live_genome_id']} "
              f"MATCH={'PASS' if attestations[pole]['hashes_match'] else 'FAIL'}")

    shas = {}
    for lbl, p, _pole in all_cells:
        if not p.is_file():
            raise SystemExit(f"REFUSING: checkpoint missing for {lbl}: {p}")
        shas[lbl] = _sha(p)

    env, _c, _o = D._build_env(device, seeds[0], "A", genomes["A"])
    obs_space, act_space = env.observation_space, env.action_space
    env.close()

    a_labels = [lbl for lbl, _p in A_TRAJECTORY]
    b_labels = [lbl for lbl, _p in B_TRAJECTORY]
    contracts = {
        "K1a_A_trajectory_checkpoints_distinct": len(set(shas[l] for l in a_labels)) == len(a_labels),
        "K1b_B_trajectory_checkpoints_distinct": len(set(shas[l] for l in b_labels)) == len(b_labels),
        "K2_A_tfinal_matches_sealed_pi_A3": shas["A_tfinal"].startswith(SEALED_A3_SHA_PREFIX),
        "K3_B_tfinal_matches_corrected_pi_B3": shas["B_tfinal"].startswith(SEALED_B3_SHA_PREFIX),
        "K3b_cross_check_is_deliberately_A_tfinal": shas[CROSS_CHECK[0]] == shas["A_tfinal"],
        "K4_both_poles_attested": all(a["hashes_match"] for a in attestations.values()),
    }
    print("\n  known-answer contracts ...")
    for k, v in contracts.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    if not all(contracts.values()):
        raise SystemExit(f"FAIL-CLOSED: contracts failed {[k for k, v in contracts.items() if not v]}")

    if args.dry_run:
        print(f"\n  --dry-run: spec frozen, {len(all_cells)} checkpoints present, both poles "
              f"attested. NO episodes run, NOTHING written.")
        return 0

    fields = ["policy", "pole", "seed", "steps", "won", "blue_score", "red_score",
              "macro_switch_rate", "all_identical_cmd", "distinct_cmds", "distinct_macros",
              "n_get_flag", "n_go_home", "n_go_to", "n_mine", "mean_pair_dist",
              "n_defenders", "n_attackers", "mean_d_own_flag", "mean_d_enemy_flag",
              "n_tagged", "any_carrying", "n_pressured", "mean_support_dist"]

    summaries = {}
    with RunLock(SD / f"{LABEL}.run.lock", run_id=LABEL):
        with rows_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for lbl, p, pole in all_cells:
                policy = load_custom_ppo_policy(str(p), obs_space, act_space, device=device)
                policy.model.eval()
                per_ep, per_step, _states = D.run_cell(policy, lbl, pole, seeds, device, genomes[pole], w, fh)
                s = D.summarize(per_ep, per_step)
                summaries[lbl] = s
                print(f"\n  {lbl}: won={s['won']:.3f} score={s['blue_score']:.2f}-{s['red_score']:.2f} "
                      f"GET_FLAG={s['macro_fraction'].get('GET_FLAG', 0):.3f} "
                      f"GO_TO={s['macro_fraction'].get('GO_TO', 0):.3f} "
                      f"switch={s['macro_switch_rate']:.3f} att={s['n_attackers']:.2f} "
                      f"carry={s['any_carrying']:.3f}", flush=True)
                out_path.write_text(json.dumps(
                    {"record": f"{LABEL} (partial)", "status": "RUNNING", "utc": _now(),
                     "cells_done": sorted(summaries), "summaries": summaries}, indent=2),
                    encoding="utf-8")
                del policy

    out_path.write_text(json.dumps({
        "record": f"{LABEL} checkpoint-trajectory behavioural comparison (A control, B primary)",
        "status": "COMPLETE_DIAGNOSTIC", "utc": _now(), "device": device,
        "arm": "DIAGNOSTIC", "confirmatory": False,
        "implements": f"{SPEC.name}",
        "question": spec.get("THE_QUESTION"),
        "seeds": {"block": [seeds[0], seeds[-1]], "n": len(seeds)},
        "checkpoints": shas, "contracts": contracts,
        "pole_attestations": {p: {k: a[k] for k in ("certified_genome_id", "live_genome_id",
                                                    "certified_config_hash", "live_config_hash",
                                                    "hashes_match")}
                              for p, a in attestations.items()},
        "summaries": summaries,
        "NOT_A_CLAIM": spec.get("WHAT_THIS_EXPERIMENT_MAY_NOT_CLAIM"),
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {out_path}\n  -> {rows_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
