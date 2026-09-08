"""SPPPO treatment-instantiation smoke: do assigned poles survive episode resets?

This is the acceptance test for AMENDMENT_ASSIGNED_POLE_PERSISTENCE. It must
cross MULTIPLE episode boundaries per environment, because that is precisely
where EXP2B and EXP2C broke: their per-env assignment was validated once at
construction and then silently resampled 50/50 for the rest of training.

Requirement, over completed episodes rather than a single reset:

    z = 0  =>  opponent = OP6   (100%)
    z = 1  =>  opponent = OP7   (100%)
    mismatches = 0

Evidence for the failure this replaces (from existing training artifacts):

    EXP2B   P(OP6|z0) 0.505   P(OP6|z1) 0.495
    EXP2C   P(OP6|z0) 0.496   P(OP6|z1) 0.493

i.e. both latents saw both opponents at chance, so the intended 16/0/0/16
treatment never ran.

Run:  python experiments/smoke_sppo_pole_persistence.py --steps 24576
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import shutil
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand"
OUT = SD / "sppo" / "pole_persistence_smoke"
RECORD = SD / "sppo" / "SPPPO_POLE_PERSISTENCE_SMOKE.json"

EXPECTED = {"0": "SCRIPTED:OP6", "1": "SCRIPTED:OP7"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=24_576,
                    help="6 rollouts of 4096; ~2 episodes per env at a 240-step horizon")
    ap.add_argument("--keep", action="store_true", help="keep the smoke run directory")
    a = ap.parse_args()

    from experiments.run_sppo_lambda_sweep import build_candidate, configure_sppo_live_environment
    from rl.training.orchestrator import orchestrate_training_run

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # A nonzero lambda so the ranking runner is live and its z/pole consistency
    # check is exercised on every minibatch, not merely the occupancy audit.
    cfg, _, parent_contract = build_candidate(0.03)
    cfg.total_timesteps = int(a.steps)
    cfg.run_tag = "sppo_pole_persistence_smoke"
    cfg.checkpoint_dir = str(OUT / "ckpts")
    cfg.metrics_csv_path = str(OUT / "metrics.csv")
    cfg.episode_csv_path = str(OUT / "episode_rows.csv")

    print("SPPPO POLE-PERSISTENCE SMOKE")
    print(f"  mode                 {cfg.mode}")
    print(f"  opponent_randomize   {cfg.opponent_randomize}")
    print(f"  steps                {cfg.total_timesteps}")
    print(f"  lambda_rank          {cfg.sppo_lambda_rank}  (ranking check live)\n", flush=True)

    orchestrate_training_run(
        cfg, pre_rollout_env_setup=partial(configure_sppo_live_environment,
                                           contract=parent_contract))

    rows = list(csv.DictReader(open(cfg.episode_csv_path, newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit("REFUSING: no completed episodes; the smoke proved nothing")
    ctr = collections.Counter((r["latent_z"], r["opponent"]) for r in rows)
    per_env = collections.Counter(r.get("env_index", "?") for r in rows)

    report, mismatches = {}, 0
    for z in ("0", "1"):
        tot = sum(v for (zz, _), v in ctr.items() if zz == z)
        if not tot:
            raise SystemExit(f"REFUSING: no completed episodes for z={z}")
        good = ctr[(z, EXPECTED[z])]
        bad = tot - good
        mismatches += bad
        report[f"z{z}"] = {"n_episodes": tot, "expected": EXPECTED[z],
                           "pct_expected": round(good / tot, 6), "mismatches": bad,
                           "observed": {o: c for (zz, o), c in ctr.items() if zz == z}}
        print(f"  z={z}  n={tot:4d}  {EXPECTED[z]} = {good/tot:.4f}   mismatches={bad}")

    n_ep_per_env = (min(per_env.values()) if per_env and "?" not in per_env else None)
    verdict = "PASS" if mismatches == 0 else "FAIL"
    rec = {
        "record": "SPPPO assigned-pole persistence smoke",
        "status": "FROZEN_RESULT",
        "requirement": "z0 -> OP6 100%, z1 -> OP7 100%, mismatches 0, across episode resets",
        "mode": str(cfg.mode),
        "opponent_randomize": bool(cfg.opponent_randomize),
        "steps": int(cfg.total_timesteps),
        "completed_episodes": len(rows),
        "min_episodes_per_env": n_ep_per_env,
        "crossed_episode_resets": len(rows) > 32,
        "by_latent": report,
        "total_mismatches": mismatches,
        "VERDICT": verdict,
        "contrast_with_prior_runs": {
            "EXP2B": {"P(OP6|z0)": 0.505, "P(OP6|z1)": 0.495},
            "EXP2C": {"P(OP6|z0)": 0.496, "P(OP6|z1)": 0.493},
            "note": "both at chance -- the 16/0/0/16 treatment never ran",
        },
    }
    RECORD.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\n  completed episodes {len(rows)}  (crossed resets: {len(rows) > 32})")
    print(f"  total mismatches   {mismatches}")
    print(f"  VERDICT: {verdict}\n  -> {RECORD}")
    if not a.keep:
        shutil.rmtree(OUT, ignore_errors=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
