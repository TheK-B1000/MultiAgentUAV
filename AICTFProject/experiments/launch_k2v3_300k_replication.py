#!/usr/bin/env python3
"""K=2v3 — predeclared 300k confirmatory replication launcher.

LOCKED 2026-07-30 (before remaining discovery 200k / behavior-audit results).
See docs/research-progress-tracker.md §"LOCKED NEXT: 300k confirmatory
replication".

Formal checkpoint is EXACTLY 300k. Checkpoint selection after seeing results
is prohibited. Discovery 1M FAIL is unchanged by this experiment.

Contains NO PPO logic -- shells out to ``rl/train_ppo.py``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

C_RUSH_OPPONENT = "OP11_ADAPTIVE_EXPLOITER"
C_SPLIT_OPPONENT = "OP9_SPLIT_LANE_FEINT"
MAP = "map_b_split_lane"

# Predeclared formal budget — do not change after launch.
TOTAL_STEPS = 300_000
FORMAL_CHECKPOINT = 300_000
N_ENVS = 16
MAX_DECISION_STEPS = 240
CKPT_STEPS = 100_000  # 100k/200k trajectory only; formal gate uses 300k (= final)
AGENTS = 2
PRESET = "no_latent_baseline"

# Fresh training seeds — disjoint from discovery (901/902xxx), confirmations
# (620/681/691/701/711xxx), pilots (821/831001), and discovery eval blocks
# (1010xxx / 1020xxx).
RUSH_SEEDS = [911001, 911002, 911003, 911004, 911005]
SPLIT_SEEDS = [912001, 912002, 912003, 912004, 912005]

# Predeclared eval blocks for the confirmatory cross-eval (64 paired eps).
EVAL_SEED_BLOCKS = {
    "C_RUSH": {"base": 1_110_001, "n": 64},
    "C_SPLIT": {"base": 1_120_001, "n": 64},
}


def build_cmd(*, specialist: str, opponent: str, seed: int, python_exe: str) -> tuple[str, list[str]]:
    run_tag = f"k2v3_{specialist}_{opponent.split('_')[0].lower()}_mapb_s{seed}"
    ckpt_dir = str(PROJECT_ROOT / "checkpoints" / f"k2v3_{specialist}")
    art_dir = PROJECT_ROOT / "artifacts" / f"k2v3_{specialist}_train_s{seed}"
    art_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe, str(PROJECT_ROOT / "rl" / "train_ppo.py"),
        "--preset", PRESET,
        "--mode", "FIXED_OPPONENT",
        "--fixed-opponent", opponent,
        "--map-layout", MAP,
        "--agents", str(AGENTS),
        "--total-steps", str(TOTAL_STEPS),
        "--max-decision-steps", str(MAX_DECISION_STEPS),
        "--periodic-checkpoint-steps", str(CKPT_STEPS),
        "--n-envs", str(N_ENVS),
        "--seed", str(seed),
        "--device", "cuda",
        "--run-tag", run_tag,
        "--checkpoint-dir", ckpt_dir,
        "--metrics-csv", str(art_dir / "metrics.csv"),
        "--episode-csv", str(art_dir / "episodes.csv"),
        "--fresh-metrics-csv",
    ]
    return run_tag, cmd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--python", default=str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"))
    args = p.parse_args()

    jobs = [("piR", C_RUSH_OPPONENT, s) for s in RUSH_SEEDS]
    jobs += [("piS", C_SPLIT_OPPONENT, s) for s in SPLIT_SEEDS]

    manifest = {
        "experiment": "k2v3_300k_replication",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PREDECLARED",
        "formal_checkpoint": FORMAL_CHECKPOINT,
        "checkpoint_selection": "prohibited",
        "contexts": {
            "C_RUSH": f"{C_RUSH_OPPONENT}|{MAP}",
            "C_SPLIT": f"{C_SPLIT_OPPONENT}|{MAP}",
        },
        "shared_config": {
            "preset": PRESET, "agents": AGENTS, "total_timesteps": TOTAL_STEPS,
            "n_envs": N_ENVS, "max_decision_steps": MAX_DECISION_STEPS,
            "periodic_checkpoint_steps": CKPT_STEPS, "device": "cuda",
            "train_domain_randomization": False,
        },
        "training_seeds": {"piR": RUSH_SEEDS, "piS": SPLIT_SEEDS},
        "eval_seed_blocks": EVAL_SEED_BLOCKS,
        "confirmatory_gates": [
            "piR > piS on C_RUSH (paired family CI95 > 0)",
            "piS > piR on C_SPLIT (paired family CI95 > 0)",
            "LCB95(delta_pool) > 0 (hierarchical clustered bootstrap)",
            "LCB95(D_policy) > 0 where D_policy = JSD_between - mean(JSD_within_piR, JSD_within_piS)",
        ],
        "branch_birth_decision": {
            "all_four_pass": "retain 300k specialists -> latent birth -> freeze branches -> router later",
            "miss_delta_pool_or_policy": "promote piR as G0; begin learned-incumbent weakness sweep",
        },
        "discovery_reference": {
            "experiment": "k2v2_specialists",
            "formal_1m": "FAIL",
            "candidate_transient_specialization_step": 300_000,
            "note": "This replication was locked before discovery 200k / behavior-audit completion.",
        },
        "runs": [],
    }

    manifest_path = PROJECT_ROOT / "artifacts" / "k2v3_300k_replication_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for specialist, opponent, seed in jobs:
            tag, cmd = build_cmd(specialist=specialist, opponent=opponent,
                                 seed=seed, python_exe=args.python)
            print(f"[dry] {tag}")
            print("     ", " ".join(cmd))
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[dry] wrote {manifest_path}")
        return 0

    procs: list[tuple[str, subprocess.Popen, object]] = []

    def flush() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2))

    for specialist, opponent, seed in jobs:
        while sum(1 for _, pr, _ in procs if pr.poll() is None) >= args.concurrency:
            time.sleep(30)
        tag, cmd = build_cmd(specialist=specialist, opponent=opponent,
                             seed=seed, python_exe=args.python)
        art_dir = PROJECT_ROOT / "artifacts" / f"k2v3_{specialist}_train_s{seed}"
        log_f = open(art_dir / "train.log", "w")
        print(f"[launch] {tag}", flush=True)
        pr = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT))
        procs.append((tag, pr, log_f))
        manifest["runs"].append({
            "run_tag": tag, "specialist": specialist, "opponent": opponent,
            "map": MAP, "seed": seed, "cmd": cmd, "launched_utc": datetime.now(timezone.utc).isoformat(),
        })
        flush()
        time.sleep(5)

    failed = []
    for tag, pr, lf in procs:
        rc = pr.wait()
        lf.close()
        print(f"[done] {tag}: rc={rc}", flush=True)
        for r in manifest["runs"]:
            if r["run_tag"] == tag:
                r["returncode"] = rc
        if rc != 0:
            failed.append(tag)
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["failed_runs"] = failed
    manifest["status"] = "FAILED" if failed else "TRAINING_COMPLETE"
    flush()
    print(f"[queue] finished; failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
