#!/usr/bin/env python3
"""Orchestration-only launcher for the K=2 LRO specialist runs (step 4).

Contains NO PPO logic -- it shells out to ``rl/train_ppo.py`` with the
frozen-context configuration and writes a shared experiment manifest so
the six runs are auditable as one experiment.

Frozen contexts (see docs/research-progress-tracker.md, 2026-07-29):
    C_RUSH  = OP11_ADAPTIVE_EXPLOITER | map_b_split_lane
    C_SPLIT = OP9_SPLIT_LANE_FEINT    | map_b_split_lane

Identical across all six runs; only opponent + seed differ:
    preset no_latent_baseline, 2v2, 1M timesteps, n_envs=16,
    max_decision_steps=240 (MATCHES the episode length both contexts were
    confirmed at -- the preset default of 400 does NOT, and the cap binds
    in >50% of confirmed-context episodes), checkpoints every 100k,
    domain randomization off.
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

TOTAL_STEPS = 1_000_000
N_ENVS = 16                 # 1M / (2048 * 16) ~= 30.5 PPO updates
MAX_DECISION_STEPS = 240    # must match frozen-context confirmations
CKPT_STEPS = 100_000
AGENTS = 2
PRESET = "no_latent_baseline"

# Fresh training seeds -- disjoint from every evaluation/confirmation block
# used so far (620001, 631001, 641001, 651001, 661001, 671001, 681001,
# 691001, 701001, 711001) and from the pilot runs (821001, 831001).
RUSH_SEEDS = [901001, 901002, 901003]
SPLIT_SEEDS = [902001, 902002, 902003]


def build_cmd(*, specialist: str, opponent: str, seed: int, python_exe: str) -> tuple[str, list[str]]:
    run_tag = f"k2v2_{specialist}_{opponent.split('_')[0].lower()}_mapb_s{seed}"
    ckpt_dir = str(PROJECT_ROOT / "checkpoints" / f"k2v2_{specialist}")
    art_dir = PROJECT_ROOT / "artifacts" / f"k2v2_{specialist}_train_s{seed}"
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
    p.add_argument("--concurrency", type=int, default=2,
                   help="How many training runs to execute at once.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--python", default=str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"))
    args = p.parse_args()

    jobs = []
    for seed in RUSH_SEEDS:
        jobs.append(("piR", C_RUSH_OPPONENT, seed))
    for seed in SPLIT_SEEDS:
        jobs.append(("piS", C_SPLIT_OPPONENT, seed))

    manifest = {
        "experiment": "k2_lro_specialists_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
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
        "note": (
            "max_decision_steps=240 matches the frozen-context confirmations; "
            "the no_latent_baseline preset default of 400 does not, and the cap "
            "binds in >50% of confirmed-context episodes. Pilot runs "
            "k2_pi_split (OP9|map_b) and k2_pi_rush (OP6|map_a) used 400 and are "
            "NOT counted toward this experiment."
        ),
        "runs": [],
    }

    procs: list[tuple[str, subprocess.Popen, object]] = []
    manifest_path = PROJECT_ROOT / "artifacts" / "k2v2_specialists_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def flush_manifest() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2))

    for specialist, opponent, seed in jobs:
        run_tag, cmd = build_cmd(
            specialist=specialist, opponent=opponent, seed=seed, python_exe=args.python
        )
        manifest["runs"].append({
            "run_tag": run_tag, "specialist": specialist,
            "opponent": opponent, "map": MAP, "seed": seed, "cmd": cmd,
        })
        if args.dry_run:
            print(f"[dry-run] {run_tag}\n    {' '.join(cmd)}")
            continue
        while len([1 for _, pr, _ in procs if pr.poll() is None]) >= max(1, args.concurrency):
            time.sleep(20)
        log_path = PROJECT_ROOT / "artifacts" / f"k2v2_{specialist}_train_s{seed}" / "train.log"
        lf = open(log_path, "w")
        print(f"[launch] {run_tag} -> {log_path}")
        pr = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT))
        procs.append((run_tag, pr, lf))
        flush_manifest()

    flush_manifest()
    if args.dry_run:
        print(f"\n[dry-run] manifest would be written to {manifest_path}")
        return 0

    failed = []
    for run_tag, pr, lf in procs:
        rc = pr.wait()
        lf.close()
        status = "ok" if rc == 0 else f"FAILED rc={rc}"
        print(f"[done] {run_tag}: {status}")
        for r in manifest["runs"]:
            if r["run_tag"] == run_tag:
                r["returncode"] = rc
        if rc != 0:
            failed.append(run_tag)
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["failed_runs"] = failed
    flush_manifest()
    print(f"\nmanifest: {manifest_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
