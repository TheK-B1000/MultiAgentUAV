#!/usr/bin/env python3
"""K=2v3 — predeclared 300k confirmatory replication launcher.

PREREGISTRATION AMENDMENT 2026-07-30 (before any training data exists).
Supersedes the earlier staged 5×64 / Δ_pool draft. See:
  artifacts/k2v3_300k_replication/PREREGISTRATION_AMENDMENT_2026-07-30.md

Formal checkpoint is EXACTLY 300k. Checkpoint selection after seeing results
is prohibited. Discovery 1M FAIL is unchanged by this experiment.

Contains NO PPO logic -- shells out to ``rl/train_ppo.py``.

DO NOT LAUNCH until discovery trajectory + behavior audit complete and GPU
is free. Audit results must NOT change sample size, behavior statistic, or
pass criteria after this freeze.
"""
from __future__ import annotations

import argparse
import hashlib
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

# Predeclared formal budget — frozen; do not change after preregistration.
TOTAL_STEPS = 300_000
FORMAL_CHECKPOINT = 300_000
N_ENVS = 16
MAX_DECISION_STEPS = 240
CKPT_STEPS = 100_000  # 100k/200k trajectory only; formal gate uses final 300k
AGENTS = 2
PRESET = "no_latent_baseline"

# 6 training seeds per family (amended from staged 5).
RUSH_SEEDS = [911001, 911002, 911003, 911004, 911005, 911006]
SPLIT_SEEDS = [912001, 912002, 912003, 912004, 912005, 912006]

# 256 paired eval episodes per context (amended from staged 64).
# No interim analysis at 64 or 128.
EVAL_SEED_BLOCKS = {
    "C_RUSH": {"base": 1_110_001, "n": 256},
    "C_SPLIT": {"base": 1_120_001, "n": 256},
}

OWN_CONTEXT = {"piR": "C_RUSH", "piS": "C_SPLIT"}


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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def build_manifest() -> dict:
    return {
        "experiment": "k2v3_300k_replication",
        "preregistration_amendment": "2026-07-30",
        "status": "PREDECLARED_AMENDED_FROZEN",
        "formal_checkpoint": FORMAL_CHECKPOINT,
        "checkpoint_selection": "prohibited",
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
        "training_seeds": {"piR": RUSH_SEEDS, "piS": SPLIT_SEEDS},
        "n_training_runs": len(RUSH_SEEDS) + len(SPLIT_SEEDS),
        "eval_seed_blocks": EVAL_SEED_BLOCKS,
        "eval_interim_analysis": "prohibited (no peek at 64 or 128)",
        "own_context": OWN_CONTEXT,
        "formal_gates": {
            "1_joint_complementary_payoff": (
                "LCB95(delta_assigned) > 0 under hierarchical clustered bootstrap; "
                "V_assigned = mean(payoff(piR, C_RUSH), payoff(piS, C_SPLIT)); "
                "V_fixed = max_f mean_c payoff(f, c); "
                "delta_assigned = V_assigned - V_fixed"
            ),
            "2_learned_policy_distinction": (
                "LCB95(D_policy) > 0 where "
                "D_policy = JSD_between - mean(JSD_within_piR, JSD_within_piS) "
                "on matched legal observations"
            ),
        },
        "reported_diagnostics_not_gates": [
            "paired directional crossover CI: piR-piS on C_RUSH",
            "paired directional crossover CI: piS-piR on C_SPLIT",
            "delta_pool (structural floor at 0; NOT a formal gate)",
        ],
        "supersedes": {
            "staged_draft": "5 seeds/family, 64 eval/context, LCB(delta_pool)>0 as formal gate",
            "reason": (
                "Staged draft was weaker than the locked confirmatory design; "
                "amended before any training data existed."
            ),
        },
        "branch_birth_decision": {
            "both_formal_gates_pass": (
                "retain 300k specialists -> latent birth -> freeze branches -> router later"
            ),
            "miss_either_gate": (
                "transient crossover not reliable enough; "
                "promote piR as G0; begin learned-incumbent weakness sweep"
            ),
        },
        "discovery_reference": {
            "experiment": "k2v2_specialists",
            "formal_1m": "FAIL",
            "candidate_transient_specialization_step": 300_000,
            "note": (
                "Amendment locked independent of remaining discovery 200k / "
                "behavior-audit outcomes; those must not change this spec."
            ),
        },
        "launch_policy": (
            "Do not launch until discovery trajectory + behavior audit finish "
            "and GPU is free. Then launch all 12 runs immediately."
        ),
        "runs": [],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--dry-run", action="store_true",
                   help="Write/update frozen manifest only; do not train.")
    p.add_argument("--write-manifest-only", action="store_true",
                   help="Alias for --dry-run (preregistration freeze).")
    p.add_argument("--python", default=str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"))
    p.add_argument("--force-launch", action="store_true",
                   help="Actually start training. Requires explicit flag.")
    args = p.parse_args()

    out_dir = PROJECT_ROOT / "artifacts" / "k2v3_300k_replication"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    jobs = [("piR", C_RUSH_OPPONENT, s) for s in RUSH_SEEDS]
    jobs += [("piS", C_SPLIT_OPPONENT, s) for s in SPLIT_SEEDS]
    assert len(jobs) == 12

    manifest = build_manifest()

    if args.dry_run or args.write_manifest_only or not args.force_launch:
        for specialist, opponent, seed in jobs:
            tag, cmd = build_cmd(specialist=specialist, opponent=opponent,
                                 seed=seed, python_exe=args.python)
            print(f"[dry] {tag}")
            manifest["runs"].append({
                "run_tag": tag, "specialist": specialist, "opponent": opponent,
                "map": MAP, "seed": seed, "cmd": cmd, "status": "not_launched",
            })
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        # Also keep legacy path in sync for any stale pointers.
        legacy = PROJECT_ROOT / "artifacts" / "k2v3_300k_replication_manifest.json"
        legacy.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[freeze] wrote {manifest_path}")
        print(f"[freeze] sha256(manifest)={_sha256(manifest_path)}")
        print(f"[freeze] sha256(launcher)={_sha256(Path(__file__))}")
        if not args.force_launch:
            print("[freeze] NOT LAUNCHED (pass --force-launch only when GPU is free "
                  "after discovery audit).")
            return 0

    # ---- actual launch (only with --force-launch) -----------------------
    procs: list[tuple[str, subprocess.Popen, object]] = []

    def flush() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    manifest["status"] = "LAUNCHING"
    manifest["launched_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["runs"] = []
    flush()

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
            "map": MAP, "seed": seed, "cmd": cmd,
            "launched_utc": datetime.now(timezone.utc).isoformat(),
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
