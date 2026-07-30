#!/usr/bin/env python3
"""Write preregistration.lock.json for K=2v3 300k replication (Revision 3)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "k2v3_300k_replication"
OUT.mkdir(parents=True, exist_ok=True)

FILES = [
    "docs/k2v2-300k-replication-preregistration.md",
    "experiments/launch_k2v3_300k_replication.py",
    "artifacts/k2v3_300k_replication/manifest.json",
    "experiments/analyze_k2_assigned_gain.py",
    "experiments/analyze_k2_specialist_crossover.py",
    "experiments/analyze_k2_specialist_behavior.py",
    "experiments/audit_k2_specialist_behavior.py",
    "experiments/run_k2_specialist_cross_eval.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    hashes = {}
    for rel in FILES:
        p = ROOT / rel
        if not p.exists():
            print(f"[warn] missing {rel}")
            continue
        hashes[rel] = sha256(p)

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT.parent if (ROOT.parent / ".git").exists() else ROOT),
        text=True,
    ).strip()
    # repo root may be MultiAgentUAV
    repo = ROOT if (ROOT / ".git").exists() else ROOT.parent
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo), text=True).strip()
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(repo), text=True
    ).strip()

    lock = {
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "revision": 3,
        "experiment": "k2v3_300k_replication",
        "status": "PREDECLARED_AMENDED_FROZEN_NOT_LAUNCHED",
        "preregistration": {
            "path": "docs/k2v2-300k-replication-preregistration.md",
            "sha256": hashes.get("docs/k2v2-300k-replication-preregistration.md"),
        },
        "file_sha256": hashes,
        "git": {"commit": commit, "branch": branch, "note": "commit SHA at freeze time; amend commit follows"},
        "design": {
            "training_seeds_per_family": 6,
            "total_runs": 12,
            "steps_per_run": 300000,
            "formal_checkpoint": 300000,
            "eval_seeds_per_context": 256,
            "one_fixed_eval_block": True,
            "interim_analysis_permitted": False,
            "piR_seeds": [911001, 911002, 911003, 911004, 911005, 911006],
            "piS_seeds": [912001, 912002, 912003, 912004, 912005, 912006],
            "eval_seed_blocks": {
                "C_RUSH": {"base": 1110001, "n": 256},
                "C_SPLIT": {"base": 1120001, "n": 256},
            },
        },
        "primary_gates": {
            "A_joint_payoff": (
                "LCB95(Delta_assigned) > 0; "
                "Delta_assigned = 0.5*min(R_R-S_R, S_S-R_S) = V_assigned - V_fixed"
            ),
            "B_policy_distinction": (
                "LCB95(D_policy) > 0; "
                "D_policy = between - mean(within_piR, within_piS)"
            ),
        },
        "diagnostics_not_gates": [
            "paired directional crossover CIs",
            "Delta_pool (structural floor at 0)",
        ],
        "void_staged_draft": {
            "description": "5 seeds/family, 64 eval/context, LCB(Delta_pool)>0",
            "launched": False,
        },
        "launch_policy": (
            "Do not launch until discovery 200k + behavior audit finish and GPU is free; "
            "then launch all 12 runs. Audit must not change this freeze."
        ),
        "note": "Frozen before any replication training run started. 1M verdict remains FAIL.",
    }
    path = OUT / "preregistration.lock.json"
    path.write_text(json.dumps(lock, indent=2) + "\n")
    # Keep legacy lock pointer updated so older paths do not resurrect 5x64.
    legacy = ROOT / "artifacts" / "k2v2_300k_replication" / "preregistration.lock.json"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(json.dumps({**lock, "redirect": str(path.relative_to(ROOT))}, indent=2) + "\n")
    print(f"wrote {path}")
    print(f"prereg sha256={lock['preregistration']['sha256']}")
    print(f"launcher sha256={hashes.get('experiments/launch_k2v3_300k_replication.py')}")
    print(f"manifest sha256={hashes.get('artifacts/k2v3_300k_replication/manifest.json')}")
    print(f"git HEAD (pre-commit)={commit} branch={branch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
