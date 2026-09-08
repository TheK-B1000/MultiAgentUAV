#!/usr/bin/env python3
"""Write preregistration.lock.json for K=2v3 300k replication (Revision 4)."""
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
    "experiments/analyze_k2_behavior_gate.py",
    "experiments/analyze_k2_specialist_crossover.py",
    "experiments/analyze_k2_specialist_behavior.py",
    "experiments/audit_k2_specialist_behavior.py",
    "experiments/run_k2_specialist_cross_eval.py",
    "experiments/watch_k2_trajectory_then_audit.py",
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

    repo = ROOT if (ROOT / ".git").exists() else ROOT.parent
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo), text=True).strip()
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(repo), text=True
    ).strip()

    lock = {
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "revision": 4,
        "experiment": "k2v3_300k_replication",
        "status": "PREDECLARED_AMENDED_FROZEN_REV4_NOT_LAUNCHED",
        "preregistration": {
            "path": "docs/k2v2-300k-replication-preregistration.md",
            "sha256": hashes.get("docs/k2v2-300k-replication-preregistration.md"),
        },
        "file_sha256": hashes,
        "git": {
            "commit": commit,
            "branch": branch,
            "note": "HEAD at freeze-script run; follow-up commit records this lock",
        },
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
            "B_distinct": (
                "LCB95(B_distinct) > 0; "
                "B_distinct = median(JSD_between) - Q_0.95(JSD_within)"
            ),
        },
        "diagnostics_not_gates": [
            "D_policy",
            "separation_ratio",
            "paired directional crossover CIs",
            "Delta_pool (structural floor at 0)",
            "argmax disagreement",
            "pairwise JSD matrices",
        ],
        "void_as_formal_gate": {
            "rev3_B": "LCB95(D_policy)>0 — passed by collapsed 1M generalists; decorative",
        },
        "launch_policy": (
            "Watcher auto-launch DISABLED. Explicit --force-launch only after this "
            "Rev 4 freeze is committed and discovery audit releases the GPU."
        ),
        "note": "Frozen before any replication training run started. 1M verdict remains FAIL.",
    }
    path = OUT / "preregistration.lock.json"
    path.write_text(json.dumps(lock, indent=2) + "\n")
    legacy = ROOT / "artifacts" / "k2v2_300k_replication" / "preregistration.lock.json"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(json.dumps({**lock, "redirect": str(path.relative_to(ROOT))}, indent=2) + "\n")
    print(f"wrote {path}")
    print(f"prereg sha256={lock['preregistration']['sha256']}")
    print(f"launcher sha256={hashes.get('experiments/launch_k2v3_300k_replication.py')}")
    print(f"B_distinct analyzer sha256={hashes.get('experiments/analyze_k2_behavior_gate.py')}")
    print(f"manifest sha256={hashes.get('artifacts/k2v3_300k_replication/manifest.json')}")
    print(f"git HEAD={commit} branch={branch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
