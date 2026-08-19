"""R2b — continue pi_A and pi_B from their 1M terminals to 2M cumulative.

Frozen protocol: artifacts/strategic_demand/R2B_CONTINUATION_FROZEN.json

The ONLY variable changed is training budget. Poles, RULESET_V3_M1, PPO
configuration, observation space, reward, and training seeds are all held at
their R1 values, and training RESUMES from the existing terminal checkpoints
rather than restarting.

Resume semantics: --additional-steps N resolves to
``total_timesteps = checkpoint_step + N`` after the checkpoint loads, so
+1,000,000 on a checkpoint at ~1,001,472 lands at ~2,001,472 cumulative.

The generalist is deliberately NOT continued here. R2 failed without pi_G
participating in either contrast, so generalist undertraining cannot explain
that failure. pi_G returns only if the 2M specialists pass, at which point it
continues 2M -> 4M to restore compute parity before R3.

Opponent resolution goes through the same asserted R0 seam used by R1, so a
specialist cannot silently continue against the wrong pole.

Run:  python experiments/run_r2b_specialist_continuation.py --policy A
      python experiments/run_r2b_specialist_continuation.py --policy ALL
      python experiments/run_r2b_specialist_continuation.py --policy ALL --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PY = str(ROOT / ".venv/Scripts/python.exe")
SD = ROOT / "artifacts/strategic_demand"
R1 = SD / "r1_training"
OUT = SD / "r2b_continuation"
PROTOCOL = SD / "R2B_CONTINUATION_FROZEN.json"

ADDITIONAL_STEPS = 1_000_000

SPECS = {
    "A": {
        "run_tag": "r2b_pi_A_specialist_2M_seed7100001",
        "resume_from": R1 / "r1_pi_A_specialist_seed7100001/ckpts/final_r1_pi_A_specialist_seed7100001.zip",
        "seed": 7_100_001,
        "pool": ("OP6",),
        "pole": "A",
    },
    "B": {
        "run_tag": "r2b_pi_B_specialist_2M_seed7200001",
        "resume_from": R1 / "r1_pi_B_specialist_seed7200001/ckpts/final_r1_pi_B_specialist_seed7200001.zip",
        "seed": 7_200_001,
        "pool": ("OP7",),
        "pole": "B",
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def preflight(policy: str) -> dict:
    if not PROTOCOL.is_file():
        raise SystemExit("REFUSING: R2B_CONTINUATION_FROZEN.json missing")
    spec = SPECS[policy]
    ck = spec["resume_from"]
    if not ck.is_file():
        raise SystemExit(f"REFUSING: R1 terminal checkpoint missing: {ck}")
    if not ck.name.startswith("final_"):
        raise SystemExit(f"REFUSING: not a terminal checkpoint: {ck.name}")
    done = OUT / spec["run_tag"] / "result_summary.json"
    if done.is_file():
        raise SystemExit(f"REFUSING: {done} exists. Continuation already run; "
                         "no re-run, no checkpoint shopping.")
    return spec


def build_argv(spec: dict) -> list[str]:
    """CLI for the production trainer, mirroring R1 exactly except for budget."""
    run_dir = OUT / spec["run_tag"]
    return [
        PY, "-u", "rl/train_ppo.py",
        "--run-tag", spec["run_tag"],
        "--load", str(spec["resume_from"]),
        "--additional-steps", str(ADDITIONAL_STEPS),
        "--seed", str(spec["seed"]),
        "--checkpoint-dir", str(run_dir / "ckpts"),
        "--metrics-csv", str(run_dir / "metrics.csv"),
        "--episode-csv", str(run_dir / "episode_rows.csv"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=("A", "B", "ALL"), required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    todo = ["A", "B"] if a.policy == "ALL" else [a.policy]
    print(f"R2b SPECIALIST CONTINUATION  {_now()}")
    print(f"  protocol         {PROTOCOL.name}")
    print(f"  additional steps {ADDITIONAL_STEPS:,} (cumulative target ~2,000,000)")
    print(f"  generalist       NOT continued (returns only if 2M specialists pass)")

    for pol in todo:
        spec = preflight(pol)
        argv = build_argv(spec)
        print(f"\n  [{pol}] resume  {spec['resume_from'].name}")
        print(f"      seed    {spec['seed']}  pole {spec['pole']}  pool {spec['pool']}")
        print(f"      cmd     {' '.join(argv[2:])}")
        if a.dry_run:
            continue
        run_dir = OUT / spec["run_tag"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "r2b_manifest.json").write_text(json.dumps({
            "protocol": str(PROTOCOL.relative_to(ROOT)),
            "policy": pol, "run_tag": spec["run_tag"],
            "resume_from": str(spec["resume_from"].relative_to(ROOT)),
            "additional_steps": ADDITIONAL_STEPS,
            "seed": spec["seed"], "pole": spec["pole"],
            "only_variable_changed": "training budget",
            "scored_checkpoint": "2M terminal only; no checkpoint shopping",
            "utc": _now(),
        }, indent=2), encoding="utf-8")
        log = run_dir / "train.log"
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.call(argv, cwd=str(ROOT), stdout=fh,
                                 stderr=subprocess.STDOUT)
        print(f"      exit    {rc}  (log: {log.relative_to(ROOT)})")
        if rc != 0:
            print("      FAILED -- stopping; do not continue the other specialist "
                  "on a broken run")
            return rc

    if a.dry_run:
        print("\nDRY RUN -- nothing launched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
