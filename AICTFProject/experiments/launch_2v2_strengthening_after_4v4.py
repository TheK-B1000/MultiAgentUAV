"""Wait for the 4v4 Stage-4 chain to package, then run the three frozen 2v2 strengthening studies.

User ordering (2026-10-05): the reserved 31xxxxxx blocks are not touched until 4v4 finishes.
This waiter only polls for 4v4/manifests/phase6_package.json; nothing else runs before it.

Then, in parallel:
  * BEHAVIOR_SIGNATURES_2V2_V1   3 shards -> merge + seal
  * Z_INTERVENTION_2V2_V1        2 shards -> merge + seal
  * REPLICATION_2V2_DUAL_BRANCH_V1  train (3 at a time) -> export -> 4 evals -> readout

Each study seals itself (write-once); a failure in one does not stop the others.

    Start-Process .venv\\Scripts\\python.exe experiments\\launch_2v2_strengthening_after_4v4.py -WindowStyle Hidden
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
GATE = ROOT / "4v4" / "manifests" / "phase6_package.json"
OUT = ROOT / "artifacts" / "strategic_demand" / "sppo" / "2v2_strengthening"
LOG = OUT / "launcher.log"
ENV = dict(os.environ, FOR_DISABLE_CONSOLE_CTRL_HANDLER="1")


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def start(argv: list[str], tag: str) -> subprocess.Popen:
    fh = (OUT / f"run_{tag}.log").open("a", encoding="utf-8")
    log(f"start {tag}: {' '.join(argv)}")
    return subprocess.Popen([PY, *argv], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=ENV)


def sharded(script: str, n: int, tag: str) -> subprocess.Popen | None:
    """Run n shards to completion, then the unsharded merge+seal call. Returns the seal process."""
    procs = [start([script, "--shard", f"{i}/{n}"], f"{tag}_shard{i}") for i in range(n)]
    for p in procs:
        p.wait()
    codes = [p.returncode for p in procs]
    log(f"{tag} shards exited {codes}")
    return start([script], f"{tag}_seal")


REQUIRED_STEPS = ("phase3_dataset", "phase4_students", "phase5_evals", "phase6_package")
STAGE4_LABELS = ("TOP50_4V4_STAGE4_SHARE_ENCODER", "TOP50_4V4_STAGE4_FULLY_SHARED_ZR", "TOP50_4V4_STAGE4_ROLE_ONLY")
OWN_TOP50_SEEDS = (ROOT / "artifacts" / "strategic_demand" / "sppo" / "dual_branch_v1" / "matched128_4v4"
                   / "DUAL_BRANCH_4V4_OWN_TOP50_seed_ids.json")


def fourv4_complete() -> list[str]:
    """Every required 4v4 stage sealed successfully. Returns the list of problems (empty = complete).

    The package marker alone is not enough: the 4v4 driver's eval step only WARNS and continues
    when a student eval exits without a result, so a package can exist over a missing eval.
    """
    import json
    problems = []
    for step in REQUIRED_STEPS:
        m = ROOT / "4v4" / "manifests" / f"{step}.json"
        if not m.is_file():
            problems.append(f"missing manifest {step}")
            continue
        if json.loads(m.read_text(encoding="utf-8")).get("status") != "COMPLETE":
            problems.append(f"manifest {step} not COMPLETE")
    own = sorted(int(s) for s in json.loads(OWN_TOP50_SEEDS.read_text(encoding="utf-8")))
    sd = ROOT / "artifacts" / "strategic_demand" / "sppo"
    for lab in STAGE4_LABELS:
        p = sd / f"{lab}_CROSSOVER_EVAL_RESULT.json"
        if not p.is_file():
            problems.append(f"missing {p.name}")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("status") != "SEALED":
            problems.append(f"{lab} status {d.get('status')!r}")
        got = sorted(int(s) for s in ((d.get("seeds") or {}).get("seed_ids") or d.get("seed_ids") or []))
        if got != own:
            problems.append(f"{lab} not evaluated on the 4v4 own top-50 seeds")
    return problems


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"waiting for 4v4 completion: manifests {REQUIRED_STEPS} COMPLETE + {len(STAGE4_LABELS)} "
        f"Stage-4 results SEALED on the own top-50 seeds")
    last = None
    while True:
        if GATE.is_file():
            problems = fourv4_complete()
            if not problems:
                break
            if problems != last:
                log(f"BLOCKED: 4v4 package exists but is incomplete: {problems}; not launching")
                last = problems
        time.sleep(120)
    log("4v4 complete (every required stage sealed); launching the three frozen 2v2 studies")

    import threading
    results = {}

    def run_sharded(script, n, tag):
        p = sharded(script, n, tag)
        p.wait()
        results[tag] = p.returncode
        log(f"{tag} seal exited {p.returncode}")

    def run_replication():
        p = start(["experiments/run_replication_2v2.py", "--phase", "all"], "replication")
        p.wait()
        results["replication"] = p.returncode
        log(f"replication exited {p.returncode}")

    threads = [
        threading.Thread(target=run_sharded, args=("experiments/run_behavior_signatures_2v2.py", 3, "behavior")),
        threading.Thread(target=run_sharded, args=("experiments/run_z_intervention_2v2.py", 2, "z")),
        threading.Thread(target=run_replication),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log(f"ALL DONE {results}")
    return 0 if all(v == 0 for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
