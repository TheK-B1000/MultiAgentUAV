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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"waiting for {GATE.relative_to(ROOT)} (4v4 Stage-4 package)")
    while not GATE.is_file():
        time.sleep(120)
    log("4v4 packaged; launching the three frozen 2v2 studies")

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
