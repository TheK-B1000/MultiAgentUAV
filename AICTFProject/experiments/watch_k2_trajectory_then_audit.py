#!/usr/bin/env python3
"""Wait for discovery trajectory (200k) to finish, then analyze + behavior-audit.

Does NOT launch the 300k replication. Does NOT alter the formal 1M FAIL.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
ROWS = ROOT / "artifacts" / "k2v2_specialist_cross_eval" / "episode_rows.csv"
LOG = ROOT / "artifacts" / "k2v2_specialist_cross_eval" / "post_trajectory.log"


def traj_done() -> bool:
    import pandas as pd
    if not ROWS.exists():
        return False
    df = pd.read_csv(ROWS)
    n200 = int((df["checkpoint_step"] == 200_000).sum())
    return n200 >= 384


def eval_still_running() -> bool:
    try:
        import psutil
    except ImportError:
        return False
    for p in psutil.process_iter(["cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
        except Exception:
            continue
        if "run_k2_specialist_cross_eval.py" in cmd and "200000" in cmd:
            return True
    return False


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as log:
        def say(msg: str) -> None:
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()

        say(f"[watcher] started; waiting for 200k rows (384)...")
        while not traj_done():
            say(f"[watcher] 200k incomplete; sleep 120s (eval_running={eval_still_running()})")
            time.sleep(120)

        say("[watcher] 200k complete — running DIAGNOSTIC analyzer")
        r = subprocess.run(
            [str(PY), "experiments/analyze_k2_specialist_crossover.py",
             "--rows", str(ROWS), "--step", "200000"],
            cwd=str(ROOT), capture_output=True, text=True,
        )
        say(r.stdout)
        if r.stderr:
            say(r.stderr)
        (ROOT / "artifacts" / "k2v2_specialist_cross_eval" / "analyze_200k.txt").write_text(
            r.stdout + r.stderr
        )

        say("[watcher] running C_SPLIT curve summary")
        subprocess.run(
            [str(PY), "experiments/summarize_k2_trajectory_curves.py",
             "--rows", str(ROWS)],
            cwd=str(ROOT), check=False,
        )

        say("[watcher] launching behavior audit @ 300k 500k 1000000 (8 eps/pair)")
        ba = subprocess.run(
            [str(PY), "experiments/audit_k2_specialist_behavior.py",
             "--checkpoints", "300000", "500000", "1000000",
             "--episodes", "8", "--device", "cuda"],
            cwd=str(ROOT),
        )
        say(f"[watcher] behavior audit rc={ba.returncode}")
        say("[watcher] discovery chain complete. Formal 1M FAIL unchanged.")
        say("[watcher] Rev 3 freeze must already be on disk; launching k2v3 12x300k NOW.")
        launch = subprocess.run(
            [str(PY), "experiments/launch_k2v3_300k_replication.py",
             "--force-launch", "--concurrency", "2"],
            cwd=str(ROOT),
        )
        say(f"[watcher] k2v3 launch rc={launch.returncode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
