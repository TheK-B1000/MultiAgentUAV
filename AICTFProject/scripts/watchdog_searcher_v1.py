"""Watchdog for the frozen Searcher V1 run -- relaunches on crash, captures
stderr, and does NOT touch the preregistered search itself.

Two silent crashes occurred in this session with no diagnostic trace: the
first survived generation 1 then died mid-candidate; the second died within
seconds of restarting, before completing even one re-derived candidate. GPU is
healthy (1% util, 2.8/12.2GB) -- no evidence of resource contention -- so the
cause is unknown rather than fixed. Capturing stderr is a mechanical
diagnostic change (does not touch treatment semantics, RNG, genome logic,
mutation, or thresholds) so it is pre-authorized without a scientific decision.

Launches EXACTLY the frozen command:
    python experiments/strategic_demand_searcher.py --mutate-from-screen --generations 6 --pop 8

Detects true completion via summary.json (only written once, at the very end
of a full run). A restart after a crash is scientifically inert: the RNG seed
is fixed (2410001) and anchor_B_cheap.json already exists, so regenerated
candidates are byte-identical to what a continuous run would have produced --
confirmed by direct comparison of SDS_INIT_0 across the two prior attempts.

Run:  python scripts/watchdog_searcher_v1.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/Scripts/python.exe")
CMD = [PY, "-u", "experiments/strategic_demand_searcher.py",
       "--mutate-from-screen", "--generations", "6", "--pop", "8"]
OUT_DIR = ROOT / "artifacts/strategic_demand/searcher_mutate"
SUMMARY = OUT_DIR / "summary.json"
LOG = OUT_DIR / "watchdog.log"
STDERR_LOG = OUT_DIR / "crash_stderr.log"

MAX_RESTARTS = 8
MIN_LIFETIME_SECONDS = 30   # a death faster than this is treated as a launch failure


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    log("watchdog start; command is UNMODIFIED from the frozen launch")
    if SUMMARY.is_file():
        log(f"{SUMMARY.name} already exists -- V1 already completed, nothing to do")
        return 0

    for attempt in range(1, MAX_RESTARTS + 1):
        log(f"launch attempt {attempt}/{MAX_RESTARTS}")
        t0 = time.time()
        with open(STDERR_LOG, "a", encoding="utf-8") as ef:
            ef.write(f"\n===== attempt {attempt} at "
                    f"{datetime.now(timezone.utc).isoformat()} =====\n")
            ef.flush()
            proc = subprocess.run(CMD, cwd=str(ROOT), stdout=subprocess.DEVNULL,
                                  stderr=ef)
        lifetime = time.time() - t0
        log(f"attempt {attempt} exited rc={proc.returncode} "
            f"after {lifetime:.1f}s")

        if SUMMARY.is_file():
            log("summary.json present -- V1 completed. Watchdog stopping.")
            return 0

        if lifetime < MIN_LIFETIME_SECONDS:
            log(f"lifetime < {MIN_LIFETIME_SECONDS}s -- looks like an immediate "
               "launch failure, not a mid-run crash. Check crash_stderr.log.")
            if attempt >= 3:
                log("3 consecutive fast failures -- stopping rather than "
                   "loop uselessly. HUMAN REVIEW NEEDED: see crash_stderr.log")
                return 2

        time.sleep(5)

    log(f"exhausted {MAX_RESTARTS} restart attempts without completion. STOPPING.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
