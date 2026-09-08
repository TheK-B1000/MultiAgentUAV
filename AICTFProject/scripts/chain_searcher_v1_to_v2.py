"""Chain V1 (via its watchdog) to V2, authorized ONLY if V1 exhausts clean.

Waits for scripts/watchdog_searcher_v1.py to reach a terminal state:
    - summary.json appears        -> V1 completed all 6 generations
    - watchdog process exits      -> either completed, or gave up after
                                     MAX_RESTARTS / 3 fast failures

Then, and only then:
    - if any archived row has development_eligible=True: STOP, write
      HUMAN_DECISION_REQUIRED. Do NOT launch V2 -- a candidate exists and
      freezing it (or not) is a human call, not an automated one.
    - if V1 exhausted with no eligible candidate: launch Searcher V2
      unattended, per STRATEGIC_DEMAND_SEARCHER_V2_FROZEN.json.
    - if V1's watchdog gave up without ever completing (infrastructure
      failure, not a scientific outcome): STOP, write HUMAN_DECISION_REQUIRED.
      V2 launching to paper over an unexplained infrastructure failure would
      not be a scientific decision.

Never launches V2 while V1 is still attempting to complete.

Run:  python scripts/chain_searcher_v1_to_v2.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/Scripts/python.exe")
V1_DIR = ROOT / "artifacts/strategic_demand/searcher_mutate"
V1_SUMMARY = V1_DIR / "summary.json"
V1_WATCHDOG_LOG = V1_DIR / "watchdog.log"
V2_CMD = [PY, "-u", "experiments/strategic_demand_searcher_v2.py", "--device", "cuda"]
SD = ROOT / "artifacts/strategic_demand"
FINAL = SD / "HUMAN_DECISION_REQUIRED_SEARCHER_CHAIN.md"

POLL_SECONDS = 120
MAX_WAIT_HOURS = 20


def log(msg: str) -> None:
    p = SD / "chain_v1_to_v2.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}"
    print(line, flush=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def watchdog_alive() -> bool:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*watchdog_searcher_v1*' }).Count"],
            capture_output=True, text=True, timeout=60).stdout.strip()
        return int(out or 0) > 0
    except Exception:
        return True   # fail safe: assume alive rather than prematurely act


def main() -> int:
    log("chain start: waiting for Searcher V1 (via watchdog) to reach a terminal state")
    t0 = time.time()
    while True:
        if V1_SUMMARY.is_file():
            log("V1 summary.json present -- V1 completed")
            break
        if not watchdog_alive():
            log("watchdog process no longer running and no summary.json -- "
               "V1 did not complete (gave up after restart budget)")
            break
        if (time.time() - t0) / 3600.0 > MAX_WAIT_HOURS:
            log(f"waited {MAX_WAIT_HOURS}h with no terminal state -- stopping to "
               "avoid an unbounded silent wait")
            FINAL.write_text(
                "# HUMAN_DECISION_REQUIRED — Searcher V1 wait exceeded budget\n\n"
                f"Waited {MAX_WAIT_HOURS}h. V1 neither completed nor visibly "
                "died. Check process state manually.\n", encoding="utf-8")
            return 1
        time.sleep(POLL_SECONDS)

    v1_completed = V1_SUMMARY.is_file()
    v1_summary = json.loads(V1_SUMMARY.read_text(encoding="utf-8")) if v1_completed else None
    v1_archive = json.loads((V1_DIR / "archive.json").read_text(encoding="utf-8")) if \
        (V1_DIR / "archive.json").is_file() else {"rows": []}
    eligible = [r for r in v1_archive.get("rows", []) if r.get("development_eligible")]

    if eligible:
        log(f"V1 found {len(eligible)} development-eligible row(s) -- "
           "STOPPING. V2 is not authorized when a candidate already exists.")
        best = max(eligible, key=lambda r: r.get("J", -9))
        FINAL.write_text(f"""# HUMAN_DECISION_REQUIRED — Searcher V1 found a candidate

V1 completed with a development-eligible candidate. V2 was NOT launched --
the chain protocol only authorizes V2 when V1 exhausts WITHOUT a candidate.

## Best candidate

```json
{json.dumps(best, indent=2)}
```

## Next step

Human decides whether to freeze this genome and spend the untouched
confirmation block 2500001. This is not automated.
""", encoding="utf-8")
        return 0

    if not v1_completed:
        log("V1 did not complete (watchdog exhausted its restart budget without "
           "reaching summary.json). This is an infrastructure failure, not a "
           "scientific outcome -- STOPPING rather than launching V2 to paper "
           "over it.")
        FINAL.write_text(f"""# HUMAN_DECISION_REQUIRED — Searcher V1 did not complete

The watchdog exhausted its restart budget without V1 ever finishing. See
{V1_WATCHDOG_LOG} and artifacts/strategic_demand/searcher_mutate/crash_stderr.log.

V2 was NOT launched. Launching V2 to route around an unexplained repeated
crash would substitute automation for diagnosis. Archive so far has
{len(v1_archive.get('rows', []))} rows, 0 development-eligible.

## Next step

Diagnose the crash (crash_stderr.log should now have a real traceback, since
the watchdog captures stderr for the first time). Once understood, either
retry V1 or authorize V2 manually.
""", encoding="utf-8")
        return 1

    log(f"V1 exhausted its {v1_summary.get('n_archive', '?')}-row search with "
       "NO development-eligible candidate. Launching Searcher V2, per "
       "STRATEGIC_DEMAND_SEARCHER_V2_FROZEN.json.")
    log(f"launching: {' '.join(V2_CMD[2:])}")
    v2_log = SD / "searcher_v2" / "v2_launch.log"
    v2_log.parent.mkdir(parents=True, exist_ok=True)
    with open(v2_log, "w", encoding="utf-8") as lf:
        subprocess.Popen(V2_CMD, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT)
    log("V2 launched. Chain supervisor exiting; V2 manages its own completion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
