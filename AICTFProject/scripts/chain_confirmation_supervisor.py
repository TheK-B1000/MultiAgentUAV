"""Keep the confirmation supervisor running until it reaches a terminal state.

The supervisor is idempotent and derives its state by inspecting artifacts, so
re-invoking it is always safe. This wrapper exists because the supervisor
instance launched before the Gate 3 recovery was authorised is running older
code: it will finish Confirmation B, score it, write AB_DECISION.json and exit
COMPLETE without attempting the recovery. Re-invoking afterwards picks up the
RUN_RECOVERY -> SCORE_GATE3 -> FINAL_DECIDE states.

Behaviour:
  * if another supervisor holds the lock, wait and retry -- never run two
  * stop immediately on BLOCKED; a blocked invariant is for a human, not a retry
  * stop when the final artifact exists

Run:  python scripts/chain_confirmation_supervisor.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/Scripts/python.exe")
SUP = str(ROOT / "scripts/run_confirmation_supervisor.py")
STATE = ROOT / "artifacts/summer_2026/confirmation_state.json"
DECISION = ROOT / "artifacts/summer_2026/AB_DECISION.json"
FINAL = ROOT / "artifacts/summer_2026/AB_DECISION_FINAL.json"
REC_FROZEN = ROOT / "artifacts/vgc_specialists/CONFIRMATION_A_RECOVERY_FROZEN.json"
LOG = ROOT / "artifacts/summer_2026/logs/supervisor_chain.log"

POLL_SECONDS = 300
MAX_HOURS = 48


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = (f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] "
            f"chain pid={os.getpid()} {msg}")
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def terminal() -> str | None:
    """Return a terminal reason, or None to keep going."""
    if STATE.is_file():
        try:
            st = json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            return None
        if st.get("state") == "BLOCKED":
            return f"BLOCKED: {st.get('blocked_reason') or st.get('detect_why')}"
    # Final artifact exists -> done. If recovery was never authorised, the
    # interim decision is the end state.
    if FINAL.is_file():
        return "AB_DECISION_FINAL.json exists"
    if DECISION.is_file() and not REC_FROZEN.is_file():
        return "AB_DECISION.json exists and no recovery was authorised"
    return None


def main() -> int:
    log("chain start")
    deadline = time.time() + MAX_HOURS * 3600
    while time.time() < deadline:
        why = terminal()
        if why:
            log(f"terminal: {why}")
            return 0
        p = subprocess.run([PY, "-u", SUP], cwd=str(ROOT),
                           capture_output=True, text=True)
        tail = (p.stdout or "").strip().splitlines()[-3:]
        if "already running" in (p.stdout or "") + (p.stderr or ""):
            log(f"supervisor busy; retry in {POLL_SECONDS}s")
        else:
            log(f"supervisor invocation rc={p.returncode} :: {' | '.join(tail)}")
            why = terminal()
            if why:
                log(f"terminal: {why}")
                return 0
            if p.returncode != 0:
                log("supervisor returned non-zero and state is not terminal; "
                    "stopping for human review rather than retrying blindly")
                return 1
        time.sleep(POLL_SECONDS)
    log("deadline reached")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
