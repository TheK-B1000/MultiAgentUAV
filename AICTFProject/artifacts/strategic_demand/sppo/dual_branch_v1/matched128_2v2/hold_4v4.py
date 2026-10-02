"""Hold the 4v4 dual-branch suite while POSTHOC_MATCHED128_2V2_DUAL_BRANCH runs (user, 2026-10-02).

    python hold_4v4.py suspend   # freeze every python process running 4v4/run_dual_branch_4v4.py
    python hold_4v4.py resume    # thaw exactly the processes this script froze (pids in hold_4v4.json)

Suspension only pauses the process (its waiting loop); nothing is killed and no state is written,
so on resume the driver continues exactly where it was. Windows only (NtSuspendProcess/NtResumeProcess).
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REC = HERE / "hold_4v4.json"
PROCESS_SUSPEND_RESUME = 0x0800


def suite_pids() -> list[int]:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object { $_.CommandLine -match 'run_dual_branch_4v4\\.py' } | ForEach-Object { $_.ProcessId }"],
        capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def _call(pid: int, fn: str) -> bool:
    k32, nt = ctypes.windll.kernel32, ctypes.windll.ntdll
    h = k32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
    if not h:
        return False
    try:
        return getattr(nt, fn)(h) == 0
    finally:
        k32.CloseHandle(h)


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if cmd == "suspend":
        if REC.is_file() and json.loads(REC.read_text(encoding="utf-8")).get("state") == "SUSPENDED":
            print("already suspended:", REC.read_text(encoding="utf-8"))
            return 0
        pids = suite_pids()
        done = [p for p in pids if _call(p, "NtSuspendProcess")]
        REC.write_text(json.dumps({"state": "SUSPENDED", "utc": now, "pids": done, "found": pids}, indent=2) + "\n",
                       encoding="utf-8")
        print(f"suspended {done} (found {pids})")
        return 0 if done == pids else 1
    if cmd == "resume":
        if not REC.is_file():
            print("nothing to resume")
            return 0
        rec = json.loads(REC.read_text(encoding="utf-8"))
        if rec.get("state") != "SUSPENDED":
            print("not suspended:", rec.get("state"))
            return 0
        done = [p for p in rec["pids"] if _call(p, "NtResumeProcess")]
        REC.write_text(json.dumps({**rec, "state": "RESUMED", "resumed_utc": now, "resumed": done}, indent=2) + "\n",
                       encoding="utf-8")
        print(f"resumed {done}")
        return 0
    raise SystemExit("usage: hold_4v4.py suspend|resume")


if __name__ == "__main__":
    raise SystemExit(main())
