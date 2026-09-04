"""After SAC-RFT CONTROL+TREATMENT leave the GPU, run the RSCFT integrity audit.

The in-flight training chain loaded run_sac_rft_both_arms.py before the audit hook existed,
so this watcher is what fires for the current run.
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
PY = ROOT / ".venv" / "Scripts" / "python.exe"
AUDIT = ROOT / "experiments" / "verify_rscft_eval_integrity.py"
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
OUT = SD / "RSCFT_EVAL_INTEGRITY.json"
MARK = SD / "RSCFT_AUDIT_WATCHER.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _training_pids() -> list[int]:
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object { $_.CommandLine -match 'run_sac_rft_production|run_sac_rft_both_arms' } | "
         "ForEach-Object { $_.ProcessId }"],
        capture_output=True, text=True)
    me = os.getpid()
    return [int(x) for x in r.stdout.split() if x.strip().isdigit() and int(x) != me]


def main() -> int:
    py = str(PY if PY.is_file() else sys.executable)
    MARK.write_text(json.dumps({
        "record": "RSCFT audit watcher",
        "status": "WAITING_FOR_SAC_RFT_TRAINING",
        "utc": _now(),
        "note": "Launches verify_rscft_eval_integrity.py when run_sac_rft_* exits.",
    }, indent=2), encoding="utf-8")
    print(f"WATCHING SAC-RFT training; audit after exit  {_now()}", flush=True)
    while True:
        pids = _training_pids()
        if not pids:
            break
        print(f"  [{_now()}] still training pids={pids}", flush=True)
        time.sleep(60)

    if OUT.is_file():
        print(f"AUDIT already present -> {OUT}", flush=True)
        return 0
    print(f"TRAINING GONE -- launching RSCFT integrity audit  {_now()}", flush=True)
    rc = subprocess.call([py, str(AUDIT)], cwd=str(ROOT))
    MARK.write_text(json.dumps({
        "record": "RSCFT audit watcher",
        "status": "AUDIT_LAUNCHED",
        "utc": _now(),
        "audit_exit": rc,
        "audit_artifact": str(OUT.relative_to(ROOT)) if OUT.is_file() else None,
    }, indent=2), encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
