"""Run SAC-RFT CONTROL then TREATMENT sequentially. One process tree, one GPU.

Does not re-run preflight (already frozen). Does not skip the RSCFT integrity audit.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
RUNNER = ROOT / "experiments" / "run_sac_rft_production.py"
AUDIT = ROOT / "experiments" / "verify_rscft_eval_integrity.py"


def main() -> int:
    py = str(PY if PY.is_file() else sys.executable)
    for arm in ("control", "treatment"):
        print(f"\n===== SAC-RFT {arm.upper()} =====", flush=True)
        rc = subprocess.call([py, str(RUNNER), "--arm", arm], cwd=str(ROOT))
        if rc != 0:
            print(f"REFUSING: {arm} exited {rc}", flush=True)
            print("Still launching RSCFT integrity audit (independent of SAC-RFT).", flush=True)
            audit_rc = subprocess.call([py, str(AUDIT)], cwd=str(ROOT))
            return audit_rc if audit_rc != 0 else rc
    print("\n===== RSCFT INTEGRITY AUDIT =====", flush=True)
    return subprocess.call([py, str(AUDIT)], cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
