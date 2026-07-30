import json, subprocess, time
from pathlib import Path
from datetime import datetime, timezone
ROOT = Path(r"K:\MultiAgentUAV\AICTFProject")
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOG = ROOT / "artifacts" / "k2v3_300k_replication" / "post_audit_launch.log"
LOG.parent.mkdir(parents=True, exist_ok=True)

def audit_running():
    try:
        import psutil
    except ImportError:
        return False
    for p in psutil.process_iter(["cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
        except Exception:
            continue
        if "audit_k2_specialist_behavior.py" in cmd:
            return True
    return False

def say(msg):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

say("waiting for discovery audit to finish before Rev4 --force-launch")
while audit_running():
    say("audit still running; sleep 120s")
    time.sleep(120)
say("audit clear; launching k2v3 12x300k with --force-launch")
rc = subprocess.call(
    [PY, "experiments/launch_k2v3_300k_replication.py", "--force-launch", "--concurrency", "2"],
    cwd=str(ROOT),
)
say(f"launch returned rc={rc}")
raise SystemExit(rc)
