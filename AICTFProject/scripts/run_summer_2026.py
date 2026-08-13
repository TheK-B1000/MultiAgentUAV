"""Summer 2026 unattended state-machine supervisor.

This is an ORCHESTRATION WRAPPER. It calls existing runners; it does not
reimplement training, evaluation, sampling, or gate logic. Every state
transition delegates to a script that already exists and was already verified
independently in this session:

    D3 preflight     experiments/run_d3_pool_preflight.py
    D1/D3 training   experiments/run_vgc_diversity.py
    cross-play eval  experiments/run_crossplay_eval.py
    FP smoke         experiments/run_fp_smoke.py (9-criterion gate; uses SNAPSHOT pool)

State is persisted atomically (write-to-temp + os.replace) so a restart after
a crash or power loss resumes from the last COMPLETED state rather than
rerunning it. A PID lock file prevents two supervisors from launching the same
experiment concurrently.

This script does not implement Phases 5-11 (crossover / oracle / selector /
latent) because those require scientific decisions (which specialists to
train, what threshold defines a meaningful crossover) that must be frozen in
their own preregistration artifacts BEFORE code exists to test them -- exactly
the discipline the rest of this project has followed. It stops cleanly at the
first state that would require inventing such a decision on the fly.

Run:  python scripts/run_summer_2026.py [--once]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATE_PATH = ROOT / "artifacts/summer_2026/state.json"
LOCK_PATH = ROOT / "artifacts/summer_2026/supervisor.lock"
PY = str(ROOT / ".venv/Scripts/python.exe")

STATES = (
    "AUDIT", "PREFLIGHT", "FP_SMOKE", "DIVERSITY_TRAIN", "DIVERSITY_EVAL",
    "DEMAND_ANALYSIS", "COMPLETE", "STOPPED_SCIENTIFIC_GATE", "STOPPED_ERROR",
)
# FP_SMOKE sits here (before the ~5h D3 campaign), not later where the new
# spec's decision tree places the full FP population loop (Phase 8). That
# placement was an explicit decision earlier this session: close FP's
# infrastructure uncertainty (BUILD COMPLETE blocker) while the GPU is idle,
# rather than contend with three D3 trainers for it. The scientific FP
# population loop remains gated behind DEMAND_ANALYSIS like every other
# post-diversity phase; only the mechanism SMOKE TEST is pulled forward.


class Lock:
    """PID-file lock. Refuses to run if another live supervisor holds it."""

    def __enter__(self):
        if LOCK_PATH.exists():
            try:
                pid = int(LOCK_PATH.read_text(encoding="utf-8").strip())
            except ValueError:
                pid = None
            if pid is not None and _pid_alive(pid):
                raise SystemExit(f"supervisor already running (pid {pid}); refusing to start a second one")
            print(f"stale lock (pid {pid}) -- previous supervisor did not exit cleanly; taking over")
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")
        return self

    def __exit__(self, *exc):
        try:
            LOCK_PATH.unlink()
        except FileNotFoundError:
            pass


def _pid_alive(pid: int) -> bool:
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue) -ne $null"],
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip().lower() == "true"
    except Exception:
        return True  # fail closed: assume alive rather than risk a double-launch


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"state": "AUDIT", "history": [], "gates": {}}


def save_state(st: dict) -> None:
    """Atomic write: temp file + os.replace, so a crash mid-write cannot
    leave a truncated/corrupt state.json that a restart would misread."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def transition(st: dict, new_state: str, **fields) -> dict:
    st["history"].append({"from": st["state"], "to": new_state,
                          "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields})
    st["state"] = new_state
    save_state(st)
    return st


def run(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
    return r.returncode


def step_audit(st: dict) -> dict:
    """AUDIT is a precondition, not an action: SUMMER_2026_AUDIT.md must
    already exist (Phase 0 is read-only and was run before this script)."""
    audit = ROOT / "SUMMER_2026_AUDIT.md"
    if not audit.exists():
        raise SystemExit("SUMMER_2026_AUDIT.md missing; run the Phase 0 audit before the supervisor")
    return transition(st, "PREFLIGHT")


def step_preflight(st: dict) -> dict:
    """MIXED_SAMPLING_PASS, in this project's existing vocabulary: D3_POOL_PREFLIGHT."""
    out = ROOT / "artifacts/vgc_diversity/D3_POOL_PREFLIGHT_RESULT.json"
    if out.exists():
        result = json.loads(out.read_text(encoding="utf-8"))
    else:
        rc = run([PY, "-u", "experiments/run_d3_pool_preflight.py"],
                 ROOT / "artifacts/summer_2026/logs/preflight.log")
        if rc != 0 or not out.exists():
            st["gates"]["MIXED_SAMPLING_PASS"] = "FAIL"
            return transition(st, "STOPPED_ERROR", reason="D3_POOL_PREFLIGHT did not produce a result",
                              rc=rc)
        result = json.loads(out.read_text(encoding="utf-8"))
    st["gates"]["MIXED_SAMPLING_PASS"] = result["verdict"]
    save_state(st)
    if not result["verdict"].endswith("PASS"):
        return transition(st, "STOPPED_ERROR", reason="D3_POOL_PREFLIGHT_FAIL; sampler needs repair")
    return transition(st, "FP_SMOKE")


def step_fp_smoke(st: dict) -> dict:
    """Mechanism-only smoke: loadability + two-checkpoint SNAPSHOT rotation.

    Runs experiments/run_fp_smoke.py against FP_SMOKE_FROZEN.json (before the
    D3 GPU commitment). Establishes mechanism only — not that FP helps.
    """
    out = ROOT / "artifacts/vgc_fp/FP_SMOKE_RESULT.json"
    if out.exists():
        result = json.loads(out.read_text(encoding="utf-8"))
    else:
        rc = run([PY, "-u", "experiments/run_fp_smoke.py"],
                 ROOT / "artifacts/summer_2026/logs/fp_smoke.log")
        if rc != 0 or not out.exists():
            st["gates"]["FP_SMOKE"] = "FAIL"
            return transition(st, "STOPPED_ERROR",
                              reason="FP_SMOKE did not produce a result", rc=rc)
        result = json.loads(out.read_text(encoding="utf-8"))
    st["gates"]["FP_SMOKE"] = result["verdict"]
    save_state(st)
    if not result["verdict"].endswith("PASS"):
        return transition(st, "STOPPED_ERROR", reason="FP_SMOKE_FAIL; repair SNAPSHOT pool seam")
    return transition(st, "DIVERSITY_TRAIN")


def _seed_complete(tag: str) -> bool:
    ck = ROOT / "artifacts/vgc_diversity" / tag / "ckpts" / f"final_{tag}.zip"
    return ck.is_file()


def step_diversity_train(st: dict) -> dict:
    """Launch only missing D3 cells. D1/D7 are already COMPLETE per the manifest."""
    seeds = (3700001, 3700002, 3700003)
    missing = [s for s in seeds if not _seed_complete(f"vgc_d3_seed{s}")]
    if not missing:
        return transition(st, "DIVERSITY_EVAL")

    # GPU-serialized: one seed at a time. D1 previously ran concurrent seeds
    # successfully, but the Summer 2026 unattended supervisor prefers serial
    # training to avoid OOM / driver contention on a single GPU overnight.
    print(f"launching D3 seeds (serialized): {missing}")
    st["diversity_train_pids"] = {}
    save_state(st)
    for s in missing:
        log = ROOT / f"artifacts/vgc_diversity/d3_seed{s}.log"
        print(f"D3 seed {s}: START", flush=True)
        with open(log, "w", encoding="utf-8") as f:
            p = subprocess.Popen([PY, "-u", "experiments/run_vgc_diversity.py",
                                  "--condition", "D3", "--seed", str(s), "--threads", "4"],
                                 cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
            st["diversity_train_pids"][str(s)] = p.pid
            save_state(st)
            rc = p.wait()
        if rc != 0 or not _seed_complete(f"vgc_d3_seed{s}"):
            return transition(st, "STOPPED_ERROR", reason=f"D3 seed {s} failed", rc=rc)
        print(f"D3 seed {s}: COMPLETE", flush=True)
    return transition(st, "DIVERSITY_EVAL")


def step_diversity_eval(st: dict) -> dict:
    out = ROOT / "artifacts/vgc_diversity/crossplay/d1_d3_d7_summary.json"
    if not out.exists():
        reg = json.loads((ROOT / "artifacts/vgc_diversity/policies_primary.json").read_text())
        for s in (3700001, 3700002, 3700003):
            t = f"vgc_d3_seed{s}"
            reg.append({"policy_id": t,
                       "checkpoint": f"artifacts/vgc_diversity/{t}/ckpts/final_{t}.zip",
                       "method": "Mixed-PPO", "diversity_condition": "D3",
                       "team_size": 2, "seed": s, "arm": "PRIMARY"})
        reg_path = ROOT / "artifacts/summer_2026/policies_d1_d3_d7.json"
        reg_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")
        rc = run([PY, "-u", "experiments/run_crossplay_eval.py",
                 "--registry", str(reg_path.relative_to(ROOT)),
                 "--episodes", "30", "--tag", "d1_d3_d7"],
                 ROOT / "artifacts/summer_2026/logs/eval_d1_d3_d7.log")
        if rc != 0 or not out.exists():
            return transition(st, "STOPPED_ERROR", reason="cross-play evaluation failed", rc=rc)
    return transition(st, "DEMAND_ANALYSIS")


def step_demand_analysis(st: dict) -> dict:
    """This is a genuine scientific-decision boundary, not an engineering one.

    Phase 5 (crossover) needs >=2 differently-trained single-opponent
    specialists (S_OP7, S_OP12) that do not exist yet. Training them is both a
    real GPU commitment and a scope decision this script must not make
    unilaterally -- doing so would be exactly the kind of automatic response
    to a gate result ("just train more things") the project's failure-handling
    rules prohibit.
    """
    st["gates"]["CROSSOVER_FOUND"] = "NOT_TESTED"
    st["gates"]["MATCHUP_DEPENDENCE"] = "NOT_TESTED"
    return transition(st, "STOPPED_SCIENTIFIC_GATE",
                      reason="D1/D3/D7 diversity-scaling data exists and is evaluated. "
                             "Phase 5 (crossover) requires S_OP7/S_OP12 specialists that "
                             "have not been trained. This is a scope decision, not a failure.")


STEP_FNS = {
    "AUDIT": step_audit, "PREFLIGHT": step_preflight, "FP_SMOKE": step_fp_smoke,
    "DIVERSITY_TRAIN": step_diversity_train, "DIVERSITY_EVAL": step_diversity_eval,
    "DEMAND_ANALYSIS": step_demand_analysis,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="run a single state transition and exit")
    args = ap.parse_args()

    with Lock():
        st = load_state()
        print(f"resuming from state: {st['state']}")
        while st["state"] not in ("COMPLETE", "STOPPED_SCIENTIFIC_GATE", "STOPPED_ERROR"):
            fn = STEP_FNS.get(st["state"])
            if fn is None:
                return transition(st, "STOPPED_ERROR", reason=f"unknown state {st['state']}") and 1
            print(f"--- {st['state']} ---", flush=True)
            st = fn(st)
            print(f"-> {st['state']}", flush=True)
            if args.once:
                break

        print(f"\nFINAL STATE: {st['state']}")
        if st["state"] == "STOPPED_SCIENTIFIC_GATE":
            print(f"  reason: {st['history'][-1].get('reason')}")
        return 0 if st["state"] != "STOPPED_ERROR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
