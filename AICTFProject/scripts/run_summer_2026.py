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
    "AUDIT", "PREFLIGHT", "DIVERSITY_TRAIN", "DIVERSITY_EVAL",
    "DEMAND_ANALYSIS", "COMPLETE", "STOPPED_SCIENTIFIC_GATE", "STOPPED_ERROR",
    # FP_SMOKE remains a reachable state only for resume/compat; it does NOT
    # sit on the D3 critical path (see step_preflight / step_fp_smoke).
    "FP_SMOKE",
)
# Critical path: PREFLIGHT → DIVERSITY_TRAIN → … 
# FP full-cycle smoke is DEFERRED. FP_PROBE_2026-08-13 showed SNAPSHOT:<path>
# provenance works but stalled at 4096/6144 without an ablation report — that
# is NOT FP_SMOKE=PASS. D3 must not wait on an unverified FP orchestration.


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
    # Record FP probe honesty so a later handoff cannot invent FP_SMOKE=PASS.
    probe = ROOT / "artifacts/vgc_fp/FP_PROBE_2026-08-13.json"
    if probe.exists():
        board = json.loads(probe.read_text(encoding="utf-8")).get("status_board", {})
        st["gates"].update({
            "FP_SNAPSHOT_FORMAT": board.get("FP_SNAPSHOT_FORMAT", "PASS"),
            "PPO_AS_OPPONENT_SEAM": board.get("PPO_AS_OPPONENT_SEAM", "OBSERVED"),
            "FP_FULL_SMOKE": board.get("FP_FULL_SMOKE", "INCOMPLETE"),
            "FP_ABLATION_REPORT": board.get("FP_ABLATION_REPORT", "NOT PRODUCED"),
            "FP_SCIENTIFIC_GATE": board.get("FP_SCIENTIFIC_GATE", "NOT PASSED"),
            "FP_SMOKE": "INCOMPLETE_PROBE_ONLY_NOT_PASS",
        })
    save_state(st)
    if not result["verdict"].endswith("PASS"):
        return transition(st, "STOPPED_ERROR", reason="D3_POOL_PREFLIGHT_FAIL; sampler needs repair")
    return transition(st, "DIVERSITY_TRAIN")


def step_fp_smoke(st: dict) -> dict:
    """Compat / resume only — not on the D3 critical path.

    If a prior supervisor left state at FP_SMOKE, skip the long smoke and
    proceed to D3. A clean FP_SMOKE must be an explicit later action; the
    2026-08-13 probe is incomplete and must not be treated as PASS.
    """
    st["gates"]["FP_SMOKE"] = "DEFERRED_INCOMPLETE"
    st["gates"]["FP_SCIENTIFIC_GATE"] = "NOT PASSED"
    save_state(st)
    return transition(st, "DIVERSITY_TRAIN",
                      reason="FP full smoke deferred; D3 is the critical path")

# Frozen 2026-08-14: 1 seed to discover; replicate only important findings.
# See artifacts/summer_2026/SEED_PROTOCOL_FROZEN.json. D3 seed 3700003 is
# grandfathered because it had already started at freeze time — do not kill it,
# and do not schedule any further D3 seeds.
D3_TRAINING_SEEDS = (3700001, 3700002, 3700003)
NEW_STAGE_N_SEEDS = 1
EVIDENCE_LABEL = "EXPLORATORY_SINGLE_SEED"


def _seed_complete(tag: str) -> bool:
    ck = ROOT / "artifacts/vgc_diversity" / tag / "ckpts" / f"final_{tag}.zip"
    return ck.is_file()


def _completed_d3_seeds() -> list[int]:
    return [s for s in D3_TRAINING_SEEDS if _seed_complete(f"vgc_d3_seed{s}")]


def step_diversity_train(st: dict) -> dict:
    """Finish any D3 seed already in flight; do not schedule extras.

    Prospective seed protocol is one canonical seed per new experiment.
    This D3 campaign is grandfathered through 3700003 only because that job
    had already started when the protocol froze.
    """
    completed = _completed_d3_seeds()
    # Never launch a seed that is not already complete AND not already running
    # according to state (3700003 grandfather). After those finish, eval.
    running = {
        int(k): int(v) for k, v in (st.get("diversity_train_pids") or {}).items()
    }
    missing = [
        s for s in D3_TRAINING_SEEDS
        if s not in completed and s in running
    ]
    if not missing and completed:
        st["gates"]["SEED_PROTOCOL"] = "ONE_CANONICAL_UNLESS_REPLICATION"
        st["gates"]["D3_SEEDS_USED"] = completed
        st["gates"]["EVIDENCE_LABEL"] = (
            "D3_THIS_CAMPAIGN_GRANDFATHERED_UP_TO_STARTED_SEEDS"
        )
        save_state(st)
        return transition(st, "DIVERSITY_EVAL")
    if not missing:
        return transition(st, "STOPPED_ERROR", reason="no D3 seeds completed or running")

    print(f"finishing in-flight D3 seeds only (no new launches): {missing}")

    # GPU-serialized: one seed at a time. D1 previously ran concurrent seeds
    # successfully, but the Summer 2026 unattended supervisor prefers serial
    # training to avoid OOM / driver contention on a single GPU overnight.
    st.setdefault("diversity_train_pids", {})
    save_state(st)
    for s in missing:
        log = ROOT / f"artifacts/vgc_diversity/d3_seed{s}.log"
        print(f"D3 seed {s}: WAIT_IN_FLIGHT (no new launch)", flush=True)
        pid = int((st.get("diversity_train_pids") or {}).get(str(s), 0))
        if pid:
            try:
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     f"Wait-Process -Id {pid} -ErrorAction SilentlyContinue"],
                    timeout=None,
                )
            except Exception:
                pass
        # Poll until final zip or process gone without a final (failure).
        while not _seed_complete(f"vgc_d3_seed{s}"):
            if pid and not _pid_alive(pid):
                return transition(st, "STOPPED_ERROR", reason=f"D3 seed {s} exited without final ckpt")
            time.sleep(30)
        print(f"D3 seed {s}: COMPLETE", flush=True)
    st["gates"]["SEED_PROTOCOL"] = "ONE_CANONICAL_UNLESS_REPLICATION"
    st["gates"]["D3_SEEDS_USED"] = _completed_d3_seeds()
    save_state(st)
    return transition(st, "DIVERSITY_EVAL")


def step_diversity_eval(st: dict) -> dict:
    out = ROOT / "artifacts/vgc_diversity/crossplay/d1_d3_d7_summary.json"
    if not out.exists():
        reg = json.loads((ROOT / "artifacts/vgc_diversity/policies_primary.json").read_text())
        d3_seeds = _completed_d3_seeds()
        if not d3_seeds:
            return transition(st, "STOPPED_ERROR", reason="no completed D3 checkpoints for eval")
        for s in d3_seeds:
            t = f"vgc_d3_seed{s}"
            reg.append({"policy_id": t,
                       "checkpoint": f"artifacts/vgc_diversity/{t}/ckpts/final_{t}.zip",
                       "method": "Mixed-PPO", "diversity_condition": "D3",
                       "team_size": 2, "seed": s, "arm": "PRIMARY",
                       "evidence_label": "D3_COMPLETED_SEEDS_ONLY"})
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
    """Path A vs Path B readout from the completed matrix.

    Criteria frozen in PAPER_PATH_READOUT_FROZEN.json before the 8-policy
    board existed. This step classifies; it does not train specialists.
    """
    rc = run([PY, "-u", "experiments/analyze_summer_2026_paper_path.py"],
             ROOT / "artifacts/summer_2026/logs/paper_path.log")
    gate_path = ROOT / "artifacts/summer_2026/gate_results.json"
    if rc != 0 or not gate_path.exists():
        return transition(st, "STOPPED_ERROR", reason="paper-path analysis failed", rc=rc)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    st["gates"]["CROSSOVER_FOUND"] = bool(gate["CROSSOVER_FOUND"])
    st["gates"]["PAPER_PATH"] = gate["paper_path"]
    st["gates"]["SEED_PROTOCOL"] = "ONE_CANONICAL_UNLESS_REPLICATION"
    st["gates"]["NEW_STAGE_N_SEEDS"] = NEW_STAGE_N_SEEDS
    st["gates"]["EVIDENCE_LABEL"] = EVIDENCE_LABEL
    save_state(st)
    return transition(
        st, "STOPPED_SCIENTIFIC_GATE",
        reason=f"{gate['paper_path']}: {gate['paper_title']}. next={gate['next']}",
        CROSSOVER_FOUND=gate["CROSSOVER_FOUND"],
    )


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
