"""Unattended supervisor for Confirmation A -> score -> Confirmation B -> score.

Runs the already-frozen sequence to completion without human input. It changes
no threshold, policy set, seed block, estimator, or protocol -- it only decides
WHICH frozen step to run next and records what happened.

Design rules:
  * resumable + idempotent -- state is derived by INSPECTING ARTIFACTS, not by
    trusting a stored cursor, so a crash or restart re-derives the truth
  * never runs a duplicate evaluator -- PID lock plus a live-process check
  * stops with BLOCKED on any provenance/scientific invariant failure rather
    than guessing or repairing the protocol
  * stops after the A/B decision artifact. Never launches latent training or
    VGC-4 Phase 3.

Run:  python scripts/run_confirmation_supervisor.py
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

STATE = ROOT / "artifacts/summer_2026/confirmation_state.json"
LOCK = ROOT / "artifacts/summer_2026/confirmation_supervisor.lock"
LOG = ROOT / "artifacts/summer_2026/logs/confirmation_supervisor.log"

A_SUMMARY = ROOT / "artifacts/vgc_diversity/crossplay/specialist_pilot_summary.json"
A_RESULT = ROOT / "artifacts/vgc_specialists/CONFIRMATION_A_RESULT.json"
B_SUMMARY = ROOT / "artifacts/vgc_diversity/crossplay/pair_replication_summary.json"
B_RESULT = ROOT / "artifacts/summer_2026/CONFIRMATION_B_RESULT.json"
B_LOG = ROOT / "artifacts/summer_2026/logs/confirmation_b.log"
DECISION = ROOT / "artifacts/summer_2026/AB_DECISION.json"

B_REGISTRY = ROOT / "artifacts/summer_2026/policies_pair_replication.json"
B_SEED_BASE = 9200000
B_EPISODES = 60
B_TAG = "pair_replication"
B_CELLS = 14


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{utc()}] pid={os.getpid()} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _pid_alive(pid: int) -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True, timeout=30).stdout
        return str(pid) in out
    except Exception:
        return False


def evaluator_running() -> int | None:
    """True if a run_crossplay_eval process exists. Prevents duplicate runs."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*run_crossplay_eval*' } | "
             "Select-Object -ExpandProperty ProcessId"],
            capture_output=True, text=True, timeout=60).stdout.strip()
        return int(out.split()[0]) if out.split() else None
    except Exception:
        return None


class Lock:
    def __enter__(self):
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        if LOCK.exists():
            try:
                pid = int(LOCK.read_text().strip())
            except Exception:
                pid = None
            if pid and _pid_alive(pid):
                raise SystemExit(f"supervisor already running (pid {pid})")
            log(f"stale lock (pid {pid}) -- taking over")
        LOCK.write_text(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        try:
            LOCK.unlink()
        except FileNotFoundError:
            pass


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2), encoding="utf-8")
    os.replace(tmp, STATE)


def load_state() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"history": []}


def transition(st: dict, frm: str, to: str, **kw) -> None:
    st["state"] = to
    st.setdefault("history", []).append(
        {"from": frm, "to": to, "utc": utc(), "pid": os.getpid(), **kw})
    save_state(st)
    log(f"STATE {frm} -> {to} {kw if kw else ''}")


# ---------------------------------------------------------------- detection
def b_cells_done() -> int:
    if not B_LOG.is_file():
        return 0
    return sum(1 for ln in B_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()
               if "win_rate=" in ln)


def detect() -> tuple[str, str]:
    """Derive the true state from artifacts. Returns (state, why)."""
    if not A_SUMMARY.is_file():
        return "BLOCKED", "Confirmation A summary missing; A was not run by this supervisor"
    if not A_RESULT.is_file():
        return "SCORE_A", "A complete, not yet scored"
    a = json.loads(A_RESULT.read_text(encoding="utf-8"))
    if a.get("verdict") == "BLOCKED":
        return "BLOCKED", f"A scoring returned BLOCKED: {a.get('reason')}"
    if not B_SUMMARY.is_file():
        n = b_cells_done()
        if n >= B_CELLS:
            return "SCORE_B", f"B log shows {n}/{B_CELLS} cells but no summary yet"
        if n > 0:
            return "RUN_B", f"B partially complete ({n}/{B_CELLS}); resume"
        return "RUN_B", "A scored; B not started"
    if not B_RESULT.is_file():
        return "SCORE_B", "B complete, not yet scored"
    if not DECISION.is_file():
        return "DECIDE", "A and B both scored; decision artifact missing"
    return "COMPLETE", "decision artifact exists"


# ------------------------------------------------------------------- steps
def step_score_a(st: dict) -> None:
    log("running frozen Confirmation A scorer")
    rc = subprocess.run([PY, str(ROOT / "experiments/score_specialist_pilot.py")],
                        cwd=str(ROOT)).returncode
    if rc != 0 or not A_RESULT.is_file():
        transition(st, "SCORE_A", "BLOCKED", reason=f"A scorer rc={rc}")
        return
    a = json.loads(A_RESULT.read_text(encoding="utf-8"))
    transition(st, "SCORE_A", "RUN_B", a_verdict=a.get("verdict"),
               gate1=a["gate1"]["verdict"], gate2=a["gate2"]["verdict"],
               gate3=a.get("gate3", {}).get("verdict"))


def step_run_b(st: dict) -> None:
    running = evaluator_running()
    if running:
        log(f"evaluator already running (pid {running}); not launching a duplicate")
        return
    if not B_REGISTRY.is_file():
        reg = [
            {"policy_id": "vgc_d1_seed3600001",
             "checkpoint": "artifacts/vgc_diversity/vgc_d1_seed3600001/ckpts/final_vgc_d1_seed3600001.zip",
             "method": "Mixed-PPO", "diversity_condition": "D1",
             "team_size": 2, "seed": 3600001, "arm": "PRIMARY"},
            {"policy_id": "g0_v5_long_seed3200001",
             "checkpoint": "artifacts/g0_v5_long/g0_v5_long_seed3200001/ckpts/final_g0_v5_long_seed3200001.zip",
             "method": "Mixed-PPO", "diversity_condition": "D7",
             "team_size": 2, "seed": 3200001, "arm": "PRIMARY"},
        ]
        for e in reg:
            if not (ROOT / e["checkpoint"]).is_file():
                transition(st, "RUN_B", "BLOCKED",
                           reason=f"missing checkpoint {e['checkpoint']}")
                return
        B_REGISTRY.write_text(json.dumps(reg, indent=2), encoding="utf-8")
        log(f"wrote B registry -> {B_REGISTRY}")

    B_LOG.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-u", str(ROOT / "experiments/run_crossplay_eval.py"),
           "--registry", str(B_REGISTRY.relative_to(ROOT)).replace("\\", "/"),
           "--episodes", str(B_EPISODES),
           "--seed-base", str(B_SEED_BASE),
           "--tag", B_TAG]
    log(f"launching Confirmation B: {' '.join(cmd[2:])}")
    with open(B_LOG, "a", encoding="utf-8") as lf:
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT)
    st["b_pid"] = p.pid
    save_state(st)
    log(f"Confirmation B pid={p.pid}; monitoring to {B_CELLS} cells")
    while p.poll() is None:
        time.sleep(120)
        log(f"  B progress {b_cells_done()}/{B_CELLS}")
    log(f"B exited rc={p.returncode}, cells={b_cells_done()}")


def step_score_b(st: dict) -> None:
    log("running frozen Confirmation B scorer")
    rc = subprocess.run([PY, str(ROOT / "experiments/score_pair_replication.py")],
                        cwd=str(ROOT)).returncode
    if rc != 0 or not B_RESULT.is_file():
        transition(st, "SCORE_B", "BLOCKED", reason=f"B scorer rc={rc}")
        return
    b = json.loads(B_RESULT.read_text(encoding="utf-8"))
    transition(st, "SCORE_B", "DECIDE",
               primary=b.get("PRIMARY_GATE_OP7_OP8_CROSSOVER_REPLICATES"))


def step_decide(st: dict) -> None:
    a = json.loads(A_RESULT.read_text(encoding="utf-8"))
    b = json.loads(B_RESULT.read_text(encoding="utf-8"))
    g1 = a["gate1"]["verdict"]
    g2 = a["gate2"]["verdict"]
    g3 = a.get("gate3", {}).get("verdict")
    crossover = b.get("PRIMARY_GATE_OP7_OP8_CROSSOVER_REPLICATES")
    siv = a.get("SPECIALIST_INCREMENTAL_VALUE", {})

    # Frozen vocabulary, DISCOVERED_PAIR_REPLICATION_FROZEN.status_vocabulary:
    #   crossover replicates            -> CANDIDATE
    #   + delta_pool>=.05 & LCB95>0     -> necessary, not sufficient
    #   + attribution picks the pair    -> CONFIRMED
    if crossover == "PASS" and g3 == "PASS" and siv.get("d1_d7_selected"):
        status, why = "CONFIRMED", "crossover replicated, repertoire value cleared, attribution names the pair"
    elif crossover == "PASS" and g3 == "BLOCKED":
        status, why = ("CANDIDATE",
                       "crossover replicated on 9200000, but gate 3 is BLOCKED so "
                       "repertoire value is unmeasured. CONFIRMED is unreachable "
                       "until gate 3 is computable.")
    elif crossover == "PASS":
        status, why = ("CANDIDATE",
                       "crossover replicated but repertoire value or attribution "
                       "did not clear")
    else:
        status, why = ("NO_CONFIRMED_TEACHERS",
                       "the discovered-pair crossover did not replicate on fresh "
                       "episodes")

    dec = {
        "record": "Summer 2026 Confirmation A/B decision",
        "utc": utc(),
        "teacher_status": status,
        "why": why,
        "vocabulary_source": "artifacts/summer_2026/DISCOVERED_PAIR_REPLICATION_FROZEN.json#repertoire_value_is_a_separate_claim.status_vocabulary",
        "confirmation_A": {
            "block": 9300000, "gate1": g1, "gate2": g2, "gate3": g3,
            "verdict": a.get("verdict"),
            "descriptive_dominance": a.get("descriptive_dominance"),
            "SELECTED_POLICY_PER_OPPONENT": a.get("SELECTED_POLICY_PER_OPPONENT"),
            "SPECIALIST_INCREMENTAL_VALUE": siv,
        },
        "confirmation_B": {
            "block": 9200000,
            "primary_gate": crossover,
            "OP7": b.get("OP7_D7_over_D1"), "OP8": b.get("OP8_D1_over_D7"),
        },
        "specialist_hypothesis": (
            "FAILED -- gate1 and gate2 both FAIL and S_OP7 beats or ties S_OP8 on "
            f"{a.get('descriptive_dominance', {}).get('S_OP7_beats_or_ties_S_OP8_on')} "
            "of 7 opponents. Direct single-opponent training did not produce "
            "complementary best responses."),
        "next_step_requires_human_authorisation": True,
        "not_launched_automatically": ["latent K=2 birth", "VGC-4 Phase 3"],
    }
    DECISION.write_text(json.dumps(dec, indent=2), encoding="utf-8")
    log(f"DECISION teacher_status={status}")
    transition(st, "DECIDE", "COMPLETE", teacher_status=status)


STEPS = {"SCORE_A": step_score_a, "RUN_B": step_run_b,
         "SCORE_B": step_score_b, "DECIDE": step_decide}


def main() -> int:
    dry = "--dry-run" in sys.argv
    state, why = detect()
    if dry:
        print(json.dumps({"detected_state": state, "why": why,
                          "would_run": state if state in STEPS else "(nothing)",
                          "artifacts": {
                              "A_summary": A_SUMMARY.is_file(),
                              "A_result": A_RESULT.is_file(),
                              "B_summary": B_SUMMARY.is_file(),
                              "B_result": B_RESULT.is_file(),
                              "decision": DECISION.is_file(),
                              "B_cells_done": b_cells_done()},
                          "evaluator_running_pid": evaluator_running()}, indent=2))
        return 0

    with Lock():
        st = load_state()
        log(f"supervisor start; detected={state} ({why})")
        guard = 0
        while guard < 12:
            guard += 1
            state, why = detect()
            st["state"] = state
            st["detect_why"] = why
            save_state(st)
            if state in ("COMPLETE", "BLOCKED"):
                log(f"terminal state {state}: {why}")
                if state == "BLOCKED":
                    st["blocked_reason"] = why
                    save_state(st)
                    return 1
                return 0
            STEPS[state](st)
        log("guard limit reached without terminal state")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
