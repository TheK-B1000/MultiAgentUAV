r"""4v4 rescue ladder: pre-registered, capped, sequential (DUAL_BRANCH_4V4_RESCUE_LADDER_SPEC.json).

    python experiments/rescue_ladder_4v4.py --freeze      # write the frozen ladder + R2/R3 specs, reserve seeds
    python experiments/rescue_ladder_4v4.py --chain       # detached: R1 -> (fail) R2 -> (fail) R3 -> stop

Frozen BEFORE R1's crossover result was seen (user, 2026-10-03: "start the next step if it fails ... don't stop
until we get a pass"; bounded here so the result stays defensible):
  * rungs, order and cap are fixed now: R1 (k=1) -> R2 (k=1, DEFEND teacher halved) -> R3 (k=1, 300k budget,
    i.e. 150k teacher-free recovery). At most three attempts; the ladder STOPS after R3 whatever happens.
  * pass rule, the same for every rung, on that rung's OWN fresh 128-seed block: the lower bounds of both
    Delta_A and Delta_B are > 0 using multiplicity-adjusted paired bootstrap intervals (Bonferroni over the
    3 possible attempts: two-sided level 1 - 0.05/3 = 98.33%; seed resampling, n=20000, rng 7). 95% intervals
    are reported alongside.
  * every rung's result is reported, failures included; the passing rung (if any) is described as the outcome
    of this sequential procedure, not as a single pre-specified test.
  * symmetric always (A and B identical recipe); each rung changes ONE thing relative to R1; fresh seeds per rung.
  * a pass STOPS the ladder; Stage 4 still waits for PI review.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
LADDER = SD / "DUAL_BRANCH_4V4_RESCUE_LADDER_SPEC.json"
R1_SPEC = SD / "DUAL_BRANCH_ROLE_COMPOSITE_V1_R1_SPEC.json"
N_ATTEMPTS = 3
ALPHA_FAMILY = 0.05
N_BOOT, BOOT_SEED = 20_000, 7

V1_TEACHER = {"lambda": 0.1, "lambda_end": 0.0, "decay_start": 50_000, "decay_end": 150_000, "cadence": 4}
RUNGS = {
    "R1": {"change_vs_R1": "none (R1 itself: k = max(1, floor(N/3 + 1/2)) = 1)", "steps": 200_000, "teacher": V1_TEACHER,
           "spec": "DUAL_BRANCH_ROLE_COMPOSITE_V1_R1_SPEC.json", "label": "DUAL_BRANCH_R1_4V4_K1",
           "suffix": "_dual_branch_r1_k1", "dir": "4v4/r1", "done": "4v4/r1/R1_STAGE3_DONE.txt"},
    "R2": {"change_vs_R1": "DEFEND-teacher strength halved: lambda 0.1 -> 0.05 (same window 50k-150k, cadence 4)",
           "steps": 200_000, "teacher": {**V1_TEACHER, "lambda": 0.05},
           "spec": "DUAL_BRANCH_ROLE_COMPOSITE_V1_R2_SPEC.json", "label": "DUAL_BRANCH_R2_4V4_K1_TEACHER_HALF",
           "suffix": "_dual_branch_r2_k1_th", "dir": "4v4/r2", "done": "4v4/r2/R2_STAGE3_DONE.txt"},
    "R3": {"change_vs_R1": "budget 200k -> 300k with the teacher schedule unchanged (ends at 150k): 150k teacher-free recovery",
           "steps": 300_000, "teacher": V1_TEACHER,
           "spec": "DUAL_BRANCH_ROLE_COMPOSITE_V1_R3_SPEC.json", "label": "DUAL_BRANCH_R3_4V4_K1_300K",
           "suffix": "_dual_branch_r3_k1_300k", "dir": "4v4/r3", "done": "4v4/r3/R3_STAGE3_DONE.txt"},
}
ORDER = ("R1", "R2", "R3")
ORDER_RATIONALE = ("R2 before R3: R1's training curve (seen before its crossover) declined while the teacher was at "
                   "peak and stayed flat (0.35-0.42) for 50k steps after the teacher reached 0, so teacher strength "
                   "is the more direct next lever; longer recovery is third. (The informal pause-packet ladder listed "
                   "these in the other order; this frozen order governs.)")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def adjusted_interval(x, attempts: int = N_ATTEMPTS, alpha: float = ALPHA_FAMILY) -> dict:
    """Paired percentile bootstrap over seeds (same procedure as eval_hog_psp_v3._mean_ci) at the
    Bonferroni-adjusted two-sided level 1 - alpha/attempts."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    boot = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(axis=1)
    a = alpha / attempts
    lo, hi = np.percentile(boot, [100 * a / 2, 100 * (1 - a / 2)])
    return {"mean": float(x.mean()), "level": 1 - a, "lcb": float(lo), "ucb": float(hi)}


def rung_deltas(label: str, seeds: list[int]) -> dict:
    rows = SD / f"{label.lower()}_specialist_crossover_eval_rows.csv"
    res = SD / f"{label}_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
    if not rows.is_file() or not res.is_file() or json.loads(res.read_text(encoding="utf-8")).get("status") != "SEALED":
        raise SystemExit(f"FAIL-CLOSED: {label} has no SEALED result + rows")
    by: dict = {}
    with rows.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by.setdefault((r["policy"][-1], r["pole"]), {})[int(r["seed"])] = float(r["win"])
    for k, d in by.items():
        if sorted(d) != seeds:
            raise SystemExit(f"FAIL-CLOSED: {label} cell {k} is not exactly its fresh block")
    v = {k: np.array([d[s] for s in seeds]) for k, d in by.items()}
    dA, dB = v[("A", "A")] - v[("B", "A")], v[("B", "B")] - v[("A", "B")]
    from experiments.eval_hog_psp_v3 import _mean_ci
    return {"cells": {f"{p}@{q}": float(v[(p, q)].mean()) for p in "AB" for q in "AB"},
            "Delta_A": {**adjusted_interval(dA), "ci95": _mean_ci(dA)},
            "Delta_B": {**adjusted_interval(dB), "ci95": _mean_ci(dB)},
            "result_sha256": sha(res), "rows_sha256": sha(rows)}


def passes(d: dict) -> bool:
    return d["Delta_A"]["lcb"] > 0 and d["Delta_B"]["lcb"] > 0


# ------------------------------------------------------------------ freeze
def freeze() -> None:
    from experiments import seed_registry as SR
    if LADDER.is_file():
        raise SystemExit(f"REFUSING: {LADDER.name} exists (frozen once)")
    if (SD / f"{RUNGS['R1']['label']}_SPECIALIST_CROSSOVER_EVAL_RESULT.json").is_file():
        raise SystemExit("REFUSING: R1's result already exists -- the ladder must be frozen before it is seen")
    r1 = json.loads(R1_SPEC.read_text(encoding="utf-8"))
    reg = {b["experiment_id"]: b for b in SR.load()["blocks"]}
    seeds = {"R1": {"A": 27_500_001, "B": 27_600_001, "eval": [27_700_001, 27_700_128],
                    "eval_id": f"{RUNGS['R1']['label']}_SPECIALIST_CROSSOVER"}}
    for rung in ("R2", "R3"):
        cfg = RUNGS[rung]
        s = {}
        for side in ("A", "B"):
            eid = f"DUAL_BRANCH_{rung}_4V4_{side}_TRAIN"
            if eid not in reg:
                lo = SR.next_free(1, "exploratory")
                SR.allocate(eid, lo, lo, "exploratory", purpose=f"4v4 rescue ladder {rung} {side} training seed",
                            spec=cfg["spec"])
                reg = {b["experiment_id"]: b for b in SR.load()["blocks"]}
            s[side] = reg[eid]["lo"]
        eid = f"{cfg['label']}_SPECIALIST_CROSSOVER"
        if eid not in reg:
            lo = SR.next_free(128, "sealed_confirmatory")
            SR.allocate(eid, lo, lo + 127, "sealed_confirmatory", purpose=f"4v4 rescue ladder {rung}: fresh 128-seed crossover",
                        spec=cfg["spec"])
            reg = {b["experiment_id"]: b for b in SR.load()["blocks"]}
        s["eval"] = [reg[eid]["lo"], reg[eid]["hi"]]
        s["eval_id"] = eid
        seeds[rung] = s
        spec = copy.deepcopy(r1)
        spec.update({
            "record_id": cfg["spec"][:-5], "status": "FROZEN_BEFORE_TRAINING", "utc": now(),
            "classification": f"{rung} rung of the pre-registered 4v4 rescue ladder (DUAL_BRANCH_4V4_RESCUE_LADDER_SPEC.json). "
                              f"Runs only if every earlier rung fails the ladder's pass rule. Not PAPER-FAITHFUL.",
            "derived_from": {"path": str(R1_SPEC.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(R1_SPEC)},
            "ONLY_CHANGE_vs_R1": cfg["change_vs_R1"],
            "TRAINING_locked": {"total_timesteps": cfg["steps"], "defend_teacher": cfg["teacher"],
                                "k_defend_4v4": 1, "run_label_suffix": cfg["suffix"]},
            "R1_RUN_locked": None,
            "RUN_locked": {"team_size": 4, "k_defend": 1, "run_label_suffix": cfg["suffix"],
                           "training_seeds": {"A": s["A"], "B": s["B"]}, "driver": f"4v4/run_dual_branch_4v4_rescue.py --rung {rung}"},
            "CONFIRMATION_locked": {**r1["CONFIRMATION_locked"], "label": cfg["label"], "registry_experiment_id": eid,
                                    "block": f"{s['eval'][0]}..{s['eval'][1]}"},
        })
        (SD / cfg["spec"]).write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    doc = {
        "record_id": "DUAL_BRANCH_4V4_RESCUE_LADDER_SPEC", "status": "FROZEN_BEFORE_R1_RESULT", "utc": now(),
        "decided_by": "user 2026-10-03: start the next step automatically if R1 fails and continue until a pass; "
                      "bounded to a capped, pre-registered, multiplicity-adjusted sequence",
        "frozen_before": f"{RUNGS['R1']['label']} sealed (its result file did not exist at freeze; its rows were not read)",
        "order": list(ORDER), "order_rationale": ORDER_RATIONALE,
        "cap": f"at most {N_ATTEMPTS} attempts (R1, R2, R3); after R3 the ladder stops whatever the outcome",
        "pass_rule": {"statement": "lower bounds of BOTH Delta_A and Delta_B > 0 on the rung's own fresh 128-seed block",
                      "interval": f"paired percentile bootstrap over seeds, n={N_BOOT}, rng {BOOT_SEED}, two-sided level "
                                  f"1 - {ALPHA_FAMILY}/{N_ATTEMPTS} = {1 - ALPHA_FAMILY / N_ATTEMPTS:.4f} (Bonferroni over the "
                                  f"possible attempts); 95% intervals reported alongside",
                      "Delta": "Delta_A = V(A,A) - V(B,A), Delta_B = V(B,B) - V(A,B), win rate"},
        "on_pass": "stop the ladder; report every rung run so far; Stage 4 waits for PI review",
        "on_fail_after_R3": "stop; report all three rungs; 4v4 is written as not rescued under these single-axis changes",
        "reporting": "every rung's sealed result is reported (no selective reporting); a passing rung is described as the "
                     "outcome of this sequential procedure",
        "rungs": {r: {k: v for k, v in RUNGS[r].items()} for r in ORDER},
        "seeds": seeds,
        "symmetry": "every rung: identical recipe for A and B, both retrained from the sealed repaired 1M specialists",
        "not_authorized": ["any rung beyond R3", "changing a rung after its result is seen", "reusing any rung's block",
                           "Stage 4 before PI review"],
    }
    LADDER.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"-> {LADDER.name}; R2/R3 specs written; seeds {json.dumps(seeds)}")


# ------------------------------------------------------------------ chain
def log(msg: str) -> None:
    line = f"{now()} {msg}"
    print(line, flush=True)
    p = ROOT / "4v4" / "rescue_ladder" / "ladder.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def chain() -> int:
    lad = json.loads(LADDER.read_text(encoding="utf-8"))
    state_p = ROOT / "4v4" / "rescue_ladder" / "LADDER_STATE.json"
    state = json.loads(state_p.read_text(encoding="utf-8")) if state_p.is_file() else {"rungs": {}}
    py = str(ROOT / ".venv" / "Scripts" / "python.exe")
    env = dict(__import__("os").environ, FOR_DISABLE_CONSOLE_CTRL_HANDLER="1", PYTHONUNBUFFERED="1")
    for rung in lad["order"]:
        cfg = lad["rungs"][rung]
        if rung in state["rungs"] and "pass" in state["rungs"][rung]:
            if state["rungs"][rung]["pass"]:
                log(f"{rung} already recorded as PASS; ladder stopped")
                return 0
            continue
        done = ROOT / cfg["done"]
        if rung == "R1":
            log("waiting for R1 Stage 3 (4v4/r1/R1_STAGE3_DONE.txt)")
            while not done.is_file():
                time.sleep(120)
        elif not done.is_file():
            log(f"starting {rung}: {cfg['change_vs_R1']}")
            with (ROOT / "4v4" / "rescue_ladder" / f"{rung}.out").open("a", encoding="utf-8") as fh:
                rc = subprocess.run([py, "4v4/run_dual_branch_4v4_rescue.py", "--rung", rung], cwd=ROOT, env=env,
                                    stdout=fh, stderr=subprocess.STDOUT).returncode
            if rc != 0 or not done.is_file():
                log(f"STOPPED: {rung} runner exited {rc} without its done marker (rerun --chain to resume)")
                return 1
        lo, hi = lad["seeds"][rung]["eval"]
        d = rung_deltas(cfg["label"], list(range(lo, hi + 1)))
        ok = passes(d)
        state["rungs"][rung] = {**d, "pass": ok, "evaluated_utc": now()}
        state_p.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        fa, fb = d["Delta_A"], d["Delta_B"]
        log(f"{rung}: Delta_A {fa['mean']:+.3f} [{fa['lcb']:+.3f}, {fa['ucb']:+.3f}] (98.33%), "
            f"Delta_B {fb['mean']:+.3f} [{fb['lcb']:+.3f}, {fb['ucb']:+.3f}] -> {'PASS' if ok else 'not passed'}")
        if ok:
            log(f"LADDER STOPS at {rung} (pass). Stage 4 waits for PI review.")
            state["outcome"] = f"PASS at {rung}"
            state_p.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
            return 0
    state["outcome"] = "NO PASS after R1-R3; ladder capped and stopped"
    state_p.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    log(state["outcome"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--freeze", action="store_true")
    g.add_argument("--chain", action="store_true")
    a = ap.parse_args()
    if a.freeze:
        freeze()
        return 0
    return chain()


if __name__ == "__main__":
    raise SystemExit(main())
