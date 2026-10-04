r"""Specialist qualification candidate procedure (SPECIALIST_QUALIFICATION_V1_SPEC.json + AMENDMENT_1), 4v4.

    python experiments/specialist_qualification.py --team-size 4 --check
    python experiments/specialist_qualification.py --team-size 4 --run        (resumable; run detached)

Invoked because the existing 4v4 pair failed its qualification gate (SCK_4V4_CONFIRM: Delta_B -0.172). Executes
the frozen procedure exactly; parallelism is execution-only (user 2026-10-04) and changes no science:
  1. train   3 A + 3 B NEW candidates: the existing b3 parent -> 1M entity-repair continuation, frozen recipe,
             pre-reserved seeds; at most 3 trainings at once
  2. select  D_select: every candidate on both poles (pairs A_i/B_i run together purely for execution; each
             policy's own two cells are read). A* = argmax V(A_i,A)-V(A_i,B), B* = argmax V(B_j,B)-V(B_j,A);
             tie: higher Q -> higher intended-pole win -> lowest seed. Sealed and locked.
  3. gate    D_gate: the selected pair, four cells; mean Delta_A^base > 0 AND Delta_B^base > 0, else STOP for
             good (no 4th candidate, no second-best, no new k).
  4. roles   unchanged role stage (defenders + shared-k + final fresh confirmation): role_stage() below.
Never starts Stage 4, 2v2 or 6v6.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
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
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
SPEC_P = SD / "SPECIALIST_QUALIFICATION_V1_SPEC.json"
MAX_PARALLEL = 3


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(rel) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


class Q:
    def __init__(self, n: int):
        if n != 4:
            raise SystemExit("REFUSING: only 4v4 is authorized")
        self.n = n
        self.spec = json.loads(SPEC_P.read_text(encoding="utf-8"))
        self.cp = self.spec["CANDIDATE_PROCEDURE"]
        self.out = ROOT / "specialist_qualification" / f"{n}v{n}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.cands = {f"{s}{i}": self._cand(s, i) for s in "AB" for i in (1, 2, 3)}

    def _cand(self, side: str, i: int) -> dict:
        suffix = f"_sq_cand{side}{i}"
        d = f"artifacts/scale_4v4_specialists/pi_{side}_specialist_4v4{suffix}"
        return {"side": side, "i": i, "seed": int(self.cp["training_seeds"][f"{side}{i}"]),
                "eid": f"SQ_4V4_CAND_{side}{i}_TRAIN", "suffix": suffix, "dir": d,
                "final": f"{d}/ckpts/final_pi_{side}_specialist_4v4{suffix}.zip"}

    # -------------------------------------------------------------- plumbing
    def log(self, msg: str) -> None:
        line = f"{now()} {msg}"
        print(line, flush=True)
        with (self.out / "sq.log").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def env(self) -> dict:
        return dict(os.environ, FOR_DISABLE_CONSOLE_CTRL_HANDLER="1", PYTHONUNBUFFERED="1")

    def popen(self, argv: list[str], tag: str) -> subprocess.Popen:
        self.log(f"exec[{tag}]: {' '.join(argv)}")
        fo = (self.out / f"{tag}.log").open("a", encoding="utf-8")
        fe = (self.out / f"{tag}.log.err").open("a", encoding="utf-8")
        return subprocess.Popen([PY, *argv], cwd=ROOT, env=self.env(), stdout=fo, stderr=fe)

    def run_parallel(self, jobs: list[tuple[str, list[str], callable]]) -> None:
        """(tag, argv, done_fn) -- at most MAX_PARALLEL at once, staggered; a job already done is skipped."""
        pending = [j for j in jobs if not j[2]()]
        live: dict[str, tuple[subprocess.Popen, callable]] = {}
        while pending or live:
            for tag, (p, done) in list(live.items()):
                if p.poll() is not None:
                    del live[tag]
                    self.log(f"{tag} exited {p.returncode}")
                    if not done():
                        raise SystemExit(f"STOPPED: {tag} exited {p.returncode} without its output (rerun --run to resume)")
            if pending and len(live) < MAX_PARALLEL:
                tag, argv, done = pending.pop(0)
                live[tag] = (self.popen(argv, tag), done)
                time.sleep(60)
            else:
                time.sleep(30)

    # -------------------------------------------------------------- 1. train
    def train_argv(self, c: dict) -> list[str]:
        recipe = shlex.split(self.cp["recipe_locked"][c["side"]])
        assert recipe[0] == "train_specialist_scale.py"
        argv = ["experiments/train_specialist_scale.py", *recipe[1:], "--seed", str(c["seed"]), "--device", "cuda",
                "--experiment-id", c["eid"], "--run-label-suffix", c["suffix"]]
        ck = sorted((ROOT / c["dir"] / "ckpts").glob("ckpt_*.zip"),
                    key=lambda p: int(p.stem.rsplit("_", 1)[-1])) if (ROOT / c["dir"] / "ckpts").is_dir() else []
        if ck:                                             # resume an interrupted continuation
            i = argv.index("--load-path")
            del argv[i:i + 2]
            argv += ["--resume", str(ck[-1].relative_to(ROOT)).replace("\\", "/")]
        return argv

    def train(self) -> None:
        for side in "AB":
            par = self.cp["recipe_locked"]["parents"][side]
            if sha(par["path"]) != par["sha256"]:
                raise SystemExit(f"FAIL-CLOSED: {side} parent sha changed")
        order = ["A1", "B1", "A2", "B2", "A3", "B3"]      # execution order only
        self.run_parallel([(f"train_{k}", self.train_argv(self.cands[k]),
                            (lambda k=k: (ROOT / self.cands[k]["final"]).is_file())) for k in order])
        self.log("all 6 candidates trained: " + ", ".join(f"{k}={sha(c['final'])[:12]}" for k, c in self.cands.items()))

    # -------------------------------------------------------------- eval helpers
    def eval_argv(self, label: str, reg: str, lo: int, hi: int, a: str, b: str, spec: str) -> list[str]:
        return ["experiments/eval_specialist_crossover_scaled.py", "--team-size", "4", "--spec", spec,
                "--seed-base", str(lo), "--n-seeds", str(hi - lo + 1), "--label", label, "--device", "cuda",
                "--pi-a-path", a, "--pi-b-path", b, "--registry-experiment-id", reg]

    @staticmethod
    def result(label: str) -> Path:
        return SD / f"{label}_SPECIALIST_CROSSOVER_EVAL_RESULT.json"

    def sealed(self, label: str) -> bool:
        p = self.result(label)
        return p.is_file() and json.loads(p.read_text(encoding="utf-8")).get("status") == "SEALED"

    @staticmethod
    def cells(label: str, seeds: list[int]) -> dict:
        by: dict = {}
        with (SD / f"{label.lower()}_specialist_crossover_eval_rows.csv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                by.setdefault((r["policy"][-1], r["pole"]), {})[int(r["seed"])] = float(r["win"])
        for k, d in by.items():
            if sorted(d) != seeds:
                raise SystemExit(f"FAIL-CLOSED: {label} cell {k} is not exactly its block")
        return {k: float(np.mean([d[s] for s in seeds])) for k, d in by.items()}

    def eval_spec(self) -> str:
        """The evaluation spec the evaluator needs (frozen; derived from the qualification spec, no new choices)."""
        p = SD / "SPECIALIST_QUALIFICATION_V1_4V4_EVAL_SPEC.json"
        if not p.is_file():
            p.write_text(json.dumps({"record_id": p.stem, "status": "FROZEN_DEVELOPMENT_EVAL", "utc": now(),
                                     "confirmatory": False, "parent": SPEC_P.name, "parent_sha256": sha(SPEC_P.relative_to(ROOT)),
                                     "purpose": "evaluator spec for D_select and D_gate (development blocks) only"},
                                    indent=2) + "\n", encoding="utf-8")
        return str(p.relative_to(ROOT)).replace("\\", "/")

    # -------------------------------------------------------------- 2. select
    def select(self) -> dict:
        rec_p = self.out / "SQ_4V4_SELECTION_SEALED.json"
        if rec_p.is_file():
            return json.loads(rec_p.read_text(encoding="utf-8"))
        lo, hi = self.cp["selection_on_D_select"]["block"]
        spec = self.eval_spec()
        jobs = []
        for i in (1, 2, 3):
            label = f"SQ_4V4_SELECT_C{i}"
            jobs.append((f"select_c{i}", self.eval_argv(label, "SQ_4V4_SELECT", lo, hi, self.cands[f"A{i}"]["final"],
                                                        self.cands[f"B{i}"]["final"], spec) + ["--resume"],
                         (lambda label=label: self.sealed(label))))
        self.run_parallel(jobs)
        seeds = list(range(lo, hi + 1))
        V = {}
        for i in (1, 2, 3):
            c = self.cells(f"SQ_4V4_SELECT_C{i}", seeds)
            V[f"A{i}"] = {"own": c[("A", "A")], "off": c[("A", "B")]}
            V[f"B{i}"] = {"own": c[("B", "B")], "off": c[("B", "A")]}
        pick = {}
        for side in "AB":
            ks = [f"{side}{i}" for i in (1, 2, 3)]
            # (1) higher Q, (2) higher intended-pole win, (3) lowest preassigned training seed
            pick[side] = sorted(ks, key=lambda k: (-(V[k]["own"] - V[k]["off"]), -V[k]["own"], self.cands[k]["seed"]))[0]
        rec = {"record_id": rec_p.stem, "status": "SEALED_SELECTION", "utc": now(), "git_head": git_head(),
               "spec": {"path": SPEC_P.name, "sha256": sha(SPEC_P.relative_to(ROOT))}, "block": [lo, hi],
               "score": "Q_A = V(A,A) - V(A,B); Q_B = V(B,B) - V(B,A)",
               "tie_break": "higher Q -> higher intended-pole win -> lowest preassigned training seed",
               "candidates": {k: {"seed": c["seed"], "final": c["final"], "sha256": sha(c["final"]),
                                  "V_own": V[k]["own"], "V_off": V[k]["off"], "Q": V[k]["own"] - V[k]["off"]}
                              for k, c in self.cands.items()},
               "result_files": {f"C{i}": {"result_sha256": sha(self.result(f"SQ_4V4_SELECT_C{i}").relative_to(ROOT))} for i in (1, 2, 3)},
               "note": "A_i and B_i were evaluated in one run per i for execution only; each policy's score uses only its own cells",
               "selected": {"A": pick["A"], "B": pick["B"]}, "locked": "no reselection"}
        rec_p.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        self.log(f"SELECTION SEALED: A*={pick['A']} B*={pick['B']}  " +
                 ", ".join(f"{k}: Q={rec['candidates'][k]['Q']:+.3f}" for k in rec["candidates"]))
        return rec

    # -------------------------------------------------------------- 3. gate
    def gate(self, sel: dict) -> dict:
        rec_p = self.out / "SQ_4V4_GATE_SEALED.json"
        if rec_p.is_file():
            return json.loads(rec_p.read_text(encoding="utf-8"))
        lo, hi = self.cp["gate_on_D_gate"]["block"]
        a, b = self.cands[sel["selected"]["A"]]["final"], self.cands[sel["selected"]["B"]]["final"]
        self.run_parallel([("gate", self.eval_argv("SQ_4V4_GATE", "SQ_4V4_GATE", lo, hi, a, b, self.eval_spec()) + ["--resume"],
                            lambda: self.sealed("SQ_4V4_GATE"))])
        c = self.cells("SQ_4V4_GATE", list(range(lo, hi + 1)))
        dA, dB = c[("A", "A")] - c[("B", "A")], c[("B", "B")] - c[("A", "B")]
        ok = dA > 0 and dB > 0
        rec = {"record_id": rec_p.stem, "status": "SEALED_GATE", "utc": now(), "git_head": git_head(), "block": [lo, hi],
               "pair": sel["selected"], "cells": {f"{p}@{q}": c[(p, q)] for p in "AB" for q in "AB"},
               "Delta_A_base": dA, "Delta_B_base": dB, "criterion": "mean Delta_A^base > 0 AND mean Delta_B^base > 0",
               "qualified": ok, "result_sha256": sha(self.result("SQ_4V4_GATE").relative_to(ROOT)),
               "on_fail": "STOP the 4v4 line: no further batch, no second-best candidate, no new k"}
        rec_p.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        self.log(f"GATE SEALED: Delta_A^base {dA:+.3f}, Delta_B^base {dB:+.3f} -> {'QUALIFIED' if ok else 'NOT QUALIFIED -- 4v4 line stops'}")
        return rec

    # -------------------------------------------------------------- 4. roles
    def role_stage(self, sel: dict) -> int:
        try:
            from experiments import specialist_qualification_roles as R
        except ImportError:
            self.log("STOPPED before the role stage: experiments/specialist_qualification_roles.py not present yet; "
                     "rerun --run once it is (everything above is sealed and is reused)")
            return 3
        return R.run(self, sel)

    def check(self) -> list[str]:
        from experiments import seed_registry as SR
        p = []
        reg = {b["experiment_id"]: b for b in SR.load()["blocks"]}
        for k, c in self.cands.items():
            b = reg.get(c["eid"])
            if b is None or b["lo"] != c["seed"]:
                p.append(f"{c['eid']} not registered at {c['seed']}")
        for eid in ("SQ_4V4_SELECT", "SQ_4V4_GATE"):
            if eid not in reg:
                p.append(f"{eid} not registered")
        for side in "AB":
            par = self.cp["recipe_locked"]["parents"][side]
            if not (ROOT / par["path"]).is_file() or sha(par["path"]) != par["sha256"]:
                p.append(f"{side} parent missing or changed")
        if not str(self.spec.get("status", "")).startswith("FROZEN"):
            p.append("qualification spec not frozen")
        return p

    def run(self) -> int:
        self.log("specialist qualification candidate procedure started (existing pair failed: SCK_4V4_CONFIRM Delta_B -0.172)")
        self.train()
        sel = self.select()
        g = self.gate(sel)
        if not g["qualified"]:
            (self.out / "SQ_4V4_STOPPED.txt").write_text(f"NOT QUALIFIED {now()}\n", encoding="utf-8")
            self.log("4v4 line STOPPED for good (frozen rule)")
            return 0
        return self.role_stage(sel)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team-size", type=int, required=True, choices=(2, 4, 6))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    q = Q(a.team_size)
    problems = q.check()
    if a.check or problems:
        print("ALL CHECKS PASS" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
        print(json.dumps({k: q.train_argv(c) for k, c in list(q.cands.items())[:2]}, indent=1))
        return 0 if not problems else 1
    return q.run()


if __name__ == "__main__":
    raise SystemExit(main())
