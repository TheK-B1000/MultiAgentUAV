r"""Specialist qualification + symmetric-role pipeline: ONE state machine for every team size (2v2 / 4v4 / 6v6).

    python experiments/specialist_qualification.py --team-size 6 --audit          # default: plan only, no side effects
    python experiments/specialist_qualification.py --team-size 4 --run [--workers 3]

Scientific values come ONLY from experiments/sq_configs/<N>v<N>.json (4v4 mirrors the frozen
SPECIALIST_QUALIFICATION_V1_SPEC.json); the code never invents a seed, block, recipe or k menu. A config whose
status is not FROZEN, or that is missing any value a stage needs, can be audited but never run.

  EXISTING_PAIR_QUALIFICATION --PASS--> ROLE_STAGE
            |
            FAIL --> CANDIDATE_TRAINING --> D_SELECT --> LOCK_SELECTED_PAIR --> D_GATE --FAIL--> STOP
                                                                                   |
                                                                                  PASS --> ROLE_STAGE
  ROLE_STAGE (defenders) --> SHARED_K_SELECTION (k_A = k_B = k*) --> FINAL_CONFIRMATION --> STOP

Rules (identical at every size): qualification = mean Delta_A^base > 0 and Delta_B^base > 0 (mean-level); exactly
`count_per_strategy` candidates per strategy, same stage/recipe/parent, only seeds differ; Q_A = V(A,A) - V(A,B),
Q_B = V(B,B) - V(B,A), tie-break higher Q -> higher intended-pole win -> lowest seed; the selected pair is locked;
a gate failure stops the line for good (no extra candidate, no second-best, no new k); shared
S(k) = [V(A,k,A) + V(B,k,B)] / 2, tie -> smaller k, k_A = k_B always. Sealed records are written once and reused on
resume; parallelism (--workers) is execution-only and never changes seeds or candidate identity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIG_DIR = ROOT / "experiments" / "sq_configs"
SCHEMA = "specialist_qualification_config_v1"
STAGES = ("EXISTING_PAIR_QUALIFICATION", "CANDIDATE_TRAINING", "D_SELECT", "LOCK_SELECTED_PAIR", "D_GATE",
          "ROLE_STAGE", "SHARED_K_SELECTION", "FINAL_CONFIRMATION", "STOP")
N_BOOT, BOOT_SEED = 20_000, 7


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ====================================================================== pure scientific rules (unit-tested)
def k_role(n: int) -> int:
    """Frozen cross-scale rule (SPECIALIST_QUALIFICATION_V1_ROLE_MENU_RULE.json): max(1, floor(N/3 + 1/2))."""
    return max(1, (2 * int(n) + 3) // 6)


def qualifies(cells: dict) -> tuple[float, float, bool]:
    """Mean-level qualification / gate: (Delta_A, Delta_B, both > 0). cells: {"A@A","B@A","A@B","B@B"}."""
    dA, dB = cells["A@A"] - cells["B@A"], cells["B@B"] - cells["A@B"]
    return dA, dB, bool(dA > 0 and dB > 0)


def select_candidates(scores: dict) -> dict:
    """scores: {key: {"side","own","off","seed"}} -> {"A": key, "B": key}. Q = own - off; tie-break higher Q ->
    higher own (intended-pole win) -> lowest preassigned seed. Independent per side."""
    out = {}
    for side in ("A", "B"):
        ks = [k for k, v in scores.items() if v["side"] == side]
        if not ks:
            raise ValueError(f"no {side} candidates")
        out[side] = sorted(ks, key=lambda k: (-(scores[k]["own"] - scores[k]["off"]), -scores[k]["own"], scores[k]["seed"]))[0]
    return out


def shared_k(intended: dict, menu: list[int]) -> int:
    """intended: {k: {"A": V(A,k,PoleA), "B": V(B,k,PoleB)}} -> one k for BOTH strategies; tie -> smaller k."""
    S = {k: (intended[k]["A"] + intended[k]["B"]) / 2 for k in menu}
    best = max(S.values())
    return min(k for k in menu if S[k] == best)


def blocks_overlap(blocks: list[list[int]]) -> list[tuple]:
    bs = [b for b in blocks if b]
    return [(a, b) for i, a in enumerate(bs) for b in bs[i + 1:] if not (a[1] < b[0] or b[1] < a[0])]


def write_sealed(path: Path, obj: dict) -> Path:
    """Write-once: a sealed record is never overwritten (identical content is accepted)."""
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != obj:
            raise SystemExit(f"REFUSING: sealed record {path.name} exists; sealed stages are never rewritten")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    return path


def validate_config(cfg: dict, *, for_run: bool) -> list[str]:
    """Schema + consistency. for_run=True additionally requires everything a run needs (frozen, seeds, blocks, menu)."""
    p = []
    if cfg.get("schema") != SCHEMA:
        p.append(f"schema != {SCHEMA}")
    n = cfg.get("team_size")
    if n not in (2, 4, 6):
        p.append(f"team_size {n!r} not in (2,4,6)")
    for side in ("A", "B"):
        e = (cfg.get("existing_specialists") or {}).get(side) or {}
        if not e.get("path") or len(str(e.get("sha256") or "")) != 64:
            p.append(f"existing_specialists.{side} needs path + sha256")
        r = ((cfg.get("candidates") or {}).get("recipe") or {}).get(side)
        if not r or "--load-path" not in r or r[r.index("--team-size") + 1] != str(n) or r[r.index("--policy") + 1] != side:
            p.append(f"candidates.recipe.{side} malformed (team size / policy / --load-path)")
        if r and "--seed" in r:
            p.append(f"candidates.recipe.{side} must not contain a seed")
    if int((cfg.get("candidates") or {}).get("count_per_strategy") or 0) < 1:
        p.append("candidates.count_per_strategy missing")
    b = cfg.get("blocks") or {}
    epq = cfg.get("existing_pair_qualification") or {}
    over = blocks_overlap([epq.get("block"), b.get("d_select"), b.get("d_gate"), b.get("role_dev"), b.get("final_confirmation")])
    if over:
        p.append(f"overlapping seed blocks: {over}")
    menu = (cfg.get("role") or {}).get("menu")
    if menu is not None and n in (2, 4, 6) and list(menu) != [0, k_role(n)]:
        p.append(f"role.menu {menu} != K(N) = [0, k_role({n})] = {[0, k_role(n)]} (SPECIALIST_QUALIFICATION_V1_ROLE_MENU_RULE)")
    if menu is not None and (cfg["role"].get("defender_k") not in menu or cfg["role"].get("defender_k") == 0):
        p.append("role.defender_k must be the nonzero menu value")
    if for_run:
        if cfg.get("status") != "FROZEN":
            p.append(f"status {cfg.get('status')!r}: only a FROZEN config may run")
        seeds = (cfg.get("candidates") or {}).get("training_seeds")
        cnt = int((cfg.get("candidates") or {}).get("count_per_strategy") or 0)
        if not seeds or sorted(seeds) != sorted(f"{s}{i}" for s in "AB" for i in range(1, cnt + 1)):
            p.append("candidates.training_seeds missing or not exactly count_per_strategy per side")
        elif len(set(seeds.values())) != len(seeds):
            p.append("candidate training seeds not unique")
        for side in ("A", "B"):
            par = ((cfg.get("candidates") or {}).get("parents") or {}).get(side) or {}
            if len(str(par.get("sha256") or "")) != 64:
                p.append(f"candidates.parents.{side}.sha256 not pinned")
        for k in ("d_select", "d_gate", "role_dev", "final_confirmation"):
            if not b.get(k):
                p.append(f"blocks.{k} not reserved")
        if menu is None:
            p.append("role.menu not prospectively defined for this scale")
        if not epq.get("block"):
            p.append("existing_pair_qualification.block not defined")
    return p


def load_config(n: int, config_dir: Path = CONFIG_DIR) -> dict:
    p = config_dir / f"{n}v{n}.json"
    if not p.is_file():
        raise SystemExit(f"FAIL-CLOSED: {p} missing")
    return json.loads(p.read_text(encoding="utf-8"))


# ====================================================================== pipeline
class Pipeline:
    def __init__(self, cfg: dict, *, root: Path = ROOT, workers: int = 3):
        self.cfg, self.root, self.workers = cfg, Path(root), max(1, int(workers))
        self.n = n = int(cfg["team_size"])
        self.N = f"{n}V{n}"
        self.sd = self.root / "artifacts" / "strategic_demand" / "sppo"
        self.out = self.root / "specialist_qualification" / f"{n}v{n}"
        self.py = str(self.root / ".venv" / "Scripts" / "python.exe")
        cnt = int(cfg["candidates"]["count_per_strategy"])
        seeds = cfg["candidates"].get("training_seeds") or {}
        self.cands = {}
        for side in "AB":
            for i in range(1, cnt + 1):        # identity fixed by config, independent of worker count / order
                suffix = f"_sq_cand{side}{i}"
                d = f"artifacts/scale_{n}v{n}_specialists/pi_{side}_specialist_{n}v{n}{suffix}"
                self.cands[f"{side}{i}"] = {"side": side, "i": i, "seed": seeds.get(f"{side}{i}"), "suffix": suffix,
                                            "eid": f"SQ_{self.N}_CAND_{side}{i}_TRAIN", "dir": d,
                                            "final": f"{d}/ckpts/final_pi_{side}_specialist_{n}v{n}{suffix}.zip"}

    # ---------------------------------------------------------------- naming (4v4 names unchanged)
    def rec(self, name: str) -> Path:
        return self.out / f"SQ_{self.N}_{name}"

    def label(self, name: str) -> str:
        return f"SQ_{self.N}_{name}"

    def result_path(self, label: str) -> Path:
        return self.sd / f"{label}_SPECIALIST_CROSSOVER_EVAL_RESULT.json"

    def rows_path(self, label: str) -> Path:
        return self.sd / f"{label.lower()}_specialist_crossover_eval_rows.csv"

    def def_dir(self, side: str) -> dict:
        k = self.cfg["role"]["defender_k"]
        d = f"artifacts/scale_{self.n}v{self.n}_specialists/pi_{side}_specialist_{self.n}v{self.n}_sq_def_k{k}"
        return {"dir": d, "final": f"{d}/ckpts/final_pi_{side}_specialist_{self.n}v{self.n}_sq_def_k{k}.zip"}

    # ---------------------------------------------------------------- io helpers
    def sha(self, rel) -> str:
        return hashlib.sha256((self.root / rel).read_bytes()).hexdigest()

    def exists(self, rel) -> bool:
        return (self.root / rel).is_file()

    def sealed(self, label: str) -> bool:
        p = self.result_path(label)
        return p.is_file() and json.loads(p.read_text(encoding="utf-8")).get("status") == "SEALED"

    def cells(self, label: str, block: list[int], field: str = "win") -> dict:
        seeds = list(range(block[0], block[1] + 1))
        by: dict = {}
        with self.rows_path(label).open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                by.setdefault((r["policy"][-1], r["pole"]), {})[int(r["seed"])] = float(r[field])
        for key, d in by.items():
            if sorted(d) != seeds:
                raise SystemExit(f"FAIL-CLOSED: {label} cell {key} is not exactly its block")
        return {f"{p}@{q}": float(np.mean([by[(p, q)][s] for s in seeds])) for p in "AB" for q in "AB"}

    def git_head(self) -> str:
        return subprocess.run(["git", "-C", str(self.root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    def log(self, msg: str) -> None:
        line = f"{now()} {msg}"
        print(line, flush=True)
        self.out.mkdir(parents=True, exist_ok=True)
        with (self.out / "sq.log").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def transition(self, frm: str, to: str, why: str) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        with self.rec("TRANSITIONS.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"utc": now(), "from": frm, "to": to, "why": why}) + "\n")
        self.log(f"TRANSITION {frm} -> {to}: {why}")

    # ---------------------------------------------------------------- execution (overridable in tests)
    def execute(self, jobs: list[tuple[str, list[str], callable]]) -> None:
        """(tag, argv, done) jobs; at most `workers` at once; finished jobs skipped; a job that exits without its
        output stops the pipeline (rerun resumes)."""
        env = dict(os.environ, FOR_DISABLE_CONSOLE_CTRL_HANDLER="1", PYTHONUNBUFFERED="1")
        pending = [j for j in jobs if not j[2]()]
        live: dict = {}
        while pending or live:
            for tag, (p, done) in list(live.items()):
                if p.poll() is not None:
                    del live[tag]
                    self.log(f"{tag} exited {p.returncode}")
                    if not done():
                        raise SystemExit(f"STOPPED: {tag} exited {p.returncode} without its output (rerun --run to resume)")
            if pending and len(live) < self.workers:
                tag, argv, done = pending.pop(0)
                self.log(f"exec[{tag}]: {' '.join(argv)}")
                fo = (self.out / f"{tag}.log").open("a", encoding="utf-8")
                fe = (self.out / f"{tag}.log.err").open("a", encoding="utf-8")
                live[tag] = (subprocess.Popen([self.py, *argv], cwd=self.root, env=env, stdout=fo, stderr=fe), done)
                time.sleep(60)
            else:
                time.sleep(30)

    # ---------------------------------------------------------------- argv builders
    def train_argv(self, c: dict) -> list[str]:
        argv = ["experiments/train_specialist_scale.py", *self.cfg["candidates"]["recipe"][c["side"]],
                "--seed", str(c["seed"]), "--device", "cuda", "--experiment-id", c["eid"], "--run-label-suffix", c["suffix"]]
        ckdir = self.root / c["dir"] / "ckpts"
        ck = sorted(ckdir.glob("ckpt_*.zip"), key=lambda p: int(p.stem.rsplit("_", 1)[-1])) if ckdir.is_dir() else []
        if ck:
            i = argv.index("--load-path")
            del argv[i:i + 2]
            argv += ["--resume", str(ck[-1].relative_to(self.root)).replace("\\", "/")]
        return argv

    def derived_spec(self, kind: str, confirmatory: bool) -> str:
        p = self.sd / f"SPECIALIST_QUALIFICATION_V1_{self.N}_{kind}_SPEC.json"
        if not p.is_file():
            write_sealed(p, {"record_id": p.stem, "status": f"FROZEN_{kind}", "utc": now(), "confirmatory": confirmatory,
                             "parent": self.cfg.get("frozen_spec"), "team_size": self.n,
                             "purpose": f"evaluator spec for {kind.lower()} runs of the {self.n}v{self.n} qualification pipeline"})
        return str(p.relative_to(self.root)).replace("\\", "/")

    def eval_argv(self, label, block, a_args, b_args, spec, reg=None, roles=False) -> list[str]:
        a = ["experiments/eval_specialist_crossover_scaled.py", "--team-size", str(self.n), "--spec", spec,
             "--seed-base", str(block[0]), "--n-seeds", str(block[1] - block[0] + 1), "--label", label, "--device", "cuda",
             *a_args, *b_args]
        if roles:
            a += ["--role-fixed-for-episode", "--role-k-defend", str(self.cfg["role"]["defender_k"])]
        if reg:
            a += ["--registry-experiment-id", reg]
        return a + ["--resume"]

    def pair_args(self, pair: dict, k: int) -> tuple[list[str], list[str]]:
        out = []
        for side in "AB":
            flag = "--pi-a-path" if side == "A" else "--pi-b-path"
            spec_path = pair[side]["path"]
            if k == 0:
                out.append([flag, spec_path])
            else:
                fa = ("--frozen-attack-path", "--frozen-attack-path-sha256") if side == "A" else ("--frozen-attack-path-b", "--frozen-attack-path-b-sha256")
                out.append([flag, self.def_dir(side)["final"], fa[0], spec_path, fa[1], pair[side]["sha256"]])
        return out[0], out[1]

    # ---------------------------------------------------------------- stage status (resume / audit)
    def status(self) -> dict:
        st = {}
        epq = self.cfg["existing_pair_qualification"]
        st["EXISTING_PAIR_QUALIFICATION"] = self.rec("EXISTING_PAIR_QUALIFICATION.json").is_file() or (
            epq.get("source") == "sealed_artifact" and self.sealed(epq["label"]))
        st["CANDIDATE_TRAINING"] = all(self.exists(c["final"]) for c in self.cands.values())
        st["D_SELECT"] = self.rec("SELECTION_SEALED.json").is_file()
        st["LOCK_SELECTED_PAIR"] = st["D_SELECT"]
        st["D_GATE"] = self.rec("GATE_SEALED.json").is_file()
        st["ROLE_STAGE"] = self.cfg["role"].get("defender_k") is not None and all(self.exists(self.def_dir(s)["final"]) for s in "AB")
        st["SHARED_K_SELECTION"] = self.rec("ROLE_SELECTION_SEALED.json").is_file()
        st["FINAL_CONFIRMATION"] = self.rec("CONFIRM_READOUT.json").is_file()
        st["STOP"] = st["FINAL_CONFIRMATION"] or self.rec("STOPPED.txt").is_file()
        return st

    # ---------------------------------------------------------------- stages
    def existing_pair_qualification(self) -> dict:
        p = self.rec("EXISTING_PAIR_QUALIFICATION.json")
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
        epq, ex = self.cfg["existing_pair_qualification"], self.cfg["existing_specialists"]
        for side in "AB":
            if self.sha(ex[side]["path"]) != ex[side]["sha256"]:
                raise SystemExit(f"FAIL-CLOSED: existing specialist {side} sha changed")
        if epq["source"] == "sealed_artifact":
            cond = epq.get("valid_iff")
            if cond:
                sel = json.loads((self.root / cond["record"]).read_text(encoding="utf-8")).get("selected")
                if sel != cond["selected"]:
                    raise SystemExit(f"FAIL-CLOSED: {cond['record']} selected {sel}, not {cond['selected']}: artifact is not the plain pair")
        else:
            self.execute([("existing_pair", self.eval_argv(epq["label"], epq["block"], ["--pi-a-path", ex["A"]["path"]],
                                                           ["--pi-b-path", ex["B"]["path"]], self.derived_spec("EVAL", False),
                                                           reg=epq.get("registry_id")), lambda: self.sealed(epq["label"]))])
        if not self.sealed(epq["label"]):
            raise SystemExit(f"FAIL-CLOSED: {epq['label']} not sealed")
        c = self.cells(epq["label"], epq["block"])
        dA, dB, ok = qualifies(c)
        return json.loads(write_sealed(p, {
            "record_id": p.stem, "status": "SEALED_QUALIFICATION", "utc": now(), "git_head": self.git_head(), "team_size": self.n,
            "pair": {s: ex[s] for s in "AB"}, "label": epq["label"], "block": epq["block"], "cells": c,
            "Delta_A_base": dA, "Delta_B_base": dB, "criterion": "mean Delta_A^base > 0 AND mean Delta_B^base > 0",
            "qualified": ok, "result_sha256": self.sha(self.result_path(epq["label"]).relative_to(self.root))}).read_text(encoding="utf-8"))

    def candidate_training(self) -> None:
        for side in "AB":
            par = self.cfg["candidates"]["parents"][side]
            if self.sha(par["path"]) != par["sha256"]:
                raise SystemExit(f"FAIL-CLOSED: {side} parent sha changed")
        order = [f"{s}{i}" for i in range(1, int(self.cfg["candidates"]["count_per_strategy"]) + 1) for s in "AB"]
        self.execute([(f"train_{k}", self.train_argv(self.cands[k]), (lambda k=k: self.exists(self.cands[k]["final"]))) for k in order])

    def d_select(self) -> dict:
        p = self.rec("SELECTION_SEALED.json")
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
        blk, spec = self.cfg["blocks"]["d_select"], self.derived_spec("EVAL", False)
        cnt = int(self.cfg["candidates"]["count_per_strategy"])
        jobs = []
        for i in range(1, cnt + 1):
            lab = self.label(f"SELECT_C{i}")
            jobs.append((f"select_c{i}", self.eval_argv(lab, blk, ["--pi-a-path", self.cands[f"A{i}"]["final"]],
                                                        ["--pi-b-path", self.cands[f"B{i}"]["final"]], spec, reg=self.label("SELECT")),
                         (lambda lab=lab: self.sealed(lab))))
        self.execute(jobs)
        scores = {}
        for i in range(1, cnt + 1):
            c = self.cells(self.label(f"SELECT_C{i}"), blk)
            scores[f"A{i}"] = {"side": "A", "own": c["A@A"], "off": c["A@B"], "seed": self.cands[f"A{i}"]["seed"]}
            scores[f"B{i}"] = {"side": "B", "own": c["B@B"], "off": c["B@A"], "seed": self.cands[f"B{i}"]["seed"]}
        pick = select_candidates(scores)
        rec = {"record_id": p.stem, "status": "SEALED_SELECTION", "utc": now(), "git_head": self.git_head(), "block": blk,
               "score": "Q_A = V(A,A) - V(A,B); Q_B = V(B,B) - V(B,A)",
               "tie_break": "higher Q -> higher intended-pole win -> lowest preassigned training seed",
               "candidates": {k: {"seed": c["seed"], "final": c["final"], "sha256": self.sha(c["final"]),
                                  "V_own": scores[k]["own"], "V_off": scores[k]["off"], "Q": scores[k]["own"] - scores[k]["off"]}
                              for k, c in self.cands.items()},
               "selected": pick, "locked": "no reselection"}
        write_sealed(p, rec)
        self.log(f"SELECTION SEALED: A*={pick['A']} B*={pick['B']}")
        return rec

    def locked_pair(self, sel: dict | None) -> dict:
        if sel is None:                                   # existing pair qualified
            ex = self.cfg["existing_specialists"]
            return {s: {"key": "existing", "path": ex[s]["path"], "sha256": ex[s]["sha256"]} for s in "AB"}
        return {s: {"key": sel["selected"][s], "path": self.cands[sel["selected"][s]]["final"],
                    "sha256": sel["candidates"][sel["selected"][s]]["sha256"]} for s in "AB"}

    def d_gate(self, pair: dict) -> dict:
        p = self.rec("GATE_SEALED.json")
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
        blk, lab = self.cfg["blocks"]["d_gate"], self.label("GATE")
        self.execute([("gate", self.eval_argv(lab, blk, ["--pi-a-path", pair["A"]["path"]], ["--pi-b-path", pair["B"]["path"]],
                                              self.derived_spec("EVAL", False), reg=lab), lambda: self.sealed(lab))])
        c = self.cells(lab, blk)
        dA, dB, ok = qualifies(c)
        rec = {"record_id": p.stem, "status": "SEALED_GATE", "utc": now(), "git_head": self.git_head(), "block": blk,
               "pair": {s: pair[s]["key"] for s in "AB"}, "cells": c, "Delta_A_base": dA, "Delta_B_base": dB,
               "criterion": "mean Delta_A^base > 0 AND mean Delta_B^base > 0", "qualified": ok,
               "result_sha256": self.sha(self.result_path(lab).relative_to(self.root)),
               "on_fail": "STOP this team-size line: no further batch, no second-best candidate, no new k"}
        write_sealed(p, rec)
        self.log(f"GATE SEALED: Delta_A {dA:+.3f}, Delta_B {dB:+.3f} -> {'QUALIFIED' if ok else 'NOT QUALIFIED'}")
        return rec

    def role_auth(self, pair: dict) -> str:
        p = self.sd / f"SPECIALIST_QUALIFICATION_V1_{self.N}_ROLE_AUTH_SPEC.json"
        if not p.is_file():
            k = self.cfg["role"]["defender_k"]
            write_sealed(p, {"record_id": p.stem, "status": "FROZEN_DERIVED", "utc": now(), "confirmatory": False,
                             "parent": self.cfg.get("frozen_spec"),
                             "purpose": "authorizes the two role-stage defender runs; recipe = frozen defender-only recipe",
                             "TRAINING_AUTHORIZED": [{"team_size": self.n, "policy": s, "role_k_defend": k,
                                                      "run_label_suffix": f"_sq_def_k{k}", "specialist_path": pair[s]["path"],
                                                      "specialist_sha256": pair[s]["sha256"],
                                                      "seed": int(self.cfg["role"]["defender_seeds"][s]),
                                                      "experiment_id": f"SQ_{self.N}_DEF_{s}_TRAIN"} for s in "AB"]})
        return str(p.relative_to(self.root)).replace("\\", "/")

    def defender_argv(self, pair: dict, side: str, auth: str) -> list[str]:
        r, k = self.cfg["role"]["defender_recipe"], self.cfg["role"]["defender_k"]
        sp = pair[side]["path"]
        return ["experiments/train_specialist_scale.py", "--team-size", str(self.n), "--policy", side,
                "--seed", str(self.cfg["role"]["defender_seeds"][side]), "--device", "cuda",
                "--total-timesteps", str(r["total_timesteps"]), "--entity-repair-enabled", "--entity-hidden-dim", str(r["entity_hidden_dim"]),
                "--role-conditioning-enabled", "--role-hold-ticks", str(r["role_hold_ticks"]), "--role-fixed-for-episode",
                "--role-k-defend", str(k), "--split-attack-defend-enabled", "--split-attack-defend-frozen-ckpt", sp,
                "--split-attack-defend-frozen-ckpt-sha256", pair[side]["sha256"], "--load-path", sp,
                "--defend-teacher-lambda", str(r["defend_teacher_lambda"]), "--defend-teacher-lambda-end", str(r["defend_teacher_lambda_end"]),
                "--defend-teacher-decay-start-step", str(r["defend_teacher_decay_start_step"]),
                "--defend-teacher-decay-end-step", str(r["defend_teacher_decay_end_step"]),
                "--defend-teacher-cadence", str(r["defend_teacher_cadence"]), "--run-label-suffix", f"_sq_def_k{k}",
                "--symmetric-role-spec", auth, "--experiment-id", f"SQ_{self.N}_DEF_{side}_TRAIN"]

    def role_stage(self, pair: dict) -> None:
        auth = self.role_auth(pair)
        self.execute([(f"def_{s}", self.defender_argv(pair, s, auth), (lambda s=s: self.exists(self.def_dir(s)["final"]))) for s in "AB"])

    def shared_k_selection(self, pair: dict) -> int:
        p = self.rec("ROLE_SELECTION_SEALED.json")
        if p.is_file():
            return int(json.loads(p.read_text(encoding="utf-8"))["selected_k"])
        menu, blk, spec = self.cfg["role"]["menu"], self.cfg["blocks"]["role_dev"], self.derived_spec("EVAL", False)
        jobs = []
        for k in menu:
            lab = self.label(f"ROLE_DEV_K{k}")
            a, b = self.pair_args(pair, k)
            jobs.append((f"role_dev_k{k}", self.eval_argv(lab, blk, a, b, spec, reg=self.label("ROLE_DEV"), roles=k > 0),
                         (lambda lab=lab: self.sealed(lab))))
        self.execute(jobs)
        V = {k: self.cells(self.label(f"ROLE_DEV_K{k}"), blk) for k in menu}
        kstar = shared_k({k: {"A": V[k]["A@A"], "B": V[k]["B@B"]} for k in menu}, menu)
        write_sealed(p, {"record_id": p.stem, "status": "SEALED_SELECTION", "utc": now(), "block": blk,
                         "rule": "S(k) = [V(A,k,A) + V(B,k,B)]/2, k* = argmax, tie -> smaller k; k_A = k_B = k*",
                         "intended_pole": {f"A@A_k{k}": V[k]["A@A"] for k in menu} | {f"B@B_k{k}": V[k]["B@B"] for k in menu},
                         "S": {str(k): (V[k]["A@A"] + V[k]["B@B"]) / 2 for k in menu}, "selected_k": kstar})
        self.log(f"ROLE SELECTION SEALED: k* = {kstar}")
        return kstar

    def final_confirmation(self, pair: dict, kstar: int) -> dict:
        p = self.rec("CONFIRM_READOUT.json")
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
        blk, lab = self.cfg["blocks"]["final_confirmation"], self.label("CONFIRM")
        a, b = self.pair_args(pair, kstar)
        self.execute([("confirm", self.eval_argv(lab, blk, a, b, self.derived_spec("CONFIRM_EVAL", True), roles=kstar > 0),
                       lambda: self.sealed(lab))])
        from experiments.eval_hog_psp_v3 import _mean_ci
        seeds = list(range(blk[0], blk[1] + 1))
        by: dict = {}
        with self.rows_path(lab).open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                by.setdefault((r["policy"][-1], r["pole"]), {})[int(r["seed"])] = (float(r["win"]), float(r["margin"]))
        out = {"record": p.stem, "utc": now(), "k_star": kstar, "pair": {s: pair[s]["key"] for s in "AB"}, "block": blk,
               "result_sha256": self.sha(self.result_path(lab).relative_to(self.root)), "levels": self.cfg["report"]["levels"]}
        for i, field in enumerate(("win", "margin")):
            v = {key: np.array([d[s][i] for s in seeds]) for key, d in by.items()}
            res = {f"{x}@{y}": float(v[(x, y)].mean()) for x in "AB" for y in "AB"}
            for name, x in (("Delta_A", v[("A", "A")] - v[("B", "A")]), ("Delta_B", v[("B", "B")] - v[("A", "B")])):
                c = _mean_ci(x)
                res[name] = {"mean": float(x.mean()), "std": float(x.std(ddof=1)), "ci95": [c["lcb95"], c["ucb95"]]}
                for lvl in self.cfg["report"]["levels"]:
                    if abs(lvl - 0.95) > 1e-12:
                        res[name][f"ci{lvl:.4f}"] = _boot(x, 1 - lvl)
            out[field] = res
        f = lambda s: f"{s['mean']:+.3f} ± {s['std']:.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]"  # noqa: E731
        md = [f"# {self.n}v{self.n} qualified pair ({pair['A']['key']}, {pair['B']['key']}) + shared-k roles (k* = {kstar}): "
              f"fresh confirmation {blk[0]}..{blk[1]}", "", "| | A@A | B@A | A@B | B@B | Δ_A (95%) | Δ_B (95%) |", "|---|---|---|---|---|---|---|"]
        for field in ("win", "margin"):
            t = out[field]
            md.append(f"| {field} | {t['A@A']:.3f} | {t['B@A']:.3f} | {t['A@B']:.3f} | {t['B@B']:.3f} | {f(t['Delta_A'])} | {f(t['Delta_B'])} |")
        out["table_markdown"] = "\n".join(md)
        write_sealed(p, out)
        p.with_suffix(".md").write_text(out["table_markdown"] + "\n", encoding="utf-8")
        return out

    # ---------------------------------------------------------------- driver
    def run(self) -> int:
        probs = validate_config(self.cfg, for_run=True)
        if probs:
            raise SystemExit("REFUSING to run:\n  " + "\n  ".join(probs))
        self.out.mkdir(parents=True, exist_ok=True)
        q = self.existing_pair_qualification()
        sel = None
        if q["qualified"]:
            self.transition("EXISTING_PAIR_QUALIFICATION", "ROLE_STAGE", "existing pair qualified")
        else:
            self.transition("EXISTING_PAIR_QUALIFICATION", "CANDIDATE_TRAINING", f"existing pair failed (dA {q['Delta_A_base']:+.3f}, dB {q['Delta_B_base']:+.3f})")
            self.candidate_training()
            self.transition("CANDIDATE_TRAINING", "D_SELECT", "all candidates trained")
            sel = self.d_select()
            self.transition("D_SELECT", "LOCK_SELECTED_PAIR", f"A*={sel['selected']['A']} B*={sel['selected']['B']}")
            g = self.d_gate(self.locked_pair(sel))
            if not g["qualified"]:
                (self.rec("STOPPED.txt")).write_text(f"NOT QUALIFIED {now()}\n", encoding="utf-8")
                self.transition("D_GATE", "STOP", "selected pair failed the gate; line stopped for good")
                return 0
            self.transition("D_GATE", "ROLE_STAGE", "selected pair qualified")
        pair = self.locked_pair(sel)
        self.role_stage(pair)
        self.transition("ROLE_STAGE", "SHARED_K_SELECTION", "defenders trained")
        k = self.shared_k_selection(pair)
        self.transition("SHARED_K_SELECTION", "FINAL_CONFIRMATION", f"k_A = k_B = {k}")
        self.final_confirmation(pair, k)
        self.transition("FINAL_CONFIRMATION", "STOP", "readout written; no Stage 4, no other scale without approval")
        return 0

    # ---------------------------------------------------------------- audit (no side effects)
    def audit(self) -> dict:
        c, ex = self.cfg, self.cfg["existing_specialists"]
        rep = {"team_size": self.n, "status": c.get("status"), "frozen_spec": c.get("frozen_spec"),
               "config_problems_for_run": validate_config(c, for_run=True), "schema_problems": validate_config(c, for_run=False)}
        rep["existing_pair"] = {s: {"path": ex[s]["path"], "sha256_pinned": ex[s]["sha256"],
                                    "present": self.exists(ex[s]["path"]),
                                    "sha_ok": self.exists(ex[s]["path"]) and self.sha(ex[s]["path"]) == ex[s]["sha256"]} for s in "AB"}
        rep["parents"] = {s: {"path": c["candidates"]["parents"][s]["path"], "present": self.exists(c["candidates"]["parents"][s]["path"]),
                              "sha256_pinned": c["candidates"]["parents"][s]["sha256"],
                              "sha256_actual": self.sha(c["candidates"]["parents"][s]["path"]) if self.exists(c["candidates"]["parents"][s]["path"]) else None}
                          for s in "AB"}
        epq = c["existing_pair_qualification"]
        qrec = self.rec("EXISTING_PAIR_QUALIFICATION.json")
        rep["existing_pair_qualification"] = {"source": epq["source"], "label": epq["label"], "block": epq.get("block"),
                                              "sealed_artifact_present": self.sealed(epq["label"]),
                                              "qualification_record": json.loads(qrec.read_text(encoding="utf-8")) if qrec.is_file() else None}
        rep["candidates_would_be_required"] = (None if not qrec.is_file()
                                               else not json.loads(qrec.read_text(encoding="utf-8"))["qualified"])
        rep["candidate_runs"] = {k: {"seed": v["seed"], "final_present": self.exists(v["final"])} for k, v in self.cands.items()}
        rep["seeds"] = {"training": c["candidates"].get("training_seeds"), "blocks": c["blocks"],
                        "existing_pair_block": epq.get("block"), "defenders": c["role"].get("defender_seeds")}
        rep["role_menu"] = c["role"].get("menu")
        rep["role_menu_note"] = c["role"].get("menu_note") or c["role"].get("menu_source")
        st = self.status()
        rep["stage_status"] = st
        rep["next_stage"] = next((s for s in STAGES if not st.get(s)), "DONE")
        cnt = int(c["candidates"]["count_per_strategy"])
        sz = lambda b: (b[1] - b[0] + 1) if b else None  # noqa: E731
        rep["estimates"] = {"candidate_trainings": 2 * cnt, "candidate_steps_each": c["candidates"]["budget_steps"],
                            "existing_pair_episodes": 4 * sz(epq.get("block")) if epq.get("block") and epq["source"] == "fresh_block" else 0,
                            "d_select_episodes": (4 * cnt * sz(c["blocks"]["d_select"])) if c["blocks"].get("d_select") else None,
                            "d_gate_episodes": (4 * sz(c["blocks"]["d_gate"])) if c["blocks"].get("d_gate") else None,
                            "role_dev_episodes": (4 * sz(c["blocks"]["role_dev"]) * len(c["role"]["menu"])) if c["blocks"].get("role_dev") and c["role"].get("menu") else None,
                            "final_episodes": (4 * sz(c["blocks"]["final_confirmation"])) if c["blocks"].get("final_confirmation") else None}
        rep["output_dir"] = str(self.out.relative_to(self.root)).replace("\\", "/")
        rep["can_run"] = not rep["config_problems_for_run"]
        return rep


def _boot(x, a: float) -> list[float]:
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    b = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(axis=1)
    lo, hi = np.percentile(b, [100 * a / 2, 100 * (1 - a / 2)])
    return [float(lo), float(hi)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team-size", type=int, required=True, choices=(2, 4, 6))
    ap.add_argument("--run", action="store_true", help="execute (default: audit only, no side effects)")
    ap.add_argument("--audit", action="store_true", help="explicit audit (the default)")
    ap.add_argument("--workers", type=int, default=3, help="parallel jobs (execution only)")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pl = Pipeline(load_config(a.team_size), workers=a.workers)
    if not a.run:
        print(json.dumps(pl.audit(), indent=1))
        return 0
    return pl.run()


if __name__ == "__main__":
    raise SystemExit(main())
