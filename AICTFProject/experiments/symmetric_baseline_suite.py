r"""Symmetric-role baseline suite for one scale (2v2, 4v4 or 6v6): one resumable driver, one output folder.

    .venv\Scripts\python.exe experiments\symmetric_baseline_suite.py --team-size 6 --out 6v6\symmetric_baseline_suite
    .venv\Scripts\python.exe experiments\symmetric_baseline_suite.py --team-size 2 --check

Gate (refuses to start otherwise): TOP50_<N>V<N>_SYMMETRIC_OURS is SEALED and its two defenders are on disk
with the sealed hashes; with --core-sealed, that SEALED.json must name the same two hashes.

Stages (each recorded in <out>/STATE.json; a rerun continues after the last finished one):
  dataset      freeze SUITE_DISTILLATION_<N>V<N>_SYM_SPEC -> collector smoke -> collect 96 episodes/pole
               (Pole A: pi_A + pi_DA, Pole B: pi_B + pi_DB, CLOSEST_DEFENDS k=ceil(N/3)) -> audit GREEN
  students     freeze STANDARDIZED_<N>V<N>_SYM_SHARING_SPEC -> Share-Encoder, Fully Shared+z, Generalist (no z),
               KL teachers = repaired pi_A / pi_B, frozen recipe
  evals        freeze STANDARDIZED_<N>V<N>_SYM_SHARING_EVAL_SPEC -> the three students on the frozen top-50 seeds
  crossovers   Delta_A / Delta_B (and Delta_G) for no-role specialists, old asymmetric Ours, symmetric Ours,
               Share-Encoder, Fully Shared+z, Generalist -- paired on the same 50 seeds
  robustness   symmetric Ours under localization / motion / delay (medium tier); nominal = symmetric Ours
  margin       paired score-margin crossovers, from the rows already sealed (no new episodes)
  bundle       BASELINE_READOUT.md/.json + checkpoints + evaluation + provenance -> one zip

Diagnostic only (post-hoc, top-50 seeds selected on the old asymmetric system); no tuning after results.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import prepare_symmetric_baselines as P  # noqa: E402
from experiments.tqdm_loop import set_postfix, tqdm_iter  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
TOP = SD / "symmetric_role_top50"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe") if os.name == "nt" else sys.executable
STAGES = ("dataset", "students", "evals", "crossovers", "robustness", "margin", "bundle")
#: ~3 GB of commit per concurrent 6v6 evaluation process; a six-way launch once exhausted memory.
COMMIT_PER_PROC_GB = 6.0


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def free_commit_gb() -> float:
    """Available commit charge (Windows), else available physical memory."""
    if os.name == "nt":
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPageFile / 1e9
    import psutil
    return psutil.virtual_memory().available / 1e9


class Suite:
    def __init__(self, n: int, out: Path, core_sealed: Path | None, max_parallel: int):
        self.n, self.out, self.core_sealed, self.max_parallel = n, out, core_sealed, max_parallel
        self.fam = f"{n}v{n}"
        self.state_p, self.log_p = out / "STATE.json", out / "provenance" / "logs" / "suite.log"
        self.lab = P.labels(n)
        self.ours_label = f"TOP50_{n}V{n}_SYMMETRIC_OURS"

    # ------------------------------------------------------------ plumbing
    def log(self, msg: str) -> None:
        line = f"{now()} [{self.fam}] {msg}"
        print(line, flush=True)
        self.log_p.parent.mkdir(parents=True, exist_ok=True)
        with self.log_p.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def state(self, **kw) -> dict:
        s = json.loads(self.state_p.read_text(encoding="utf-8")) if self.state_p.is_file() else {"stages": {}}
        stage = kw.pop("stage", None)
        s.update(kw)
        if stage:
            s["stages"][stage] = now()
        s["updated_utc"] = now()
        self.state_p.parent.mkdir(parents=True, exist_ok=True)
        self.state_p.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
        return s

    def done(self, stage: str) -> bool:
        return self.state_p.is_file() and stage in json.loads(self.state_p.read_text(encoding="utf-8"))["stages"]

    def fail(self, msg: str):
        self.log(f"STOPPED: {msg}")
        self.state(status="STOPPED", reason=msg)
        raise SystemExit(1)

    def _env(self) -> dict:
        e = dict(os.environ)
        e["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
        return e

    def run(self, args: list[str], name: str) -> int:
        lp = self.out / "provenance" / "logs" / f"{name}.log"
        lp.parent.mkdir(parents=True, exist_ok=True)
        self.command(name, args)
        with lp.open("a", encoding="utf-8") as fh:
            return subprocess.run([PY, *args], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=self._env()).returncode

    def command(self, name: str, args: list[str]) -> None:
        cp = self.out / "provenance" / "commands.txt"
        cp.parent.mkdir(parents=True, exist_ok=True)
        with cp.open("a", encoding="utf-8") as fh:
            fh.write(f"{now()} {name}: python {' '.join(args)}\n")

    def run_many(self, jobs: list[tuple[str, list[str], Path]]) -> None:
        """Run (name, args, result) jobs, at most max_parallel at once, staggered, each started only with
        enough free commit; a job whose result exists is skipped. Sequential when max_parallel == 1."""
        pending = [j for j in jobs if not j[2].is_file()]
        live: dict[str, tuple[subprocess.Popen, Path]] = {}
        while pending or live:
            for name, (pr, res) in list(live.items()):
                if pr.poll() is not None:
                    del live[name]
                    self.log(f"{name}: exited {pr.returncode}")
                    if not res.is_file():
                        self.fail(f"{name} exited {pr.returncode} without {res.name} (rerun to resume)")
            if pending and len(live) < self.max_parallel and (not live or free_commit_gb() > COMMIT_PER_PROC_GB):
                name, args, res = pending.pop(0)
                lp = self.out / "provenance" / "logs" / f"{name}.log"
                lp.parent.mkdir(parents=True, exist_ok=True)
                self.command(name, args)
                fh = lp.open("a", encoding="utf-8")
                live[name] = (subprocess.Popen([PY, *args], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                               env=self._env()), res)
                self.log(f"{name}: started pid={live[name][0].pid}")
                time.sleep(90 if pending else 5)
            else:
                time.sleep(30)

    # ------------------------------------------------------------ gate
    def gate(self) -> list[str]:
        p = []
        try:
            ours = P.sealed_ours(self.n)
        except SystemExit as exc:
            return [str(exc)]
        if self.core_sealed is not None:
            if not self.core_sealed.is_file():
                p.append(f"{self.core_sealed} missing (Phase 1 not sealed)")
            else:
                s = json.loads(self.core_sealed.read_text(encoding="utf-8"))
                want = {"pi_DA": ours["pins"]["pi_DA"]["sha256"], "pi_DB": ours["pins"]["pi_DB"]["sha256"]}
                if s.get("status") != "SEALED" or s.get("defenders_sha256") != want or not s.get("crossover_completed"):
                    p.append(f"{self.core_sealed.name} does not seal these defenders / a completed crossover")
        from experiments import seed_registry as SR
        reg = {b["experiment_id"]: b for b in SR.load()["blocks"]}
        for _k, (eid, lo, hi) in P.blocks(self.n).items():
            b = reg.get(eid)
            if b is None or (b["lo"], b["hi"]) != (lo, hi):
                p.append(f"{eid} not registered at {lo}..{hi} (git pull / prepare --reserve)")
        if self.norole_rows() is None:
            p.append(f"no-role top-50 rows for {self.fam} missing")
        if not (TOP / f"top50_{self.fam}_asymmetric_ours_rows.csv").is_file():
            p.append(f"old asymmetric top-50 rows for {self.fam} missing")
        return p

    def norole_rows(self) -> Path | None:
        for f in (SD / f"top50_{self.fam}_norole_specialist_crossover_eval_rows.csv", TOP / f"top50_{self.fam}_norole_rows.csv"):
            if f.is_file():
                return f
        return None

    # ------------------------------------------------------------ stages
    def stage_dataset(self) -> None:
        n, T = self.n, P.TAG
        spec = P.collection_spec(n)
        self.log(f"collection spec frozen: {spec.name}")
        man = SD / f"SUITE_DISTILLATION_{n}V{n}_{T}_DATASET.json"
        base = ["experiments/collect_suite_distillation_states.py", "--team-size", str(n), "--dataset-tag", T, "--device", "cuda"]
        if not man.is_file():
            if not (SD / f"SUITE_DISTILLATION_{n}V{n}_{T}_DATASET_SMOKE.json").is_file():
                if self.run(base + ["--smoke"], "collect_smoke") != 0:
                    self.fail("collector smoke failed; see provenance/logs/collect_smoke.log")
                self.log("collector smoke passed")
            if self.run(base + ["--resume"], "collect") != 0 or not man.is_file():
                self.fail("collection failed (rerun to resume from its shards); see provenance/logs/collect.log")
        self.log(f"dataset frozen: {man.name}")
        aud = P.audit_record(n)
        if not aud.is_file():
            rc = self.run(["experiments/audit_suite_datasets_cross_scale.py", "--scales", f"{self.fam}_sym", "--write",
                           "--out", str(aud)], "audit")
            if rc != 0:
                self.fail(f"dataset audit RED; see provenance/logs/audit.log and {aud.name}")
        if json.loads(aud.read_text(encoding="utf-8")).get("verdict") != "GREEN":
            self.fail(f"{aud.name} is not GREEN")
        self.log("dataset audit GREEN")

    def student_dir(self, arm: str) -> Path:
        return SD / "suite_sharing_std" / f"{self.fam}_sym" / P.ARM_TAG[arm]

    def stage_students(self) -> None:
        spec = P.sharing_spec(self.n)
        self.log(f"sharing spec frozen: {spec.name}")
        for arm in P.ARMS:                                          # the spec's arm order
            if (self.student_dir(arm) / "STUDENT_FROZEN.json").is_file():
                continue
            base = ["experiments/run_suite_sharing_distillation.py", "--arm", arm, "--team-size", str(self.n),
                    "--spec-tag", P.TAG, "--device", "cuda"]
            if self.run(base + ["--preflight"], f"train_{arm}_preflight") != 0:
                self.fail(f"{arm} preflight failed; see provenance/logs/train_{arm}_preflight.log")
            if self.run(base, f"train_{arm}") != 0 or not (self.student_dir(arm) / "STUDENT_FROZEN.json").is_file():
                self.fail(f"{arm} training failed; see provenance/logs/train_{arm}.log")
            self.log(f"{arm} frozen")

    def eval_result(self, label: str, sharing: bool) -> Path:
        return SD / (f"{label}_CROSSOVER_EVAL_RESULT.json" if sharing else f"{label}_SPECIALIST_CROSSOVER_EVAL_RESULT.json")

    def stage_evals(self) -> None:
        spec = P.eval_spec(self.n)
        self.log(f"eval spec frozen: {spec.name}")
        jobs = []
        for arm in P.ARMS:
            res = self.eval_result(self.lab[arm], True)
            args = ["experiments/eval_suite_sharing_crossover.py", "--team-size", str(self.n), "--arm", arm,
                    "--spec-tag", P.TAG, "--device", "cuda"]
            partial = SD / f"{self.lab[arm].lower()}_crossover_eval_rows.PARTIAL.jsonl"
            if not res.is_file() and not partial.is_file() and self.run(args + ["--dry-run"], f"eval_{arm}_dryrun") != 0:
                self.fail(f"{arm} evaluation dry-run failed; see provenance/logs/eval_{arm}_dryrun.log")
            jobs.append((f"eval_{arm}", args + ["--resume"], res))
        self.run_many(jobs)

    def launch_args(self, key: str) -> list[str]:
        spec = json.loads((SD / f"STANDARDIZED_{self.n}V{self.n}_{P.TAG}_SHARING_EVAL_SPEC.json").read_text(encoding="utf-8"))
        toks = spec["LAUNCH"][key].split()
        assert toks[0].endswith("python.exe")
        return toks[1:]

    def stage_robustness(self) -> None:
        jobs = []
        for k in P.ROBUSTNESS:
            key = f"robust_{k.lower()}"
            label = self.lab[key]
            args = self.launch_args(key)
            res = self.eval_result(label, False)
            partial = SD / f"{label.lower()}_specialist_crossover_eval_rows.PARTIAL.jsonl"
            if not res.is_file() and not partial.is_file():
                dry = [a for a in args if a != "--resume"] + ["--dry-run"]
                if self.run(dry, f"{key}_dryrun") != 0:
                    self.fail(f"{key} dry-run failed; see provenance/logs/{key}_dryrun.log")
            jobs.append((key, args, res))
        self.run_many(jobs)

    # ------------------------------------------------------------ readouts
    def rows(self) -> dict[str, Path]:
        r = {"specialists_no_role": self.norole_rows(),
             "ours_asymmetric_old": TOP / f"top50_{self.fam}_asymmetric_ours_rows.csv",
             "ours_symmetric": SD / f"{self.ours_label.lower()}_specialist_crossover_eval_rows.csv",
             "share_encoder": SD / f"{self.lab['share_encoder'].lower()}_crossover_eval_rows.csv",
             "fully_shared_z": SD / f"{self.lab['fully_shared'].lower()}_crossover_eval_rows.csv",
             "generalist": SD / f"{self.lab['generalist'].lower()}_crossover_eval_rows.csv"}
        for k in P.ROBUSTNESS:
            r[f"ours_symmetric_{k.lower()}"] = SD / f"{self.lab[f'robust_{k.lower()}'].lower()}_specialist_crossover_eval_rows.csv"
        return r

    def seeds(self) -> list[int]:
        e = P._diag_entry(self.n)
        return sorted(int(s) for s in json.loads((ROOT / e["seed_ids_file"]).read_text(encoding="utf-8")))

    @staticmethod
    def _cells(path: Path, field: str) -> dict:
        """{(policy 'A'|'B', pole): {seed: value}} from specialist rows (policy pi_A/pi_B) or student rows (z 0/1)."""
        out: dict = {}
        with path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                pol = row["policy"][-1] if "policy" in row else ("A" if row["z"] == "0" else "B")
                out.setdefault((pol, row["pole"]), {})[int(row["seed"])] = float(row[field])
        return out

    def _stat(self, x) -> dict:
        import numpy as np
        from experiments.eval_hog_psp_v3 import _mean_ci
        x = np.asarray(x, dtype=np.float64)
        ci = _mean_ci(x)
        return {"mean": float(x.mean()), "std": float(x.std(ddof=1)), "lcb95": ci["lcb95"], "ucb95": ci["ucb95"], "n": int(x.size)}

    def crossover_table(self, field: str) -> dict:
        import numpy as np
        seeds, out = self.seeds(), {}
        sym = self._cells(self.rows()["ours_symmetric"], field)
        for name, path in self.rows().items():
            if name.startswith("ours_symmetric_"):              # robustness rows: robustness_table
                continue
            if path is None or not path.is_file():
                self.fail(f"rows for {name} missing: {path}")
            c = self._cells(path, field)
            v = {k: np.array([d[s] for s in seeds]) for k, d in c.items()}
            if name == "generalist":
                g_a, g_b = v[("A", "A")], v[("A", "B")]                    # one policy (z=0) on both poles
                ours_a = np.array([sym[("A", "A")][s] for s in seeds])
                ours_b = np.array([sym[("B", "B")][s] for s in seeds])
                out[name] = {"V_on_A": self._stat(g_a), "V_on_B": self._stat(g_b),
                             "Delta_G_A_vs_ours_symmetric": self._stat(ours_a - g_a),
                             "Delta_G_B_vs_ours_symmetric": self._stat(ours_b - g_b)}
                continue
            out[name] = {"A_on_A": float(v[("A", "A")].mean()), "B_on_A": float(v[("B", "A")].mean()),
                         "A_on_B": float(v[("A", "B")].mean()), "B_on_B": float(v[("B", "B")].mean()),
                         "Delta_A": self._stat(v[("A", "A")] - v[("B", "A")]),
                         "Delta_B": self._stat(v[("B", "B")] - v[("A", "B")])}
        return out

    def robustness_table(self) -> dict:
        import numpy as np
        seeds, rows = self.seeds(), self.rows()
        nom = self._cells(rows["ours_symmetric"], "win")
        d = lambda c, s: (c[("A", "A")][s] - c[("B", "A")][s], c[("B", "B")][s] - c[("A", "B")][s])  # noqa: E731
        out = {}
        for k in P.ROBUSTNESS:
            c = self._cells(rows[f"ours_symmetric_{k.lower()}"], "win")
            da = np.array([d(c, s)[0] - d(nom, s)[0] for s in seeds])
            db = np.array([d(c, s)[1] - d(nom, s)[1] for s in seeds])
            out[k.lower()] = {"severity": P.ROBUSTNESS_SEVERITY,
                              "Delta_A": self._stat([d(c, s)[0] for s in seeds]),
                              "Delta_B": self._stat([d(c, s)[1] for s in seeds]),
                              "change_vs_nominal_Delta_A": self._stat(da), "change_vs_nominal_Delta_B": self._stat(db),
                              "own_pole_WR_A": float(np.mean([c[("A", "A")][s] for s in seeds])),
                              "own_pole_WR_B": float(np.mean([c[("B", "B")][s] for s in seeds]))}
        return out

    def write_readout(self, with_robustness: bool = True) -> dict:
        ro = {"record": f"SYMMETRIC_BASELINE_READOUT_{self.fam.upper()}", "utc": now(), "team_size": self.n,
              "k_defend": P.k_sym(self.n), "seeds": {"n": len(self.seeds()), "file": P._diag_entry(self.n)["seed_ids_file"]},
              "classification": "DIAGNOSTIC, post-hoc on the frozen Ours top-50 seeds; not confirmatory",
              "caveat": "The 50 seeds were selected on the old asymmetric Ours, which biases every comparison toward it "
                        "(and toward old pi_B on Pole B). Mean effects only; no tuning after results.",
              "crossovers_win": self.crossover_table("win"),
              "robustness_ours_symmetric": self.robustness_table() if with_robustness else None,
              "score_margin": self.crossover_table("margin")}
        f = lambda s: f"{s['mean']:+.3f} ± {s['std']:.3f}"  # noqa: E731
        md = [f"# {self.fam} symmetric-role baselines (top-50 seeds, post-hoc diagnostic)", "",
              f"k = {ro['k_defend']}, n = {ro['seeds']['n']} seeds. {ro['caveat']}", "",
              "## Crossover (win rate)", "", "| System | A@A | B@A | A@B | B@B | Δ_A | Δ_B |", "|---|---|---|---|---|---|---|"]
        for name, t in ro["crossovers_win"].items():
            if name == "generalist":
                continue
            md.append(f"| {name} | {t['A_on_A']:.3f} | {t['B_on_A']:.3f} | {t['A_on_B']:.3f} | {t['B_on_B']:.3f} | "
                      f"{f(t['Delta_A'])} | {f(t['Delta_B'])} |")
        g = ro["crossovers_win"]["generalist"]
        md += ["", f"Generalist (no z): V on A {g['V_on_A']['mean']:.3f}, V on B {g['V_on_B']['mean']:.3f}; "
                   f"Δ_G vs symmetric Ours: A {f(g['Delta_G_A_vs_ours_symmetric'])}, B {f(g['Delta_G_B_vs_ours_symmetric'])}",
               "", "## Robustness of symmetric Ours (medium tier; change is paired vs nominal)", "",
               "| Perturbation | Δ_A | Δ_B | change Δ_A | change Δ_B |", "|---|---|---|---|---|"]
        for k, t in (ro["robustness_ours_symmetric"] or {}).items():
            md.append(f"| {k} | {f(t['Delta_A'])} | {f(t['Delta_B'])} | {f(t['change_vs_nominal_Delta_A'])} | "
                      f"{f(t['change_vs_nominal_Delta_B'])} |")
        md += ["", "## Score margin (paired crossover on blue − red)", "", "| System | Δ_A margin | Δ_B margin |", "|---|---|---|"]
        for name, t in ro["score_margin"].items():
            if name != "generalist":
                md.append(f"| {name} | {f(t['Delta_A'])} | {f(t['Delta_B'])} |")
        ro["table_markdown"] = "\n".join(md)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "BASELINE_READOUT.json").write_text(json.dumps(ro, indent=2) + "\n", encoding="utf-8")
        (self.out / "BASELINE_READOUT.md").write_text(ro["table_markdown"] + "\n", encoding="utf-8")
        return ro

    # ------------------------------------------------------------ bundle
    def bundle(self, zip_path: Path) -> Path:
        n, T, o = self.n, P.TAG, self.out
        def cp(src: Path, dst_dir: Path):
            if src is not None and src.is_file():
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst_dir / src.name)
        for arm in P.ARMS:
            for f in self.student_dir(arm).rglob("*"):
                if f.is_file():
                    cp(f, o / "checkpoints" / P.ARM_TAG[arm] / f.parent.relative_to(self.student_dir(arm)))
        for f in (SD / f"SUITE_DISTILLATION_{n}V{n}_{T}_DATASET.json", P.audit_record(n)):
            cp(f, o / "dataset")
        labels = [self.ours_label, *self.lab.values()]
        for lab in labels:
            for f in SD.glob(f"{lab}_*.json"):
                cp(f, o / "evaluation")
            for f in SD.glob(f"{lab.lower()}_*rows.csv"):
                cp(f, o / "evaluation")
        for name, f in self.rows().items():
            cp(f, o / "evaluation" / "rows_used")
        for f in (SD / f"SUITE_DISTILLATION_{n}V{n}_{T}_SPEC.json", SD / f"STANDARDIZED_{n}V{n}_{T}_SHARING_SPEC.json",
                  SD / f"STANDARDIZED_{n}V{n}_{T}_SHARING_EVAL_SPEC.json", SD / "SYMMETRIC_ROLE_TOP50_DIAGNOSTIC_SPEC.json",
                  SD / "GENERALIST_DEFINITION_V1.json", ROOT / "artifacts" / "SEED_REGISTRY.json"):
            cp(f, o / "provenance" / "specs_and_configs")
        cp(ROOT / P._diag_entry(n)["seed_ids_file"], o / "provenance" / "seed_lists")
        git = lambda *a: subprocess.run(["git", "-C", str(ROOT.parent), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
        (o / "provenance" / "git_commit.json").write_text(json.dumps(
            {"head": git("rev-parse", "HEAD"), "scientific_status": git("status", "--short", "--", "AICTFProject/experiments",
                                                                         "AICTFProject/rl", "AICTFProject/gpu_env"),
             "host": platform.node(), "python": sys.version.split()[0], "utc": now()}, indent=2) + "\n", encoding="utf-8")
        hashes = {str(f.relative_to(o)).replace("\\", "/"): sha(f) for f in sorted(o.rglob("*"))
                  if f.is_file() and f.name != "HASHES.json"}
        (o / "provenance" / "HASHES.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
        (o / "README.txt").write_text(
            f"{self.fam} symmetric-role baseline suite (post-hoc diagnostic on the frozen Ours top-50 seeds)\n\n"
            "BASELINE_READOUT.md / .json  crossovers, Generalist, robustness, score margin\n"
            "checkpoints/                 Generalist (no z), Share-Encoder, Fully Shared+z (weights, freeze records, metrics)\n"
            "dataset/                     symmetric dataset manifest + audit (state shards stay on the collecting PC)\n"
            "evaluation/                  sealed results, audits, run states, per-seed rows (rows_used = exactly what the readout read)\n"
            "provenance/                  frozen specs, seed list, seed registry snapshot, commands, logs, git, HASHES.json\n",
            encoding="utf-8")
        if zip_path.exists():
            zip_path.unlink()
        z = shutil.make_archive(str(zip_path.with_suffix("")), "zip", o)
        return Path(z)

    # ------------------------------------------------------------ driver
    def main(self, zip_path: Path) -> int:
        problems = self.gate()
        if problems:
            self.fail("gate: " + "; ".join(problems))
        self.state(status="RUNNING", pid=os.getpid())
        self.log("symmetric baseline suite started")
        work = {"dataset": self.stage_dataset, "students": self.stage_students, "evals": self.stage_evals,
                "robustness": self.stage_robustness}
        remaining = [st for st in STAGES if not self.done(st)]
        overall = tqdm_iter(remaining, desc=f"{self.fam} suite OVERALL", total=len(remaining), unit="stage")
        for st in overall:
            set_postfix(overall, st)
            self.log(f"stage start: {st}  ({overall.n + 1}/{len(remaining)})")
            if st == "crossovers":                     # baselines' crossovers; readout without robustness yet
                ro = self.write_readout(with_robustness=False)
                self.log("\n" + ro["table_markdown"])
            elif st == "margin":                       # final readout: crossovers + robustness + score margin
                ro = self.write_readout()
                self.log("\n" + ro["table_markdown"])
            elif st == "bundle":
                z = self.bundle(zip_path)
                self.log(f"bundle: {self.out} and {z} ({z.stat().st_size / 1e6:.0f} MB)")
            else:
                work[st]()
            self.state(stage=st)
            self.log(f"stage done: {st}")
        self.state(status="DONE")
        self.log(f"DONE -- send {zip_path}")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team-size", type=int, required=True, choices=(2, 4, 6))
    ap.add_argument("--out", default=None, help="output folder (default symmetric_baselines/<N>v<N>)")
    ap.add_argument("--zip", default=None, help="zip path (default <out>/../symmetric_baselines_<N>v<N>.zip)")
    ap.add_argument("--core-sealed", default=None, help="Phase-1 SEALED.json that must name the same defenders")
    ap.add_argument("--max-parallel", type=int, default=2, help="concurrent evaluation processes (memory-checked)")
    ap.add_argument("--check", action="store_true", help="evaluate the gate only; change nothing")
    a = ap.parse_args()
    n = a.team_size
    out = (ROOT / a.out) if a.out else ROOT / "symmetric_baselines" / f"{n}v{n}"
    zp = (ROOT / a.zip) if a.zip else out.parent / f"symmetric_baselines_{n}v{n}.zip"
    s = Suite(n, out, (ROOT / a.core_sealed) if a.core_sealed else None, max(1, a.max_parallel))
    if a.check:
        p = s.gate()
        print("GATE PASS" if not p else "GATE CLOSED:\n  " + "\n  ".join(p))
        return 0 if not p else 1
    return s.main(zp)


if __name__ == "__main__":
    raise SystemExit(main())
