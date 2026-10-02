r"""6v6 symmetric-role pipeline, school PC: one launcher, two independent bundles.

    cd <repo>; git pull; cd AICTFProject
    .venv\Scripts\python.exe 6v6\run_symmetric_6v6.py --check      (no side effects; must print ALL CHECKS PASS)
    powershell -ExecutionPolicy Bypass -File 6v6\run_symmetric_6v6.ps1   (detached; safe to close the window)

PHASE 1 -- symmetric_core (SYMMETRIC_ROLE_TOP50_DIAGNOSTIC_SPEC.json; k = ceil(6/3) = 2 on BOTH sides)
   1 check     everything below, no side effects
   2 smoke     2,048-step B-side defender run; fails fast
   3 train     pi_DA <- repaired pi_A and pi_DB <- repaired pi_B, 200k steps each, identical recipe
               (seeds 26100001 / 26200001; resume from the latest periodic checkpoint)
   4 crossover TOP50_6V6_SYMMETRIC_OURS on the frozen 50 seeds
               A: ATTACK -> frozen pi_A, DEFEND -> pi_DA     B: ATTACK -> frozen pi_B, DEFEND -> pi_DB
   5 seal      READOUT (no-role vs old asymmetric k=1 vs new symmetric k=2) + SEALED.json
               -> 6v6\symmetric_core\ and 6v6\symmetric_results.zip
   Phase 1 never waits for Phase 2: its zip exists as soon as stage 5 finishes.

PHASE 2 -- symmetric_baseline_suite (experiments/symmetric_baseline_suite.py; the same code runs 2v2/4v4)
   gate: symmetric_core\SEALED.json exists, names the defenders the sealed crossover used, crossover completed
   6 dataset   symmetric 6v6 teacher-state set (Pole A: pi_A + pi_DA, Pole B: pi_B + pi_DB, 96 episodes/pole),
               KL teachers stay repaired pi_A / pi_B; audit GREEN
   7 students  Generalist (no z), Share-Encoder, Fully Shared+z on that dataset, frozen recipe
   8 evals     the three students on the same frozen 50 seeds
   9 crossovers Delta_A / Delta_B for specialists (no role), old asymmetric Ours, symmetric Ours, the students;
               Delta_G for the Generalist
  10 robustness symmetric Ours under localization / motion / delay (medium tier)
  11 margin     paired score-margin crossovers
  12 bundle     6v6\symmetric_baseline_suite\ and 6v6\symmetric_baselines.zip

Re-running the same command after a restart continues from the last finished stage.
Phases run one after the other, never in parallel on one GPU.
Does NOT touch: specialists, the old 6v6 defender or students, any sealed result.
"""
from __future__ import annotations

import argparse
import csv
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

PROJ = Path(__file__).resolve().parents[1]                 # AICTFProject
REPO = PROJ.parent
SD = PROJ / "artifacts" / "strategic_demand" / "sppo"
TOP = SD / "symmetric_role_top50"
OUTDIR = PROJ / "6v6" / "symmetric_core"
CORE_ZIP = PROJ / "6v6" / "symmetric_results.zip"
SEALED = OUTDIR / "SEALED.json"
SUITE_OUT, SUITE_ZIP = "6v6/symmetric_baseline_suite", "6v6/symmetric_baselines.zip"
STATE = PROJ / "6v6" / "symmetric_6v6_state.json"
LOG = PROJ / "6v6" / "symmetric_6v6.log"
PY = str(PROJ / ".venv" / "Scripts" / "python.exe")
SPEC = "artifacts/strategic_demand/sppo/SYMMETRIC_ROLE_TOP50_DIAGNOSTIC_SPEC.json"
SEEDS_FILE = "artifacts/strategic_demand/sppo/symmetric_role_top50/6v6_ours_top50_seed_ids.json"
REQUIRED_COMMIT = "3b7eedfd"            # symmetric role support (B defender, B-side splice, seed lists)

A = "artifacts/scale_6v6_specialists/pi_A_specialist_6v6_c2_entity_repair/ckpts/final_pi_A_specialist_6v6_c2_entity_repair.zip"
B = "artifacts/scale_6v6_specialists/pi_B_specialist_6v6_c2_entity_repair/ckpts/final_pi_B_specialist_6v6_c2_entity_repair.zip"
SHA = {A: "3298000480acd899e715eb78312938a9d20610a1eb0bcf73d382fd268577009e",
       B: "fc0043235d3abc23f87f5157a46920921f5eced0e9aa252347084d2b500258b5"}
K = 2
TEACHER = ["--defend-teacher-lambda", "0.1", "--defend-teacher-lambda-end", "0.0", "--defend-teacher-decay-start-step", "50000",
           "--defend-teacher-decay-end-step", "150000", "--defend-teacher-cadence", "4"]
RUNS = {  # identical recipe; only the policy, its specialist, seed and suffix differ
    "A": {"seed": 26100001, "eid": "SYMMETRIC_ROLE_TOP50_6V6_A_DEFENDER_TRAINING", "spec_ck": A, "suffix": "_sym_A_k2_top50"},
    "B": {"seed": 26200001, "eid": "SYMMETRIC_ROLE_TOP50_6V6_B_DEFENDER_TRAINING", "spec_ck": B, "suffix": "_sym_B_top50"},
}
for _p, _r in RUNS.items():
    _r["run_dir"] = f"artifacts/scale_6v6_specialists/pi_{_p}_specialist_6v6{_r['suffix']}"
    _r["final"] = f"{_r['run_dir']}/ckpts/final_pi_{_p}_specialist_6v6{_r['suffix']}.zip"
EVAL_LABEL = "TOP50_6V6_SYMMETRIC_OURS"
EVAL_REG = "STANDARDIZED_6V6_SEPARATED_EVAL"
OLD_ROWS = {"old_asymmetric_ours": TOP / "top50_6v6_asymmetric_ours_rows.csv", "no_role": TOP / "top50_6v6_norole_rows.csv"}


# ------------------------------------------------------------------ plumbing
def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"{now()} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def state(**kw) -> dict:
    s = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {"steps": {}}
    s.update({k: v for k, v in kw.items() if k != "step"})
    if "step" in kw:
        s["steps"][kw["step"]] = now()
    s["updated_utc"] = now()
    STATE.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
    return s


def done(step: str) -> bool:
    return STATE.is_file() and step in json.loads(STATE.read_text(encoding="utf-8")).get("steps", {})


def fail(msg: str) -> None:
    log(f"STOPPED: {msg}")
    state(status="STOPPED", reason=msg)
    sys.exit(1)


def sha(rel: str) -> str:
    h = hashlib.sha256()
    with (PROJ / rel).open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()


def env() -> dict:
    e = dict(os.environ)
    e["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"   # a console-close event killed 6v6 runs once (Intel Fortran runtime)
    return e


def train_args(pol: str, steps: int, seed: int, eid: str | None, suffix: str, smoke: bool) -> list[str]:
    r = RUNS[pol]
    a = ["experiments/train_specialist_scale.py", "--team-size", "6", "--policy", pol, "--seed", str(seed), "--device", "cuda",
         "--total-timesteps", str(steps), "--entity-repair-enabled", "--entity-hidden-dim", "32", "--role-conditioning-enabled",
         "--role-hold-ticks", "8", "--role-fixed-for-episode", "--role-k-defend", str(K), "--split-attack-defend-enabled",
         "--split-attack-defend-frozen-ckpt", r["spec_ck"], "--split-attack-defend-frozen-ckpt-sha256", SHA[r["spec_ck"]],
         "--load-path", r["spec_ck"], *TEACHER, "--run-label-suffix", suffix]
    if pol != "A":
        a += ["--symmetric-role-spec", SPEC]
    a += ["--smoke"] if smoke else ["--experiment-id", eid]
    return a


def eval_args() -> list[str]:
    return ["experiments/eval_specialist_crossover_scaled.py", "--team-size", "6", "--spec", SPEC, "--post-hoc-ablation-spec", SPEC,
            "--seed-list", SEEDS_FILE, "--registry-experiment-id", EVAL_REG, "--label", EVAL_LABEL, "--device", "cuda",
            "--pi-a-path", RUNS["A"]["final"], "--pi-b-path", RUNS["B"]["final"], "--role-fixed-for-episode", "--role-k-defend", str(K),
            "--frozen-attack-path", A, "--frozen-attack-path-sha256", SHA[A],
            "--frozen-attack-path-b", B, "--frozen-attack-path-b-sha256", SHA[B]]


# ------------------------------------------------------------------ checks
def check() -> list[str]:
    """Everything the run needs, with no side effects. Returns the list of problems."""
    p = []
    head = git("rev-parse", "--short", "HEAD")
    if subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", REQUIRED_COMMIT, "HEAD"]).returncode != 0:
        p.append(f"code too old: HEAD {head} does not contain {REQUIRED_COMMIT} -- run `git pull`")
    for rel, want in SHA.items():
        if not (PROJ / rel).is_file():
            p.append(f"missing checkpoint {rel}")
        elif sha(rel) != want:
            p.append(f"checkpoint {rel} sha256 differs from its seal")
    spec_p = PROJ / SPEC
    if not spec_p.is_file():
        p.append(f"missing {SPEC} -- run `git pull`")
    else:
        s = json.loads(spec_p.read_text(encoding="utf-8"))
        if not str(s.get("status", "")).startswith("FROZEN"):
            p.append("spec not frozen")
        auth = {(e["policy"], e["role_k_defend"], e["run_label_suffix"]) for e in s.get("TRAINING_AUTHORIZED", []) if e["team_size"] == 6}
        if ("B", K, RUNS["B"]["suffix"]) not in auth:
            p.append("spec does not authorize the 6v6 B defender")
        ent = (s.get("POST_HOC_MATCHED_ROLE_ABLATIONS") or {}).get(EVAL_LABEL) or {}
        seeds = json.loads((PROJ / SEEDS_FILE).read_text(encoding="utf-8")) if (PROJ / SEEDS_FILE).is_file() else None
        if seeds is None:
            p.append(f"missing {SEEDS_FILE}")
        elif sorted(seeds) != sorted(ent.get("seed_ids") or []) or len(seeds) != 50:
            p.append("frozen 6v6 seed list disagrees with the spec")
    try:
        sys.path.insert(0, str(PROJ))
        from experiments import seed_registry as SR
        reg = {b["experiment_id"]: b for b in SR.load()["blocks"]}
        for pol, r in RUNS.items():
            b = reg.get(r["eid"])
            if b is None or b["lo"] != r["seed"]:
                p.append(f"seed block {r['eid']} not registered at {r['seed']} -- run `git pull`")
            elif b["status"] != "RESERVED" and not (PROJ / r["final"]).is_file():
                p.append(f"seed block {r['eid']} is {b['status']} but its final checkpoint is missing")
        if reg.get(EVAL_REG, {}).get("status") != "SPENT":
            p.append(f"{EVAL_REG} must be SPENT (post-hoc reuse)")
        # Phase 2 prerequisites: the generic suite, its pre-registered seed blocks, the parent specs it derives from
        from experiments import prepare_symmetric_baselines as PSB
        from experiments import symmetric_baseline_suite  # noqa: F401  (importable = Phase 2 code present)
        for _k, (eid, lo, hi) in PSB.blocks(6).items():
            b = reg.get(eid)
            if b is None or (b["lo"], b["hi"]) != (lo, hi):
                p.append(f"Phase 2 seed block {eid} not registered at {lo}..{hi} -- run `git pull`")
        for f in (PSB.COLLECTION_PARENT[6], "STANDARDIZED_6V6_SHARING_SPEC.json", "STANDARDIZED_6V6_SHARING_EVAL_SPEC.json",
                  "GENERALIST_DEFINITION_V1.json", "DEPLOYMENT_ROBUSTNESS_SPEC.json"):
            if not (SD / f).is_file():
                p.append(f"Phase 2 needs {f} -- run `git pull`")
    except Exception as exc:                              # noqa: BLE001
        p.append(f"registry unreadable: {exc}")
    for f in OLD_ROWS.values():
        if not f.is_file():
            p.append(f"missing {f.relative_to(PROJ)} (old results for the comparison) -- run `git pull`")
    if not (SD / "STANDARDIZED_6V6_SPLIT_K1_CONFIRMATORY_SPECIALIST_CROSSOVER_EVAL_RESULT.json").is_file():
        p.append("missing the 6v6 confirmatory record the evaluation is matched to")
    for pol, r in RUNS.items():
        if (PROJ / r["run_dir"]).is_dir() and not done(f"train_{pol}"):
            p.append(f"note (not an error if resuming): {r['run_dir']} already exists")
    try:
        import torch
        if not torch.cuda.is_available():
            p.append("CUDA not available")
    except Exception as exc:                              # noqa: BLE001
        p.append(f"torch import failed: {exc}")
    free_gb = shutil.disk_usage(PROJ).free / 1e9
    if free_gb < 10:
        p.append(f"only {free_gb:.1f} GB free disk (need ~10)")
    return [x for x in p if not x.startswith("note")], [x for x in p if x.startswith("note")]


# ------------------------------------------------------------------ steps
def run_logged(args: list[str], name: str) -> int:
    with (PROJ / "6v6" / f"symmetric_{name}.log").open("a", encoding="utf-8") as out:
        return subprocess.run([PY, *args], cwd=PROJ, stdout=out, stderr=subprocess.STDOUT, env=env()).returncode


def latest_periodic(run_dir: str) -> str | None:
    cks = sorted((PROJ / run_dir / "ckpts").glob("ckpt_*.zip"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    return str(cks[-1].relative_to(PROJ)) if cks else None


def train_both() -> None:
    procs = {}
    for pol, r in RUNS.items():
        if (PROJ / r["final"]).is_file():
            log(f"pi_D{pol}: final checkpoint already exists")
            continue
        args = train_args(pol, 200_000, r["seed"], r["eid"], r["suffix"], smoke=False)
        resume = latest_periodic(r["run_dir"])
        if resume:
            args += ["--resume", resume]
            log(f"pi_D{pol}: resuming from {resume}")
        out = (PROJ / "6v6" / f"symmetric_train_{pol}.log").open("a", encoding="utf-8")
        procs[pol] = subprocess.Popen([PY, *args], cwd=PROJ, stdout=out, stderr=subprocess.STDOUT, env=env())
        log(f"pi_D{pol}: training started pid={procs[pol].pid}")
        time.sleep(120)                                   # stagger the two starts
    for pol, pr in procs.items():
        rc = pr.wait()
        log(f"pi_D{pol}: training exited {rc}")
    for pol, r in RUNS.items():
        if not (PROJ / r["final"]).is_file():
            fail(f"pi_D{pol} training ended without {r['final']} (rerun this script to resume)")
        state(**{f"pi_D{pol}_sha256": sha(r["final"])})


def readout() -> dict:
    """Paired comparison on the frozen 50 seeds: four win rates, Delta_A, Delta_B, sample std."""
    import numpy as np
    seeds = sorted(json.loads((PROJ / SEEDS_FILE).read_text(encoding="utf-8")))

    def table(path: Path) -> dict:
        by = {}
        for row in csv.DictReader(path.open(encoding="utf-8")):
            by.setdefault((row["policy"], row["pole"]), {})[int(row["seed"])] = float(row["win"])
        v = {k: np.array([c[s] for s in seeds]) for k, c in by.items()}
        da, db = v[("pi_A", "A")] - v[("pi_B", "A")], v[("pi_B", "B")] - v[("pi_A", "B")]
        return {"n": len(seeds), "WR_A_on_A": float(v[("pi_A", "A")].mean()), "WR_B_on_A": float(v[("pi_B", "A")].mean()),
                "WR_A_on_B": float(v[("pi_A", "B")].mean()), "WR_B_on_B": float(v[("pi_B", "B")].mean()),
                "Delta_A_mean": float(da.mean()), "Delta_A_std": float(da.std(ddof=1)),
                "Delta_B_mean": float(db.mean()), "Delta_B_std": float(db.std(ddof=1)), "source_rows": path.name}

    new_rows = SD / f"{EVAL_LABEL.lower()}_specialist_crossover_eval_rows.csv"
    res = {"record": "SYMMETRIC_ROLE_TOP50_6V6_READOUT", "utc": now(), "seeds": "frozen Ours top-50 (post-hoc; development only)",
           "systems": {"no_role_specialists": table(OLD_ROWS["no_role"]),
                       "old_asymmetric_ours_k1": table(OLD_ROWS["old_asymmetric_ours"]),
                       "new_symmetric_ours_k2": table(new_rows)},
           "caveat": "Seeds were selected on the old asymmetric Ours (pi_A won A and pi_B won B on all 50), so old pi_B's Pole-B win rate "
                     "is 1.0 by construction; a new B can only tie or fall short on Pole B here. Read A-side effects and A-vs-B gaps. "
                     "Not confirmatory; does not replace any n=128 result."}
    lines = ["| 6v6, 50 frozen seeds | WR A@A | WR B@A | WR A@B | WR B@B | Delta_A (mean +- std) | Delta_B (mean +- std) |", "|---|---|---|---|---|---|---|"]
    for name, t in res["systems"].items():
        lines.append(f"| {name} | {t['WR_A_on_A']:.3f} | {t['WR_B_on_A']:.3f} | {t['WR_A_on_B']:.3f} | {t['WR_B_on_B']:.3f} | "
                     f"{t['Delta_A_mean']:+.3f} +- {t['Delta_A_std']:.3f} | {t['Delta_B_mean']:+.3f} +- {t['Delta_B_std']:.3f} |")
    res["table_markdown"] = "\n".join(lines)
    return res


def bundle(ro: dict) -> None:
    """Phase 1 output: 6v6/symmetric_core/ (sealed by SEALED.json, written last) + symmetric_results.zip."""
    if SEALED.is_file():
        return
    if OUTDIR.exists():
        shutil.rmtree(OUTDIR)
    defenders = {}
    for pol, name in (("A", "pi_DA_k2"), ("B", "pi_DB_k2")):     # final weights + every training record
        r = RUNS[pol]
        dst = OUTDIR / "checkpoints" / name
        shutil.copytree(PROJ / r["run_dir"], dst / "training_run", ignore=shutil.ignore_patterns("ckpt_*.zip"))
        shutil.copy2(PROJ / r["final"], dst / Path(r["final"]).name)
        defenders[f"pi_D{pol}"] = sha(r["final"])
    ev = OUTDIR / "evaluation"
    ev.mkdir(parents=True)
    for f in SD.glob(f"{EVAL_LABEL}_*.json"):
        shutil.copy2(f, ev / f.name)
    new_rows = SD / f"{EVAL_LABEL.lower()}_specialist_crossover_eval_rows.csv"
    shutil.copy2(new_rows, ev)
    for name, f in (("no_role", OLD_ROWS["no_role"]), ("old_asymmetric", OLD_ROWS["old_asymmetric_ours"]),
                    ("new_symmetric", new_rows)):
        (OUTDIR / "comparison" / name).mkdir(parents=True)
        shutil.copy2(f, OUTDIR / "comparison" / name / f.name)
    (OUTDIR / "READOUT.json").write_text(json.dumps(ro, indent=2) + "\n", encoding="utf-8")
    (OUTDIR / "READOUT.md").write_text("# 6v6 symmetric-role diagnostic (top-50, post-hoc)\n\n" + ro["table_markdown"] + "\n\n" + ro["caveat"] + "\n", encoding="utf-8")
    prov = OUTDIR / "provenance"
    for sub, files in (("frozen_spec", [PROJ / SPEC, PROJ / "artifacts" / "SEED_REGISTRY.json"]),
                       ("seed_list", [PROJ / SEEDS_FILE]), ("logs", [STATE, LOG, *(PROJ / "6v6").glob("symmetric_*.log")])):
        (prov / sub).mkdir(parents=True, exist_ok=True)
        for f in files:
            if f.is_file():
                shutil.copy2(f, prov / sub / f.name)
    (prov / "commands").mkdir()
    (prov / "commands" / "commands.json").write_text(json.dumps({
        "smoke_B": train_args("B", 2048, 99902002, None, RUNS["B"]["suffix"], smoke=True),
        "train_A": train_args("A", 200_000, RUNS["A"]["seed"], RUNS["A"]["eid"], RUNS["A"]["suffix"], smoke=False),
        "train_B": train_args("B", 200_000, RUNS["B"]["seed"], RUNS["B"]["eid"], RUNS["B"]["suffix"], smoke=False),
        "crossover": eval_args() + ["--resume"]}, indent=2) + "\n", encoding="utf-8")
    (prov / "git_commit").mkdir()
    (prov / "git_commit" / "MACHINE.json").write_text(json.dumps({"git_head": git("rev-parse", "HEAD"), "git_status_short": git("status", "--short", "--", "AICTFProject/experiments", "AICTFProject/rl", "AICTFProject/gpu_env"),
                                                   "host": platform.node(), "python": sys.version.split()[0], "utc": now()}, indent=2) + "\n", encoding="utf-8")
    (OUTDIR / "README.txt").write_text(
        "6v6 symmetric-role core result (Phase 1) -- everything from the school-PC run.\n\n"
        "READOUT.md / READOUT.json   no-role vs old asymmetric (k=1) vs new symmetric (k=2), same 50 seeds, paired\n"
        "SEALED.json                 seal: defender hashes + the sealed crossover result\n"
        "checkpoints/pi_DA_k2, pi_DB_k2   final weights + each run's configs, manifests, metrics, episode rows\n"
        "evaluation/                 sealed TOP50_6V6_SYMMETRIC_OURS result, audit, run state, per-seed rows\n"
        "comparison/                 the per-seed rows of each system in the readout\n"
        "provenance/                 frozen spec, seed list, seed registry snapshot, commands, logs, git HEAD\n\n"
        "Send 6v6/symmetric_results.zip as one file.\n", encoding="utf-8")
    res = SD / f"{EVAL_LABEL}_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
    rec = json.loads(res.read_text(encoding="utf-8"))
    sealed = {"record": "SYMMETRIC_CORE_6V6_SEALED", "status": "SEALED", "utc": now(),
              "defenders_sha256": {"pi_DA": defenders["pi_DA"], "pi_DB": defenders["pi_DB"]},
              "crossover_completed": rec.get("status") == "SEALED"
                                     and rec["checkpoints"] == {"pi_A": defenders["pi_DA"], "pi_B": defenders["pi_DB"]},
              "crossover_result": res.name, "crossover_result_sha256": sha(str(res.relative_to(PROJ))),
              "k_defend": K, "seeds": SEEDS_FILE}
    if not sealed["crossover_completed"]:
        fail("the sealed crossover does not match the trained defenders; not sealing")
    SEALED.write_text(json.dumps(sealed, indent=2) + "\n", encoding="utf-8")      # last: the seal
    if CORE_ZIP.exists():
        CORE_ZIP.unlink()
    zp = shutil.make_archive(str(CORE_ZIP.with_suffix("")), "zip", OUTDIR)
    log(f"Phase 1 sealed: {OUTDIR} and {zp} ({Path(zp).stat().st_size / 1e6:.0f} MB)")


def phase2() -> int:
    """Stages 6-12: the generic symmetric baseline suite, gated on SEALED.json (checked again inside)."""
    args = ["experiments/symmetric_baseline_suite.py", "--team-size", "6", "--out", SUITE_OUT, "--zip", SUITE_ZIP,
            "--core-sealed", str(SEALED.relative_to(PROJ))]
    log("Phase 2 (symmetric_baseline_suite) starting; progress also in 6v6/symmetric_baseline_suite/provenance/logs/suite.log")
    return subprocess.run([PY, *args], cwd=PROJ, env=env()).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify everything; change nothing")
    ap.add_argument(
        "--allow-exploratory-ablation",
        action="store_true",
        help=(
            "required for any non --check run: this pipeline is defender-only "
            "(frozen ATTACK). Main candidate is DUAL_BRANCH_ROLE_COMPOSITE_V1_SPEC.json."
        ),
    )
    a = ap.parse_args()
    problems, notes = check()
    if a.check:
        for n in notes:
            print("  " + n)
        print("ALL CHECKS PASS" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
        return 0 if not problems else 1
    if not a.allow_exploratory_ablation:
        print(
            "REFUSING: 6v6/run_symmetric_6v6.py is the defender-only symmetric diagnostic.\n"
            "Main candidate is artifacts/strategic_demand/sppo/DUAL_BRANCH_ROLE_COMPOSITE_V1_SPEC.json\n"
            "(dual-branch ATTACK+DEFEND). Pass --allow-exploratory-ablation only for intentional ablation."
        )
        return 2
    if problems:
        fail("pre-run checks failed: " + "; ".join(problems))
    state(status="RUNNING", pid=os.getpid(), git_head=git("rev-parse", "HEAD"))
    log("6v6 symmetric diagnostic started (exploratory ablation; not dual-branch V1)")

    if not done("smoke_B"):
        # same suffix as the authorized run (the authorization matches it); --smoke prefixes the directory with
        # smoke_, so it can never collide with the production run
        rc = run_logged(train_args("B", 2048, 99902002, None, RUNS["B"]["suffix"], smoke=True), "smoke_B")
        if rc != 0:
            fail(f"B-side smoke failed (exit {rc}); see 6v6/symmetric_smoke_B.log")
        state(step="smoke_B")
        log("B-side smoke passed")
    if not done("train"):
        train_both()
        state(step="train")
    if not done("eval"):
        rc = run_logged(eval_args() + ["--dry-run", "--resume"], "eval_dryrun")
        if rc != 0:
            fail(f"evaluation dry-run failed (exit {rc}); see 6v6/symmetric_eval_dryrun.log")
        log("evaluation dry-run passed; running 200 episodes")
        rc = run_logged(eval_args() + ["--resume"], "eval")
        if not (SD / f"{EVAL_LABEL}_SPECIALIST_CROSSOVER_EVAL_RESULT.json").is_file():
            fail(f"evaluation exited {rc} without a sealed result (rerun this script to resume)")
        state(step="eval")
    if not done("core_sealed"):
        ro = readout()
        log("\n" + ro["table_markdown"])
        bundle(ro)
        state(step="core_sealed", status="PHASE1_SEALED")
        log("PHASE 1 DONE -- 6v6/symmetric_results.zip is ready to send; starting Phase 2")
    if not done("baselines"):
        rc = phase2()
        if rc != 0:
            fail(f"Phase 2 stopped (exit {rc}); Phase 1 stays sealed. See 6v6/symmetric_baseline_suite/STATE.json "
                 f"and provenance/logs/suite.log; rerun this script to continue")
        state(step="baselines", status="DONE")
    log("DONE -- send 6v6/symmetric_results.zip and 6v6/symmetric_baselines.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
