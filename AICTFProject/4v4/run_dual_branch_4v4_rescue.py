r"""Run one later rung (R2 or R3) of the pre-registered 4v4 rescue ladder (DUAL_BRANCH_4V4_RESCUE_LADDER_SPEC.json).

    .venv\Scripts\python.exe 4v4\run_dual_branch_4v4_rescue.py --rung R2 [--check]

Same construction as the R1 runner: the 4v4 V1 driver's own training, ATTACK export, deploy manifest and technical
seal, with k = 1 (R1 rule) and only the rung's single change (teacher strength for R2, budget for R3) plus its
labels, seeds and output paths taken from the frozen ladder spec. Then the rung's fresh 128-seed four-cell crossover
and readout. Started by experiments/rescue_ladder_4v4.py --chain only if every earlier rung failed the pass rule.
Stage 4 is never run here.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
SD = PROJ / "artifacts" / "strategic_demand" / "sppo"
LADDER = json.loads((SD / "DUAL_BRANCH_4V4_RESCUE_LADDER_SPEC.json").read_text(encoding="utf-8"))


def load_driver(rung: str):
    cfg, seeds = LADDER["rungs"][rung], LADDER["seeds"][rung]
    spec = importlib.util.spec_from_file_location(f"v1_driver_4v4_{rung}", PROJ / "4v4" / "run_dual_branch_4v4.py")
    D = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(D)
    out = PROJ / cfg["dir"]
    t = cfg["teacher"]
    D.K = 1
    D.SPEC = f"artifacts/strategic_demand/sppo/{cfg['spec']}"
    D.TEACHER = ["--defend-teacher-lambda", str(t["lambda"]), "--defend-teacher-lambda-end", str(t["lambda_end"]),
                 "--defend-teacher-decay-start-step", str(t["decay_start"]),
                 "--defend-teacher-decay-end-step", str(t["decay_end"]), "--defend-teacher-cadence", str(t["cadence"])]
    D.LOG, D.STATE, D.MANIFESTS = out / f"{rung.lower()}.log", out / "STATE.json", out / "manifests"
    D.SEAL = out / f"DUAL_BRANCH_4V4_{rung}_TECHNICAL_SEAL.json"
    D.STAGE4_TEACHERS = out / f"STAGE4_4V4_{rung}_TEACHERS_SEALED.json"
    D.MANIFEST = f"{cfg['dir']}/dual_branch_deploy_manifest_{rung.lower()}.json"
    D.LEGACY_DIR, D.LEGACY_LOG, D.LEGACY_LAUNCH = out, out / "no_legacy_chain.log", out / "no_legacy_launch.json"
    D.LEGACY_DONE = D.LOCAL_DONE = out / f"{rung}_TRAIN_DONE.txt"
    D.OWNER = out / f"{rung.lower()}.owner.json"
    smoke = {"R2": (99_906_003, 99_906_004), "R3": (99_907_003, 99_907_004)}[rung]
    D.RUNS = {p: {"seed": seeds[p], "eid": f"DUAL_BRANCH_{rung}_4V4_{p}_TRAIN", "spec_ck": getattr(D, p),
                  "suffix": cfg["suffix"], "smoke_seed": smoke[i]} for i, p in enumerate(("A", "B"))}
    for p, r in D.RUNS.items():
        r["run_dir"] = f"artifacts/scale_4v4_specialists/pi_{p}_specialist_4v4{r['suffix']}"
        r["final"] = f"{r['run_dir']}/ckpts/final_pi_{p}_specialist_4v4{r['suffix']}.zip"
        r["attack"] = f"{r['run_dir']}/ckpts/attack_pi_{p}_specialist_4v4{r['suffix']}.zip"
    orig_args, orig_log = D.train_args, D.run_logged
    steps = int(cfg["steps"])
    D.train_args = lambda pol, s, *a, **k: orig_args(pol, steps if s == 200_000 else s, *a, **k)
    D.run_logged = lambda argv, tag: orig_log(argv, f"{rung.lower()}_{tag}")
    return D, cfg, seeds, out


def eval_args(D, cfg, seeds) -> list[str]:
    R = D.RUNS
    lo, hi = seeds["eval"]
    return ["experiments/eval_specialist_crossover_scaled.py", "--team-size", "4", "--spec", D.SPEC,
            "--seed-base", str(lo), "--n-seeds", str(hi - lo + 1), "--label", cfg["label"], "--device", "cuda",
            "--pi-a-path", R["A"]["final"], "--pi-b-path", R["B"]["final"], "--role-fixed-for-episode", "--role-k-defend", "1",
            "--frozen-attack-path", R["A"]["attack"], "--frozen-attack-path-sha256", D.sha(R["A"]["attack"]),
            "--frozen-attack-path-b", R["B"]["attack"], "--frozen-attack-path-b-sha256", D.sha(R["B"]["attack"]),
            "--dual-branch-deploy-manifest", D.MANIFEST]


def check(D, cfg, seeds) -> list[str]:
    from experiments import seed_registry as SR
    from rl.custom_ppo.split_attack_defend import dual_branch_k
    p = []
    spec = json.loads((PROJ / D.SPEC).read_text(encoding="utf-8"))
    if "FROZEN" not in str(spec.get("status", "")) or dual_branch_k(spec, 4) != 1:
        p.append("rung spec not frozen or k != 1")
    reg = {b["experiment_id"]: b for b in SR.load()["blocks"]}
    for side in ("A", "B"):
        b = reg.get(D.RUNS[side]["eid"])
        if b is None or b["lo"] != seeds[side]:
            p.append(f"{D.RUNS[side]['eid']} not registered")
        if "_dual_branch_v1" in D.RUNS[side]["run_dir"] or "_r1_" in D.RUNS[side]["run_dir"]:
            p.append("rung would write a V1/R1 run dir")
    if seeds["eval_id"] not in reg:
        p.append(f"{seeds['eval_id']} not registered")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rung", required=True, choices=("R2", "R3"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    D, cfg, seeds, out = load_driver(a.rung)
    out.mkdir(parents=True, exist_ok=True)
    problems = check(D, cfg, seeds)
    if a.check or problems:
        print("ALL CHECKS PASS" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
        return 0 if not problems else 1
    D.state(status="RUNNING", pid=os.getpid(), rung=a.rung, change=cfg["change_vs_R1"])
    D.log(f"{a.rung} started: {cfg['change_vs_R1']}")
    for pol in ("A", "B"):
        if not D.done(f"phase1_smoke_{pol}"):
            rc = D.run_logged(D.train_args(pol, 5000, D.RUNS[pol]["smoke_seed"], None, D.RUNS[pol]["suffix"], smoke=True),
                              f"smoke_{pol}")
            if rc != 0:
                D.fail(f"{a.rung} smoke {pol} failed exit {rc}")
            D.manifest(f"phase1_smoke_{pol}")
    if not D.done("phase1_train"):
        D.train_both()
        D.mark_train_done()
        D.manifest("phase1_train")
    if not D.done("phase1_export"):
        for pol, r in D.RUNS.items():
            if not (PROJ / r["attack"]).is_file():
                D.export_attack_branch(r["final"], r["attack"])
        D.write_manifest()
        man = json.loads((PROJ / D.MANIFEST).read_text(encoding="utf-8"))
        man.update({"allocator_rule": "max(1, floor(N/3 + 1/2))", "spec": D.SPEC, "rescue": a.rung,
                    "change_vs_R1": cfg["change_vs_R1"]})
        (PROJ / D.MANIFEST).write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
        D.manifest("phase1_export", deploy_manifest=D.MANIFEST)
    if not D.done("phase1_technical_seal"):
        D.technical_seal()
        D.manifest("phase1_technical_seal")
    res = SD / f"{cfg['label']}_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
    if not D.done("crossover"):
        if not res.is_file():
            if D.run_logged(eval_args(D, cfg, seeds) + ["--dry-run"], "crossover_dryrun") != 0:
                D.fail(f"{a.rung} crossover dry-run failed")
            D.run_logged(eval_args(D, cfg, seeds) + ["--resume"], "crossover")
        if not res.is_file():
            D.fail(f"{a.rung} crossover exited without a sealed result (rerun to resume)")
        D.manifest("crossover", label=cfg["label"])
    D.run_logged(["experiments/diagnose_dual_branch_training.py", "--out", cfg["dir"]], "training_diagnostic")
    (PROJ / cfg["done"]).write_text(f"DONE {D.now()}\n", encoding="utf-8")
    D.state(status=f"{a.rung}_STAGE3_DONE")
    D.log(f"{a.rung} Stage 3 DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
