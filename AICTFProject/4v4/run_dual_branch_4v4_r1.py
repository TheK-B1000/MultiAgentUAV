r"""R1 rescue / mechanism study at 4v4 (DUAL_BRANCH_ROLE_COMPOSITE_V1_R1_SPEC.json): the V1 recipe with k = 1.

    .venv\Scripts\python.exe 4v4\run_dual_branch_4v4_r1.py --check
    .venv\Scripts\python.exe 4v4\run_dual_branch_4v4_r1.py            (resumable; detached via run_dual_branch_4v4_r1.ps1)

The only change from V1 is the allocator rule k = max(1, floor(N/3 + 1/2)) -> k = 1 at 4v4, for BOTH A and B.
Training, ATTACK export, deploy manifest and technical seal are the 4v4 V1 driver's own functions
(4v4/run_dual_branch_4v4.py), reused with only k, spec, labels, seeds and output paths overridden -- so the
recipe is V1 by construction. Every output goes under 4v4/r1/ or the new _dual_branch_r1_k1 run dirs; no V1
file is written. Steps:
  smoke A, smoke B (5000 steps) -> train A, B (200k each, from the sealed 1M specialists) -> export ATTACK ->
  deploy manifest -> technical seal -> fresh 128-seed four-cell crossover (27700001..27700128, both sides
  spliced) -> readout (win + margin, CIs) + training diagnostic -> R1_STAGE3_DONE.txt. Stage 4 is NOT run
  (not authorized before the PI reviews R1 Stage 3). No old-system comparison (PI 2026-10-03).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
_spec = importlib.util.spec_from_file_location("v1_driver_4v4", PROJ / "4v4" / "run_dual_branch_4v4.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)                                  # defines functions/constants only (main is guarded)

R1 = PROJ / "4v4" / "r1"
R1_SPEC = "artifacts/strategic_demand/sppo/DUAL_BRANCH_ROLE_COMPOSITE_V1_R1_SPEC.json"
LABEL = "DUAL_BRANCH_R1_4V4_K1"
EVAL_REG = "DUAL_BRANCH_R1_4V4_K1_SPECIALIST_CROSSOVER"
SEED_BASE, N_SEEDS = 27_700_001, 128
SD = PROJ / "artifacts" / "strategic_demand" / "sppo"

# ---- overrides: k, spec, labels, seeds, every output path (nothing else)
D.K = 1
D.SPEC = R1_SPEC
D.LOG = R1 / "dual_branch_4v4_r1.log"
D.STATE = R1 / "STATE.json"
D.MANIFESTS = R1 / "manifests"
D.SEAL = R1 / "DUAL_BRANCH_4V4_R1_TECHNICAL_SEAL.json"
D.STAGE4_TEACHERS = R1 / "STAGE4_4V4_R1_TEACHERS_SEALED.json"     # never the V1 STAGE4_4V4_TEACHERS_SEALED.json
D.MANIFEST = "4v4/r1/dual_branch_deploy_manifest_r1.json"
D.LEGACY_DIR = R1
D.LEGACY_LOG = R1 / "no_legacy_chain.log"
D.LEGACY_LAUNCH = R1 / "no_legacy_launch.json"
D.LEGACY_DONE = R1 / "R1_TRAIN_DONE.txt"
D.LOCAL_DONE = R1 / "R1_TRAIN_DONE.txt"
D.OWNER = R1 / "r1.owner.json"
D.RUNS = {
    "A": {"seed": 27_500_001, "eid": "DUAL_BRANCH_R1_4V4_A_TRAIN", "spec_ck": D.A, "suffix": "_dual_branch_r1_k1",
          "smoke_seed": 99_905_003},
    "B": {"seed": 27_600_001, "eid": "DUAL_BRANCH_R1_4V4_B_TRAIN", "spec_ck": D.B, "suffix": "_dual_branch_r1_k1",
          "smoke_seed": 99_905_004},
}
for _p, _r in D.RUNS.items():
    _r["run_dir"] = f"artifacts/scale_4v4_specialists/pi_{_p}_specialist_4v4{_r['suffix']}"
    _r["final"] = f"{_r['run_dir']}/ckpts/final_pi_{_p}_specialist_4v4{_r['suffix']}.zip"
    _r["attack"] = f"{_r['run_dir']}/ckpts/attack_pi_{_p}_specialist_4v4{_r['suffix']}.zip"
_orig_run_logged = D.run_logged
D.run_logged = lambda argv, tag: _orig_run_logged(argv, f"r1_{tag}")   # logs: 4v4/dual_branch_r1_<tag>.log


def check() -> list[str]:
    from experiments import seed_registry as SR
    from rl.custom_ppo.split_attack_defend import dual_branch_k
    p = []
    spec = json.loads((PROJ / R1_SPEC).read_text(encoding="utf-8"))
    if "FROZEN" not in str(spec.get("status", "")):
        p.append("R1 spec not frozen")
    if dual_branch_k(spec, 4) != 1 or D.K != 1:
        p.append("R1 spec does not give k=1 at 4v4")
    for pol, r in D.RUNS.items():
        if D.sha(r["spec_ck"]) != D.SHA[r["spec_ck"]]:
            p.append(f"foundation {pol} sha mismatch")
        if "_dual_branch_v1" in r["run_dir"]:
            p.append(f"R1 {pol} run dir is a V1 path")
    reg = {b["experiment_id"]: b for b in SR.load()["blocks"]}
    for eid, lo, hi in (("DUAL_BRANCH_R1_4V4_A_TRAIN", 27_500_001, 27_500_001),
                        ("DUAL_BRANCH_R1_4V4_B_TRAIN", 27_600_001, 27_600_001),
                        (EVAL_REG, SEED_BASE, SEED_BASE + N_SEEDS - 1)):
        b = reg.get(eid)
        if b is None or (b["lo"], b["hi"]) != (lo, hi):
            p.append(f"{eid} not registered at {lo}..{hi}")
    if (SD / f"{LABEL}_SPECIALIST_CROSSOVER_EVAL_RESULT.json").is_file() and not D.done("r1_crossover"):
        p.append("an R1 crossover result exists but the state does not record it")
    return p


def eval_args() -> list[str]:
    R = D.RUNS
    return ["experiments/eval_specialist_crossover_scaled.py", "--team-size", "4", "--spec", R1_SPEC,
            "--seed-base", str(SEED_BASE), "--n-seeds", str(N_SEEDS), "--label", LABEL, "--device", "cuda",
            "--pi-a-path", R["A"]["final"], "--pi-b-path", R["B"]["final"],
            "--role-fixed-for-episode", "--role-k-defend", "1",
            "--frozen-attack-path", R["A"]["attack"], "--frozen-attack-path-sha256", D.sha(R["A"]["attack"]),
            "--frozen-attack-path-b", R["B"]["attack"], "--frozen-attack-path-b-sha256", D.sha(R["B"]["attack"]),
            "--dual-branch-deploy-manifest", D.MANIFEST]


def readout() -> dict:
    import numpy as np
    from experiments.eval_hog_psp_v3 import _mean_ci
    rows = SD / f"{LABEL.lower()}_specialist_crossover_eval_rows.csv"
    seeds = list(range(SEED_BASE, SEED_BASE + N_SEEDS))
    by: dict = {}
    with rows.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by.setdefault((r["policy"][-1], r["pole"]), {})[int(r["seed"])] = (float(r["win"]), float(r["margin"]))

    def st(x):
        c = _mean_ci(np.asarray(x, dtype=float))
        return {"mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)), "lcb95": float(c["lcb95"]), "ucb95": float(c["ucb95"])}
    out = {"record": f"{LABEL}_READOUT", "utc": D.now(), "spec": R1_SPEC, "seeds": f"{SEED_BASE}..{SEED_BASE + N_SEEDS - 1}",
           "rows": str(rows.relative_to(PROJ)).replace("\\", "/"), "k_defend": 1}
    for i, field in enumerate(("win", "margin")):
        v = {k: np.array([d[s][i] for s in seeds]) for k, d in by.items()}
        out[field] = {**{f"{p}@{q}": float(v[(p, q)].mean()) for p in "AB" for q in "AB"},
                      "Delta_A": st(v[("A", "A")] - v[("B", "A")]), "Delta_B": st(v[("B", "B")] - v[("A", "B")])}
    f = lambda s: f"{s['mean']:+.3f} ± {s['std']:.3f} [{s['lcb95']:+.3f}, {s['ucb95']:+.3f}]"  # noqa: E731
    md = [f"# {LABEL} (R1: 4v4, k=1; fresh seeds {out['seeds']})", "",
          "| | A@A | B@A | A@B | B@B | Δ_A | Δ_B |", "|---|---|---|---|---|---|---|"]
    for field in ("win", "margin"):
        t = out[field]
        md.append(f"| {field} | {t['A@A']:.3f} | {t['B@A']:.3f} | {t['A@B']:.3f} | {t['B@B']:.3f} | "
                  f"{f(t['Delta_A'])} | {f(t['Delta_B'])} |")
    out["table_markdown"] = "\n".join(md)
    (R1 / "R1_STAGE3_READOUT.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    (R1 / "R1_STAGE3_READOUT.md").write_text(out["table_markdown"] + "\n", encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    R1.mkdir(parents=True, exist_ok=True)
    problems = check()
    if a.check or problems:
        print("ALL CHECKS PASS" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
        return 0 if not problems else 1
    D.state(status="RUNNING", pid=__import__("os").getpid(), k_defend=1, spec=R1_SPEC)
    D.log("R1 rescue (4v4, k=1) started")
    if not D.done("phase0_preflight"):
        D.manifest("phase0_preflight", spec=R1_SPEC, allocator_rule="max(1, floor(N/3 + 1/2))")
    for pol in ("A", "B"):
        if not D.done(f"phase1_smoke_{pol}"):
            rc = D.run_logged(D.train_args(pol, 5000, D.RUNS[pol]["smoke_seed"], None, D.RUNS[pol]["suffix"], smoke=True),
                              f"smoke_{pol}")
            if rc != 0:
                D.fail(f"R1 smoke {pol} failed exit {rc}; see 4v4/dual_branch_r1_smoke_{pol}.log.err")
            D.manifest(f"phase1_smoke_{pol}")
            D.log(f"R1 smoke {pol} PASS")
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
        man.update({"allocator_rule": "max(1, floor(N/3 + 1/2))", "spec": R1_SPEC, "rescue": "R1"})
        (PROJ / D.MANIFEST).write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
        D.manifest("phase1_export", deploy_manifest=D.MANIFEST)
    if not D.done("phase1_technical_seal"):
        D.technical_seal()
        D.manifest("phase1_technical_seal", seal=str(D.SEAL.relative_to(PROJ)).replace("\\", "/"))
    if not D.done("r1_crossover"):
        res = SD / f"{LABEL}_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
        if not res.is_file():
            if D.run_logged(eval_args() + ["--dry-run"], "crossover_dryrun") != 0:
                D.fail("R1 crossover dry-run failed; see 4v4/dual_branch_r1_crossover_dryrun.log.err")
            D.run_logged(eval_args() + ["--resume"], "crossover")
        if not res.is_file():
            D.fail("R1 crossover exited without a sealed result (rerun to resume)")
        D.manifest("r1_crossover", label=LABEL)
    ro = readout()
    D.log("\n" + ro["table_markdown"])
    D.run_logged(["experiments/diagnose_dual_branch_training.py", "--out", "4v4/r1"], "training_diagnostic")
    (R1 / "R1_STAGE3_DONE.txt").write_text(f"DONE {D.now()}\n", encoding="utf-8")
    D.state(status="R1_STAGE3_DONE")
    D.log("R1 Stage 3 DONE -- Stage 4 not run (needs PI review)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
