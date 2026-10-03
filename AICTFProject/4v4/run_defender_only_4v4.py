r"""4v4 symmetric defender-only (frozen ATTACK), k=1 -- SYMMETRIC_DEFENDER_ONLY_4V4_K1_SPEC.json.

    .venv\Scripts\python.exe 4v4\run_defender_only_4v4.py --check
    .venv\Scripts\python.exe 4v4\run_defender_only_4v4.py            (resumable; run detached)

For A and B identically: the sealed 1M specialist plays every ATTACK slot FROZEN; a defender warm-started from
that same specialist is trained (200k, N' teacher 0.1 -> 0 over 50k-150k, cadence 4) and plays the one DEFEND
slot (k=1). Same flags as the 2v2 defender-only runs. Then the fresh 128-seed four-cell crossover (both sides
spliced), the readout and the training diagnostic. Stage 4 is not run. Outputs: 4v4/defonly/, the
*_sym_defonly_k1 run dirs, and the R4_SYM_DEFENDER_ONLY_4V4_K1 result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
SD = PROJ / "artifacts" / "strategic_demand" / "sppo"
SPEC_REL = "artifacts/strategic_demand/sppo/SYMMETRIC_DEFENDER_ONLY_4V4_K1_SPEC.json"
SPEC = json.loads((PROJ / SPEC_REL).read_text(encoding="utf-8"))
OUT = PROJ / "4v4" / "defonly"
PY = str(PROJ / ".venv" / "Scripts" / "python.exe")
CONF = SPEC["CONFIRMATION_locked"]
LABEL = CONF["label"]
LO, HI = (int(x) for x in CONF["block"].split(".."))
AUTH = {e["policy"]: e for e in SPEC["TRAINING_AUTHORIZED"]}
SUFFIX = "_sym_defonly_k1"
SMOKE = {"A": 99_908_003, "B": 99_908_004}
RUN = {p: {"dir": f"artifacts/scale_4v4_specialists/pi_{p}_specialist_4v4{SUFFIX}"} for p in "AB"}
for p, r in RUN.items():
    r["final"] = f"{r['dir']}/ckpts/final_pi_{p}_specialist_4v4{SUFFIX}.zip"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"{now()} {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "defonly.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sha(rel: str) -> str:
    return hashlib.sha256((PROJ / rel).read_bytes()).hexdigest()


def done(step: str) -> bool:
    return (OUT / "manifests" / f"{step}.json").is_file()


def mark(step: str, **kw) -> None:
    (OUT / "manifests").mkdir(parents=True, exist_ok=True)
    (OUT / "manifests" / f"{step}.json").write_text(json.dumps({"step": step, "utc": now(), **kw}, indent=2) + "\n",
                                                   encoding="utf-8")


def fail(msg: str):
    log(f"STOPPED: {msg}")
    raise SystemExit(1)


def run(argv: list[str], tag: str) -> int:
    env = dict(os.environ, FOR_DISABLE_CONSOLE_CTRL_HANDLER="1", PYTHONUNBUFFERED="1")
    log(f"exec: {' '.join(argv)}")
    with (OUT / f"{tag}.log").open("w", encoding="utf-8") as fo, (OUT / f"{tag}.log.err").open("w", encoding="utf-8") as fe:
        return subprocess.run([PY, *argv], cwd=PROJ, env=env, stdout=fo, stderr=fe).returncode


def train_args(pol: str, steps: int, seed: int, smoke: bool) -> list[str]:
    e = AUTH[pol]
    a = ["experiments/train_specialist_scale.py", "--team-size", "4", "--policy", pol, "--seed", str(seed), "--device", "cuda",
         "--total-timesteps", str(steps), "--entity-repair-enabled", "--entity-hidden-dim", "32",
         "--role-conditioning-enabled", "--role-hold-ticks", "8", "--role-fixed-for-episode", "--role-k-defend", "1",
         "--split-attack-defend-enabled", "--split-attack-defend-frozen-ckpt", e["specialist_path"],
         "--split-attack-defend-frozen-ckpt-sha256", e["specialist_sha256"], "--load-path", e["specialist_path"],
         "--defend-teacher-lambda", "0.1", "--defend-teacher-lambda-end", "0.0",
         "--defend-teacher-decay-start-step", "50000", "--defend-teacher-decay-end-step", "150000",
         "--defend-teacher-cadence", "4", "--run-label-suffix", SUFFIX, "--symmetric-role-spec", SPEC_REL]
    a += ["--smoke"] if smoke else ["--experiment-id", e["experiment_id"]]
    return a


def eval_args() -> list[str]:
    a, b = AUTH["A"], AUTH["B"]
    return ["experiments/eval_specialist_crossover_scaled.py", "--team-size", "4", "--spec", SPEC_REL,
            "--seed-base", str(LO), "--n-seeds", str(HI - LO + 1), "--label", LABEL, "--device", "cuda",
            "--pi-a-path", RUN["A"]["final"], "--pi-b-path", RUN["B"]["final"], "--role-fixed-for-episode", "--role-k-defend", "1",
            "--frozen-attack-path", a["specialist_path"], "--frozen-attack-path-sha256", a["specialist_sha256"],
            "--frozen-attack-path-b", b["specialist_path"], "--frozen-attack-path-b-sha256", b["specialist_sha256"]]


def check() -> list[str]:
    from experiments import seed_registry as SR
    p = []
    if not str(SPEC.get("status", "")).startswith("FROZEN"):
        p.append("spec not frozen")
    for pol, e in AUTH.items():
        if sha(e["specialist_path"]) != e["specialist_sha256"]:
            p.append(f"{pol} specialist sha mismatch")
    reg = {b["experiment_id"]: b for b in SR.load()["blocks"]}
    for eid, lo, hi in ((AUTH["A"]["experiment_id"], AUTH["A"]["seed"], AUTH["A"]["seed"]),
                        (AUTH["B"]["experiment_id"], AUTH["B"]["seed"], AUTH["B"]["seed"]),
                        (CONF["registry_experiment_id"], LO, HI)):
        b = reg.get(eid)
        if b is None or (b["lo"], b["hi"]) != (lo, hi):
            p.append(f"{eid} not registered at {lo}..{hi}")
    return p


def integrity() -> dict:
    """Technical only: finals exist, are role-conditioned defenders at k=1, and the frozen specialists are untouched."""
    import torch
    c = {}
    for pol, r in RUN.items():
        c[f"{pol}_final_exists"] = (PROJ / r["final"]).is_file()
        if c[f"{pol}_final_exists"]:
            cfg = dict(torch.load(PROJ / r["final"], map_location="cpu", weights_only=False).get("cfg") or {})
            c[f"{pol}_role_conditioned"] = bool(cfg.get("role_conditioning_enabled"))
            c[f"{pol}_k1"] = int(cfg.get("role_k_defend", -1)) == 1
            c[f"{pol}_split_frozen_attack"] = bool(cfg.get("split_attack_defend_enabled"))
            c[f"{pol}_not_dual_branch"] = not bool(cfg.get("dual_branch_role_composite_enabled"))
        c[f"{pol}_frozen_specialist_unchanged"] = sha(AUTH[pol]["specialist_path"]) == AUTH[pol]["specialist_sha256"]
    rec = {"utc": now(), "checks": c, "ok": all(c.values()),
           "finals": {p: {"path": r["final"], "sha256": sha(r["final"])} for p, r in RUN.items() if (PROJ / r["final"]).is_file()}}
    (OUT / "TECHNICAL_SEAL.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return rec


def readout() -> dict:
    import numpy as np
    from experiments.eval_hog_psp_v3 import _mean_ci
    from experiments.rescue_ladder_4v4 import adjusted_interval
    rows = SD / f"{LABEL.lower()}_specialist_crossover_eval_rows.csv"
    seeds = list(range(LO, HI + 1))
    by: dict = {}
    with rows.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by.setdefault((r["policy"][-1], r["pole"]), {})[int(r["seed"])] = (float(r["win"]), float(r["margin"]))
    out = {"record": f"{LABEL}_READOUT", "utc": now(), "spec": SPEC_REL, "seeds": CONF["block"], "k_defend": 1}
    for i, field in enumerate(("win", "margin")):
        v = {k: np.array([d[s][i] for s in seeds]) for k, d in by.items()}
        res = {f"{p}@{q}": float(v[(p, q)].mean()) for p in "AB" for q in "AB"}
        for name, x in (("Delta_A", v[("A", "A")] - v[("B", "A")]), ("Delta_B", v[("B", "B")] - v[("A", "B")])):
            c, adj = _mean_ci(x), adjusted_interval(x, attempts=2)
            res[name] = {"mean": float(x.mean()), "std": float(x.std(ddof=1)), "ci95": [c["lcb95"], c["ucb95"]],
                         "ci97_5_two_rescue_attempts": [adj["lcb"], adj["ucb"]]}
        out[field] = res
    f = lambda s: f"{s['mean']:+.3f} ± {s['std']:.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]"  # noqa: E731
    md = [f"# {LABEL} (4v4 defender-only, frozen ATTACK, k=1; fresh seeds {CONF['block']})", "",
          "| | A@A | B@A | A@B | B@B | Δ_A (95%) | Δ_B (95%) |", "|---|---|---|---|---|---|---|"]
    for field in ("win", "margin"):
        t = out[field]
        md.append(f"| {field} | {t['A@A']:.3f} | {t['B@A']:.3f} | {t['A@B']:.3f} | {t['B@B']:.3f} | {f(t['Delta_A'])} | {f(t['Delta_B'])} |")
    w = out["win"]
    md += ["", f"97.5% (two post-hoc rescue attempts): Δ_A [{w['Delta_A']['ci97_5_two_rescue_attempts'][0]:+.3f}, "
               f"{w['Delta_A']['ci97_5_two_rescue_attempts'][1]:+.3f}], Δ_B [{w['Delta_B']['ci97_5_two_rescue_attempts'][0]:+.3f}, "
               f"{w['Delta_B']['ci97_5_two_rescue_attempts'][1]:+.3f}]"]
    out["table_markdown"] = "\n".join(md)
    (OUT / "DEFONLY_READOUT.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    (OUT / "DEFONLY_READOUT.md").write_text(out["table_markdown"] + "\n", encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    problems = check()
    if a.check or problems:
        print("ALL CHECKS PASS" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
        return 0 if not problems else 1
    log("4v4 defender-only (frozen ATTACK, k=1) started")
    for pol in ("A", "B"):
        if not done(f"smoke_{pol}"):
            if run(train_args(pol, 5000, SMOKE[pol], smoke=True), f"smoke_{pol}") != 0:
                fail(f"smoke {pol} failed; see 4v4/defonly/smoke_{pol}.log.err")
            mark(f"smoke_{pol}")
            log(f"smoke {pol} PASS")
    for pol in ("A", "B"):
        if not (PROJ / RUN[pol]["final"]).is_file():
            argv = train_args(pol, 200_000, AUTH[pol]["seed"], smoke=False)
            ck = sorted((PROJ / RUN[pol]["dir"] / "ckpts").glob("ckpt_*.zip")) if (PROJ / RUN[pol]["dir"] / "ckpts").is_dir() else []
            if ck:
                i = argv.index("--load-path")
                del argv[i:i + 2]
                argv += ["--resume", str(ck[-1].relative_to(PROJ)).replace("\\", "/")]
            run(argv, f"train_{pol}")
            if not (PROJ / RUN[pol]["final"]).is_file():
                fail(f"{pol} training ended without {RUN[pol]['final']} (rerun to resume)")
            log(f"{pol} defender trained sha={sha(RUN[pol]['final'])[:16]}")
    if not done("technical_seal"):
        rec = integrity()
        if not rec["ok"]:
            fail(f"technical seal failed: {[k for k, v in rec['checks'].items() if not v]}")
        mark("technical_seal")
        log("technical seal PASS")
    res = SD / f"{LABEL}_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
    if not done("crossover"):
        if not res.is_file():
            if run(eval_args() + ["--dry-run"], "crossover_dryrun") != 0:
                fail("crossover dry-run failed; see 4v4/defonly/crossover_dryrun.log.err")
            run(eval_args() + ["--resume"], "crossover")
        if not res.is_file():
            fail("crossover exited without a sealed result (rerun to resume)")
        mark("crossover")
    ro = readout()
    log("\n" + ro["table_markdown"])
    run(["experiments/diagnose_dual_branch_training.py", "--out", "4v4/defonly"], "training_diagnostic")
    (OUT / "DEFONLY_DONE.txt").write_text(f"DONE {now()}\n", encoding="utf-8")
    log("DONE -- Stage 4 not run (needs PI review); 2v2/6v6 must follow the same construction if adopted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
