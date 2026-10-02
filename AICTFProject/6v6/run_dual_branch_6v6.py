r"""6v6 DUAL_BRANCH_ROLE_COMPOSITE_V1 school-PC launcher.

    cd <repo>; git pull; cd AICTFProject
    .venv\Scripts\python.exe 6v6\run_dual_branch_6v6.py --check
    powershell -ExecutionPolicy Bypass -File 6v6\run_dual_branch_6v6.ps1

School-PC 6v6 suite under DUAL_BRANCH_ROLE_COMPOSITE_V1:

  smoke A, smoke B, joint 200k A, joint 200k B,
  export trained ATTACK branches, four-cell diagnostic crossover on the
  frozen top-50 seeds (post-hoc, not confirmatory), zip the bundle.

k=ceil(6/3)=2. ATTACK and DEFEND are both trained. Do not run
run_symmetric_6v6.py (defender-only ablation). The sharing ladder
(Share-Encoder / Fully Shared+z+r / role-only) is Stage 4 and is not
this launch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
REPO = PROJ.parent
os.chdir(PROJ)
sys.path.insert(0, str(PROJ))

LOG = PROJ / "6v6" / "dual_branch_6v6.log"
STATE = PROJ / "6v6" / "dual_branch_6v6_STATE.json"
SPEC = "artifacts/strategic_demand/sppo/DUAL_BRANCH_ROLE_COMPOSITE_V1_SPEC.json"
EVAL_SPEC = "artifacts/strategic_demand/sppo/DUAL_BRANCH_6V6_SCHOOL_DIAGNOSTIC_SPEC.json"
EVAL_LABEL = "DUAL_BRANCH_6V6_ROLE_COMPOSITE"
EVAL_REG = "STANDARDIZED_6V6_SEPARATED_EVAL"
SEEDS_FILE = "artifacts/strategic_demand/sppo/symmetric_role_top50/6v6_ours_top50_seed_ids.json"
MANIFEST = "6v6/dual_branch_deploy_manifest.json"
BUNDLE = PROJ / "6v6" / "dual_branch_6v6_results.zip"

A = "artifacts/scale_6v6_specialists/pi_A_specialist_6v6_c2_entity_repair/ckpts/final_pi_A_specialist_6v6_c2_entity_repair.zip"
B = "artifacts/scale_6v6_specialists/pi_B_specialist_6v6_c2_entity_repair/ckpts/final_pi_B_specialist_6v6_c2_entity_repair.zip"
SHA = {
    A: "3298000480acd899e715eb78312938a9d20610a1eb0bcf73d382fd268577009e",
    B: "fc0043235d3abc23f87f5157a46920921f5eced0e9aa252347084d2b500258b5",
}
K = 2
TEACHER = [
    "--defend-teacher-lambda", "0.1", "--defend-teacher-lambda-end", "0.0",
    "--defend-teacher-decay-start-step", "50000", "--defend-teacher-decay-end-step", "150000",
    "--defend-teacher-cadence", "4",
]
RUNS = {
    "A": {
        "seed": 26900005,
        "eid": "DUAL_BRANCH_ROLE_COMPOSITE_V1_6V6_A_TRAIN",
        "spec_ck": A,
        "suffix": "_dual_branch_v1",
        "smoke_seed": 99903005,
    },
    "B": {
        "seed": 26900006,
        "eid": "DUAL_BRANCH_ROLE_COMPOSITE_V1_6V6_B_TRAIN",
        "spec_ck": B,
        "suffix": "_dual_branch_v1",
        "smoke_seed": 99903006,
    },
}
for _p, _r in RUNS.items():
    _r["run_dir"] = f"artifacts/scale_6v6_specialists/pi_{_p}_specialist_6v6{_r['suffix']}"
    _r["final"] = f"{_r['run_dir']}/ckpts/final_pi_{_p}_specialist_6v6{_r['suffix']}.zip"
    _r["attack"] = f"{_r['run_dir']}/ckpts/attack_pi_{_p}_specialist_6v6{_r['suffix']}.zip"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"{now()} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def state(**kw) -> dict:
    s = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {"steps": {}}
    s.update({k: v for k, v in kw.items() if k != "step"})
    if "step" in kw:
        s.setdefault("steps", {})[kw["step"]] = now()
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


def env() -> dict:
    e = dict(os.environ)
    e["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
    return e


def train_args(pol: str, steps: int, seed: int, eid: str | None, suffix: str, smoke: bool) -> list[str]:
    r = RUNS[pol]
    a = [
        "experiments/train_specialist_scale.py",
        "--team-size", "6", "--policy", pol, "--seed", str(seed), "--device", "cuda",
        "--total-timesteps", str(steps),
        "--entity-repair-enabled", "--entity-hidden-dim", "32",
        "--role-conditioning-enabled", "--role-hold-ticks", "8", "--role-fixed-for-episode",
        "--role-k-defend", str(K),
        "--split-attack-defend-enabled",
        "--split-attack-defend-frozen-ckpt", r["spec_ck"],
        "--split-attack-defend-frozen-ckpt-sha256", SHA[r["spec_ck"]],
        "--load-path", r["spec_ck"],
        "--dual-branch-role-composite-enabled", "--dual-branch-spec", SPEC,
        *TEACHER, "--run-label-suffix", suffix,
    ]
    a += ["--smoke"] if smoke else ["--experiment-id", str(eid)]
    return a


def run_logged(argv: list[str], tag: str) -> int:
    out = PROJ / "6v6" / f"dual_branch_{tag}.log"
    err = PROJ / "6v6" / f"dual_branch_{tag}.log.err"
    log(f"exec: {' '.join(argv)}")
    with out.open("w", encoding="utf-8") as fo, err.open("w", encoding="utf-8") as fe:
        p = subprocess.run([str(PROJ / ".venv" / "Scripts" / "python.exe"), *argv],
                           cwd=str(PROJ), env=env(), stdout=fo, stderr=fe)
    return int(p.returncode)


def check() -> tuple[list[str], list[str]]:
    problems: list[str] = []
    notes: list[str] = []
    if not (PROJ / ".venv" / "Scripts" / "python.exe").is_file():
        problems.append("missing .venv")
    if not (PROJ / SPEC).is_file():
        problems.append(f"missing {SPEC}")
    else:
        spec = json.loads((PROJ / SPEC).read_text(encoding="utf-8"))
        if "PASSED" not in str((spec.get("JOINT_200K_TRAINING_locked") or {}).get("IMPLEMENTATION_GATE", {}).get("status", "")):
            problems.append("DUAL_BRANCH IMPLEMENTATION_GATE has not PASSED")
        notes.append(f"spec status={spec.get('status')}")
    for pol, r in RUNS.items():
        if not (PROJ / r["spec_ck"]).is_file():
            problems.append(f"missing foundation {pol}: {r['spec_ck']}")
        else:
            got = sha(r["spec_ck"])
            if got != SHA[r["spec_ck"]]:
                problems.append(f"foundation hash mismatch {pol}: {got} != {SHA[r['spec_ck']]}")
            else:
                notes.append(f"foundation {pol} sha OK")
    if not (PROJ / EVAL_SPEC).is_file():
        problems.append(f"missing {EVAL_SPEC}")
    if not (PROJ / SEEDS_FILE).is_file():
        problems.append(f"missing {SEEDS_FILE}")
    primary = PROJ / "artifacts/strategic_demand/sppo/STANDARDIZED_6V6_SPLIT_K1_CONFIRMATORY_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
    if not primary.is_file():
        problems.append(f"missing post-hoc primary record {primary.name}")
    try:
        from experiments import seed_registry as SR
        reg = {b["experiment_id"]: b for b in SR.load()["blocks"]}
        for pol, r in RUNS.items():
            b = reg.get(r["eid"])
            if b is None or int(b["lo"]) != int(r["seed"]):
                problems.append(f"seed block {r['eid']} is not reserved at {r['seed']}")
            elif b.get("status") != "RESERVED" and not (PROJ / r["final"]).is_file():
                problems.append(f"{r['eid']} is {b.get('status')} but the final checkpoint is missing")
        if reg.get(EVAL_REG, {}).get("status") != "SPENT":
            problems.append(f"{EVAL_REG} must be SPENT for the post-hoc diagnostic")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"seed registry unreadable: {exc}")
    try:
        import torch
        if not torch.cuda.is_available():
            problems.append("CUDA not available")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"torch import failed: {exc}")
    return problems, notes


def export_attack_branch(dual_rel: str, out_rel: str) -> None:
    """Write a deploy zip whose weights are the trained ATTACK branch.

    Eval splices this non-role network into ATTACK slots. It is not the
    foundation specialist.
    """
    import torch

    payload = torch.load(PROJ / dual_rel, map_location="cpu", weights_only=False)
    attack_sd = payload.get("attack_branch_state_dict")
    if not isinstance(attack_sd, dict) or not attack_sd:
        fail(f"{dual_rel} has no attack_branch_state_dict")
    cfg = dict(payload.get("cfg") or {})
    cfg["role_conditioning_enabled"] = False
    cfg["dual_branch_role_composite_enabled"] = False
    cfg["split_attack_defend_enabled"] = False
    drop = {
        "attack_branch_state_dict", "attack_branch_optimizer_state", "dual_branch_role_composite",
        "optimizer_state_dict", "actor_optimizer_state_dict", "critic_optimizer_state_dict",
        "actor_cf_optimizer_state_dict", "router_optimizer_state_dict",
    }
    out = {k: v for k, v in payload.items() if k not in drop}
    out["model_state_dict"] = attack_sd
    out["cfg"] = cfg
    out["dual_branch_attack_export"] = True
    out["exported_from"] = dual_rel.replace("\\", "/")
    dest = PROJ / out_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dest)
    log(f"exported ATTACK branch {out_rel} sha={sha(out_rel)[:16]}...")


def write_manifest() -> None:
    pins = {}
    for pol, r in RUNS.items():
        pins[f"pi_{pol}_defend"] = {"path": r["final"], "sha256": sha(r["final"])}
        pins[f"pi_{pol}_attack"] = {"path": r["attack"], "sha256": sha(r["attack"])}
    doc = {
        "architecture": "DUAL_BRANCH_ROLE_COMPOSITE_V1",
        "team_size": 6,
        "k_defend": K,
        "diagnostic": True,
        "confirmatory": False,
        "note": "ATTACK zips are trained branches exported from the dual-branch finals, not the 1M foundations.",
        **pins,
    }
    (PROJ / MANIFEST).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {MANIFEST}")


def eval_args() -> list[str]:
    return [
        "experiments/eval_specialist_crossover_scaled.py",
        "--team-size", "6", "--spec", EVAL_SPEC, "--post-hoc-ablation-spec", EVAL_SPEC,
        "--seed-list", SEEDS_FILE, "--registry-experiment-id", EVAL_REG,
        "--label", EVAL_LABEL, "--device", "cuda",
        "--pi-a-path", RUNS["A"]["final"], "--pi-b-path", RUNS["B"]["final"],
        "--role-fixed-for-episode", "--role-k-defend", str(K),
        "--frozen-attack-path", RUNS["A"]["attack"],
        "--frozen-attack-path-sha256", sha(RUNS["A"]["attack"]),
        "--frozen-attack-path-b", RUNS["B"]["attack"],
        "--frozen-attack-path-b-sha256", sha(RUNS["B"]["attack"]),
        "--dual-branch-deploy-manifest", MANIFEST,
    ]


def bundle() -> None:
    import shutil
    out = PROJ / "6v6" / "dual_branch_6v6_bundle"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for pol, r in RUNS.items():
        dst = out / "checkpoints" / pol
        dst.mkdir(parents=True)
        shutil.copy2(PROJ / r["final"], dst / Path(r["final"]).name)
        shutil.copy2(PROJ / r["attack"], dst / Path(r["attack"]).name)
    ev = out / "evaluation"
    ev.mkdir()
    sd = PROJ / "artifacts" / "strategic_demand" / "sppo"
    for f in sd.glob(f"{EVAL_LABEL}*"):
        if f.is_file():
            shutil.copy2(f, ev / f.name)
    shutil.copy2(PROJ / MANIFEST, out / Path(MANIFEST).name)
    (out / "README.txt").write_text(
        "6v6 dual-branch role composite (school PC).\n"
        "ATTACK and DEFEND were both trained from the same repaired specialist.\n"
        "k=ceil(6/3)=2. The crossover is a post-hoc diagnostic on the frozen top-50 seeds.\n"
        "It is not the paper confirmatory n=128, and it is not the sharing ladder.\n",
        encoding="utf-8",
    )
    if BUNDLE.exists():
        BUNDLE.unlink()
    shutil.make_archive(str(BUNDLE.with_suffix("")), "zip", out)
    log(f"bundled {BUNDLE}")


def train_both() -> None:
    for pol in ("A", "B"):
        r = RUNS[pol]
        final = PROJ / r["final"]
        if final.is_file():
            log(f"{pol} already built: {r['final']}")
            continue
        # resume from latest periodic if present
        ckpts = sorted((PROJ / r["run_dir"] / "ckpts").glob("ckpt_*.zip")) if (PROJ / r["run_dir"] / "ckpts").is_dir() else []
        argv = train_args(pol, 200_000, r["seed"], r["eid"], r["suffix"], smoke=False)
        if ckpts:
            i = argv.index("--load-path")
            del argv[i:i + 2]
            resume = str(ckpts[-1].relative_to(PROJ)).replace("\\", "/")
            argv += ["--resume", resume]
            log(f"{pol} resuming from {ckpts[-1].name}")
        rc = run_logged(argv, f"train_{pol}")
        if not final.is_file():
            fail(f"{pol} training exited {rc} without final zip {r['final']}")
        log(f"{pol} TRAIN DONE sha={sha(r['final'])[:16]}...")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    problems, notes = check()
    if a.check:
        for n in notes:
            print("  " + n)
        print("ALL CHECKS PASS" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
        return 0 if not problems else 1
    if problems:
        fail("pre-run checks failed: " + "; ".join(problems))
    state(status="RUNNING", pid=os.getpid())
    log("6v6 DUAL_BRANCH_ROLE_COMPOSITE_V1 started")

    if not done("smoke_A"):
        rc = run_logged(
            train_args("A", 5000, RUNS["A"]["smoke_seed"], None, RUNS["A"]["suffix"], smoke=True),
            "smoke_A",
        )
        if rc != 0:
            fail(f"A smoke failed exit {rc}")
        state(step="smoke_A")
        log("A smoke PASS")
    if not done("smoke_B"):
        rc = run_logged(
            train_args("B", 5000, RUNS["B"]["smoke_seed"], None, RUNS["B"]["suffix"], smoke=True),
            "smoke_B",
        )
        if rc != 0:
            fail(f"B smoke failed exit {rc}")
        state(step="smoke_B")
        log("B smoke PASS")
    if not done("train"):
        train_both()
        state(step="train", status="TRAIN_DONE")
        log("TRAIN DONE")
    if not done("export"):
        for pol, r in RUNS.items():
            export_attack_branch(r["final"], r["attack"])
        write_manifest()
        state(step="export")
    if not done("eval"):
        rc = run_logged(eval_args() + ["--dry-run"], "eval_dryrun")
        if rc != 0:
            fail(f"evaluation dry-run failed exit {rc}")
        log("evaluation dry-run passed")
        rc = run_logged(eval_args() + ["--resume"], "eval")
        result = PROJ / "artifacts/strategic_demand/sppo" / f"{EVAL_LABEL}_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
        if not result.is_file():
            fail(f"evaluation exited {rc} without {result.name}")
        state(step="eval")
        log("EVAL DONE (post-hoc top-50 diagnostic, not confirmatory)")
    if not done("bundle"):
        bundle()
        state(step="bundle", status="DONE")
    log("DONE -- send 6v6/dual_branch_6v6_results.zip. Sharing ladder is a later stage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
