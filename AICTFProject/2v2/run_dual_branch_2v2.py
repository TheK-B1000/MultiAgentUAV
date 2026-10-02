r"""2v2 frozen pipeline: Ours-Teachers -> Stage 3 -> Stage 4 Ours-Shared. k=ceil(2/3)=1.

Resumable. Technical integrity only -- a bad Delta does not stop the chain.
Does not start a second copy of a live Phase-1 train.

    cd AICTFProject
    .venv\Scripts\python.exe 2v2\run_dual_branch_2v2.py --check
    powershell -ExecutionPolicy Bypass -File 2v2\run_dual_branch_2v2.ps1

If artifacts/.../dual_branch_v1/run_2v2_dual_branch_sequential.ps1 is already
training, this process waits for SYM_DUAL_BRANCH_2V2_TRAIN_DONE.txt, then
exports ATTACK branches and continues Phase 2 through the zip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
os.chdir(PROJ)
sys.path.insert(0, str(PROJ))

LOG = PROJ / "2v2" / "dual_branch_2v2.log"
STATE = PROJ / "2v2" / "dual_branch_2v2_STATE.json"
MANIFESTS = PROJ / "2v2" / "manifests"
SEAL = PROJ / "2v2" / "DUAL_BRANCH_2V2_TECHNICAL_SEAL.json"
STAGE4_TEACHERS = PROJ / "artifacts" / "strategic_demand" / "sppo" / "STAGE4_2V2_TEACHERS_SEALED.json"
SPEC = "artifacts/strategic_demand/sppo/DUAL_BRANCH_ROLE_COMPOSITE_V1_SPEC.json"
AUTH = "artifacts/strategic_demand/sppo/STAGE4_2V2_FULL_SUITE_SPEC.json"
EVAL_SPEC = "artifacts/strategic_demand/sppo/DUAL_BRANCH_2V2_DIAGNOSTIC_SPEC.json"
EVAL_LABEL = "DUAL_BRANCH_2V2_ROLE_COMPOSITE"
EVAL_REG = "STANDARDIZED_2V2_SPLIT_K1_CONFIRMATORY_SPECIALIST_CROSSOVER"
PRIMARY = "STANDARDIZED_2V2_SPLIT_K1_CONFIRMATORY_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
SEEDS_FILE = "artifacts/strategic_demand/sppo/symmetric_role_top50/2v2_ours_top50_seed_ids.json"
MANIFEST = "2v2/dual_branch_deploy_manifest.json"
BUNDLE = PROJ / "2v2" / "dual_branch_2v2_full_suite.zip"
PY = str(PROJ / ".venv" / "Scripts" / "python.exe")
LEGACY_DIR = PROJ / "artifacts" / "strategic_demand" / "sppo" / "dual_branch_v1"
LEGACY_DONE = LEGACY_DIR / "SYM_DUAL_BRANCH_2V2_TRAIN_DONE.txt"
LEGACY_LOG = LEGACY_DIR / "chain_2v2.log"
LEGACY_LAUNCH = LEGACY_DIR / "CHAIN_LAUNCH_2V2.json"
LOCAL_DONE = PROJ / "2v2" / "SYM_DUAL_BRANCH_2V2_TRAIN_DONE.txt"

A = "artifacts/scale_2v2_specialists/pi_A_specialist_2v2_std_entity_repair/ckpts/final_pi_A_specialist_2v2_std_entity_repair.zip"
B = "artifacts/scale_2v2_specialists/pi_B_specialist_2v2_std_entity_repair/ckpts/final_pi_B_specialist_2v2_std_entity_repair.zip"
SHA = {
    A: "858805dde3588a686868ae15cfcc4b4c2d63f30d23222dfe6ae421ab7a563b7f",
    B: "9ee024ad6356ea79ee765c7d55bba184a510a73fe8a4738a1f7203630636fd0d",
}
K = 1
TEACHER = [
    "--defend-teacher-lambda", "0.1", "--defend-teacher-lambda-end", "0.0",
    "--defend-teacher-decay-start-step", "50000", "--defend-teacher-decay-end-step", "150000",
    "--defend-teacher-cadence", "4",
]
RUNS = {
    "A": {
        "seed": 26900001,
        "eid": "DUAL_BRANCH_ROLE_COMPOSITE_V1_2V2_A_TRAIN",
        "spec_ck": A,
        "suffix": "_dual_branch_v1",
        "smoke_seed": 99903001,
    },
    "B": {
        "seed": 26900002,
        "eid": "DUAL_BRANCH_ROLE_COMPOSITE_V1_2V2_B_TRAIN",
        "spec_ck": B,
        "suffix": "_dual_branch_v1",
        "smoke_seed": 99903002,
    },
}
for _p, _r in RUNS.items():
    _r["run_dir"] = f"artifacts/scale_2v2_specialists/pi_{_p}_specialist_2v2{_r['suffix']}"
    _r["final"] = f"{_r['run_dir']}/ckpts/final_pi_{_p}_specialist_2v2{_r['suffix']}.zip"
    _r["attack"] = f"{_r['run_dir']}/ckpts/attack_pi_{_p}_specialist_2v2{_r['suffix']}.zip"

STAGE4_ARMS = ("share_encoder", "fully_shared", "role_only")
WAIT_S = 60


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
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
    return s


def manifest(step: str, **extra) -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    doc = {"step": step, "utc": now(), "status": "COMPLETE", "team_size": 2, "k_defend": K, **extra}
    (MANIFESTS / f"{step}.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    state(step=step)


def done(step: str) -> bool:
    return (MANIFESTS / f"{step}.json").is_file()


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
    e["PYTHONUNBUFFERED"] = "1"
    return e


def run_logged(argv: list[str], tag: str) -> int:
    out = PROJ / "2v2" / f"dual_branch_{tag}.log"
    err = PROJ / "2v2" / f"dual_branch_{tag}.log.err"
    log(f"exec: {' '.join(argv)}")
    with out.open("w", encoding="utf-8") as fo, err.open("w", encoding="utf-8") as fe:
        p = subprocess.run([PY, *argv], cwd=str(PROJ), env=env(), stdout=fo, stderr=fe)
    return int(p.returncode)


def finals_ready() -> bool:
    return all((PROJ / RUNS[p]["final"]).is_file() for p in ("A", "B"))


def train_done_marked() -> bool:
    return LEGACY_DONE.is_file() or LOCAL_DONE.is_file()


def mark_train_done() -> None:
    text = "DONE\n"
    LOCAL_DONE.write_text(text, encoding="utf-8")
    if not LEGACY_DONE.is_file():
        LEGACY_DIR.mkdir(parents=True, exist_ok=True)
        LEGACY_DONE.write_text(text, encoding="utf-8")


def _pid_alive(pid: int) -> bool:
    if pid <= 0 or pid == os.getpid():
        return False
    import ctypes
    SYNCHRONIZE = 0x00100000
    handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, int(pid))
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def foreign_phase1_owners() -> list[str]:
    """Live sequential chain or its 2v2 dual-branch trainer. Never this orchestrator."""
    owners: list[str] = []
    if LEGACY_LAUNCH.is_file():
        try:
            pid = int(json.loads(LEGACY_LAUNCH.read_text(encoding="utf-8")).get("chain_pid") or 0)
        except (OSError, ValueError, TypeError):
            pid = 0
        if _pid_alive(pid):
            known_pid = pid
        else:
            known_pid = 0
    else:
        known_pid = 0
    try:
        raw = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine } | "
             "ForEach-Object { '{0}|{1}' -f $_.ProcessId, ($_.CommandLine -replace '\\s+',' ') }"],
            text=True, timeout=40, cwd=str(PROJ),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log(f"WARN: process scan failed ({exc})")
        if known_pid:
            return [f"chain_pid={known_pid}"]
        return owners
    me = os.getpid()
    for line in raw.splitlines():
        if "|" not in line:
            continue
        pid_s, cmd = line.split("|", 1)
        try:
            pid = int(pid_s.strip())
        except ValueError:
            continue
        if pid == me:
            continue
        c = cmd.lower()
        if "run_dual_branch_2v2.py" in c:
            continue
        sequential = "run_2v2_dual_branch_sequential.ps1" in c
        trainer = (
            "train_specialist_scale.py" in c
            and "--team-size 2" in c
            and "dual-branch-role-composite-enabled" in c
            and "_dual_branch_v1" in c
        )
        if sequential or trainer or (known_pid and pid == known_pid and "run_2v2_dual_branch" in c):
            owners.append(f"pid={pid}")
    if not owners and known_pid and _pid_alive(known_pid):
        owners.append(f"chain_pid={known_pid}")
    return owners


def wait_for_foreign_phase1() -> None:
    """Block while another process owns 2v2 dual-branch training. Do not spawn a second copy."""
    announced = False
    while True:
        if train_done_marked() and finals_ready():
            log("Phase 1 already sealed by the existing chain (both finals + DONE marker)")
            return
        owners = foreign_phase1_owners()
        if not owners:
            if not finals_ready():
                log("No live 2v2 dual-branch chain. This orchestrator will resume Phase 1 if needed.")
            return
        if not announced:
            log("WAITING: live Phase 1 owns the GPU (" + ", ".join(owners)
                + "). Not launching another smoke or 200k.")
            announced = True
        else:
            log("still waiting on Phase 1 (" + ", ".join(owners) + ")")
        time.sleep(WAIT_S)


def smoke_already(pol: str) -> bool:
    if done(f"phase1_smoke_{pol}"):
        return True
    text = LEGACY_LOG.read_text(encoding="utf-8", errors="replace") if LEGACY_LOG.is_file() else ""
    train_log = LEGACY_DIR / f"train_2v2_{pol}.log"
    if f"smoke {pol} PASS" in text or train_log.is_file() or (PROJ / RUNS[pol]["final"]).is_file():
        manifest(f"phase1_smoke_{pol}", source="existing dual_branch_v1 chain")
        log(f"smoke {pol} already finished; skipping")
        return True
    return False


def train_args(pol: str, steps: int, seed: int, eid: str | None, suffix: str, smoke: bool) -> list[str]:
    r = RUNS[pol]
    a = [
        "experiments/train_specialist_scale.py",
        "--team-size", "2", "--policy", pol, "--seed", str(seed), "--device", "cuda",
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


def check() -> tuple[list[str], list[str]]:
    problems: list[str] = []
    notes: list[str] = []
    if not Path(PY).is_file():
        problems.append("missing .venv")
    if not (PROJ / SPEC).is_file():
        problems.append(f"missing {SPEC}")
    else:
        spec = json.loads((PROJ / SPEC).read_text(encoding="utf-8"))
        gate = str((spec.get("JOINT_200K_TRAINING_locked") or {}).get("IMPLEMENTATION_GATE", {}).get("status", ""))
        if "PASSED" not in gate:
            problems.append("DUAL_BRANCH IMPLEMENTATION_GATE has not PASSED")
        notes.append(f"spec status={spec.get('status')} gate={gate}")
    if not (PROJ / AUTH).is_file():
        problems.append(f"missing {AUTH}")
    for pol, r in RUNS.items():
        if not (PROJ / r["spec_ck"]).is_file():
            problems.append(f"missing foundation {pol}: {r['spec_ck']}")
        else:
            got = sha(r["spec_ck"])
            if got != SHA[r["spec_ck"]]:
                problems.append(f"foundation hash mismatch {pol}")
            else:
                notes.append(f"foundation {pol} sha OK")
    if K != -(-2 // 3):
        problems.append(f"k_defend {K} != ceil(2/3)")
    else:
        notes.append("k_defend=1 = ceil(2/3)")
    if not (PROJ / EVAL_SPEC).is_file():
        problems.append(f"missing {EVAL_SPEC}")
    else:
        entry = json.loads((PROJ / EVAL_SPEC).read_text(encoding="utf-8"))
        entry = (entry.get("POST_HOC_MATCHED_ROLE_ABLATIONS") or {}).get(EVAL_LABEL) or {}
        seeds = json.loads((PROJ / SEEDS_FILE).read_text(encoding="utf-8")) if (PROJ / SEEDS_FILE).is_file() else []
        if [int(s) for s in entry.get("seed_ids") or []] != [int(s) for s in seeds]:
            problems.append("diagnostic spec seed_ids != frozen 2v2 top-50 file")
        else:
            notes.append(f"top-50 diagnostic seeds n={len(seeds)}")
    if not (PROJ / SEEDS_FILE).is_file():
        problems.append(f"missing {SEEDS_FILE}")
    primary = PROJ / "artifacts/strategic_demand/sppo" / PRIMARY
    if not primary.is_file():
        problems.append(f"missing post-hoc primary record {PRIMARY}")
    try:
        from experiments import seed_registry as SR
        from experiments import prepare_stage4_baselines as P4
        reg = {b["experiment_id"]: b for b in SR.load()["blocks"]}
        for pol, r in RUNS.items():
            b = reg.get(r["eid"])
            if b is None or int(b["lo"]) != int(r["seed"]):
                problems.append(f"seed block {r['eid']} is not reserved at {r['seed']}")
            elif b.get("status") != "RESERVED" and not (PROJ / r["final"]).is_file():
                problems.append(f"{r['eid']} is {b.get('status')} but the final checkpoint is missing")
            else:
                notes.append(f"{r['eid']} {b.get('status')}")
        ev = reg.get(EVAL_REG)
        if ev is None or ev.get("status") != "SPENT":
            problems.append(f"{EVAL_REG} must be SPENT for the post-hoc diagnostic")
        else:
            notes.append(f"{EVAL_REG} SPENT {ev['lo']}..{ev['hi']}")
        for _k, (eid, lo, hi) in P4.blocks(2).items():
            b = reg.get(eid)
            if b is None:
                problems.append(f"Stage4 seed block {eid} missing (run prepare_stage4_baselines --reserve)")
            elif (b["lo"], b["hi"]) != (lo, hi):
                problems.append(f"{eid} range mismatch")
            else:
                notes.append(f"Stage4 block {eid} OK")
        if P4.k_sym(2) != 1 or P4.k_sym(4) != 2 or P4.k_sym(6) != 2:
            problems.append("k_sym scale table drifted")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"seed registry unreadable: {exc}")
    try:
        import torch
        if not torch.cuda.is_available():
            problems.append("CUDA not available")
        else:
            notes.append("CUDA available")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"torch import failed: {exc}")
    return problems, notes


def export_attack_branch(dual_rel: str, out_rel: str) -> None:
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
        "team_size": 2,
        "k_defend": K,
        "diagnostic": True,
        "confirmatory": False,
        "note": "ATTACK zips are trained branches exported from the dual-branch finals, not the 1M foundations.",
        **pins,
    }
    (PROJ / MANIFEST).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {MANIFEST}")


def technical_seal() -> dict:
    """Integrity gate only. Does not look at win rates or Delta."""
    import torch
    from rl.custom_ppo import load_custom_ppo_policy
    import experiments.r2_learned_crossover as R2

    checks: dict[str, bool] = {}
    detail: dict = {}
    for pol, r in RUNS.items():
        checks[f"{pol}_final_exists"] = (PROJ / r["final"]).is_file()
        checks[f"{pol}_attack_exists"] = (PROJ / r["attack"]).is_file()
        if not checks[f"{pol}_final_exists"]:
            continue
        payload = torch.load(PROJ / r["final"], map_location="cpu", weights_only=False)
        cfg = dict(payload.get("cfg") or {})
        checks[f"{pol}_dual_branch_flag"] = bool(cfg.get("dual_branch_role_composite_enabled"))
        checks[f"{pol}_role_conditioning"] = bool(cfg.get("role_conditioning_enabled"))
        checks[f"{pol}_k_defend"] = int(cfg.get("role_k_defend", -1)) == K
        checks[f"{pol}_has_attack_branch_sd"] = isinstance(payload.get("attack_branch_state_dict"), dict)
        checks[f"{pol}_foundation_sha"] = sha(r["spec_ck"]) == SHA[r["spec_ck"]]
        detail[f"{pol}_final_sha"] = sha(r["final"])
        detail[f"{pol}_attack_sha"] = sha(r["attack"]) if checks[f"{pol}_attack_exists"] else None
        try:
            R2.AGENTS = 2
            probe = R2.build_env("cpu", 99_999_002)
            obs_s, act_s = probe.observation_space, probe.action_space
            probe.close()
            d = load_custom_ppo_policy(str(PROJ / r["final"]), obs_s, act_s, device="cpu")
            a = load_custom_ppo_policy(str(PROJ / r["attack"]), obs_s, act_s, device="cpu")
            checks[f"{pol}_defend_loads"] = True
            checks[f"{pol}_attack_loads"] = True
            checks[f"{pol}_attack_not_role_conditioned"] = not bool(
                getattr(a.model, "role_conditioning_enabled", False)
            )
            checks[f"{pol}_defend_is_role_conditioned"] = bool(
                getattr(d.model, "role_conditioning_enabled", False)
            )
            del d, a
        except Exception as exc:  # noqa: BLE001
            checks[f"{pol}_defend_loads"] = False
            checks[f"{pol}_attack_loads"] = False
            detail[f"{pol}_load_error"] = str(exc)

    checks["manifest_exists"] = (PROJ / MANIFEST).is_file()
    if checks["manifest_exists"]:
        man = json.loads((PROJ / MANIFEST).read_text(encoding="utf-8"))
        checks["manifest_architecture"] = man.get("architecture") == "DUAL_BRANCH_ROLE_COMPOSITE_V1"
        checks["manifest_k"] = int(man.get("k_defend", -1)) == K
        for pol, r in RUNS.items():
            pin_d = man.get(f"pi_{pol}_defend") or {}
            pin_a = man.get(f"pi_{pol}_attack") or {}
            checks[f"manifest_{pol}_defend_sha"] = pin_d.get("sha256") == sha(r["final"])
            checks[f"manifest_{pol}_attack_sha"] = pin_a.get("sha256") == sha(r["attack"])

    ok = all(checks.values())
    seal = {
        "record_id": "DUAL_BRANCH_2V2_TECHNICAL_SEAL",
        "status": "SEALED" if ok else "FAILED",
        "utc": now(),
        "gate": "technical_integrity_only",
        "not_a_gate": ["win_rate", "Delta_A", "Delta_B", "strategy_signature_quality"],
        "k_defend": K,
        "checks": checks,
        "detail": detail,
        "continue_even_if_strategy_results_ugly": True,
    }
    SEAL.write_text(json.dumps(seal, indent=2) + "\n", encoding="utf-8")
    if not ok:
        failed = [k for k, v in checks.items() if not v]
        fail(f"TECHNICAL SEAL FAILED: {failed}")
    man = json.loads((PROJ / MANIFEST).read_text(encoding="utf-8"))
    STAGE4_TEACHERS.write_text(json.dumps({
        "record_id": "STAGE4_2V2_TEACHERS_SEALED",
        "status": "SEALED",
        "utc": now(),
        "parent": MANIFEST,
        "technical_seal": str(SEAL.relative_to(PROJ)).replace("\\", "/"),
        "pins": {k: man[k] for k in ("pi_A_defend", "pi_A_attack", "pi_B_defend", "pi_B_attack")},
        "teachers_are_dual_branch_composites": True,
        "not_foundation_specialists": True,
    }, indent=2) + "\n", encoding="utf-8")
    log(f"TECHNICAL SEAL PASS -> {SEAL.name}")
    return seal


def eval_args() -> list[str]:
    return [
        "experiments/eval_specialist_crossover_scaled.py",
        "--team-size", "2", "--spec", EVAL_SPEC, "--post-hoc-ablation-spec", EVAL_SPEC,
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


def train_both() -> None:
    for pol in ("A", "B"):
        r = RUNS[pol]
        final = PROJ / r["final"]
        if final.is_file():
            log(f"{pol} already built: {r['final']}")
            continue
        owners = foreign_phase1_owners()
        if owners:
            fail("refusing to start training while a live chain still owns Phase 1: " + ", ".join(owners))
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


def write_phase2_diagnostic(result: Path) -> None:
    """Record Delta and the four cells. Ugly numbers do not stop the suite."""
    doc = json.loads(result.read_text(encoding="utf-8"))
    gate = doc.get("PRIMARY_GATE") or {}
    rows = PROJ / "artifacts/strategic_demand/sppo" / f"{EVAL_LABEL.lower()}_specialist_crossover_eval_rows.csv"
    cells = {}
    if rows.is_file():
        import csv
        with rows.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = f"{row.get('policy')}@{row.get('pole')}"
                cells[key] = cells.get(key, 0) + 1
    need = {"pi_A@A", "pi_B@A", "pi_A@B", "pi_B@B"}
    out = {
        "record_id": "DUAL_BRANCH_2V2_PHASE2_DIAGNOSTIC",
        "status": "POST_HOC_DIAGNOSTIC",
        "utc": now(),
        "confirmatory": False,
        "stops_the_chain": False,
        "result": str(result.relative_to(PROJ)).replace("\\", "/"),
        "rows": str(rows.relative_to(PROJ)).replace("\\", "/") if rows.is_file() else None,
        "delta_A": (gate.get("delta_A") or {}).get("mean"),
        "delta_B": (gate.get("delta_B") or {}).get("mean"),
        "gate_passes": gate.get("passes"),
        "four_cells": {k: cells.get(k, 0) for k in sorted(need)},
        "four_cells_complete": need <= set(cells),
        "note": "Post-hoc on the frozen 2v2 top-50. A bad Delta does not stop Stage 4.",
    }
    dest = PROJ / "2v2" / "PHASE2_DIAGNOSTIC.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    log(f"Phase 2 diagnostic delta_A={out['delta_A']} delta_B={out['delta_B']} "
        f"gate_passes={out['gate_passes']} four_cells_complete={out['four_cells_complete']}")


def assert_four_teacher_cells(man_path: Path) -> None:
    import numpy as np
    man = json.loads(man_path.read_text(encoding="utf-8"))
    seen: set[tuple[str, str]] = set()
    for sh in man.get("shards") or []:
        z = np.load(PROJ / str(sh["file"]))
        roles = np.asarray(z["roles"])
        pole = str(sh["pole"])
        if (roles < 0.5).any():
            seen.add((pole, "DEFEND"))
        if (roles >= 0.5).any():
            seen.add((pole, "ATTACK"))
    need = {("A", "ATTACK"), ("A", "DEFEND"), ("B", "ATTACK"), ("B", "DEFEND")}
    if not need <= seen:
        fail(f"Stage4 dataset missing teacher cells {sorted(need - seen)}")
    log(f"Stage4 dataset has all four (z, r) cells: {sorted(seen)}")


def stage4_dataset() -> None:
    rc = run_logged(["experiments/prepare_stage4_baselines.py", "--reserve"], "stage4_reserve")
    if rc != 0:
        fail(f"Stage4 --reserve exited {rc}")
    rc = run_logged(
        ["experiments/prepare_stage4_baselines.py", "--team-size", "2", "--collection"],
        "stage4_freeze_collection",
    )
    if rc != 0:
        fail(f"Stage4 collection freeze exited {rc}")
    man = PROJ / "artifacts/strategic_demand/sppo/SUITE_DISTILLATION_2V2_STAGE4_DATASET.json"
    base = [
        "experiments/collect_suite_distillation_states.py",
        "--team-size", "2", "--dataset-tag", "STAGE4", "--device", "cuda",
    ]
    if not man.is_file():
        rc = run_logged(base + ["--smoke"], "stage4_collect_smoke")
        if rc != 0:
            fail(f"Stage4 collect smoke exited {rc}")
        rc = run_logged(base + ["--resume"], "stage4_collect")
        if not man.is_file():
            fail(f"Stage4 collect exited {rc} without {man.name}")
    audit = PROJ / "artifacts/strategic_demand/sppo/SUITE_DATASETS_STAGE4_AUDIT_2V2.json"
    rc = run_logged(
        ["experiments/audit_suite_datasets_cross_scale.py",
         "--scales", "2v2_stage4", "--write",
         "--out", str(audit.relative_to(PROJ))],
        "stage4_audit",
    )
    if rc != 0 or not audit.is_file():
        fail(f"Stage4 dataset audit exited {rc}")
    aud = json.loads(audit.read_text(encoding="utf-8"))
    if aud.get("verdict") != "GREEN":
        fail(f"Stage4 dataset audit is {aud.get('verdict')!r}, not GREEN")
    m = json.loads(man.read_text(encoding="utf-8"))
    if m.get("dataset_mode") != "dual_branch_teachers":
        fail("Stage4 dataset is not dual_branch_teachers")
    if m.get("teachers", {}).get("mode") != "dual_branch_role_gated":
        fail("Stage4 teachers are not dual_branch_role_gated")
    for side in ("pi_A", "pi_B"):
        for half in ("defend", "attack"):
            pin = m["teachers"][side][half]
            if pin["sha256"] in SHA.values():
                fail(f"Stage4 teacher {side}.{half} equals a foundation specialist")
    assert_four_teacher_cells(man)
    log("Stage4 dataset GREEN under dual-branch teachers")


def stage4_students() -> None:
    rc = run_logged(
        ["experiments/prepare_stage4_baselines.py", "--team-size", "2", "--sharing"],
        "stage4_freeze_sharing",
    )
    if rc != 0:
        fail(f"Stage4 sharing freeze exited {rc}")
    for arm in STAGE4_ARMS:
        frozen = (
            PROJ / "artifacts/strategic_demand/sppo/suite_sharing_std/2v2_stage4"
            / ("fully_shared_z_r" if arm == "fully_shared" else arm)
            / "STUDENT_FROZEN.json"
        )
        if frozen.is_file():
            log(f"Stage4 {arm} already frozen")
            continue
        rc = run_logged(
            ["experiments/run_suite_sharing_distillation.py",
             "--arm", arm, "--team-size", "2", "--spec-tag", "STAGE4",
             "--device", "cuda", "--preflight"],
            f"stage4_{arm}_preflight",
        )
        if rc != 0:
            fail(f"Stage4 {arm} preflight exited {rc}")
        rc = run_logged(
            ["experiments/run_suite_sharing_distillation.py",
             "--arm", arm, "--team-size", "2", "--spec-tag", "STAGE4", "--device", "cuda"],
            f"stage4_{arm}_train",
        )
        if not frozen.is_file():
            fail(f"Stage4 {arm} train exited {rc} without STUDENT_FROZEN.json")
        log(f"Stage4 {arm} FROZEN")


def stage4_evals() -> None:
    """Post-hoc top-50 eval of Stage-4 students. Ugly Delta does not stop the suite."""
    rc = run_logged(
        ["experiments/prepare_stage4_baselines.py", "--team-size", "2", "--eval"],
        "stage4_freeze_eval",
    )
    if rc != 0:
        fail(f"Stage4 eval freeze exited {rc}")
    from experiments import prepare_stage4_baselines as P4
    for arm, label_key in (
        ("share_encoder", "share_encoder"),
        ("fully_shared", "fully_shared"),
        ("role_only", "role_only"),
    ):
        lab = P4.labels(2)[label_key]
        result = PROJ / "artifacts/strategic_demand/sppo" / f"{lab}_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
        if result.is_file():
            log(f"Stage4 eval {arm} already sealed")
            continue
        rc = run_logged(
            ["experiments/eval_suite_sharing_crossover.py",
             "--team-size", "2", "--arm", arm, "--spec-tag", "STAGE4",
             "--device", "cuda", "--resume"],
            f"stage4_eval_{arm}",
        )
        if not result.is_file():
            log(f"WARN: Stage4 eval {arm} exited {rc} without result; continuing (diagnostic)")
        else:
            log(f"Stage4 eval {arm} DONE")


def bundle() -> None:
    import shutil
    out = PROJ / "2v2" / "dual_branch_2v2_bundle"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for pol, r in RUNS.items():
        dst = out / "checkpoints" / pol
        dst.mkdir(parents=True)
        if (PROJ / r["final"]).is_file():
            shutil.copy2(PROJ / r["final"], dst / Path(r["final"]).name)
        if (PROJ / r["attack"]).is_file():
            shutil.copy2(PROJ / r["attack"], dst / Path(r["attack"]).name)
    ev = out / "evaluation"
    ev.mkdir()
    sd = PROJ / "artifacts" / "strategic_demand" / "sppo"
    for f in sd.glob(f"{EVAL_LABEL}*"):
        if f.is_file():
            shutil.copy2(f, ev / f.name)
    for f in list(sd.glob("TOP50_2V2_STAGE4*")) + list(sd.glob("STANDARDIZED_2V2_STAGE4*")):
        if f.is_file():
            shutil.copy2(f, ev / f.name)
    diag = PROJ / "2v2" / "PHASE2_DIAGNOSTIC.json"
    if diag.is_file():
        shutil.copy2(diag, ev / diag.name)
    if (PROJ / MANIFEST).is_file():
        shutil.copy2(PROJ / MANIFEST, out / Path(MANIFEST).name)
    if SEAL.is_file():
        shutil.copy2(SEAL, out / SEAL.name)
    if STAGE4_TEACHERS.is_file():
        shutil.copy2(STAGE4_TEACHERS, out / STAGE4_TEACHERS.name)
    if MANIFESTS.is_dir():
        shutil.copytree(MANIFESTS, out / "manifests", dirs_exist_ok=True)
    s4 = out / "stage4"
    s4.mkdir()
    for arm_dir in (sd / "suite_sharing_std" / "2v2_stage4").glob("*"):
        if arm_dir.is_dir():
            shutil.copytree(arm_dir, s4 / arm_dir.name, dirs_exist_ok=True)
    man = sd / "SUITE_DISTILLATION_2V2_STAGE4_DATASET.json"
    if man.is_file():
        shutil.copy2(man, s4 / man.name)
    aud = sd / "SUITE_DATASETS_STAGE4_AUDIT_2V2.json"
    if aud.is_file():
        shutil.copy2(aud, s4 / aud.name)
    (out / "README.txt").write_text(
        "2v2 Ours-Teachers + Stage 4 (Ours-Shared). Same pipeline as 6v6. k=ceil(2/3)=1.\n"
        "Phase 1: Ours-Teachers dual-branch (ATTACK+DEFEND), technical seal.\n"
        "Phase 2: frozen top-50 four-cell diagnostic (post-hoc; not a redesign gate).\n"
        "Phase 3: dataset (o, z, r, teacher_target) from the sealed dual-branch teachers.\n"
        "Phase 4: Share-Encoder / Ours-Shared Fully Shared+z+r / Role-only ablation.\n"
        "Phase 5: student evals on the same top-50. Ugly Delta does not invalidate the run.\n"
        "Section: Strategic Representation Under Parameter Sharing. No Generalist rung.\n",
        encoding="utf-8",
    )
    if BUNDLE.exists():
        BUNDLE.unlink()
    shutil.make_archive(str(BUNDLE.with_suffix("")), "zip", out)
    log(f"bundled {BUNDLE.name}")


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
    state(status="RUNNING", pid=os.getpid(), k_defend=K)
    log("2v2 DUAL_BRANCH + STAGE4 frozen pipeline started (k=1)")
    manifest("phase0_preflight", gate="IMPLEMENTATION_GATE", k_defend=K)

    wait_for_foreign_phase1()

    if not smoke_already("A"):
        if foreign_phase1_owners():
            fail("smoke A would duplicate a live chain")
        rc = run_logged(
            train_args("A", 5000, RUNS["A"]["smoke_seed"], None, RUNS["A"]["suffix"], smoke=True),
            "smoke_A",
        )
        if rc != 0:
            fail(f"A smoke failed exit {rc}")
        manifest("phase1_smoke_A")
        log("A smoke PASS")
    if not smoke_already("B"):
        if foreign_phase1_owners():
            fail("smoke B would duplicate a live chain")
        rc = run_logged(
            train_args("B", 5000, RUNS["B"]["smoke_seed"], None, RUNS["B"]["suffix"], smoke=True),
            "smoke_B",
        )
        if rc != 0:
            fail(f"B smoke failed exit {rc}")
        manifest("phase1_smoke_B")
        log("B smoke PASS")
    if not done("phase1_train"):
        if not (train_done_marked() and finals_ready()):
            train_both()
        if not finals_ready():
            fail("Phase 1 ended without both dual-branch finals")
        mark_train_done()
        manifest("phase1_train", done_marker=str(LEGACY_DONE.relative_to(PROJ)).replace("\\", "/"))
        log("TRAIN DONE")
    if not done("phase1_export"):
        for pol, r in RUNS.items():
            if not (PROJ / r["attack"]).is_file():
                export_attack_branch(r["final"], r["attack"])
            else:
                log(f"ATTACK export already present: {r['attack']}")
        write_manifest()
        manifest("phase1_export", deploy_manifest=MANIFEST)
    if not done("phase1_technical_seal"):
        technical_seal()
        manifest("phase1_technical_seal", seal=str(SEAL.relative_to(PROJ)).replace("\\", "/"))
        state(status="TECHNICALLY_SEALED")

    if not done("phase2_diagnostic"):
        result = PROJ / "artifacts/strategic_demand/sppo" / f"{EVAL_LABEL}_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
        if not result.is_file():
            rc = run_logged(eval_args() + ["--dry-run"], "eval_dryrun")
            if rc != 0:
                fail(f"evaluation dry-run failed exit {rc}")
            log("evaluation dry-run passed")
            rc = run_logged(eval_args() + ["--resume"], "eval")
            if not result.is_file():
                fail(f"evaluation exited {rc} without {result.name}")
        write_phase2_diagnostic(result)
        manifest("phase2_diagnostic", label="POST_HOC_DIAGNOSTIC", stops_the_chain=False)
        log("EVAL DONE (post-hoc top-50 diagnostic; ugly Delta does NOT stop Stage 4)")

    if not done("phase3_dataset"):
        if not SEAL.is_file() or json.loads(SEAL.read_text(encoding="utf-8")).get("status") != "SEALED":
            fail("Stage 4 requires TECHNICAL SEAL = SEALED")
        stage4_dataset()
        manifest("phase3_dataset")

    if not done("phase4_students"):
        stage4_students()
        manifest("phase4_students", ladder=["Separated=teachers", "share_encoder", "fully_shared_z_r", "role_only"])

    if not done("phase5_evals"):
        stage4_evals()
        manifest("phase5_evals", stops_the_chain=False)

    if not done("phase6_package"):
        bundle()
        manifest("phase6_package", zip=str(BUNDLE.relative_to(PROJ)).replace("\\", "/"))
        state(status="DONE")
    log(f"DONE -- full 2v2 pipeline. Zip: {BUNDLE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
