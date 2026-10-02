r"""6v6 frozen pipeline: Ours-Teachers -> Stage 3 diagnostics -> Stage 4 Ours-Shared.

Resumable. Technical integrity gate only -- never stops because Delta looks bad.

    cd <repo>; git pull; cd AICTFProject
    .venv\Scripts\python.exe 6v6\run_dual_branch_6v6.py --check
    powershell -ExecutionPolicy Bypass -File 6v6\run_dual_branch_6v6.ps1

Phases (STATE.json records each completion):
  Phase 1  smoke A/B, 200k A, 200k B, export ATTACK, write manifest, TECHNICAL SEAL
           (= Ours-Teachers)
  Phase 2  old top-50 four-cell diagnostic (post-hoc; not a redesign gate)
  Phase 2b post-hoc matched-128 on historical block 25800001..25800128
  Phase 2c dual-branch own top-50 (historical rule; CPU; no new episodes)
  Phase 3  Stage-4 dataset from dual-branch teachers
  Phase 4  Share-Encoder -> Ours-Shared Fully Shared+z+r -> Role-only ablation
  Phase 5  Stage-4 student evals (post-hoc top-50)
  Phase 6  populate 6v6/FOR_PROFESSOR/

k=ceil(6/3)=2. Do not run run_symmetric_6v6.py. Do not distill Generalist.
Framing: EXPERIMENTAL_FRAMING_OURS_TEACHERS_SHARED_V1.json
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
from typing import Any

PROJ = Path(__file__).resolve().parents[1]
REPO = PROJ.parent
os.chdir(PROJ)
sys.path.insert(0, str(PROJ))

from experiments.tqdm_loop import set_postfix, tqdm_iter  # noqa: E402

LOG = PROJ / "6v6" / "dual_branch_6v6.log"
STATE = PROJ / "6v6" / "dual_branch_6v6_STATE.json"
OVERALL_ERR = PROJ / "6v6" / "dual_branch_OVERALL.log.err"
OVERALL_JSON = PROJ / "6v6" / "dual_branch_OVERALL_PROGRESS.json"
SEAL = PROJ / "6v6" / "DUAL_BRANCH_6V6_TECHNICAL_SEAL.json"
STAGE4_TEACHERS = PROJ / "artifacts" / "strategic_demand" / "sppo" / "STAGE4_6V6_TEACHERS_SEALED.json"
SPEC = "artifacts/strategic_demand/sppo/DUAL_BRANCH_ROLE_COMPOSITE_V1_SPEC.json"
AUTH = "artifacts/strategic_demand/sppo/STAGE4_6V6_SCHOOL_SUITE_SPEC.json"
EVAL_SPEC = "artifacts/strategic_demand/sppo/DUAL_BRANCH_6V6_SCHOOL_DIAGNOSTIC_SPEC.json"
EVAL_LABEL = "DUAL_BRANCH_6V6_ROLE_COMPOSITE"
EVAL_REG = "STANDARDIZED_6V6_SEPARATED_EVAL"
SEEDS_FILE = "artifacts/strategic_demand/sppo/symmetric_role_top50/6v6_ours_top50_seed_ids.json"
MATCHED128_SPEC = "artifacts/strategic_demand/sppo/DUAL_BRANCH_6V6_POSTHOC_MATCHED128_SPEC.json"
MATCHED128_LABEL = "POSTHOC_MATCHED128_6V6_DUAL_BRANCH"
MATCHED128_DIR = "artifacts/strategic_demand/sppo/dual_branch_v1/matched128_6v6"
MATCHED128_SEED_BASE = 25800001
MATCHED128_N = 128
OWN_TOP50_LABEL = "DUAL_BRANCH_6V6_OWN_TOP50"
OLD_OURS_ROWS = "artifacts/strategic_demand/sppo/standardized_6v6_split_k1_confirmatory_specialist_crossover_eval_rows.csv"
PRIMARY_6V6 = "artifacts/strategic_demand/sppo/STANDARDIZED_6V6_SPLIT_K1_CONFIRMATORY_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
MANIFEST = "6v6/dual_branch_deploy_manifest.json"
BUNDLE = PROJ / "6v6" / "dual_branch_6v6_results.zip"
STAGE4_OUT = PROJ / "6v6" / "stage4_baseline_suite"
PY = str(PROJ / ".venv" / "Scripts" / "python.exe")

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

STAGE4_ARMS = ("share_encoder", "fully_shared", "role_only")

# Weighted units for the suite-level ETA bar (roughly proportional to wall time).
W_SMOKE = 5_000
W_TRAIN = 200_000
W_EVAL_CELLS = 50 * 4          # old top-50 four-cell diagnostic
W_MATCHED128 = 128 * 4         # post-hoc matched-128 (fair overall comparison)
W_OWN_TOP50 = 1                # CPU re-rank of sealed matched-128 rows
W_COLLECT = 96 * 2             # Stage-4 dataset episodes
W_DISTILL = 20                # distillation epochs per arm
W_STAGE4_EVAL = 3 * (50 * 4)  # three student arms
W_BOOKKEEP = 1

# Filled by main(); child helpers advance the suite-level bar through run_logged.
PROGRESS: dict[str, Any] = {"bar": None, "base": 0, "total": 1}


def _advance_bookkeeping(tag: str, weight: int = W_BOOKKEEP) -> None:
    bar = PROGRESS.get("bar")
    if bar is None:
        return
    total = int(PROGRESS["total"])
    base = int(PROGRESS["base"])
    target = min(base + weight, total)
    delta = target - int(bar.n)
    if delta > 0:
        bar.update(delta)
    set_postfix(bar, f"{tag} done")
    PROGRESS["base"] = target
    _heartbeat(tag, target, total, "stage_done")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"{now()} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    with OVERALL_ERR.open("a", encoding="utf-8") as fh:
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
    e["PYTHONUNBUFFERED"] = "1"
    return e


def _read_global_step(metrics: Path | None) -> int:
    if metrics is None or not metrics.is_file():
        return 0
    try:
        with metrics.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return 0
        last = rows[-1]
        for key in ("global_step", "timesteps", "total_timesteps", "step"):
            if key in last and last[key] not in (None, ""):
                return int(float(last[key]))
    except Exception:
        return 0
    return 0


def _heartbeat(phase: str, overall_done: int, overall_total: int, detail: str) -> None:
    payload = {
        "utc": now(),
        "phase": phase,
        "overall_done": overall_done,
        "overall_total": overall_total,
        "frac": (overall_done / overall_total) if overall_total else 0.0,
        "detail": detail,
    }
    OVERALL_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    line = (
        f"[{payload['utc']}] phase={phase} "
        f"overall={overall_done}/{overall_total} "
        f"({100.0 * payload['frac']:.1f}%) {detail}\n"
    )
    with OVERALL_ERR.open("a", encoding="utf-8") as fh:
        fh.write(line)


def run_logged(
    argv: list[str],
    tag: str,
    *,
    metrics_rel: str | None = None,
    weight: int = W_BOOKKEEP,
    overall_base: int = 0,
    overall_total: int = 1,
    overall_bar: Any | None = None,
) -> int:
    """Run a child; optional metrics polling advances the suite-level overall bar."""
    out = PROJ / "6v6" / f"dual_branch_{tag}.log"
    err = PROJ / "6v6" / f"dual_branch_{tag}.log.err"
    log(f"exec: {' '.join(argv)}")
    log(f"  stage tqdm -> {err.relative_to(PROJ)}")
    metrics = PROJ / metrics_rel if metrics_rel else None
    with out.open("w", encoding="utf-8") as fo, err.open("w", encoding="utf-8") as fe:
        proc = subprocess.Popen([PY, *argv], cwd=str(PROJ), env=env(), stdout=fo, stderr=fe)
        last_report = -1
        try:
            while True:
                rc = proc.poll()
                step = min(max(_read_global_step(metrics), 0), weight) if metrics else 0
                if overall_bar is not None:
                    overall_now = min(overall_base + (step if metrics else 0), overall_total)
                    delta = overall_now - int(overall_bar.n)
                    if delta > 0:
                        overall_bar.update(min(delta, overall_total - int(overall_bar.n)))
                    set_postfix(overall_bar, f"{tag} {step}/{weight}" if metrics else tag)
                if metrics and step != last_report and (
                    step - last_report >= max(1, weight // 200) or rc is not None
                ):
                    _heartbeat(tag, overall_base + step, overall_total, f"stage_step={step}/{weight}")
                    last_report = step
                if rc is not None:
                    break
                time.sleep(10.0 if metrics else 2.0)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
        rc = int(proc.returncode or 0)
    if overall_bar is not None:
        target = min(overall_base + weight, overall_total)
        delta = target - int(overall_bar.n)
        if delta > 0:
            overall_bar.update(delta)
        set_postfix(overall_bar, f"{tag} done")
        _heartbeat(tag, target, overall_total, "stage_done")
    return rc


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


def check() -> tuple[list[str], list[str]]:
    problems: list[str] = []
    notes: list[str] = []
    if not Path(PY).is_file():
        problems.append("missing .venv")
    if not (PROJ / SPEC).is_file():
        problems.append(f"missing {SPEC}")
    else:
        spec = json.loads((PROJ / SPEC).read_text(encoding="utf-8"))
        if "PASSED" not in str((spec.get("JOINT_200K_TRAINING_locked") or {}).get("IMPLEMENTATION_GATE", {}).get("status", "")):
            problems.append("DUAL_BRANCH IMPLEMENTATION_GATE has not PASSED")
        notes.append(f"spec status={spec.get('status')}")
    if not (PROJ / AUTH).is_file():
        problems.append(f"missing {AUTH}")
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
    primary = PROJ / PRIMARY_6V6
    if not primary.is_file():
        problems.append(f"missing post-hoc primary record {Path(PRIMARY_6V6).name}")
    if not (PROJ / OLD_OURS_ROWS).is_file():
        problems.append(f"missing old Ours 128 rows {Path(OLD_OURS_ROWS).name}")
    try:
        rc = subprocess.run(
            [PY, "experiments/select_own_top50.py", "--verify-historical", "--scale", "6"],
            cwd=str(PROJ), capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if rc.returncode != 0:
            problems.append(f"own-top50 historical rule failed for 6v6: {(rc.stderr or rc.stdout)[:200]}")
        else:
            notes.append("own-top50 historical rule reproduces frozen 6v6 list")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"own-top50 verify failed: {exc}")
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
        b128 = reg.get(EVAL_REG)
        if b128 is not None and (int(b128["lo"]), int(b128["hi"])) != (MATCHED128_SEED_BASE, MATCHED128_SEED_BASE + MATCHED128_N - 1):
            problems.append(f"{EVAL_REG} range is not {MATCHED128_SEED_BASE}..{MATCHED128_SEED_BASE + MATCHED128_N - 1}")
        from experiments import prepare_stage4_baselines as P4
        for _k, (eid, lo, hi) in P4.blocks(6).items():
            b = reg.get(eid)
            if b is None:
                problems.append(f"Stage4 seed block {eid} missing (run prepare_stage4_baselines --reserve)")
            elif (b["lo"], b["hi"]) != (lo, hi):
                problems.append(f"{eid} range mismatch")
            else:
                notes.append(f"Stage4 block {eid} OK")
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
        # Readable: load both branches into a probe env.
        try:
            R2.AGENTS = 6
            probe = R2.build_env("cpu", 99_999_001)
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
            pin_d, pin_a = man.get(f"pi_{pol}_defend") or {}, man.get(f"pi_{pol}_attack") or {}
            checks[f"manifest_{pol}_defend_sha"] = pin_d.get("sha256") == sha(r["final"])
            checks[f"manifest_{pol}_attack_sha"] = pin_a.get("sha256") == sha(r["attack"])

    # Stage-4 output dirs must not collide with historical asymmetric / SYM suites.
    hist = [
        PROJ / "artifacts/strategic_demand/sppo/suite_sharing_std/6v6",
        PROJ / "artifacts/strategic_demand/sppo/suite_sharing_std/6v6_sym",
    ]
    stage4_root = PROJ / "artifacts/strategic_demand/sppo/suite_sharing_std/6v6_stage4"
    checks["stage4_output_isolated"] = True  # path name itself isolates; refuse writing into hist
    detail["stage4_root"] = str(stage4_root.relative_to(PROJ)).replace("\\", "/")
    detail["historical_roots_not_written"] = [str(h.relative_to(PROJ)).replace("\\", "/") for h in hist]

    ok = all(checks.values())
    seal = {
        "record_id": "DUAL_BRANCH_6V6_TECHNICAL_SEAL",
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
    # Teachers sealed for Stage 4 prepare.
    man = json.loads((PROJ / MANIFEST).read_text(encoding="utf-8"))
    STAGE4_TEACHERS.write_text(json.dumps({
        "record_id": "STAGE4_6V6_TEACHERS_SEALED",
        "status": "SEALED",
        "utc": now(),
        "parent": MANIFEST,
        "technical_seal": str(SEAL.relative_to(PROJ)).replace("\\", "/"),
        "pins": {
            "pi_A_defend": man["pi_A_defend"],
            "pi_A_attack": man["pi_A_attack"],
            "pi_B_defend": man["pi_B_defend"],
            "pi_B_attack": man["pi_B_attack"],
        },
        "teachers_are_dual_branch_composites": True,
        "not_foundation_specialists": True,
    }, indent=2) + "\n", encoding="utf-8")
    log(f"TECHNICAL SEAL PASS -> {SEAL.name}")
    return seal


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


def write_matched128_spec() -> None:
    """Pin the post-hoc matched-128 command to the sealed dual-branch deploy manifest."""
    man = json.loads((PROJ / MANIFEST).read_text(encoding="utf-8"))
    hi = MATCHED128_SEED_BASE + MATCHED128_N - 1
    rows = f"artifacts/strategic_demand/sppo/{MATCHED128_LABEL.lower()}_specialist_crossover_eval_rows.csv"
    eval_cmd = (
        f".venv/Scripts/python.exe experiments/eval_specialist_crossover_scaled.py "
        f"--team-size 6 --spec {MATCHED128_SPEC} --post-hoc-ablation-spec {MATCHED128_SPEC} "
        f"--seed-base {MATCHED128_SEED_BASE} --n-seeds {MATCHED128_N} "
        f"--registry-experiment-id {EVAL_REG} --label {MATCHED128_LABEL} --device cuda "
        f"--pi-a-path {RUNS['A']['final']} --pi-b-path {RUNS['B']['final']} "
        f"--role-fixed-for-episode --role-k-defend {K} "
        f"--frozen-attack-path {RUNS['A']['attack']} "
        f"--frozen-attack-path-sha256 {man['pi_A_attack']['sha256']} "
        f"--frozen-attack-path-b {RUNS['B']['attack']} "
        f"--frozen-attack-path-b-sha256 {man['pi_B_attack']['sha256']} "
        f"--dual-branch-deploy-manifest {MANIFEST} --resume"
    )
    doc = {
        "record_id": "DUAL_BRANCH_6V6_POSTHOC_MATCHED128_SPEC",
        "status": "FROZEN_DIAGNOSTIC",
        "arm": "POST_HOC_ABLATION",
        "confirmatory": False,
        "utc": now(),
        "classification": (
            "POST-HOC MATCHED EVALUATION ON THE HISTORICAL 128-SEED 6V6 BLOCK. Not confirmatory. "
            "Not fresh seeds. Not PAPER-FAITHFUL. Does not replace any future untouched evaluation "
            "of the dual-branch method."
        ),
        "decided_by": (
            "PI, 2026-10-02: same three-view pattern as 2v2 — old top-50 (provenance), "
            "dual-branch own top-50 (best-case capability), matched 128 (fair overall comparison)."
        ),
        "governed_by": [
            "DUAL_BRANCH_ROLE_COMPOSITE_V1_SPEC.json",
            "DUAL_BRANCH_6V6_SCHOOL_DIAGNOSTIC_SPEC.json",
            "EXPERIMENTAL_FRAMING_OURS_TEACHERS_SHARED_V1.json",
        ],
        "THE_QUESTION": (
            "On the identical 128 historical 6v6 seeds, how do the dual-branch role composite's "
            "crossover deltas (win rate and score margin) compare with the old asymmetric Ours "
            "sealed on that block?"
        ),
        "SYSTEM_locked": {
            "deploy_manifest": MANIFEST,
            "pi_A_defend": man["pi_A_defend"],
            "pi_A_attack": man["pi_A_attack"],
            "pi_B_defend": man["pi_B_defend"],
            "pi_B_attack": man["pi_B_attack"],
            "k_defend": K,
            "allocator": f"CLOSEST_DEFENDS(k={K}), roles fixed for the episode",
            "unchanged_from_top50_diagnostic": (
                "every flag of the DUAL_BRANCH_6V6_ROLE_COMPOSITE command except the seeds "
                "(full block instead of --seed-list) and the label"
            ),
        },
        "POST_HOC_MATCHED_ROLE_ABLATIONS": {
            MATCHED128_LABEL: {
                "registry_experiment_id": EVAL_REG,
                "block": f"{MATCHED128_SEED_BASE}..{hi}",
                "primary_record": Path(PRIMARY_6V6).name,
                "system": "dual-branch role composite",
                "frozen_attack_A": man["pi_A_attack"],
                "frozen_attack_B": man["pi_B_attack"],
                "episodes": MATCHED128_N * 4,
                "note": (
                    "all four cells (A@A, B@A, A@B, B@B) on all 128 seeds of the SPENT block; "
                    "no seed spent, block status unchanged"
                ),
            }
        },
        "READING": {
            "per_system": (
                "V(A,A), V(B,A), V(A,B), V(B,B); Delta_A = V(A,A) - V(B,A); "
                "Delta_B = V(B,B) - V(A,B); the same deltas on score margin (blue - red)"
            ),
            "comparison": (
                f"paired within seed against the old asymmetric Ours rows sealed on this block "
                f"({Path(OLD_OURS_ROWS).name})"
            ),
            "statistics": (
                "mean +- std (ddof=1) per delta; 95% paired percentile bootstrap "
                "(n=20000, alpha=0.05, rng 7) via eval_hog_psp_v3._mean_ci"
            ),
            "own_top50": (
                "after this seal, experiments/select_own_top50.py --scale 6 applies the historical "
                "rule to these rows (best-case capability; not an unbiased estimate)"
            ),
        },
        "LAUNCH": {
            "when": "inside 6v6/run_dual_branch_6v6.py after the technical seal and old top-50 diagnostic",
            "eval": eval_cmd,
            "readout": (
                f".venv/Scripts/python.exe experiments/readout_posthoc_matched_crossover.py "
                f"--spec {MATCHED128_SPEC}"
            ),
            "own_top50": (
                f".venv/Scripts/python.exe experiments/select_own_top50.py --rows {rows} "
                f"--label {OWN_TOP50_LABEL} --out {MATCHED128_DIR} --scale 6"
            ),
        },
        "NOT_AUTHORIZED_BY_THIS_SPEC": [
            "changing dual-branch training, k, or any branch",
            "tuning anything in response to this result",
            "calling these seeds fresh or this evaluation confirmatory",
            "replacing a future untouched evaluation of the dual-branch method with this result",
        ],
        "READOUT_SYSTEMS": {
            "dual_branch": rows,
            "old_asymmetric_ours": OLD_OURS_ROWS,
        },
        "READOUT_OUT": f"{MATCHED128_DIR}/MATCHED128_6V6_READOUT",
    }
    out = PROJ / MATCHED128_SPEC
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    (PROJ / MATCHED128_DIR).mkdir(parents=True, exist_ok=True)
    log(f"wrote {MATCHED128_SPEC}")


def matched128_args() -> list[str]:
    return [
        "experiments/eval_specialist_crossover_scaled.py",
        "--team-size", "6", "--spec", MATCHED128_SPEC, "--post-hoc-ablation-spec", MATCHED128_SPEC,
        "--seed-base", str(MATCHED128_SEED_BASE), "--n-seeds", str(MATCHED128_N),
        "--registry-experiment-id", EVAL_REG, "--label", MATCHED128_LABEL, "--device", "cuda",
        "--pi-a-path", RUNS["A"]["final"], "--pi-b-path", RUNS["B"]["final"],
        "--role-fixed-for-episode", "--role-k-defend", str(K),
        "--frozen-attack-path", RUNS["A"]["attack"],
        "--frozen-attack-path-sha256", sha(RUNS["A"]["attack"]),
        "--frozen-attack-path-b", RUNS["B"]["attack"],
        "--frozen-attack-path-b-sha256", sha(RUNS["B"]["attack"]),
        "--dual-branch-deploy-manifest", MANIFEST,
    ]


def matched128_eval() -> None:
    """Post-hoc matched-128 on the historical SPENT block, then paired readout vs old Ours."""
    write_matched128_spec()
    result = PROJ / "artifacts/strategic_demand/sppo" / f"{MATCHED128_LABEL}_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
    done_mark = PROJ / MATCHED128_DIR / "MATCHED128_DONE.txt"
    if not result.is_file():
        rc = run_logged(matched128_args() + ["--dry-run"], "matched128_dryrun")
        if rc != 0:
            fail(f"matched-128 dry-run failed exit {rc}")
        log("matched-128 dry-run passed")
        base = int(PROGRESS["base"])
        rc = run_logged(
            matched128_args() + ["--resume"],
            "matched128",
            weight=W_MATCHED128,
            overall_base=base,
            overall_total=int(PROGRESS["total"]),
            overall_bar=PROGRESS["bar"],
        )
        PROGRESS["base"] = base + W_MATCHED128
        if not result.is_file():
            fail(f"matched-128 exited {rc} without {result.name}")
    else:
        log(f"matched-128 already sealed: {result.name}")
        _advance_bookkeeping("matched128", W_MATCHED128)
    rc = run_logged(
        ["experiments/readout_posthoc_matched_crossover.py", "--spec", MATCHED128_SPEC],
        "matched128_readout",
    )
    if rc != 0:
        fail(f"matched-128 readout failed exit {rc}")
    done_mark.write_text(f"DONE {now()}\n", encoding="utf-8")
    log(f"MATCHED128 DONE -> {MATCHED128_DIR}/MATCHED128_6V6_READOUT.md")


def own_top50() -> None:
    """Historical top-50 rule on dual-branch's own matched-128 rows (CPU; no new episodes)."""
    rows = PROJ / "artifacts/strategic_demand/sppo" / f"{MATCHED128_LABEL.lower()}_specialist_crossover_eval_rows.csv"
    if not rows.is_file():
        fail(f"own-top50 needs matched-128 rows: {rows.name}")
    out_dir = PROJ / MATCHED128_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / f"{OWN_TOP50_LABEL}.md"
    done_mark = out_dir / "OWN_TOP50_DONE.txt"
    if md.is_file() and done_mark.is_file():
        log(f"own-top50 already written: {md.name}")
        _advance_bookkeeping("own_top50", W_OWN_TOP50)
        return
    rc = run_logged(
        [
            "experiments/select_own_top50.py",
            "--rows", str(rows.relative_to(PROJ)).replace("\\", "/"),
            "--label", OWN_TOP50_LABEL,
            "--out", MATCHED128_DIR,
            "--scale", "6",
        ],
        "own_top50",
    )
    if rc != 0 or not md.is_file():
        fail(f"own-top50 selection failed exit {rc}")
    done_mark.write_text(f"DONE {now()}\n", encoding="utf-8")
    log(f"OWN_TOP50 DONE -> {MATCHED128_DIR}/{OWN_TOP50_LABEL}.md")
    _advance_bookkeeping("own_top50", W_OWN_TOP50)


def train_both() -> None:
    for pol in ("A", "B"):
        r = RUNS[pol]
        final = PROJ / r["final"]
        metrics_rel = f"{r['run_dir']}/metrics.csv"
        if final.is_file():
            log(f"{pol} already built: {r['final']}")
            _advance_bookkeeping(f"train_{pol}", W_TRAIN)
            continue
        ckpts = sorted((PROJ / r["run_dir"] / "ckpts").glob("ckpt_*.zip")) if (PROJ / r["run_dir"] / "ckpts").is_dir() else []
        argv = train_args(pol, 200_000, r["seed"], r["eid"], r["suffix"], smoke=False)
        if ckpts:
            i = argv.index("--load-path")
            del argv[i:i + 2]
            resume = str(ckpts[-1].relative_to(PROJ)).replace("\\", "/")
            argv += ["--resume", resume]
            log(f"{pol} resuming from {ckpts[-1].name}")
        base = int(PROGRESS["base"])
        rc = run_logged(
            argv,
            f"train_{pol}",
            metrics_rel=metrics_rel,
            weight=W_TRAIN,
            overall_base=base,
            overall_total=int(PROGRESS["total"]),
            overall_bar=PROGRESS["bar"],
        )
        PROGRESS["base"] = base + W_TRAIN
        if not final.is_file():
            fail(f"{pol} training exited {rc} without final zip {r['final']}")
        log(f"{pol} TRAIN DONE sha={sha(r['final'])[:16]}...")


def stage4_dataset() -> None:
    rc = run_logged(["experiments/prepare_stage4_baselines.py", "--reserve"], "stage4_reserve")
    if rc != 0:
        fail(f"Stage4 --reserve exited {rc}")
    rc = run_logged(
        ["experiments/prepare_stage4_baselines.py", "--team-size", "6", "--collection"],
        "stage4_freeze_collection",
    )
    if rc != 0:
        fail(f"Stage4 collection freeze exited {rc}")
    man = PROJ / "artifacts/strategic_demand/sppo/SUITE_DISTILLATION_6V6_STAGE4_DATASET.json"
    base = [
        "experiments/collect_suite_distillation_states.py",
        "--team-size", "6", "--dataset-tag", "STAGE4", "--device", "cuda",
    ]
    if not man.is_file():
        rc = run_logged(base + ["--smoke"], "stage4_collect_smoke")
        if rc != 0:
            fail(f"Stage4 collect smoke exited {rc}")
        rc = run_logged(base + ["--resume"], "stage4_collect")
        if not man.is_file():
            fail(f"Stage4 collect exited {rc} without {man.name}")
    audit = PROJ / "artifacts/strategic_demand/sppo/SUITE_DATASETS_STAGE4_AUDIT_6V6.json"
    rc = run_logged(
        ["experiments/audit_suite_datasets_cross_scale.py",
         "--scales", "6v6_stage4", "--write", "--out", str(audit.relative_to(PROJ))],
        "stage4_audit",
    )
    if rc != 0 or not audit.is_file():
        fail(f"Stage4 dataset audit exited {rc}")
    aud = json.loads(audit.read_text(encoding="utf-8"))
    if aud.get("verdict") != "GREEN":
        fail(f"Stage4 dataset audit is {aud.get('verdict')!r}, not GREEN")
    # Integrity: teachers must be dual-branch, not foundations.
    m = json.loads(man.read_text(encoding="utf-8"))
    if m.get("dataset_mode") != "dual_branch_teachers":
        fail("Stage4 dataset is not dual_branch_teachers")
    if m.get("teachers", {}).get("mode") != "dual_branch_role_gated":
        fail("Stage4 teachers are not dual_branch_role_gated")
    for side in ("pi_A", "pi_B"):
        for half in ("defend", "attack"):
            pin = m["teachers"][side][half]
            if pin["sha256"] in SHA.values():
                fail(f"Stage4 teacher {side}.{half} equals a foundation specialist -- wrong teachers")
    log("Stage4 dataset GREEN under dual-branch teachers")


def stage4_students() -> None:
    rc = run_logged(
        ["experiments/prepare_stage4_baselines.py", "--team-size", "6", "--sharing"],
        "stage4_freeze_sharing",
    )
    if rc != 0:
        fail(f"Stage4 sharing freeze exited {rc}")
    for arm in STAGE4_ARMS:
        frozen = (
            PROJ / "artifacts/strategic_demand/sppo/suite_sharing_std/6v6_stage4"
            / ("fully_shared_z_r" if arm == "fully_shared" else arm)
            / "STUDENT_FROZEN.json"
        )
        if frozen.is_file():
            log(f"Stage4 {arm} already frozen")
            _advance_bookkeeping(f"stage4_{arm}", W_DISTILL)
            continue
        for mode in ("--preflight",):
            rc = run_logged(
                ["experiments/run_suite_sharing_distillation.py",
                 "--arm", arm, "--team-size", "6", "--spec-tag", "STAGE4",
                 "--device", "cuda", mode],
                f"stage4_{arm}_preflight",
            )
            if rc != 0:
                fail(f"Stage4 {arm} preflight exited {rc}")
        rc = run_logged(
            ["experiments/run_suite_sharing_distillation.py",
             "--arm", arm, "--team-size", "6", "--spec-tag", "STAGE4", "--device", "cuda"],
            f"stage4_{arm}_train",
        )
        if not frozen.is_file():
            fail(f"Stage4 {arm} train exited {rc} without STUDENT_FROZEN.json")
        log(f"Stage4 {arm} FROZEN")
        _advance_bookkeeping(f"stage4_{arm}", W_DISTILL)


def stage4_evals() -> None:
    """Post-hoc top-50 eval of Stage-4 students. Diagnostic; ugly Delta does not stop the suite."""
    rc = run_logged(
        ["experiments/prepare_stage4_baselines.py", "--team-size", "6", "--eval"],
        "stage4_freeze_eval",
    )
    if rc != 0:
        fail(f"Stage4 eval freeze exited {rc}")
    # Reuse eval_specialist path via sharing eval if available; otherwise record deferred.
    eval_spec = PROJ / "artifacts/strategic_demand/sppo/STANDARDIZED_6V6_STAGE4_SHARING_EVAL_SPEC.json"
    if not eval_spec.is_file():
        fail(f"missing {eval_spec.name}")
    # Best-effort: run eval_suite_sharing_crossover with STAGE4 tag when the CLI supports it.
    try:
        from experiments import eval_suite_sharing_crossover as EV  # noqa: F401
        has_tag = True
    except Exception:  # noqa: BLE001
        has_tag = False
    if has_tag:
        for arm, label_key in (
            ("share_encoder", "share_encoder"),
            ("fully_shared", "fully_shared"),
            ("role_only", "role_only"),
        ):
            from experiments import prepare_stage4_baselines as P4
            lab = P4.labels(6)[label_key]
            result = PROJ / "artifacts/strategic_demand/sppo" / f"{lab}_CROSSOVER_EVAL_RESULT.json"
            if result.is_file():
                log(f"Stage4 eval {arm} already sealed")
                continue
            argv = [
                "experiments/eval_suite_sharing_crossover.py",
                "--team-size", "6", "--arm", arm, "--spec-tag", "STAGE4",
                "--device", "cuda", "--resume",
            ]
            rc = run_logged(argv, f"stage4_eval_{arm}")
            if not result.is_file():
                log(f"WARN: Stage4 eval {arm} exited {rc} without result; continuing (diagnostic)")
            else:
                log(f"Stage4 eval {arm} DONE")
    else:
        log("Stage4 eval CLI unavailable; evals deferred to post-bundle tooling")


def _cp_file(src: Path, dst_dir: Path) -> None:
    import shutil
    if src.is_file():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)


def _write_summary(out: Path) -> None:
    """Plain-text summary a professor can open first."""
    sd = PROJ / "artifacts" / "strategic_demand" / "sppo"
    lines = [
        "6v6 dual-branch + Stage-4 summary",
        "=================================",
        f"written_utc: {now()}",
        f"k_defend: {K}  (= ceil(6/3))",
        "pipeline: smoke -> dual-branch train A/B -> export ATTACK -> technical seal",
        "          -> Stage-3 old top-50 -> matched-128 -> own top-50",
        "          -> Stage-4 dataset/students/evals",
        "",
        "Rule: if it is not in this FOR_PROFESSOR folder, you do not need it.",
        "",
        "Three Stage-3 views (same rule as 2v2):",
        "  old top-50          provenance / matched diagnostic on the old system's best 50",
        "  dual-branch own-50  best-case strategic capability (A@A/B@B pushed toward 1 by construction;",
        "                      informative cells are B@A and A@B)",
        "  matched 128         fair overall comparison on the identical historical block",
        "",
    ]
    result = sd / f"{EVAL_LABEL}_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
    if result.is_file():
        rec = json.loads(result.read_text(encoding="utf-8"))
        gate = rec.get("PRIMARY_GATE") or {}
        dA = (gate.get("delta_A") or {})
        dB = (gate.get("delta_B") or {})
        lines += [
            "Stage-3 old top-50 diagnostic (post-hoc; not a redesign gate)",
            f"  status: {rec.get('status')}",
            f"  Delta_A mean: {dA.get('mean')}",
            f"  Delta_B mean: {dB.get('mean')}",
            f"  record: {result.name}",
            "",
        ]
    own_md = PROJ / MATCHED128_DIR / f"{OWN_TOP50_LABEL}.md"
    if own_md.is_file():
        lines += [
            "Dual-branch own top-50 (best-case capability; historical rule)",
            f"  report: MATCHED128/{OWN_TOP50_LABEL}.md",
            "",
        ]
    readout = PROJ / MATCHED128_DIR / "MATCHED128_6V6_READOUT.md"
    if readout.is_file():
        lines += [
            "Matched 128 (fair overall comparison)",
            "  report: MATCHED128/MATCHED128_6V6_READOUT.md",
            "",
        ]
    if SEAL.is_file():
        seal = json.loads(SEAL.read_text(encoding="utf-8"))
        lines += [
            "Technical seal",
            f"  status: {seal.get('status')}",
            f"  file: {SEAL.name}",
            "",
        ]
    arms_dir = sd / "suite_sharing_std" / "6v6_stage4"
    if arms_dir.is_dir():
        lines.append("Stage-4 students")
        for arm_dir in sorted(arms_dir.glob("*")):
            if not arm_dir.is_dir():
                continue
            fr = arm_dir / "STUDENT_FROZEN.json"
            st = "?"
            if fr.is_file():
                st = json.loads(fr.read_text(encoding="utf-8")).get("status", "?")
            lines.append(f"  {arm_dir.name}: {st}")
        lines.append("")
    lines += [
        "Folders",
        "  TEACHERS/            dual-branch DEFEND finals + exported ATTACK branches",
        "  STAGE3_EVALUATION/   old top-50 four-cell diagnostic result / rows / audit",
        "  MATCHED128/          matched-128 readout + dual-branch own top-50",
        "  STAGE4_SHARING/      Strategic Representation Under Parameter Sharing:\n"
        "                      Share-Encoder (comparison), Ours-Shared Fully Shared+z+r,\n"
        "                      Role-only ablation + dataset",
        "  SEALS/               technical seal, teacher seal, deploy manifest",
        "  PROVENANCE/          STATE, overall progress logs, pipeline log",
        "  SUMMARY/             this file",
        "",
    ]
    (out / "SUMMARY").mkdir(parents=True, exist_ok=True)
    (out / "SUMMARY" / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def bundle() -> None:
    """Populate 6v6/FOR_PROFESSOR/ so the professor can zip that folder in File Explorer."""
    import shutil

    out = PROJ / "6v6" / "FOR_PROFESSOR"
    if out.exists():
        shutil.rmtree(out)
    for name in ("SUMMARY", "TEACHERS", "STAGE3_EVALUATION", "MATCHED128", "STAGE4_SHARING", "SEALS", "PROVENANCE"):
        (out / name).mkdir(parents=True)

    # Teachers: dual-branch DEFEND finals + exported ATTACK branches.
    for pol, r in RUNS.items():
        dst = out / "TEACHERS" / pol
        dst.mkdir(parents=True)
        shutil.copy2(PROJ / r["final"], dst / Path(r["final"]).name)
        shutil.copy2(PROJ / r["attack"], dst / Path(r["attack"]).name)

    sd = PROJ / "artifacts" / "strategic_demand" / "sppo"
    # Stage-3 diagnostic eval artifacts.
    for f in sd.glob(f"{EVAL_LABEL}*"):
        _cp_file(f, out / "STAGE3_EVALUATION")
    for f in sd.glob(f"{MATCHED128_LABEL}*"):
        _cp_file(f, out / "MATCHED128")
    for f in sd.glob(f"{MATCHED128_LABEL.lower()}*"):
        _cp_file(f, out / "MATCHED128")
    mdir = PROJ / MATCHED128_DIR
    if mdir.is_dir():
        for f in mdir.iterdir():
            if f.is_file():
                _cp_file(f, out / "MATCHED128")
    _cp_file(PROJ / MATCHED128_SPEC, out / "MATCHED128")
    # Stage-4 sharing students + dataset + their evals.
    for arm_dir in (sd / "suite_sharing_std" / "6v6_stage4").glob("*"):
        if arm_dir.is_dir():
            shutil.copytree(arm_dir, out / "STAGE4_SHARING" / arm_dir.name, dirs_exist_ok=True)
    for name in (
        "SUITE_DISTILLATION_6V6_STAGE4_DATASET.json",
        "SUITE_DISTILLATION_6V6_STAGE4_SPEC.json",
        "STANDARDIZED_6V6_STAGE4_SHARING_SPEC.json",
        "STANDARDIZED_6V6_STAGE4_SHARING_EVAL_SPEC.json",
        "SUITE_DATASETS_STAGE4_AUDIT_6V6.json",
    ):
        _cp_file(sd / name, out / "STAGE4_SHARING")
    for f in sd.glob("TOP50_6V6_STAGE4*"):
        _cp_file(f, out / "STAGE4_SHARING")

    # Seals / pins.
    _cp_file(PROJ / MANIFEST, out / "SEALS")
    _cp_file(SEAL, out / "SEALS")
    _cp_file(STAGE4_TEACHERS, out / "SEALS")
    _cp_file(PROJ / AUTH, out / "SEALS")
    _cp_file(PROJ / SPEC, out / "SEALS")
    _cp_file(PROJ / MATCHED128_SPEC, out / "SEALS")

    # Provenance.
    _cp_file(STATE, out / "PROVENANCE")
    for p in (OVERALL_ERR, OVERALL_JSON, LOG):
        _cp_file(p, out / "PROVENANCE")
    for tag_log in (PROJ / "6v6").glob("dual_branch_*.log*"):
        _cp_file(tag_log, out / "PROVENANCE" / "logs")

    _write_summary(out)

    (out / "START_HERE.txt").write_text(
        "ZIP THIS FOLDER AND SEND IT\n"
        "===========================\n\n"
        "In File Explorer:\n"
        "  1. Go up one level to AICTFProject\\6v6\\\n"
        "  2. Right-click the FOR_PROFESSOR folder\n"
        "  3. Choose Compress to ZIP file (or Send to > Compressed folder)\n"
        "  4. Email / Drive / USB that ZIP\n\n"
        "Open SUMMARY\\SUMMARY.txt next for the short readout.\n"
        "Everything you need is inside this folder. Ignore the rest of the repo.\n",
        encoding="utf-8",
    )
    (out / "README.txt").write_text(
        "6v6 Ours-Teachers + Stage 4 (Ours-Shared) — professor package\n"
        "=============================================================\n\n"
        "This folder is complete. Zip FOR_PROFESSOR in File Explorer and send it.\n"
        "No PowerShell. No rebuild script. Do not dig in artifacts/.\n\n"
        "TEACHERS/              Ours-Teachers: dual-branch DEFEND + ATTACK (A and B)\n"
        "STAGE3_EVALUATION/     old top-50 four-cell diagnostic (RESULT / rows / audit)\n"
        "MATCHED128/            matched-128 fair comparison + dual-branch own top-50\n"
        "STAGE4_SHARING/        Strategic Representation Under Parameter Sharing:\n"
        "                       Share-Encoder (comparison), Ours-Shared Fully Shared+z+r,\n"
        "                       Role-only ablation + dataset/evals\n"
        "SEALS/                 technical seal, teacher seal, deploy manifest, auth specs\n"
        "PROVENANCE/            STATE, overall progress, per-stage logs\n"
        "SUMMARY/SUMMARY.txt    short human readout\n\n"
        f"k = ceil(6/3) = {K}. Generalist pi(a|o) is not part of Stage 4.\n"
        "Ours-Shared is part of the proposed framework, not a neutral baseline.\n"
        "Ugly Delta on Stage-3 diagnostics does not invalidate this package.\n"
        "Own top-50 is best-case capability, not an unbiased estimate.\n",
        encoding="utf-8",
    )
    (out / "READY_TO_ZIP.txt").write_text(
        f"READY {now()}\nZip this FOR_PROFESSOR folder in File Explorer and send it.\n",
        encoding="utf-8",
    )

    # Optional convenience zip next to the folder (professor still can Explorer-zip).
    if BUNDLE.exists():
        BUNDLE.unlink()
    try:
        shutil.make_archive(str(BUNDLE.with_suffix("")), "zip", out)
        log(f"optional convenience zip -> {BUNDLE.name}")
    except Exception as exc:  # noqa: BLE001
        log(f"optional zip skipped ({exc}); FOR_PROFESSOR/ is still complete")

    log(f"FOR_PROFESSOR ready -> {out.relative_to(PROJ)}")
    log("Professor: right-click 6v6/FOR_PROFESSOR -> Compress to ZIP file -> send")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--skip-stage4", action="store_true",
                    help="stop after Phase 2 (diagnostic); default is full frozen pipeline")
    a = ap.parse_args()
    problems, notes = check()
    if a.check:
        for n in notes:
            print("  " + n)
        print("ALL CHECKS PASS" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
        return 0 if not problems else 1
    if problems:
        fail("pre-run checks failed: " + "; ".join(problems))

    # Remaining weighted work (already-finished STATE steps are omitted so ETA is honest on resume).
    plan: list[tuple[str, int]] = []
    if not done("smoke_A"):
        plan.append(("smoke_A", W_SMOKE))
    if not done("smoke_B"):
        plan.append(("smoke_B", W_SMOKE))
    if not done("train"):
        plan.append(("train_A", W_TRAIN))
        plan.append(("train_B", W_TRAIN))
    if not done("export"):
        plan.append(("export", W_BOOKKEEP))
    if not done("technical_seal"):
        plan.append(("technical_seal", W_BOOKKEEP))
    if not done("eval"):
        plan.append(("eval", W_EVAL_CELLS))
    if not done("matched128"):
        plan.append(("matched128", W_MATCHED128))
    if not done("own_top50"):
        plan.append(("own_top50", W_OWN_TOP50))
    if not a.skip_stage4:
        if not done("stage4_dataset"):
            plan.append(("stage4_dataset", W_COLLECT))
        if not done("stage4_students"):
            for arm in STAGE4_ARMS:
                plan.append((f"stage4_{arm}", W_DISTILL))
        if not done("stage4_evals"):
            plan.append(("stage4_evals", W_STAGE4_EVAL))
    if not done("bundle"):
        plan.append(("bundle", W_BOOKKEEP))
    overall_total = max(1, sum(w for _, w in plan))

    OVERALL_ERR.parent.mkdir(parents=True, exist_ok=True)
    OVERALL_ERR.write_text("", encoding="utf-8")
    state(status="RUNNING", pid=os.getpid())
    log("6v6 DUAL_BRANCH + STAGE4 frozen pipeline started")
    log(f"overall tqdm -> {OVERALL_ERR.relative_to(PROJ)}  (also watch dual_branch_<stage>.log.err)")
    log(f"remaining weighted units={overall_total}: " + ", ".join(f"{n}={w}" for n, w in plan))

    overall_bar = tqdm_iter(
        range(overall_total),
        desc="dual_branch_6v6_OVERALL",
        total=overall_total,
        unit="unit",
        leave=True,
    )
    overall_bar.n = 0
    overall_bar.refresh()
    PROGRESS["bar"] = overall_bar
    PROGRESS["base"] = 0
    PROGRESS["total"] = overall_total

    def _run_ppo(tag: str, argv: list[str], metrics_rel: str, weight: int) -> int:
        base = int(PROGRESS["base"])
        rc = run_logged(
            argv,
            tag,
            metrics_rel=metrics_rel,
            weight=weight,
            overall_base=base,
            overall_total=overall_total,
            overall_bar=overall_bar,
        )
        PROGRESS["base"] = base + weight
        return rc

    # ---- Phase 1: dual-branch teachers ----
    if not done("smoke_A"):
        smoke_dir = f"artifacts/scale_6v6_specialists/pi_A_specialist_6v6{RUNS['A']['suffix']}"
        rc = _run_ppo(
            "smoke_A",
            train_args("A", 5000, RUNS["A"]["smoke_seed"], None, RUNS["A"]["suffix"], smoke=True),
            f"{smoke_dir}/metrics.csv",
            W_SMOKE,
        )
        if rc != 0:
            fail(f"A smoke failed exit {rc}")
        state(step="smoke_A")
        log("A smoke PASS")
    if not done("smoke_B"):
        smoke_dir = f"artifacts/scale_6v6_specialists/pi_B_specialist_6v6{RUNS['B']['suffix']}"
        rc = _run_ppo(
            "smoke_B",
            train_args("B", 5000, RUNS["B"]["smoke_seed"], None, RUNS["B"]["suffix"], smoke=True),
            f"{smoke_dir}/metrics.csv",
            W_SMOKE,
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
        _advance_bookkeeping("export")
    if not done("technical_seal"):
        technical_seal()
        state(step="technical_seal", status="TECHNICALLY_SEALED")
        _advance_bookkeeping("technical_seal")

    # ---- Phase 2: Stage 3 diagnostics (not a redesign gate) ----
    if not done("eval"):
        rc = run_logged(eval_args() + ["--dry-run"], "eval_dryrun")
        if rc != 0:
            fail(f"evaluation dry-run failed exit {rc}")
        log("evaluation dry-run passed")
        base = int(PROGRESS["base"])
        rc = run_logged(
            eval_args() + ["--resume"],
            "eval",
            weight=W_EVAL_CELLS,
            overall_base=base,
            overall_total=overall_total,
            overall_bar=overall_bar,
        )
        PROGRESS["base"] = base + W_EVAL_CELLS
        result = PROJ / "artifacts/strategic_demand/sppo" / f"{EVAL_LABEL}_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
        if not result.is_file():
            fail(f"evaluation exited {rc} without {result.name}")
        state(step="eval")
        log("EVAL DONE (post-hoc old top-50 diagnostic; ugly Delta does NOT stop Stage 4)")

    # ---- Phase 2b/2c: matched-128 + dual-branch own top-50 ----
    if not done("matched128"):
        matched128_eval()
        state(step="matched128")
        log("MATCHED128 DONE (fair overall comparison on historical 128)")
    if not done("own_top50"):
        own_top50()
        state(step="own_top50")
        log("OWN_TOP50 DONE (best-case capability; historical rule; no new episodes)")

    if a.skip_stage4:
        if not done("bundle"):
            bundle()
            state(step="bundle", status="DONE_CORE_ONLY")
            _advance_bookkeeping("bundle")
        log("DONE core ( --skip-stage4 ). Stage 4 not run.")
        log("Professor: right-click 6v6/FOR_PROFESSOR -> Compress to ZIP file -> send")
        return 0

    # ---- Phase 3: Stage 4 dataset ----
    if not done("stage4_dataset"):
        if not SEAL.is_file() or json.loads(SEAL.read_text(encoding="utf-8")).get("status") != "SEALED":
            fail("Stage 4 requires TECHNICAL SEAL = SEALED")
        stage4_dataset()
        state(step="stage4_dataset")
        _advance_bookkeeping("stage4_dataset", W_COLLECT)

    # ---- Phase 4: sharing ladder ----
    if not done("stage4_students"):
        stage4_students()
        state(step="stage4_students")

    # ---- Phase 5: Stage 4 evals ----
    if not done("stage4_evals"):
        stage4_evals()
        state(step="stage4_evals")
        _advance_bookkeeping("stage4_evals", W_STAGE4_EVAL)

    # ---- Phase 6: bundle ----
    if not done("bundle"):
        bundle()
        state(step="bundle", status="DONE")
        _advance_bookkeeping("bundle")
    log("DONE -- full frozen pipeline.")
    log("Professor: right-click 6v6/FOR_PROFESSOR -> Compress to ZIP file -> send")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
