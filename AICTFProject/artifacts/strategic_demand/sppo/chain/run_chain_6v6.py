r"""Chain stage 2: after 4v4 closes, run the whole 6v6 suite unattended (PI 2026-09-29).

    .venv\Scripts\python.exe <this file>          (started early; waits for CHAIN_4V4_NOISE_DONE.json)

Every step fails closed, records itself in CHAIN_6V6_STATE.json and is skipped on a rerun:
   0  wait for stage 1 (4v4 noise) DONE
   1  merge branch prep-6v6 into the working branch (abort on any conflict)
   2  regression: the original 2v2/4v4 dataset audit must still be GREEN
   3  registry: mark the school-PC 6v6 exploratory block SPENT (its result sealed there) and retire the
      unspendable confirmatory block 22800001..128 (never spent; superseded by a standardized block)
   4  freeze SUITE_DISTILLATION_6V6_SPEC (reserve 2 x 96 collection seeds), smoke, collect, commit
   5  6v6 cross-scale audit vs 2v2 (executed-code identity) must be GREEN, commit
   6  freeze the 6v6 sharing spec (derived from the frozen 4v4 one), preflight x3, train x3, commit
   7  freeze the 6v6 sharing eval spec + the 6v6 Separated spec (confirmatory main + no-role, paired),
      dry-run all six evaluations, launch them detached, wait, commit
   8  freeze the 6v6 noise spec (mirror of the sealed 2v2 one), dry-run, launch four, wait, commit
   9  CHAIN_6V6_DONE.json
Never edits scientific code after the merge in step 1.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"K:\MultiAgentUAV")                       # always the main checkout, wherever this file lives
ROOT = REPO / "AICTFProject"
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
CHAIN = SD / "chain"
STATE = CHAIN / "CHAIN_6V6_STATE.json"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
STAGE1_DONE = CHAIN / "CHAIN_4V4_NOISE_DONE.json"
PREP_BRANCH = "prep-6v6"
POLL_S = 300
CHECK = "--check" in sys.argv

C2A = "artifacts/scale_6v6_specialists/pi_A_specialist_6v6_c2_entity_repair/ckpts/final_pi_A_specialist_6v6_c2_entity_repair.zip"
C2B = "artifacts/scale_6v6_specialists/pi_B_specialist_6v6_c2_entity_repair/ckpts/final_pi_B_specialist_6v6_c2_entity_repair.zip"
PID = "artifacts/scale_6v6_specialists/pi_A_specialist_6v6_split_defend_k1_v1/ckpts/final_pi_A_specialist_6v6_split_defend_k1_v1.zip"
SEAL = {C2A: "3298000480acd899e715eb78312938a9d20610a1eb0bcf73d382fd268577009e",
        C2B: "fc0043235d3abc23f87f5157a46920921f5eced0e9aa252347084d2b500258b5",
        PID: "77261f919009ff96ca81cb5a7c8e8b2346b77c16566775cf72570d9e73bc5ecc"}
SHARING_LABELS = {"share_encoder": "STANDARDIZED_6V6_SHARE_ENCODER", "fully_shared_z": "STANDARDIZED_6V6_FULLY_SHARED_Z",
                  "generalist": "STANDARDIZED_6V6_GENERALIST"}
REF_LABEL = "STANDARDIZED_6V6_SEPARATED_GENERALIST_REF"
MAIN_LABEL, NOROLE_LABEL = "STANDARDIZED_6V6_SPLIT_K1_CONFIRMATORY", "STANDARDIZED_6V6_NOROLE"


# ------------------------------------------------------------------ plumbing
def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"{now()} {msg}"
    print(line, flush=True)
    CHAIN.mkdir(parents=True, exist_ok=True)
    with (CHAIN / "chain_6v6.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def state(**kw) -> None:
    s = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {"record": "CHAIN_6V6", "steps": {}}
    s.update({k: v for k, v in kw.items() if k != "step"})
    if "step" in kw:
        s["steps"][kw["step"]] = now()
    s["updated_utc"] = now()
    STATE.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")


def done(step: str) -> bool:
    return STATE.is_file() and step in json.loads(STATE.read_text(encoding="utf-8"))["steps"]


def fail(msg: str) -> None:
    log(f"ABORT: {msg}")
    state(status="ABORTED", reason=msg)
    sys.exit(1)


def git(*a: str, ok_fail: bool = False) -> str:
    r = subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)
    if r.returncode != 0 and not ok_fail:
        fail(f"git {' '.join(a)}: {(r.stderr or r.stdout).strip()[:400]}")
    return r.stdout.strip()


def commit(paths: list[str], title: str, body: str) -> None:
    git("add", "--", *paths)
    if not git("diff", "--cached", "--name-only"):
        log(f"nothing to commit: {title}")
        return
    git("commit", "-q", "-m", title, "-m", body, "-m", "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    log(f"committed {git('log', '--oneline', '-1')}")


def run(args: list[str], what: str, ok_codes=(0,)) -> int:
    log(f"run: {what}")
    r = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True)
    (CHAIN / "logs").mkdir(exist_ok=True)
    (CHAIN / "logs" / f"{what.replace(' ', '_')}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    if r.returncode not in ok_codes:
        fail(f"{what} exit {r.returncode}: {(r.stdout + r.stderr)[-500:]}")
    return r.returncode


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(REPO)).replace("\\", "/")


def alive(pid: int) -> bool:
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
    return str(pid) in r.stdout


def result(label: str, separated: bool) -> Path:
    return SD / (f"{label}_SPECIALIST_CROSSOVER_EVAL_RESULT.json" if separated else f"{label}_CROSSOVER_EVAL_RESULT.json")


def launch_and_wait(cmds: dict[str, tuple[str, bool]], logdir: Path, step: str) -> None:
    """cmds: label -> (args without interpreter, separated evaluator?). Dry-runs all, launches detached, waits."""
    if not done(f"{step}_dry_runs"):
        for lab, (a, _sep) in cmds.items():
            r = subprocess.run([PY, *a.replace(" --resume", "").split(), "--dry-run"], cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                fail(f"dry-run {lab} exit {r.returncode}: {(r.stdout + r.stderr)[-400:]}")
            log(f"dry-run PASS {lab}")
        state(step=f"{step}_dry_runs")
    s = json.loads(STATE.read_text(encoding="utf-8"))
    pids = s.get(f"{step}_pids") or {}
    logdir.mkdir(parents=True, exist_ok=True)
    for lab, (a, sep) in cmds.items():
        if result(lab, sep).is_file() or (lab in pids and alive(pids[lab])):
            continue
        out = open(logdir / f"{lab}.log", "a", encoding="utf-8")
        err = open(logdir / f"{lab}.log.err", "a", encoding="utf-8")
        p = subprocess.Popen([PY, *a.split()], cwd=ROOT, stdout=out, stderr=err,
                             creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008)
        pids[lab] = p.pid
        log(f"launched {lab} pid={p.pid}")
    state(**{f"{step}_pids": pids, f"{step}_launch_commit": git("rev-parse", "--short", "HEAD")})
    while True:
        missing = [l for l, (_a, sep) in cmds.items() if not result(l, sep).is_file()]
        if not missing:
            break
        dead = [l for l in missing if l in pids and not alive(pids[l])]
        if dead:
            fail(f"{step}: exited without a sealed record: {dead} (rerun the chain; runs resume from PARTIAL)")
        time.sleep(POLL_S)
    for l, (_a, sep) in cmds.items():
        if json.loads(result(l, sep).read_text(encoding="utf-8")).get("status") != "SEALED":
            fail(f"{l} not SEALED")
    log(f"{step}: all sealed")


def split_rows(manifest: dict) -> dict:
    """Row counts under the distillation loader's own rule (decision rows; holdout = episode % 10 == 9)."""
    import numpy as np
    a = b = h = 0
    for s in manifest["shards"]:
        d = np.load(ROOT / s["file"])
        if int(d["step"].shape[0]) == 0:
            continue
        m = d["decision_mask"].any(axis=1)
        ep, pole = d["episode"][m], d["pole"][m]
        hold = (ep % 10) == 9
        h += int(hold.sum())
        a += int(((~hold) & (pole == 0)).sum())
        b += int(((~hold) & (pole == 1)).sum())
    return {"train_pole_A": a, "train_pole_B": b, "holdout": h}


# ------------------------------------------------------------------ the chain
def main() -> int:
    sys.path.insert(0, str(ROOT))
    CHAIN.mkdir(parents=True, exist_ok=True)
    if CHECK:
        return check()
    state(status="WAITING_FOR_STAGE_1", pid=os.getpid())
    log("stage 2 (6v6) started; waiting for CHAIN_4V4_NOISE_DONE.json")
    while not STAGE1_DONE.is_file():
        s1 = CHAIN / "CHAIN_4V4_NOISE_STATE.json"
        if s1.is_file() and json.loads(s1.read_text(encoding="utf-8")).get("status") == "ABORTED":
            fail("stage 1 (4v4 noise) ABORTED; 6v6 does not start")
        time.sleep(POLL_S)
    state(status="RUNNING")

    # 1: merge the prepared 6v6 code/specs
    if not done("merged"):
        if git("status", "--porcelain", "--", "AICTFProject/experiments", "AICTFProject/rl", "AICTFProject/gpu_env",
               "AICTFProject/configs", "AICTFProject/tests"):
            fail("scientific tree not clean before the prep-6v6 merge")
        r = subprocess.run(["git", "-C", str(REPO), "merge", "--no-ff", PREP_BRANCH, "-m",
                            "Merge prep-6v6: executed-code identity audit, 6v6 collection spec, 6v6 chain"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            subprocess.run(["git", "-C", str(REPO), "merge", "--abort"], capture_output=True)
            fail(f"merge of {PREP_BRANCH} failed (aborted, tree unchanged): {(r.stdout + r.stderr)[-400:]}")
        log(f"merged {PREP_BRANCH}: {git('log', '--oneline', '-1')}")
        state(step="merged")

    # 2: regression on the original audit
    if not done("regression_audit"):
        run(["experiments/audit_suite_datasets_cross_scale.py", "--scales", "2v2,4v4"], "regression audit 2v2 4v4")
        state(step="regression_audit")

    from experiments import seed_registry as SR

    # 3: registry reconciliation for the school-PC 6v6 blocks
    if not done("registry_reconciled"):
        reg = {b["experiment_id"]: b for b in SR.load()["blocks"]}
        if reg.get("SCALE_6V6_SPLIT_CROSSOVER_EXPLORATORY_V1", {}).get("status") == "RESERVED":
            if not (SD.parent / "EXPLORATORY_6V6_SPLIT_K1_SPECIALIST_CROSSOVER_EVAL_RESULT.json").is_file() and \
               not (SD / "EXPLORATORY_6V6_SPLIT_K1_SPECIALIST_CROSSOVER_EVAL_RESULT.json").is_file():
                fail("6v6 exploratory block is RESERVED here but its sealed result is not on disk")
            SR.set_status("SCALE_6V6_SPLIT_CROSSOVER_EXPLORATORY_V1", "SPENT",
                          "reconciled 2026-09-29: spent on the school PC; EXPLORATORY_6V6_SPLIT_K1 sealed there (registries were not synced)")
        if reg.get("SCALE_6V6_SPLIT_CROSSOVER_CONFIRMATORY_V1", {}).get("status") == "RESERVED":
            SR.set_status("SCALE_6V6_SPLIT_CROSSOVER_CONFIRMATORY_V1", "RETIRED",
                          "never spent; registered without shared labels under an id no evaluator label maps to; superseded by "
                          "the standardized 6v6 confirmatory block (PI 2026-09-29: run it as a paper-parity measurement)")
        commit([rel(ROOT / "artifacts" / "SEED_REGISTRY.json")], "Reconcile the school-PC 6v6 seed blocks",
               "Exploratory block SPENT (sealed on the school PC); unspendable confirmatory block RETIRED, never spent.")
        state(step="registry_reconciled")

    # 4: freeze the collection spec, smoke, collect
    cs = SD / "SUITE_DISTILLATION_6V6_SPEC.json"
    if not done("collection_spec_frozen"):
        s = json.loads(cs.read_text(encoding="utf-8"))
        for p, want in SEAL.items():
            if sha(p) != want:
                fail(f"{p} sha differs from its school-PC seal")
        blocks = {}
        for blk, pole in (("collection_A", "A"), ("collection_B", "B")):
            lo = SR.next_free(96)
            eid = s["SEEDS"]["registry_experiment_ids"][blk]
            SR.allocate(eid, lo, lo + 95, "exploratory", f"6v6 suite distillation dataset, Pole {pole}", spec=cs.name)
            s["SEEDS"][blk] = f"{lo}..{lo + 95}"
            blocks[blk] = s["SEEDS"][blk]
        s.update(status="FROZEN_BEFORE_COLLECTION", utc=now())
        cs.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
        commit([rel(cs), rel(ROOT / "artifacts" / "SEED_REGISTRY.json")], "Freeze the 6v6 collection spec; reserve its seeds",
               f"SUITE_DISTILLATION_6V6_SPEC FROZEN_BEFORE_COLLECTION; {blocks}.")
        state(step="collection_spec_frozen", collection_blocks=blocks)
    if not done("collected"):
        run(["experiments/collect_suite_distillation_states.py", "--team-size", "6", "--device", "cuda", "--smoke"], "collect 6v6 smoke")
        run(["experiments/collect_suite_distillation_states.py", "--team-size", "6", "--device", "cuda", "--resume"], "collect 6v6")
        man = json.loads((SD / "SUITE_DISTILLATION_6V6_DATASET.json").read_text(encoding="utf-8"))
        if man.get("status") != "FROZEN_DATASET":
            fail(f"6v6 manifest status {man.get('status')!r}")
        commit([rel(SD / "SUITE_DISTILLATION_6V6_DATASET.json"), rel(SD / "suite_distillation_6v6"),
                rel(SD / "SUITE_DISTILLATION_6V6_DATASET_SMOKE.json"), rel(ROOT / "artifacts" / "SEED_REGISTRY.json")],
               "Record the frozen 6v6 suite distillation dataset", "Collected by the 6v6 chain under SUITE_DISTILLATION_6V6_SPEC.")
        state(step="collected")

    # 5: 6v6 audit
    audit6 = SD / "SUITE_DATASETS_CROSS_SCALE_AUDIT_6V6.json"
    if not done("audited"):
        run(["experiments/audit_suite_datasets_cross_scale.py", "--scales", "2v2,6v6", "--write", "--out", str(audit6)],
            "audit 2v2 6v6")
        if json.loads(audit6.read_text(encoding="utf-8")).get("verdict") != "GREEN":
            fail("6v6 dataset audit is not GREEN")
        commit([rel(audit6)], "6v6 dataset audit GREEN (executed-code identity with the 2v2 collection)", "Written by the 6v6 chain.")
        state(step="audited")

    # 6: sharing spec, preflights, training
    ss = SD / "STANDARDIZED_6V6_SHARING_SPEC.json"
    if not done("sharing_spec_frozen"):
        derive_sharing_spec(ss, audit6, SR)
        state(step="sharing_spec_frozen")
    if not done("students_frozen"):
        for arm in ("share_encoder", "fully_shared", "generalist"):
            run(["experiments/run_suite_sharing_distillation.py", "--arm", arm, "--team-size", "6", "--preflight", "--device", "cuda"],
                f"preflight 6v6 {arm}")
        for arm in ("share_encoder", "fully_shared", "generalist"):
            run(["experiments/run_suite_sharing_distillation.py", "--arm", arm, "--team-size", "6", "--device", "cuda"],
                f"train 6v6 {arm}", ok_codes=(0, 2))
        commit([rel(SD / "suite_sharing_std" / "6v6"), rel(ROOT / "artifacts" / "SEED_REGISTRY.json")],
               "Freeze the three 6v6 sharing students", "Trained in the frozen order by the 6v6 chain.")
        state(step="students_frozen")

    # 7: evaluation (sharing arms + Separated reference, and the Separated main + no-role pair)
    es, sep = SD / "STANDARDIZED_6V6_SHARING_EVAL_SPEC.json", SD / "STANDARDIZED_6V6_SEPARATED_SPEC.json"
    if not done("eval_specs_frozen"):
        derive_eval_specs(es, sep, ss, SR)
        state(step="eval_specs_frozen")
    e, p = json.loads(es.read_text(encoding="utf-8")), json.loads(sep.read_text(encoding="utf-8"))
    cmds = {SHARING_LABELS[k]: (e["LAUNCH"][k].split(" ", 1)[1], False) for k in SHARING_LABELS}
    cmds[REF_LABEL] = (e["LAUNCH"]["separated_reference"].split(" ", 1)[1], True)
    cmds[MAIN_LABEL] = (p["LAUNCH"][MAIN_LABEL].split(" ", 1)[1], True)
    cmds[NOROLE_LABEL] = (p["LAUNCH"][NOROLE_LABEL].split(" ", 1)[1], True)
    if not done("evals_committed"):
        launch_and_wait(cmds, SD / "suite_sharing_std" / "6v6" / "evals", "evals")
        commit([f"{rel(SD)}/STANDARDIZED_6V6_*", f"{rel(SD)}/standardized_6v6_*_eval_rows.csv",
                rel(ROOT / "artifacts" / "SEED_REGISTRY.json")],
               "Seal the 6v6 baselines, the confirmatory main row and the no-role row", "Six labels sealed by the 6v6 chain.")
        state(step="evals_committed")

    # 8: noise
    ns = SD / "STANDARDIZED_6V6_NOISE_SPEC.json"
    if not done("noise_spec_frozen"):
        derive_noise_spec(ns, SR)
        state(step="noise_spec_frozen")
    n = json.loads(ns.read_text(encoding="utf-8"))
    common = n["LAUNCH"]["common"].split(" ", 1)[1]
    ncmds = {lab: (f"{common} {extra}", True) for lab, extra in n["LAUNCH"]["per_label"].items()}
    if not done("noise_committed"):
        launch_and_wait(ncmds, SD / "suite_sharing_std" / "6v6" / "noise", "noise")
        commit([f"{rel(SD)}/STANDARDIZED_6V6_NOISE_*", f"{rel(SD)}/standardized_6v6_noise_*_eval_rows.csv",
                rel(ROOT / "artifacts" / "SEED_REGISTRY.json")], "Seal the 6v6 deployment-noise suite", "Four conditions sealed by the 6v6 chain.")
        state(step="noise_committed")

    (CHAIN / "CHAIN_6V6_DONE.json").write_text(json.dumps({"record": "CHAIN_6V6_DONE", "utc": now()}, indent=2) + "\n", encoding="utf-8")
    state(status="DONE")
    log("6v6 chain DONE: every 6v6 row is sealed; fill experiments.tex and close 6v6")
    return 0


# ------------------------------------------------------------------ spec derivations
def derive_sharing_spec(out: Path, audit6: Path, SR) -> None:
    four = json.loads((SD / "STANDARDIZED_4V4_SHARING_SPEC.json").read_text(encoding="utf-8"))
    two = json.loads((SD / "STANDARDIZED_2V2_SHARING_SPEC.json").read_text(encoding="utf-8"))
    mp = SD / "SUITE_DISTILLATION_6V6_DATASET.json"
    man = json.loads(mp.read_text(encoding="utf-8"))
    import importlib
    RSD = importlib.import_module("experiments.run_suite_sharing_distillation")
    s = json.loads(json.dumps(four))
    s.update(record_id="STANDARDIZED_6V6_SHARING_SPEC", status="FROZEN_BEFORE_TRAINING", utc=now(),
             derived_from="STANDARDIZED_4V4_SHARING_SPEC.json (itself derived from the 2v2 spec)",
             decided_by="PI, 2026-09-29: after 4v4 closes, 6v6 with the same frozen definitions, started automatically.",
             frozen_after="CHAIN_4V4_NOISE_DONE.json")
    for k in ("scope", "THE_QUESTION"):
        s[k] = s[k].replace("4v4", "6v6")
    s["SCALE"] = {"N": 6, "k_defend": 1, "allocator": four["SCALE"]["allocator"]}
    s["DATASET_locked"].update(manifest=rel(mp).replace("AICTFProject/", ""), manifest_sha256=RSD._sha(mp),
                               content_sha256=RSD.dataset_content_sha256(man),
                               collected_at=man["collector"]["git_sha"][:8] + " (executed-code identity: COLLECTOR_EXECUTED_CODE_ATTESTATION.json)",
                               decision_rows=None, audit_record=rel(audit6).replace("AICTFProject/", ""))
    rows = split_rows(man)
    s["DATASET_locked"]["decision_rows"] = sum(rows.values())
    s["TEACHERS_locked"]["pi_A"] = dict(man["teachers"]["pi_A"])
    s["TEACHERS_locked"]["pi_B"] = dict(man["teachers"]["pi_B"])
    s["POLES_locked"]["A"] = {"pole_config_hash": man["poles"]["A"]["pole_config_hash"]}
    s["POLES_locked"]["B"] = {"pole_config_hash": man["poles"]["B"]["pole_config_hash"]}
    s["SPLIT_AND_SAMPLING_locked"]["rows"] = rows
    for key in ("ARM_ORDER", "RECIPE_locked", "PREFLIGHT_REQUIRED"):
        if s[key] != two[key]:
            fail(f"6v6 sharing spec {key} differs from 2v2")
    tr = {}
    for arm, n in (("share_encoder", 2), ("fully_shared", 1), ("generalist", 1)):
        eid = s["ARMS_locked"][arm]["seed_block"].replace("4V4", "6V6")
        lo = SR.next_free(n)
        SR.allocate(eid, lo, lo + n - 1, "exploratory", f"6v6 {arm} distillation init", spec=out.name)
        tr[eid] = f"{lo}..{lo + n - 1}"
        s["ARMS_locked"][arm]["seed_block"] = eid
        s["ARMS_locked"][arm]["init"] = f"fresh; seeds {tr[eid]} ({eid})"
    ev = s["SEEDS_locked"]["evaluation"]
    labels = [l.replace("4V4", "6V6") for l in ev["shared_by_labels"]]
    lo = SR.next_free(128)
    SR.allocate("STANDARDIZED_6V6_SHARING_EVAL", lo, lo + 127, "sealed_confirmatory",
                "6v6 matched evaluation of Share-Encoder, Fully Shared+z, Generalist and the Separated Delta_G reference",
                spec=out.name, shared_by_labels=labels)
    s["SEEDS_locked"] = {"training": tr, "evaluation": {
        "STANDARDIZED_6V6_SHARING_EVAL": f"{lo}..{lo + 127}", "seed_class": "sealed_confirmatory",
        "shared_by_labels": labels, "why_one_block": ev["why_one_block"],
        "separated_ref_note": "the Separated run on this block exists only to pair Delta_G; the 6v6 main value is STANDARDIZED_6V6_SPLIT_K1_CONFIRMATORY"},
        "all_must_be": four["SEEDS_locked"]["all_must_be"]}
    s["EVALUATION_planned"]["frozen_later_in"] = "STANDARDIZED_6V6_SHARING_EVAL_SPEC.json"
    s["OUTPUTS"] = {"root": "artifacts/strategic_demand/sppo/suite_sharing_std/6v6/<arm tag>/",
                    "why_a_new_root": "one root per scale under suite_sharing_std"}
    s["FORBIDDEN_INPUTS"] = ["TEACHER_DISTILLATION_6V6_DATASET.json (legacy, unrepaired specialists)",
                             "the historical 6v6 specialists artifacts/scale_6v6_specialists/pi_{A,B}_specialist_6v6 (not suite teachers)",
                             "RUNG1_6V6 / sharing_ladder_6v6 students"]
    s["NOT_AUTHORIZED_BY_THIS_SPEC"] = ["changing the dataset, teachers, recipe, split or arm order",
                                        "the evaluation run (needs STANDARDIZED_6V6_SHARING_EVAL_SPEC.json)",
                                        "the deployment-noise cases", "claiming PAPER-FAITHFUL"]
    out.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
    commit([rel(out), rel(ROOT / "artifacts" / "SEED_REGISTRY.json")], "Freeze the 6v6 sharing spec; reserve its blocks",
           f"Derived from the frozen 4v4 spec (arm order, recipe, preflight identical to 2v2); training {tr}; eval {lo}..{lo + 127}.")


def derive_eval_specs(es: Path, sep: Path, ss: Path, SR) -> None:
    four = json.loads((SD / "STANDARDIZED_4V4_SHARING_EVAL_SPEC.json").read_text(encoding="utf-8"))
    tr = json.loads(ss.read_text(encoding="utf-8"))
    s = json.loads(json.dumps(four))
    s.update(record_id="STANDARDIZED_6V6_SHARING_EVAL_SPEC", status="FROZEN_BEFORE_SEED_SPEND", utc=now(),
             derived_from="STANDARDIZED_4V4_SHARING_EVAL_SPEC.json", parent=["STANDARDIZED_6V6_SHARING_SPEC.json (training)",
                                                                             "GENERALIST_DEFINITION_V1.json#SCORING_locked"])
    s["claim_boundary"] = s["claim_boundary"].replace("4v4", "6v6")
    for key, tag in (("share_encoder", "share_encoder"), ("fully_shared_z", "fully_shared_z"), ("generalist", "generalist")):
        fr = json.loads((SD / "suite_sharing_std" / "6v6" / tag / "STUDENT_FROZEN.json").read_text(encoding="utf-8"))
        ck = fr["checkpoint"].replace("\\", "/")
        if sha(ck) != fr["sha256"]:
            fail(f"6v6 {key} checkpoint sha differs from STUDENT_FROZEN")
        s["ARMS"][key] = {**s["ARMS"][key], "label": SHARING_LABELS[key], "checkpoint": ck, "sha256": fr["sha256"]}
    ref = s["SEPARATED_REFERENCE"]
    ref.update(label=REF_LABEL,
               why=ref["why"].replace("4v4 main Separated value stays DEFEND_ATTACK_SPLIT_POLICY_A_V1_CONFIRMATORY_V1 (+0.156/+0.406)",
                                      "6v6 main Separated value is STANDARDIZED_6V6_SPLIT_K1_CONFIRMATORY"),
               system="6v6 split composite (SCHOOL_PC_6V6_LOCKED_PIPELINE): split k1 pi_D on DEFEND + c2 pi_A frozen on ATTACK, roles fixed for the episode, CLOSEST_DEFENDS k=1; pi_B the c2 pi_B",
               checkpoints={"pi_D": {"path": PID, "sha256": SEAL[PID]}, "pi_A_frozen_attack": {"path": C2A, "sha256": SEAL[C2A]},
                            "pi_B": {"path": C2B, "sha256": SEAL[C2B]}})
    ev = tr["SEEDS_locked"]["evaluation"]
    block = ev["STANDARDIZED_6V6_SHARING_EVAL"]
    lo = block.split("..")[0]
    s["SEEDS"] = {"registry_experiment_id": "STANDARDIZED_6V6_SHARING_EVAL", "block": block, "n": 128,
                  "seed_class": "sealed_confirmatory", "shared_by_labels": ev["shared_by_labels"], "spent_when": "all four labels have sealed"}
    s["POLES"] = {"A": tr["POLES_locked"]["A"], "B": tr["POLES_locked"]["B"], "source": four["POLES"]["source"]}
    L = s["LAUNCH"]
    for k in ("share_encoder", "fully_shared_z", "generalist"):
        L[k] = L[k].replace("--team-size 4", "--team-size 6")
    sep_args = (f"--team-size 6 --pi-a-path {PID} --pi-b-path {C2B} --device cuda --role-fixed-for-episode "
                f"--frozen-attack-path {C2A} --frozen-attack-path-sha256 {SEAL[C2A]} --role-k-defend 1")
    L["separated_reference"] = (f".venv/Scripts/python.exe experiments/eval_specialist_crossover_scaled.py {sep_args} "
                                f"--spec artifacts/strategic_demand/sppo/{es.name} --seed-base {lo} --n-seeds 128 "
                                f"--label {REF_LABEL} --registry-experiment-id STANDARDIZED_6V6_SHARING_EVAL --resume")
    s["NOT_AUTHORIZED_BY_THIS_SPEC"] = [x.replace("4v4 main Separated", "6v6 main Separated").replace("any 6v6 evaluation", "any 8v8 evaluation")
                                        for x in four["NOT_AUTHORIZED_BY_THIS_SPEC"]]
    es.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")

    # the Separated main row (confirmatory, n=128) and the no-role row, paired on one fresh block
    lo2 = SR.next_free(128)
    SR.allocate("STANDARDIZED_6V6_SEPARATED_EVAL", lo2, lo2 + 127, "sealed_confirmatory",
                "6v6 main Separated confirmatory (paper parity) + no-role specialists, paired",
                spec=sep.name, shared_by_labels=[MAIN_LABEL, NOROLE_LABEL])
    base = f".venv/Scripts/python.exe experiments/eval_specialist_crossover_scaled.py --team-size 6 --spec artifacts/strategic_demand/sppo/{sep.name} --seed-base {lo2} --n-seeds 128 --device cuda --registry-experiment-id STANDARDIZED_6V6_SEPARATED_EVAL --resume"
    p = {"record_id": "STANDARDIZED_6V6_SEPARATED_SPEC", "status": "FROZEN_BEFORE_SEED_SPEND", "arm": "SEALED_CONFIRMATORY",
         "confirmatory": True, "utc": now(),
         "decided_by": "PI, 2026-09-29: run the 6v6 confirmatory n=128 as a paper-parity measurement (2v2/4v4 report n=128 "
                       "confirmatory values), not as a gate-conditioned confirmation; the sealed exploratory n=64 "
                       "(EXPLORATORY_6V6_SPLIT_K1) stays on record unchanged.",
         "amends": "SCHOOL_PC_6V6_LOCKED_PIPELINE.json step 8 ('confirmatory IF exploratory PASS'); its block 22800001..128 is RETIRED unspent",
         "LABELS": {MAIN_LABEL: "the 6v6 main row: split composite, CLOSEST_DEFENDS k=1, roles fixed for the episode",
                    NOROLE_LABEL: "specialists without roles: c2 pi_A vs c2 pi_B, plain (the teachers of the distilled students)"},
         "why_paired": "both on the same 128 fresh seeds, so the role-allocation effect is read seed by seed (as the 2v2 diagnostic pair)",
         "CHECKPOINTS_locked": {"pi_D": {"path": PID, "sha256": SEAL[PID]}, "pi_A": {"path": C2A, "sha256": SEAL[C2A]},
                                "pi_B": {"path": C2B, "sha256": SEAL[C2B]}},
         "SEEDS": {"registry_experiment_id": "STANDARDIZED_6V6_SEPARATED_EVAL", "block": f"{lo2}..{lo2 + 127}", "n": 128,
                   "seed_class": "sealed_confirmatory", "shared_by_labels": [MAIN_LABEL, NOROLE_LABEL]},
         "LAUNCH": {MAIN_LABEL: f"{base} --pi-a-path {PID} --pi-b-path {C2B} --label {MAIN_LABEL} --role-fixed-for-episode "
                                f"--frozen-attack-path {C2A} --frozen-attack-path-sha256 {SEAL[C2A]} --role-k-defend 1",
                    NOROLE_LABEL: f"{base} --pi-a-path {C2A} --pi-b-path {C2B} --label {NOROLE_LABEL}"},
         "NOT_AUTHORIZED_BY_THIS_SPEC": ["re-running either label", "changing k, the splice or the checkpoints", "claiming PAPER-FAITHFUL"]}
    sep.write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
    commit([rel(es), rel(sep), rel(ROOT / "artifacts" / "SEED_REGISTRY.json")],
           "Freeze the 6v6 sharing eval spec and the 6v6 Separated spec (confirmatory main + no-role)",
           f"Sharing eval block {block}; Separated main + no-role paired on {lo2}..{lo2 + 127} (PI: paper-parity confirmatory).")


def derive_noise_spec(out: Path, SR) -> None:
    two = json.loads((SD / "STANDARDIZED_2V2_NOISE_SPEC.json").read_text(encoding="utf-8"))
    lo = SR.next_free(128)
    labels = [l.replace("2V2", "6V6") for l in two["SEEDS"]["shared_by_labels"]]
    s = json.loads(json.dumps(two))
    s.update(record_id="STANDARDIZED_6V6_NOISE_SPEC", status="FROZEN_BEFORE_SEED_SPEND", utc=now(),
             derived_from="STANDARDIZED_2V2_NOISE_SPEC.json (only the system, seeds, labels and team size change)",
             decided_by="PI, 2026-09-29: 6v6 runs unattended after 4v4, mirroring the sealed noise suites.",
             THE_QUESTION=two["THE_QUESTION"].replace("2v2", "6v6"))
    s["SYSTEM_locked"] = {"name": "final 6v6 Separated system (STANDARDIZED_6V6_SPLIT_K1_CONFIRMATORY)",
                          "pole_A_policy": "split composite: split k1 pi_D on DEFEND + c2 pi_A frozen on ATTACK, roles fixed for the episode, CLOSEST_DEFENDS k=1",
                          "pole_B_policy": "c2 pi_B",
                          "checkpoints": {"pi_D": {"path": PID, "sha256": SEAL[PID]}, "pi_A_frozen_attack": {"path": C2A, "sha256": SEAL[C2A]},
                                          "pi_B": {"path": C2B, "sha256": SEAL[C2B]}}}
    s["CONDITIONS_locked"] = {k.replace("2V2", "6V6"): v for k, v in two["CONDITIONS_locked"].items()}
    s["SEEDS"] = {**two["SEEDS"], "registry_experiment_id": "STANDARDIZED_6V6_NOISE_EVAL", "block": f"{lo}..{lo + 127}",
                  "shared_by_labels": labels}
    s["READINGS"] = {**two["READINGS"], "headline": two["READINGS"]["headline"].replace(
        "the 2v2 main value stays +0.117/+0.539", "the 6v6 main value is STANDARDIZED_6V6_SPLIT_K1_CONFIRMATORY")}
    s["LAUNCH"] = {**two["LAUNCH"], "common": (
        f".venv/Scripts/python.exe experiments/eval_specialist_crossover_scaled.py --team-size 6 "
        f"--spec artifacts/strategic_demand/sppo/{out.name} --pi-a-path {PID} --pi-b-path {C2B} --seed-base {lo} --n-seeds 128 "
        f"--device cuda --role-fixed-for-episode --frozen-attack-path {C2A} --frozen-attack-path-sha256 {SEAL[C2A]} "
        f"--role-k-defend 1 --registry-experiment-id STANDARDIZED_6V6_NOISE_EVAL --resume"),
        "per_label": {k.replace("2V2", "6V6"): v.replace("2V2", "6V6") for k, v in two["LAUNCH"]["per_label"].items()},
        "when": "automatically after the 6v6 evaluations sealed (chain stage 2)"}
    s["NOT_AUTHORIZED_BY_THIS_SPEC"] = [x.replace("any 4v4 or 6v6 evaluation", "any 8v8 evaluation") for x in two["NOT_AUTHORIZED_BY_THIS_SPEC"]]
    out.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
    SR.allocate("STANDARDIZED_6V6_NOISE_EVAL", lo, lo + 127, "sealed_confirmatory",
                "6v6 deployment-noise suite on the final system: nominal + medium localization/motion/delay, matched seeds",
                spec=out.name, shared_by_labels=labels)
    commit([rel(out), rel(ROOT / "artifacts" / "SEED_REGISTRY.json")], "Freeze the 6v6 noise spec; reserve its matched block",
           f"STANDARDIZED_6V6_NOISE_EVAL {lo}..{lo + 127} shared by {labels}.")


def check() -> int:
    """--check: no side effects. Confirms inputs exist and match their seals, and that the branch merges."""
    problems = []
    for p, want in SEAL.items():
        if not (ROOT / p).is_file() or sha(p) != want:
            problems.append(f"{p}: missing or sha differs from its seal")
    for f in ("STANDARDIZED_2V2_SHARING_SPEC.json", "STANDARDIZED_4V4_SHARING_SPEC.json", "STANDARDIZED_4V4_SHARING_EVAL_SPEC.json",
              "STANDARDIZED_2V2_NOISE_SPEC.json", "GENERALIST_DEFINITION_V1.json"):
        if not (SD / f).is_file():
            problems.append(f"{f} missing on main")
    br = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify", "--quiet", PREP_BRANCH], capture_output=True, text=True)
    if not br.stdout.strip():
        problems.append(f"branch {PREP_BRANCH} missing")
    mt = subprocess.run(["git", "-C", str(REPO), "merge-tree", "--write-tree", "HEAD", PREP_BRANCH], capture_output=True, text=True)
    if mt.returncode != 0:
        problems.append(f"{PREP_BRANCH} does not merge cleanly into HEAD: {mt.stdout[-300:]}")
    print("--check:", "OK" if not problems else problems)
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
