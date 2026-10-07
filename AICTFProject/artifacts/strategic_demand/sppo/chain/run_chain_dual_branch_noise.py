r"""TOP50 deployment-robustness diagnostic for the symmetric dual-branch method.

One frozen spec, three team sizes, three medium disturbances. The 50 seeds at
each scale are the already-sealed own-TOP50 IDs from that scale's dual-branch
matched-128 nominal. They are not recomputed and not re-ranked under noise.
Nominal episodes are not rerun. New work is 3 x 200 = 600 episodes per team
size. 2v2 finishes before 4v4 starts. 6v6 stays in the frozen spec and is not launched
unless this script is given --include-6v6.

    .venv\Scripts\python.exe artifacts/strategic_demand/sppo/chain/run_chain_dual_branch_noise.py --freeze
    .venv\Scripts\python.exe artifacts/strategic_demand/sppo/chain/run_chain_dual_branch_noise.py

This is a post-hoc descriptive diagnostic, not an unbiased confirmatory test.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # AICTFProject
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
CHAIN = SD / "chain"
SPEC_PATH = SD / "DUAL_BRANCH_TOP50_NOISE_SPEC.json"
STATE = CHAIN / "CHAIN_DUAL_BRANCH_TOP50_NOISE_STATE.json"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
sys.path.insert(0, str(ROOT))

POLL_S = 120
STALE_PARTIAL_S = 3 * 3600

# Own-TOP50 records already sealed from the dual-branch matched-128 nominal.
# This chain reads those seed IDs. It does not call select_own_top50.
SCALES = (
    {
        "n": 2,
        "k_defend": 1,
        "seed_base_block": (23600001, 23600128),
        "registry_experiment_id": "STANDARDIZED_2V2_SPLIT_K1_CONFIRMATORY_SPECIALIST_CROSSOVER",
        "nominal_result": "POSTHOC_MATCHED128_2V2_DUAL_BRANCH_SPECIALIST_CROSSOVER_EVAL_RESULT.json",
        "nominal_rows_sha256": "286cb10c3df4f6472a137149e708381faf9dfbd119d0c85f0f0848fa83f5c97d",
        "manifest": "2v2/dual_branch_deploy_manifest.json",
        "top50_record": "dual_branch_v1/matched128_2v2/DUAL_BRANCH_2V2_OWN_TOP50.json",
        "seed_list": "dual_branch_v1/matched128_2v2/DUAL_BRANCH_2V2_OWN_TOP50_seed_ids.json",
        "question": (
            "On this own-TOP50 subset the nominal win deltas are two-sided "
            "(Delta_A +0.32, Delta_B +0.66). Report whether both stay positive "
            "and the paired change against those same seeds."
        ),
    },
    {
        "n": 4,
        "k_defend": 2,
        "seed_base_block": (21800001, 21800128),
        "registry_experiment_id": "DEFEND_ATTACK_SPLIT_POLICY_A_V1_CONFIRMATORY_V1_EVAL",
        "nominal_result": "POSTHOC_MATCHED128_4V4_DUAL_BRANCH_SPECIALIST_CROSSOVER_EVAL_RESULT.json",
        "nominal_rows_sha256": "ea985b7ea547b6c8d5e2d606bc36220e3e7db0cbb66e0c3227dadacedaba38a0",
        "manifest": "4v4/dual_branch_deploy_manifest.json",
        "top50_record": "dual_branch_v1/matched128_4v4/DUAL_BRANCH_4V4_OWN_TOP50.json",
        "seed_list": "dual_branch_v1/matched128_4v4/DUAL_BRANCH_4V4_OWN_TOP50_seed_ids.json",
        "question": (
            "Matched-128 nominal is one-sided (Delta_B negative). Own-TOP50 nominal "
            "on these seeds is also one-sided (Delta_A +0.68, Delta_B -0.10). "
            "Report whether that own-TOP50 pattern is retained. Do not read the "
            "result as two-sided specialization of the method."
        ),
    },
    {
        "n": 6,
        "k_defend": 2,
        "seed_base_block": (25800001, 25800128),
        "registry_experiment_id": "STANDARDIZED_6V6_SEPARATED_EVAL",
        "nominal_result": "POSTHOC_MATCHED128_6V6_DUAL_BRANCH_SPECIALIST_CROSSOVER_EVAL_RESULT.json",
        "nominal_rows_sha256": "c855343a13026c5f27b69dcf3f8ba8438cb5d12bfde2b6017a99cedbbc716777",
        "manifest": "6v6/dual_branch_deploy_manifest.json",
        "top50_record": "dual_branch_v1/matched128_6v6/DUAL_BRANCH_6V6_OWN_TOP50.json",
        "seed_list": "dual_branch_v1/matched128_6v6/DUAL_BRANCH_6V6_OWN_TOP50_seed_ids.json",
        "question": (
            "Matched-128 nominal is one-sided (Delta_A negative). Own-TOP50 nominal "
            "on these selected seeds is two-sided in the mean (Delta_A +0.12, "
            "Delta_B +0.66) because the subset maximizes A-on-A plus B-on-B. "
            "Report whether that own-TOP50 pattern is retained. It does not say "
            "whether the matched-128 Pole-A result is repaired."
        ),
    },
)

CONDITIONS = (
    ("LOCALIZATION_MEDIUM", "localization_noise", "medium"),
    ("MOTION_MEDIUM", "motion_error", "medium"),
    ("DELAY_MEDIUM", "control_delay", "medium"),
)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"{now()} {msg}"
    print(line, flush=True)
    CHAIN.mkdir(parents=True, exist_ok=True)
    with (CHAIN / "chain_dual_branch_top50_noise.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def label_of(n: int, cond: str) -> str:
    return f"DUAL_BRANCH_{n}V{n}_TOP50_NOISE_{cond}"


def result_path(label: str) -> Path:
    return SD / f"{label}_SPECIALIST_CROSSOVER_EVAL_RESULT.json"


def rows_path(label: str) -> Path:
    return SD / f"{label.lower()}_specialist_crossover_eval_rows.csv"


def _ckpt(n: int, side: str, kind: str) -> str:
    name = f"pi_{side}_specialist_{n}v{n}_dual_branch_v1"
    leaf = f"final_{name}.zip" if kind == "defend" else f"attack_{name}.zip"
    return f"artifacts/scale_{n}v{n}_specialists/{name}/ckpts/{leaf}"


def load_seed_ids(scale: dict) -> list[int]:
    rec = json.loads((SD / scale["top50_record"]).read_text(encoding="utf-8"))
    listed = json.loads((SD / scale["seed_list"]).read_text(encoding="utf-8"))
    ids = [int(s) for s in rec["selected_seed_ids"]]
    if sorted(ids) != sorted(int(s) for s in listed):
        raise SystemExit(f"REFUSING: {scale['seed_list']} does not match {scale['top50_record']}")
    if len(ids) != 50 or len(set(ids)) != 50:
        raise SystemExit(f"REFUSING: {scale['top50_record']} does not contain 50 distinct seeds")
    lo, hi = scale["seed_base_block"]
    if any(not (lo <= s <= hi) for s in ids):
        raise SystemExit(f"REFUSING: a {scale['n']}v{scale['n']} TOP50 seed is outside {lo}..{hi}")
    if rec.get("rows") != (
        f"artifacts/strategic_demand/sppo/posthoc_matched128_{scale['n']}v{scale['n']}"
        f"_dual_branch_specialist_crossover_eval_rows.csv"
    ):
        raise SystemExit(f"REFUSING: {scale['top50_record']} was not selected from the dual-branch nominal rows")
    return sorted(ids)


def build_spec() -> dict:
    import hashlib
    scales_out = {}
    ablations = {}
    by_scale = {}
    for scale in SCALES:
        n = int(scale["n"])
        rows = SD / f"posthoc_matched128_{n}v{n}_dual_branch_specialist_crossover_eval_rows.csv"
        digest = hashlib.sha256(rows.read_bytes()).hexdigest()
        if digest != scale["nominal_rows_sha256"]:
            raise SystemExit(f"REFUSING: nominal rows hash changed for {n}v{n}")
        man = json.loads((ROOT / scale["manifest"]).read_text(encoding="utf-8"))
        if man.get("architecture") != "DUAL_BRANCH_ROLE_COMPOSITE_V1" or int(man["k_defend"]) != int(scale["k_defend"]):
            raise SystemExit(f"REFUSING: {scale['manifest']} is not the expected dual-branch manifest")
        pins = {}
        for key, side, kind in (
            ("pi_A_defend", "A", "defend"), ("pi_A_attack", "A", "attack"),
            ("pi_B_defend", "B", "defend"), ("pi_B_attack", "B", "attack"),
        ):
            rel = _ckpt(n, side, kind)
            if man[key]["path"].replace("\\", "/") != rel:
                raise SystemExit(f"REFUSING: {key} path {man[key]['path']} != {rel}")
            pins[key] = {"path": rel, "sha256": str(man[key]["sha256"]).lower()}
        seed_ids = load_seed_ids(scale)
        lo, hi = scale["seed_base_block"]
        block = f"{lo}..{hi}"
        labels = [label_of(n, cond) for cond, _, _ in CONDITIONS]
        for lab in labels:
            ablations[lab] = {
                "registry_experiment_id": scale["registry_experiment_id"],
                "block": block,
                "seed_ids": seed_ids,
                "primary_record": scale["nominal_result"],
                "system": "symmetric dual-branch role composite",
                "frozen_attack_A": pins["pi_A_attack"],
                "frozen_attack_B": pins["pi_B_attack"],
                "episodes": 200,
                "note": "post-hoc TOP50 diagnostic; seed IDs copied from the sealed own-TOP50 list; not re-ranked; block status unchanged",
            }
        seed_list = f"artifacts/strategic_demand/sppo/{scale['seed_list']}"
        common = (
            f".venv/Scripts/python.exe experiments/eval_specialist_crossover_scaled.py --team-size {n} "
            f"--spec artifacts/strategic_demand/sppo/DUAL_BRANCH_TOP50_NOISE_SPEC.json "
            f"--post-hoc-ablation-spec artifacts/strategic_demand/sppo/DUAL_BRANCH_TOP50_NOISE_SPEC.json "
            f"--seed-list {seed_list} --registry-experiment-id {scale['registry_experiment_id']} "
            f"--device cuda --pi-a-path {pins['pi_A_defend']['path']} --pi-b-path {pins['pi_B_defend']['path']} "
            f"--role-fixed-for-episode --role-k-defend {scale['k_defend']} "
            f"--frozen-attack-path {pins['pi_A_attack']['path']} "
            f"--frozen-attack-path-sha256 {pins['pi_A_attack']['sha256']} "
            f"--frozen-attack-path-b {pins['pi_B_attack']['path']} "
            f"--frozen-attack-path-b-sha256 {pins['pi_B_attack']['sha256']} "
            f"--dual-branch-deploy-manifest {scale['manifest']} --resume"
        )
        by_scale[str(n)] = {
            "common": common,
            "per_label": {
                label_of(n, cond): f"--label {label_of(n, cond)} --perturbation {family} --severity {severity}"
                for cond, family, severity in CONDITIONS
            },
        }
        scales_out[f"{n}v{n}"] = {
            "k_defend": int(scale["k_defend"]),
            "deploy_manifest": scale["manifest"],
            **pins,
            "registry_experiment_id": scale["registry_experiment_id"],
            "block": block,
            "seed_list": seed_list,
            "seed_ids": seed_ids,
            "nominal_result": scale["nominal_result"],
            "nominal_rows": f"artifacts/strategic_demand/sppo/posthoc_matched128_{n}v{n}_dual_branch_specialist_crossover_eval_rows.csv",
            "nominal_rows_sha256": digest,
            "top50_record": f"artifacts/strategic_demand/sppo/{scale['top50_record']}",
            "question": scale["question"],
            "labels": labels,
        }
    return {
        "record_id": "DUAL_BRANCH_TOP50_NOISE_SPEC",
        "status": "FROZEN",
        "arm": "POST_HOC_ABLATION",
        "confirmatory": False,
        "utc": "2026-10-07T21:30:00Z",
        "paper_label": "TOP50 deployment-robustness diagnostic",
        "classification": (
            "POST-HOC DESCRIPTIVE DIAGNOSTIC. The 50 seeds at each team size were "
            "already selected from that scale's dual-branch matched-128 nominal. "
            "Not an unbiased confirmatory robustness test. Not a fresh seed block. "
            "Not PAPER-FAITHFUL."
        ),
        "decided_by": (
            "user, 2026-10-07: under a one-day budget, evaluate medium localization, "
            "motion error, and 2-tick delay on the existing own-TOP50 seed IDs. "
            "Same IDs across the three disturbances. Do not re-rank. 2v2, then 4v4, "
            "then 6v6, on this machine."
        ),
        "parent": [
            "DEPLOYMENT_ROBUSTNESS_SPEC.json (medium tier only)",
            "experiments/select_own_top50.py (selection rule already applied; not rerun here)",
        ],
        "THE_QUESTION": (
            "On the already-chosen own-TOP50 seeds, how do medium localization noise, "
            "motion error, and control delay change Delta_A and Delta_B relative to "
            "the nominal rows of those same seeds?"
        ),
        "SEED_RULE": (
            "Copy selected_seed_ids from the sealed DUAL_BRANCH_*_OWN_TOP50 records. "
            "Do not recompute the ranking. Do not rank again under a disturbance."
        ),
        "INTERPRETATION_locked": {
            "2v2": SCALES[0]["question"],
            "4v4": SCALES[1]["question"],
            "6v6": SCALES[2]["question"],
        },
        "SYSTEMS_locked": scales_out,
        "CONDITIONS_locked": {
            "localization_noise": {"severity": "medium", "sigma_cells": 0.06},
            "motion_error": {"severity": "medium", "sigma_cells": 0.06},
            "control_delay": {"severity": "medium", "ticks": 2, "seconds": 1.0},
            "tier_source": "DEPLOYMENT_ROBUSTNESS_SPEC.json#TIERS via eval_specialist_crossover_scaled.resolve_perturbation",
            "nominal": "not rerun; paired against the matched-128 rows restricted to these 50 seeds",
        },
        "POST_HOC_MATCHED_ROLE_ABLATIONS": ablations,
        "READINGS": {
            "per_condition": "Delta_A and Delta_B on the 50 seeds; mean and sample std (ddof=1)",
            "paired_change": "disturbance Delta minus nominal Delta on the same seed; 95% paired bootstrap via eval_hog_psp_v3._mean_ci",
            "caption_caveat": "For localization and motion perturbations, we use a matched numeric severity schedule of 0.03, 0.06, and 0.12 cells to enable controlled comparison across perturbation families. These levels are not assumed to represent equivalent behavioral difficulty.",
        },
        "LAUNCH": {"order": [2, 4, 6], "by_scale": by_scale, "episodes_new_per_scale": 600},
        "NOT_AUTHORIZED_BY_THIS_SPEC": [
            "re-ranking seeds under noise",
            "re-running nominal",
            "low or high tiers",
            "a fresh seed block",
            "any retraining",
            "calling the table an unbiased confirmatory robustness result",
            "claiming PAPER-FAITHFUL",
        ],
    }


def freeze_specs() -> None:
    if SPEC_PATH.is_file():
        raise SystemExit(f"REFUSING: {SPEC_PATH.name} already exists; the spec is not rewritten")
    spec = build_spec()
    SPEC_PATH.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    log(f"froze {SPEC_PATH.name}")


def state(**kw) -> dict:
    s = json.loads(STATE.read_text(encoding="utf-8-sig")) if STATE.is_file() else {
        "record": "CHAIN_DUAL_BRANCH_TOP50_NOISE", "steps": {}, "pids": {},
    }
    s.update({k: v for k, v in kw.items() if k != "step"})
    if "step" in kw:
        s.setdefault("steps", {})[kw["step"]] = now()
    if "status" in kw and kw["status"] != "ABORTED":
        s.pop("reason", None)
    s["updated_utc"] = now()
    STATE.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
    return s


def done(step: str) -> bool:
    return STATE.is_file() and step in json.loads(STATE.read_text(encoding="utf-8-sig")).get("steps", {})


def fail(msg: str) -> None:
    log(f"ABORT: {msg}")
    state(status="ABORTED", reason=msg)
    sys.exit(1)


def _alive(pid: int) -> bool:
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
    return str(pid) in (r.stdout or "")


def _label_alive(label: str) -> bool:
    """True if any python process is running this evaluation label.

    On Windows a venv ``python.exe`` is a launcher that keeps a base-interpreter
    child; either PID may be the one stored in state, so match by --label.
    """
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
         f"Where-Object {{ $_.CommandLine -match '--label {label}(\\s|$)' }} | "
         f"Select-Object -First 1 -ExpandProperty ProcessId"],
        capture_output=True, text=True,
    )
    return bool((r.stdout or "").strip())


def _partial_stale(label: str) -> bool:
    partial = SD / f"{label.lower()}_specialist_crossover_eval_rows.PARTIAL.jsonl"
    return partial.is_file() and time.time() - partial.stat().st_mtime > STALE_PARTIAL_S


def checkpoints_present(spec: dict, n: int) -> list[str]:
    info = spec["SYSTEMS_locked"][f"{n}v{n}"]
    missing = []
    for key in ("pi_A_defend", "pi_A_attack", "pi_B_defend", "pi_B_attack"):
        rel = info[key]["path"]
        if not (ROOT / rel).is_file():
            missing.append(rel)
    return missing


def commands_for(spec: dict, n: int) -> dict[str, str]:
    block = spec["LAUNCH"]["by_scale"][str(n)]
    common = block["common"].split(" ", 1)[1]
    return {lab: f"{common} {extra}" for lab, extra in block["per_label"].items()}


def dry_run(n: int, cmds: dict[str, str]) -> None:
    step = f"dry_run_{n}v{n}"
    if done(step):
        return
    for lab, args in cmds.items():
        if result_path(lab).is_file():
            continue
        r = subprocess.run(
            [PY, *args.replace(" --resume", "").split(), "--dry-run"],
            cwd=ROOT, capture_output=True, text=True,
        )
        if r.returncode != 0:
            tail = (r.stdout + r.stderr)[-800:]
            fail(f"dry-run {lab} exit {r.returncode}: {tail}")
        log(f"dry-run PASS {lab}")
    state(step=step)


def launch_missing(n: int, cmds: dict[str, str]) -> None:
    st = json.loads(STATE.read_text(encoding="utf-8-sig")) if STATE.is_file() else {}
    pids = dict(st.get("pids") or {})
    logdir = SD / "dual_branch_v1" / "noise_top50" / f"{n}v{n}"
    logdir.mkdir(parents=True, exist_ok=True)
    for lab, args in cmds.items():
        if result_path(lab).is_file():
            continue
        pid = pids.get(lab)
        if (pid and _alive(int(pid))) or _label_alive(lab):
            continue
        out = open(logdir / f"{lab}.log", "a", encoding="utf-8")
        err = open(logdir / f"{lab}.log.err", "a", encoding="utf-8")
        p = subprocess.Popen(
            [PY, *args.split()], cwd=ROOT, stdout=out, stderr=err,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008,
        )
        pids[lab] = p.pid
        log(f"launched {lab} pid={p.pid}")
    state(pids=pids, step=f"launched_{n}v{n}")


def wait_scale(n: int, cmds: dict[str, str]) -> None:
    while True:
        missing = [lab for lab in cmds if not result_path(lab).is_file()]
        if not missing:
            break
        st = json.loads(STATE.read_text(encoding="utf-8-sig"))
        pids = st.get("pids") or {}
        dead = [
            lab for lab in missing
            if not _label_alive(lab) and not (lab in pids and _alive(int(pids[lab])))
        ]
        if dead:
            fail(f"{n}v{n} exited without a sealed record: {dead}")
        if any(_partial_stale(lab) for lab in missing):
            fail(f"{n}v{n}: PARTIAL has not grown for 3 h ({missing})")
        time.sleep(POLL_S)
    for lab in cmds:
        rec = json.loads(result_path(lab).read_text(encoding="utf-8"))
        if rec.get("status") != "SEALED":
            fail(f"{lab} status {rec.get('status')!r}")
    log(f"{n}v{n} TOP50 noise: all three sealed")


def _deltas(rows_file: Path, seeds: set[int]) -> dict[int, tuple[float, float]]:
    by: dict[int, dict] = {}
    with rows_file.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            seed = int(row["seed"])
            if seed not in seeds:
                continue
            by.setdefault(seed, {})[(row["policy"], row["pole"])] = int(row["win"])
    out = {}
    for seed, cells in by.items():
        out[seed] = (
            cells[("pi_A", "A")] - cells[("pi_B", "A")],
            cells[("pi_B", "B")] - cells[("pi_A", "B")],
        )
    return out


def write_readout(n: int, spec: dict) -> None:
    step = f"readout_{n}v{n}"
    if done(step):
        return
    import numpy as np
    from experiments.eval_hog_psp_v3 import _mean_ci

    info = spec["SYSTEMS_locked"][f"{n}v{n}"]
    seeds = [int(s) for s in info["seed_ids"]]
    nominal = _deltas(ROOT / info["nominal_rows"], set(seeds))
    if set(nominal) != set(seeds):
        fail(f"{n}v{n} nominal rows are missing TOP50 seeds")
    report = {
        "record": f"DUAL_BRANCH_{n}V{n}_TOP50_NOISE_READOUT",
        "paper_label": "TOP50 deployment-robustness diagnostic",
        "classification": spec["classification"],
        "utc": now(),
        "question": info["question"],
        "n_seeds": 50,
        "seed_ids": seeds,
        "conditions": {},
    }
    order = sorted(seeds)
    nom_a = np.array([nominal[s][0] for s in order], dtype=np.float64)
    nom_b = np.array([nominal[s][1] for s in order], dtype=np.float64)
    report["nominal_top50"] = {
        "Delta_A": {"mean": float(nom_a.mean()), "std": float(nom_a.std(ddof=1))},
        "Delta_B": {"mean": float(nom_b.mean()), "std": float(nom_b.std(ddof=1))},
    }
    for lab in spec["LAUNCH"]["by_scale"][str(n)]["per_label"]:
        got = _deltas(rows_path(lab), set(seeds))
        if set(got) != set(seeds):
            fail(f"{lab} does not cover the frozen TOP50 seeds")
        a = np.array([got[s][0] for s in order], dtype=np.float64)
        b = np.array([got[s][1] for s in order], dtype=np.float64)
        report["conditions"][lab] = {
            "Delta_A": {"mean": float(a.mean()), "std": float(a.std(ddof=1))},
            "Delta_B": {"mean": float(b.mean()), "std": float(b.std(ddof=1))},
            "paired_change_A": _mean_ci(a - nom_a),
            "paired_change_B": _mean_ci(b - nom_b),
        }
    out = SD / f"DUAL_BRANCH_{n}V{n}_TOP50_NOISE_READOUT.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {out.name}")
    state(step=step)


def main() -> int:
    if "--freeze" in sys.argv:
        freeze_specs()
        return 0
    if not SPEC_PATH.is_file():
        fail(f"missing {SPEC_PATH.name}; run --freeze before any episode")
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        fail("spec is not frozen")
    # Re-check that the spec's seed lists still equal the sealed own-TOP50 files.
    for scale in SCALES:
        ids = load_seed_ids(scale)
        frozen = spec["SYSTEMS_locked"][f"{scale['n']}v{scale['n']}"]["seed_ids"]
        if ids != [int(s) for s in frozen]:
            fail(f"{scale['n']}v{scale['n']} spec seed_ids differ from the sealed own-TOP50 list")
    active = [2, 4]
    if "--include-6v6" in sys.argv:
        active = [2, 4, 6]
    else:
        log("6v6 deferred (user 2026-10-07); frozen spec unchanged, no 6v6 episodes")
    state(status="RUNNING", active_scales=active)
    for n in active:
        if done(f"sealed_{n}v{n}"):
            log(f"{n}v{n} already sealed; skipping")
            continue
        missing = checkpoints_present(spec, n)
        if missing:
            hold = {
                "record": f"DUAL_BRANCH_{n}V{n}_TOP50_NOISE_HOLD",
                "utc": now(),
                "reason": "dual-branch checkpoints are not on this machine",
                "missing": missing,
                "spec": SPEC_PATH.name,
                "what_remains": (
                    f"copy the four {n}v{n} dual-branch zips onto this machine "
                    "(or run this chain where they already live), then rerun the chain; "
                    "it resumes and does not re-rank seeds"
                ),
            }
            hold_path = SD / f"DUAL_BRANCH_{n}V{n}_TOP50_NOISE_HOLD.json"
            hold_path.write_text(json.dumps(hold, indent=2) + "\n", encoding="utf-8")
            log(f"HOLD {n}v{n}: checkpoints missing ({missing[0]})")
            state(status=f"HOLD_{n}v{n}_CHECKPOINTS_MISSING", step=f"hold_{n}v{n}")
            return 0
        cmds = commands_for(spec, n)
        dry_run(n, cmds)
        log(f"starting {n}v{n} TOP50 noise (3 x 200 episodes)")
        launch_missing(n, cmds)
        wait_scale(n, cmds)
        write_readout(n, spec)
        state(step=f"sealed_{n}v{n}", status=f"{n}v{n}_SEALED")
    if active == [2, 4, 6]:
        state(status="ALL_SCALES_SEALED", step="all_scales_sealed")
        log("TOP50 noise diagnostic: 2v2, 4v4, and 6v6 sealed")
    else:
        state(status="2V2_AND_4V4_SEALED", step="stopped_after_4v4")
        log("TOP50 noise diagnostic: 2v2 and 4v4 sealed. 6v6 not started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
