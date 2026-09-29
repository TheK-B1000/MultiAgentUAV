r"""Chain stage 1: 4v4 baselines sealed -> 4v4 noise suite (mirror of the sealed 2v2 one) -> sealed.

    .venv\Scripts\python.exe artifacts/strategic_demand/sppo/chain/run_chain_4v4_noise.py

Runs detached, unattended. Every step fails closed and records its state in CHAIN_4V4_NOISE_STATE.json:
  1. wait for the four 4v4 sharing-eval labels to seal (abort if one's process vanished without a record)
  2. commit the sealed 4v4 baseline records
  3. derive STANDARDIZED_4V4_NOISE_SPEC.json from the sealed 2v2 noise spec (only the system, seeds,
     labels and team size change), reserve a fresh 128-seed block shared by the four labels, commit
  4. dry-run all four commands; abort on any failure
  5. launch nominal / localization / motion / delay detached with --resume, wait for all four to seal
  6. commit the sealed noise records; write CHAIN_4V4_NOISE_DONE.json (the 6v6 stage waits on it)
Never edits scientific code; resumable (a rerun skips finished steps).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]          # AICTFProject
REPO = ROOT.parent
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
CHAIN = SD / "chain"
STATE = CHAIN / "CHAIN_4V4_NOISE_STATE.json"
DONE = CHAIN / "CHAIN_4V4_NOISE_DONE.json"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
sys.path.insert(0, str(ROOT))

BASE_LABELS = ["STANDARDIZED_4V4_SHARE_ENCODER", "STANDARDIZED_4V4_FULLY_SHARED_Z", "STANDARDIZED_4V4_GENERALIST"]
REF_LABEL = "STANDARDIZED_4V4_SEPARATED_GENERALIST_REF"
NOISE_SPEC = SD / "STANDARDIZED_4V4_NOISE_SPEC.json"
POLL_S = 300


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"{now()} {msg}"
    print(line, flush=True)
    with (CHAIN / "chain_4v4_noise.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def state(**kw) -> dict:
    s = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {"record": "CHAIN_4V4_NOISE", "steps": {}}
    s.update({k: v for k, v in kw.items() if k != "step"})
    if "step" in kw:
        s["steps"][kw["step"]] = now()
    s["updated_utc"] = now()
    STATE.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
    return s


def done(step: str) -> bool:
    return STATE.is_file() and step in json.loads(STATE.read_text(encoding="utf-8"))["steps"]


def fail(msg: str) -> None:
    log(f"ABORT: {msg}")
    state(status="ABORTED", reason=msg)
    sys.exit(1)


def git(*a: str) -> str:
    r = subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)
    if r.returncode != 0:
        fail(f"git {' '.join(a)}: {r.stderr.strip()[:300]}")
    return r.stdout.strip()


def commit(paths: list[str], title: str, body: str) -> None:
    git("add", "--", *paths)
    if not git("diff", "--cached", "--name-only"):
        log(f"nothing to commit for: {title}")
        return
    git("commit", "-q", "-m", title, "-m", body, "-m", "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    log(f"committed {git('log', '--oneline', '-1')}")


def result_path(label: str, separated: bool) -> Path:
    return SD / (f"{label}_SPECIALIST_CROSSOVER_EVAL_RESULT.json" if separated else f"{label}_CROSSOVER_EVAL_RESULT.json")


def _alive(pid: int) -> bool:
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
    return str(pid) in r.stdout


def rel(p: Path) -> str:
    return str(p.relative_to(REPO)).replace("\\", "/")


CHECK = "--check" in sys.argv


def main() -> int:
    CHAIN.mkdir(parents=True, exist_ok=True)
    if not CHECK:
        state(status="RUNNING", pid=__import__("os").getpid())

    # 1-2: 4v4 baselines sealed -> commit their records
    if not CHECK and not done("baselines_committed"):
        labels = {**{l: False for l in BASE_LABELS}, REF_LABEL: True}
        log("waiting for the four 4v4 baseline evaluations to seal")
        while True:
            missing = [l for l, sep in labels.items() if not result_path(l, sep).is_file()]
            if not missing:
                break
            partial = SD / f"{REF_LABEL.lower()}_specialist_crossover_eval_rows.PARTIAL.jsonl"
            stale = partial.is_file() and time.time() - partial.stat().st_mtime > 3 * 3600
            if stale:
                fail(f"4v4 baselines: {missing} not sealed and the reference PARTIAL has not grown for 3 h")
            time.sleep(POLL_S)
        for l in BASE_LABELS + [REF_LABEL]:
            rec = json.loads(result_path(l, l == REF_LABEL).read_text(encoding="utf-8"))
            if rec.get("status") != "SEALED":
                fail(f"{l} status {rec.get('status')!r}")
        pats = [f"{rel(SD)}/{p}" for p in ("STANDARDIZED_4V4_SHARE_ENCODER_*", "STANDARDIZED_4V4_FULLY_SHARED_Z_*",
                                          "STANDARDIZED_4V4_GENERALIST_*", "STANDARDIZED_4V4_SEPARATED_GENERALIST_REF_*",
                                          "standardized_4v4_*_crossover_eval_rows.csv")]
        commit(pats + [rel(ROOT / "artifacts" / "SEED_REGISTRY.json")],
               "Seal the 4v4 baselines: Share-Encoder, Fully Shared+z, Generalist (+ Separated reference)",
               "All four labels on STANDARDIZED_4V4_SHARING_EVAL sealed; records committed by the 4v4->noise chain.")
        state(step="baselines_committed")

    # 3: derive + freeze the 4v4 noise spec, reserve its block
    if CHECK or not done("noise_spec_frozen"):
        from experiments import seed_registry as SR
        two = json.loads((SD / "STANDARDIZED_2V2_NOISE_SPEC.json").read_text(encoding="utf-8"))
        ev = json.loads((SD / "STANDARDIZED_4V4_SHARING_EVAL_SPEC.json").read_text(encoding="utf-8"))
        ck = ev["SEPARATED_REFERENCE"]["checkpoints"]
        lo = SR.next_free(128)
        labels = [l.replace("2V2", "4V4") for l in two["SEEDS"]["shared_by_labels"]]
        s = json.loads(json.dumps(two))
        s.update(record_id="STANDARDIZED_4V4_NOISE_SPEC", status="FROZEN_BEFORE_SEED_SPEND", utc=now(),
                 derived_from="STANDARDIZED_2V2_NOISE_SPEC.json (only the system, seeds, labels and team size change)",
                 decided_by="PI, 2026-09-29: 4v4 noise starts automatically after the 4v4 baselines, mirroring the sealed 2v2 noise suite.",
                 THE_QUESTION=two["THE_QUESTION"].replace("2v2", "4v4"))
        s["SYSTEM_locked"] = {
            "name": "final 4v4 Separated system (the 4v4 main result's system, DEFEND_ATTACK_SPLIT_POLICY_A_V1_CONFIRMATORY_V1)",
            "pole_A_policy": "split composite: a4 pi_D on DEFEND + b3 pi_A frozen on ATTACK, roles fixed for the episode, CLOSEST_DEFENDS k=2",
            "pole_B_policy": "b3-corrected pi_B", "checkpoints": ck}
        s["CONDITIONS_locked"] = {k.replace("2V2", "4V4"): v for k, v in two["CONDITIONS_locked"].items()}
        s["SEEDS"] = {**two["SEEDS"], "registry_experiment_id": "STANDARDIZED_4V4_NOISE_EVAL",
                      "block": f"{lo}..{lo + 127}", "shared_by_labels": labels}
        s["READINGS"] = {**two["READINGS"], "headline": two["READINGS"]["headline"].replace(
            "the 2v2 main value stays +0.117/+0.539", "the 4v4 main value stays +0.156/+0.406")}
        common = (f".venv/Scripts/python.exe experiments/eval_specialist_crossover_scaled.py --team-size 4 "
                  f"--spec artifacts/strategic_demand/sppo/STANDARDIZED_4V4_NOISE_SPEC.json "
                  f"--pi-a-path {ck['pi_D']['path']} --pi-b-path {ck['pi_B']['path']} --seed-base {lo} --n-seeds 128 "
                  f"--device cuda --role-fixed-for-episode --frozen-attack-path {ck['pi_A_frozen_attack']['path']} "
                  f"--frozen-attack-path-sha256 {ck['pi_A_frozen_attack']['sha256']} --role-k-defend 2 "
                  f"--registry-experiment-id STANDARDIZED_4V4_NOISE_EVAL --resume")
        s["LAUNCH"] = {**two["LAUNCH"], "common": common,
                       "per_label": {k.replace("2V2", "4V4"): v.replace("2V2", "4V4") for k, v in two["LAUNCH"]["per_label"].items()},
                       "when": "automatically after the four 4v4 baseline labels sealed (chain stage 1)"}
        s["NOT_AUTHORIZED_BY_THIS_SPEC"] = [x.replace("any 4v4 or 6v6 evaluation", "any 6v6 evaluation")
                                            for x in two["NOT_AUTHORIZED_BY_THIS_SPEC"]]
        if CHECK:
            out = CHAIN / "_CHECK_STANDARDIZED_4V4_NOISE_SPEC.json"
            out.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
            print(f"--check: derived spec written to {out}; nothing reserved, launched or committed")
            return 0
        NOISE_SPEC.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
        SR.allocate("STANDARDIZED_4V4_NOISE_EVAL", lo, lo + 127, "sealed_confirmatory",
                    "4v4 deployment-noise suite on the final system: nominal + medium localization/motion/delay, matched seeds",
                    spec=NOISE_SPEC.name, shared_by_labels=labels)
        commit([rel(NOISE_SPEC), rel(ROOT / "artifacts" / "SEED_REGISTRY.json")],
               "Freeze the 4v4 noise spec (mirror of the sealed 2v2 one); reserve its matched block",
               f"STANDARDIZED_4V4_NOISE_EVAL {lo}..{lo + 127} shared by {labels}. Derived by the 4v4->noise chain.")
        state(step="noise_spec_frozen", block=f"{lo}..{lo + 127}")

    spec = json.loads(NOISE_SPEC.read_text(encoding="utf-8"))
    common = spec["LAUNCH"]["common"].split(" ", 1)[1]
    cmds = {lab: f"{common} {extra}" for lab, extra in spec["LAUNCH"]["per_label"].items()}

    # 4: dry-runs
    if not done("dry_runs_passed"):
        for lab, a in cmds.items():
            r = subprocess.run([PY, *a.replace(" --resume", "").split(), "--dry-run"], cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                fail(f"dry-run {lab} exit {r.returncode}: {(r.stdout + r.stderr)[-400:]}")
            log(f"dry-run PASS {lab}")
        state(step="dry_runs_passed")

    # 5: launch + wait
    st = json.loads(STATE.read_text(encoding="utf-8"))
    pids = st.get("noise_pids") or {}
    if not done("noise_launched"):
        logdir = SD / "suite_sharing_std" / "4v4" / "noise"
        logdir.mkdir(parents=True, exist_ok=True)
        for lab, a in cmds.items():
            if result_path(lab, True).is_file():
                continue
            out = open(logdir / f"{lab}.log", "a", encoding="utf-8")
            err = open(logdir / f"{lab}.log.err", "a", encoding="utf-8")
            p = subprocess.Popen([PY, *a.split()], cwd=ROOT, stdout=out, stderr=err,
                                 creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008)  # DETACHED_PROCESS
            pids[lab] = p.pid
            log(f"launched {lab} pid={p.pid}")
        state(step="noise_launched", noise_pids=pids, launch_commit=git("rev-parse", "--short", "HEAD"))
    while True:
        missing = [l for l in cmds if not result_path(l, True).is_file()]
        if not missing:
            break
        dead = [l for l in missing if l in pids and not _alive(pids[l])]
        if dead:
            fail(f"noise: exited without a sealed record: {dead} (rerun this chain; runs resume from PARTIAL)")
        time.sleep(POLL_S)
    for l in cmds:
        if json.loads(result_path(l, True).read_text(encoding="utf-8")).get("status") != "SEALED":
            fail(f"{l} not SEALED")
    log("4v4 noise: all four sealed")

    # 6: commit + done marker
    if not done("noise_committed"):
        commit([f"{rel(SD)}/STANDARDIZED_4V4_NOISE_*", f"{rel(SD)}/standardized_4v4_noise_*_specialist_crossover_eval_rows.csv",
                rel(ROOT / "artifacts" / "SEED_REGISTRY.json")],
               "Seal the 4v4 deployment-noise suite", "Nominal + medium localization/motion/delay on one matched block; committed by the chain.")
        state(step="noise_committed")
    DONE.write_text(json.dumps({"record": "CHAIN_4V4_NOISE_DONE", "utc": now(),
                                "noise_labels": list(cmds)}, indent=2) + "\n", encoding="utf-8")
    state(status="DONE")
    log("chain stage 1 DONE -> 6v6 stage may start")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
