"""REPLICATION_2V2_DUAL_BRANCH_V1: does the 2v2 dual-branch recipe repeatedly produce A/B separation?

Frozen spec: artifacts/strategic_demand/sppo/2v2_strengthening/REPLICATION_2V2_DUAL_BRANCH_V1_SPEC.json

Phases (each idempotent; finished work is skipped):
  train    six 200k runs (rep1-3 x A/B) with the EXACT argv of 2v2/run_dual_branch_2v2.train_args,
           only --seed / --experiment-id / --run-label-suffix differ; 3 at a time
  export   ATTACK branch of each final, same export as the 2v2 driver; per-replicate deploy manifest
  eval     original + rep1-3 on the shared block 31200001..31200128 via
           eval_specialist_crossover_scaled.py (dual-branch composite flags of the 2v2 matched-128)
  readout  k/3 headline + descriptive LCB count; write-once RESULT

    .venv/Scripts/python.exe experiments/run_replication_2v2.py --smoke          # 999xxxxx, 5k steps
    .venv/Scripts/python.exe experiments/run_replication_2v2.py --phase all      # after 4v4 finishes
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.strengthening_2v2_common as C  # noqa: E402

SPEC_ID = "REPLICATION_2V2_DUAL_BRANCH_V1"
EVAL_REG_ID = "REPLICATION_2V2_DUAL_BRANCH_V1_EVAL"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
OUT = C.STRENGTH_DIR / "replication"
EVAL_SPEC = OUT / "REPLICATION_2V2_DUAL_BRANCH_V1_EVAL_SPEC.json"
LABELS = {"original": "REPLICATION_2V2_ORIGINAL", "rep1": "REPLICATION_2V2_REP1",
          "rep2": "REPLICATION_2V2_REP2", "rep3": "REPLICATION_2V2_REP3"}
SMOKE_SEEDS = {"A": 99973001, "B": 99973002}
SMOKE_STEPS = 5000
STEPS = 200_000
PARALLEL = 3


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"{_now()} {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "replication.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def fail(msg: str) -> None:
    log(f"FAIL: {msg}")
    raise SystemExit(msg)


def driver():
    """The 2v2 dual-branch driver module: its train_args IS the frozen recipe."""
    spec = importlib.util.spec_from_file_location("run_dual_branch_2v2", ROOT / "2v2" / "run_dual_branch_2v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.log, mod.fail = log, fail          # keep our messages out of the 2v2 driver's own log
    return mod


def runs(spec: dict, smoke: bool) -> dict:
    """{(rep, side): {seed, eid, suffix, final, attack}}"""
    out = {}
    reps = {"repsmoke": {"A": SMOKE_SEEDS["A"], "B": SMOKE_SEEDS["B"]}} if smoke else spec["replicates"]["training_seeds"]
    for rep, sides in reps.items():
        for side, seed in sides.items():
            suffix = f"_dual_branch_{rep}"
            run_dir = f"artifacts/scale_2v2_specialists/pi_{side}_specialist_2v2{suffix}"
            out[(rep, side)] = {
                "seed": int(seed), "suffix": suffix, "smoke": smoke,
                "eid": None if smoke else f"REPLICATION_2V2_{rep.upper()}_{side}_TRAIN",
                "run_dir": run_dir,
                "final": f"{run_dir}/ckpts/final_pi_{side}_specialist_2v2{suffix}.zip",
                "attack": f"{run_dir}/ckpts/attack_pi_{side}_specialist_2v2{suffix}.zip",
            }
    return out


def find_final(r: dict) -> Path | None:
    p = ROOT / r["final"]
    if p.is_file():
        return p
    if r["smoke"]:   # smoke runs are labelled smoke_*; locate the newest matching final
        hits = sorted((ROOT / "artifacts").rglob(f"final_*pi_{r['final'].split('pi_')[-1].split('_specialist')[0]}"
                                                 f"_specialist_2v2{r['suffix']}.zip"), key=lambda q: q.stat().st_mtime)
        return hits[-1] if hits else None
    return None


# ------------------------------------------------------------------ phases
def phase_train(spec: dict, smoke: bool) -> None:
    D = driver()
    todo = [(k, r) for k, r in runs(spec, smoke).items() if find_final(r) is None]
    log(f"train: {len(todo)} run(s) to launch ({'smoke ' + str(SMOKE_STEPS) if smoke else STEPS} steps)")
    live: list[tuple] = []
    env = dict(os.environ, FOR_DISABLE_CONSOLE_CTRL_HANDLER="1")
    while todo or live:
        while todo and len(live) < PARALLEL:
            (rep, side), r = todo.pop(0)
            argv = D.train_args(side, SMOKE_STEPS if smoke else STEPS, r["seed"], r["eid"], r["suffix"], smoke=smoke)
            logf = (OUT / f"train_{rep}_{side}.log").open("w", encoding="utf-8")
            log(f"launch {rep} {side} seed {r['seed']}: {' '.join(argv)}")
            p = subprocess.Popen([PY, *argv], cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, env=env)
            live.append(((rep, side), r, p, logf))
        time.sleep(15)
        for item in list(live):
            (rep, side), r, p, logf = item
            if p.poll() is not None:
                logf.close()
                live.remove(item)
                if p.returncode != 0 or find_final(r) is None:
                    fail(f"{rep} {side} training exited {p.returncode} without a final checkpoint "
                         f"(see train_{rep}_{side}.log); a failed replicate is REPORTED, never retrained")
                log(f"{rep} {side} TRAIN DONE sha {C.sha256(find_final(r))[:16]}")


def phase_export(spec: dict, smoke: bool) -> dict:
    D = driver()
    manifests = {}
    rs = runs(spec, smoke)
    for rep in sorted({k[0] for k in rs}):
        pins = {}
        for side in ("A", "B"):
            r = rs[(rep, side)]
            final = find_final(r)
            if final is None:
                fail(f"{rep} {side}: no final checkpoint to export")
            attack = final.parent / final.name.replace("final_", "attack_", 1)
            if not attack.is_file():
                D.export_attack_branch(str(final.relative_to(ROOT)).replace("\\", "/"),
                                       str(attack.relative_to(ROOT)).replace("\\", "/"))
            pins[f"pi_{side}_defend"] = {"path": str(final.relative_to(ROOT)).replace("\\", "/"), "sha256": C.sha256(final)}
            pins[f"pi_{side}_attack"] = {"path": str(attack.relative_to(ROOT)).replace("\\", "/"), "sha256": C.sha256(attack)}
        man = {"architecture": "DUAL_BRANCH_ROLE_COMPOSITE_V1", "team_size": 2, "k_defend": 1,
               "replicate": rep, "spec": SPEC_ID, "smoke": smoke,
               "note": "ATTACK zips are trained branches exported from the dual-branch finals.", **pins}
        path = OUT / f"{rep}_deploy_manifest.json"
        if path.is_file() and not smoke:
            old = json.loads(path.read_text(encoding="utf-8"))
            if any(old[k]["sha256"] != man[k]["sha256"] for k in pins):
                fail(f"{path.name} exists with different pins; refusing to overwrite")
        path.write_text(json.dumps(man, indent=2), encoding="utf-8")
        manifests[rep] = path
        log(f"{rep} deploy manifest {path.name}")
    return manifests


def write_eval_spec(spec: dict, manifests: dict, smoke: bool) -> Path:
    """Derived evaluator spec (FROZEN, confirmatory), pinned to the frozen parent by sha."""
    path = OUT / ("SMOKE_" + EVAL_SPEC.name if smoke else EVAL_SPEC.name)
    doc = {
        "record_id": "REPLICATION_2V2_DUAL_BRANCH_V1_EVAL_SPEC",
        "status": "FROZEN_DERIVED",
        "confirmatory": True,
        "derived_from": {"spec": f"2v2_strengthening/{SPEC_ID}_SPEC.json",
                         "sha256": C.sha256(C.STRENGTH_DIR / f"{SPEC_ID}_SPEC.json")},
        "block": spec["evaluation_frozen"]["block"],
        "registry_experiment_id": EVAL_REG_ID,
        "labels": LABELS,
        "pairs": {rep: {"deploy_manifest": str(Path(m).relative_to(ROOT)).replace("\\", "/"),
                        **{k: v for k, v in json.loads(Path(m).read_text(encoding="utf-8")).items() if k.startswith("pi_")}}
                  for rep, m in manifests.items()},
        "smoke": smoke,
    }
    if path.is_file() and not smoke:
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["pairs"] != doc["pairs"]:
            fail(f"{path.name} exists with different pins; refusing to overwrite")
        return path
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def eval_argv(label: str, man_path: Path, spec_path: Path, block: list[int], dry: bool) -> list[str]:
    man = json.loads(Path(man_path).read_text(encoding="utf-8"))
    for k in ("pi_A_defend", "pi_B_defend", "pi_A_attack", "pi_B_attack"):
        if C.sha256(ROOT / man[k]["path"]) != man[k]["sha256"]:
            fail(f"{label}: {k} does not match {Path(man_path).name}")
    lo, hi = block
    return ["experiments/eval_specialist_crossover_scaled.py", "--team-size", "2",
            "--spec", str(spec_path.relative_to(ROOT)).replace("\\", "/"),
            "--seed-base", str(lo), "--n-seeds", str(hi - lo + 1),
            "--registry-experiment-id", EVAL_REG_ID, "--label", label, "--device", "cuda",
            "--pi-a-path", man["pi_A_defend"]["path"], "--pi-b-path", man["pi_B_defend"]["path"],
            "--role-fixed-for-episode", "--role-k-defend", "1",
            "--frozen-attack-path", man["pi_A_attack"]["path"],
            "--frozen-attack-path-sha256", man["pi_A_attack"]["sha256"],
            "--frozen-attack-path-b", man["pi_B_attack"]["path"],
            "--frozen-attack-path-b-sha256", man["pi_B_attack"]["sha256"],
            "--dual-branch-deploy-manifest", str(Path(man_path).relative_to(ROOT)).replace("\\", "/"),
            *(["--dry-run"] if dry else ["--resume"])]


def phase_eval(spec: dict, manifests: dict, smoke: bool) -> None:
    pairs = dict(manifests)
    if not smoke:
        pairs = {"original": C.DEPLOY_MANIFEST, **pairs}
    spec_path = write_eval_spec(spec, pairs, smoke)
    block = spec["evaluation_frozen"]["block"]
    env = dict(os.environ, FOR_DISABLE_CONSOLE_CTRL_HANDLER="1")
    procs = []
    for rep, man in pairs.items():
        label = LABELS.get(rep, "REPLICATION_2V2_REP1")    # smoke dry-runs borrow a declared label
        result = C.SD / f"{label}_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
        if result.is_file() and not smoke:
            log(f"eval {label} already sealed")
            continue
        argv = eval_argv(label, man, spec_path, block, dry=smoke)
        logf = (OUT / f"eval_{rep}{'_dryrun' if smoke else ''}.log").open("w", encoding="utf-8")
        log(f"eval {rep} -> {label}{' (DRY RUN)' if smoke else ''}")
        procs.append((label, subprocess.Popen([PY, *argv], cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, env=env), logf))
        if smoke:
            break
    for label, p, logf in procs:
        p.wait()
        logf.close()
        log(f"eval {label} exited {p.returncode}")
        if smoke and p.returncode != 0:
            fail(f"smoke dry-run for {label} failed")


def phase_readout(spec: dict) -> None:
    result_path = OUT / f"{SPEC_ID}_RESULT.json"
    if result_path.exists():
        fail(f"{result_path.name} already sealed (write-once)")
    pairs = {}
    for rep, label in LABELS.items():
        p = C.SD / f"{label}_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
        if not p.is_file():
            fail(f"missing {p.name}; readout needs all four pairs")
        g = json.loads(p.read_text(encoding="utf-8"))["PRIMARY_GATE"]
        pairs[rep] = {k: {x: g[k][x] for x in ("mean", "lcb95", "ucb95")} for k in ("delta_A", "delta_B")}
    new = [pairs[r] for r in ("rep1", "rep2", "rep3")]
    k_mean = sum(1 for v in new if v["delta_A"]["mean"] > 0 and v["delta_B"]["mean"] > 0)
    k_lcb = sum(1 for v in new if v["delta_A"]["lcb95"] > 0 and v["delta_B"]["lcb95"] > 0)
    summary = {d: {"mean_across_new": sum(v[d]["mean"] for v in new) / 3,
                   "range_across_new": [min(v[d]["mean"] for v in new), max(v[d]["mean"] for v in new)]}
               for d in ("delta_A", "delta_B")}
    result = {"id": SPEC_ID, "status": "SEALED", "utc": _now(),
              "spec_sha256": C.sha256(C.STRENGTH_DIR / f"{SPEC_ID}_SPEC.json"),
              "headline_k_of_3_both_means_positive": k_mean,
              "secondary_descriptive_k_of_3_both_lcb95_positive": k_lcb,
              "pairs": pairs, "across_new_replicates": summary,
              "note": "the secondary count is descriptive only and is not a gate (spec)"}
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"READOUT: {k_mean}/3 new replicates with both mean deltas > 0; {k_lcb}/3 with both LCB95 > 0")
    for rep, v in pairs.items():
        log(f"  {rep:8s} dA {v['delta_A']['mean']:+.3f} [{v['delta_A']['lcb95']:+.3f}, {v['delta_A']['ucb95']:+.3f}]  "
            f"dB {v['delta_B']['mean']:+.3f} [{v['delta_B']['lcb95']:+.3f}, {v['delta_B']['ucb95']:+.3f}]")


def smoke_play(manifest: Path, device: str) -> None:
    """Two throwaway episodes through the smoke replicate composite (999xxxxx seeds)."""
    genomes, identity, _ = C.resolve_poles()
    actors = C.load_ours(device, manifest_path=manifest)
    for i, (side, pole) in enumerate((("A", "A"), ("B", "B"))):
        res = C.run_episode(actors[side], pole, 99973101 + i, device, genomes, identity, context="replication smoke")
        log(f"smoke play {side}@{pole}: {res}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--phase", choices=("train", "export", "eval", "readout", "all"), default="all")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    spec = C.load_frozen_spec(SPEC_ID)
    if not args.smoke:
        for (rep, side), r in runs(spec, False).items():
            ok_block = [b for b in __import__("experiments.seed_registry", fromlist=["load"]).load()["blocks"]
                        if b["experiment_id"] == r["eid"]]
            if len(ok_block) != 1 or not (ok_block[0]["lo"] <= r["seed"] <= ok_block[0]["hi"]):
                fail(f"{r['eid']} seed {r['seed']} is not registered")
    log(f"{SPEC_ID} {'SMOKE' if args.smoke else 'RUN'} phase={args.phase}")
    if args.phase in ("train", "all"):
        phase_train(spec, args.smoke)
    manifests = {}
    if args.phase in ("export", "eval", "all"):
        manifests = phase_export(spec, args.smoke)
    if args.smoke:
        smoke_play(manifests["repsmoke"], args.device)
    if args.phase in ("eval", "all"):
        phase_eval(spec, manifests, args.smoke)
    if args.phase in ("readout", "all") and not args.smoke:
        phase_readout(spec)
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
