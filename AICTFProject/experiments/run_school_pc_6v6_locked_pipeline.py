"""SCHOOL_PC_6V6_LOCKED_PIPELINE — unattended orchestrator with dual tqdm.

Per-stage PPO/eval bars go to ``school_pc_6v6_<tag>.err`` (ASCII-safe for
``Get-Content -Wait``). An **overall** bar on this process's stderr tracks
weighted pipeline progress (1M A + 1M B + 200k split + 256 eval cells).

Portable handoff (models + results + seals + logs) is kept under::

    AICTFProject/6v6/

Zip that folder for your professor::

    powershell -ExecutionPolicy Bypass -File experiments/pack_6v6_handoff.ps1

Launch detached::

    powershell -ExecutionPolicy Bypass -File experiments/launch_school_pc_6v6_detached.ps1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tqdm_loop import tqdm_iter  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
HANDOFF = ROOT / "6v6"
SPEC_PATH = SD / "SCHOOL_PC_6V6_LOCKED_PIPELINE.json"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.is_file():
    PY = Path(sys.executable)

A_BASE = "artifacts/scale_6v6_specialists/pi_A_specialist_6v6/ckpts/final_pi_A_specialist_6v6.zip"
B_BASE = "artifacts/scale_6v6_specialists/pi_B_specialist_6v6/ckpts/final_pi_B_specialist_6v6.zip"
A_REPAIR = (
    "artifacts/scale_6v6_specialists/pi_A_specialist_6v6_c2_entity_repair/"
    "ckpts/final_pi_A_specialist_6v6_c2_entity_repair.zip"
)
B_REPAIR = (
    "artifacts/scale_6v6_specialists/pi_B_specialist_6v6_c2_entity_repair/"
    "ckpts/final_pi_B_specialist_6v6_c2_entity_repair.zip"
)
PI_D = (
    "artifacts/scale_6v6_specialists/pi_A_specialist_6v6_split_defend_k1_v1/"
    "ckpts/final_pi_A_specialist_6v6_split_defend_k1_v1.zip"
)

WEIGHT_REPAIR = 1_000_000
WEIGHT_SPLIT = 200_000
WEIGHT_EVAL_CELLS = 2 * 2 * 64  # 256


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _handoff_dirs() -> dict[str, Path]:
    dirs = {
        "root": HANDOFF,
        "models": HANDOFF / "models",
        "results": HANDOFF / "results",
        "seals": HANDOFF / "seals",
        "progress": HANDOFF / "progress",
        "logs": HANDOFF / "logs",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def _handoff_update_manifest(**fields: Any) -> None:
    path = HANDOFF / "MANIFEST.json"
    doc: dict[str, Any] = {}
    if path.is_file():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            doc = {}
    doc.setdefault("record", "6v6_school_handoff")
    doc["utc_updated"] = _now()
    doc.update(fields)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _handoff_copy(src: Path, dest: Path) -> None:
    if not src.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"HANDOFF copy {src.name} -> {dest.relative_to(ROOT)}", flush=True)


def _write_seal(name: str, rel: str, *, handoff_seal_name: str | None = None) -> str:
    path = ROOT / rel
    if not path.is_file():
        raise SystemExit(f"missing checkpoint for seal: {path}")
    digest = _sha256(path)
    payload = {
        "utc": _now(),
        "path": rel.replace("\\", "/"),
        "sha256": digest,
        "bytes": path.stat().st_size,
    }
    (SD / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _handoff_dirs()
    seal_name = handoff_seal_name or name
    (HANDOFF / "seals" / seal_name).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"SEAL {name} sha256={digest}", flush=True)
    return digest


def _handoff_model(rel: str, short_name: str) -> None:
    src = ROOT / rel
    dest = HANDOFF / "models" / short_name
    _handoff_copy(src, dest)
    if dest.is_file():
        _handoff_update_manifest(
            **{
                f"model_{short_name}": {
                    "handoff": f"6v6/models/{short_name}",
                    "source": rel.replace("\\", "/"),
                    "sha256": _sha256(dest),
                    "bytes": dest.stat().st_size,
                }
            }
        )


def _read_global_step(metrics_csv: Path) -> int:
    if not metrics_csv.is_file():
        return 0
    try:
        import csv

        with metrics_csv.open("r", encoding="utf-8", newline="") as fh:
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
        "utc": _now(),
        "phase": phase,
        "overall_done": overall_done,
        "overall_total": overall_total,
        "frac": (overall_done / overall_total) if overall_total else 0.0,
        "detail": detail,
    }
    text = json.dumps(payload, indent=2) + "\n"
    (SD / "school_pc_6v6_OVERALL_PROGRESS.json").write_text(text, encoding="utf-8")
    _handoff_dirs()
    (HANDOFF / "progress" / "OVERALL_PROGRESS.json").write_text(text, encoding="utf-8")
    line = (
        f"[{payload['utc']}] phase={phase} "
        f"overall={overall_done}/{overall_total} "
        f"({100.0 * payload['frac']:.2f}%) {detail}\n"
    )
    with (SD / "school_pc_6v6_OVERALL_PROGRESS.log").open("a", encoding="utf-8") as fh:
        fh.write(line)
    with (HANDOFF / "progress" / "OVERALL_PROGRESS.log").open("a", encoding="utf-8") as fh:
        fh.write(line)


def _run_preflight(stage: str) -> None:
    cmd = [str(PY), "experiments/run_school_pc_6v6_preflight.py", "--stage", stage]
    print(f"=== preflight {stage} ===", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise SystemExit(f"preflight {stage} FAIL — refusing grind")
    _handoff_copy(
        SD / "SCHOOL_PC_6V6_PREFLIGHT_RESULT.json",
        HANDOFF / "seals" / "PREFLIGHT_RESULT.json",
    )


def _run_stage(
    tag: str,
    argv: list[str],
    *,
    metrics_rel: str | None,
    weight: int,
    overall_base: int,
    overall_total: int,
    overall_bar: Any,
) -> None:
    SD.mkdir(parents=True, exist_ok=True)
    _handoff_dirs()
    log = SD / f"school_pc_6v6_{tag}.log"
    err = SD / f"school_pc_6v6_{tag}.err"
    print(f"=== {tag} ===", flush=True)
    print(" ".join(argv), flush=True)
    print(f"  stage tqdm -> {err}", flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [str(PY), *argv],
        cwd=str(ROOT),
        stdout=log.open("w", encoding="utf-8"),
        stderr=err.open("w", encoding="utf-8"),
        env=env,
    )
    metrics = ROOT / metrics_rel if metrics_rel else None
    last_report = -1
    try:
        while True:
            rc = proc.poll()
            step = _read_global_step(metrics) if metrics else 0
            step = min(max(step, 0), weight)
            overall_now = overall_base + step
            delta = overall_now - int(overall_bar.n)
            if delta > 0:
                overall_bar.update(min(delta, overall_total - int(overall_bar.n)))
            if step != last_report and (
                step - last_report >= max(1, weight // 200) or rc is not None
            ):
                _heartbeat(
                    tag,
                    min(overall_now, overall_total),
                    overall_total,
                    f"stage_step={step}/{weight}",
                )
                last_report = step
            if rc is not None:
                break
            time.sleep(15.0)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()

    if proc.returncode != 0:
        tail = err.read_text(encoding="utf-8", errors="replace")[-4000:]
        print(tail, file=sys.stderr, flush=True)
        raise SystemExit(f"{tag} failed exit={proc.returncode}")

    target = min(overall_base + weight, overall_total)
    delta = target - int(overall_bar.n)
    if delta > 0:
        overall_bar.update(delta)
    _heartbeat(tag, target, overall_total, "stage_complete")
    _handoff_copy(err, HANDOFF / "logs" / f"{tag}.err")
    _handoff_copy(log, HANDOFF / "logs" / f"{tag}.log")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-repair", action="store_true")
    ap.add_argument("--skip-split", action="store_true")
    ap.add_argument(
        "--skip-preflight",
        action="store_true",
        help="dangerous; only for resume after a verified PASS",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    SD.mkdir(parents=True, exist_ok=True)
    _handoff_dirs()
    print(f"overall tqdm -> {SD / 'school_pc_6v6_OVERALL.err'}", flush=True)
    print(f"handoff folder: {HANDOFF}", flush=True)
    print(f"started utc={_now()}", flush=True)
    _handoff_update_manifest(
        status="RUNNING",
        handoff_root="6v6",
        note="Professor: zip this folder with experiments/pack_6v6_handoff.ps1",
    )

    if args.dry_run:
        print("dry-run only; not launching", flush=True)
        return 0

    if not args.skip_preflight:
        _run_preflight("foundation")

    weights: list[tuple[str, int]] = []
    if not args.skip_repair:
        weights += [("repair_A", WEIGHT_REPAIR), ("repair_B", WEIGHT_REPAIR)]
    if not args.skip_split:
        weights.append(("split_k1", WEIGHT_SPLIT))
    weights.append(("crossover_exploratory", WEIGHT_EVAL_CELLS))
    overall_total = sum(w for _, w in weights)

    overall_bar = tqdm_iter(
        range(overall_total),
        desc="school_pc_6v6_OVERALL",
        total=overall_total,
        unit="unit",
        leave=True,
    )
    overall_bar.n = 0
    overall_bar.refresh()

    overall_base = 0
    a_sha = ""

    try:
        if not args.skip_repair:
            _run_stage(
                "repair_A",
                [
                    "experiments/train_specialist_scale.py",
                    "--team-size", "6", "--policy", "A", "--seed", "22500001",
                    "--device", "cuda", "--total-timesteps", "1000000",
                    "--entity-repair-enabled", "--entity-hidden-dim", "32",
                    "--load-path", A_BASE,
                    "--run-label-suffix", "_c2_entity_repair",
                ],
                metrics_rel=(
                    "artifacts/scale_6v6_specialists/pi_A_specialist_6v6_c2_entity_repair/"
                    "metrics.csv"
                ),
                weight=WEIGHT_REPAIR,
                overall_base=overall_base,
                overall_total=overall_total,
                overall_bar=overall_bar,
            )
            _write_seal(
                "SCHOOL_PC_6V6_PI_A_REPAIR_SEAL.json",
                A_REPAIR,
                handoff_seal_name="PI_A_REPAIR_SEAL.json",
            )
            _handoff_model(A_REPAIR, "final_pi_A_repair.zip")
            overall_base += WEIGHT_REPAIR

            _run_stage(
                "repair_B",
                [
                    "experiments/train_specialist_scale.py",
                    "--team-size", "6", "--policy", "B", "--seed", "22500002",
                    "--device", "cuda", "--total-timesteps", "1000000",
                    "--entity-repair-enabled", "--entity-hidden-dim", "32",
                    "--load-path", B_BASE,
                    "--run-label-suffix", "_c2_entity_repair",
                ],
                metrics_rel=(
                    "artifacts/scale_6v6_specialists/pi_B_specialist_6v6_c2_entity_repair/"
                    "metrics.csv"
                ),
                weight=WEIGHT_REPAIR,
                overall_base=overall_base,
                overall_total=overall_total,
                overall_bar=overall_bar,
            )
            _write_seal(
                "SCHOOL_PC_6V6_PI_B_REPAIR_SEAL.json",
                B_REPAIR,
                handoff_seal_name="PI_B_REPAIR_SEAL.json",
            )
            _handoff_model(B_REPAIR, "final_pi_B_repair.zip")
            overall_base += WEIGHT_REPAIR
        else:
            a_sha = _write_seal(
                "SCHOOL_PC_6V6_PI_A_REPAIR_SEAL.json",
                A_REPAIR,
                handoff_seal_name="PI_A_REPAIR_SEAL.json",
            )
            _write_seal(
                "SCHOOL_PC_6V6_PI_B_REPAIR_SEAL.json",
                B_REPAIR,
                handoff_seal_name="PI_B_REPAIR_SEAL.json",
            )
            _handoff_model(A_REPAIR, "final_pi_A_repair.zip")
            _handoff_model(B_REPAIR, "final_pi_B_repair.zip")

        a_sha = a_sha or _sha256(ROOT / A_REPAIR)

        if not args.skip_preflight:
            _run_preflight("split")

        if not args.skip_split:
            _run_stage(
                "split_k1",
                [
                    "experiments/train_specialist_scale.py",
                    "--team-size", "6", "--policy", "A", "--seed", "22600001",
                    "--device", "cuda", "--total-timesteps", "200000",
                    "--entity-repair-enabled", "--entity-hidden-dim", "32",
                    "--role-conditioning-enabled", "--role-fixed-for-episode",
                    "--role-k-defend", "1",
                    "--split-attack-defend-enabled",
                    "--split-attack-defend-frozen-ckpt", A_REPAIR,
                    "--split-attack-defend-frozen-ckpt-sha256", a_sha,
                    "--load-path", A_REPAIR,
                    "--defend-teacher-lambda", "0.1",
                    "--defend-teacher-lambda-end", "0.0",
                    "--defend-teacher-decay-start-step", "50000",
                    "--defend-teacher-decay-end-step", "150000",
                    "--defend-teacher-cadence", "4",
                    "--run-label-suffix", "_split_defend_k1_v1",
                ],
                metrics_rel=(
                    "artifacts/scale_6v6_specialists/pi_A_specialist_6v6_split_defend_k1_v1/"
                    "metrics.csv"
                ),
                weight=WEIGHT_SPLIT,
                overall_base=overall_base,
                overall_total=overall_total,
                overall_bar=overall_bar,
            )
            _write_seal(
                "SCHOOL_PC_6V6_SPLIT_K1_SEAL.json",
                PI_D,
                handoff_seal_name="SPLIT_K1_SEAL.json",
            )
            _handoff_model(PI_D, "final_pi_D_split.zip")
            overall_base += WEIGHT_SPLIT
        else:
            _write_seal(
                "SCHOOL_PC_6V6_SPLIT_K1_SEAL.json",
                PI_D,
                handoff_seal_name="SPLIT_K1_SEAL.json",
            )
            _handoff_model(PI_D, "final_pi_D_split.zip")

        _run_stage(
            "crossover_exploratory",
            [
                "experiments/eval_specialist_crossover_scaled.py",
                "--team-size", "6",
                "--spec", str(SPEC_PATH.relative_to(ROOT)).replace("\\", "/"),
                "--pi-a-path", PI_D,
                "--pi-b-path", B_REPAIR,
                "--frozen-attack-path", A_REPAIR,
                "--frozen-attack-path-sha256", a_sha,
                "--role-fixed-for-episode", "--role-k-defend", "1",
                "--seed-base", "22700001", "--n-seeds", "64",
                "--label", "EXPLORATORY_6V6_SPLIT_K1",
                "--device", "cuda",
            ],
            metrics_rel=None,
            weight=WEIGHT_EVAL_CELLS,
            overall_base=overall_base,
            overall_total=overall_total,
            overall_bar=overall_bar,
        )
        _handoff_copy(
            SD / "EXPLORATORY_6V6_SPLIT_K1_SPECIALIST_CROSSOVER_EVAL_RESULT.json",
            HANDOFF / "results" / "exploratory_crossover_RESULT.json",
        )
        _handoff_copy(
            SD / "exploratory_6v6_split_k1_specialist_crossover_eval_rows.csv",
            HANDOFF / "results" / "exploratory_crossover_rows.csv",
        )
    finally:
        try:
            overall_bar.close()
        except Exception:
            pass

    _handoff_copy(SD / "school_pc_6v6_OVERALL.err", HANDOFF / "logs" / "OVERALL.err")
    _handoff_copy(SD / "school_pc_6v6_OVERALL.log", HANDOFF / "logs" / "OVERALL.log")

    done = {
        "utc": _now(),
        "status": "FINISHED_THROUGH_EXPLORATORY_EVAL",
        "overall_total_units": overall_total,
        "handoff": "6v6",
        "pack_command": (
            "powershell -ExecutionPolicy Bypass -File experiments/pack_6v6_handoff.ps1"
        ),
        "checkpoints": {"A_repair": A_REPAIR, "B_repair": B_REPAIR, "pi_D": PI_D},
        "note": (
            "If exploratory PASS, run confirmatory manually (seed-base 22800001, n=128). "
            "Zip 6v6/ for the professor."
        ),
    }
    (SD / "SCHOOL_PC_6V6_PIPELINE_DONE.json").write_text(
        json.dumps(done, indent=2) + "\n", encoding="utf-8"
    )
    (HANDOFF / "seals" / "PIPELINE_DONE.json").write_text(
        json.dumps(done, indent=2) + "\n", encoding="utf-8"
    )
    _handoff_update_manifest(status="FINISHED_THROUGH_EXPLORATORY_EVAL", done=done)
    _heartbeat("DONE", overall_total, overall_total, "exploratory eval finished")
    print(json.dumps(done, indent=2), flush=True)
    print(f"\nSEND FOLDER: {HANDOFF}", flush=True)
    print(
        "PACK: powershell -ExecutionPolicy Bypass -File experiments/pack_6v6_handoff.ps1",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
