r"""Evaluation-only Stage-4 re-score on the full unselected matched-128 block.

Pins the SAME frozen Share-Encoder / Fully Shared+(z+r) / Role-only students as
TOP50/OWN50. Writes NEW M128_* result artifacts. Does not retrain, re-freeze
students, or touch sealed TOP50_* / OWN50_* results.

Authorized by STAGE4_2V2_MATCHED128_REEVAL_V1.json (n=2).

    python experiments/reeval_stage4_on_matched128.py --team-size 2
    python experiments/reeval_stage4_on_matched128.py --team-size 2 --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import prepare_stage4_baselines as P4  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
ARMS = ("share_encoder", "fully_shared", "role_only")
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(n: int, msg: str) -> None:
    here = SD / "dual_branch_v1" / f"matched128_{n}v{n}"
    here.mkdir(parents=True, exist_ok=True)
    line = f"{_now()} {msg}"
    print(line, flush=True)
    with (here / "m128_stage4_reeval.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _env() -> dict:
    e = dict(os.environ)
    e["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
    e["PYTHONUNBUFFERED"] = "1"
    return e


def _run(argv: list[str], n: int, tag: str) -> int:
    here = SD / "dual_branch_v1" / f"matched128_{n}v{n}"
    out = here / f"m128_stage4_reeval_{tag}.out"
    err = here / f"m128_stage4_reeval_{tag}.err"
    _log(n, f"exec: {' '.join(argv)}")
    with out.open("w", encoding="utf-8") as fo, err.open("w", encoding="utf-8") as fe:
        p = subprocess.run([PY, *argv], cwd=str(ROOT), env=_env(), stdout=fo, stderr=fe)
    return int(p.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team-size", type=int, required=True, choices=(2, 4, 6))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    n = int(a.team_size)
    auth = SD / f"STAGE4_{n}V{n}_MATCHED128_REEVAL_V1.json"
    if not auth.is_file():
        raise SystemExit(f"REFUSING: auth missing: {auth.name}")
    hist = SD / f"STANDARDIZED_{n}V{n}_STAGE4_SHARING_EVAL_SPEC.json"
    if not hist.is_file():
        raise SystemExit(
            f"REFUSING: historical Stage-4 eval spec missing ({hist.name}); "
            f"students must already be frozen before matched-128 re-eval"
        )
    teacher = SD / f"POSTHOC_MATCHED128_{n}V{n}_DUAL_BRANCH_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
    if not teacher.is_file():
        raise SystemExit(f"REFUSING: teacher matched-128 RESULT missing: {teacher.name}")

    _log(n, f"freezing M128 Stage-4 eval spec for {n}v{n}")
    spec_p = P4.matched128_eval_spec(n)
    _log(n, f"froze {spec_p.name}")

    labs = P4.matched128_labels(n)
    for arm in ARMS:
        lab = labs[arm]
        result = SD / f"{lab}_CROSSOVER_EVAL_RESULT.json"
        if result.is_file():
            _log(n, f"M128 Stage4 eval {arm} already sealed: {result.name}")
            continue
        argv = [
            "experiments/eval_suite_sharing_crossover.py",
            "--team-size", str(n),
            "--arm", arm,
            "--spec-tag", "STAGE4_M128",
            "--device", a.device,
        ]
        if a.dry_run:
            argv.append("--dry-run")
        else:
            argv.append("--resume")
        rc = _run(argv, n, arm)
        if a.dry_run:
            if rc != 0:
                raise SystemExit(f"M128 Stage4 dry-run {arm} exited {rc}")
            _log(n, f"M128 Stage4 dry-run {arm} PASS")
            continue
        if not result.is_file():
            raise SystemExit(
                f"M128 Stage4 eval {arm} exited {rc} without {result.name}; "
                f"see dual_branch_v1/matched128_{n}v{n}/m128_stage4_reeval_{arm}.err"
            )
        _log(n, f"M128 Stage4 eval {arm} DONE → {result.name}")

    done = SD / "dual_branch_v1" / f"matched128_{n}v{n}" / "M128_STAGE4_REEVAL_DONE.txt"
    if not a.dry_run:
        done.write_text(f"DONE {_now()}\n", encoding="utf-8")
        _log(n, f"wrote {done.name}")
    _log(n, "M128 Stage-4 re-eval complete (TOP50_* and OWN50_* artifacts untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
