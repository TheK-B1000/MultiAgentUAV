r"""Evaluation-only Stage-4 re-score on dual-branch own top-50 seeds.

Pins the SAME frozen Share-Encoder / Ours-Shared / Role-only students as the
historical-seed Stage-4 diagnostic. Writes NEW OWN50_* result artifacts.
Does not retrain, re-freeze students, or touch sealed TOP50_* results.

Authorized by STAGE4_2V2_SALVAGE_AND_OWN50_REEVAL_V1.json (n=2) and the
scale's STAGE4_*_FULL_SUITE / SCHOOL suite auth for n in {4, 6}.

    python experiments/reeval_stage4_on_own_top50.py --team-size 2
    python experiments/reeval_stage4_on_own_top50.py --team-size 2 --dry-run
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
    with (here / "own50_stage4_reeval.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _env() -> dict:
    e = dict(os.environ)
    e["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
    e["PYTHONUNBUFFERED"] = "1"
    return e


def _run(argv: list[str], n: int, tag: str) -> int:
    here = SD / "dual_branch_v1" / f"matched128_{n}v{n}"
    out = here / f"own50_stage4_reeval_{tag}.out"
    err = here / f"own50_stage4_reeval_{tag}.err"
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
    own = (
        SD / "dual_branch_v1" / f"matched128_{n}v{n}"
        / f"DUAL_BRANCH_{n}V{n}_OWN_TOP50_seed_ids.json"
    )
    if not own.is_file():
        raise SystemExit(f"REFUSING: own-top50 seed list missing: {own.relative_to(ROOT)}")
    hist = SD / f"STANDARDIZED_{n}V{n}_STAGE4_SHARING_EVAL_SPEC.json"
    if not hist.is_file():
        raise SystemExit(
            f"REFUSING: historical Stage-4 eval spec missing ({hist.name}); "
            f"students must already be frozen before own50 re-eval"
        )

    _log(n, f"freezing OWN50 Stage-4 eval spec for {n}v{n}")
    spec_p = P4.own_top50_eval_spec(n)
    _log(n, f"froze {spec_p.name}")

    labs = P4.own50_labels(n)
    for arm in ARMS:
        lab = labs[arm]
        result = SD / f"{lab}_CROSSOVER_EVAL_RESULT.json"
        if result.is_file():
            _log(n, f"OWN50 Stage4 eval {arm} already sealed: {result.name}")
            continue
        argv = [
            "experiments/eval_suite_sharing_crossover.py",
            "--team-size", str(n),
            "--arm", arm,
            "--spec-tag", "STAGE4_OWN50",
            "--device", a.device,
        ]
        if a.dry_run:
            argv.append("--dry-run")
        else:
            argv.append("--resume")
        rc = _run(argv, n, arm)
        if a.dry_run:
            if rc != 0:
                raise SystemExit(f"OWN50 Stage4 dry-run {arm} exited {rc}")
            _log(n, f"OWN50 Stage4 dry-run {arm} PASS")
            continue
        if not result.is_file():
            raise SystemExit(
                f"OWN50 Stage4 eval {arm} exited {rc} without {result.name}; "
                f"see dual_branch_v1/matched128_{n}v{n}/own50_stage4_reeval_{arm}.err"
            )
        _log(n, f"OWN50 Stage4 eval {arm} DONE → {result.name}")

    done = SD / "dual_branch_v1" / f"matched128_{n}v{n}" / "OWN50_STAGE4_REEVAL_DONE.txt"
    if not a.dry_run:
        done.write_text(f"DONE {_now()}\n", encoding="utf-8")
        _log(n, f"wrote {done.name}")
    _log(n, "OWN50 Stage-4 re-eval complete (historical TOP50_* artifacts untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
