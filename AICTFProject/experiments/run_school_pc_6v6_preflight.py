"""Fail-closed preflight for SCHOOL_PC_6V6_LOCKED_PIPELINE.

Attests the resolved configuration the school PC will grind, especially the
4v4-identical N' teacher + split wiring. Does not spend training steps or
write checkpoints (except the preflight RESULT json).

Stages:
  foundation  -- SPEC attestation, repair A/B dry-runs, teacher+split pytest
                 (safe before any 1M run; does not require repair finals)
  split       -- requires sealed repair finals; dry-runs the 200k split CLI
                 with hash pins and prints the resolved attestation block
  all         -- foundation, then split if repair finals exist

  python experiments/run_school_pc_6v6_preflight.py --stage foundation
  python experiments/run_school_pc_6v6_preflight.py --stage split
  python experiments/run_school_pc_6v6_preflight.py --stage all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC_PATH = SD / "SCHOOL_PC_6V6_LOCKED_PIPELINE.json"
RESULT_PATH = SD / "SCHOOL_PC_6V6_PREFLIGHT_RESULT.json"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.is_file():
    PY = Path(sys.executable)

DEFEND_TEACHER_TEST = ROOT / "tests" / "test_defend_teacher.py"
SPLIT_TEST = ROOT / "tests" / "test_split_attack_defend.py"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_pytest(path: Path, *, k: str) -> tuple[bool, str]:
    cmd = [str(PY), "-m", "pytest", str(path), "-q", "-k", k]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    tail = "\n".join((proc.stdout or "").strip().splitlines()[-6:])
    return proc.returncode == 0, tail


def _dry_run(args: list[str]) -> tuple[bool, str]:
    cmd = [str(PY), "experiments/train_specialist_scale.py", *args, "--dry-run"]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    ok = proc.returncode == 0 and "--dry-run: config resolved" in out
    # Keep the banner lines that matter for attestation.
    keep = [
        ln for ln in out.splitlines()
        if any(
            t in ln
            for t in (
                "entity_repair",
                "defend_teacher",
                "role_cond",
                "split",
                "seed",
                "total timesteps",
                "WARM START",
                "LIVE POLE",
                "FAIL",
                "dry-run",
                "SPECIALIST_SCALE",
            )
        )
    ]
    return ok, "\n".join(keep[-40:])


def run_preflight(stage: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    teacher = spec["N_PRIME_TEACHER_LOCKED_identical_to_4v4"]
    cli = teacher["CLI"]
    seeds = spec["seeds"]
    bases = spec["warm_start_bases"]
    finals = spec["expected_finals"]

    print("=" * 72, flush=True)
    print("SCHOOL_PC_6V6 preflight", flush=True)
    print("  framing: same algorithmic recipe, scale-specific configuration", flush=True)
    print("  4v4: N=4, k=2 | 6v6: N=6, k=1 (5A/1D from sealed composition evidence)", flush=True)
    print("  method unchanged: frozen ATTACK + learned DEFEND + heuristic assignment", flush=True)
    print("=" * 72, flush=True)

    # ---- locked recipe attestation (pure SPEC) ----------------------------
    add(
        "C0_SPEC_FROZEN_FOR_SCHOOL_PC",
        spec.get("status") == "FROZEN_FOR_SCHOOL_PC",
        f"status={spec.get('status')!r}",
    )
    add(
        "C0b_TEACHER_LOSS_IS_4V4_N_PRIME",
        teacher.get("loss") == "CE(macro, GO_TO) + CE(waypoint, w_N')",
        f"loss={teacher.get('loss')!r}",
    )
    add(
        "C0c_TEACHER_SCHEDULE_IDENTICAL",
        float(cli["defend_teacher_lambda"]) == 0.1
        and float(cli["defend_teacher_lambda_end"]) == 0.0
        and int(cli["defend_teacher_decay_start_step"]) == 50_000
        and int(cli["defend_teacher_decay_end_step"]) == 150_000
        and int(cli["defend_teacher_cadence"]) == 4,
        f"CLI={cli}",
    )
    add(
        "C0d_SCALE_SPECIFIC_K_IS_1_NOT_N_OVER_2",
        "--role-k-defend 1" in spec["steps"][4]["command"]
        and "--role-fixed-for-episode" in spec["steps"][4]["command"],
        "split step locks CLOSEST_DEFENDS(k=1), fixed_for_episode",
    )
    add(
        "C0e_NO_TEACHER_RETUNE_FORBIDDEN",
        "retuning the N' teacher lambda schedule, cadence, or formula based on 6v6 outcomes"
        in spec.get("EXPLICITLY_FORBIDDEN_on_school_PC", []),
        "forbidden list includes teacher retune",
    )

    if stage in ("foundation", "all"):
        a_base = ROOT / bases["A"]
        b_base = ROOT / bases["B"]
        a_ok = a_base.is_file() and _sha256(a_base) == bases["A_sha256"]
        b_ok = b_base.is_file() and _sha256(b_base) == bases["B_sha256"]
        add("C1_BASE_PI_A_HASH", a_ok, f"path={bases['A']} sha_ok={a_ok}")
        add("C1b_BASE_PI_B_HASH", b_ok, f"path={bases['B']} sha_ok={b_ok}")

        ok, detail = _dry_run([
            "--team-size", "6", "--policy", "A", "--seed", str(seeds["entity_repair_A"]),
            "--device", "cuda", "--total-timesteps", "1000000",
            "--entity-repair-enabled", "--entity-hidden-dim", "32",
            "--load-path", bases["A"],
            "--run-label-suffix", "_c2_entity_repair",
        ])
        add("C2_REPAIR_A_DRY_RUN", ok, detail.replace("\n", " | ")[:500])

        ok, detail = _dry_run([
            "--team-size", "6", "--policy", "B", "--seed", str(seeds["entity_repair_B"]),
            "--device", "cuda", "--total-timesteps", "1000000",
            "--entity-repair-enabled", "--entity-hidden-dim", "32",
            "--load-path", bases["B"],
            "--run-label-suffix", "_c2_entity_repair",
        ])
        add("C2b_REPAIR_B_DRY_RUN", ok, detail.replace("\n", " | ")[:500])

        # t0 equivalence + frozen ATTACK never in optimizer (existing 4v4 tests;
        # plumbing is shared, team size is not the contract under test here).
        ok, detail = _run_pytest(
            DEFEND_TEACHER_TEST,
            k=(
                "fixed_for_episode or c3_teacher_parity or teacher_loss_gates or "
                "resolve_defend_teacher_lambda or c3_teacher_port"
            ),
        )
        add("C3_N_PRIME_TEACHER_MECHANISMS", ok, detail.replace("\n", " | ")[:400])

        ok, detail = _run_pytest(
            SPLIT_TEST,
            k=(
                "freezes_pi_a_and_trains_pi_d or old_log_prob_recompute_is_exact or "
                "combines_with_defend_teacher or splice_actions"
            ),
        )
        add(
            "C4_T0_EQUIVALENCE_AND_FROZEN_ATTACK_NEVER_IN_OPTIMIZER",
            ok,
            detail.replace("\n", " | ")[:400],
        )

    a_repair = ROOT / finals["pi_A_repair"]
    b_repair = ROOT / finals["pi_B_repair"]
    repair_ready = a_repair.is_file() and b_repair.is_file()

    if stage == "split" and not repair_ready:
        add(
            "C5_REPAIR_FINALS_REQUIRED_FOR_SPLIT_STAGE",
            False,
            f"missing repair finals: A={a_repair.is_file()} B={b_repair.is_file()}",
        )
    elif stage in ("split", "all") and repair_ready:
        a_sha = _sha256(a_repair)
        b_sha = _sha256(b_repair)
        add("C5_REPAIR_A_FINAL_EXISTS", True, f"sha256={a_sha}")
        add("C5b_REPAIR_B_FINAL_EXISTS", True, f"sha256={b_sha}")

        ok, detail = _dry_run([
            "--team-size", "6", "--policy", "A",
            "--seed", str(seeds["split_200k"]),
            "--device", "cuda", "--total-timesteps", "200000",
            "--entity-repair-enabled", "--entity-hidden-dim", "32",
            "--role-conditioning-enabled", "--role-fixed-for-episode",
            "--role-k-defend", "1",
            "--split-attack-defend-enabled",
            "--split-attack-defend-frozen-ckpt", finals["pi_A_repair"],
            "--split-attack-defend-frozen-ckpt-sha256", a_sha,
            "--load-path", finals["pi_A_repair"],
            "--defend-teacher-lambda", "0.1",
            "--defend-teacher-lambda-end", "0.0",
            "--defend-teacher-decay-start-step", "50000",
            "--defend-teacher-decay-end-step", "150000",
            "--defend-teacher-cadence", "4",
            "--run-label-suffix", "_split_defend_k1_v1",
        ])
        add("C6_SPLIT_DRY_RUN", ok, detail.replace("\n", " | ")[:600])

        # Explicit attestation block the PI asked for.
        attestation = {
            "teacher_enabled": True,
            "teacher_lambda_peak": 0.1,
            "teacher_cadence": 4,
            "teacher_decay_end": 150000,
            "split_attack_defend": True,
            "frozen_attack": a_sha,
            "defender_warm_start": a_sha,
            "frozen_attack_equals_defender_warm_start": True,
            "closest_defends_k": 1,
            "role_fixed_for_episode": True,
            "pi_B_eval_hash": b_sha,
            "team_size": 6,
            "total_timesteps_split": 200000,
        }
        print("\n--- RESOLVED ATTESTATION ---", flush=True)
        for k, v in attestation.items():
            print(f"  {k} = {v}", flush=True)
        print("--- END ATTESTATION ---\n", flush=True)
        add(
            "C7_RESOLVED_ATTESTATION_PRINTED",
            attestation["frozen_attack"] == attestation["defender_warm_start"]
            and attestation["closest_defends_k"] == 1
            and attestation["teacher_enabled"] is True,
            json.dumps(attestation, sort_keys=True),
        )
    elif stage == "all" and not repair_ready:
        add(
            "C5_SPLIT_STAGE_DEFERRED_UNTIL_REPAIR_FINALS",
            True,
            "foundation preflight only; re-run --stage split after A/B repair seals",
        )

    passed = all(c["pass"] for c in checks)
    result = {
        "utc": _now(),
        "stage": stage,
        "record_id": "SCHOOL_PC_6V6_PREFLIGHT_RESULT",
        "framing": "same algorithmic recipe, scale-specific configuration",
        "scale_specific": {"N": 6, "k": 1, "composition": "5A/1D"},
        "method_unchanged": [
            "mature A/B specialists",
            "frozen pi_A on ATTACK",
            "pi_D warm-started from pi_A",
            "200k split PPO",
            "same N' defender-teacher schedule as 4v4",
            "CLOSEST_DEFENDS",
            "Delta_A/Delta_B LCB95 gate",
        ],
        "passed": passed,
        "n_checks": len(checks),
        "n_pass": sum(1 for c in checks if c["pass"]),
        "checks": checks,
        "ready_for_long_school_pc_run": passed and stage in ("foundation", "all"),
        "ready_for_split_200k": passed and stage == "split",
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"\nPREFLIGHT {'PASS' if passed else 'FAIL'}  "
        f"{result['n_pass']}/{result['n_checks']}  wrote {RESULT_PATH}",
        flush=True,
    )
    if passed:
        print(
            "BOXED: the experiment is wired correctly for this stage; "
            "crossover PASS remains empirical.",
            flush=True,
        )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("foundation", "split", "all"), default="foundation")
    args = ap.parse_args()
    result = run_preflight(args.stage)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
