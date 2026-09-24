"""DEFEND_ATTACK_SPLIT_POLICY_A_V1_SPEC.json.

Contracts-only runner for CONTRACTS_before_training (C0-C13). Spends no
seed, writes no checkpoint (C13). Training and evaluation reuse the
existing generic launchers (experiments/train_specialist_scale.py,
experiments/eval_specialist_crossover_scaled.py) once these contracts pass
and the training/evaluation seeds are reserved via seed_registry -- this
script does not itself launch either.

C2-C8 (fixed-for-episode role hold, N' teacher parity, teacher loss
gating, lambda schedule, warm-start t0-equivalence, fresh optimizer) are
REUSED UNCHANGED from DEFEND_TEACHER_ROLE_CONDITIONING_A_V1 -- this script
re-runs that spec's own test file to confirm nothing regressed, rather
than re-implementing those checks a second time. C9-C12 (the properties
genuinely new to this spec: executed-action provenance, DEFEND-gated main
PPO loss, frozen-pi_A immutability, structural absence when disabled) are
covered by tests/test_split_attack_defend.py, which this script also runs
and maps into the named contracts below.

  contracts   The only stage this script implements: C0-C13 below.
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

STEM = "DEFEND_ATTACK_SPLIT_POLICY_A_V1"
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC_PATH = SD / f"{STEM}_SPEC.json"
CONTRACT_PATH = SD / f"{STEM}_CONTRACT_RESULT.json"

PI_A_CKPT = ROOT / (
    "artifacts/scale_4v4_specialists/pi_A_specialist_4v4_b3_entity_repair/"
    "ckpts/final_pi_A_specialist_4v4_b3_entity_repair.zip"
)
PI_A_SHA256 = "94dde69d091a79344db3390d5464dbb4bcf51677a175df93969ab252b2e0f478"

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


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _run_pytest(path: Path, *, k: str | None = None) -> tuple[bool, str]:
    """Run a pytest file (optionally filtered by -k) in-process's own venv
    python, returning (passed, summary_line). Never spends a seed or
    touches a checkpoint -- these are all cpu-only unit/integration tests."""
    cmd = [sys.executable, "-m", "pytest", str(path), "-q"]
    if k:
        cmd += ["-k", k]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    tail = "\n".join(proc.stdout.strip().splitlines()[-5:])
    return proc.returncode == 0, tail


def run_contracts() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    # ---- C0: spec frozen --------------------------------------------------
    spec = _load_json(SPEC_PATH)
    add(
        "C0_SPEC_FROZEN",
        spec.get("status") == "FROZEN_BEFORE_ANY_TRAINING",
        f"spec status = {spec.get('status')!r}",
    )

    # ---- C1: pi_A checkpoint pin (used for BOTH frozen copy and pi_D warm-start)
    ckpt_ok = PI_A_CKPT.is_file()
    ckpt_sha = _sha256(PI_A_CKPT) if ckpt_ok else ""
    add(
        "C1_PI_A_CHECKPOINT_EQUALS_PIN",
        ckpt_ok and ckpt_sha == PI_A_SHA256,
        f"exists={ckpt_ok} sha256_matches_pin={ckpt_sha == PI_A_SHA256} (used for pi_A_frozen AND pi_D's warm-start)",
    )
    supersedes = spec.get("SUPERSEDES", {})
    add(
        "C1b_SUPERSEDES_DEFEND_TEACHER_RECORD_PRESENT",
        supersedes.get("record") == "DEFEND_TEACHER_ROLE_CONDITIONING_A_V1_SPEC.json"
        and bool(supersedes.get("exception_fired")),
        f"SUPERSEDES.record={supersedes.get('record')!r} exception_fired_present={bool(supersedes.get('exception_fired'))}",
    )

    # ---- C2-C8: reused unchanged from DEFEND_TEACHER_ROLE_CONDITIONING_A_V1 --
    ok, detail = _run_pytest(
        DEFEND_TEACHER_TEST,
        k="fixed_for_episode or c3_teacher_parity or teacher_loss_gates or "
          "resolve_defend_teacher_lambda or c3_teacher_port",
    )
    add(
        "C2_C8_DEFEND_TEACHER_MECHANISMS_REUSED_UNCHANGED",
        ok,
        f"re-ran DEFEND_TEACHER_ROLE_CONDITIONING_A_V1's own tests (fixed_for_episode role hold, "
        f"N' teacher parity, teacher loss gating, lambda schedule) for regressions; {detail}",
    )

    # ---- C9: pure splice/gating math (executed-action provenance) -----------
    ok, detail = _run_pytest(
        SPLIT_TEST, k="splice_actions or role_broadcast_mask or defend_gated_sum",
    )
    add(
        "C9_EXECUTED_ACTION_PROVENANCE_AND_DEFEND_GATING_MATH",
        ok,
        f"ATTACK slots from frozen pi_A, DEFEND slots from pi_D, ATTACK-slot values excluded "
        f"entirely from the DEFEND-gated scalar; {detail}",
    )

    # ---- C10: frozen pi_A immutability + pi_D nonzero gradient + exact recompute
    ok, detail = _run_pytest(
        SPLIT_TEST,
        k="freezes_pi_a_and_trains_pi_d or old_log_prob_recompute_is_exact",
    )
    add(
        "C10_FROZEN_PI_A_IMMUTABLE_PI_D_TRAINS_RATIO_EXACT_AT_T0",
        ok,
        f"frozen pi_A parameters/gradients/optimizer-membership/content-hash unchanged after a "
        f"real collect+update cycle; pi_D receives nonzero gradient; DEFEND-gated old/new "
        f"log-prob recompute is exact before any parameter change; {detail}",
    )

    # ---- C11: combined with the DEFEND-teacher runner (the real training config)
    ok, detail = _run_pytest(SPLIT_TEST, k="combines_with_defend_teacher")
    add(
        "C11_COMBINES_WITH_DEFEND_TEACHER_RUNNER",
        ok,
        f"split-policy PPO functions correctly with the N'-teacher runner attached "
        f"(lambda>0, the real training configuration); {detail}",
    )

    # ---- C12: mutual exclusion + structural absence when disabled -----------
    ok, detail = _run_pytest(
        SPLIT_TEST,
        k="isolation_split_attack_defend or maybe_attach_split_attack_defend or "
          "other_aux_losses_reject_split_attack_defend or non_split_path_unchanged",
    )
    add(
        "C12_MUTUAL_EXCLUSION_AND_STRUCTURAL_ABSENCE",
        ok,
        f"guarded both directions against SAPPO/EXP2/sibling/role_pres/getflag; "
        f"split_attack_defend_enabled=False reproduces the non-split path with no new buffer "
        f"fields; {detail}",
    )

    # ---- C13: contracts consume no training step, write no checkpoint -------
    src = Path(__file__).read_text(encoding="utf-8")
    forbidden = [".le" + "arn(", "optimizer" + ".step(", "torch.sa" + "ve("]
    hits = [t for t in forbidden if t in src]
    add(
        "C13_CONTRACTS_ARE_READ_ONLY",
        not hits,
        "no training/checkpoint-writing call in this script's own source"
        if not hits else f"found {hits}",
    )

    decision = "CONTRACTS_PASS" if all(c["pass"] for c in checks) else "CONTRACTS_FAIL"
    result = {
        "record_id": f"{STEM}_CONTRACT_RESULT",
        "implements": SPEC_PATH.name,
        "utc": _now(),
        "DECISION": decision,
        "spec_sha256": _sha256(SPEC_PATH),
        "script_sha256": _sha256(Path(__file__)),
        "checks": checks,
    }
    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"\n  DEFEND_ATTACK_SPLIT_POLICY_A CONTRACTS: {decision}  "
        f"({sum(not c['pass'] for c in checks)}/{len(checks)} failed)",
        flush=True,
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("contracts",), default="contracts")
    ap.parse_args()
    result = run_contracts()
    return 0 if result["DECISION"] == "CONTRACTS_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
