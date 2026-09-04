"""Authorize and optionally launch SAC-RFT after sealed RSCFT FAIL.

Does NOT touch the in-flight RSCFT EVAL. Writes SAC_RFT_ACTIVATION.json only when the
sealed RSCFT outcome authorizes the successor, then can start CONTROL (and optionally
chain TREATMENT).

Run:
  python experiments/launch_sac_rft_after_rscft_fail.py --check
  python experiments/launch_sac_rft_after_rscft_fail.py --authorize
  python experiments/launch_sac_rft_after_rscft_fail.py --wait --authorize --launch-control
  python experiments/launch_sac_rft_after_rscft_fail.py --wait --authorize --launch-both
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
RSCFT_RESULT = SD / "RSCFT_EVAL_RESULT.json"
RSCFT_INTEGRITY_REQ = SD / "RSCFT_EVAL_INTEGRITY_REQUIRED.json"
RSCFT_INTEGRITY = SD / "RSCFT_EVAL_INTEGRITY.json"
EARLY_OVERRIDE = SD / "SAC_RFT_EARLY_ACTIVATION_OVERRIDE.json"
AUTH = SD / "SAC_RFT_ACTIVATION.json"
SPEC = SD / "SAC_RFT_SPEC.json"
_VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON = str(_VENV_PY) if _VENV_PY.is_file() else sys.executable


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def assess() -> dict:
    """Return authorization assessment without writing anything."""
    if RSCFT_RESULT.is_file():
        data = json.loads(RSCFT_RESULT.read_text(encoding="utf-8"))
        gate = data.get("PRIMARY_GATE", {})
        treat = gate.get("TREATMENT_retention", {})
        control = gate.get("CONTROL_causal_only", {})
        treat_pass = bool(treat.get("passes"))
        if treat_pass:
            return {
                "authorized": False,
                "reason": "RSCFT TREATMENT primary gate PASS -- take SCALING PASS branch, "
                          "do not launch SAC-RFT",
                "path": "RSCFT_EVAL_RESULT",
                "treatment_passes": True,
                "control_passes": bool(control.get("passes")),
                "reading": data.get("READING"),
            }
        return {
            "authorized": True,
            "reason": "RSCFT_EVAL_RESULT present and TREATMENT primary gate FAIL",
            "path": "RSCFT_EVAL_RESULT",
            "treatment_passes": False,
            "control_passes": bool(control.get("passes")),
            "reading": data.get("READING"),
            "claim_to_write": (
                "EMA temporal consistency reduced unrestricted actor drift, but was "
                "insufficient to preserve strategy-specific crossover"),
        }

    if RSCFT_INTEGRITY_REQ.is_file():
        if not RSCFT_INTEGRITY.is_file():
            if EARLY_OVERRIDE.is_file():
                ov = json.loads(EARLY_OVERRIDE.read_text(encoding="utf-8"))
                if ov.get("status") != "FROZEN":
                    return {
                        "authorized": False,
                        "reason": "early-activation override present but not FROZEN",
                        "path": "EARLY_OVERRIDE_INVALID",
                    }
                flag = json.loads(RSCFT_INTEGRITY_REQ.read_text(encoding="utf-8"))
                return {
                    "authorized": True,
                    "kind": "EARLY_COMPUTE_OVERRIDE",
                    "scientific_clearance": False,
                    "rscft_audit_still_mandatory": True,
                    "reason": ("PI early-compute override: sealed EVAL finished and wrote "
                               "INTEGRITY_REQUIRED; GPU may start SAC-RFT. This is NOT "
                               "RSCFT scientific clearance. Integrity audit remains mandatory."),
                    "path": "EARLY_COMPUTE_OVERRIDE",
                    "triggered_by": flag.get("triggered_by"),
                    "claim_to_write": None,
                    "override_file": EARLY_OVERRIDE.name,
                }
            return {
                "authorized": False,
                "reason": "RSCFT_EVAL_INTEGRITY_REQUIRED.json present but integrity audit "
                          "not yet written -- wait for GENUINE_* audit before activating",
                "path": "INTEGRITY_REQUIRED_PENDING_AUDIT",
            }
        audit = json.loads(RSCFT_INTEGRITY.read_text(encoding="utf-8"))
        verdict = str(audit.get("VERDICT", ""))
        if not verdict.startswith("GENUINE"):
            return {
                "authorized": False,
                "reason": f"integrity audit VERDICT={verdict!r} is not GENUINE_* -- "
                          "do not activate SAC-RFT on a suspected evaluator defect",
                "path": "INTEGRITY_NOT_GENUINE",
                "integrity_verdict": verdict,
            }
        # Genuine reversal / tie means no PASS claim; successor is authorized.
        return {
            "authorized": True,
            "reason": "integrity audit confirms GENUINE rows with tie/reversal; "
                      "no RSCFT PASS claim exists",
            "path": "INTEGRITY_GENUINE",
            "integrity_verdict": verdict,
            "triggered_by": audit.get("triggered_by"),
            "claim_to_write": (
                "EMA temporal consistency reduced unrestricted actor drift, but was "
                "insufficient to preserve strategy-specific crossover"),
        }

    return {
        "authorized": False,
        "reason": "RSCFT sealed EVAL not finished (no RESULT and no INTEGRITY_REQUIRED)",
        "path": "WAITING",
    }


def write_activation(assessment: dict) -> Path:
    if not assessment.get("authorized"):
        raise SystemExit(f"REFUSING to authorize: {assessment.get('reason')}")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    payload = {
        "record": "SAC_RFT_ACTIVATION",
        "status": "AUTHORIZED",
        "utc": _now(),
        "implements": "SAC_RFT_SPEC.json",
        "authorized_by": assessment,
        "kind": assessment.get("kind", "SCIENTIFIC"),
        "scientific_clearance": bool(assessment.get("scientific_clearance", True)),
        "rscft_audit_still_mandatory": bool(
            assessment.get("rscft_audit_still_mandatory", False)),
        "rscft_fail_claim": assessment.get("claim_to_write"),
        "next": [
            "python experiments/sac_rft_preflight.py --device cuda",
            "python experiments/run_sac_rft_production.py --arm control",
            "python experiments/run_sac_rft_production.py --arm treatment",
        ],
        "spec_status_at_authorization": spec.get("status"),
    }
    AUTH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Flip spec status so production accepts ACTIVATED
    if spec.get("status") == "FROZEN_PENDING_RSCFT_FAIL_ACTIVATION":
        spec["status"] = "ACTIVATED"
        spec["activated_utc"] = _now()
        SPEC.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return AUTH


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="print assessment and exit")
    ap.add_argument("--authorize", action="store_true",
                    help="write SAC_RFT_ACTIVATION.json if authorized")
    ap.add_argument("--wait", action="store_true",
                    help="poll until RSCFT finishes (RESULT or INTEGRITY_REQUIRED+audit)")
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--launch-control", action="store_true")
    ap.add_argument("--launch-both", action="store_true",
                    help="launch CONTROL then TREATMENT sequentially after authorize")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.wait:
        print(f"Waiting for sealed RSCFT outcome (poll every {args.poll_seconds}s)...",
              flush=True)
        while True:
            a = assess()
            if a["path"] != "WAITING" and a["path"] != "INTEGRITY_REQUIRED_PENDING_AUDIT":
                print(f"  [{_now()}] {a['path']}: {a['reason']}", flush=True)
                break
            print(f"  [{_now()}] {a['path']}: {a['reason']}", flush=True)
            time.sleep(max(5, int(args.poll_seconds)))

    assessment = assess()
    print(json.dumps(assessment, indent=2))
    if args.check and not (args.authorize or args.launch_control or args.launch_both):
        return 0 if assessment["authorized"] else 2

    if not assessment["authorized"]:
        if args.authorize or args.launch_control or args.launch_both:
            raise SystemExit(f"REFUSING: {assessment['reason']}")
        return 2

    if args.authorize or args.launch_control or args.launch_both:
        path = write_activation(assessment)
        print(f"\nAUTHORIZED -> {path}", flush=True)

    if args.launch_control or args.launch_both:
        # Preflight first (fail closed before 500k)
        pre = subprocess.run(
            [PYTHON, str(ROOT / "experiments" / "sac_rft_preflight.py"),
             "--device", args.device],
            cwd=str(ROOT))
        if pre.returncode != 0:
            raise SystemExit(f"REFUSING: sac_rft_preflight failed with code {pre.returncode}")
        ctrl = subprocess.run(
            [PYTHON, str(ROOT / "experiments" / "run_sac_rft_production.py"),
             "--arm", "control"],
            cwd=str(ROOT))
        if ctrl.returncode != 0:
            raise SystemExit(f"REFUSING: CONTROL failed with code {ctrl.returncode}")
        if args.launch_both:
            treat = subprocess.run(
                [PYTHON, str(ROOT / "experiments" / "run_sac_rft_production.py"),
                 "--arm", "treatment"],
                cwd=str(ROOT))
            if treat.returncode != 0:
                raise SystemExit(f"REFUSING: TREATMENT failed with code {treat.returncode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
