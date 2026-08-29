"""Freeze the pre-DEV RASR-PPO implementation gate.

This runner performs no scorer fit, environment collection, or policy update.
It authorizes DEV collection only when the implementation and paper-regression
contracts pass. Policy launch remains false pending a separate scorer
qualification PASS artifact.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_rasrppo_ladder import (  # noqa: E402
    IMPLEMENTATION_GATE,
    PROTOCOL,
    SCORER_QUALIFICATION,
    build_config,
)
from experiments.run_sppo_production import build_production_config  # noqa: E402

IDENTITY_FIELDS = {
    "checkpoint_dir",
    "episode_csv_path",
    "metrics_csv_path",
    "run_tag",
}


def _diff(left, right) -> list[str]:
    a, b = dataclasses.asdict(left), dataclasses.asdict(right)
    return sorted(key for key in a if a[key] != b[key])


def _final_is_sealed() -> tuple[bool, list[str]]:
    rasr_dir = IMPLEMENTATION_GATE.parent
    offenders = []
    for path in rasr_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if (
            "final_eval" in name
            or "terminal_evaluation" in name
            or (path.suffix.lower() == ".csv" and "106" in name)
        ):
            offenders.append(str(path.relative_to(ROOT)))
    if (rasr_dir / "FINAL").exists():
        offenders.append(str((rasr_dir / "FINAL").relative_to(ROOT)))
    return not offenders, offenders


def _run_tests() -> tuple[bool, list[str], int]:
    tests = [
        "tests/test_rasrppo_core.py",
        "tests/test_rasrppo_ladder_contracts.py",
        "tests/test_v5i4_paper_faithful.py",
    ]
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *tests],
        cwd=ROOT,
        check=False,
    )
    return completed.returncode == 0, tests, completed.returncode


def build_gate() -> dict:
    tests_pass, tests, returncode = _run_tests()
    production, _ = build_production_config()
    s0, _ = build_config("S0")
    r1, _ = build_config("R1")
    r2, _ = build_config("R2")
    r3, _ = build_config("R3")
    diffs = {
        "S0_vs_SPPPO_V1": _diff(production, s0),
        "R1_vs_S0": _diff(s0, r1),
        "R2_vs_R1": _diff(r1, r2),
        "R3_vs_R2": _diff(r2, r3),
    }
    expected_r1 = IDENTITY_FIELDS | {
        "rasr_regime_qpsi",
        "rasr_regime_qpsi_path",
    }
    if r1.rasr_regime_qpsi_sha256 != s0.rasr_regime_qpsi_sha256:
        expected_r1.add("rasr_regime_qpsi_sha256")
    diff_ok = (
        set(diffs["S0_vs_SPPPO_V1"]) == IDENTITY_FIELDS | {"seed"}
        and set(diffs["R1_vs_S0"]) == expected_r1
        and set(diffs["R2_vs_R1"])
        == IDENTITY_FIELDS | {"rasr_private_critic_heads"}
        and set(diffs["R3_vs_R2"])
        == IDENTITY_FIELDS | {"rasr_directed_identity"}
    )
    final_sealed, final_offenders = _final_is_sealed()
    verdict = "PASS" if tests_pass and diff_ok and final_sealed else "FAIL"

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    checklist = []
    tested_items = {
        0: diff_ok,
        1: diff_ok,
        2: diff_ok,
        3: diff_ok,
        4: tests_pass,
        5: tests_pass,
        6: tests_pass,
        7: tests_pass,
        8: tests_pass,
        9: tests_pass,
        10: tests_pass,
        11: tests_pass,
        12: tests_pass,
        13: tests_pass,
        14: tests_pass,
        15: final_sealed,
    }
    for index, item in enumerate(
        protocol["implementation_gate_before_any_DEV_or_policy_step"]
    ):
        substitute = index in {12, 13}
        checklist.append(
            {
                "item": item,
                "status": (
                    "PASS_CONTRACT_SUBSTITUTE"
                    if substitute and tested_items[index]
                    else "PASS" if tested_items[index] else "FAIL"
                ),
                "evidence": (
                    "lightweight builder/attachment contract substitute; no live "
                    "environment, DEV seed, or policy step was opened"
                    if substitute
                    else
                    "focused CPU contract tests and resolved-config audit"
                    if index < 15
                    else "filesystem FINAL seal audit"
                ),
            }
        )

    return {
        "record_id": "RASR_PPO_IMPLEMENTATION_GATE",
        "protocol": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
        "verdict": verdict,
        "dev_collection_authorized": verdict == "PASS",
        "policy_launch_authorized": False,
        "policy_launch_blocker": (
            f"{SCORER_QUALIFICATION.relative_to(ROOT)} must exist with verdict PASS"
        ),
        "scope": (
            "CPU contract gate; integrated runner lifecycle and 16/0/0/16 behavior "
            "are covered by coherent builder/attachment contracts without DEV, FINAL, "
            "or a 1M policy run"
        ),
        "tests": {
            "paths": tests,
            "returncode": returncode,
            "passed": tests_pass,
        },
        "resolved_config_delta_keys": diffs,
        "final_sealed": final_sealed,
        "final_offenders": final_offenders,
        "checklist": checklist,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    gate = build_gate()
    rendered = json.dumps(gate, indent=2) + "\n"
    if IMPLEMENTATION_GATE.exists():
        existing = IMPLEMENTATION_GATE.read_text(encoding="utf-8")
        if existing != rendered and not args.force:
            raise RuntimeError(
                f"refusing to overwrite different gate {IMPLEMENTATION_GATE}; use --force"
            )
        if existing == rendered:
            print(f"unchanged -> {IMPLEMENTATION_GATE}")
            return 0 if gate["verdict"] == "PASS" else 1
    IMPLEMENTATION_GATE.write_text(rendered, encoding="utf-8")
    print(f"{gate['verdict']} -> {IMPLEMENTATION_GATE}")
    return 0 if gate["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
