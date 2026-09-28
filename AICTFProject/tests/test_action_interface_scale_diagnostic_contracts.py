"""Executable G0–G8 contracts for ACTION_INTERFACE_SCALE_DIAGNOSTIC.

The attestation is written to ``*_CONTRACT_ATTESTATION.json``, never to
``*_CONTRACT_RESULT.json``. The latter is this diagnostic's pre-run authorization record
AND a G0-protected artifact of the downstream ACTION_INTERFACE_COMMITMENT_MECHANISM
probe. This test used to rewrite it on every run, which re-stamped its utc on
2026-09-22 and silently broke that probe's frozen pin (undetected at the time, because
G0 recorded hashes without comparing them).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ATTESTATION = (ROOT / "artifacts/strategic_demand/sppo/"
               "ACTION_INTERFACE_SCALE_DIAGNOSTIC_CONTRACT_ATTESTATION.json")


@pytest.mark.timeout(120)
def test_action_interface_scale_diagnostic_contracts_pass():
    from experiments.eval_action_interface_scale_diagnostic import run_contracts

    result = run_contracts()
    ATTESTATION.parent.mkdir(parents=True, exist_ok=True)
    ATTESTATION.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    assert result["overall_pass"], json.dumps(result["gates"], indent=2)
