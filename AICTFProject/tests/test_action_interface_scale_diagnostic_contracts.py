"""Executable G0–G8 contracts for ACTION_INTERFACE_SCALE_DIAGNOSTIC."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/strategic_demand/sppo/ACTION_INTERFACE_SCALE_DIAGNOSTIC_CONTRACT_RESULT.json"


@pytest.mark.timeout(120)
def test_action_interface_scale_diagnostic_contracts_pass():
    from experiments.eval_action_interface_scale_diagnostic import run_contracts

    result = run_contracts()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    assert result["overall_pass"], json.dumps(result["gates"], indent=2)
