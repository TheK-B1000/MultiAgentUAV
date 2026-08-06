from __future__ import annotations

import pytest

import experiments.run_c3_decision_proximal_discovery as runner
from experiments.run_c3_decision_proximal_discovery import (
    CONTROLLABILITY_SCOPE,
    DISCOVERY_SEED_BASE,
    _parse_runtime_contract,
)


def test_runtime_contract_requires_frozen_cells_without_code_defaults():
    with pytest.raises(ValueError, match="runtime_cells"):
        _parse_runtime_contract({"runtime_cells": {}})


def test_runtime_contract_reads_trace_horizon_delta_and_utility_definition():
    contract = _parse_runtime_contract(
        {
            "runtime_cells": {
                "T_trace": 17,
                "H_response": 23,
                "delta": 0.125,
                "minimum_fork_rate": 0.20,
                "U": {
                    "name": "return",
                    "doomed_at_or_below": -2.0,
                },
            }
        }
    )
    assert contract.t_trace == 17
    assert contract.h_response == 23
    assert contract.delta == 0.125
    assert contract.utility_name == "return"
    assert contract.doomed_utility_threshold == -2.0
    assert contract.minimum_fork_rate == 0.20


def test_runtime_contract_requires_minimum_fork_rate():
    with pytest.raises(ValueError, match="minimum_fork_rate"):
        _parse_runtime_contract(
            {
                "runtime_cells": {
                    "T_trace": 40,
                    "H_response": 30,
                    "delta": 0.10,
                    "U": {"name": "carrier_survival", "doomed_at_or_below": 0.0},
                }
            }
        )


def test_stage3_scope_is_controllability_only():
    assert CONTROLLABILITY_SCOPE == "CONTROLLABILITY_SCREEN_ONLY"


def test_discovery_seed_block_remains_natural_replay_block():
    assert DISCOVERY_SEED_BASE == 9_400_000


def test_authorization_guard_fails_closed_before_contract_or_rollout(monkeypatch):
    monkeypatch.setattr(
        runner,
        "AUTH_PATH",
        runner.PROJECT_ROOT / "artifacts" / "c3_discovery" / "TEST_MISSING_AUTH.json",
    )
    with pytest.raises(SystemExit, match="Execution is prohibited"):
        runner._require_c3_execution_authorization()
