from __future__ import annotations

import numpy as np

from experiments.eval_exp2_k2_terminal import (
    N_PAIRED,
    _load_protocol,
    _paired_mean_ci,
    _ratio_ci,
    guard_rails,
)


def test_frozen_terminal_contract_and_checkpoint_hashes():
    protocol = _load_protocol()
    assert protocol["seed_blocks"]["evaluation"]["range"] == "8300001..8300192"
    preflight = guard_rails(launch=False)
    assert set(preflight["checkpoint_hashes"]) == {"student", "pi_A", "pi_B"}


def test_paired_bootstrap_keeps_seed_level_vector():
    values = np.ones(N_PAIRED, dtype=float)
    result = _paired_mean_ci(values)
    assert result == {"mean": 1.0, "lcb95": 1.0, "ucb95": 1.0}


def test_retention_ratio_bootstrap_uses_paired_seed_rows():
    numer = np.full(N_PAIRED, 0.9)
    denom = np.ones(N_PAIRED)
    result = _ratio_ci(numer, denom)
    assert abs(result["rho"] - 0.9) < 1e-12
    assert abs(result["lcb95"] - 0.9) < 1e-12
    assert abs(result["ucb95"] - 0.9) < 1e-12
