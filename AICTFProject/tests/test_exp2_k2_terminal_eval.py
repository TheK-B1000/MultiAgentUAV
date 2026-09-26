from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

# experiments/eval_sppo_v1_terminal.py is built BY rebinding this module's globals
# (PROTOCOL, SEED_BASE, EXPECTED_HASHES, ...) at import time -- that is how the frozen SPPO
# terminal evaluator reuses the EXP2 code path, and it must not change. Any earlier test
# that imports it therefore turns `experiments.eval_exp2_k2_terminal` into the SPPO
# evaluator for the rest of the session. The EXP2 contract is checked on a private,
# pristine copy of the module instead, so it cannot depend on test order.
_SRC = Path(__file__).resolve().parents[1] / "experiments" / "eval_exp2_k2_terminal.py"
_spec = importlib.util.spec_from_file_location("_exp2_k2_terminal_pristine", _SRC)
_exp2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_exp2)

N_PAIRED = _exp2.N_PAIRED
_load_protocol = _exp2._load_protocol
_paired_mean_ci = _exp2._paired_mean_ci
_ratio_ci = _exp2._ratio_ci
guard_rails = _exp2.guard_rails


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
