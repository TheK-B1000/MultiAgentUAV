from __future__ import annotations

import numpy as np
import torch

from experiments.validate_phase0_env_reuse import _compare, _digest


def _branch_record(marker: int = 1):
    digest = _digest(np.asarray([marker], dtype=np.int64))
    return {
        "restored": {
            "observation": {"grid": digest},
            "action_masks": {"core_blue": digest},
            "core_numeric_ledger": {"ticks": digest},
            "behavior_telemetry": digest,
        },
        "first_action": [marker, marker],
        "after_first": {
            "observation": {"grid": digest},
            "action_masks": {"core_blue": digest},
            "core_numeric_ledger": {"ticks": digest},
            "reward": digest,
            "done": digest,
        },
        "continuation": {
            "blue": marker,
            "red": 0,
            "win": 1,
            "margin": marker,
            "steps": marker,
            "return_digest": digest,
            "terminal_core_numeric_ledger": {"ticks": digest},
        },
    }


def test_exact_branch_records_pass_every_required_check():
    row = _branch_record()
    result = _compare(row, row)
    assert result["exact"] is True
    assert result["failed_checks"] == []


def test_hidden_ledger_mismatch_rejects_reuse():
    reference = _branch_record()
    reuse = _branch_record()
    reuse["restored"]["core_numeric_ledger"]["ticks"] = _digest(
        np.asarray([99], dtype=np.int64)
    )
    result = _compare(reference, reuse)
    assert result["exact"] is False
    assert "restored_core_ledger" in result["failed_checks"]


def test_digest_is_dtype_shape_and_rng_sensitive():
    assert _digest(np.asarray([1], dtype=np.int32)) != _digest(
        np.asarray([1], dtype=np.int64)
    )
    assert _digest(np.asarray([1, 2])) != _digest(np.asarray([[1, 2]]))
    g0 = torch.Generator().manual_seed(1)
    g1 = torch.Generator().manual_seed(2)
    assert _digest(g0) != _digest(g1)
