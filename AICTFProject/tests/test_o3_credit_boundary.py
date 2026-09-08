"""Adversarial tests for the O3 credit boundary.

Two independent defences, tested separately because they fail differently:

STRUCTURAL -- only credited rows are indexed into each consumer. Catches a wrong
index.

ARITHMETIC -- prefix rows carry sentinels so absurd that any path reaching
around the index produces an impossible result. Catches a wrong path, e.g. code
that normalizes advantages before filtering.

A correct mask sitting beside code that reaches around it passes the first and
fails the second, which is why both exist.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch")

from experiments.o3_credit_boundary import (  # noqa: E402
    CREDIT_FIELD,
    CreditAudit,
    filter_batch_to_credited,
)

# Sentinels chosen so a leak is arithmetically visible, not merely statistical.
ADV_SENTINEL = 1e6
RET_SENTINEL = -9.9e5
LOGP_SENTINEL = 1234.5


def _poisoned_batch(n: int, credited: list[int]) -> dict:
    """Credited rows carry ordinary values; prefix rows carry sentinels."""
    credit = torch.zeros(n, dtype=torch.bool)
    credit[credited] = True
    adv = torch.where(credit, torch.linspace(-1.0, 1.0, n), torch.full((n,), ADV_SENTINEL))
    ret = torch.where(credit, torch.linspace(0.0, 1.0, n), torch.full((n,), RET_SENTINEL))
    logp = torch.where(credit, torch.zeros(n), torch.full((n,), LOGP_SENTINEL))
    return {
        CREDIT_FIELD: credit,
        "advantages": adv,
        "returns": ret,
        "log_probs": logp,
        "values": torch.arange(n, dtype=torch.float32),
        "row_id": torch.arange(n, dtype=torch.int64),
    }


def test_structural_only_credited_rows_survive():
    audit = CreditAudit()
    out = filter_batch_to_credited(_poisoned_batch(8, [1, 4, 6]), audit)
    assert out["row_id"].tolist() == [1, 4, 6]
    assert out[CREDIT_FIELD].all()
    assert audit.rows_offered == 8 and audit.rows_credited == 3


def test_arithmetic_no_sentinel_survives_into_any_consumer():
    audit = CreditAudit()
    out = filter_batch_to_credited(_poisoned_batch(10, [0, 5, 9]), audit)
    for name, sentinel in (("advantages", ADV_SENTINEL),
                           ("returns", RET_SENTINEL),
                           ("log_probs", LOGP_SENTINEL)):
        assert not torch.isclose(
            out[name], torch.tensor(sentinel)
        ).any(), f"{name} carries a prefix sentinel into the update"


def test_advantage_normalization_is_uncontaminated():
    """The exact leak the sentinel is designed to expose."""
    audit = CreditAudit()
    batch = _poisoned_batch(64, list(range(0, 64, 8)))   # 8 credited of 64
    out = filter_batch_to_credited(batch, audit)

    adv = out["advantages"]
    normed = (adv - adv.mean()) / (adv.std() + 1e-8)
    assert torch.isfinite(normed).all()
    assert normed.abs().max().item() < 10.0, "normalization sees sentinel-scale data"

    # What a leak would have produced, for contrast.
    leaked = batch["advantages"]
    leaked_normed = (leaked - leaked.mean()) / (leaked.std() + 1e-8)
    assert leaked_normed.abs().max().item() > 1.0
    assert not torch.allclose(normed.mean(), leaked_normed.mean())


def test_five_abort_counts_are_zero_on_a_correct_filter():
    audit = CreditAudit()
    filter_batch_to_credited(_poisoned_batch(16, [2, 3, 11]), audit)
    audit.assert_clean()
    d = audit.to_dict()
    for k in ("pre_handoff_actor_samples", "pre_handoff_critic_samples",
              "pre_handoff_entropy_samples", "pre_handoff_norm_samples",
              "pre_handoff_return_targets"):
        assert d[k] == 0


def test_assert_clean_aborts_when_a_prefix_row_survives():
    """Consumer-derived counting must catch a leak the mask alone would miss."""
    audit = CreditAudit()
    audit.actor_input_prefix_count = 3
    with pytest.raises(AssertionError, match="credit boundary violated"):
        audit.assert_clean()


def test_fully_uncredited_minibatch_is_dropped_not_passed_through():
    audit = CreditAudit()
    assert filter_batch_to_credited(_poisoned_batch(8, []), audit) is None
    assert audit.minibatches_dropped_empty == 1
    assert audit.rows_credited == 0


def test_missing_credit_field_refuses_to_train():
    """Absent mask must abort, never default to crediting everything."""
    audit = CreditAudit()
    batch = _poisoned_batch(4, [0])
    del batch[CREDIT_FIELD]
    with pytest.raises(KeyError, match="o3_credit"):
        filter_batch_to_credited(batch, audit)


def test_all_fields_are_filtered_consistently():
    """Every per-row field must be indexed by the same rows."""
    audit = CreditAudit()
    out = filter_batch_to_credited(_poisoned_batch(12, [1, 7]), audit)
    n = out["row_id"].shape[0]
    for name, value in out.items():
        assert value.shape[0] == n, f"{name} desynchronized from the filtered rows"
    ids = out["row_id"].tolist()
    assert out["values"].tolist() == [float(i) for i in ids]
