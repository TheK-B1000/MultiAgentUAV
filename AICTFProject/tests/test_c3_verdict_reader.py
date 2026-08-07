"""A software failure must never become a scientifically plausible negative.

The C3 analyzer originally guessed at verdict/status/fork_verdict. The persisted
Stage-3 field is `episode_status`, so it found none of them and silently scored
every anchor as unqualified. That would have written C3_NOT_PASS after ~9 hours
of compute, indistinguishable from a real negative result.

These tests pin the corrected contract: read the real field, and abort on
anything this reader cannot interpret rather than defaulting to "not qualified".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("numpy")

from experiments.analyze_c3_stage3 import (  # noqa: E402
    NOT_QUALIFIED,
    QUALIFIED,
    VERDICT_FIELD,
    is_qualified,
)


def test_qualified_is_true():
    assert is_qualified({VERDICT_FIELD: QUALIFIED}) is True


def test_not_qualified_is_false():
    assert is_qualified({VERDICT_FIELD: NOT_QUALIFIED}) is False


def test_missing_field_hard_fails():
    """The original bug: no recognised field, silently scored as unqualified."""
    with pytest.raises(SystemExit, match="has no 'episode_status' field"):
        is_qualified({"train_seed": 3200001, "opponent": "OP6"})


def test_legacy_guessed_field_names_do_not_satisfy_the_reader():
    """verdict/status/fork_verdict were the guesses; none is the real field."""
    for legacy in ("verdict", "status", "fork_verdict"):
        with pytest.raises(SystemExit, match="has no 'episode_status' field"):
            is_qualified({legacy: QUALIFIED})


@pytest.mark.parametrize(
    "value",
    ["", "qualified", "QUALIFIED", "SOMETHING_NEW", "None", "0", "ERROR"],
)
def test_unknown_verdict_hard_fails(value):
    """An unrecognised value must abort, not silently count as not-qualified."""
    with pytest.raises(SystemExit, match="unrecognised episode_status"):
        is_qualified({VERDICT_FIELD: value})


def test_real_stage3_rows_are_readable_when_present():
    """Against live artifacts, every persisted row must parse without aborting."""
    import json

    p = PROJECT_ROOT / "artifacts" / "c3_discovery" / "C3_STAGE3_ANCHOR_RESULTS.jsonl"
    if not p.is_file() or p.stat().st_size == 0:
        pytest.skip("no Stage-3 results on disk yet")
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    verdicts = [is_qualified(r) for r in rows]
    assert len(verdicts) == len(rows)
    # A reader that scored everything False is the exact failure being guarded.
    assert any(verdicts), "no qualified anchors parsed — the reader is misreading"
