"""C6 code-safety gate: OP5-OP12 profiles must be byte-identical to 55cf5e7.

The C6 freeze requires new opponent families to be strictly ADDITIVE. Silently
altering an existing profile would invalidate G0 training, C3, C4 and C5 at once,
and every one of those verdicts would still *look* valid -- there is no downstream
check that would notice.

The expected values below were read from profile_for_opponent_key at 55cf5e7,
before any C6 profile existed. Do not regenerate them from current code: that
would make the test assert only that the code equals itself.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from gpu_env._core._bt_profiles import (
    LRO_AUDITED_OPPONENT_POOL,
    OPPONENT_ALIASES,
    profile_for_opponent_key,
)

LEGACY = ["OP5", "OP6", "OP7", "OP8", "OP9", "OP10", "OP11", "OP12"]
BASELINE = Path(__file__).resolve().parent / "data" / "c6_legacy_profile_baseline.json"


@pytest.fixture(scope="module")
def legacy_baseline() -> dict:
    """Profiles as they were at 55cf5e7, extracted from git history.

    Captured by loading the pre-C6 module from `git show`, NOT by dumping the
    current code -- a self-generated baseline would assert only that the code
    equals itself.
    """
    return json.loads(BASELINE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("key", LEGACY)
def test_legacy_profile_fields_unchanged(key, legacy_baseline):
    """Every field of every legacy profile matches the recorded baseline."""
    current = dataclasses.asdict(profile_for_opponent_key(key))
    expected = legacy_baseline[key]
    assert current == expected, (
        f"{key} profile changed. C6 must be ADDITIVE -- altering an existing "
        f"profile invalidates G0 training, C3, C4 and C5 silently.\n"
        f"differing fields: "
        f"{ {k: (expected.get(k), current.get(k)) for k in current if current[k] != expected.get(k)} }"
    )


def test_legacy_levels_unchanged(legacy_baseline):
    """Level identity is stable; C6 must not renumber an existing opponent."""
    for key in LEGACY:
        assert profile_for_opponent_key(key).level == legacy_baseline[key]["level"]


def test_c6_families_are_new_levels():
    """C6 occupies levels no legacy opponent uses."""
    legacy_levels = {profile_for_opponent_key(k).level for k in LEGACY}
    for key in ("C6A", "C6B"):
        assert profile_for_opponent_key(key).level not in legacy_levels


def test_c6_not_on_the_permanent_board():
    """C6 families are experimental fixtures, not OP13/OP14."""
    assert len(LRO_AUDITED_OPPONENT_POOL) == 7
    for name in LRO_AUDITED_OPPONENT_POOL:
        assert not name.startswith("C6")
    for short in OPPONENT_ALIASES:
        assert not short.startswith("OP13") and not short.startswith("OP14")

    from experiments.run_g0_v2_seed import OPPONENTS
    assert "C6A" not in OPPONENTS and "C6B" not in OPPONENTS
    assert len(OPPONENTS) == 7


def test_c6_families_differ_on_the_intended_axis():
    """The two families sit on opposite sides of defensive allocation.

    This asserts the MECHANICAL separation the freeze specifies. It says nothing
    about which response should win -- that is prohibited and is Stage 2's job.
    """
    a = profile_for_opponent_key("C6A")
    b = profile_for_opponent_key("C6B")
    assert a.enable_defender is False and b.enable_defender is True
    assert a.intercept_block_base < b.intercept_block_base
    assert a.counter_always is True and b.counter_always is False
