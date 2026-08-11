"""C6 code-safety gate: legacy opponent RESOLUTION is unchanged from 55cf5e7.

The sibling test pins the BTProfile knobs. This one pins the identity plumbing —
canonicalization, level mapping, BT membership, and the alias/synonym tables —
because that seam is where opponent-naming mistakes have actually landed twice in
this project: once reading a synonym's deprecated side as canonical, and once
before that.

A profile can be byte-identical while an alias silently reroutes a key to a
different level, which would repoint every historical experiment that names an
opponent by string. Baseline captured from git history at 55cf5e7, not from
current code.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_env._core._bt_profiles import (
    OPPONENT_ALIASES,
    OPPONENT_SYNONYMS,
    LRO_AUDITED_OPPONENT_POOL,
    canonicalize_opponent_key,
    is_bt_opponent,
    normalize_bt_level,
)
from rl.evaluation.opponent_resolution import validate_opponent_name

BASELINE = Path(__file__).resolve().parent / "data" / "c6_legacy_resolution_baseline.json"


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def _keys(baseline: dict):
    return [k for k in baseline if not k.startswith("_")]


def test_canonicalization_unchanged(baseline):
    """Every legacy key and historical synonym resolves where it always did."""
    for key in _keys(baseline):
        assert canonicalize_opponent_key(key) == baseline[key]["canonical"], (
            f"{key} now canonicalizes differently. Aliases reroute every historical "
            f"experiment that names this opponent by string.")


def test_levels_unchanged(baseline):
    for key in _keys(baseline):
        assert normalize_bt_level(key) == baseline[key]["level"]


def test_bt_membership_unchanged(baseline):
    for key in _keys(baseline):
        assert is_bt_opponent(key) is baseline[key]["is_bt"]


def test_alias_and_synonym_tables_are_supersets(baseline):
    """C6 may ADD entries; it may not change or remove an existing one."""
    for table, recorded in ((OPPONENT_ALIASES, baseline["_ALIASES"]),
                            (OPPONENT_SYNONYMS, baseline["_SYNONYMS"])):
        for k, v in recorded.items():
            assert table.get(k) == v, f"{k} remapped from {v!r} to {table.get(k)!r}"


def test_audited_pool_unchanged(baseline):
    """The permanent board is exactly what it was; C6 is not on it."""
    assert list(LRO_AUDITED_OPPONENT_POOL) == baseline["_LRO_POOL"]


def test_c6_keys_resolve_and_are_distinct(baseline):
    """C6 keys work through the same plumbing without colliding with legacy."""
    legacy_levels = {baseline[k]["level"] for k in _keys(baseline)}
    for key in ("C6A", "C6B"):
        assert validate_opponent_name(key) == key
        assert is_bt_opponent(key)
        assert normalize_bt_level(key) not in legacy_levels
