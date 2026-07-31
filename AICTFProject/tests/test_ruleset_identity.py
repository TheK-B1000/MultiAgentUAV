"""Tests for ruleset stamping and mismatch rejection.

The failure this prevents is concrete: the old G0 family in
``checkpoints/k2v2_piR/`` was trained under RULESET_V1, where a lone defender
could not tag at all. Loading it into a V2 environment would produce a
V2-labelled result from a policy that learned a different game.
"""
from __future__ import annotations

import pytest

from rl.ruleset_identity import (
    LEGACY_UNKNOWN,
    RulesetMismatchError,
    classify,
    compare,
    enforce,
    fingerprint,
    is_complete,
    stamp,
)

V2 = {
    "ruleset_id": "RULESET_V2_AQUATICUS_10S",
    "taggers_required": 1,
    "tag_min_interval_seconds": 10.0,
    "tag_nearest_only": True,
    "tag_channel_seconds": 0.0,
    "suppression_attackers_required": 2,
}
V1 = {
    "ruleset_id": "RULESET_V1_TWO_TAGGER",
    "taggers_required": 2,
    "tag_min_interval_seconds": 0.0,
    "tag_nearest_only": False,
    "tag_channel_seconds": 1.0,
    "suppression_attackers_required": 2,
}


def test_matching_ruleset_loads():
    r = enforce(dict(V2), dict(V2), context="unit")
    assert r["match"] is True
    assert r["formal_result_eligible"] is True
    assert r["ruleset_mismatch_override"] is False


def test_v1_checkpoint_into_v2_env_is_hard_error():
    with pytest.raises(RulesetMismatchError) as ei:
        enforce(dict(V1), dict(V2), context="k2v2_piR 1M checkpoint")
    assert "RULESET_V1_TWO_TAGGER" in str(ei.value)
    assert "RULESET_V2_AQUATICUS_10S" in str(ei.value)


def test_missing_metadata_is_legacy_unknown_and_rejected():
    assert classify(None) == LEGACY_UNKNOWN
    assert classify({}) == LEGACY_UNKNOWN
    with pytest.raises(RulesetMismatchError):
        enforce(None, dict(V2))


def test_incomplete_metadata_is_legacy_unknown():
    partial = {"ruleset_id": "RULESET_V2_AQUATICUS_10S", "taggers_required": 1}
    assert not is_complete(partial)
    assert classify(partial) == LEGACY_UNKNOWN
    with pytest.raises(RulesetMismatchError):
        enforce(partial, dict(V2))


def test_comparison_uses_full_fields_not_just_the_label():
    """Same friendly id, one differing field -> still a mismatch."""
    sneaky = dict(V2)
    sneaky["tag_min_interval_seconds"] = 30.0   # label says 10s, value says 30s
    res = compare(sneaky, V2)
    assert res["match"] is False
    assert "tag_min_interval_seconds" in res["differing_fields"]
    with pytest.raises(RulesetMismatchError):
        enforce(sneaky, dict(V2))


def test_override_warns_and_marks_result_ineligible():
    with pytest.warns(RuntimeWarning, match="RULESET MISMATCH OVERRIDE"):
        r = enforce(dict(V1), dict(V2), allow_mismatch=True)
    assert r["match"] is False
    assert r["formal_result_eligible"] is False
    assert r["ruleset_mismatch_override"] is True


def test_fingerprint_from_live_config():
    torch = pytest.importorskip("torch")  # noqa: F841
    from gpu_env import GPUFieldConfig

    cfg = GPUFieldConfig(n_envs=1, device="cpu", map_layout="map_a")
    fp = fingerprint(cfg)
    assert is_complete(fp)
    assert fp["ruleset_id"] == "RULESET_V2_AQUATICUS_10S"
    assert fp["taggers_required"] == 1
    # A live V2 env must load a V2 checkpoint and reject a V1 one.
    assert enforce(dict(V2), fp)["match"] is True
    with pytest.raises(RulesetMismatchError):
        enforce(dict(V1), fp)


def test_stamp_writes_all_fields():
    target = {"run_tag": "g0v2_s1"}
    stamp(target, dict(V2))
    for k in V2:
        assert target[k] == V2[k]
    assert target["run_tag"] == "g0v2_s1"
