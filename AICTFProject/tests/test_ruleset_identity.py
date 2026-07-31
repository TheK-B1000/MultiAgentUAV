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
    RULESET_FIELDS,
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


def test_legacy_v1_checkpoint_on_disk_is_rejected(tmp_path):
    """The real G0 checkpoints predate stamping and must not enter a V2 run."""
    torch = pytest.importorskip("torch")
    from gpu_env import GPUFieldConfig
    from rl.custom_ppo.checkpoints.loader import (
        read_checkpoint_ruleset, verify_checkpoint_ruleset,
    )

    # A legacy checkpoint: no "ruleset" key at all.
    legacy = tmp_path / "legacy_v1.zip"
    torch.save({"model_state_dict": {}, "cfg": {}}, legacy)
    assert read_checkpoint_ruleset(str(legacy)) == {}

    env_cfg = GPUFieldConfig(n_envs=1, device="cpu", map_layout="map_a")
    with pytest.raises(RulesetMismatchError):
        verify_checkpoint_ruleset(str(legacy), env_cfg)

    # Override is permitted but marks the result ineligible.
    with pytest.warns(RuntimeWarning):
        r = verify_checkpoint_ruleset(str(legacy), env_cfg, allow_mismatch=True)
    assert r["formal_result_eligible"] is False
    assert r["checkpoint_ruleset"] == LEGACY_UNKNOWN


def test_stamped_v2_checkpoint_loads(tmp_path):
    torch = pytest.importorskip("torch")
    from gpu_env import GPUFieldConfig
    from rl.custom_ppo.checkpoints.loader import verify_checkpoint_ruleset

    env_cfg = GPUFieldConfig(n_envs=1, device="cpu", map_layout="map_a")
    good = tmp_path / "v2.zip"
    torch.save({"model_state_dict": {}, "cfg": {}, "ruleset": dict(V2)}, good)
    r = verify_checkpoint_ruleset(str(good), env_cfg)
    assert r["match"] is True and r["formal_result_eligible"] is True

    bad = tmp_path / "v1.zip"
    torch.save({"model_state_dict": {}, "cfg": {}, "ruleset": dict(V1)}, bad)
    with pytest.raises(RulesetMismatchError):
        verify_checkpoint_ruleset(str(bad), env_cfg)


def test_save_fails_closed_when_ruleset_unavailable():
    """An unstamped checkpoint would reject itself on load, wasting the run."""
    from rl.custom_ppo.checkpoints.loader import _env_ruleset_fingerprint
    from rl.ruleset_identity import RulesetFingerprintError

    class NoEnvTrainer:
        pass

    with pytest.raises(RulesetFingerprintError):
        _env_ruleset_fingerprint(NoEnvTrainer())


def test_diagnostic_mode_may_save_unstamped_with_warning():
    from rl.custom_ppo.checkpoints.loader import _env_ruleset_fingerprint

    class DiagTrainer:
        allow_unstamped_checkpoint = True

    with pytest.warns(RuntimeWarning, match="UNSTAMPED"):
        assert _env_ruleset_fingerprint(DiagTrainer()) == {}


def test_save_finds_ruleset_through_env_core_cfg():
    torch = pytest.importorskip("torch")  # noqa: F841
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    from rl.custom_ppo.checkpoints.loader import _env_ruleset_fingerprint

    env = GPUCTFVecEnv(GPUFieldConfig(n_envs=1, device="cpu", map_layout="map_a"))
    try:
        class T:
            pass
        t = T()
        t.env = env
        fp = _env_ruleset_fingerprint(t)
        assert fp["ruleset_id"] == "RULESET_V2_AQUATICUS_10S"
        assert is_complete(fp)
    finally:
        env.close()


def test_stamp_artifact_includes_full_fingerprint_and_eligibility():
    from rl.ruleset_identity import artifact_row_fields, stamp_artifact

    cfg_like = dict(V2)
    art = stamp_artifact({"run_tag": "g0v2_s1"}, cfg_like)
    for k in RULESET_FIELDS:
        assert k in art, f"artifact missing {k}"
    assert art["formal_result_eligible"] is True
    assert art["run_tag"] == "g0v2_s1"

    row = artifact_row_fields(cfg_like, formal_result_eligible=False)
    assert row["formal_result_eligible"] is False
    assert row["ruleset_id"] == "RULESET_V2_AQUATICUS_10S"
    # a label alone is insufficient -- the value fields must travel with it
    assert row["tag_min_interval_seconds"] == 10.0
    assert row["taggers_required"] == 1


def test_artifact_stamp_from_live_env_config():
    torch = pytest.importorskip("torch")  # noqa: F841
    from gpu_env import GPUFieldConfig
    from rl.ruleset_identity import artifact_row_fields

    cfg = GPUFieldConfig(n_envs=1, device="cpu", map_layout="map_a")
    row = artifact_row_fields(cfg)
    assert row["ruleset_id"] == "RULESET_V2_AQUATICUS_10S"
    assert is_complete(row)
