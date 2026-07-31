"""Checkpoint <-> RunIdentity join.

A checkpoint used to carry its own homemade passport: a ``ruleset`` payload
unrelated to the run-level identity block. Two parallel systems drift.

``verify_checkpoint_run_identity`` is the single boundary both save and load
pass through, so three things must be three representations of ONE fact:

    checkpoint["ruleset"]
    checkpoint["artifact_identity"]
    live RunIdentity

A matching ``ruleset_id`` alone is never sufficient -- the whole point is that a
label can agree while a field that changes play does not.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gpu_env import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402
from rl.ruleset_identity import (  # noqa: E402
    ARTIFACT_IDENTITY_KEY,
    CHECKPOINT_RULESET_KEY,
    RunIdentity,
    RunIdentityError,
    build_checkpoint_identity_payload,
    build_formal_run_identity,
    ruleset_fingerprint_hash,
    verify_checkpoint_run_identity,
)

V2 = dict(taggers_required=1, tag_min_interval_seconds=10.0, tag_nearest_only=True,
          tag_channel_seconds=0.0, suppression_attackers_required=2)
V1 = dict(taggers_required=2, tag_min_interval_seconds=0.0, tag_nearest_only=False,
          tag_channel_seconds=1.0, suppression_attackers_required=2,
          ruleset_id="RULESET_V1_TWO_TAGGER")


@pytest.fixture
def env():
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2, map_set="train",
        map_layout="map_a", max_decision_steps=240, aquaticus_profile=True,
        rules_profile="OURS", device="cpu", seed=2_700_001,
        obstacle_obs_channel=True, **V2)
    e = GPUCTFVecEnv(cfg)
    e.reset()
    yield e
    e.close()


@pytest.fixture
def ident(env):
    return build_formal_run_identity(env, run_id="train_run")


def payload_for(identity: RunIdentity, **over):
    p = {"model_state_dict": {}, "global_step": 1000}
    p.update(build_checkpoint_identity_payload(identity))
    p.update(over)
    return p


# --- happy path -------------------------------------------------------------

def test_formal_save_and_load_agree(ident):
    p = payload_for(ident)
    assert verify_checkpoint_run_identity(p, ident, operation="save")["match"]
    assert verify_checkpoint_run_identity(p, ident, operation="load")["match"]


def test_payload_carries_both_representations(ident):
    p = payload_for(ident)
    assert p[CHECKPOINT_RULESET_KEY]["ruleset_id"] == ident.ruleset_id
    assert p[ARTIFACT_IDENTITY_KEY]["ruleset_fingerprint"] == ident.ruleset_fingerprint
    assert ruleset_fingerprint_hash(p[CHECKPOINT_RULESET_KEY]) == ident.ruleset_fingerprint


def test_builder_refuses_a_non_identity(env):
    with pytest.raises(RunIdentityError):
        build_checkpoint_identity_payload(env.core.cfg)   # type: ignore[arg-type]


# --- rejections -------------------------------------------------------------

def test_missing_artifact_identity_rejected(ident):
    p = payload_for(ident)
    del p[ARTIFACT_IDENTITY_KEY]
    with pytest.raises(RunIdentityError, match="artifact_identity"):
        verify_checkpoint_run_identity(p, ident, operation="load")


def test_legacy_checkpoint_rejected(ident):
    with pytest.raises(RunIdentityError):
        verify_checkpoint_run_identity(
            {"model_state_dict": {}}, ident, operation="load")


def test_missing_ruleset_payload_rejected(ident):
    p = payload_for(ident)
    del p[CHECKPOINT_RULESET_KEY]
    with pytest.raises(RunIdentityError, match="ruleset"):
        verify_checkpoint_run_identity(p, ident, operation="load")


def test_v1_checkpoint_under_live_v2_rejected(ident):
    p = payload_for(ident)
    p[CHECKPOINT_RULESET_KEY] = dict(V1)
    p[ARTIFACT_IDENTITY_KEY] = dict(
        p[ARTIFACT_IDENTITY_KEY],
        ruleset_id="RULESET_V1_TWO_TAGGER",
        ruleset_fingerprint=ruleset_fingerprint_hash(V1))
    with pytest.raises(RunIdentityError):
        verify_checkpoint_run_identity(p, ident, operation="load")


def test_same_label_changed_cooldown_rejected(ident):
    """The label agrees; one field that changes play does not."""
    p = payload_for(ident)
    tampered = dict(ident.ruleset, tag_min_interval_seconds=30.0)
    p[CHECKPOINT_RULESET_KEY] = tampered
    p[ARTIFACT_IDENTITY_KEY] = dict(
        p[ARTIFACT_IDENTITY_KEY],
        ruleset_fingerprint=ruleset_fingerprint_hash(tampered))
    assert p[ARTIFACT_IDENTITY_KEY]["ruleset_id"] == ident.ruleset_id
    with pytest.raises(RunIdentityError):
        verify_checkpoint_run_identity(p, ident, operation="load")


def test_matching_fingerprint_different_map_rejected(ident):
    p = payload_for(ident)
    p[ARTIFACT_IDENTITY_KEY] = dict(p[ARTIFACT_IDENTITY_KEY], canonical_map="map_b")
    with pytest.raises(RunIdentityError, match="canonical_map"):
        verify_checkpoint_run_identity(p, ident, operation="load")


def test_internal_disagreement_rejected(ident):
    """artifact_identity vs the checkpoint's own ruleset payload."""
    p = payload_for(ident)
    p[ARTIFACT_IDENTITY_KEY] = dict(
        p[ARTIFACT_IDENTITY_KEY], ruleset_fingerprint="0" * 64)
    with pytest.raises(RunIdentityError, match="internally inconsistent"):
        verify_checkpoint_run_identity(p, ident, operation="load")


def test_run_id_mismatch_rejected_by_default(ident, env):
    other = build_formal_run_identity(env, run_id="a_different_run")
    p = payload_for(other)
    with pytest.raises(RunIdentityError, match="run_id"):
        verify_checkpoint_run_identity(p, ident, operation="load")


def test_standalone_evaluation_may_allow_different_run_id(ident, env):
    """Explicit argument, not a permissive comparison."""
    other = build_formal_run_identity(env, run_id="a_different_run")
    p = payload_for(other)
    res = verify_checkpoint_run_identity(
        p, ident, operation="load", allow_different_run_id=True)
    assert res["match"] and res["different_run_id_allowed"] is True
    assert res["checkpoint_run_id"] != res["live_run_id"]


def test_diagnostic_checkpoint_cannot_pass_as_formal(ident):
    diag = RunIdentity(
        run_id=ident.run_id, canonical_map=ident.canonical_map,
        resolved_map=ident.resolved_map, ruleset_id=ident.ruleset_id,
        ruleset_fingerprint=ident.ruleset_fingerprint, ruleset=ident.ruleset,
        formal_result_eligible=False, identity_override_used=True)
    p = payload_for(diag)
    with pytest.raises(RunIdentityError, match="formal_result_eligible"):
        verify_checkpoint_run_identity(p, ident, operation="load")
    # ...but is consistent within its own diagnostic universe.
    assert verify_checkpoint_run_identity(p, diag, operation="load")["match"]


def test_invalid_operation_rejected(ident):
    with pytest.raises(RunIdentityError, match="operation"):
        verify_checkpoint_run_identity(payload_for(ident), ident, operation="peek")


# --- model-execution sentinel ----------------------------------------------

class _Sentinel:
    """Fails loudly if weights are touched during a rejected load."""

    def __init__(self):
        self.load_state_dict_calls = 0
        self.forward_calls = 0

    def load_state_dict(self, *a, **k):
        self.load_state_dict_calls += 1

    def __call__(self, *a, **k):
        self.forward_calls += 1


def test_rejected_load_touches_no_weights(ident):
    """Identity must be checked BEFORE load_state_dict and any forward pass."""
    sentinel = _Sentinel()
    p = payload_for(ident)
    p[ARTIFACT_IDENTITY_KEY] = dict(p[ARTIFACT_IDENTITY_KEY], canonical_map="map_b")

    def load_checkpoint_into(model, payload, live_identity):
        verify_checkpoint_run_identity(payload, live_identity, operation="load")
        model.load_state_dict(payload["model_state_dict"])
        model()

    with pytest.raises(RunIdentityError):
        load_checkpoint_into(sentinel, p, ident)
    assert sentinel.load_state_dict_calls == 0, "weights loaded despite rejection"
    assert sentinel.forward_calls == 0, "forward pass ran despite rejection"


def test_accepted_load_does_touch_weights(ident):
    """Control: the sentinel would have caught a no-op verifier."""
    sentinel = _Sentinel()
    p = payload_for(ident)
    verify_checkpoint_run_identity(p, ident, operation="load")
    sentinel.load_state_dict(p["model_state_dict"])
    sentinel()
    assert sentinel.load_state_dict_calls == 1
    assert sentinel.forward_calls == 1


# --- manifest agreement -----------------------------------------------------

def test_checkpoint_identity_must_match_training_manifest(ident, env):
    from rl.ruleset_identity import stamp_json_artifact, validate_bundle

    other = build_formal_run_identity(env, run_id="a_different_run")
    manifest = stamp_json_artifact({"seeds": [1]}, ident)
    ck = payload_for(other)
    with pytest.raises(RunIdentityError, match="run_id differs"):
        validate_bundle({"training_manifest.json": manifest, "checkpoint": ck})


def test_checkpoint_and_manifest_agree_when_same_run(ident):
    from rl.ruleset_identity import stamp_json_artifact, validate_bundle

    manifest = stamp_json_artifact({"seeds": [1]}, ident)
    ck = payload_for(ident)
    ref = validate_bundle({"training_manifest.json": manifest, "checkpoint": ck})
    assert ref["run_id"] == ident.run_id
