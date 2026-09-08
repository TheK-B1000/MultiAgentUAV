"""Evaluation-manifest checkpoint lineage: no self-fallback.

The original writer defaulted lineage to the evaluation run's own identity:

    source_training_run_id or identity.run_id
    source_checkpoint_ruleset_fingerprint or identity.ruleset_fingerprint
    source_checkpoint_fingerprint or ""

An omitted value therefore produced a manifest asserting the checkpoint matched
itself -- a border check comparing a passport to its own reflection, which can
never detect a mismatch.

Lineage is now a validated object constructible only by the compatibility
verifier, and a formal manifest cannot be written without one. A diagnostic
evaluation records explicit nulls: null is honest, self-fallback is camouflage.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from gpu_env import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402
from rl.ruleset_identity import (  # noqa: E402
    RunIdentity,
    RunIdentityError,
    VerifiedCheckpointLineage,
    build_formal_run_identity,
    fingerprint,
    ruleset_fingerprint_hash,
    verify_checkpoint_lineage,
)
from rl.training.run_artifacts import write_evaluation_manifest_json  # noqa: E402

V2 = dict(taggers_required=1, tag_min_interval_seconds=10.0, tag_nearest_only=True,
          tag_channel_seconds=0.0, suppression_attackers_required=2)
V1 = dict(taggers_required=2, tag_min_interval_seconds=0.0, tag_nearest_only=False,
          tag_channel_seconds=1.0, suppression_attackers_required=2)


@pytest.fixture
def env():
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2, map_set="train",
        map_layout="map_a", max_decision_steps=240, aquaticus_profile=True,
        rules_profile="OURS", device="cpu", seed=2_600_001,
        obstacle_obs_channel=True, **V2)
    e = GPUCTFVecEnv(cfg)
    e.reset()
    yield e
    e.close()


@pytest.fixture
def ckpt(tmp_path, env):
    """A checkpoint file whose ruleset matches the live environment."""
    p = tmp_path / "v2_ckpt.zip"
    torch.save({"model_state_dict": {}, "ruleset": fingerprint(env.core.cfg)}, p)
    return p


def v2_fields(env):
    return fingerprint(env.core.cfg)


def good_lineage(env, ckpt):
    return verify_checkpoint_lineage(
        checkpoint_path=str(ckpt),
        identity=build_formal_run_identity(env, run_id="eval_run"),
        checkpoint_ruleset=v2_fields(env),
        source_training_run_id="training_run_abc")


# --- formal path ------------------------------------------------------------

def test_formal_evaluation_with_verified_lineage_is_accepted(tmp_path, env, ckpt):
    ident = build_formal_run_identity(env, run_id="eval_run")
    lin = good_lineage(env, ckpt)
    path = write_evaluation_manifest_json(
        str(tmp_path / "evaluation_manifest.json"),
        run_identity=ident, evaluation_run_id="eval_run", lineage=lin)
    m = json.loads(open(path, encoding="utf-8").read())
    assert m["checkpoint_lineage_complete"] is True
    assert m["source_training_run_id"] == "training_run_abc"
    assert len(m["source_checkpoint_fingerprint"]) == 64
    assert m["source_checkpoint_ruleset_fingerprint"] == ident.ruleset_fingerprint
    # the discovered-defect guard
    assert m["source_training_run_id"] != m["evaluation_run_id"]


def test_formal_evaluation_without_lineage_is_rejected(tmp_path, env):
    ident = build_formal_run_identity(env, run_id="eval_run")
    with pytest.raises(RunIdentityError, match="VerifiedCheckpointLineage"):
        write_evaluation_manifest_json(
            str(tmp_path / "m.json"), run_identity=ident, lineage=None)
    assert not (tmp_path / "m.json").exists()


def test_loose_strings_cannot_stand_in_for_verified_lineage(tmp_path, env):
    """A caller must not fabricate lineage from plain values."""
    ident = build_formal_run_identity(env, run_id="eval_run")
    with pytest.raises(RunIdentityError):
        write_evaluation_manifest_json(
            str(tmp_path / "m.json"), run_identity=ident,
            lineage={"source_training_run_id": "x",
                     "source_checkpoint_fingerprint": "y",
                     "source_checkpoint_ruleset_fingerprint": "z"})


# --- verifier rejections ----------------------------------------------------

def test_missing_training_run_id_rejected(env, ckpt):
    with pytest.raises(RunIdentityError, match="source_training_run_id is required"):
        verify_checkpoint_lineage(
            checkpoint_path=str(ckpt),
            identity=build_formal_run_identity(env, run_id="eval_run"),
            checkpoint_ruleset=v2_fields(env), source_training_run_id="")


def test_checkpoint_ruleset_mismatch_rejected(env, ckpt):
    with pytest.raises(RunIdentityError, match="does not match the live evaluation"):
        verify_checkpoint_lineage(
            checkpoint_path=str(ckpt),
            identity=build_formal_run_identity(env, run_id="eval_run"),
            checkpoint_ruleset=dict(V1, ruleset_id="RULESET_V1_TWO_TAGGER"),
            source_training_run_id="training_run_abc")


def test_legacy_checkpoint_without_ruleset_rejected(env, ckpt):
    with pytest.raises(RunIdentityError, match="LEGACY_UNKNOWN"):
        verify_checkpoint_lineage(
            checkpoint_path=str(ckpt),
            identity=build_formal_run_identity(env, run_id="eval_run"),
            checkpoint_ruleset=None, source_training_run_id="training_run_abc")


def test_missing_checkpoint_file_rejected(tmp_path, env):
    with pytest.raises(RunIdentityError, match="file not found"):
        verify_checkpoint_lineage(
            checkpoint_path=str(tmp_path / "nope.zip"),
            identity=build_formal_run_identity(env, run_id="eval_run"),
            checkpoint_ruleset=v2_fields(env),
            source_training_run_id="training_run_abc")


def test_verifier_refuses_for_a_diagnostic_identity(env, ckpt):
    """The verifier is for formal runs; diagnostics record explicit nulls."""
    base = build_formal_run_identity(env, run_id="eval_run")
    diag = RunIdentity(
        run_id=base.run_id, canonical_map=base.canonical_map,
        resolved_map=base.resolved_map, ruleset_id=base.ruleset_id,
        ruleset_fingerprint=base.ruleset_fingerprint, ruleset=base.ruleset,
        formal_result_eligible=False, identity_override_used=True)
    with pytest.raises(RunIdentityError, match="diagnostic"):
        verify_checkpoint_lineage(
            checkpoint_path=str(ckpt), identity=diag,
            checkpoint_ruleset=v2_fields(env),
            source_training_run_id="training_run_abc")


# --- diagnostic path --------------------------------------------------------

def test_diagnostic_override_records_explicit_nulls(tmp_path, env):
    base = build_formal_run_identity(env, run_id="eval_run")
    diag = RunIdentity(
        run_id=base.run_id, canonical_map=base.canonical_map,
        resolved_map=base.resolved_map, ruleset_id=base.ruleset_id,
        ruleset_fingerprint=base.ruleset_fingerprint, ruleset=base.ruleset,
        formal_result_eligible=False, identity_override_used=True)
    path = write_evaluation_manifest_json(
        str(tmp_path / "diag.json"), run_identity=diag, lineage=None)
    m = json.loads(open(path, encoding="utf-8").read())
    assert m["source_training_run_id"] is None
    assert m["source_checkpoint_fingerprint"] is None
    assert m["source_checkpoint_ruleset_fingerprint"] is None
    assert m["checkpoint_lineage_complete"] is False
    # It must never masquerade as formal.
    ai = m["artifact_identity"]
    assert ai["formal_result_eligible"] is False
    assert ai["identity_override_used"] is True


def test_diagnostic_manifest_never_validates_as_formal(tmp_path, env):
    from rl.ruleset_identity import validate_bundle

    base = build_formal_run_identity(env, run_id="eval_run")
    diag = RunIdentity(
        run_id=base.run_id, canonical_map=base.canonical_map,
        resolved_map=base.resolved_map, ruleset_id=base.ruleset_id,
        ruleset_fingerprint=base.ruleset_fingerprint, ruleset=base.ruleset,
        formal_result_eligible=False, identity_override_used=True)
    path = write_evaluation_manifest_json(
        str(tmp_path / "diag.json"), run_identity=diag, lineage=None)
    m = json.loads(open(path, encoding="utf-8").read())
    with pytest.raises(RunIdentityError):
        validate_bundle({"evaluation_manifest.json": m}, require_formal=True)


def test_self_fallback_pattern_is_gone(tmp_path, env, ckpt):
    """Regression guard for the exact discovered defect."""
    ident = build_formal_run_identity(env, run_id="eval_run")
    lin = good_lineage(env, ckpt)
    path = write_evaluation_manifest_json(
        str(tmp_path / "m.json"), run_identity=ident,
        evaluation_run_id="eval_run", lineage=lin)
    m = json.loads(open(path, encoding="utf-8").read())
    assert m["source_training_run_id"] != m["evaluation_run_id"], (
        "lineage must come from the checkpoint, not the evaluation run")
    assert m["source_checkpoint_fingerprint"] != ident.ruleset_fingerprint


# --- in-training evaluation -------------------------------------------------

def test_in_training_lineage_requires_a_real_checkpoint_fingerprint(env):
    ident = build_formal_run_identity(env, run_id="train_run")
    with pytest.raises(RunIdentityError, match="real checkpoint fingerprint"):
        VerifiedCheckpointLineage.for_in_training_evaluation(ident, "")


def test_in_training_lineage_is_explicit_not_inferred(tmp_path, env):
    """Shared run id is legitimate in-training, but must be a declared claim."""
    ident = build_formal_run_identity(env, run_id="train_run")
    lin = VerifiedCheckpointLineage.for_in_training_evaluation(ident, "a" * 64)
    path = write_evaluation_manifest_json(
        str(tmp_path / "m.json"), run_identity=ident,
        evaluation_run_id=ident.run_id, lineage=lin, extra={"scope": "in_training"})
    m = json.loads(open(path, encoding="utf-8").read())
    assert m["source_training_run_id"] == m["evaluation_run_id"] == "train_run"
    assert m["checkpoint_lineage_complete"] is True
    assert m["source_checkpoint_fingerprint"] == "a" * 64
    # Still impossible to get here by omitting the argument.
    with pytest.raises(RunIdentityError):
        write_evaluation_manifest_json(
            str(tmp_path / "m2.json"), run_identity=ident,
            evaluation_run_id=ident.run_id, lineage=None)


def test_in_training_constructor_refuses_diagnostic_identity(env):
    base = build_formal_run_identity(env, run_id="train_run")
    diag = RunIdentity(
        run_id=base.run_id, canonical_map=base.canonical_map,
        resolved_map=base.resolved_map, ruleset_id=base.ruleset_id,
        ruleset_fingerprint=base.ruleset_fingerprint, ruleset=base.ruleset,
        formal_result_eligible=False, identity_override_used=True)
    with pytest.raises(RunIdentityError, match="diagnostic"):
        VerifiedCheckpointLineage.for_in_training_evaluation(diag, "b" * 64)
