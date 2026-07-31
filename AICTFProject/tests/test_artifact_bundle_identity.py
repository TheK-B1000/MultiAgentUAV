"""Artifact bundle identity: one run, one passport.

Identity is resolved ONCE from the live environment and handed to every writer.
Writers must never rebuild it from configuration defaults -- that is how five
artifacts end up carrying five subtly different versions of "V2".

The mutation suite is the load-bearing part: a validator that never rejects
anything is worthless, and this project has already produced five checkers that
failed to catch what they were built for.
"""
from __future__ import annotations

import copy
import json

import pytest

torch = pytest.importorskip("torch")

from gpu_env import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402
from rl.ruleset_identity import (  # noqa: E402
    ARTIFACT_IDENTITY_KEY,
    RunIdentity,
    RunIdentityError,
    build_formal_run_identity,
    ruleset_fingerprint_hash,
    stamp_csv_row,
    stamp_json_artifact,
    validate_bundle,
)

V2 = dict(taggers_required=1, tag_min_interval_seconds=10.0, tag_nearest_only=True,
          tag_channel_seconds=0.0, suppression_attackers_required=2)


@pytest.fixture
def env():
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2, map_set="train",
        map_layout="map_a", max_decision_steps=240, aquaticus_profile=True,
        rules_profile="OURS", device="cpu", seed=2_400_001,
        obstacle_obs_channel=True, **V2)
    e = GPUCTFVecEnv(cfg)
    e.reset()
    yield e
    e.close()


def build_bundle(identity: RunIdentity):
    """A tiny but complete artifact bundle, all stamped from ONE identity."""
    jsons = {
        "run_config.json": stamp_json_artifact({"total_steps": 50_000}, identity),
        "training_manifest.json": stamp_json_artifact({"seeds": [1, 2, 3]}, identity),
        "evaluation_manifest.json": stamp_json_artifact({"episodes": 32}, identity),
        "result_summary.json": stamp_json_artifact({"verdict": "SMOKE"}, identity),
        "checkpoint_payload": stamp_json_artifact({"global_step": 50_000}, identity),
    }
    rows = [stamp_csv_row({"episode_index": i, "env_index": 0,
                           "episode_id": i, "reset_sequence": i}, identity)
            for i in range(4)]
    return jsons, {"episode_rows.csv": rows}


# --- resolution -------------------------------------------------------------

def test_identity_resolves_from_live_env(env):
    ident = build_formal_run_identity(env)
    assert ident.canonical_map == "map_a"
    assert ident.resolved_map == "map_a_open"
    assert ident.ruleset_id == "RULESET_V2_AQUATICUS_10S"
    assert ident.formal_result_eligible is True
    assert ident.identity_override_used is False
    assert len(ident.ruleset_fingerprint) == 64
    assert ident.ruleset["taggers_required"] == 1
    assert ident.ruleset["tag_min_interval_seconds"] == 10.0


def test_no_silent_fallback_when_env_unresolvable():
    """A writer must never invent identity from defaults."""
    class NotAnEnv:
        pass

    with pytest.raises(RunIdentityError):
        build_formal_run_identity(NotAnEnv())


def test_writers_cannot_pass_a_config_instead_of_the_identity(env):
    with pytest.raises(RunIdentityError):
        stamp_json_artifact({}, env.core.cfg)      # type: ignore[arg-type]
    with pytest.raises(RunIdentityError):
        stamp_csv_row({}, dict(V2))                # type: ignore[arg-type]


# --- happy path -------------------------------------------------------------

def test_full_bundle_shares_one_passport(env):
    ident = build_formal_run_identity(env)
    jsons, csvs = build_bundle(ident)
    ref = validate_bundle(jsons, csvs)
    assert ref["run_id"] == ident.run_id
    assert ref["ruleset_fingerprint"] == ident.ruleset_fingerprint
    # every artifact agrees byte-for-byte
    fps = {j[ARTIFACT_IDENTITY_KEY]["ruleset_fingerprint"] for j in jsons.values()}
    assert len(fps) == 1
    ids = {j[ARTIFACT_IDENTITY_KEY]["run_id"] for j in jsons.values()}
    assert len(ids) == 1


def test_checkpoint_identity_matches_manifest(env):
    ident = build_formal_run_identity(env)
    jsons, _ = build_bundle(ident)
    ck = jsons["checkpoint_payload"][ARTIFACT_IDENTITY_KEY]
    mf = jsons["training_manifest.json"][ARTIFACT_IDENTITY_KEY]
    assert ck == mf


def test_every_csv_row_matches_bundle_identity(env):
    ident = build_formal_run_identity(env)
    _, csvs = build_bundle(ident)
    for r in csvs["episode_rows.csv"]:
        assert r["run_id"] == ident.run_id
        assert r["ruleset_fingerprint"] == ident.ruleset_fingerprint
        assert r["canonical_map"] == "map_a"
        assert r["resolved_map"] == "map_a_open"


# --- mutation suite: each must FAIL CLOSED ---------------------------------

def test_missing_stamp_rejected(env):
    ident = build_formal_run_identity(env)
    jsons, csvs = build_bundle(ident)
    del jsons["run_config.json"][ARTIFACT_IDENTITY_KEY]
    with pytest.raises(RunIdentityError):
        validate_bundle(jsons, csvs)


def test_missing_fingerprint_rejected(env):
    ident = build_formal_run_identity(env)
    jsons, csvs = build_bundle(ident)
    jsons["run_config.json"][ARTIFACT_IDENTITY_KEY]["ruleset_fingerprint"] = ""
    with pytest.raises(RunIdentityError):
        validate_bundle(jsons, csvs)


def test_mixed_fingerprints_rejected(env):
    ident = build_formal_run_identity(env)
    jsons, csvs = build_bundle(ident)
    jsons["evaluation_manifest.json"][ARTIFACT_IDENTITY_KEY][
        "ruleset_fingerprint"] = "0" * 64
    with pytest.raises(RunIdentityError, match="ruleset_fingerprint differs"):
        validate_bundle(jsons, csvs)


def test_mixed_run_ids_rejected(env):
    ident = build_formal_run_identity(env)
    jsons, csvs = build_bundle(ident)
    jsons["training_manifest.json"][ARTIFACT_IDENTITY_KEY]["run_id"] = "other-run"
    with pytest.raises(RunIdentityError, match="run_id differs"):
        validate_bundle(jsons, csvs)


def test_mixed_maps_rejected(env):
    ident = build_formal_run_identity(env)
    jsons, csvs = build_bundle(ident)
    jsons["result_summary.json"][ARTIFACT_IDENTITY_KEY]["canonical_map"] = "map_b"
    with pytest.raises(RunIdentityError, match="canonical_map differs"):
        validate_bundle(jsons, csvs)


def test_single_mismatched_csv_row_rejected(env):
    """One bad row out of many must sink the bundle."""
    ident = build_formal_run_identity(env)
    jsons, csvs = build_bundle(ident)
    csvs["episode_rows.csv"][2]["ruleset_fingerprint"] = "deadbeef" * 8
    with pytest.raises(RunIdentityError, match="row 2"):
        validate_bundle(jsons, csvs)


def test_concatenated_foreign_rows_rejected(env):
    """Accidental concatenation of another run's CSV must be caught."""
    ident = build_formal_run_identity(env)
    jsons, csvs = build_bundle(ident)
    foreign = copy.deepcopy(csvs["episode_rows.csv"][0])
    foreign["run_id"] = "some-other-run"
    csvs["episode_rows.csv"].append(foreign)
    with pytest.raises(RunIdentityError, match="run_id mismatch"):
        validate_bundle(jsons, csvs)


def test_v1_artifact_inserted_is_rejected(env):
    ident = build_formal_run_identity(env)
    jsons, csvs = build_bundle(ident)
    v1_fields = {"ruleset_id": "RULESET_V1_TWO_TAGGER", "taggers_required": 2,
                 "tag_min_interval_seconds": 0.0, "tag_nearest_only": False,
                 "tag_channel_seconds": 1.0, "suppression_attackers_required": 2}
    jsons["v1_intruder.json"] = {ARTIFACT_IDENTITY_KEY: {
        "run_id": ident.run_id, "canonical_map": "map_a",
        "resolved_map": "map_a_open", "ruleset_id": "RULESET_V1_TWO_TAGGER",
        "ruleset_fingerprint": ruleset_fingerprint_hash(v1_fields),
        "formal_result_eligible": True}}
    with pytest.raises(RunIdentityError):
        validate_bundle(jsons, csvs)


def test_legacy_unstamped_checkpoint_rejected(env):
    ident = build_formal_run_identity(env)
    jsons, csvs = build_bundle(ident)
    jsons["legacy_checkpoint"] = {"global_step": 1_000_000}   # no identity at all
    with pytest.raises(RunIdentityError, match="missing"):
        validate_bundle(jsons, csvs)


def test_changed_ruleset_field_with_same_id_rejected(env):
    """The label alone must never be sufficient."""
    ident = build_formal_run_identity(env)
    jsons, csvs = build_bundle(ident)
    tampered = dict(ident.ruleset)
    tampered["tag_min_interval_seconds"] = 30.0      # id unchanged, value differs
    jsons["run_config.json"][ARTIFACT_IDENTITY_KEY]["ruleset_fingerprint"] = \
        ruleset_fingerprint_hash(tampered)
    with pytest.raises(RunIdentityError, match="ruleset_fingerprint differs"):
        validate_bundle(jsons, csvs)


def test_diagnostic_override_accepted_only_as_ineligible(env):
    """Override may exist, but it can never masquerade as a formal result."""
    ident = build_formal_run_identity(env)
    override = RunIdentity(
        run_id=ident.run_id, canonical_map=ident.canonical_map,
        resolved_map=ident.resolved_map, ruleset_id=ident.ruleset_id,
        ruleset_fingerprint=ident.ruleset_fingerprint, ruleset=ident.ruleset,
        formal_result_eligible=False, identity_override_used=True)
    jsons, csvs = build_bundle(override)
    # Rejected for a formal run...
    with pytest.raises(RunIdentityError):
        validate_bundle(jsons, csvs, require_formal=True)
    # ...but readable as an explicitly ineligible diagnostic bundle.
    ref = validate_bundle(jsons, csvs, require_formal=False)
    assert ref["formal_result_eligible"] is False
    assert ref["identity_override_used"] is True


def test_empty_bundle_rejected():
    with pytest.raises(RunIdentityError):
        validate_bundle({})


def test_fingerprint_is_order_independent_but_value_sensitive():
    a = {"ruleset_id": "X", "taggers_required": 1, "tag_min_interval_seconds": 10.0,
         "tag_nearest_only": True, "tag_channel_seconds": 0.0,
         "suppression_attackers_required": 2}
    b = dict(reversed(list(a.items())))
    assert ruleset_fingerprint_hash(a) == ruleset_fingerprint_hash(b)
    c = dict(a, taggers_required=2)
    assert ruleset_fingerprint_hash(a) != ruleset_fingerprint_hash(c)
