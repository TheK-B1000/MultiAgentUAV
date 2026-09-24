"""Regression tests for the exact failure in PI_B3_TRAIN_EVAL_POLE_MISMATCH_INVALIDATION.json.

pi_B3 trained against canonical Pole B (SDS_PARENT_OP7) while every evaluation
scored it against the certified B3-3 candidate (SDS2_B3_LOCKDEF10_2V1). The old
guard only demanded --pole-b-genome-json when the certification FILENAME
contained "CONFIRMATORY_REDESIGN"; the B3-3 record certifies a candidate genome
the same way but does not match that substring, so nothing fired. ~17.5
GPU-hours were spent answering the wrong experiment.

The rule these tests enforce: no experiment may start unless the live resolved
environment is proven identical to the governing certified configuration.
Launch arguments alone are never evidence of correctness.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.pole_attestation import (
    PoleAttestationError, assert_resolved_matches_certification, certified_pole,
    cross_policy_parity, pole_config_hash, resolve_pole_genome,
)

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
B3_CERT = SD / "STRATEGIC_DEMAND_4v4_POLE_B3_3_N192_CERTIFICATION.json"
B3_GENOME = SD / "pole_b2_candidates" / "B3-3_lockdef10_2v1.json"

pytestmark = pytest.mark.skipif(
    not (B3_CERT.is_file() and B3_GENOME.is_file()),
    reason="B3-3 certification / candidate genome not present")


# --------------------------------------------------------- 1: the actual bug --
def test_policy_B_without_certified_genome_override_fails_closed():
    """THE REGRESSION. This is the launch that was actually run and must now be refused."""
    genome = resolve_pole_genome("B", 4, None)          # canonical -- what the bad run used
    assert genome.genome_id == "SDS_PARENT_OP7"
    with pytest.raises(PoleAttestationError) as exc:
        assert_resolved_matches_certification("B", 4, B3_CERT, genome, is_smoke=False)
    msg = str(exc.value)
    assert "SDS2_B3_LOCKDEF10_2V1" in msg, "the refusal must name the genome that WAS certified"
    assert "pole-b-genome-json" in msg, "the refusal must tell the operator how to fix it"


# ------------------------------------------------------- 2: the wrong genome --
def test_policy_B_with_the_WRONG_certified_genome_fails_closed(tmp_path):
    """Passing *a* genome is not enough -- it must be the certified one. Uses the
    B2-1 candidate, a real genome from a superseded certification."""
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({
        "genome_id": "SDS2_B2_LOCKDEF10", "derived_from": "OP7", "base_opponent": "OP7",
        "overlay": {"lock_defender": 10}, "opening_hold_steps": 0}), encoding="utf-8")
    genome = resolve_pole_genome("B", 4, str(wrong))
    with pytest.raises(PoleAttestationError) as exc:
        assert_resolved_matches_certification("B", 4, B3_CERT, genome, is_smoke=False)
    assert "SDS2_B2_LOCKDEF10" in str(exc.value) and "SDS2_B3_LOCKDEF10_2V1" in str(exc.value)


# ------------------------------------------------------ 3: the correct genome --
def test_policy_B_with_the_certified_genome_passes():
    genome = resolve_pole_genome("B", 4, str(B3_GENOME))
    att = assert_resolved_matches_certification("B", 4, B3_CERT, genome, is_smoke=False)
    assert att["hashes_match"] is True
    assert att["live_genome_id"] == "SDS2_B3_LOCKDEF10_2V1"
    # the overlay fields the incident hinged on
    assert att["live_overlay"]["lock_defender"] == 10
    assert att["live_overlay"]["enable_2v1"] is True
    assert att["live_overlay"]["min_alive_for_defender"] == 4


# ------------------------------------ 4: A is unaffected by the B-only override --
def test_policy_A_still_passes_with_no_override():
    genome = resolve_pole_genome("A", 4, None)
    att = assert_resolved_matches_certification("A", 4, B3_CERT, genome, is_smoke=False)
    assert att["hashes_match"] is True
    assert att["certified_genome_id"] == "<canonical>"


def test_policy_A_rejects_a_pole_b_override_rather_than_ignoring_it():
    """A silently-ignored argument is how an operator ends up believing a run is
    something it is not."""
    with pytest.raises(PoleAttestationError, match="policy A"):
        resolve_pole_genome("A", 4, str(B3_GENOME))


# ----------------------------- 5: mismatch is caught before env / GPU work ----
def test_certification_mismatch_is_detected_without_building_an_environment(monkeypatch):
    """assert_resolved_matches_certification must reach its verdict from records and
    the resolved genome alone. If it ever constructed an env, this test's sabotage
    of the env factory would raise something other than PoleAttestationError."""
    import rl.training.env_factory as ef

    def _boom(*a, **k):
        raise AssertionError("environment was constructed during a pre-GPU check")

    monkeypatch.setattr(ef, "build_training_env", _boom, raising=False)
    genome = resolve_pole_genome("B", 4, None)
    with pytest.raises(PoleAttestationError):
        assert_resolved_matches_certification("B", 4, B3_CERT, genome, is_smoke=False)


def test_team_size_mismatch_fails_closed():
    """A 4v4 certification must not license a 6v6 run."""
    with pytest.raises(PoleAttestationError, match="team_size"):
        certified_pole(B3_CERT, "B", 6)


def test_certification_without_a_poles_block_fails_closed(tmp_path):
    """Absence is an error state: a record that cannot say what it certified is
    not a basis for spending GPU hours."""
    bare = tmp_path / "STRATEGIC_DEMAND_4v4_BARE_CERTIFICATION.json"
    bare.write_text(json.dumps({"VERDICT": "CERTIFIED", "team_size": 4}), encoding="utf-8")
    with pytest.raises(PoleAttestationError, match="poles"):
        certified_pole(bare, "B", 4)


# --------------------- 6: warm start / resume cannot swap the opponent ---------
def test_warm_start_cannot_silently_swap_the_opponent_definition():
    """A warm start re-resolves its pole and is attested like any other launch --
    inheriting weights from a checkpoint grants no exemption. The contaminated run
    was itself a warm start, so this is the path that actually failed."""
    canonical = resolve_pole_genome("B", 4, None)
    certified = resolve_pole_genome("B", 4, str(B3_GENOME))
    assert pole_config_hash("B", 4, canonical.genome_id, canonical.overlay) != \
           pole_config_hash("B", 4, certified.genome_id, certified.overlay), \
        "canonical and certified Pole B must not hash alike, or a swap would be invisible"
    with pytest.raises(PoleAttestationError):
        assert_resolved_matches_certification("B", 4, B3_CERT, canonical, is_smoke=False)
    ok = assert_resolved_matches_certification("B", 4, B3_CERT, certified, is_smoke=False)
    assert ok["hashes_match"] is True


# ------------------------------------------------- cross-policy parity check --
def test_cross_policy_parity_exposes_the_asymmetry_that_was_assumed_away():
    """The incident's root cause was assuming the A and B launches were symmetric.
    Preflight must state the asymmetry out loud."""
    parity = cross_policy_parity(4, B3_CERT)
    assert parity["A_requires_override"] is None
    assert parity["B_requires_override"] == "SDS2_B3_LOCKDEF10_2V1"
    assert "lock_defender" in parity["overlay_only_in_B"]
    assert "enable_2v1" in parity["overlay_only_in_B"]


def test_hash_is_deterministic_and_order_independent():
    a = pole_config_hash("B", 4, "X", {"lock_defender": 10, "enable_2v1": True})
    b = pole_config_hash("B", 4, "X", {"enable_2v1": True, "lock_defender": 10})
    assert a == b
    assert a != pole_config_hash("B", 4, "X", {"lock_defender": 11, "enable_2v1": True})
