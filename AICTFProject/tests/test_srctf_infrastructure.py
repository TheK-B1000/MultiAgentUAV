"""Tests for the C7-independent SRCTF infrastructure.

Content-free: no SRCTF map, opponent profile, rollout or qualification datum is
created or consumed anywhere in this file. Everything is exercised on synthetic
inputs so the machinery is proven before the science touches it.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from srctf import attempt, capacity, registry


# ---------------------------------------------------------------- registry --

@dataclasses.dataclass(frozen=True)
class FakeProfile:
    name: str
    knob: float = 1.0


@pytest.fixture(autouse=True)
def _clean_registry():
    registry._clear_for_tests()
    yield
    registry._clear_for_tests()


def _register(name="RAIDER", knob=1.0, traj="t" * 64):
    prof = FakeProfile(name=name, knob=knob)
    return prof, registry.register(
        name, prof,
        expected_profile_sha256=registry.profile_sha256(prof),
        expected_trajectory_sha256=traj)


def _admit_kw(prof, traj="t" * 64, *, canonical=None, dispatch=13, synonyms=()):
    return dict(
        canonicalize=lambda n: canonical if canonical is not None else n,
        resolve_profile=lambda n: prof,
        dispatch_level=lambda n: dispatch,
        trajectory_hash=lambda n: traj,
        synonym_tables=synonyms,
    )


def test_frozen_names_exact():
    assert registry.SRCTF_CANONICAL_OPPONENTS == {
        "RAIDER", "FORTRESS", "ESCORT", "SPLIT", "INTERCEPTOR", "ADAPTIVE"}


def test_turtle_is_not_a_name():
    """TURTLE is a reserved historical confusable: OP6_TURTLE is a dual rush."""
    assert "TURTLE" not in registry.SRCTF_CANONICAL_OPPONENTS


def test_names_carry_no_op_prefix_or_numbering():
    for n in registry.SRCTF_CANONICAL_OPPONENTS:
        assert not n.startswith("OP")
        assert not any(ch.isdigit() for ch in n)


def test_unknown_name_rejected_at_registration():
    with pytest.raises(registry.SRCTFRegistryError):
        registry.register("TURTLE", FakeProfile("TURTLE"),
                          expected_profile_sha256="x", expected_trajectory_sha256="y")


def test_admit_passes_when_all_six_hold():
    prof, _ = _register()
    registry.admit("RAIDER", **_admit_kw(prof))


def test_admit_fails_on_non_canonical_name():
    prof, _ = _register()
    with pytest.raises(registry.SRCTFRegistryError, match="canonical"):
        registry.admit("RAIDER", **_admit_kw(prof, canonical="SOMETHING_ELSE"))


def test_admit_fails_when_a_synonym_table_references_the_name():
    prof, _ = _register()
    with pytest.raises(registry.SRCTFRegistryError, match="synonym"):
        registry.admit("RAIDER", **_admit_kw(prof, synonyms=({"OLD_RAIDER": "RAIDER"},)))


def test_admit_fails_when_profile_hash_drifts():
    prof, _ = _register()
    drifted = FakeProfile(name="RAIDER", knob=2.0)   # same name, different knobs
    kw = _admit_kw(prof)
    kw["resolve_profile"] = lambda n: drifted
    with pytest.raises(registry.SRCTFRegistryError, match="profile hash"):
        registry.admit("RAIDER", **kw)


def test_admit_fails_when_profile_name_disagrees():
    prof, _ = _register()
    kw = _admit_kw(prof)
    kw["resolve_profile"] = lambda n: FakeProfile(name="FORTRESS")
    with pytest.raises(registry.SRCTFRegistryError, match="named"):
        registry.admit("RAIDER", **kw)


def test_admit_fails_when_it_never_reaches_the_bt_brain():
    """The C6 failure: config is perfect, the opponent is never driven."""
    prof, _ = _register()
    with pytest.raises(registry.SRCTFRegistryError, match="does NOT reach the BT brain"):
        registry.admit("RAIDER", **_admit_kw(prof, dispatch=None))


def test_admit_fails_when_behaviour_drifts():
    prof, _ = _register()
    kw = _admit_kw(prof)
    kw["trajectory_hash"] = lambda n: "z" * 64
    with pytest.raises(registry.SRCTFRegistryError, match="trajectory hash"):
        registry.admit("RAIDER", **kw)


def test_admit_all_fails_closed_on_a_partial_registry():
    prof, _ = _register()
    with pytest.raises(registry.SRCTFRegistryError, match="not registered"):
        registry.admit_all(**_admit_kw(prof))


# ---------------------------------------------------------------- capacity --

def _corr_pair(n, r_target, seed=0):
    """Synthesize two vectors with approximately the requested correlation."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = r_target * x + np.sqrt(max(0.0, 1 - r_target ** 2)) * z
    return x.tolist(), y.tolist()


import numpy as np  # noqa: E402  (after helper definition for readability)


def test_zero_variance_fails_closed():
    r = capacity.affordance_capacity([0.3] * 8, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    assert r.verdict == "AFFORDANCE_CAPACITY_FAIL"
    assert "zero variance" in r.reason


def test_zero_variance_in_contest_also_fails():
    r = capacity.affordance_capacity([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [0.5] * 8)
    assert r.verdict == "AFFORDANCE_CAPACITY_FAIL"


def test_too_few_opponents_fails_closed():
    assert capacity.affordance_capacity([0.1, 0.2], [0.3, 0.1]).verdict == \
        "AFFORDANCE_CAPACITY_FAIL"


def test_historical_cramped_correlation_fails():
    """The 2v2 baseline itself must NOT pass the gate it defines."""
    x, y = _corr_pair(60, -0.898, seed=1)
    r = capacity.affordance_capacity(x, y)
    assert r.verdict == "AFFORDANCE_CAPACITY_FAIL"
    assert not r.point_ok


def test_clearly_decoupled_population_passes():
    x, y = _corr_pair(60, -0.10, seed=2)
    r = capacity.affordance_capacity(x, y)
    assert r.verdict == "AFFORDANCE_CAPACITY_PASS"
    assert r.point_ok and r.ci_ok


def test_point_estimate_alone_is_insufficient():
    """A small noisy sample whose point estimate bounces upward must not pass."""
    x, y = _corr_pair(6, -0.60, seed=7)
    r = capacity.affordance_capacity(x, y)
    if r.point_ok:                       # the CI must be what stops it
        assert r.verdict == "AFFORDANCE_CAPACITY_FAIL"
        assert not r.ci_ok


def test_thresholds_are_the_frozen_ones():
    assert capacity.POINT_THRESHOLD == -0.70
    assert capacity.HISTORICAL_R == -0.898
    assert capacity.RESAMPLES == 2000 and capacity.BOOTSTRAP_SEED == 12345


def test_capacity_is_deterministic():
    x, y = _corr_pair(20, -0.5, seed=3)
    assert capacity.affordance_capacity(x, y) == capacity.affordance_capacity(x, y)


# ----------------------------------------------------------------- attempt --

def test_invalid_attempt_costs_nothing():
    o = attempt.invalid_attempt("srctf_v1", 9940000, evidence="dispatch guard failed")
    assert o.block_spent is False and o.rerun_allowed is True
    assert o.reportable == ()


def test_invalid_execution_spends_the_block_and_reports_nothing():
    o = attempt.invalid_execution("srctf_v1", 9940000,
                                  deterministic_test="tests/test_x.py::test_target_coords",
                                  shows_measured_quantity_wrong=True)
    assert o.block_spent is True
    assert o.rerun_allowed is False
    assert o.finding == "INCONCLUSIVE"
    assert o.reportable == ()


@pytest.mark.parametrize("claim", ["PASS", "FAIL", "NO_REVERSAL"])
def test_invalid_execution_may_not_report_anything(claim):
    o = attempt.invalid_execution("srctf_v1", 9940000, deterministic_test="t",
                                  shows_measured_quantity_wrong=True)
    with pytest.raises(attempt.AttemptError):
        attempt.assert_reportable(o, claim)


def test_invalid_execution_requires_a_deterministic_test():
    with pytest.raises(attempt.AttemptError, match="deterministic"):
        attempt.invalid_execution("srctf_v1", 9940000, deterministic_test="",
                                  shows_measured_quantity_wrong=True)


def test_a_defect_that_did_not_change_the_measurement_leaves_the_result_valid():
    with pytest.raises(attempt.AttemptError, match="MEASURED quantity"):
        attempt.invalid_execution("srctf_v1", 9940000, deterministic_test="t",
                                  shows_measured_quantity_wrong=False)


def test_valid_may_report_its_finding():
    o = attempt.valid("srctf_v1", 9940000, finding="NO_REVERSAL")
    attempt.assert_reportable(o, "NO_REVERSAL")
    assert o.block_spent and not o.rerun_allowed


def test_valid_rejects_an_unlisted_finding():
    with pytest.raises(attempt.AttemptError):
        attempt.valid("srctf_v1", 9940000, finding="INCONCLUSIVE")


def test_block_ledger_spending_is_irreversible(tmp_path):
    led = attempt.BlockLedger(tmp_path / "ledger.json")
    led.require_unspent(9940000)
    led.spend(9940000, design="srctf_v1", utc="2026-08-11T00:00:00Z")
    assert led.is_spent(9940000)
    with pytest.raises(attempt.AttemptError, match="already consumed"):
        led.require_unspent(9940000)
    with pytest.raises(attempt.AttemptError):
        led.spend(9940000, design="srctf_v1_retry", utc="2026-08-11T01:00:00Z")


def test_annotating_never_unspends(tmp_path):
    led = attempt.BlockLedger(tmp_path / "ledger.json")
    led.spend(9940000, design="srctf_v1", utc="u")
    led.annotate(9940000, state="INVALID_EXECUTION", finding="INCONCLUSIVE")
    assert led.is_spent(9940000)
    rec = json.loads((tmp_path / "ledger.json").read_text())["spent"]["9940000"]
    assert rec["state"] == "INVALID_EXECUTION"
