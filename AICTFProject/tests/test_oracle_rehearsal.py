"""Tests for the oracle-gated rehearsal bank.

The sign convention is load-bearing: an inverted mapping would anchor each latent to
the wrong teacher and still produce plausible-looking differentiation, so it is
asserted here rather than trusted. Torch-free.
"""
from __future__ import annotations

import numpy as np
import pytest

from rl.oracle_rehearsal import (
    A_PREFERRED,
    B_PREFERRED,
    LATENT_FOR,
    NOT_ESTABLISHED,
    TEACHER_FOR,
    RehearsalBank,
    RehearsalError,
)


def _bank(deltas, rng_seed=31):
    n = len(deltas)
    d = np.array(deltas, dtype=np.int64)
    return RehearsalBank(
        obs={"grid": np.zeros((n, 2, 7, 20, 20), np.float32),
             "vec": np.zeros((n, 2, 20), np.float32),
             "agent_mask": np.ones((n, 2), np.float32),
             "mask": np.ones((n, 110), np.float32)},
        label=np.sign(d).astype(np.int64), delta=d,
        teacher_action=np.tile(np.arange(4), (n, 1)),
        cell=np.array(["A_r0_late"] * n), seed=np.zeros(n, np.int64),
        rng_seed=rng_seed)


# --------------------------------------------------------- sign convention

def test_sign_mapping_matches_the_frozen_convention():
    """Delta = M(pi_B) - M(pi_A), so Delta > 0 means pi_B scored higher."""
    assert A_PREFERRED == -1 and B_PREFERRED == 1 and NOT_ESTABLISHED == 0
    assert LATENT_FOR[A_PREFERRED] == 0 and TEACHER_FOR[A_PREFERRED] == "pi_A"
    assert LATENT_FOR[B_PREFERRED] == 1 and TEACHER_FOR[B_PREFERRED] == "pi_B"


def test_a_preferred_states_route_to_latent_zero():
    b = _bank([-1, -2, -3])
    z = b.sample(3)["z_idx"]
    assert set(z.tolist()) == {0}


def test_b_preferred_states_route_to_latent_one():
    b = _bank([1, 2, 3])
    assert set(b.sample(3)["z_idx"].tolist()) == {1}


# ------------------------------------------------- ties exert zero pressure

def test_tied_states_are_never_sampled():
    """The core invariant of the whole abstention arc."""
    b = _bank([0, 0, 0, -1, 1])
    for _ in range(50):
        batch = b.sample(2)
        assert NOT_ESTABLISHED not in batch["label"].tolist()
    b.assert_zero_tied_pressure()


def test_tied_states_are_loaded_but_excluded():
    """Loaded, not dropped -- so zero-pressure is MEASURED, not merely structural."""
    b = _bank([0, 0, 0, -1, 1])
    assert b.n_states == 5 and b.n_eligible == 2 and b.n_tied == 3


def test_zero_tied_pressure_assertion_can_actually_fail():
    """A guard that cannot fail proves nothing."""
    b = _bank([0, -1, 1])
    b.tied_exposures = 3
    with pytest.raises(RehearsalError, match="zero A/B pressure"):
        b.assert_zero_tied_pressure()


def test_bank_with_no_resolvable_states_is_refused():
    with pytest.raises(RehearsalError, match="no resolvable states"):
        _bank([0, 0, 0])


# ------------------------------------------------------------- composition

def test_composition_reports_the_thin_margin_limitation():
    """76.4% of the real bank turns on |Delta| = 1; that must be visible."""
    c = _bank([-1, 1, -2, 3, -1]).composition()
    assert c["eligible"] == 5 and c["abs_delta_1"] == 3
    assert c["abs_delta_1_frac"] == pytest.approx(0.6)


def test_composition_counts_both_directions():
    c = _bank([-1, -1, 1, 0]).composition()
    assert c["A_preferred"] == 2 and c["B_preferred"] == 1
    assert c["tied_excluded_from_sampling"] == 1


# ---------------------------------------------------------------- sampling

def test_sampling_is_uniform_not_adaptive():
    """Repeated draws must not concentrate; the spec prohibits hard-example mining."""
    b = _bank([-1] * 50 + [1] * 50)
    for _ in range(200):
        b.sample(10)
    counts = np.array([b.exposures.get(i, 0) for i in range(100)])
    assert counts.min() > 0
    assert counts.max() / counts.mean() < 2.0        # no runaway concentration


def test_sampling_never_exceeds_the_eligible_pool():
    b = _bank([-1, 1])
    assert len(b.sample(64)["index"]) == 2


def test_sampling_is_deterministic_for_a_given_seed():
    a = _bank([-1, 1, -2, 2, -3], rng_seed=7).sample(3)["index"]
    c = _bank([-1, 1, -2, 2, -3], rng_seed=7).sample(3)["index"]
    assert a.tolist() == c.tolist()


def test_different_seeds_give_different_draws():
    a = _bank([-1, 1, -2, 2, -3, 3, -1, 1], rng_seed=1).sample(3)["index"].tolist()
    c = _bank([-1, 1, -2, 2, -3, 3, -1, 1], rng_seed=2).sample(3)["index"].tolist()
    assert a != c


# --------------------------------------------------------------- telemetry

def test_telemetry_exposes_replay_factor():
    """Replay intensity is the primary validity threat, so it must be measured."""
    b = _bank([-1] * 10 + [1] * 10)
    for _ in range(10):
        b.sample(20)
    t = b.telemetry()
    assert t["total_exposures"] == 200
    assert t["replay_factor"] == pytest.approx(10.0)
    assert t["never_sampled"] == 0


def test_telemetry_tracks_per_latent_exposure():
    b = _bank([-1, -1, -1, 1])
    b.sample(4)
    t = b.telemetry()
    assert t["latent_exposures"]["z0"] == 3
    assert t["latent_exposures"]["z1"] == 1


def test_telemetry_reports_never_sampled_states():
    b = _bank([-1] * 10 + [1] * 10)
    b.sample(2)
    assert b.telemetry()["never_sampled"] == 18
