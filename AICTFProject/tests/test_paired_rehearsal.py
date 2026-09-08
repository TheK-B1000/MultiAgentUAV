"""OG-PSP paired rehearsal bank tests.

Stricter than V1's, because V1's smoke could not have detected V1's actual failure:
it verified both latents received *some* pressure, never that they received
*contradictory* pressure on the *same* state. Torch-free.
"""
from __future__ import annotations

import numpy as np
import pytest

from rl.paired_rehearsal import (
    LATENT_TO_TEACHER,
    PairedBank,
    PairedRehearsalError,
)

N_HEADS = 4


def _bank(deltas, *, pi_a=None, pi_b=None, rng_seed=37):
    n = len(deltas)
    a = np.tile(np.array([1, 1, 1, 1]), (n, 1)) if pi_a is None else np.asarray(pi_a)
    b = np.tile(np.array([2, 2, 2, 2]), (n, 1)) if pi_b is None else np.asarray(pi_b)
    return PairedBank(
        obs={"grid": np.zeros((n, 2, 7, 20, 20), np.float32),
             "vec": np.zeros((n, 2, 20), np.float32),
             "agent_mask": np.ones((n, 2), np.float32),
             "mask": np.ones((n, 110), np.float32)},
        delta=np.asarray(deltas, np.int64),
        pi_a_action=a, pi_b_action=b,
        cell=np.array(["A_r0_late"] * n), seed=np.zeros(n, np.int64), rng_seed=rng_seed)


# ------------------------------------------------- the core mechanism change

def test_same_state_yields_both_latent_targets():
    """The single thing V1 did not do."""
    b = _bank([-1, 1, -2])
    s = b.sample(3)
    assert s["n_states"] == 3 and s["n_pairs"] == 6
    # every sampled state appears exactly twice, once per latent
    for sid in set(s["state_id"].tolist()):
        rows = s["state_id"] == sid
        assert rows.sum() == 2
        assert sorted(s["z_idx"][rows].tolist()) == [0, 1]


def test_z0_always_gets_pi_A_and_z1_always_gets_pi_B():
    """Pairing is by latent identity, NEVER by which teacher is locally preferred."""
    b = _bank([-1, 1])                       # one A-preferred, one B-preferred
    s = b.sample(2)
    for i, sid in enumerate(s["state_id"]):
        z = s["z_idx"][i]
        expected = b.pi_a_action[sid] if z == 0 else b.pi_b_action[sid]
        assert np.array_equal(s["teacher_action"][i], expected)
    assert LATENT_TO_TEACHER == {0: "pi_A", 1: "pi_B"}


def test_locally_worse_specialist_is_still_taught():
    """On an A-preferred state z1 still receives pi_B -- the deliberate departure.

    If both latents chased the locally optimal teacher they would collapse into one
    adaptive generalist, which is exactly what V1 produced.
    """
    b = _bank([-5])                          # strongly A-preferred
    s = b.sample(1)
    z1_row = np.nonzero(s["z_idx"] == 1)[0][0]
    assert np.array_equal(s["teacher_action"][z1_row], b.pi_b_action[0])
    assert not np.array_equal(s["teacher_action"][z1_row], b.pi_a_action[0])


# ------------------------------------------------------ positive control

def test_positive_control_targets_actually_differ_where_teachers_disagree():
    """Guards against 'calls both paths but feeds effectively identical targets'."""
    b = _bank([-1, 1],
              pi_a=[[1, 1, 1, 1], [1, 1, 1, 1]],
              pi_b=[[9, 9, 9, 9], [9, 9, 9, 9]])
    s = b.sample(2)
    for sid in set(s["state_id"].tolist()):
        rows = np.nonzero(s["state_id"] == sid)[0]
        t0 = s["teacher_action"][rows[s["z_idx"][rows] == 0][0]]
        t1 = s["teacher_action"][rows[s["z_idx"][rows] == 1][0]]
        assert not np.array_equal(t0, t1), "paired targets must differ on a disagreement state"
    assert s["teachers_disagree"].all()


def test_identical_teachers_are_reported_as_no_contrast():
    """A state where the teachers agree supplies no differentiating signal."""
    b = _bank([-1, 1], pi_a=[[3, 3, 3, 3]] * 2, pi_b=[[3, 3, 3, 3]] * 2)
    assert not b.sample(2)["teachers_disagree"].any()
    assert b.composition()["teacher_disagreement_frac"] == 0.0


def test_disagreement_is_decision_masked():
    """Disagreement on a locked head is not usable supervision."""
    b = _bank([-1], pi_a=[[1, 1, 5, 5]], pi_b=[[1, 1, 9, 9]])
    b.obs["agent_mask"] = np.array([[1.0, 0.0]])     # second agent inactive
    assert not b.teachers_disagree(np.array([0]))[0], (
        "differences confined to masked heads must not count as contrast")
    b.obs["agent_mask"] = np.array([[0.0, 1.0]])     # now the differing heads are live
    assert b.teachers_disagree(np.array([0]))[0]


# --------------------------------------------------------- ties and invariants

def test_tied_states_are_never_sampled():
    b = _bank([0, 0, 0, -1, 1])
    for _ in range(50):
        assert not np.any(b.sample(2)["delta"] == 0)
    b.assert_invariants()


def test_tied_states_are_loaded_but_excluded():
    b = _bank([0, 0, -1, 1])
    assert b.n_states == 4 and b.n_eligible == 2
    assert b.composition()["tied_excluded_from_sampling"] == 2


def test_latent_exposure_must_stay_balanced():
    """Unequal exposure means the pairing broke -- the V1 defect returning."""
    b = _bank([-1, 1, -2])
    b.sample(3)
    t = b.telemetry()
    assert t["latent_exposures"]["z0"] == t["latent_exposures"]["z1"] == 3
    b.assert_invariants()


def test_invariant_check_can_actually_fail():
    """A guard that cannot fail proves nothing."""
    b = _bank([-1, 1])
    b.sample(2)
    b.latent_exposures[0] += 1                       # simulate a broken pairing
    with pytest.raises(PairedRehearsalError, match="expose both latents equally"):
        b.assert_invariants()
    b2 = _bank([0, -1, 1])
    b2.tied_exposures = 2
    with pytest.raises(PairedRehearsalError, match="zero strategic pressure"):
        b2.assert_invariants()


def test_empty_eligible_bank_is_refused():
    with pytest.raises(PairedRehearsalError, match="no resolvable states"):
        _bank([0, 0, 0])


# ------------------------------------------------------------- accounting

def test_replay_factor_counts_states_not_pairs():
    """2 targets per state must not be double-counted as 2x the replay."""
    b = _bank([-1] * 10 + [1] * 10)
    for _ in range(5):
        b.sample(20)
    t = b.telemetry()
    assert t["state_exposures_total"] == 100
    assert t["replay_factor"] == pytest.approx(5.0)
    assert t["latent_exposures"]["z0"] == 100


def test_sampling_is_uniform_and_deterministic_per_seed():
    a = _bank([-1, 1, -2, 2, -3, 3], rng_seed=11).sample(3)["state_id"][:3]
    c = _bank([-1, 1, -2, 2, -3, 3], rng_seed=11).sample(3)["state_id"][:3]
    assert a.tolist() == c.tolist()
    b = _bank([-1] * 40 + [1] * 40)
    for _ in range(100):
        b.sample(10)
    counts = np.array([b.state_exposures.get(i, 0) for i in range(80)])
    assert counts.min() > 0 and counts.max() / counts.mean() < 2.0


def test_composition_reports_available_contrast():
    b = _bank([-1, 1, -1], pi_a=[[1, 1, 1, 1]] * 3,
              pi_b=[[1, 1, 1, 1], [9, 9, 9, 9], [9, 9, 9, 9]])
    c = b.composition()
    assert c["eligible"] == 3
    assert c["eligible_with_teacher_disagreement"] == 2
    assert c["teacher_disagreement_frac"] == pytest.approx(2 / 3, abs=1e-4)
    assert c["targets_per_batch_state"] == 2
