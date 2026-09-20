"""The composition instrument, pinned on synthetic traces with a known answer.

Simulator-free. The instrument reads only positions and status flags, so a trace
built by hand fixes exactly what it must report.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from experiments.probe_learned_composition import (  # noqa: E402
    N_AGENTS,
    home_distance,
    instrument,
    instrument_reference,
)

R = 6.0
HOME = np.array([2.0, 10.0], dtype=np.float32)


def _trace(T=4, near=(), carrying=(), tagged=(), dead=()):
    """Agents in `near` sit 1 cell from home; every other agent sits 20 cells away."""
    pos = np.tile(np.array([22.0, 10.0], np.float32), (T, N_AGENTS, 1))
    for i in near:
        pos[:, i] = HOME + np.array([1.0, 0.0], np.float32)
    def mask(idx):
        m = np.zeros((T, N_AGENTS), bool); m[:, list(idx)] = True; return m
    alive = ~mask(dead) if dead else np.ones((T, N_AGENTS), bool)
    return {"pos": pos, "alive": alive, "tagged": mask(tagged), "carrying": mask(carrying),
            "flag_home": np.tile(HOME, (T, 1)), "flag_pos": np.tile(HOME, (T, 1))}


@pytest.mark.parametrize("d", range(0, 7))
def test_recovers_the_number_of_home_agents(d):
    out = instrument(_trace(near=range(d)), R)
    assert (out["k"] == d).all() and (out["n_active"] == 6).all()
    assert np.allclose(out["share"], d / 6)


def test_a_carrier_near_home_is_not_defend_like():
    """A carrier returning to score passes through home; that is an attack, not a defence."""
    out = instrument(_trace(near=(0, 1), carrying=(0,)), R)
    assert (out["k"] == 1).all()


def test_tagged_and_dead_agents_are_excluded_from_both_counts():
    out = instrument(_trace(near=(0, 1, 2), tagged=(0,), dead=(1,)), R)
    assert (out["k"] == 1).all()
    assert (out["n_active"] == 4).all()                       # 6 - 1 tagged - 1 dead
    assert np.allclose(out["share"], 1 / 4)


def test_share_is_nan_when_nobody_is_active():
    out = instrument(_trace(tagged=range(6)), R)
    assert (out["n_active"] == 0).all() and np.isnan(out["share"]).all()


def test_the_radius_boundary_is_inclusive():
    t = _trace()
    t["pos"][:, 0] = HOME + np.array([R, 0.0], np.float32)
    assert (instrument(t, R)["k"] == 1).all()
    t["pos"][:, 0] = HOME + np.array([R + 1e-3, 0.0], np.float32)
    assert (instrument(t, R)["k"] == 0).all()


def test_distance_is_measured_from_the_flags_HOME_not_its_current_position():
    """A dropped or stolen flag must not move the reference point."""
    t = _trace(near=(0,))
    t["flag_pos"] = np.tile(np.array([30.0, 10.0], np.float32), (4, 1))
    assert (instrument(t, R)["k"] == 1).all()


def test_vectorised_and_reference_implementations_agree_on_random_traces():
    rng = np.random.default_rng(5)
    for _ in range(25):
        T = int(rng.integers(3, 30))
        tr = {"pos": rng.uniform(0, 40, size=(T, N_AGENTS, 2)).astype(np.float32),
              "alive": rng.random((T, N_AGENTS)) > 0.1, "tagged": rng.random((T, N_AGENTS)) > 0.8,
              "carrying": rng.random((T, N_AGENTS)) > 0.85,
              "flag_home": np.tile(HOME, (T, 1)), "flag_pos": np.tile(HOME, (T, 1))}
        a, b = instrument(tr, R), instrument_reference(tr, R)
        assert np.array_equal(a["k"], b["k"]) and np.array_equal(a["n_active"], b["n_active"])
        assert np.allclose(a["share"], b["share"], equal_nan=True)


def test_home_distance_shape_and_value():
    d = home_distance(_trace(near=(0,)))
    assert d.shape == (4, N_AGENTS)
    assert np.allclose(d[:, 0], 1.0)            # near agent: 1 cell from home
    assert np.allclose(d[:, 1], 20.0)           # far agent at (22,10), home at (2,10)


# --------------------------------------------------------------------------
# The intent instrument: reads the RESOLVED TARGET, not position or macro id.
# --------------------------------------------------------------------------
from experiments.probe_learned_composition import (  # noqa: E402
    R_INTENT,
    intent_instrument,
    intent_instrument_reference,
)


def _itrace(T=4, home_directed=(), carrying=(), tagged=(), dead=()):
    """Every agent SITS far from home; only the intent differs. Agents in home_directed are
    being sent to the flag, everyone else to the enemy flag at (17, 10)."""
    t = _trace(T=T, carrying=carrying, tagged=tagged, dead=dead)
    intent = np.tile(np.array([17.0, 10.0], np.float32), (T, N_AGENTS, 1))
    for i in home_directed:
        intent[:, i] = HOME
    t["intent"] = intent
    return t


@pytest.mark.parametrize("d", range(0, 7))
def test_intent_counts_home_directed_agents_regardless_of_where_they_stand(d):
    out = intent_instrument(_itrace(home_directed=range(d)))
    assert (out["k"] == d).all() and np.allclose(out["share"], d / 6)


def test_position_is_ignored_by_the_intent_instrument():
    """An agent standing AT home but being sent to the enemy flag is attack-directed."""
    t = _itrace()
    t["pos"][:, 0] = HOME + np.array([0.5, 0.0], np.float32)
    assert (intent_instrument(t)["k"] == 0).all()


def test_a_carrier_is_never_home_directed_even_though_its_target_is_home():
    out = intent_instrument(_itrace(home_directed=(0, 1), carrying=(0,)))
    assert (out["k"] == 1).all()


def test_tagged_and_dead_agents_are_excluded():
    out = intent_instrument(_itrace(home_directed=(0, 1, 2), tagged=(0,), dead=(1,)))
    assert (out["k"] == 1).all() and (out["n_active"] == 4).all()


def test_the_target_radius_boundary_is_inclusive():
    t = _itrace()
    t["intent"][:, 0] = HOME + np.array([R_INTENT, 0.0], np.float32)
    assert (intent_instrument(t)["k"] == 1).all()
    t["intent"][:, 0] = HOME + np.array([R_INTENT + 1e-3, 0.0], np.float32)
    assert (intent_instrument(t)["k"] == 0).all()


def test_state_masks_are_reported_for_slicing():
    out = intent_instrument(_itrace(carrying=(3,), tagged=(4,)))
    assert out["any_carry"].all() and out["any_tagged"].all()
    clean = intent_instrument(_itrace())
    assert not clean["any_carry"].any() and not clean["any_tagged"].any()


def test_intent_vectorised_and_reference_implementations_agree_on_random_traces():
    rng = np.random.default_rng(9)
    for _ in range(25):
        T = int(rng.integers(3, 30))
        tr = {"pos": rng.uniform(0, 20, size=(T, N_AGENTS, 2)).astype(np.float32),
              "intent": rng.uniform(0, 20, size=(T, N_AGENTS, 2)).astype(np.float32),
              "alive": rng.random((T, N_AGENTS)) > 0.1, "tagged": rng.random((T, N_AGENTS)) > 0.8,
              "carrying": rng.random((T, N_AGENTS)) > 0.85,
              "flag_home": np.tile(HOME, (T, 1)), "flag_pos": np.tile(HOME, (T, 1))}
        a, b = intent_instrument(tr), intent_instrument_reference(tr)
        assert np.array_equal(a["k"], b["k"]) and np.array_equal(a["n_active"], b["n_active"])
        assert np.allclose(a["share"], b["share"], equal_nan=True)


def test_r_intent_is_the_predeclared_value():
    assert R_INTENT == 4.5


# --------------------------------------------------------------------------
# Role assignments: PREFIX (all prior sweeps) versus SUFFIX (the id-geometry control).
# --------------------------------------------------------------------------
from experiments.probe_learned_composition import roles_for  # noqa: E402

COMPS7 = ["6A_0D", "5A_1D", "4A_2D", "3A_3D", "2A_4D", "1A_5D", "0A_6D"]


@pytest.mark.parametrize("comp", COMPS7)
def test_both_assignments_field_the_same_number_of_defenders(comp):
    d = int(comp.split("_")[1][:-1])
    assert sum(roles_for(comp, "PREFIX")) == d and sum(roles_for(comp, "SUFFIX")) == d


def test_prefix_matches_the_frozen_sweep_helper():
    from experiments.run_pyquaticus_6v6_role_composition_sweep import composition_roles_n
    for c in COMPS7:
        assert roles_for(c, "PREFIX") == composition_roles_n(c, 6)


def test_suffix_places_defenders_on_the_last_ids():
    assert roles_for("5A_1D", "SUFFIX") == (0, 0, 0, 0, 0, 1)
    assert roles_for("4A_2D", "SUFFIX") == (0, 0, 0, 0, 1, 1)
    assert roles_for("1A_5D", "SUFFIX") == (0, 1, 1, 1, 1, 1)


def test_the_two_assignments_coincide_only_at_the_extremes():
    same = [c for c in COMPS7 if roles_for(c, "PREFIX") == roles_for(c, "SUFFIX")]
    assert same == ["6A_0D", "0A_6D"]


def test_unknown_assignment_is_rejected():
    with pytest.raises(ValueError):
        roles_for("3A_3D", "MIDDLE")
