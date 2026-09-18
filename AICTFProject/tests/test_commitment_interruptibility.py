"""Rule 12 known-answer contracts for INTERRUPT_CONDITION_R.

R is the entire scientific content of COMMITMENT_INTERRUPTIBILITY_INTERVENTION_V1.
If it fires more eagerly than specified the intervention degenerates into h=1, which
is the already-falsified experiment; if it fires less it cannot move the mechanism.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments import action_interface_scale_diagnostic as diag


GO_TO = 0


def _state(n: int = 1) -> diag.ArmState:
    return diag.ArmState(
        name="I2R_W50_INTERRUPTIBLE_COMMIT",
        core=None,
        x=None,
        y=None,
        heading=None,
        speed=None,
        r_streak=np.zeros(n, dtype=np.int32),
        current_run=np.zeros(n, dtype=np.int32),
    )


def _fire(state, *, requested_idx, committed_idx, desired, committed, active=True):
    return diag._interrupt_condition_r(
        state,
        np.asarray([[GO_TO, requested_idx]], dtype=np.int64),
        np.asarray([[GO_TO, committed_idx]], dtype=np.int64),
        np.asarray([desired], dtype=np.float64),
        np.asarray([committed], dtype=np.float64),
        np.asarray([active], dtype=bool),
    )


EAST = (1.0, 0.0)
NORTH = (0.0, 1.0)          # 90 degrees from EAST, clears theta=45
SLIGHT = (1.0, 0.36)        # ~20 degrees from EAST, below theta=45


@pytest.fixture(autouse=True)
def _repair_on():
    previous = diag.R_ENABLED
    diag.R_ENABLED = True
    yield
    diag.R_ENABLED = previous


def test_operating_point_is_the_frozen_one():
    assert diag.R_THETA_DEG == 45.0
    assert diag.R_K_TICKS == 2


def test_fires_only_after_k_consecutive_ticks():
    state = _state()
    first = _fire(state, requested_idx=7, committed_idx=3, desired=NORTH, committed=EAST)
    assert not first[0], "k=2 means a single qualifying tick must not interrupt"
    second = _fire(state, requested_idx=7, committed_idx=3, desired=NORTH, committed=EAST)
    assert second[0], "two consecutive qualifying ticks must interrupt"


def test_r1_alone_never_fires():
    """Target identity changed but the bearing barely moved."""
    state = _state()
    for _ in range(10):
        fires = _fire(state, requested_idx=7, committed_idx=3, desired=SLIGHT, committed=EAST)
        assert not fires[0]


def test_r2_alone_never_fires():
    """Bearing diverged but the committed action is still the one we would pick."""
    state = _state()
    for _ in range(10):
        fires = _fire(state, requested_idx=3, committed_idx=3, desired=NORTH, committed=EAST)
        assert not fires[0]


def test_streak_resets_when_condition_lapses():
    state = _state()
    _fire(state, requested_idx=7, committed_idx=3, desired=NORTH, committed=EAST)
    _fire(state, requested_idx=3, committed_idx=3, desired=NORTH, committed=EAST)  # lapse
    fires = _fire(state, requested_idx=7, committed_idx=3, desired=NORTH, committed=EAST)
    assert not fires[0], "a lapse must restart the k counter, not resume it"


def test_streak_resets_after_firing():
    state = _state()
    _fire(state, requested_idx=7, committed_idx=3, desired=NORTH, committed=EAST)
    assert _fire(state, requested_idx=7, committed_idx=3, desired=NORTH, committed=EAST)[0]
    assert not _fire(state, requested_idx=7, committed_idx=3, desired=NORTH, committed=EAST)[0]


def test_inactive_agent_never_fires_and_is_not_counted():
    state = _state()
    for _ in range(5):
        fires = _fire(
            state, requested_idx=7, committed_idx=3,
            desired=NORTH, committed=EAST, active=False,
        )
        assert not fires[0]
    assert state.r_eligible_ticks == 0, "inactive ticks must not enter the interruption-rate denominator"
    assert state.r_fire_count == 0


def test_exactly_at_theta_fires():
    """theta is a >= threshold; the boundary case must be inside, not outside."""
    state = _state()
    at_45 = (1.0, 1.0)
    _fire(state, requested_idx=7, committed_idx=3, desired=at_45, committed=EAST)
    assert _fire(state, requested_idx=7, committed_idx=3, desired=at_45, committed=EAST)[0]


def test_repair_off_never_fires():
    diag.R_ENABLED = False
    state = _state()
    for _ in range(20):
        assert not _fire(
            state, requested_idx=7, committed_idx=3, desired=NORTH, committed=EAST,
        )[0]
    assert state.r_fire_count == 0
    assert state.r_eligible_ticks == 0


def test_interruption_rate_denominator_counts_only_eligible_ticks():
    state = _state()
    for _ in range(4):
        _fire(state, requested_idx=7, committed_idx=3, desired=NORTH, committed=EAST)
    assert state.r_eligible_ticks == 4
    assert state.r_fire_count == 2


def test_run_length_tracking_matches_hand_count():
    state = _state()
    active = np.asarray([True], dtype=bool)
    schedule = [True, False, False, False, True, False, False]
    for boundary in schedule:
        diag._track_run_lengths(state, np.asarray([boundary], dtype=bool), active)
    assert state.run_lengths == [4], f"expected one closed run of 4 ticks, got {state.run_lengths}"
    assert int(state.current_run[0]) == 3


def test_dead_agent_does_not_accumulate_run_length():
    state = _state()
    diag._track_run_lengths(state, np.asarray([True], dtype=bool), np.asarray([False], dtype=bool))
    for _ in range(5):
        diag._track_run_lengths(
            state, np.asarray([False], dtype=bool), np.asarray([False], dtype=bool),
        )
    assert int(state.current_run[0]) == 0
    assert state.run_lengths == []
