"""Properties of the frozen full-window startup guard (GUARDED_ROUTED_COMPOSITION_CONFIRMATORY_V1_SPEC).

Simulator-free: these drive the router with feature sequences, so a logic error in
the guard shows up in seconds rather than after a contract stage or a two-hour run.
Torch-gated because the router lives in modules that import the CPU evaluation harness.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

pytest.importorskip("torch")

from experiments.run_routed_composition_outcome import (  # noqa: E402
    DEFAULT_COMPOSITION,
    DWELL,
    HYSTERESIS,
    THRESHOLD,
    TRIGGERED_COMPOSITION,
    WINDOW,
    OnlineRouter,
)
from experiments.size_routed_startup_guard import GuardedRouter  # noqa: E402

GUARD_TICK = WINDOW - 1


def _seq(router, vals):
    return [1 if router.update(v) == TRIGGERED_COMPOSITION else 0 for v in vals]


def test_first_departure_is_never_before_the_window_fills():
    r = GuardedRouter(guard=True)
    s = _seq(r, [-6.0] * 240)
    assert s.index(1) == GUARD_TICK
    assert r.first_block_tick == 0 and r.blocked_departures == GUARD_TICK


def test_guard_holds_on_random_sequences_and_is_inert_when_unguarded_departs_late():
    rng = np.random.default_rng(3)
    early = late = 0
    for _ in range(300):
        v = list(rng.normal(-0.6, 1.4, 240))
        g, u = _seq(GuardedRouter(guard=True), v), _seq(GuardedRouter(guard=False), v)
        if 1 in g:
            assert g.index(1) >= GUARD_TICK
        if 1 in u and u.index(1) >= GUARD_TICK:
            late += 1
            assert g == u, "guard altered a sequence whose first departure was already late"
        elif 1 in u:
            early += 1
    assert early > 0 and late > 0, "test would be vacuous on one branch"


def test_blocks_only_occur_before_the_window_fills():
    rng = np.random.default_rng(5)
    for _ in range(200):
        r = GuardedRouter(guard=True)
        _seq(r, list(rng.normal(-0.6, 1.4, 240)))
        if r.blocked_departures:
            assert r.first_block_tick < GUARD_TICK


def test_guard_disabled_is_the_frozen_router_exactly():
    rng = np.random.default_rng(9)
    for _ in range(100):
        v = list(rng.normal(-0.5, 1.5, 240))
        a, b = OnlineRouter(), GuardedRouter(guard=False)
        assert _seq(a, v) == _seq(b, v) and a.switches == b.switches


def test_release_is_never_blocked():
    fx = [-6.0] * 60 + [6.0] * 180
    g, u = _seq(GuardedRouter(guard=True), fx), _seq(GuardedRouter(guard=False), fx)

    def release(seq):
        seen = False
        for t, s in enumerate(seq):
            seen = seen or bool(s)
            if seen and not s:
                return t
        return None
    assert release(g) is not None and release(g) == release(u)
    assert g[GUARD_TICK:] == u[GUARD_TICK:]


def test_the_guard_reads_only_the_buffer_length_not_the_feature():
    """Two very different feature streams must be blocked at exactly the same ticks
    for as long as both want to depart: the guard's decision is a function of the
    tick index alone."""
    a, b = GuardedRouter(guard=True), GuardedRouter(guard=True)
    _seq(a, [-6.0] * 100)
    _seq(b, [-3.0] * 100)
    assert a.first_block_tick == b.first_block_tick == 0
    assert a.blocked_departures == b.blocked_departures == GUARD_TICK


def test_router_interface_admits_only_the_scalar_feature():
    params = set(inspect.signature(GuardedRouter.update).parameters) - {"self"}
    assert params == {"d_value"}


def test_first_switch_dwell_sentinel_and_operating_point_are_unchanged():
    r = GuardedRouter(guard=True)
    assert r._last_switch == -10_000
    assert (r.window, r.threshold, r.hysteresis, r.dwell) == (40, -0.8333333333333334, 0.2, 10)
    assert (WINDOW, THRESHOLD, HYSTERESIS, DWELL) == (40, -0.8333333333333334, 0.2, 10)


def test_default_composition_at_tick_zero():
    assert GuardedRouter(guard=True).update(-6.0) == DEFAULT_COMPOSITION
