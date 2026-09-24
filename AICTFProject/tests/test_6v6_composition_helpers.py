"""C2 parity: the n-agent composition helpers must reproduce the frozen 4v4 ones.

Simulator-free half of the contract. The live tick-by-tick action-adapter parity
runs in the contract stage; these catch a logic error in seconds instead.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from experiments.run_pyquaticus_4v4_role_composition_sweep import (  # noqa: E402
    COMPOSITIONS as COMPOSITIONS_4V4,
    composition_roles as composition_roles_4v4,
)
from experiments.run_pyquaticus_6v6_role_composition_sweep import (  # noqa: E402
    BASELINE,
    COMPOSITIONS,
    N_AGENTS,
    action_for_roles_n,
    composition_roles_n,
)


def test_role_mapping_matches_the_frozen_4v4_helper():
    assert len(COMPOSITIONS_4V4) == 5
    for c in COMPOSITIONS_4V4:
        assert composition_roles_n(c, 4) == composition_roles_4v4(c), c


def test_six_compositions_are_exhaustive_and_prefix_ordered():
    assert len(COMPOSITIONS) == 7 and BASELINE == "3A_3D"
    seen_d = set()
    for c in COMPOSITIONS:
        roles = composition_roles_n(c, N_AGENTS)
        d = int(c.split("_")[1][:-1])
        seen_d.add(d)
        assert len(roles) == N_AGENTS
        assert roles[:d] == (1,) * d and roles[d:] == (0,) * (N_AGENTS - d)
    assert seen_d == set(range(7)), "defender counts must cover 0..6 exhaustively"


def test_composition_that_does_not_cover_the_team_is_rejected():
    with pytest.raises(ValueError):
        composition_roles_n("2A_2D", 6)
    with pytest.raises(ValueError):
        composition_roles_n("3A_3D", 4)


class _FakeCore:
    """Minimal stand-in exposing only what the action adapter reads."""

    def __init__(self, carrying: bool):
        self._carrying = carrying

    class _Arr:
        def __init__(self, v): self._v = v
        def any(self): return self
        def item(self): return self._v

    @property
    def blue_carrying(self):
        return {0: _FakeCore._Arr(self._carrying)}


def test_action_adapter_matches_the_frozen_helper_on_both_carrier_branches():
    from experiments.run_pyquaticus_4v4_team_evaluation import _action_for_roles as frozen
    for carrying in (False, True):
        core = _FakeCore(carrying)
        for c in COMPOSITIONS_4V4:
            roles = composition_roles_4v4(c)
            assert np.array_equal(action_for_roles_n(core, roles), frozen(core, roles)), (c, carrying)


def test_action_shape_and_vocabulary_at_six():
    for carrying in (False, True):
        core = _FakeCore(carrying)
        for c in COMPOSITIONS:
            a = action_for_roles_n(core, composition_roles_n(c, N_AGENTS))
            assert a.shape == (1, N_AGENTS, 2)
            assert int(a[..., 0].max()) < 8
            assert int(a[..., 1].max()) == 0


def test_defenders_get_defend_and_attackers_follow_the_carrier_branch():
    from macro_actions import MacroAction
    roles = composition_roles_n("4A_2D", N_AGENTS)
    a_free = action_for_roles_n(_FakeCore(False), roles)[0, :, 0]
    a_carry = action_for_roles_n(_FakeCore(True), roles)[0, :, 0]
    assert list(a_free[:2]) == [int(MacroAction.DEFEND)] * 2
    assert list(a_carry[:2]) == [int(MacroAction.DEFEND)] * 2
    assert list(a_free[2:]) == [int(MacroAction.GET_FLAG)] * 4
    assert list(a_carry[2:]) == [int(MacroAction.GO_HOME)] * 4
