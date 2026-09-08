"""Correctness tests for the Q_psi action-conditioned scorer.

The load-bearing claim is amendment 2 of the Phase 0 protocol: that

    V_hat = b + u^T mu_1 + v^T mu_2 + mu_1^T M mu_2

is the EXACT expectation of Q_psi(o,a1,a2,p) under a factorised policy, not an
approximation. If that is wrong, every Gate 0B number is wrong, so it is tested
against brute-force enumeration rather than assumed.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rl.scorer.qpsi import N_ACTIONS, N_WAYPOINT, QPsi, QPsiConfig, joint_action_index


def _obs(n, gen):
    return (torch.rand(n, 2, 7, 20, 20, generator=gen),
            torch.rand(n, 2, 20, generator=gen) * 2 - 1,
            torch.ones(n, 2),
            torch.randint(0, 2, (n,), generator=gen))


def test_joint_action_index_matches_agent_major_layout():
    a = torch.tensor([[0, 5, 4, 49], [3, 0, 1, 12]])
    a1, a2 = joint_action_index(a)
    assert a1.tolist() == [0 * N_WAYPOINT + 5, 3 * N_WAYPOINT + 0]
    assert a2.tolist() == [4 * N_WAYPOINT + 49, 1 * N_WAYPOINT + 12]
    assert int(a1.max()) < N_ACTIONS and int(a2.max()) < N_ACTIONS


def test_joint_action_index_rejects_wrong_width():
    with pytest.raises(ValueError):
        joint_action_index(torch.zeros(2, 3, dtype=torch.long))


def test_analytic_expectation_equals_brute_force_enumeration():
    """V_hat must equal sum_{a1,a2} p1(a1)p2(a2) Q(o,a1,a2,p), exactly."""
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    # small action space keeps the 62,500-term enumeration tractable in a test
    m = QPsi(QPsiConfig(action_dim=6, rank=3, hidden=32, conv_width=16)).eval()
    k = 12                                    # enumerate over a k x k sub-block
    grid, vec, am, pole = _obs(2, gen)

    p1 = torch.rand(2, N_ACTIONS, generator=gen)
    p2 = torch.rand(2, N_ACTIONS, generator=gen)
    p1[:, k:] = 0.0                           # support confined to first k actions
    p2[:, k:] = 0.0
    p1 = p1 / p1.sum(-1, keepdim=True)
    p2 = p2 / p2.sum(-1, keepdim=True)

    with torch.no_grad():
        analytic = m.expected_value(grid, vec, am, pole, p1, p2)
        brute = torch.zeros(2)
        for i in range(k):
            for j in range(k):
                a1 = torch.full((2,), i, dtype=torch.long)
                a2 = torch.full((2,), j, dtype=torch.long)
                q = m(grid, vec, am, pole, a1, a2)
                brute += p1[:, i] * p2[:, j] * q

    assert torch.allclose(analytic, brute, atol=1e-5), (analytic, brute)


def test_expectation_over_point_mass_equals_forward():
    """A deterministic policy must reduce V_hat to a plain Q lookup."""
    gen = torch.Generator().manual_seed(1)
    torch.manual_seed(1)
    m = QPsi(QPsiConfig(action_dim=8, rank=4, hidden=32, conv_width=16)).eval()
    grid, vec, am, pole = _obs(3, gen)
    a1 = torch.tensor([7, 100, 249])
    a2 = torch.tensor([0, 42, 13])
    p1 = torch.zeros(3, N_ACTIONS).scatter_(1, a1[:, None], 1.0)
    p2 = torch.zeros(3, N_ACTIONS).scatter_(1, a2[:, None], 1.0)
    with torch.no_grad():
        assert torch.allclose(m.expected_value(grid, vec, am, pole, p1, p2),
                              m(grid, vec, am, pole, a1, a2), atol=1e-6)


def test_interaction_term_is_not_additively_separable():
    """The protocol rejects Q = Q1(o,a1) + Q2(o,a2). Verify M can break it.

    Additive separability implies Q(a,b) - Q(a,b') - Q(a',b) + Q(a',b') == 0 for
    all action pairs. A trained-up interaction head must be able to violate it.
    """
    torch.manual_seed(2)
    gen = torch.Generator().manual_seed(2)
    m = QPsi(QPsiConfig(action_dim=8, rank=4, hidden=32, conv_width=16)).eval()
    # heads start near zero by design, so give the interaction real weight
    with torch.no_grad():
        m.head_P.weight.normal_(std=0.5); m.head_Q.weight.normal_(std=0.5)
    grid, vec, am, pole = _obs(1, gen)
    o = (grid, vec, am, pole)
    t = lambda i: torch.tensor([i])
    with torch.no_grad():
        cross = (m(*o, t(3), t(9)) - m(*o, t(3), t(40))
                 - m(*o, t(77), t(9)) + m(*o, t(77), t(40)))
    assert cross.abs().item() > 1e-4, "interaction term collapsed to additive"


def test_policy_identity_is_not_an_input():
    """Q_psi's signature must expose no policy/latent argument."""
    import inspect
    params = set(inspect.signature(QPsi.forward).parameters)
    for forbidden in ("policy", "policy_id", "z", "latent", "teacher"):
        assert forbidden not in params
    params_v = set(inspect.signature(QPsi.expected_value).parameters)
    for forbidden in ("policy_id", "z", "latent", "teacher"):
        assert forbidden not in params_v
