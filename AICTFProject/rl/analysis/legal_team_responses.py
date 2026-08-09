"""Authoritative blue legal-team-response legality, shared by C3 and O3.

This module exists so that one implementation of legality serves both the
counterfactual screen and the O3 training path, and so the O3 trainer does not
transitively depend on ``counterfactual_actionability`` -- a module whose
broader purpose is counterfactual evaluation, and which the response-supervision
prohibition (46c9e17) keeps out of training.

    enumerate_legal_team_responses(core)
        exact team-response enumeration, B == 1 only
        used by C3 Stage-3 and by the frozen precursor audit
        behaviour preserved EXACTLY from the original implementation

    count_legal_team_responses_batched(core)
        one exact count per environment, shape (B,)
        used by CForkDetector in the 16-env training collector

WHY THE EXACT COUNT, NOT A SHORTCUT
-----------------------------------
The frozen C_fork predicate reads ``n_legal_team_responses >= 2``. Since every
agent is required to have at least one legal macro, that threshold is equivalent
to "some agent has >= 2 choices" -- but the batched function returns the exact
Cartesian-product count anyway. Two reasons: CForkDetector keeps exposing the
same quantity the frozen predicate names, and the equivalence test can compare
integers against the enumerator's ``len(...)`` rather than having to prove an
additional logical shortcut.

THE SCIENTIFIC BRIDGE
---------------------
The 0.7006 precursor audit measured a detector built on the B=1 enumerator. That
audit is frozen and is NOT rerun. What carries its validity across this refactor
is ``tests/test_legal_team_responses_equivalence.py``, which asserts the batched
implementation returns identical counts to the enumerator on real replayed
states. Semantics transfer by proof, not by re-measurement.
"""
from __future__ import annotations

import itertools

import numpy as np


def _np(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _per_agent_legal_macros(core) -> np.ndarray:
    """-> (B, n_agents) count of legal macros per agent, from the authoritative mask.

    Reads exactly what the original enumerator read: ``_build_action_mask``,
    sliced to the macro block, counted as strictly-positive entries.
    """
    mask = _np(core._build_action_mask(side="blue"))
    if mask.ndim != 2:
        raise ValueError(f"expected 2-D blue action mask, got shape={mask.shape}")
    alive = _np(core.blue_alive)
    if alive.ndim != 2:
        raise ValueError(f"expected 2-D blue_alive, got shape={alive.shape}")

    batch = int(mask.shape[0])
    n_agents = int(alive.shape[1])
    n_macros = int(core.cfg.n_macros)
    n_targets = int(core.cfg.n_targets)
    per_agent = mask.reshape(batch, n_agents, n_macros + n_targets)
    return (per_agent[:, :, :n_macros] > 0.0).sum(axis=2).astype(np.int64)


def enumerate_legal_team_responses(core) -> tuple[tuple[int, ...], ...]:
    """Enumerate the Cartesian product of authoritative per-agent macro masks.

    B == 1 only. Behaviour preserved exactly from the original implementation in
    ``counterfactual_actionability``, including the batch-shape validation and
    the RuntimeError when an agent has no legal macro.
    """
    mask = _np(core._build_action_mask(side="blue"))
    if mask.ndim != 2 or mask.shape[0] != 1:
        raise ValueError(f"C3 requires B=1 action mask, got shape={mask.shape}")

    alive = _np(core.blue_alive)
    if alive.ndim != 2 or alive.shape[0] != 1:
        raise ValueError(f"C3 requires B=1 blue_alive, got shape={alive.shape}")

    n_agents = int(alive.shape[1])
    n_macros = int(core.cfg.n_macros)
    n_targets = int(core.cfg.n_targets)
    per_agent = mask.reshape(1, n_agents, n_macros + n_targets)[0]
    legal_macros = []
    for agent_i in range(n_agents):
        macros = tuple(
            int(macro_i)
            for macro_i in np.flatnonzero(per_agent[agent_i, :n_macros] > 0.0)
        )
        if not macros:
            raise RuntimeError(f"authoritative mask exposes no macro for blue agent {agent_i}")
        legal_macros.append(macros)
    return tuple(tuple(int(m) for m in response) for response in itertools.product(*legal_macros))


def count_legal_team_responses_batched(core) -> np.ndarray:
    """-> (B,) exact Cartesian-product count of legal team responses per env.

    Equivalent to ``len(enumerate_legal_team_responses(core))`` when B == 1, and
    defined for any B. Raises when any agent in any environment has no legal
    macro, matching the enumerator's failure mode rather than silently
    producing a zero count.
    """
    counts = _per_agent_legal_macros(core)
    if (counts <= 0).any():
        bad = np.argwhere(counts <= 0)
        env_i, agent_i = int(bad[0][0]), int(bad[0][1])
        raise RuntimeError(
            f"authoritative mask exposes no macro for blue agent {agent_i} in env {env_i}"
        )
    return counts.prod(axis=1).astype(np.int64)


__all__ = [
    "count_legal_team_responses_batched",
    "enumerate_legal_team_responses",
]
