"""C_fork precursor detector — natural state only.

Frozen definition: artifacts/o3_preregistration/O3_CFORK_RECALL_AMENDMENT.json
(ce7949f), which supersedes the predicate block of O3_PROTOCOL_FROZEN.json.

    C_fork = blue is carrying
             AND n_legal_team_responses >= 2
             -> first such decision in each carry phase, one-way

Two responsibilities and nothing else: evaluate that predicate, and track carry
phase boundaries so it fires at most once per phase.

DELIBERATELY ABSENT
-------------------
No geometry. No pressure radius. No utility. No access to any C3 or Stage-4
artifact, and no counterfactual field. This module must remain importable by the
O3 trainer, which the response-supervision prohibition (46c9e17) forbids from
touching fork labels. Keeping it label-free is what makes that guarantee
structural rather than promised.

Legality comes from ``rl.analysis.legal_team_responses``, the neutral module
shared with C3. That keeps the counterfactual module out of O3's transitive
training path while preserving the exact semantics the precursor audit
certified -- equivalence is pinned by
tests/test_legal_team_responses_equivalence.py rather than by shared code alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rl.analysis.legal_team_responses import (
    count_legal_team_responses_batched,
    enumerate_legal_team_responses,
)

MIN_LEGAL_TEAM_RESPONSES = 2


def _blue_carrying(core) -> bool:
    import numpy as np

    carrying = core.blue_carrying
    arr = carrying.detach().cpu().numpy() if hasattr(carrying, "detach") else np.asarray(carrying)
    return bool(arr[0].astype(bool).any())


def c_fork_conditions(core) -> tuple[bool, int]:
    """-> (blue_carrying, n_legal_team_responses). Natural state only."""
    carrying = _blue_carrying(core)
    if not carrying:
        # The legality enumeration is only meaningful as a team-response count
        # when a carrier exists; report it anyway for auditability.
        return False, len(enumerate_legal_team_responses(core))
    return True, len(enumerate_legal_team_responses(core))


@dataclass
class CForkDetector:
    """Stateful, one-way per carry phase.

    A carry phase begins when blue possession starts and ends when it ends. A
    new pickup starts a fresh eligible phase, so an episode with three pickups
    can produce up to three onsets -- but at most one per phase.
    """

    carry_phase_index: int = -1
    _in_carry: bool = False
    _fired_this_phase: bool = False
    firings: list[dict] = field(default_factory=list)

    def reset(self) -> None:
        self.carry_phase_index = -1
        self._in_carry = False
        self._fired_this_phase = False
        self.firings = []

    def step(self, core, step_index: int) -> bool:
        """Evaluate at one decision, BEFORE the action is taken. -> fired now."""
        carrying, n_legal = c_fork_conditions(core)

        if carrying and not self._in_carry:
            self._in_carry = True
            self._fired_this_phase = False
            self.carry_phase_index += 1
        elif not carrying and self._in_carry:
            self._in_carry = False
            self._fired_this_phase = False

        if not carrying or self._fired_this_phase:
            return False
        if n_legal < MIN_LEGAL_TEAM_RESPONSES:
            return False

        self._fired_this_phase = True
        self.firings.append({
            "step": int(step_index),
            "carry_phase_index": int(self.carry_phase_index),
            "n_legal_team_responses": int(n_legal),
        })
        return True

    @property
    def active(self) -> bool:
        """True once this carry phase has fired -- the one-way handoff latch."""
        return self._fired_this_phase


__all__ = ["CForkDetector", "MIN_LEGAL_TEAM_RESPONSES", "c_fork_conditions"]
