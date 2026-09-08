"""Causal-advantage supervision for the successor latent policy.

Implements CCP_SUCCESSOR_BUILD_CONTRACT.json, built while the Phase 1 bank was still being
collected and before any outcome statistic existed.

    L_causal = sum_{t,i} d_ti * w_ti * L_teacher(s_t, i, z)  /  sum_{t,i} d_ti * w_ti

    d_ti = [commit_ticks_left_i <= 0]      per agent, the runtime decision predicate
    w_ti = |delta_Q(s_t, i)|               the weight frozen in the Phase 1 spec

Two properties this is built to guarantee, both of which the ladder got wrong:

1. A COMMITTED agent contributes neither gradient nor denominator mass. OG-PSP, V3 and V4
   passed ``obs["agent_mask"]`` -- all ones in the teacher bank -- as the decision mask, so
   committed heads inflated the denominator while contributing exactly zero, scaling the
   effective lambda to ~0.662 of nominal (CCP_FORCED_HEAD_DILUTION.json). Here the predicate
   is the real one and it gates the denominator too.

2. A ZERO-WEIGHT sample contributes no gradient. A decision with no measured payoff effect
   receives no strategic pressure by construction, not by filtering -- which is what makes
   w = |delta_Q| a threshold-free rule.

The loss is an auxiliary POLICY loss. It never touches reward, returns, GAE or value targets.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

# Pole -> (latent that must carry it, pole-matched specialist, the other specialist)
POLE_ROUTING = {"A": (0, "pi_A", "pi_B"), "B": (1, "pi_B", "pi_A")}


class CausalRoutingError(RuntimeError):
    """Raised when a record would supervise a latent with the wrong pole's teacher."""


@dataclass(frozen=True)
class _WinnerDirected:
    """Shared derivation for every causal supervision unit.

    Both the local record and the segment inherit this rather than reimplementing the rule.
    That is deliberate: the defect this replaced came from a teacher field that could disagree
    with the measurement, and giving the segment its own copy of the logic would recreate the
    same failure one abstraction level higher. There is exactly one place where sign becomes
    teacher.
    """
    pole: str
    delta_q: float                # SIGNED, pole-oriented

    @property
    def latent(self) -> int:
        return POLE_ROUTING[self.pole][0]

    @property
    def weight(self) -> float:
        return abs(self.delta_q)

    @property
    def teacher(self) -> str | None:
        """None when delta_q is exactly zero: no measured effect, so no supervision."""
        _, matched, other = POLE_ROUTING[self.pole]
        if self.delta_q > 0:
            return matched
        if self.delta_q < 0:
            return other
        return None

    def _assert_routing(self, ident: str, declared_teacher: str | None) -> None:
        if self.pole not in POLE_ROUTING:
            raise CausalRoutingError(f"{ident}: unknown pole {self.pole!r}")
        if declared_teacher is not None and declared_teacher != self.teacher:
            raise CausalRoutingError(
                f"{ident}: delta_q {self.delta_q:+.4f} on pole {self.pole} implies teacher "
                f"{self.teacher}, but {declared_teacher} was declared. Supervising the loser of "
                f"a measured contrast would train the latent away from payoff.")
        if self.teacher is None and self.weight != 0.0:
            raise CausalRoutingError(f"{ident}: no teacher but non-zero weight")


@dataclass(frozen=True)
class CausalRecord(_WinnerDirected):
    """One Phase 1 measurement, as the trainer consumes it.

    WINNER-DIRECTED ROUTING. ``delta_q`` is stored SIGNED, under the pole-oriented convention
    (Pole A: Q(pi_A) - Q(pi_B); Pole B: Q(pi_B) - Q(pi_A)), so positive means the pole-matched
    specialist was causally better at this boundary. Both the weight and the teacher are
    DERIVED from it:

        weight  = |delta_q|
        teacher = pole-matched specialist if delta_q > 0, the OTHER specialist if delta_q < 0

    Deriving rather than storing the teacher is deliberate. An earlier version stored the
    teacher independently and pinned it to the pole-matched specialist, which meant a boundary
    measuring delta_q = -0.50 would carry weight 0.50 and still train the latent toward the
    specialist the measurement had just shown was WORSE there -- training hardest on exactly
    the decisions it should have avoided. That state is now unrepresentable.

    The LATENT never flips: z0 carries Pole A and z1 carries Pole B, because that is the
    deployment condition the crossover gate scores. Only the target flips, so the latent learns
    whichever behaviour causally improved payoff rather than preserving specialist identity for
    its own sake.
    """
    state_id: str = ""
    agent_id: int | None = None   # None means a joint record covering every agent
    intervention_mode: str = "single_macro"

    def assert_routing(self, declared_teacher: str | None = None) -> None:
        self._assert_routing(self.state_id, declared_teacher)


@dataclass(frozen=True)
class CausalSegment(_WinnerDirected):
    """A SEQUENCE-mode supervision unit spanning several commitment decisions.

    Carries signed delta_q and inherits the identical derivation, so it cannot express a
    teacher that disagrees with its measurement. ``active_until`` records how far the segment
    supervises; the RULE that sets it is deliberately NOT frozen here, because Phase 1 decides
    whether sequence mode is used at all and what its termination should be.
    """
    segment_id: str = ""
    start_state_id: str = ""
    controlled_agents: tuple[int, ...] = ()
    active_until: int | None = None      # rule not frozen; Phase 1 decides

    def assert_routing(self, declared_teacher: str | None = None) -> None:
        self._assert_routing(self.segment_id, declared_teacher)
        if not self.controlled_agents and self.weight != 0.0:
            raise CausalRoutingError(
                f"{self.segment_id}: non-zero weight but no controlled agents")


def decision_mask_from_core(core, n_agents: int, *, side: str = "blue") -> torch.Tensor:
    """d_ti = [commit_ticks_left_i <= 0] read from the live environment.

    Missing state is fatal rather than defaulted: a silently all-True mask is exactly the
    defect this module exists to prevent.
    """
    attr = f"{side}_commit_ticks_left"
    ticks = getattr(core, attr, None)
    if ticks is None:
        raise CausalRoutingError(
            f"core exposes no {attr}; the decision predicate cannot be established, and "
            "absence is not evidence that every agent is free")
    if ticks.shape[-1] != n_agents:
        raise CausalRoutingError(f"{attr} has {ticks.shape[-1]} agents, expected {n_agents}")
    return (ticks <= 0)


def causal_supervision_loss(
    model,
    obs: dict,
    teacher_actions: torch.Tensor,
    *,
    z_idx: torch.Tensor,
    decision_mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted negative log-likelihood of the teacher action at live decisions only.

    Parameters
    ----------
    decision_mask : (batch, n_agents) bool
        True where that agent is free to commit a new macro.
    weights : (batch, n_agents) float
        |delta_Q| per (state, agent). Zero contributes nothing, to numerator or denominator.
    """
    from rl.custom_ppo.strategy_anchor import action_log_prob

    per_head = action_log_prob(model, obs, teacher_actions, z_idx=z_idx)
    n_heads = per_head.shape[1]
    n_agents = decision_mask.shape[1]
    if n_heads % n_agents:
        raise CausalRoutingError(f"{n_heads} heads do not divide across {n_agents} agents")
    if weights.shape != decision_mask.shape:
        raise CausalRoutingError(
            f"weights {tuple(weights.shape)} must match decision_mask {tuple(decision_mask.shape)}")
    if bool((weights < 0).any()):
        raise CausalRoutingError("weights must be non-negative; pass |delta_Q|, not delta_Q")

    per_agent = n_heads // n_agents
    m = decision_mask.to(per_head.dtype).repeat_interleave(per_agent, dim=1)
    w = weights.to(per_head.dtype).repeat_interleave(per_agent, dim=1)
    mw = m * w
    denom = mw.sum()
    if float(denom) <= 0.0:
        # legitimate: a batch where nothing carries measured payoff effect. No pressure,
        # and no division by zero. Returned as a real zero tensor so .backward() is safe.
        return (per_head * 0.0).sum()
    return -(per_head * mw).sum() / denom
