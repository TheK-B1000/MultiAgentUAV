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

# Pole -> (latent id that must carry it, teacher whose actions supervise it)
POLE_ROUTING = {"A": (0, "pi_A"), "B": (1, "pi_B")}


class CausalRoutingError(RuntimeError):
    """Raised when a record would supervise a latent with the wrong pole's teacher."""


@dataclass(frozen=True)
class CausalRecord:
    """One Phase 1 measurement, as the trainer consumes it.

    ``weight`` is |delta_Q| for this (state, agent). ``teacher`` names the specialist whose
    action is the target; it must be the pole-matched one, and ``assert_routing`` enforces it.
    """
    state_id: str
    pole: str
    agent_id: int | None          # None means a joint record covering every agent
    teacher: str
    weight: float
    intervention_mode: str        # "single_macro" or "full_takeover"

    @property
    def latent(self) -> int:
        return POLE_ROUTING[self.pole][0]

    def assert_routing(self) -> None:
        want_z, want_teacher = POLE_ROUTING[self.pole]
        if self.teacher != want_teacher:
            raise CausalRoutingError(
                f"{self.state_id}: pole {self.pole} must be supervised by {want_teacher}, "
                f"got {self.teacher}. Routing z{want_z} to the wrong specialist would train "
                f"the latent to carry the opposite strategy.")


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
