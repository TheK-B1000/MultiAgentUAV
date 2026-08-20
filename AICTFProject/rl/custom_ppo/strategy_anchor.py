"""SAPPO V1 — strategy-anchor loss over the EXISTING training-side distribution.

Frozen protocol: artifacts/strategic_demand/STRATEGY_ANCHORED_PPO_V1_FROZEN.json

    L_actor = L_PPO + lambda_anchor * L_anchor
    L_anchor = -log pi_theta(a_teacher | o)

The loss was frozen before implementation, so the interface adapts to the loss
rather than the loss being reshaped around whatever API was convenient.

Why this file adds no new distribution machinery
------------------------------------------------
``SharedActorCentralizedCritic.get_distribution()`` is already a public contract
method returning the same ``MultiHeadActionDistribution`` the PPO actor update
uses. This module only evaluates that distribution at a known action. No
sampling, no argmax, no inference wrapper, and no new factorization: the head
structure is whatever ``action_space.nvec`` already defines.

Head ordering is agent-major, mirroring ``action_dims = action_space.nvec`` and
``heads_per_agent = len(action_dims) // n_agents``. A demonstration action of
shape ``(batch, n_agents, heads_per_agent)`` therefore flattens to head order
directly -- the same flattening used when the demonstrations were generated, so
labels cannot silently transpose.

Decision-point masking
----------------------
The engine latches a macro only when ``blue_commit_ticks_left <= 0``. Anchor
supervision applies only to agents that actually had a choice on that step;
locked agents contribute no gradient.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

__all__ = ["action_log_prob", "anchor_loss", "teacher_agreement"]


def action_log_prob(
    model,
    obs: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    *,
    z_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """log pi_theta(a | o), summed over action heads.

    Parameters
    ----------
    model : SharedActorCentralizedCritic
        The TRAINING-side model, not an inference wrapper.
    obs : dict of tensors
        Observation batch as the policy consumes it.
    actions : LongTensor
        ``(batch, n_agents, heads_per_agent)`` or ``(batch, n_heads)``.
    z_idx : optional LongTensor
        Required when the model uses latent strategy; passed straight through
        so this helper never invents a silent default.

    Returns
    -------
    Tensor ``(batch, n_heads)`` of per-head log-probabilities. The caller
    decides how to reduce, because decision-point masking is per agent-head.
    """
    dist = model.get_distribution(obs, z_idx=z_idx)
    heads = dist.heads
    a = actions.reshape(actions.shape[0], -1).long()
    if a.shape[1] != len(heads):
        raise ValueError(
            f"action encoding mismatch: got {a.shape[1]} action columns for "
            f"{len(heads)} policy heads. Demonstration actions must be "
            f"agent-major and match action_space.nvec exactly; no remapping is "
            f"performed here.")
    out = []
    for h, head in enumerate(heads):
        idx = a[:, h]
        if int(idx.max()) >= head.action_dim or int(idx.min()) < 0:
            raise ValueError(
                f"head {h}: action index out of range [0, {head.action_dim}).")
        logp = head.logits.log_softmax(dim=-1)
        out.append(logp.gather(1, idx.unsqueeze(1)).squeeze(1))
    return torch.stack(out, dim=1)


def anchor_loss(
    model,
    obs: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    decision_mask: Optional[torch.Tensor] = None,
    *,
    z_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean negative log-probability of the teacher action.

    ``decision_mask`` is ``(batch, n_agents)`` and True where that agent had a
    macro decision available. It is broadcast across that agent's heads, so a
    locked agent contributes nothing to the loss.
    """
    per_head = action_log_prob(model, obs, actions, z_idx=z_idx)
    if decision_mask is None:
        return -per_head.mean()

    n_heads = per_head.shape[1]
    n_agents = decision_mask.shape[1]
    if n_heads % n_agents:
        raise ValueError(f"{n_heads} heads do not divide across {n_agents} agents")
    per_agent = n_heads // n_agents
    m = decision_mask.to(per_head.dtype).repeat_interleave(per_agent, dim=1)
    denom = m.sum().clamp_min(1.0)
    return -(per_head * m).sum() / denom


@torch.no_grad()
def teacher_agreement(
    model,
    obs: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    decision_mask: Optional[torch.Tensor] = None,
    *,
    z_idx: Optional[torch.Tensor] = None,
) -> float:
    """Fraction of decision-point heads whose argmax equals the teacher action.

    Diagnostic only. Reported for the train split and the held-out 10% split so
    memorisation is visible, but it is NOT a gate.
    """
    dist = model.get_distribution(obs, z_idx=z_idx)
    a = actions.reshape(actions.shape[0], -1).long()
    hits = torch.stack([(h.argmax_actions == a[:, i]).float()
                        for i, h in enumerate(dist.heads)], dim=1)
    if decision_mask is None:
        return float(hits.mean())
    n_heads, n_agents = hits.shape[1], decision_mask.shape[1]
    m = decision_mask.to(hits.dtype).repeat_interleave(n_heads // n_agents, dim=1)
    return float((hits * m).sum() / m.sum().clamp_min(1.0))
