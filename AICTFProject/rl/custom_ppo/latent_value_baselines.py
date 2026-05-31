"""Value-function baselines for the latent strategy router (q_phi).

The legacy episode-credit advantage uses ``V(s, z_picked)`` as the baseline:

    adv_z = R - V(s, z_picked)
          ~ R - E[R | s, z_picked]
          ~ noise that does NOT encode "z_picked vs other z"

That mathematically cancels the cross-z signal q_phi needs to specialize -- the
centralized critic absorbs E[R | s, z] before the router ever sees the gradient.

This module provides the z-marginal baseline, which is the variance-optimal
advantage-actor-critic formula for a discrete latent policy:

    V_marginal(s) = E_{z' ~ q_phi(s)}[V(s, z')] = sum_k pi_phi(k|s) * V(s, k)
    adv_z         = R - V_marginal(s)

Now ``adv_z`` is non-zero exactly when V(s, z_picked) differs from the
policy-weighted average -- i.e. it encodes "is this z above or below the
average available z in this context?", which is exactly the signal q_phi
needs to learn contextual specialization.

Plan-faithful: no labels, no aux heads, no opponent IDs. The only change is
the choice of baseline inside the existing q_phi policy-gradient update.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def compute_z_marginal_strategy_value(
    model: Any,
    states: torch.Tensor,
    latent_k: int,
    *,
    policy_weighted: bool = True,
) -> torch.Tensor:
    """Return the z-marginal value baseline ``E_{z' ~ q_phi}[V(s, z')]``.

    Parameters
    ----------
    model
        Object exposing ``episode_strategy_value(states, z_idx)`` and
        ``strategy_logits(states)``. Concretely this is the policy network
        produced by :class:`SharedActorCentralizedCritic` (it carries both
        the q_phi logits head and the episode-strategy value head).
    states
        ``(B, q_phi_input_dim)`` context tensor that q_phi consumes (ctx170
        in the current setup). Must already live on the model's device.
    latent_k
        Number of latent strategy slots ``K``.
    policy_weighted
        When ``True`` (default) the marginal is computed under the *current*
        q_phi policy: ``sum_k pi_phi(k|s) * V(s, k)``. This is the
        variance-optimal AAC baseline. When ``False`` a uniform mean
        ``(1/K) * sum_k V(s, k)`` is used; both are valid baselines (zero
        bias on the policy gradient), but the policy-weighted version has
        lower variance once q_phi specializes.

    Returns
    -------
    torch.Tensor
        Shape ``(B,)`` baseline values, **always detached**. q_phi's
        gradient must NOT flow into the value head through this baseline;
        the value head receives its own gradient through the standard
        ``v_loss = MSE(V(s, z_picked), R)`` term so coupling the baseline
        path would double-count and destabilize the value learner.
    """
    if states.dim() != 2:
        raise ValueError(
            f"compute_z_marginal_strategy_value expects a 2-D (B, D) states tensor; got {tuple(states.shape)}"
        )
    if latent_k < 1:
        raise ValueError(f"latent_k must be >= 1; got {latent_k}")
    if latent_k == 1:
        z_only = torch.zeros((states.shape[0],), dtype=torch.long, device=states.device)
        return model.episode_strategy_value(states, z_only).detach().reshape(-1)

    batch = int(states.shape[0])
    device = states.device

    z_grid = torch.arange(latent_k, dtype=torch.long, device=device)
    z_all = z_grid.repeat_interleave(batch)
    s_rep = states.repeat(latent_k, 1)
    v_all = model.episode_strategy_value(s_rep, z_all)
    if v_all.dim() > 1:
        v_all = v_all.squeeze(-1)
    v_all = v_all.reshape(latent_k, batch).detach()

    if not policy_weighted:
        return v_all.mean(dim=0)

    logits = model.strategy_logits(states).detach()
    probs = F.softmax(logits, dim=-1)
    weights = probs.transpose(0, 1)
    return (weights * v_all).sum(dim=0)


__all__ = ["compute_z_marginal_strategy_value"]
