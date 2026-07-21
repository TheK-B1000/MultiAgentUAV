"""Pure-tensor losses for the latent team strategy PPO update.

These helpers are the **building blocks** consumed by
:meth:`rl.custom_ppo.CustomPPOTrainer.update`. Each function:

* Takes only tensors and scalar hyperparameters (no config object, no
  trainer reference). This keeps them unit-testable and makes the contract
  explicit.
* Returns ``(loss, stats)`` where ``loss`` is a 0-d ``torch.Tensor`` that the
  caller adds into the total ``latent_loss``, and ``stats`` is a
  ``dict[str, float]`` of plain-Python diagnostics the caller can write to
  CSV.

Composition order (summing each ``loss`` into ``latent_loss``) is the caller's
responsibility. Floating-point addition is not associative; the trainer
preserves the historical order so byte-level output is unchanged.

The functions mirror — exactly — the inline math that previously lived in
``CustomPPOTrainer.update``. They are intentionally minimal: no clever fusion,
no autograd shortcuts. The only goal is "same numbers, tested in isolation."
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from rl.latent_marl import expected_strategy_switch_penalty
from rl.ppo_core import ppo_policy_loss


_LossStats = Dict[str, float]


def _zero_scalar(device: torch.device) -> Tensor:
    return torch.zeros((), dtype=torch.float32, device=device)


def strategy_entropy_loss(
    strategy_entropy: Tensor,
    resample_mask: Tensor,
    *,
    objective: str,
    lam_h: float,
    device: torch.device,
) -> Tuple[Tensor, _LossStats]:
    """λ_H * mean H(q_phi(z|s))[resample_mask], with sign per ``objective``.

    * ``objective="maximize"`` (paper default): sign = -1, so minimizing the
      loss increases entropy.
    * ``objective="minimize"``: sign = +1, sharpening q_phi.
    * ``objective="none"`` or ``lam_h <= 0``: returns a zero loss, no grad.

    Stats: ``{"strategy_entropy_term_mean": <mean H>}``. The mean is taken
    only over the resample subset (matching the trainer); when the mask is
    empty, returns 0.
    """
    if resample_mask.any():
        h_mean = strategy_entropy[resample_mask].mean()
    else:
        h_mean = _zero_scalar(device)

    obj = str(objective or "maximize").lower()
    if obj == "none" or lam_h <= 0.0:
        return _zero_scalar(device), {"strategy_entropy_term_mean": float(h_mean.detach().cpu().item())}
    if obj == "minimize":
        return lam_h * h_mean, {"strategy_entropy_term_mean": float(h_mean.detach().cpu().item())}
    return -lam_h * h_mean, {"strategy_entropy_term_mean": float(h_mean.detach().cpu().item())}


def feedforward_router_entropy_loss(
    strategy_entropy: Tensor,
    decision_mask: Tensor,
    *,
    router_ent_coef: float,
    device: torch.device,
) -> Tuple[Tensor, _LossStats]:
    """Conditional router entropy for feedforward sparse-routing updates.

    Matches ``RouterSequenceUpdater`` semantics: maximize entropy at router
    decision steps only via ``router_ent_coef * (-mean H)``.
    """
    if not bool(decision_mask.any()) or router_ent_coef <= 0.0:
        return _zero_scalar(device), {
            "feedforward_router_entropy_mean": 0.0,
            "feedforward_router_entropy_loss": 0.0,
        }
    h_mean = strategy_entropy[decision_mask].mean()
    loss = router_ent_coef * (-h_mean)
    return loss, {
        "feedforward_router_entropy_mean": float(h_mean.detach().cpu().item()),
        "feedforward_router_entropy_loss": float(loss.detach().cpu().item()),
    }


def strategy_marginal_entropy_loss(
    strategy_logits: Tensor,
    resample_mask: Tensor,
    *,
    objective: str,
    lam_h: float,
    latent_k: int,
    device: torch.device,
) -> Tuple[Tensor, _LossStats]:
    """Per-minibatch marginal entropy loss (PARITY-ONLY, NOT USED IN v5i6).

    Computes ``KL(mean_s q_phi(z|s) || Uniform)`` over the resample subset of
    the *current PPO minibatch*. The mean is taken over fewer than 1 / N of
    the rollout's resample-decision points, and KL is convex in ``p_bar``, so
    by Jensen ``E_B[KL(p_bar_B || U)] >= KL(E_B[p_bar_B] || U)``: the
    per-minibatch loss is a strict upper bound on the rollout-level marginal
    KL whenever the per-minibatch ``p_bar_B`` are not constant. The bias is
    closed by the gradient softening individual ``q_phi(z|s)`` rows toward
    uniform — the conditional-entropy regression v5i6 was designed to avoid.

    .. deprecated:: v5i6 (rollout-level aggregation)
       Retained ONLY for parity tests against the historical implementation
       and as a comparison baseline. Production code MUST use
       ``rollout_marginal_entropy_loss`` (computed once per PPO inner epoch
       over the full rollout resample subset). The PPO updater no longer
       wires this function into any production preset; if you find it
       active in a v5i6-family run, that is a regression.
    """
    obj = str(objective or "maximize").lower()
    if not bool(resample_mask.any()):
        return _zero_scalar(device), {
            "strategy_marginal_entropy_nats": 0.0,
            "strategy_marginal_entropy_kl": 0.0,
        }

    probs = torch.softmax(strategy_logits[resample_mask], dim=-1)
    p_bar = probs.mean(dim=0).clamp_min(1e-8)
    marginal_entropy = -(p_bar * torch.log(p_bar)).sum()
    usage_kl = (
        p_bar
        * (torch.log(p_bar) + torch.log(p_bar.new_tensor(float(latent_k))))
    ).sum()

    if obj == "none" or lam_h <= 0.0:
        loss = _zero_scalar(device)
    elif obj == "minimize":
        loss = -float(lam_h) * usage_kl
    else:
        loss = float(lam_h) * usage_kl
    return loss, {
        "strategy_marginal_entropy_nats": float(
            marginal_entropy.detach().cpu().item()
        ),
        "strategy_marginal_entropy_kl": float(usage_kl.detach().cpu().item()),
    }


def rollout_marginal_entropy_loss(
    rollout_resample_logits: Tensor,
    *,
    objective: str,
    lam_h: float,
    latent_k: int,
    device: torch.device,
) -> Tuple[Tensor, _LossStats]:
    """Rollout-level marginal entropy loss for the v5i6 contract.

    Caller responsibilities:
        ``rollout_resample_logits`` MUST be the differentiable router logits
        evaluated at *every* router-decision point in the current rollout
        (i.e. all rows where ``z_resampled == True``), not just those landing
        in the current PPO minibatch. The caller is responsible for gathering
        these states from the rollout buffer and running ONE forward pass
        through ``q_phi`` so gradients can flow.

    Loss math (for ``objective="maximize"``):

    .. math::

        \\bar q = \\frac{1}{N}\\sum_{i=1}^{N} \\mathrm{softmax}(\\ell_i)

        L = \\lambda_H \\cdot \\mathrm{KL}(\\bar q \\,\\Vert\\, U)
              = \\lambda_H \\cdot (\\log K - H(\\bar q))

    Minimizing ``L`` maximizes the *rollout-aggregated* marginal entropy
    ``H(\\bar q)``. Because the average is taken over the entire rollout
    population in a single forward+backward, this implementation does NOT
    suffer the per-minibatch Jensen bias of ``strategy_marginal_entropy_loss``
    (see that function's docstring).

    Stats include rollout-level diagnostics from the same population:

    * ``rollout_marginal_entropy_nats``: ``H(\\bar q)``
    * ``rollout_marginal_entropy_kl``: ``KL(\\bar q || U)``
    * ``rollout_conditional_entropy_nats``: ``mean_i H(softmax(\\ell_i))``
    * ``rollout_mi_proxy_nats``:
      ``rollout_marginal_entropy_nats - rollout_conditional_entropy_nats``
    * ``rollout_resample_count``: ``N`` (so analysis can spot empty rollouts)
    """
    obj = str(objective or "maximize").lower()
    if rollout_resample_logits.numel() == 0 or rollout_resample_logits.shape[0] == 0:
        return _zero_scalar(device), {
            "rollout_marginal_entropy_nats": 0.0,
            "rollout_marginal_entropy_kl": 0.0,
            "rollout_conditional_entropy_nats": 0.0,
            "rollout_mi_proxy_nats": 0.0,
            "rollout_resample_count": 0.0,
        }

    probs = torch.softmax(rollout_resample_logits, dim=-1)
    p_bar = probs.mean(dim=0).clamp_min(1e-8)
    marginal_entropy = -(p_bar * torch.log(p_bar)).sum()
    usage_kl = (
        p_bar
        * (torch.log(p_bar) + torch.log(p_bar.new_tensor(float(latent_k))))
    ).sum()
    per_state_entropy = -(probs.clamp_min(1e-8) * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
    conditional_entropy = per_state_entropy.mean()
    mi_proxy = marginal_entropy - conditional_entropy

    if obj == "none" or lam_h <= 0.0:
        loss = _zero_scalar(device)
    elif obj == "minimize":
        loss = -float(lam_h) * usage_kl
    else:
        loss = float(lam_h) * usage_kl
    return loss, {
        "rollout_marginal_entropy_nats": float(marginal_entropy.detach().cpu().item()),
        "rollout_marginal_entropy_kl": float(usage_kl.detach().cpu().item()),
        "rollout_conditional_entropy_nats": float(
            conditional_entropy.detach().cpu().item()
        ),
        "rollout_mi_proxy_nats": float(mi_proxy.detach().cpu().item()),
        "rollout_resample_count": float(rollout_resample_logits.shape[0]),
    }


def rollout_router_soft_diagnostics(
    rollout_resample_logits: Tensor,
    *,
    latent_k: int,
) -> _LossStats:
    """Non-gradient soft router diagnostics over the rollout resample subset.

    Returns the same ``H_marginal``, ``H_conditional``, ``MI_proxy`` measured
    by ``rollout_marginal_entropy_loss``, plus soft-decision occupancy:

    * ``router_rollout_soft_marginal_entropy_nats``: ``H(\\bar q)``
    * ``router_rollout_soft_conditional_entropy_nats``: ``mean_i H(q_i)``
    * ``router_rollout_soft_mi_proxy_nats``: difference of the two above
    * ``router_rollout_soft_p_bar_z<k>``: per-z entries of ``\\bar q``
    * ``router_rollout_soft_argmax_occupancy_max``: max over z of
      ``mean_i 1[\\arg\\max q_i = z]`` (soft-argmax population fraction)
    * ``router_rollout_soft_argmax_occupancy_min``: corresponding min
    * ``router_rollout_soft_argmax_occupancy_ratio``: max / max(min, eps)

    These are intentionally distinct from the *sampled-z* counterparts in
    ``rl/custom_ppo/latent_diagnostics.py`` (``latent_occupancy_*``,
    ``latent_marginal_entropy_nats``, ``effective_num_latents``), which are
    one-sample-per-state empirical histograms over the categorical samples.

    No gradients are computed; safe to call inside a ``no_grad`` context.
    """
    K = int(latent_k)
    if (
        rollout_resample_logits.numel() == 0
        or rollout_resample_logits.shape[0] == 0
    ):
        out: _LossStats = {
            "router_rollout_soft_marginal_entropy_nats": 0.0,
            "router_rollout_soft_conditional_entropy_nats": 0.0,
            "router_rollout_soft_mi_proxy_nats": 0.0,
            "router_rollout_soft_argmax_occupancy_max": 0.0,
            "router_rollout_soft_argmax_occupancy_min": 0.0,
            "router_rollout_soft_argmax_occupancy_ratio": 0.0,
            "router_rollout_resample_count": 0.0,
        }
        for k in range(K):
            out[f"router_rollout_soft_p_bar_z{k}"] = 0.0
        return out

    with torch.no_grad():
        probs = torch.softmax(rollout_resample_logits, dim=-1)
        p_bar = probs.mean(dim=0).clamp_min(1e-12)
        marg_h = -(p_bar * torch.log(p_bar)).sum()
        cond_h = -(probs.clamp_min(1e-12) * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean()

        argmax = probs.argmax(dim=-1)
        counts = torch.bincount(argmax, minlength=K).to(torch.float32)
        occ = counts / max(1, int(argmax.shape[0]))
        occ_max = float(occ.max().item())
        occ_min = float(occ.min().item())
        occ_ratio = occ_max / max(occ_min, 1e-12) if occ_min > 0.0 else float("inf")
        # ``inf`` is unfriendly to CSV consumers; cap at a large sentinel and
        # rely on ``occ_min`` to disambiguate true zero-population z's.
        if not math.isfinite(occ_ratio):
            occ_ratio = float(K) * float(occ_max) * 1e6

    out = {
        "router_rollout_soft_marginal_entropy_nats": float(marg_h.item()),
        "router_rollout_soft_conditional_entropy_nats": float(cond_h.item()),
        "router_rollout_soft_mi_proxy_nats": float((marg_h - cond_h).item()),
        "router_rollout_soft_argmax_occupancy_max": occ_max,
        "router_rollout_soft_argmax_occupancy_min": occ_min,
        "router_rollout_soft_argmax_occupancy_ratio": occ_ratio,
        "router_rollout_resample_count": float(int(argmax.shape[0])),
    }
    for k in range(K):
        out[f"router_rollout_soft_p_bar_z{k}"] = float(p_bar[k].item())
    return out


def strategy_persistence_loss(
    strategy_logits: Tensor,
    prev_z: Tensor,
    persist_mask: Tensor,
    *,
    lam_p: float,
    device: torch.device,
) -> Tuple[Tensor, _LossStats]:
    """Mean ``1 - p(z_t = z_{t-1})`` over ``persist_mask``, scaled by ``lam_p``.

    Returns the **un-coefficient'd** persist value via stats, and the
    coefficient-applied tensor as the loss so the caller can add it to the
    aggregated ``latent_loss`` directly.

    When ``persist_mask`` is all-False the persist value is the zero scalar,
    matching the trainer's historical fallback.
    """
    switch = expected_strategy_switch_penalty(strategy_logits, prev_z)
    if persist_mask.any():
        persist = switch[persist_mask].mean()
    else:
        persist = _zero_scalar(device)
    return float(lam_p) * persist, {"persist_term": float(persist.detach().cpu().item())}


def strategy_kl_consecutive_loss(
    z_logits: Tensor,
    z_logits_prev: Tensor,
    valid_mask: Tensor,
    *,
    coef: float,
) -> Tuple[Tensor, _LossStats]:
    """KL( q_phi(z|s_t) || q_phi(z|s_{t-1}) ) averaged over ``valid_mask``.

    When ``coef <= 0`` returns a zero loss and emits ``kl_mean=0.0`` without
    forming the KL graph — matching the trainer's short-circuit in the
    ``latent_kl_consecutive == 0.0`` branch.
    """
    device = z_logits.device
    if coef <= 0.0:
        return _zero_scalar(device), {"kl_mean": 0.0}
    log_p = F.log_softmax(z_logits, dim=-1)
    log_q = F.log_softmax(z_logits_prev.detach(), dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)
    v = valid_mask.float()
    denom = v.sum().clamp_min(1.0)
    kl_mean = (kl * v).sum() / denom
    return float(coef) * kl_mean, {"kl_mean": float(kl_mean.detach().cpu().item())}


def strategy_phase_aux_loss(
    phase_logits: Tensor,
    phase_target: Tensor,
    *,
    coef: float,
) -> Tuple[Tensor, _LossStats]:
    """Cross-entropy auxiliary loss predicting game phase from strategy logits.

    Returns ``coef * CE(phase_logits, phase_target)`` and the un-scaled CE as
    a stat. Short-circuits to zero loss with ``phase_term=0.0`` when
    ``coef <= 0`` — matching the trainer.
    """
    device = phase_logits.device
    if coef <= 0.0:
        return _zero_scalar(device), {"phase_term": 0.0}
    ce = F.cross_entropy(phase_logits, phase_target.long())
    return float(coef) * ce, {"phase_term": float(ce.detach().cpu().item())}


def strategy_ppo_loss(
    strategy_log_prob: Tensor,
    strategy_log_prob_old: Tensor,
    advantages: Tensor,
    resample_mask: Tensor,
    *,
    clip_range: float,
    coef: float,
    device: torch.device,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Clipped PPO loss on z applied to the resample subset only.

    The caller passes ``advantages`` that have already been (globally)
    normalized for the action loss. This function re-normalizes across the
    resample subset before scoring, matching the trainer's historical
    behavior.

    Returns ``(coef * policy_loss, ppo_stats)``. ``ppo_stats`` is the dict
    produced by :func:`rl.ppo_core.ppo_policy_loss` augmented with
    ``"policy_loss"`` so callers can log the un-scaled value.
    """
    if not bool(resample_mask.any()):
        return _zero_scalar(device), {
            "approx_kl": _zero_scalar(device),
            "clip_fraction": _zero_scalar(device),
            "ratio": torch.ones((1,), dtype=torch.float32, device=device),
            "policy_loss": _zero_scalar(device),
        }
    strategy_adv = advantages[resample_mask].detach()
    if strategy_adv.numel() > 1:
        strategy_adv = (strategy_adv - strategy_adv.mean()) / (
            strategy_adv.std(unbiased=False) + 1e-8
        )
    pol_loss, stats = ppo_policy_loss(
        strategy_log_prob[resample_mask],
        strategy_log_prob_old[resample_mask],
        strategy_adv,
        float(clip_range),
    )
    out_stats: Dict[str, Tensor] = dict(stats)
    out_stats["policy_loss"] = pol_loss
    return float(coef) * pol_loss, out_stats


def strategy_aux_return_loss(
    pred_all: Tensor,
    z: Tensor,
    returns_normalized: Tensor,
    resample_mask: Tensor,
    *,
    latent_k: int,
    coef: float,
    device: torch.device,
) -> Tuple[Tensor, _LossStats]:
    """MSE between predicted per-z return and normalized return at resample steps.

    ``pred_all`` has shape ``(B, K)``; we gather the column corresponding to
    the sampled ``z`` at each resample-step row.

    Short-circuits to zero loss when ``coef <= 0`` or no rows are masked —
    matching the trainer.
    """
    if coef <= 0.0 or not bool(resample_mask.any()):
        return _zero_scalar(device), {"aux_return_term": 0.0}
    z_sel = z[resample_mask].long().clamp(min=0, max=int(latent_k) - 1)
    pred_selected = pred_all[resample_mask].gather(1, z_sel.reshape(-1, 1)).squeeze(1)
    mse = F.mse_loss(pred_selected, returns_normalized)
    return float(coef) * mse, {"aux_return_term": float(mse.detach().cpu().item())}


def compute_v6i5_router_loss(
    model,
    *,
    router_contexts: Tensor,
    previous_router_contexts: Tensor,
    executed_z: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    opportunity_mask: Tensor,
    persistence_mask: Tensor,
    clip_range: float,
    latent_k: int,
    ppo_coef: float,
    persistence_coef: float,
    entropy_coef: float,
    entropy_objective: str,
    include_rollout_marginal_entropy: bool,
    device: torch.device,
) -> tuple[Tensor, dict[str, Tensor | float]]:
    """Authoritative v6i5 q_phi loss over corrected router contexts.

    ``router_contexts`` and ``previous_router_contexts`` are 68-wide
    ``current_34 || delta_34`` rows captured during rollout collection. Actor
    and critic context stay separate; this helper only trains q_phi.
    """
    zero = _zero_scalar(device)
    if router_contexts.numel() == 0:
        return zero, {
            "policy_loss": zero,
            "approx_kl": zero,
            "clip_fraction": zero,
            "ratio": torch.ones((1,), dtype=torch.float32, device=device),
            "persistence_loss": zero,
            "marginal_entropy_nats": 0.0,
            "marginal_entropy_kl": 0.0,
            "conditional_entropy_nats": 0.0,
            "application_count": 0.0,
            "row_count": 0.0,
            "effective_coefficient": 0.0,
        }

    logits = model.strategy_logits(router_contexts)
    previous_logits = model.strategy_logits(previous_router_contexts)
    probs = torch.softmax(logits, dim=-1)
    previous_probs = torch.softmax(previous_logits.detach(), dim=-1)

    if include_rollout_marginal_entropy:
        entropy_loss, entropy_stats = rollout_marginal_entropy_loss(
            logits,
            objective=entropy_objective,
            lam_h=float(entropy_coef),
            latent_k=int(latent_k),
            device=device,
        )
        return entropy_loss, {
            "policy_loss": zero,
            "approx_kl": zero,
            "clip_fraction": zero,
            "ratio": torch.ones((1,), dtype=torch.float32, device=device),
            "persistence_loss": zero,
            "marginal_entropy_nats": float(entropy_stats["rollout_marginal_entropy_nats"]),
            "marginal_entropy_kl": float(entropy_stats["rollout_marginal_entropy_kl"]),
            "conditional_entropy_nats": 0.0,
            "application_count": 1.0,
            "row_count": float(entropy_stats["rollout_resample_count"]),
            "effective_coefficient": float(entropy_coef),
        }

    dist = torch.distributions.Categorical(logits=logits)
    z = executed_z.long().clamp(min=0, max=int(latent_k) - 1)
    strategy_log_prob = dist.log_prob(z)
    policy_loss, ppo_stats = strategy_ppo_loss(
        strategy_log_prob,
        old_log_probs,
        advantages,
        opportunity_mask.bool(),
        clip_range=float(clip_range),
        coef=float(ppo_coef),
        device=device,
    )

    expected_stay = (probs * previous_probs).sum(dim=-1)
    persist_rows = persistence_mask.bool()
    if bool(persist_rows.any()):
        persist_raw = (1.0 - expected_stay[persist_rows]).mean()
    else:
        persist_raw = zero
    persistence_loss = float(persistence_coef) * persist_raw
    total = policy_loss + persistence_loss
    return total, {
        "policy_loss": ppo_stats["policy_loss"],
        "approx_kl": ppo_stats["approx_kl"],
        "clip_fraction": ppo_stats["clip_fraction"],
        "ratio": ppo_stats["ratio"],
        "persistence_loss": persist_raw,
        "marginal_entropy_nats": 0.0,
        "marginal_entropy_kl": 0.0,
        "conditional_entropy_nats": 0.0,
        "application_count": 0.0,
        "row_count": 0.0,
        "effective_coefficient": 0.0,
    }


__all__ = [
    "compute_v6i5_router_loss",
    "rollout_marginal_entropy_loss",
    "rollout_router_soft_diagnostics",
    "strategy_entropy_loss",
    "strategy_marginal_entropy_loss",
    "strategy_persistence_loss",
    "strategy_kl_consecutive_loss",
    "strategy_phase_aux_loss",
    "strategy_ppo_loss",
    "strategy_aux_return_loss",
]
