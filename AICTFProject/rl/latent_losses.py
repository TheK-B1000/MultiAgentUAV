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


__all__ = [
    "strategy_entropy_loss",
    "strategy_persistence_loss",
    "strategy_kl_consecutive_loss",
    "strategy_phase_aux_loss",
    "strategy_ppo_loss",
    "strategy_aux_return_loss",
]
