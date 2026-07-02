"""Feedforward router credit selection and telemetry helpers."""

from __future__ import annotations

from typing import Any

import torch


def is_feedforward_sparse_router(cfg: Any) -> bool:
    return bool(getattr(cfg, "router_reward_enabled", False)) and int(
        getattr(cfg, "recurrent_selector_hidden_dim", 0) or 0
    ) == 0


def is_recurrent_router(cfg: Any, model: Any | None = None) -> bool:
    if int(getattr(cfg, "recurrent_selector_hidden_dim", 0) or 0) > 0:
        return True
    if model is not None and getattr(model, "selector_gru", None) is not None:
        return True
    return False


def resolve_strategy_advantages(
    *,
    cfg: Any,
    batch: dict[str, torch.Tensor],
    actor_advantages: torch.Tensor,
) -> tuple[torch.Tensor, str]:
    """Select advantages for feedforward/recurrent strategy PPO."""
    if is_feedforward_sparse_router(cfg):
        if "router_advantages" not in batch:
            raise RuntimeError(
                "router_reward_enabled=True for feedforward router but batch is missing "
                "'router_advantages'. Sparse router credit cannot be applied."
            )
        return batch["router_advantages"], "router"

    if getattr(cfg, "latent_q_phi_option_advantage", False):
        if "option_advantages" not in batch:
            raise RuntimeError(
                "latent_q_phi_option_advantage=True but batch is missing 'option_advantages'."
            )
        return batch["option_advantages"], "option"

    return actor_advantages, "actor_gae"


def router_decision_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if "router_decision_valid" in batch:
        return batch["router_decision_valid"].bool()
    return batch["z_resampled"].bool()


def router_advantage_telemetry(
    batch: dict[str, torch.Tensor],
    decision_mask: torch.Tensor,
) -> dict[str, float]:
    if "router_advantages" not in batch:
        return {
            "router_advantage_mean": 0.0,
            "router_advantage_std": 0.0,
            "router_advantage_positive_fraction": 0.0,
            "router_decision_count": float(decision_mask.float().sum().detach().cpu().item()),
        }
    adv = batch["router_advantages"].float()
    sel = adv[decision_mask] if bool(decision_mask.any()) else adv.reshape(-1)[:0]
    if sel.numel() == 0:
        return {
            "router_advantage_mean": 0.0,
            "router_advantage_std": 0.0,
            "router_advantage_positive_fraction": 0.0,
            "router_decision_count": 0.0,
        }
    return {
        "router_advantage_mean": float(sel.mean().detach().cpu().item()),
        "router_advantage_std": float(sel.std(unbiased=False).detach().cpu().item()),
        "router_advantage_positive_fraction": float((sel > 0).float().mean().detach().cpu().item()),
        "router_decision_count": float(decision_mask.float().sum().detach().cpu().item()),
    }


def strategy_advantage_source_code(source: str) -> float:
    return {"actor_gae": 0.0, "option": 1.0, "router": 2.0}.get(source, -1.0)


def encoder_grad_norm_from_loss(loss: torch.Tensor, encoder: Any) -> float:
    if not bool(getattr(loss, "requires_grad", False)):
        return 0.0
    params = [p for p in encoder.parameters() if p.requires_grad]
    if not params:
        return 0.0
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    sq = 0.0
    for grad in grads:
        if grad is not None:
            sq += float(grad.detach().pow(2).sum().cpu().item())
    return float(sq**0.5)


__all__ = [
    "encoder_grad_norm_from_loss",
    "is_feedforward_sparse_router",
    "is_recurrent_router",
    "resolve_strategy_advantages",
    "router_advantage_telemetry",
    "router_decision_mask",
    "strategy_advantage_source_code",
]
