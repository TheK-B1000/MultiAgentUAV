from __future__ import annotations

from typing import Any
import torch


def _return_norm_std(trainer: Any) -> float:
    return max(1e-3, float(trainer._return_norm_var) ** 0.5)


def _normalize_value_targets(trainer: Any, returns: torch.Tensor) -> torch.Tensor:
    if not trainer.normalize_returns:
        return returns.float()
    return (returns.float() - float(trainer._return_norm_mean)) / _return_norm_std(trainer)


def _denormalize_values(trainer: Any, values: torch.Tensor) -> torch.Tensor:
    if not trainer.normalize_returns:
        return values.float()
    return values.float() * _return_norm_std(trainer) + float(trainer._return_norm_mean)


def _update_return_norm_stats(trainer: Any, returns: torch.Tensor) -> None:
    if not trainer.normalize_returns:
        return
    values = returns.detach().float().reshape(-1)
    if values.numel() <= 0:
        return
    batch_count = float(values.numel())
    batch_mean = float(values.mean().detach().cpu().item())
    batch_var = float(values.var(unbiased=False).detach().cpu().item()) if values.numel() > 1 else 0.0

    count = float(trainer._return_norm_count)
    delta = batch_mean - float(trainer._return_norm_mean)
    total_count = count + batch_count
    new_mean = float(trainer._return_norm_mean) + delta * batch_count / max(1e-6, total_count)
    m_a = float(trainer._return_norm_var) * count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta * delta * count * batch_count / max(1e-6, total_count)
    trainer._return_norm_mean = new_mean
    trainer._return_norm_var = max(1e-6, m2 / max(1e-6, total_count))
    trainer._return_norm_count = total_count


def _update_strategy_return_stats(trainer: Any, buffer: Any) -> None:
    """Update running return normalization stats for sampled z targets."""
    if not trainer.latent_strategy_aux_return_head or "z_resampled" not in buffer.fields:
        return
    sampled = buffer.fields["z_resampled"][: int(buffer.pos)].reshape(-1).bool()
    returns = buffer.fields["returns"][: int(buffer.pos)].reshape(-1).detach().float()
    if not bool(sampled.any().item()):
        return
    values = returns[sampled]
    batch_count = float(values.numel())
    batch_mean = float(values.mean().detach().cpu().item())
    batch_var = float(values.var(unbiased=False).detach().cpu().item()) if values.numel() > 1 else 0.0

    count = float(trainer._strategy_return_count)
    delta = batch_mean - float(trainer._strategy_return_mean)
    total_count = count + batch_count
    new_mean = float(trainer._strategy_return_mean) + delta * batch_count / max(1e-6, total_count)
    m_a = float(trainer._strategy_return_var) * count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta * delta * count * batch_count / max(1e-6, total_count)
    trainer._strategy_return_mean = new_mean
    trainer._strategy_return_var = max(1e-6, m2 / max(1e-6, total_count))
    trainer._strategy_return_count = total_count


def _normalize_strategy_returns(trainer: Any, returns: torch.Tensor) -> torch.Tensor:
    std = max(1e-3, float(trainer._strategy_return_var) ** 0.5)
    return (returns.detach().float() - float(trainer._strategy_return_mean)) / std
