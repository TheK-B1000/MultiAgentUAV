"""Optimizer-owned gradient clipping."""

from __future__ import annotations

import torch


def clip_optimizer_grad_norm(optimizer: torch.optim.Optimizer, max_norm: float) -> float:
    """Clip gradients for parameters owned by this optimizer only."""
    params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
    if not params:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(params, float(max_norm)))
