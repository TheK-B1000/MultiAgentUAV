"""Shared types for latent router sampling and credit records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class RouterActionSource(str, Enum):
    ROUTER = "router"
    EPSILON_MIXTURE = "epsilon_mixture"
    FORCED_REHEARSAL = "forced_rehearsal"
    FIXED = "fixed"
    EVENT_REFRESH = "event_refresh"


@dataclass(frozen=True)
class RouterAction:
    sampled_z: torch.Tensor
    executed_z: torch.Tensor
    router_probs: torch.Tensor
    behavior_probs: torch.Tensor
    behavior_log_prob: torch.Tensor
    router_log_prob: torch.Tensor
    source: RouterActionSource


@dataclass(frozen=True)
class RouterPPOConfig:
    coef: float = 1.0
    value_coef: float = 0.5
    clip_epsilon: float = 0.2
    epochs: int = 4
    target_kl: float | None = None
    normalize_advantages: bool = True


@dataclass(frozen=True)
class RouterPPOBatch:
    states: torch.Tensor
    executed_z: torch.Tensor
    old_behavior_log_prob: torch.Tensor
    fixed_advantages: torch.Tensor
    returns: torch.Tensor
    selector_hidden: torch.Tensor | None = None


@dataclass(frozen=True)
class RouterStepResult:
    stepped: bool
    grad_norm: float
    finite: bool
    optimizer_steps: int
    grad_splits: dict[str, float] | None = None
    q_phi_entropy: float = 0.0
    q_phi_mean_max_prob: float = 0.0


class LifecycleError(RuntimeError):
    """Illegal episode lifecycle transition."""
