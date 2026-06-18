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
    proposed_z: torch.Tensor
    executed_z: torch.Tensor
    router_probs: torch.Tensor
    behavior_probs: torch.Tensor
    behavior_log_prob: torch.Tensor
    router_log_prob: torch.Tensor
    source: RouterActionSource

    @property
    def sampled_z(self) -> torch.Tensor:
        return self.proposed_z


@dataclass(frozen=True)
class OpponentResolution:
    value: int
    valid: bool
    reason: str | None = None


@dataclass(frozen=True)
class EpisodeRouterBatch:
    states: torch.Tensor
    executed_z: torch.Tensor
    old_behavior_log_prob: torch.Tensor
    episode_returns: torch.Tensor
    opponent_ids: torch.Tensor
    bucket_ids: torch.Tensor
    selector_hidden: torch.Tensor | None
    action_sources: tuple[RouterActionSource, ...]

    def __post_init__(self) -> None:
        n = int(self.states.shape[0])
        if int(self.executed_z.shape[0]) != n:
            raise ValueError("executed_z batch size mismatch")
        if int(self.old_behavior_log_prob.shape[0]) != n:
            raise ValueError("old_behavior_log_prob batch size mismatch")
        if int(self.episode_returns.shape[0]) != n:
            raise ValueError("episode_returns batch size mismatch")
        if len(self.action_sources) != n:
            raise ValueError("action_sources length mismatch")
        if not torch.isfinite(self.old_behavior_log_prob).all():
            raise ValueError("old_behavior_log_prob must be finite")
        if not torch.isfinite(self.episode_returns).all():
            raise ValueError("episode_returns must be finite")
        if any(s is RouterActionSource.FORCED_REHEARSAL for s in self.action_sources):
            raise ValueError("forced-rehearsal episodes must not enter on-policy batch")


@dataclass
class LossComponent:
    raw: torch.Tensor
    scaled: torch.Tensor
    active_fraction: float = 0.0


@dataclass
class EpisodeAuxiliaryLossBundle:
    entropy: LossComponent
    usage_balance: LossComponent
    specialist: LossComponent
    preference: LossComponent
    commitment: LossComponent
    awrd: LossComponent
    refresh_preference: LossComponent

    def total_scaled(self) -> torch.Tensor:
        return (
            self.entropy.scaled
            + self.usage_balance.scaled
            + self.specialist.scaled
            + self.preference.scaled
            + self.commitment.scaled
            + self.awrd.scaled
            + self.refresh_preference.scaled
        )


@dataclass
class RouterPPOConfig:
    coef: float = 1.0
    value_coef: float = 0.5
    clip_epsilon: float = 0.2
    epochs: int = 4
    target_kl: float | None = None
    target_kl_multiplier: float = 1.5
    normalize_advantages: bool = True
    max_grad_norm: float = 0.5
    objective_name: str = "episode_credit"


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
