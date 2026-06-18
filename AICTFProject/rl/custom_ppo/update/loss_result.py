"""Typed loss / telemetry results for one minibatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class PairwiseSeparationMeasurement:
    """CF-batch pair JSD vector with explicit validity (never infer from CSV zeros)."""

    values: torch.Tensor | None
    valid: bool
    reason: str | None
    active_fraction: float
    valid_groups: int

    def as_list(self) -> list[float] | None:
        if not self.valid or self.values is None:
            return None
        return [float(v) for v in self.values.detach().cpu().tolist()]


@dataclass
class LossComponent:
    name: str
    scaled_loss: torch.Tensor
    raw_value: torch.Tensor
    active: bool
    metrics: dict[str, float | torch.Tensor] = field(default_factory=dict)


@dataclass
class MinibatchUpdateResult:
    policy: LossComponent
    value: LossComponent
    entropy: LossComponent
    latent_components: tuple[LossComponent, ...]

    action_kl: float
    strategy_kl: float
    should_stop: bool
    stop_reason: str | None

    grad_norms: dict[str, float]
    telemetry: dict[str, float]
    separation_measurement: PairwiseSeparationMeasurement | None = None

    def total_loss(self, *, vf_coef: float, ent_coef: float) -> torch.Tensor:
        total = (
            self.policy.scaled_loss
            + float(vf_coef) * self.value.scaled_loss
            + float(ent_coef) * self.entropy.scaled_loss
        )
        for component in self.latent_components:
            if component.active:
                total = total + component.scaled_loss
        return total


def measurement_from_pair_tensor(
    pair_jsd: torch.Tensor | None,
    *,
    active_fraction: float,
    valid_groups: int,
    reason: str | None = None,
) -> PairwiseSeparationMeasurement:
    if pair_jsd is None or int(pair_jsd.numel()) <= 0:
        return PairwiseSeparationMeasurement(
            values=None,
            valid=False,
            reason=reason or "missing_pair_jsd",
            active_fraction=float(active_fraction),
            valid_groups=int(valid_groups),
        )
    return PairwiseSeparationMeasurement(
        values=pair_jsd.detach(),
        valid=True,
        reason=None,
        active_fraction=float(active_fraction),
        valid_groups=int(valid_groups),
    )
