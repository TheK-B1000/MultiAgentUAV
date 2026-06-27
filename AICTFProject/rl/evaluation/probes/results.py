"""Canonical probe result types for policy diagnostic evaluation.

Design
------
``ProbeResult[T]`` is the generic envelope.  ``T`` is a frozen dataclass
of measured metrics (``WeightProbeMetrics``, ``GradientProbeMetrics``,
``CounterfactualProbeMetrics``).

Invariants enforced in ``__post_init__``:
* SUCCESS requires ``metrics`` and no ``error``.
* ERROR requires ``error`` and no ``metrics``.
* A failed probe cannot contain zero-filled metrics.

The probe status strings are lowercase enum members (``"success"``,
``"error"``) following the convention used elsewhere in the project.

Backward compatibility
----------------------
The old flat dataclasses (``WeightProbeResult``, ``GradientProbeResult``,
``CounterfactualProbeResult``) from ``rl.custom_ppo.probe_result`` remain
importable from that path.  New evaluation code should import from here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, Optional, TypeVar


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------

class ProbeStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Error envelope
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProbeError:
    error_type: str
    message: str
    traceback: Optional[str] = None

    @classmethod
    def from_exception(cls, exc: BaseException, tb: Optional[str] = None) -> "ProbeError":
        return cls(
            error_type=type(exc).__name__,
            message=str(exc),
            traceback=tb,
        )

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "error_type": self.error_type,
            "message": self.message,
        }
        if self.traceback is not None:
            d["traceback"] = self.traceback
        return d


# ---------------------------------------------------------------------------
# Typed metric payloads
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WeightProbeMetrics:
    has_obstacle_channel: bool
    cnn_channels: int
    obstacle_weight_l2: Optional[float]
    obstacle_weight_abs_mean: Optional[float]
    obstacle_weight_abs_max: Optional[float]
    obstacle_weight_nonzero_fraction: Optional[float]

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "has_obstacle_channel": self.has_obstacle_channel,
            "cnn_channels": self.cnn_channels,
        }
        if self.obstacle_weight_l2 is not None:
            d["obstacle_weight_l2"] = self.obstacle_weight_l2
        if self.obstacle_weight_abs_mean is not None:
            d["obstacle_weight_abs_mean"] = self.obstacle_weight_abs_mean
        if self.obstacle_weight_abs_max is not None:
            d["obstacle_weight_abs_max"] = self.obstacle_weight_abs_max
        if self.obstacle_weight_nonzero_fraction is not None:
            d["obstacle_weight_nonzero_fraction"] = self.obstacle_weight_nonzero_fraction
        return d


@dataclass(frozen=True)
class GradientProbeMetrics:
    obstacle_gradient_l2: float
    obstacle_gradient_abs_mean: float
    obstacle_gradient_max: float
    obstacle_gradient_nonzero_fraction: float
    sampled_state_count: int
    diagnostic_loss: Optional[float] = None

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "obstacle_gradient_l2": self.obstacle_gradient_l2,
            "obstacle_gradient_abs_mean": self.obstacle_gradient_abs_mean,
            "obstacle_gradient_max": self.obstacle_gradient_max,
            "obstacle_gradient_nonzero_fraction": self.obstacle_gradient_nonzero_fraction,
            "sampled_state_count": self.sampled_state_count,
        }
        if self.diagnostic_loss is not None:
            d["diagnostic_loss"] = self.diagnostic_loss
        return d


@dataclass(frozen=True)
class CounterfactualProbeMetrics:
    mean_action_kl: float
    max_action_kl: float
    mean_logit_l2: float
    max_logit_l2: float
    argmax_action_change_rate: float
    sampled_state_count: int
    observation_tensor: Optional[str] = None
    obstacle_nonzero_fraction: Optional[float] = None
    obstacle_standard_deviation: Optional[float] = None

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "mean_action_kl": self.mean_action_kl,
            "max_action_kl": self.max_action_kl,
            "mean_logit_l2": self.mean_logit_l2,
            "max_logit_l2": self.max_logit_l2,
            "argmax_action_change_rate": self.argmax_action_change_rate,
            "sampled_state_count": self.sampled_state_count,
        }
        if self.observation_tensor is not None:
            d["observation_tensor"] = self.observation_tensor
        if self.obstacle_nonzero_fraction is not None:
            d["obstacle_nonzero_fraction"] = self.obstacle_nonzero_fraction
        if self.obstacle_standard_deviation is not None:
            d["obstacle_standard_deviation"] = self.obstacle_standard_deviation
        return d


# ---------------------------------------------------------------------------
# Generic envelope
# ---------------------------------------------------------------------------

T = TypeVar("T")


@dataclass(frozen=True)
class ProbeResult(Generic[T]):
    """Typed probe result envelope.

    Invariants
    ----------
    * ``SUCCESS`` requires ``metrics is not None`` and ``error is None``.
    * ``ERROR`` requires ``error is not None`` and ``metrics is None``.
    * An error result cannot contain any metric values — not even zeroes.
    """

    status: ProbeStatus
    metrics: Optional[T] = None
    error: Optional[ProbeError] = None

    def __post_init__(self) -> None:
        if self.status is ProbeStatus.SUCCESS:
            if self.metrics is None:
                raise ValueError("SUCCESS result must include metrics.")
            if self.error is not None:
                raise ValueError("SUCCESS result must not include an error.")
        elif self.status is ProbeStatus.ERROR:
            if self.error is None:
                raise ValueError("ERROR result must include a ProbeError.")
            if self.metrics is not None:
                raise ValueError(
                    "ERROR result must not include metrics — "
                    "metrics=None prevents silent zero-measurement conversion."
                )

    @property
    def is_success(self) -> bool:
        return self.status is ProbeStatus.SUCCESS

    def to_json_dict(self) -> dict[str, object]:
        d: dict[str, object] = {"status": str(self.status)}
        if self.metrics is not None and hasattr(self.metrics, "to_dict"):
            d.update(self.metrics.to_dict())  # type: ignore[union-attr]
        if self.error is not None:
            d.update(self.error.to_dict())
        return d

    @classmethod
    def success(cls, metrics: T) -> "ProbeResult[T]":
        return cls(status=ProbeStatus.SUCCESS, metrics=metrics)

    @classmethod
    def from_exception(
        cls, exc: BaseException, tb: Optional[str] = None
    ) -> "ProbeResult[T]":
        return cls(
            status=ProbeStatus.ERROR,
            error=ProbeError.from_exception(exc, tb),
        )


# Typed aliases for the three probe kinds
WeightProbeResult = ProbeResult[WeightProbeMetrics]
GradientProbeResult = ProbeResult[GradientProbeMetrics]
CounterfactualProbeResult = ProbeResult[CounterfactualProbeMetrics]


__all__ = [
    "ProbeStatus",
    "ProbeError",
    "WeightProbeMetrics",
    "GradientProbeMetrics",
    "CounterfactualProbeMetrics",
    "ProbeResult",
    "WeightProbeResult",
    "GradientProbeResult",
    "CounterfactualProbeResult",
]
