"""Typed probe result containers for policy diagnostic use.

Every probe function must return one of these dataclasses.  The ``status``
field is always ``"SUCCESS"`` or ``"ERROR"`` — callers **must** check
``result.is_success`` before reading metric fields.

Metric fields are ``None`` (not zero) when the probe could not run.  This
prevents silent conversion of exceptions into valid zero-valued scientific
measurements (constraint #8 of the refactoring spec).

Dependency: no local rl.* imports (stdlib only).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

ProbeStatus = Literal["SUCCESS", "ERROR"]

PROBE_SUCCESS: ProbeStatus = "SUCCESS"
PROBE_ERROR: ProbeStatus = "ERROR"


@dataclass
class ProbeResult:
    """Base class for all probe results."""

    status: ProbeStatus
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.status == PROBE_SUCCESS


@dataclass
class WeightProbeResult(ProbeResult):
    """Result of inspecting first-CNN-layer weights for a specific input channel."""

    has_obstacle_channel: bool = False
    cnn_channels: int = 0
    obstacle_weight_l2: Optional[float] = None
    obstacle_weight_abs_mean: Optional[float] = None
    obstacle_weight_abs_max: Optional[float] = None
    obstacle_weight_nonzero_fraction: Optional[float] = None

    def to_json_dict(self) -> Dict[str, object]:
        d: Dict[str, object] = {
            "has_obstacle_channel": self.has_obstacle_channel,
            "cnn_channels": self.cnn_channels,
            "obstacle_weight_l2": self.obstacle_weight_l2,
            "obstacle_weight_abs_mean": self.obstacle_weight_abs_mean,
            "obstacle_weight_abs_max": self.obstacle_weight_abs_max,
            "obstacle_weight_nonzero_fraction": self.obstacle_weight_nonzero_fraction,
        }
        if self.error is not None:
            d["error"] = self.error
        return d


@dataclass
class GradientProbeResult(ProbeResult):
    """Result of a gradient-flow probe through the CNN obstacle channel."""

    obstacle_gradient_l2: Optional[float] = None
    obstacle_gradient_abs_mean: Optional[float] = None
    diagnostic_loss: Optional[float] = None

    def to_json_dict(self) -> Dict[str, object]:
        d: Dict[str, object] = {}
        if self.obstacle_gradient_l2 is not None:
            d["obstacle_gradient_l2"] = self.obstacle_gradient_l2
        if self.obstacle_gradient_abs_mean is not None:
            d["obstacle_gradient_abs_mean"] = self.obstacle_gradient_abs_mean
        if self.diagnostic_loss is not None:
            d["diagnostic_loss"] = self.diagnostic_loss
        if self.error is not None:
            d["error"] = self.error
        return d


@dataclass
class CounterfactualProbeResult(ProbeResult):
    """Result of comparing real vs. zeroed-obstacle-channel action distributions."""

    states_evaluated: int = 0
    observation_tensor: Optional[str] = None
    mean_action_kl: Optional[float] = None
    max_action_kl: Optional[float] = None
    mean_logit_l2: Optional[float] = None
    max_logit_l2: Optional[float] = None
    argmax_action_change_rate: Optional[float] = None

    def to_json_dict(self) -> Dict[str, object]:
        d: Dict[str, object] = {"states_evaluated": self.states_evaluated}
        if self.observation_tensor is not None:
            d["observation_tensor"] = self.observation_tensor
        for attr in (
            "mean_action_kl",
            "max_action_kl",
            "mean_logit_l2",
            "max_logit_l2",
            "argmax_action_change_rate",
        ):
            v = getattr(self, attr)
            if v is not None:
                d[attr] = v
        if self.error is not None:
            d["error"] = self.error
        return d


__all__ = [
    "ProbeStatus",
    "PROBE_SUCCESS",
    "PROBE_ERROR",
    "ProbeResult",
    "WeightProbeResult",
    "GradientProbeResult",
    "CounterfactualProbeResult",
]
