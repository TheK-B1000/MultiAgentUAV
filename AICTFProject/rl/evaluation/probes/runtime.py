"""Shared runtime dependencies for V6I9 obstacle probes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class ObstacleProbeRuntime:
    make_env: Callable[..., Any]
    model: Callable[[Any], torch.nn.Module]
    policy_device: Callable[[Any, str], torch.device]
    reset_obs: Callable[[Any], Any]
    set_opponent: Callable[[Any, str], str]
    to_torch: Callable[[Any, torch.device], Any]
    zero_obstacle_channel: Callable[[Any], tuple[Any, str]]
    head_argmax_change_rate: Callable[[list[torch.Tensor], list[torch.Tensor]], float]
    predict: Callable[[Any, Any], Any]
    unpack_step: Callable[[Any], tuple[Any, Any, Any, Any]]
    done: Callable[[Any], bool]


__all__ = ["ObstacleProbeRuntime"]
