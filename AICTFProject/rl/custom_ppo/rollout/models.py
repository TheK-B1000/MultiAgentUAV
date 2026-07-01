"""Shared data models for rollout collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class StepFrame:
    """All per-step inputs needed to fill one row of the rollout buffer."""

    obs: Dict[str, np.ndarray]
    context_state: torch.Tensor
    decision_global_state_np: np.ndarray
    actions_t: torch.Tensor
    log_probs_t: torch.Tensor
    values_t: torch.Tensor
    values_norm_t: torch.Tensor
    next_values_t: torch.Tensor
    reward_component: Dict[str, torch.Tensor]
    terminated: np.ndarray
    truncated: np.ndarray
    opp_row: torch.Tensor
    infos: List[Dict[str, Any]]
    strategy_aux: Optional[Dict[str, torch.Tensor]] = None
    behavior_telemetry: Optional[torch.Tensor] = None
    spread_bucket: Optional[torch.Tensor] = None
    role_bucket: Optional[torch.Tensor] = None
    pressure_bucket: Optional[torch.Tensor] = None
    attack_defense_ratio_bucket: Optional[torch.Tensor] = None
    blue_ahead: Optional[torch.Tensor] = None
    message_aux: Optional[Dict[str, torch.Tensor]] = None


__all__ = ["StepFrame"]
