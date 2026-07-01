"""Lightweight tactical snapshot for a single environment index.

Used by ctfviewer and tests — not imported during training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .._core._state import _StateMixin  # pragma: no cover


@dataclass(frozen=True)
class BTTelemetry:
    """Per-environment behavior-tree telemetry snapshot (lifetime counters)."""
    # Cumulative event counters for this episode.
    escort_attempts: int
    intercept_attempts: int
    counter_captures: int
    objective_changes: int
    successful_tags: int
    stuck_steps: int
    # Current role per red agent (list of ints matching ROLE_* constants in _bt_red.py).
    red_roles: "list[int]"
    # Active BT branch per red agent (same encoding as roles).
    active_branches: "list[int]"


@dataclass(frozen=True)
class TacticalContext:
    env_idx: int
    blue_score: int
    red_score: int
    step: int
    max_steps: int
    # Per-side carrier indices (-1 = no carrier)
    blue_carrier_idx: int
    red_carrier_idx: int
    # Snapshot of red debug targets (may be None if not stored)
    red_target_x: "list[float] | None"
    red_target_y: "list[float] | None"
    # BT telemetry (None when BT opponent not active)
    bt: "BTTelemetry | None" = None

    @property
    def time_remaining_frac(self) -> float:
        return max(0.0, 1.0 - self.step / max(1, self.max_steps))

    @property
    def late_game(self) -> bool:
        return self.time_remaining_frac < 0.25

    @property
    def red_leading(self) -> bool:
        return self.red_score > self.blue_score

    @property
    def red_trailing(self) -> bool:
        return self.red_score < self.blue_score


def extract_tactical_context(core: "_StateMixin", env_idx: int = 0) -> TacticalContext:
    """Extract a TacticalContext snapshot from a live BatchedCTFCore."""
    import torch

    i = env_idx

    def _int(t: "torch.Tensor") -> int:
        return int(t[i].item())

    def _carrier(carrying: "torch.Tensor") -> int:
        c = carrying[i]
        idx = int(c.to(torch.int64).argmax().item())
        return idx if bool(c.any().item()) else -1

    red_tx = None
    red_ty = None
    if hasattr(core, "_debug_red_target_x"):
        red_tx = core._debug_red_target_x[i].tolist()  # type: ignore[attr-defined]
        red_ty = core._debug_red_target_y[i].tolist()  # type: ignore[attr-defined]

    # Extract BT telemetry if available (BT attributes allocated by _BTRedMixin).
    bt_tel: Optional[BTTelemetry] = None
    if hasattr(core, "bt_tel_escort_attempts"):
        bt_tel = BTTelemetry(
            escort_attempts=int(core.bt_tel_escort_attempts[i].item()),   # type: ignore[attr-defined]
            intercept_attempts=int(core.bt_tel_intercept_attempts[i].item()),  # type: ignore[attr-defined]
            counter_captures=int(core.bt_tel_counter_captures[i].item()),  # type: ignore[attr-defined]
            objective_changes=int(core.bt_tel_objective_changes[i].item()),  # type: ignore[attr-defined]
            successful_tags=int(core.bt_tel_successful_tags[i].item()),   # type: ignore[attr-defined]
            stuck_steps=int(core.bt_tel_stuck_steps[i].item()),          # type: ignore[attr-defined]
            red_roles=core.bt_red_role[i].tolist(),                       # type: ignore[attr-defined]
            active_branches=core.bt_active_branch[i].tolist(),            # type: ignore[attr-defined]
        )

    return TacticalContext(
        env_idx=i,
        blue_score=_int(core.blue_score),
        red_score=_int(core.red_score),
        step=_int(core.step_count),
        max_steps=int(core.max_steps),
        blue_carrier_idx=_carrier(core.blue_carrying),
        red_carrier_idx=_carrier(core.red_carrying),
        red_target_x=red_tx,
        red_target_y=red_ty,
        bt=bt_tel,
    )
