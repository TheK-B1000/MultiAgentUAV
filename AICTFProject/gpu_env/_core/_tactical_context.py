"""Lightweight tactical snapshot for a single environment index.

Used by ctfviewer and tests — not imported during training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._core._state import _StateMixin  # pragma: no cover


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
    )
