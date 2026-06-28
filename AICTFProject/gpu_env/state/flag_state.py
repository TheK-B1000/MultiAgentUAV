"""Flag position and score tensor allocation.

Allocates the four flag tensors (blue/red home + current position) and the two
score counters.  Flag home positions are deterministic: vertical center of the
field, two cells inward from each side edge.
"""
from __future__ import annotations

import torch


class _FlagStateMixin:
    """Owns score and flag position tensor allocation."""

    def _alloc_flags_and_scores(
        self,
        B: int,
        dev: torch.device,
        f32: torch.dtype,
    ) -> None:
        self.blue_score = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.red_score = torch.zeros((B,), dtype=torch.int32, device=dev)

        # Flags at vertical center; two cells inward from edges toward middle.
        home_y = float(self.rows // 2)
        inward = 2.0
        blue_x = min(inward, float(max(0, self.cols - 1)))
        red_x = max(float(self.cols - 1) - inward, 0.0)
        self.blue_flag_home = torch.stack(
            [
                torch.full((B,), blue_x, dtype=f32, device=dev),
                torch.full((B,), home_y, dtype=f32, device=dev),
            ],
            dim=1,
        )
        self.red_flag_home = torch.stack(
            [
                torch.full((B,), red_x, dtype=f32, device=dev),
                torch.full((B,), home_y, dtype=f32, device=dev),
            ],
            dim=1,
        )
        self.blue_flag_pos = self.blue_flag_home.clone()
        self.red_flag_pos = self.red_flag_home.clone()
