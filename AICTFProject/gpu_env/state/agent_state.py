"""Agent position, speed, and liveness state — allocation and respawn.

Owns ``_alloc_agent_state`` (tensor allocation) and ``_respawn_side``
(per-episode random initial placement for blue and red agents).
"""
from __future__ import annotations

import math
from typing import Optional

import torch


class _AgentStateMixin:
    """Manages per-agent kinematic and liveness tensors."""

    @property
    def blue_pos(self) -> torch.Tensor:
        """Backward-compatible ``(..., 2)`` view of blue agent positions."""
        return torch.stack((self.blue_x, self.blue_y), dim=-1)

    @property
    def red_pos(self) -> torch.Tensor:
        """Backward-compatible ``(..., 2)`` view of red agent positions."""
        return torch.stack((self.red_x, self.red_y), dim=-1)

    def _alloc_agent_state(
        self,
        B: int,
        Nb: int,
        Nr: int,
        dev: torch.device,
        f32: torch.dtype,
    ) -> None:
        self.blue_x = torch.zeros((B, Nb), dtype=f32, device=dev)
        self.blue_y = torch.zeros((B, Nb), dtype=f32, device=dev)
        self.blue_heading = torch.zeros((B, Nb), dtype=f32, device=dev)
        self.blue_speed = torch.zeros((B, Nb), dtype=f32, device=dev)
        self.blue_alive = torch.ones((B, Nb), dtype=torch.bool, device=dev)
        self.blue_tagged = torch.zeros((B, Nb), dtype=torch.bool, device=dev)
        self.blue_carrying = torch.zeros((B, Nb), dtype=torch.bool, device=dev)
        self.blue_respawn = torch.zeros((B, Nb), dtype=f32, device=dev)

        self.red_x = torch.zeros((B, Nr), dtype=f32, device=dev)
        self.red_y = torch.zeros((B, Nr), dtype=f32, device=dev)
        self.red_heading = torch.zeros((B, Nr), dtype=f32, device=dev)
        self.red_speed = torch.zeros((B, Nr), dtype=f32, device=dev)
        self.red_alive = torch.ones((B, Nr), dtype=torch.bool, device=dev)
        self.red_tagged = torch.zeros((B, Nr), dtype=torch.bool, device=dev)
        self.red_carrying = torch.zeros((B, Nr), dtype=torch.bool, device=dev)
        self.red_respawn = torch.zeros((B, Nr), dtype=f32, device=dev)

    def _respawn_side(self, blue: bool, env_mask: Optional[torch.Tensor] = None) -> None:
        if env_mask is None:
            env_mask = torch.ones((self.B,), dtype=torch.bool, device=self.device)
        idx = torch.where(env_mask)[0]
        if idx.numel() == 0:
            return
        E = int(idx.numel())
        # NOTE: ``self.blue_speed[idx].zero_()`` is a PyTorch *no-op* when
        # ``idx`` is a LongTensor (advanced indexing returns a fresh tensor,
        # whose ``.zero_()`` does not reach back to the original storage).
        # Use ``self.blue_speed[idx] = 0.0`` instead, which goes through the
        # in-place ``__setitem__`` path.  The old pattern silently left blue/red
        # speed, heading, alive, and respawn carrying over from the last frame
        # of the prior episode — breaking the matched-start contract that
        # q_probe / local CF rely on.
        if blue:
            x_lo, x_hi = 0.0, max(1.0, float(self.cols // 3 - 1))
            self.blue_x[idx] = self._rand_uniform((E, self.Nb), x_lo, x_hi)
            self.blue_y[idx] = self._rand_uniform((E, self.Nb), 0.0, float(max(0, self.rows - 1)))
            self.blue_heading[idx] = 0.0
            self.blue_speed[idx] = 0.0
            self.blue_alive[idx] = True
            self.blue_tagged[idx] = False
            self.blue_carrying[idx] = False
            self.blue_respawn[idx] = 0
        else:
            x_lo = max(0.0, float(self.cols - max(1, self.cols // 3)))
            x_hi = float(max(0, self.cols - 1))
            self.red_x[idx] = self._rand_uniform((E, self.Nr), x_lo, x_hi)
            self.red_y[idx] = self._rand_uniform((E, self.Nr), 0.0, float(max(0, self.rows - 1)))
            self.red_heading[idx] = math.pi
            self.red_speed[idx] = 0.0
            self.red_alive[idx] = True
            self.red_tagged[idx] = False
            self.red_carrying[idx] = False
            self.red_respawn[idx] = 0
