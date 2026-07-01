"""Team-perspective utilities: symmetric tensor access and coordinate mirroring.

``_side_tensors`` returns a side-agnostic view of all agent/flag/mine tensors
so that blue and red code paths can be written symmetrically.  ``_mirror_x``
and ``_mirror_heading`` flip coordinates to the red team's frame of reference.
"""
from __future__ import annotations

import math
from typing import Dict

import torch


class _TeamStateMixin:
    """Provides symmetric side-tensor access and red-frame coordinate mirroring."""

    def _side_tensors(self, side: str) -> Dict[str, torch.Tensor]:
        if side == "red":
            return {
                "own_x": self.red_x,
                "own_y": self.red_y,
                "own_heading": self.red_heading,
                "own_speed": self.red_speed,
                "own_alive": self.red_alive,
                "own_carrying": self.red_carrying,
                "own_flag": self.red_flag_pos,
                "own_flag_home": self.red_flag_home,
                "own_mine_x": self.red_mine_x,
                "own_mine_y": self.red_mine_y,
                "own_mine_active": self.red_mine_active,
                "own_mine_charges": self.red_mine_charges,
                "enemy_x": self.blue_x,
                "enemy_y": self.blue_y,
                "enemy_alive": self.blue_alive,
                "enemy_flag": self.blue_flag_pos,
                "n_agents": torch.tensor(self.Nr, device=self.device),
            }
        return {
            "own_x": self.blue_x,
            "own_y": self.blue_y,
            "own_heading": self.blue_heading,
            "own_speed": self.blue_speed,
            "own_alive": self.blue_alive,
            "own_carrying": self.blue_carrying,
            "own_flag": self.blue_flag_pos,
            "own_flag_home": self.blue_flag_home,
            "own_mine_x": self.blue_mine_x,
            "own_mine_y": self.blue_mine_y,
            "own_mine_active": self.blue_mine_active,
            "own_mine_charges": self.blue_mine_charges,
            "enemy_x": self.red_x,
            "enemy_y": self.red_y,
            "enemy_alive": self.red_alive,
            "enemy_flag": self.red_flag_pos,
            "n_agents": torch.tensor(self.Nb, device=self.device),
        }

    def _mirror_x(self, x: torch.Tensor, side: str) -> torch.Tensor:
        if side == "red":
            return float(max(0, self.cols - 1)) - x
        return x

    def _mirror_heading(self, heading: torch.Tensor, side: str) -> torch.Tensor:
        if side == "red":
            mirrored = math.pi - heading
            return (mirrored + math.pi) % (2.0 * math.pi) - math.pi
        return heading
