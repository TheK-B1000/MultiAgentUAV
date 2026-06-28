"""Map layout state: obstacle geometry, collision detection, and pathfinding.

Handles the per-episode split-lane wall (Map B variants) including tensor
allocation, per-episode reset (mirroring), axis-aligned collision detection,
wall-slide physics, and waypoint routing around obstacles.
"""
from __future__ import annotations

from typing import Tuple

import torch

from .._maps import (
    MAP_B_SPLIT_LANE_V2,
    is_split_lane_layout,
    norm_rect_to_cells,
    split_lane_rect_norm,
    split_lane_v2_rect_norm,
)


class _MapStateMixin:
    """Owns all map-layout and obstacle state: allocation, reset, and geometry queries."""

    def _alloc_map_state(
        self,
        B: int,
        dev: torch.device,
        f32: torch.dtype,
    ) -> None:
        self.max_obstacles = 1
        self.map_vertical_mirror = torch.zeros((B,), dtype=torch.bool, device=dev)
        self.obstacle_rects = torch.zeros((B, self.max_obstacles, 4), dtype=f32, device=dev)
        self.obstacle_active = torch.zeros((B, self.max_obstacles), dtype=torch.bool, device=dev)

    def _reset_map_layout(self, env_mask: torch.Tensor) -> None:
        idx = torch.where(env_mask)[0]
        if idx.numel() == 0:
            return
        self.map_vertical_mirror[idx] = False
        self.obstacle_rects[idx] = 0.0
        self.obstacle_active[idx] = False
        if not is_split_lane_layout(self.map_layout):
            return
        mirror_p = max(
            0.0, min(1.0, float(getattr(self.cfg, "map_b_vertical_mirror_prob", 0.5)))
        )
        mirrors = torch.rand((idx.numel(),), generator=self._rng, device=self.device) < mirror_p
        self.map_vertical_mirror[idx] = mirrors
        if self.map_layout == MAP_B_SPLIT_LANE_V2:
            base_norm = split_lane_v2_rect_norm(mirror_y=False)
            mirror_norm = split_lane_v2_rect_norm(mirror_y=True)
        else:
            base_norm = split_lane_rect_norm(
                x_min=float(self.cfg.map_b_wall_x_min_norm),
                x_max=float(self.cfg.map_b_wall_x_max_norm),
                y_min=float(self.cfg.map_b_wall_y_min_norm),
                y_max=float(self.cfg.map_b_wall_y_max_norm),
                mirror_y=False,
            )
            mirror_norm = split_lane_rect_norm(
                x_min=float(self.cfg.map_b_wall_x_min_norm),
                x_max=float(self.cfg.map_b_wall_x_max_norm),
                y_min=float(self.cfg.map_b_wall_y_min_norm),
                y_max=float(self.cfg.map_b_wall_y_max_norm),
                mirror_y=True,
            )
        base_rect = norm_rect_to_cells(base_norm, cols=self.cols, rows=self.rows)
        mirror_rect = norm_rect_to_cells(mirror_norm, cols=self.cols, rows=self.rows)
        base = torch.tensor(base_rect, dtype=self.obstacle_rects.dtype, device=self.device)
        mirrored = torch.tensor(mirror_rect, dtype=self.obstacle_rects.dtype, device=self.device)
        rects = torch.where(mirrors[:, None], mirrored[None, :], base[None, :])
        self.obstacle_rects[idx, 0, :] = rects
        self.obstacle_active[idx, 0] = True

    def _points_in_obstacles(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        if not hasattr(self, "obstacle_active") or not bool(self.obstacle_active.any().item()):
            return torch.zeros_like(x, dtype=torch.bool)
        rects = self.obstacle_rects.to(dtype=x.dtype, device=x.device)
        active = self.obstacle_active.to(device=x.device)
        px = x[..., None]
        py = y[..., None]
        inside = (
            active[:, None, :]
            & (px >= rects[:, None, :, 0])
            & (px <= rects[:, None, :, 2])
            & (py >= rects[:, None, :, 1])
            & (py <= rects[:, None, :, 3])
        )
        return inside.any(dim=-1)

    def _segments_hit_obstacles(
        self,
        prev_x: torch.Tensor,
        prev_y: torch.Tensor,
        next_x: torch.Tensor,
        next_y: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(self, "obstacle_active") or not bool(self.obstacle_active.any().item()):
            return torch.zeros_like(next_x, dtype=torch.bool)
        hit = torch.zeros_like(next_x, dtype=torch.bool)
        for frac in (0.25, 0.5, 0.75, 1.0):
            sx = prev_x + (next_x - prev_x) * float(frac)
            sy = prev_y + (next_y - prev_y) * float(frac)
            hit = hit | self._points_in_obstacles(sx, sy)
        return hit

    def _revert_obstacle_hits(
        self,
        prev_x: torch.Tensor,
        prev_y: torch.Tensor,
        next_x: torch.Tensor,
        next_y: torch.Tensor,
        speed: torch.Tensor,
        alive: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hit = self._segments_hit_obstacles(prev_x, prev_y, next_x, next_y) & alive
        if not bool(hit.any().item()):
            return next_x, next_y, speed, hit

        # Wall sliding: try axis-aligned moves before doing a full positional revert.
        # Prevents corner-sticking where the agent freezes with speed=0 for many
        # steps while slowly turning away from the wall.
        x_blocked = self._segments_hit_obstacles(prev_x, prev_y, next_x, prev_y)
        y_blocked = self._segments_hit_obstacles(prev_x, prev_y, prev_x, next_y)

        can_slide_x = hit & ~x_blocked
        can_slide_y = hit & ~y_blocked & ~can_slide_x
        full_revert = hit & ~can_slide_x & ~can_slide_y

        x_out = torch.where(
            can_slide_x,
            next_x,
            torch.where(can_slide_y | full_revert, prev_x, next_x),
        )
        y_out = torch.where(
            can_slide_x,
            prev_y,
            torch.where(can_slide_y, next_y, torch.where(full_revert, prev_y, next_y)),
        )
        speed_out = torch.where(hit, torch.zeros_like(speed), speed)
        return x_out, y_out, speed_out, hit

    def _route_targets_around_obstacles(
        self,
        own_x: torch.Tensor,
        own_y: torch.Tensor,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not is_split_lane_layout(self.map_layout) or not bool(
            self.obstacle_active.any().item()
        ):
            return target_x, target_y
        rect = self.obstacle_rects[:, 0, :].to(dtype=target_x.dtype, device=target_x.device)
        active = self.obstacle_active[:, 0].to(device=target_x.device)
        x0 = rect[:, 0:1]
        y0 = rect[:, 1:2]
        x1 = rect[:, 2:3]
        y1 = rect[:, 3:4]
        center_y = (y0 + y1) * 0.5

        denom_x = target_x - own_x
        safe_denom_x = torch.where(
            torch.abs(denom_x) < 1e-6, torch.full_like(denom_x, 1e-6), denom_x
        )
        wall_x = (x0 + x1) * 0.5
        t = (wall_x - own_x) / safe_denom_x
        line_y = own_y + (target_y - own_y) * t
        crosses_wall_x = (t >= 0.0) & (t <= 1.0)
        crosses_blocked_y = (line_y >= y0) & (line_y <= y1)

        denom_y = target_y - own_y
        safe_denom_y = torch.where(
            torch.abs(denom_y) < 1e-6, torch.full_like(denom_y, 1e-6), denom_y
        )
        t_top = (y0 - own_y) / safe_denom_y
        line_x_top = own_x + (target_x - own_x) * t_top
        crosses_top_face = (
            (t_top > 0.0) & (t_top <= 1.0) & (line_x_top >= x0) & (line_x_top <= x1)
        )
        t_bot = (y1 - own_y) / safe_denom_y
        line_x_bot = own_x + (target_x - own_x) * t_bot
        crosses_bot_face = (
            (t_bot > 0.0) & (t_bot <= 1.0) & (line_x_bot >= x0) & (line_x_bot <= x1)
        )

        target_inside = self._points_in_obstacles(target_x, target_y)
        own_inside = self._points_in_obstacles(own_x, own_y)
        needs_route = active[:, None] & (
            (crosses_wall_x & crosses_blocked_y)
            | crosses_top_face
            | crosses_bot_face
            | target_inside
            | own_inside
        )
        if not bool(needs_route.any().item()):
            return target_x, target_y

        clearance = 2.0 if self.map_layout == MAP_B_SPLIT_LANE_V2 else 1.5
        upper_y = torch.clamp(y0 - clearance, 0.0, float(max(0, self.rows - 1)))
        lower_y = torch.clamp(y1 + clearance, 0.0, float(max(0, self.rows - 1)))
        prefer_upper = torch.where(
            target_y < y0,
            torch.ones_like(target_y, dtype=torch.bool),
            torch.where(
                target_y > y1,
                torch.zeros_like(target_y, dtype=torch.bool),
                own_y <= center_y,
            ),
        )
        route_y = torch.where(prefer_upper, upper_y, lower_y)
        current_left = own_x < x0
        current_right = own_x > x1
        target_right = target_x > x1
        target_left = target_x < x0
        # When the target is inside the wall, treat it as on the far side so the
        # agent routes through the gap rather than stalling at the gap entrance.
        target_right_eff = target_right | (target_inside & current_left)
        target_left_eff = target_left | (target_inside & current_right)
        moving_right = current_left & target_right_eff
        moving_left = current_right & target_left_eff
        current_side_x = torch.where(
            current_left,
            torch.clamp(x0 - clearance, 0.0, float(max(0, self.cols - 1))),
            torch.where(
                current_right,
                torch.clamp(x1 + clearance, 0.0, float(max(0, self.cols - 1))),
                own_x,
            ),
        )
        far_side_x = torch.where(
            moving_right,
            torch.clamp(x1 + clearance, 0.0, float(max(0, self.cols - 1))),
            torch.where(
                moving_left,
                torch.clamp(x0 - clearance, 0.0, float(max(0, self.cols - 1))),
                current_side_x,
            ),
        )
        y_staging = (
            (torch.abs(own_y - route_y) > 0.75)
            & (own_y >= (y0 - (clearance * 0.5)))
            & (own_y <= (y1 + (clearance * 0.5)))
        )
        waypoint_x = torch.where(y_staging, current_side_x, far_side_x)
        waypoint_y = route_y
        return (
            torch.where(needs_route, waypoint_x, target_x),
            torch.where(needs_route, waypoint_y, target_y),
        )
