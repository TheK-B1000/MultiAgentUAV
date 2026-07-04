"""Map layout state: obstacle geometry, collision detection, and pathfinding.

Handles the per-episode split-lane wall (Map B variants) including tensor
allocation, per-episode reset (mirroring), axis-aligned collision detection,
wall-slide physics, and waypoint routing around obstacles.
"""
from __future__ import annotations

from typing import Tuple

import torch

from .._maps import (
    MAP_B_SPLIT_LANE,
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
        mirror_p = max(
            0.0, min(1.0, float(getattr(self.cfg, "map_b_vertical_mirror_prob", 0.5)))
        )
        for env_i in idx.detach().cpu().tolist():
            env_i = int(env_i)
            layout = self._map_layout_for_env(env_i)
            self.map_vertical_mirror[env_i] = False
            self.obstacle_rects[env_i] = 0.0
            self.obstacle_active[env_i] = False
            if not is_split_lane_layout(layout):
                continue
            mirror = bool(
                (torch.rand((1,), generator=self._rng, device=self.device) < mirror_p).item()
            )
            self.map_vertical_mirror[env_i] = mirror
            if layout == MAP_B_SPLIT_LANE_V2:
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
            rect = mirrored if mirror else base
            self.obstacle_rects[env_i, 0, :] = rect
            self.obstacle_active[env_i, 0] = True

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

        # Safety net: if the agent somehow began the step *inside* an obstacle
        # (numerical edge case, never expected in normal play), do not trap it.
        # Eject it to the nearest face so it can never freeze permanently inside.
        prev_inside = self._points_in_obstacles(prev_x, prev_y) & alive
        if bool(prev_inside.any().item()):
            ex, ey = self._nearest_exit(prev_x, prev_y)
            x_eject = torch.where(prev_inside, ex, prev_x)
            y_eject = torch.where(prev_inside, ey, prev_y)
            prev_x = torch.where(prev_inside, x_eject, prev_x)
            prev_y = torch.where(prev_inside, y_eject, prev_y)
            # Re-evaluate the move from the ejected (outside) baseline.
            hit = self._segments_hit_obstacles(prev_x, prev_y, next_x, next_y) & alive

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

    def _nearest_exit(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Nearest point just outside the active obstacle for each (x, y).

        Used only as a safety eject for agents that begin a step inside an
        obstacle; pushes to whichever of the four faces is closest.
        """
        rect = self.obstacle_rects[:, 0, :].to(dtype=x.dtype, device=x.device)
        x0 = rect[:, 0:1]
        y0 = rect[:, 1:2]
        x1 = rect[:, 2:3]
        y1 = rect[:, 3:4]
        eps = 0.05
        dist_left = x - x0
        dist_right = x1 - x
        dist_top = y - y0
        dist_bot = y1 - y
        min_d = torch.minimum(
            torch.minimum(dist_left, dist_right),
            torch.minimum(dist_top, dist_bot),
        )
        out_x = x.clone()
        out_y = y.clone()
        out_x = torch.where(dist_left == min_d, x0 - eps, out_x)
        out_x = torch.where(dist_right == min_d, x1 + eps, out_x)
        out_y = torch.where(dist_top == min_d, y0 - eps, out_y)
        out_y = torch.where(dist_bot == min_d, y1 + eps, out_y)
        max_x = float(max(0, self.cols - 1))
        max_y = float(max(0, self.rows - 1))
        return torch.clamp(out_x, 0.0, max_x), torch.clamp(out_y, 0.0, max_y)

    def _segment_intersects_rect(
        self,
        ox: torch.Tensor,
        oy: torch.Tensor,
        tx: torch.Tensor,
        ty: torch.Tensor,
        x0: torch.Tensor,
        y0: torch.Tensor,
        x1: torch.Tensor,
        y1: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized segment vs axis-aligned-rectangle test (Liang-Barsky).

        Returns a bool tensor (broadcast of the inputs) that is True where the
        segment ``(ox, oy) -> (tx, ty)`` overlaps the rectangle. This is exact
        (unlike fixed-fraction point sampling) so the routing trigger never
        misses a thin clip of the wall.
        """
        dx = tx - ox
        dy = ty - oy
        t0 = torch.zeros_like(dx)
        t1 = torch.ones_like(dx)
        valid = torch.ones_like(dx, dtype=torch.bool)
        for p, q in (
            (-dx, ox - x0),
            (dx, x1 - ox),
            (-dy, oy - y0),
            (dy, y1 - oy),
        ):
            parallel = p.abs() < 1e-9
            outside = parallel & (q < 0)
            valid = valid & ~outside
            safe_p = torch.where(parallel, torch.ones_like(p), p)
            r = q / safe_p
            t0 = torch.where((~parallel) & (p < 0), torch.maximum(t0, r), t0)
            t1 = torch.where((~parallel) & (p > 0), torch.minimum(t1, r), t1)
        return valid & (t0 <= t1)

    def _route_targets_around_obstacles(
        self,
        own_x: torch.Tensor,
        own_y: torch.Tensor,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Redirect a straight-line target to a corner waypoint when the direct
        path would cross the wall.

        Corner-routing contract (per agent, fully vectorized):
          * Route: own -> nearC -> farC -> target, where nearC=(near_x, detour_y)
            and farC=(far_x, detour_y) are corners just outside the wall.
          * Detour end (above / below) chosen by minimum path cost.
          * Each committed leg is validated with Liang-Barsky against expanded
            wall bounds (agent radius + margin) before being issued.  If the
            near-corner leg is blocked — agent inside wall x-range or approaching
            with a heading that clips the wall during a yaw maneuver — the router
            falls back to a lateral escape: (near_x, own_y).  This sidesteps the
            wall in x without requiring a diagonal that crosses the wall face.
        """
        if not bool(self.obstacle_active.any().item()):
            return target_x, target_y
        rect = self.obstacle_rects[:, 0, :].to(dtype=target_x.dtype, device=target_x.device)
        active = self.obstacle_active[:, 0:1].to(device=target_x.device)
        x0 = rect[:, 0:1]
        y0 = rect[:, 1:2]
        x1 = rect[:, 2:3]
        y1 = rect[:, 3:4]

        needs_route = active & self._segment_intersects_rect(
            own_x, own_y, target_x, target_y, x0, y0, x1, y1
        )
        if not bool(needs_route.any().item()):
            return target_x, target_y

        max_x = float(max(0, self.cols - 1))
        max_y = float(max(0, self.rows - 1))
        if getattr(self, "_map_pool", ()):
            clearance = 2.0
        else:
            clearance = 2.0 if self.map_layout == MAP_B_SPLIT_LANE_V2 else 1.5
        center_x = (x0 + x1) * 0.5

        # Detour-end corridor lines (y just above / below the wall).
        top_y = torch.clamp(y0 - clearance, 0.0, max_y)
        bot_y = torch.clamp(y1 + clearance, 0.0, max_y)
        # A detour end is only usable if the corridor is actually clear of the
        # wall after clamping to the grid (e.g. wall flush against an edge).
        top_ok = top_y < (y0 - 1e-3)
        bot_ok = bot_y > (y1 + 1e-3)

        # Near corner stays on the agent's x-side; far corner on the target's.
        left_side_x = torch.clamp(x0 - clearance, 0.0, max_x)
        right_side_x = torch.clamp(x1 + clearance, 0.0, max_x)
        own_left = own_x <= center_x
        target_left = target_x <= center_x
        near_x = torch.where(own_left, left_side_x, right_side_x)
        far_x = torch.where(target_left, left_side_x, right_side_x)

        def _path_cost(det_y: torch.Tensor) -> torch.Tensor:
            d0 = torch.sqrt((near_x - own_x) ** 2 + (det_y - own_y) ** 2 + 1e-8)
            d1 = torch.sqrt((far_x - near_x) ** 2 + 1e-8)
            d2 = torch.sqrt((target_x - far_x) ** 2 + (target_y - det_y) ** 2 + 1e-8)
            return d0 + d1 + d2

        inf = torch.full_like(own_x, 1e9)
        cost_top = torch.where(top_ok.expand_as(own_x), _path_cost(top_y), inf)
        cost_bot = torch.where(bot_ok.expand_as(own_x), _path_cost(bot_y), inf)
        use_top = cost_top <= cost_bot
        detour_y = torch.where(use_top, top_y.expand_as(own_x), bot_y.expand_as(own_x))

        # Leg selection (stateless, monotonic — cannot oscillate):
        #   * crossed: the agent has reached the target's x-side of the wall
        #     -> aim straight at the real target.
        #   * vertically clear of the wall band (above for a top detour, below
        #     for a bottom detour) -> aim the far corner to cross the corridor.
        #   * otherwise -> aim the near corner to enter the corridor while
        #     staying on the agent's own side.
        margin_v = 0.25
        clear_top = own_y <= (y0 - margin_v)
        clear_bot = own_y >= (y1 + margin_v)
        vertical_clear = torch.where(use_top, clear_top.expand_as(own_x), clear_bot.expand_as(own_x))
        crossed = torch.where(
            target_left.expand_as(own_x),
            own_x <= x0.expand_as(own_x),
            own_x >= x1.expand_as(own_x),
        )

        # Segment validation: before committing to the near-corner leg, verify
        # it does not clip the wall when expanded by agent radius + a small
        # clearance margin.  An agent inside the wall x-range (drifted in via
        # sampled-point collision misses) or approaching with a heading still
        # pointing into the wall zone will trigger this check.  The fallback
        # is a lateral-only escape to (near_x, own_y) so the x-exit happens
        # without a diagonal that crosses the wall face.
        agent_r = 0.3
        wx0 = x0 - agent_r
        wy0 = y0 - agent_r
        wx1 = x1 + agent_r
        wy1 = y1 + agent_r
        near_corner_blocked = self._segment_intersects_rect(
            own_x, own_y,
            near_x.expand_as(own_x), detour_y,
            wx0, wy0, wx1, wy1,
        )
        near_aim_y = torch.where(near_corner_blocked, own_y, detour_y)

        aim_x = torch.where(
            crossed,
            target_x,
            torch.where(vertical_clear, far_x.expand_as(own_x), near_x.expand_as(own_x)),
        )
        aim_y = torch.where(
            crossed,
            target_y,
            torch.where(vertical_clear, detour_y, near_aim_y),
        )

        return (
            torch.where(needs_route, aim_x, target_x),
            torch.where(needs_route, aim_y, target_y),
        )
