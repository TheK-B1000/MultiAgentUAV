"""Per-episode telemetry tensor allocation and reset.

Two groups of telemetry are managed here:
- **Metric buffers** (``metric_*``): episode-aggregate scalars summarised into
  ``info["episode_result"]`` at terminal time.
- **Navigation telemetry buffers** (``nav_*``): per-agent and per-env counters
  tracking obstacle collisions, lane usage, route switches, etc.
"""
from __future__ import annotations

import torch

from .._constants import METRIC_ZONE_COLS, METRIC_ZONE_ROWS
from .._navigation_telemetry import RouteCode


class _TelemetryStateMixin:
    """Allocates and resets metric / navigation telemetry tensors."""

    def _alloc_metric_buffers(
        self, B: int, dev: torch.device, f32: torch.dtype
    ) -> None:
        self.metric_time_to_first_score = torch.full((B,), -1.0, dtype=f32, device=dev)
        self.metric_inter_robot_dist_sum = torch.zeros((B,), dtype=f32, device=dev)
        self.metric_inter_robot_dist_count = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.metric_collision_events = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.metric_obstacle_collision_events = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.metric_near_misses = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.metric_blue_route_upper_crossings = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.metric_blue_route_lower_crossings = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.metric_red_route_upper_crossings = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.metric_red_route_lower_crossings = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.metric_blue_attack_upper_crossings = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.metric_blue_attack_lower_crossings = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.metric_blue_return_upper_crossings = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.metric_blue_return_lower_crossings = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.metric_blue_intercept_upper_crossings = torch.zeros(
            (B,), dtype=torch.int32, device=dev
        )
        self.metric_blue_intercept_lower_crossings = torch.zeros(
            (B,), dtype=torch.int32, device=dev
        )
        self.metric_red_attack_upper_crossings = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.metric_red_attack_lower_crossings = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.metric_red_return_upper_crossings = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.metric_red_return_lower_crossings = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.metric_red_intercept_upper_crossings = torch.zeros(
            (B,), dtype=torch.int32, device=dev
        )
        self.metric_red_intercept_lower_crossings = torch.zeros(
            (B,), dtype=torch.int32, device=dev
        )
        self.metric_blue_zone_visited = torch.zeros(
            (B, METRIC_ZONE_ROWS * METRIC_ZONE_COLS),
            dtype=torch.bool,
            device=dev,
        )

    def _alloc_navigation_telemetry_buffers(
        self, B: int, Nb: int, Nr: int, dev: torch.device
    ) -> None:
        for side in ("blue", "red"):
            n_agents = Nb if side == "blue" else Nr
            setattr(
                self,
                f"nav_{side}_obstacle_collision_events",
                torch.zeros((B,), dtype=torch.int32, device=dev),
            )
            setattr(
                self,
                f"nav_{side}_blocked_movement_events",
                torch.zeros((B,), dtype=torch.int32, device=dev),
            )
            setattr(
                self,
                f"nav_{side}_stuck_steps",
                torch.zeros((B,), dtype=torch.int32, device=dev),
            )
            setattr(
                self,
                f"nav_{side}_repeated_blocked_movement_events",
                torch.zeros((B,), dtype=torch.int32, device=dev),
            )
            setattr(
                self,
                f"nav_{side}_upper_lane_steps",
                torch.zeros((B,), dtype=torch.int32, device=dev),
            )
            setattr(
                self,
                f"nav_{side}_lower_lane_steps",
                torch.zeros((B,), dtype=torch.int32, device=dev),
            )
            setattr(
                self,
                f"nav_{side}_neutral_lane_steps",
                torch.zeros((B,), dtype=torch.int32, device=dev),
            )
            setattr(
                self,
                f"nav_{side}_route_switches",
                torch.zeros((B,), dtype=torch.int32, device=dev),
            )
            setattr(
                self,
                f"nav_{side}_movement_attempts",
                torch.zeros((B,), dtype=torch.int32, device=dev),
            )
            setattr(
                self,
                f"nav_{side}_successful_movement_steps",
                torch.zeros((B,), dtype=torch.int32, device=dev),
            )
            setattr(
                self,
                f"nav_{side}_consecutive_blocked_steps",
                torch.zeros((B, n_agents), dtype=torch.int32, device=dev),
            )
            setattr(
                self,
                f"nav_{side}_repeated_blocked_direction_steps",
                torch.zeros((B, n_agents), dtype=torch.int32, device=dev),
            )
            setattr(
                self,
                f"nav_{side}_last_blocked_dir_x",
                torch.zeros((B, n_agents), dtype=torch.int8, device=dev),
            )
            setattr(
                self,
                f"nav_{side}_last_blocked_dir_y",
                torch.zeros((B, n_agents), dtype=torch.int8, device=dev),
            )
            setattr(
                self,
                f"nav_{side}_last_route",
                torch.full(
                    (B, n_agents),
                    int(RouteCode.UNKNOWN),
                    dtype=torch.int8,
                    device=dev,
                ),
            )

    def _reset_navigation_telemetry(self, idx: torch.Tensor) -> None:
        if idx.numel() == 0:
            return
        for side in ("blue", "red"):
            for name in (
                "obstacle_collision_events",
                "blocked_movement_events",
                "stuck_steps",
                "repeated_blocked_movement_events",
                "upper_lane_steps",
                "lower_lane_steps",
                "neutral_lane_steps",
                "route_switches",
                "movement_attempts",
                "successful_movement_steps",
                "consecutive_blocked_steps",
                "repeated_blocked_direction_steps",
                "last_blocked_dir_x",
                "last_blocked_dir_y",
            ):
                getattr(self, f"nav_{side}_{name}")[idx] = 0
            getattr(self, f"nav_{side}_last_route")[idx] = int(RouteCode.UNKNOWN)
