"""Environment-owned navigation telemetry contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum


NAVIGATION_TELEMETRY_VERSION = "phase1.6.v1"
ROUTE_CLASSIFIER_VERSION = "split-lane-route-classifier.v1"
MAP_ROUTE_METADATA_VERSION = "split-lane-route-metadata.v1"

BLOCKED_DISPLACEMENT_THRESHOLD_CELLS = 1e-3
STUCK_DISPLACEMENT_EPSILON_CELLS = 1e-3
STUCK_CONSECUTIVE_STEP_WINDOW = 3
REPEATED_BLOCKED_DIRECTION_WINDOW = 3


class RouteId(StrEnum):
    UPPER = "upper"
    LOWER = "lower"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class RouteCode(IntEnum):
    UNKNOWN = 0
    UPPER = 1
    LOWER = 2
    NEUTRAL = 3


@dataclass
class TeamNavigationTelemetry:
    obstacle_collision_events: int = 0
    blocked_movement_events: int = 0
    stuck_steps: int = 0
    repeated_blocked_movement_events: int = 0
    upper_lane_steps: int = 0
    lower_lane_steps: int = 0
    neutral_lane_steps: int = 0
    route_switches: int = 0
    movement_attempts: int = 0
    successful_movement_steps: int = 0


@dataclass
class EpisodeNavigationTelemetry:
    blue: TeamNavigationTelemetry = field(default_factory=TeamNavigationTelemetry)
    red: TeamNavigationTelemetry = field(default_factory=TeamNavigationTelemetry)


__all__ = [
    "BLOCKED_DISPLACEMENT_THRESHOLD_CELLS",
    "EpisodeNavigationTelemetry",
    "MAP_ROUTE_METADATA_VERSION",
    "NAVIGATION_TELEMETRY_VERSION",
    "REPEATED_BLOCKED_DIRECTION_WINDOW",
    "ROUTE_CLASSIFIER_VERSION",
    "RouteCode",
    "RouteId",
    "STUCK_CONSECUTIVE_STEP_WINDOW",
    "STUCK_DISPLACEMENT_EPSILON_CELLS",
    "TeamNavigationTelemetry",
]
