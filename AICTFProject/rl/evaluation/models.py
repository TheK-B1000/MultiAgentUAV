"""Typed episode result data models for the map-awareness evaluation.

All dataclasses are frozen (immutable after construction).  ``None`` means
a value was unavailable; callers must not interpret ``None`` as zero.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class EvaluationCondition:
    """Identity of one (policy × map × opponent × seed) evaluation cell."""

    policy_name: Literal["baseline", "candidate"]
    map_name: str
    requested_opponent: str
    seed: int


@dataclass(frozen=True)
class EpisodeMetrics:
    """Navigation telemetry collected during one episode.

    Fields sourced from exact env telemetry when available, or from the
    ``InstrumentedEnv`` position proxy otherwise.  ``None`` means neither
    source was available.
    """

    obstacle_collisions: float | None
    stuck_steps: float | None
    blocked_movements: float | None
    upper_lane_crossings: float | None
    lower_lane_crossings: float | None
    route_switches: float | None
    episode_steps: int


@dataclass(frozen=True)
class EpisodeResult:
    """Complete result of one evaluated episode."""

    condition: EvaluationCondition
    resolved_opponent: str
    blue_score: float
    red_score: float
    won: bool
    lost: bool
    draw: bool
    score_margin: float
    episode_steps: int
    metrics: EpisodeMetrics

    def to_row(self) -> dict[str, Any]:
        """Return a flat dict compatible with the existing CSV schema."""
        upper = self.metrics.upper_lane_crossings
        lower = self.metrics.lower_lane_crossings
        return {
            "policy": self.condition.policy_name,
            "map": self.condition.map_name,
            "requested_opponent": self.condition.requested_opponent,
            "resolved_opponent": self.resolved_opponent,
            "opponent": self.resolved_opponent,
            "seed": self.condition.seed,
            "blue_score": self.blue_score,
            "red_score": self.red_score,
            "win": int(self.won),
            "loss": int(self.lost),
            "draw": int(self.draw),
            "score_margin": self.score_margin,
            "wall_collisions": self.metrics.obstacle_collisions,
            "stuck_steps": self.metrics.stuck_steps,
            "repeated_blocked_movement": self.metrics.blocked_movements,
            "upper_lane_use": upper,
            "lower_lane_use": lower,
            "route_switches": self.metrics.route_switches,
            "episode_steps": self.episode_steps,
        }


@dataclass(frozen=True)
class ConditionSummary:
    """Aggregated metrics across all episodes of one (policy, map, opponent) cell."""

    policy_name: str
    map_name: str
    requested_opponent: str
    resolved_opponent: str
    episode_count: int
    # Means — None when no valid episodes contributed
    mean_blue_score: float | None
    mean_red_score: float | None
    win_rate: float | None
    loss_rate: float | None
    draw_rate: float | None
    mean_score_margin: float | None
    mean_wall_collisions: float | None
    mean_stuck_steps: float | None
    mean_blocked_movements: float | None
    mean_upper_lane_use: float | None
    mean_lower_lane_use: float | None
    mean_route_switches: float | None
    mean_episode_steps: float | None
    route_crossings: float | None
    upper_lane_fraction: float | None

    def to_row(self) -> dict[str, Any]:
        return {
            "policy": self.policy_name,
            "map": self.map_name,
            "requested_opponent": self.requested_opponent,
            "resolved_opponent": self.resolved_opponent,
            "opponent": self.resolved_opponent,
            "episodes": self.episode_count,
            "blue_score": self.mean_blue_score,
            "red_score": self.mean_red_score,
            "win": self.win_rate,
            "loss": self.loss_rate,
            "draw": self.draw_rate,
            "score_margin": self.mean_score_margin,
            "wall_collisions": self.mean_wall_collisions,
            "stuck_steps": self.mean_stuck_steps,
            "repeated_blocked_movement": self.mean_blocked_movements,
            "upper_lane_use": self.mean_upper_lane_use,
            "lower_lane_use": self.mean_lower_lane_use,
            "route_switches": self.mean_route_switches,
            "episode_steps": self.mean_episode_steps,
            "route_crossings": self.route_crossings,
            "upper_lane_fraction": self.upper_lane_fraction,
        }


def _safe_mean(values: list[float]) -> float | None:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return None
    return sum(finite) / len(finite)
