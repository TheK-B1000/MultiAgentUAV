"""
Schema versioning and standardization for episode results and info fields.

Ensures consistent field names, types, and versions across the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# Schema versions
ENV_SCHEMA_VERSION = 1  # Environment schema version
VEC_SCHEMA_VERSION = 1  # Vector observation schema version


@dataclass
class EpisodeResultSchema:
    """
    Standardized episode result schema.
    
    All episode results must conform to this schema for consistency.
    Field names and types are fixed - no ad-hoc additions.
    """
    # Scores
    blue_score: int
    red_score: int
    win_by: int  # blue_score - red_score
    success: int  # 1 if blue won, 0 otherwise
    
    # Episode metadata
    phase_name: str
    league_mode: bool
    blue_rewards_total: float
    decision_steps: int
    
    # Opponent info
    opponent_kind: str  # "scripted" | "species" | "snapshot"
    opponent_snapshot: Optional[str]
    species_tag: Optional[str]
    scripted_tag: Optional[str]
    
    # Action usage
    macro_order: List[str]
    macro_counts: List[List[int]]
    mine_counts: List[Dict[str, int]]
    blue_mine_kills: int
    mines_placed_enemy_half: int
    mines_triggered_by_red: int
    
    # Environment config
    dynamics_config: Optional[Dict[str, Any]]
    
    # Top 5 IROS-style metrics
    time_to_first_score: Optional[float]
    time_to_game_over: Optional[float]
    collisions_per_episode: int
    collision_events_per_episode: int  # Canonical collision metric
    near_misses_per_episode: int
    collision_free_episode: int  # 1 if collision_events_per_episode == 0, else 0
    mean_inter_robot_dist: Optional[float]
    std_inter_robot_dist: Optional[float]
    zone_coverage: float
    
    # MARL correctness
    dropped_reward_events: int
    
    # Schema versioning
    vec_schema_version: int = VEC_SCHEMA_VERSION
    env_schema_version: int = ENV_SCHEMA_VERSION
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for info['episode_result'])."""
        return {
            "blue_score": self.blue_score,
            "red_score": self.red_score,
            "win_by": self.win_by,
            "success": self.success,
            "phase_name": self.phase_name,
            "league_mode": self.league_mode,
            "blue_rewards_total": self.blue_rewards_total,
            "opponent_kind": self.opponent_kind,
            "opponent_snapshot": self.opponent_snapshot,
            "species_tag": self.species_tag,
            "scripted_tag": self.scripted_tag,
            "decision_steps": self.decision_steps,
            "macro_order": list(self.macro_order),
            "macro_counts": [list(row) for row in self.macro_counts],
            "mine_counts": [dict(row) for row in self.mine_counts],
            "blue_mine_kills": self.blue_mine_kills,
            "mines_placed_enemy_half": self.mines_placed_enemy_half,
            "mines_triggered_by_red": self.mines_triggered_by_red,
            "dynamics_config": (
                dict(self.dynamics_config) if self.dynamics_config is not None else None
            ),
            "time_to_first_score": self.time_to_first_score,
            "time_to_game_over": self.time_to_game_over,
            "collisions_per_episode": self.collisions_per_episode,
            "collision_events_per_episode": self.collision_events_per_episode,
            "near_misses_per_episode": self.near_misses_per_episode,
            "collision_free_episode": self.collision_free_episode,
            "mean_inter_robot_dist": self.mean_inter_robot_dist,
            "std_inter_robot_dist": self.std_inter_robot_dist,
            "zone_coverage": self.zone_coverage,
            "dropped_reward_events": self.dropped_reward_events,
            "vec_schema_version": self.vec_schema_version,
            "env_schema_version": self.env_schema_version,
        }


@dataclass
class StepInfoSchema:
    """
    Standardized step info schema.
    
    All step info dicts must include these fields for consistency.
    """
    reward_mode: str  # "TEAM_SUM" | "PER_AGENT" | "SHAPLEY_APPROX"
    reward_blue_per_agent: List[float]
    reward_blue_team: float
    flip_count_step: int
    macro_flip_count_step: int
    target_flip_count_step: int
    num_agents: int
    action_components: int  # Always 2 (macro + target per agent)
    phase: str
    dropped_reward_events: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for info dict)."""
        return {
            "reward_mode": self.reward_mode,
            "reward_blue_per_agent": list(self.reward_blue_per_agent),
            "reward_blue_team": float(self.reward_blue_team),
            "flip_count_step": int(self.flip_count_step),
            "macro_flip_count_step": int(self.macro_flip_count_step),
            "target_flip_count_step": int(self.target_flip_count_step),
            "num_agents": int(self.num_agents),
            "action_components": int(self.action_components),
            "phase": str(self.phase),
            "dropped_reward_events": int(self.dropped_reward_events),
        }


__all__ = [
    "ENV_SCHEMA_VERSION",
    "VEC_SCHEMA_VERSION",
    "EpisodeResultSchema",
    "StepInfoSchema",
]
