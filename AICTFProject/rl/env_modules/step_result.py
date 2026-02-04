"""
Unified step-result/reward structure for CTFGameFieldSB3Env.

Provides a single canonical structure for step results, rewards, and episode termination info.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepReward:
    """Reward breakdown for a single step."""
    team_total: float = 0.0
    per_agent: List[float] = field(default_factory=list)
    reward_events_total: float = 0.0
    stall_penalty_total: float = 0.0
    terminal_bonus: float = 0.0  # Only set on terminal steps

    def __post_init__(self) -> None:
        if not self.per_agent:
            self.per_agent = []


@dataclass
class StepInfo:
    """Info dict for a single step."""
    reward_mode: str = "TEAM_SUM"
    reward_blue_per_agent: List[float] = field(default_factory=list)
    reward_blue_team: float = 0.0
    flip_count_step: int = 0
    macro_flip_count_step: int = 0
    target_flip_count_step: int = 0
    num_agents: int = 0
    action_components: int = 2
    phase: str = "OP1"
    dropped_reward_events: int = 0

    def __post_init__(self) -> None:
        if not self.reward_blue_per_agent:
            self.reward_blue_per_agent = []


@dataclass
class EpisodeResult:
    """Episode termination result (only set on terminal steps)."""
    blue_score: int = 0
    red_score: int = 0
    win_by: int = 0
    phase_name: str = "OP3"
    league_mode: bool = False
    blue_rewards_total: float = 0.0
    opponent_kind: str = "scripted"
    opponent_snapshot: Optional[str] = None
    species_tag: Optional[str] = None
    scripted_tag: Optional[str] = None
    decision_steps: int = 0
    macro_order: List[str] = field(default_factory=list)
    macro_counts: List[List[int]] = field(default_factory=list)
    mine_counts: List[Dict[str, int]] = field(default_factory=list)
    blue_mine_kills: int = 0
    mines_placed_enemy_half: int = 0
    mines_triggered_by_red: int = 0
    dynamics_config: Optional[Dict[str, Any]] = None
    # Top 5 IROS-style metrics
    success: int = 0
    time_to_first_score: Optional[float] = None
    time_to_game_over: Optional[float] = None
    collisions_per_episode: int = 0
    collision_events_per_episode: int = 0
    near_misses_per_episode: int = 0
    collision_free_episode: int = 0
    mean_inter_robot_dist: Optional[float] = None
    std_inter_robot_dist: Optional[float] = None
    zone_coverage: float = 0.0
    dropped_reward_events: int = 0
    vec_schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.macro_order:
            self.macro_order = []
        if not self.macro_counts:
            self.macro_counts = []
        if not self.mine_counts:
            self.mine_counts = []


@dataclass
class StepResult:
    """
    Unified step result structure.
    
    Contains:
    - observation: Dict[str, np.ndarray] (from gym.Env.step)
    - reward: float (scalar team reward for SB3)
    - terminated: bool
    - truncated: bool
    - reward_breakdown: StepReward (detailed reward info)
    - step_info: StepInfo (step-level info)
    - episode_result: Optional[EpisodeResult] (only set on terminal steps)
    """
    observation: Dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    reward_breakdown: StepReward
    step_info: StepInfo
    episode_result: Optional[EpisodeResult] = None

    def to_gym_tuple(self) -> tuple:
        """Convert to (obs, reward, terminated, truncated, info) tuple for gym.Env.step()."""
        info: Dict[str, Any] = {
            "reward_mode": self.step_info.reward_mode,
            "reward_blue_per_agent": list(self.step_info.reward_blue_per_agent),
            "reward_blue_team": float(self.step_info.reward_blue_team),
            "flip_count_step": int(self.step_info.flip_count_step),
            "macro_flip_count_step": int(self.step_info.macro_flip_count_step),
            "target_flip_count_step": int(self.step_info.target_flip_count_step),
            "num_agents": int(self.step_info.num_agents),
            "action_components": int(self.step_info.action_components),
            "phase": str(self.step_info.phase),
            "dropped_reward_events": int(self.step_info.dropped_reward_events),
        }

        if self.episode_result is not None:
            info["episode_result"] = {
                "blue_score": self.episode_result.blue_score,
                "red_score": self.episode_result.red_score,
                "win_by": self.episode_result.win_by,
                "phase_name": self.episode_result.phase_name,
                "league_mode": self.episode_result.league_mode,
                "blue_rewards_total": self.episode_result.blue_rewards_total,
                "opponent_kind": self.episode_result.opponent_kind,
                "opponent_snapshot": self.episode_result.opponent_snapshot,
                "species_tag": self.episode_result.species_tag,
                "scripted_tag": self.episode_result.scripted_tag,
                "decision_steps": self.episode_result.decision_steps,
                "macro_order": list(self.episode_result.macro_order),
                "macro_counts": [list(row) for row in self.episode_result.macro_counts],
                "mine_counts": [dict(row) for row in self.episode_result.mine_counts],
                "blue_mine_kills": self.episode_result.blue_mine_kills,
                "mines_placed_enemy_half": self.episode_result.mines_placed_enemy_half,
                "mines_triggered_by_red": self.episode_result.mines_triggered_by_red,
                "dynamics_config": (
                    dict(self.episode_result.dynamics_config)
                    if self.episode_result.dynamics_config is not None
                    else None
                ),
                "success": self.episode_result.success,
                "time_to_first_score": self.episode_result.time_to_first_score,
                "time_to_game_over": self.episode_result.time_to_game_over,
                "collisions_per_episode": self.episode_result.collisions_per_episode,
                "collision_events_per_episode": self.episode_result.collision_events_per_episode,
                "near_misses_per_episode": self.episode_result.near_misses_per_episode,
                "collision_free_episode": self.episode_result.collision_free_episode,
                "mean_inter_robot_dist": self.episode_result.mean_inter_robot_dist,
                "std_inter_robot_dist": self.episode_result.std_inter_robot_dist,
                "zone_coverage": self.episode_result.zone_coverage,
                "dropped_reward_events": self.episode_result.dropped_reward_events,
                "vec_schema_version": self.episode_result.vec_schema_version,
                "env_schema_version": 1,  # Standardized schema version
            }

        return (
            self.observation,
            float(self.reward),
            bool(self.terminated),
            bool(self.truncated),
            info,
        )

    @classmethod
    def from_components(
        cls,
        observation: Dict[str, Any],
        reward_breakdown: StepReward,
        step_info: StepInfo,
        terminated: bool,
        truncated: bool,
        episode_result: Optional[EpisodeResult] = None,
    ) -> StepResult:
        """Create StepResult from components."""
        # Team reward for SB3 (scalar)
        team_reward = reward_breakdown.team_total + reward_breakdown.terminal_bonus
        return cls(
            observation=observation,
            reward=team_reward,
            terminated=terminated,
            truncated=truncated,
            reward_breakdown=reward_breakdown,
            step_info=step_info,
            episode_result=episode_result,
        )


__all__ = ["StepReward", "StepInfo", "EpisodeResult", "StepResult"]
