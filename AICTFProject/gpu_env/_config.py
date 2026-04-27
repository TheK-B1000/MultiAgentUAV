from __future__ import annotations

from dataclasses import MISSING, dataclass, fields as dataclass_fields, replace as dataclass_replace
from typing import Any, List, Optional

from game_manager import (
    ACTION_FAILED_PUNISHMENT,
    DEFAULT_SCORE_LIMIT,
    DRAW_TEAM_PENALTY,
    ENEMY_MAV_KILL_REWARD,
    FLAG_CARRY_HOME_REWARD,
    FLAG_PICKUP_REWARD,
    LOSE_TEAM_PUNISH,
    WIN_TEAM_REWARD,
)

from ._constants import MAP_SET_SEED_OFFSETS


@dataclass(frozen=True)
class RewardConfig:
    """Paper-facing reward profile for GPU PPO training.

    `GPUFieldConfig` keeps flat fields for backward compatibility, but normalizes
    them through this object during `__post_init__`.
    """

    enabled_mine_reward: float = 0.2
    flag_pickup_reward: float = FLAG_PICKUP_REWARD
    flag_carry_home_reward: float = FLAG_CARRY_HOME_REWARD
    enemy_mav_kill_reward: float = ENEMY_MAV_KILL_REWARD
    action_failed_punishment: float = ACTION_FAILED_PUNISHMENT
    win_team_reward: float = WIN_TEAM_REWARD
    lose_team_punish: float = LOSE_TEAM_PUNISH
    draw_team_penalty: float = DRAW_TEAM_PENALTY
    pbrs_gamma: float = 0.995
    pbrs_attack_coef: float = 1.0
    pbrs_return_coef: float = 1.0
    pbrs_defense_coef: float = 1.0
    team_defense_presence_reward: float = 0.03
    team_escort_reward: float = 0.02
    team_intercept_reward: float = 0.02
    sparse_weight: float = 1.0
    dense_weight: float = 0.5
    reward_scale: float = 2.0
    reward_clip: float = 1.0
    stalemate_max_steps: int = 120
    stalemate_progress_eps: float = 0.002
    stalemate_penalty: float = -0.15
    spin_penalty_coef: float = 0.05
    idle_penalty_coef: float = 0.03

    @classmethod
    def from_object(cls, obj: Any) -> "RewardConfig":
        return cls(**{f.name: getattr(obj, f.name) for f in dataclass_fields(cls)})


# Historical name for papers/experiment configs that call this a "profile".
RewardProfile = RewardConfig
REWARD_FIELD_NAMES = frozenset(f.name for f in dataclass_fields(RewardConfig))


@dataclass(init=False)
class GPUFieldConfig:
    n_envs: int = 64
    n_agents_per_team: Optional[int] = None
    # Aquaticus standard setup is 2v2.
    max_blue_agents: int = 2
    max_red_agents: int = 2
    map_set: str = "train"
    map_rows: int = 20
    map_cols: int = 20
    # Paper-aligned timing (Table 3):
    #   - maxGameTime ~= 200 s
    #   - dt_sim ~= 0.5 s
    #   => 400 decision steps per episode when using dt = decision_interval_seconds.
    max_decision_steps: int = 400
    decision_interval_seconds: float = 0.5

    # Dynamics (matching BoatSimConfig defaults in game_field.py)
    max_speed_cps: float = 2.2
    max_accel_cps2: float = 2.0
    max_yaw_rate_rps: float = 4.0
    min_turn_radius_cells: float = 0.75
    current_strength_cps: float = 0.0
    drift_sigma_cells: float = 0.0
    sensor_range_cells: float = 9999.0
    sensor_noise_sigma_cells: float = 0.0
    sensor_dropout_prob: float = 0.0
    suppression_range_cells: float = 2.0
    # Local tag radius (in map cells) used by Aquaticus-style tagging.
    tag_range_cells: float = 2.5
    home_untag_radius_cells: float = 2.0
    avoid_collision_radius_cells: float = 0.75
    opregion_safe_speed_cps: float = 0.8

    n_macros: int = 5
    # Number of discrete 2D macro targets for GoTo/PlaceMine.
    # Paper-aligned default: 50 fixed positions sampled over the map.
    n_targets: int = 50
    score_limit: int = DEFAULT_SCORE_LIMIT

    # Mines: pickups spawn; agents must GRAB_MINE to get a charge, then PLACE_MINE to place anywhere.
    # For realism, each team can have at most 2 active mines on the field, and there are 4 pickups total
    # (2 on the blue side, 2 on the red side). Pickups are single-use with no respawn.
    max_mines_per_team: int = 2
    max_mine_charges_per_agent: int = 2
    mine_trigger_radius_cells: float = 1.5
    mine_pickup_radius_cells: float = 1.2
    mine_pickup_respawn_steps: int = 0   # 0 => no respawn; pickups are single-use
    n_mine_pickups: int = 4
    macro_commit_go_to_ticks: int = 4
    macro_commit_grab_ticks: int = 3
    macro_commit_get_flag_ticks: int = 4
    macro_commit_place_ticks: int = 2
    macro_commit_go_home_ticks: int = 4
    macro_arrival_radius_cells: float = 1.0

    # Tagging channel controls:
    # - tag_channel_seconds: pressure >= 2 must be sustained for this many seconds before a tag is applied.
    tag_channel_seconds: float = 1.0

    # Profile and reward controls
    aquaticus_profile: bool = False
    rules_profile: str = "OURS"

    # Viewer-only convenience: when True, newly tagged agents are snapped back
    # to their home flag position instead of returning under their own motion.
    # Training configs should generally leave this as False.
    teleport_tagged_home: bool = False
    # Number of consecutive frames an agent must be inside home radius while
    # carrying to count as a capture (filters single-frame tunneling at speed).
    capture_confirm_frames: int = 2
    # No score for grab/capture in the first N steps (avoids spurious points at game start).
    score_grace_steps: int = 10

    # PPO stability

    reward_config: Optional[RewardConfig] = None
    device: str = "cpu"
    seed: int = 42

    def __init__(self, **kwargs: Any) -> None:
        reward_kwargs = {
            name: kwargs.pop(name)
            for name in list(kwargs)
            if name in REWARD_FIELD_NAMES
        }
        reward_config = kwargs.pop("reward_config", None)
        config_fields = [f for f in dataclass_fields(type(self)) if f.name != "reward_config"]
        config_field_names = {f.name for f in config_fields}
        unknown = sorted(set(kwargs) - config_field_names)
        if unknown:
            unexpected = unknown[0]
            raise TypeError(f"GPUFieldConfig.__init__() got an unexpected keyword argument {unexpected!r}")

        for f in config_fields:
            if f.name in kwargs:
                value = kwargs.pop(f.name)
            elif f.default is not MISSING:
                value = f.default
            elif f.default_factory is not MISSING:  # type: ignore[attr-defined]
                value = f.default_factory()  # type: ignore[misc]
            else:
                raise TypeError(f"GPUFieldConfig.__init__() missing required keyword argument {f.name!r}")
            object.__setattr__(self, f.name, value)

        if reward_config is None:
            reward_config = RewardConfig(**reward_kwargs)
        object.__setattr__(self, "reward_config", reward_config)
        self.__post_init__()

    def __getattr__(self, name: str) -> Any:
        if name in REWARD_FIELD_NAMES:
            reward_config = self.__dict__.get("reward_config")
            if reward_config is None:
                reward_config = RewardConfig()
                object.__setattr__(self, "reward_config", reward_config)
            return getattr(reward_config, name)
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in REWARD_FIELD_NAMES:
            reward_config = self.__dict__.get("reward_config") or RewardConfig()
            object.__setattr__(self, "reward_config", dataclass_replace(reward_config, **{name: value}))
            return
        object.__setattr__(self, name, value)

    def __dir__(self) -> List[str]:
        return sorted(set(super().__dir__()) | REWARD_FIELD_NAMES)

    def __post_init__(self) -> None:
        self.reward_config = RewardConfig.from_object(self)

        if self.n_agents_per_team is not None:
            n = max(1, int(self.n_agents_per_team))
            self.max_blue_agents = n
            self.max_red_agents = n
        self.map_set = str(self.map_set).lower()
        if self.map_set not in MAP_SET_SEED_OFFSETS:
            allowed = ", ".join(sorted(MAP_SET_SEED_OFFSETS))
            raise ValueError(f"map_set must be one of {{{allowed}}}, got {self.map_set!r}")


assert REWARD_FIELD_NAMES <= set(dir(GPUFieldConfig()))


__all__ = ["GPUFieldConfig", "REWARD_FIELD_NAMES", "RewardConfig", "RewardProfile"]
