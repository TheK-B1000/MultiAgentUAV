from __future__ import annotations

from dataclasses import MISSING, dataclass, fields as dataclass_fields, replace as dataclass_replace
from typing import Any, List, Optional, Tuple

from game_manager import (
    ACTION_FAILED_PUNISHMENT,
    DEFAULT_SCORE_LIMIT,
    DRAW_TEAM_PENALTY,
    ENEMY_MAV_KILL_REWARD,
    FLAG_CARRY_HOME_REWARD,
    FLAG_PICKUP_REWARD,
    LOSE_TEAM_PUNISH,
    SPARSE_MINE_TAG_POINTS,
    SPARSE_OOB_POINTS,
    SPARSE_TAG_NO_FLAG_POINTS,
    SPARSE_TAG_WITH_FLAG_POINTS,
    WIN_TEAM_REWARD,
)

from ._constants import MAP_SET_SEED_OFFSETS
from ._constants import NUM_CNN_CHANNELS
from ._maps import MAP_A_OPEN, normalize_map_layout


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
    pbrs_attack_coef: float = 0.5
    pbrs_return_coef: float = 0.5
    pbrs_defense_coef: float = 0.5
    team_defense_presence_reward: float = 0.03
    team_escort_reward: float = 0.02
    team_intercept_reward: float = 0.02
    sparse_weight: float = 1.0
    # Points for tagging an opponent who is NOT carrying a flag. Applied
    # symmetrically: BLUE earns it for tagging, and pays it when tagged.
    #
    # The default (+100) equals SPARSE_FLAG_CAPTURE_POINTS, so a routine
    # defensive tag pays exactly what scoring a flag pays -- while being far
    # more frequent and far less risky. It also pays DOUBLE what tagging the
    # enemy flag carrier pays (SPARSE_TAG_WITH_FLAG_POINTS = 50). This is the
    # leading suspect for the passive tag-farming attractor that collapsed two
    # of three G0-v2 seeds; exposed as a knob so it can be ablated without
    # editing the shared game_manager constant.
    sparse_tag_no_flag_points: float = float(SPARSE_TAG_NO_FLAG_POINTS)
    # Points for tagging the enemy FLAG CARRIER. With sparse_tag_no_flag_points
    # zeroed this becomes the only remaining tag payoff, and the seed drawing the
    # largest share of its sparse reward from it was the one that failed. Exposed
    # so the whole tag-reward family can be closed in a single experiment.
    sparse_tag_with_flag_points: float = float(SPARSE_TAG_WITH_FLAG_POINTS)
    # Out-of-bounds points. Exposed for MEASUREMENT and future budgeting only --
    # deliberately left at its original value in Reward V3 because the OOB event
    # rate has never been measured, and budgeting an unmeasured term is the
    # mistake this whole exercise exists to correct.
    sparse_oob_points: float = float(SPARSE_OOB_POINTS)
    # OOB split into its two halves, because they are different incentives.
    # own: a penalty for leaving the field yourself (keep, but bounded).
    # opponent: a REWARD for the enemy leaving the field. At the historical
    # +100 this was a points farm -- V3 seed 2900002 drove red off the field
    # 2.39x/episode, earning +1.9/episode (3.1x its terminal signal) while
    # losing 89% of its games. Defaults preserve the original behaviour exactly.
    sparse_own_oob_points: float = float(SPARSE_OOB_POINTS)
    sparse_opponent_oob_points: float = float(-SPARSE_OOB_POINTS)
    # Mine tags are paid twice: sparse points AND enemy_mav_kill_reward,
    # because blue_kill_count includes them. Exposed for measurement first.
    sparse_mine_tag_points: float = float(SPARSE_MINE_TAG_POINTS)
    dense_weight: float = 0.25
    reward_scale: float = 4.0
    reward_clip: float = 1.0
    stalemate_max_steps: int = 120
    stalemate_progress_eps: float = 0.002
    stalemate_penalty: float = -0.08
    spin_penalty_coef: float = 0.05
    idle_penalty_coef: float = 0.03
    surface_score_margin_coef: float = 0.0
    surface_blue_capture_tempo_bonus: float = 0.0
    surface_red_flag_touch_penalty: float = 0.0
    surface_red_carrier_progress_penalty: float = 0.0
    surface_blue_near_cap_bonus: float = 0.0

    @classmethod
    def from_object(cls, obj: Any) -> "RewardConfig":
        return cls(**{f.name: getattr(obj, f.name) for f in dataclass_fields(cls)})


# Historical name for papers/experiment configs that call this a "profile".
RewardProfile = RewardConfig
REWARD_FIELD_NAMES = frozenset(f.name for f in dataclass_fields(RewardConfig))


@dataclass(frozen=True)
class RouterRewardConfig:
    """Sparse team-consequence reward for the V6I7 GRU router.

    Passed as ``router_reward_config`` to ``GPUFieldConfig`` from
    ``env_factory.build_training_env`` when ``router_reward_enabled=True``.
    The environment uses this to compute an exact per-step router reward from
    flag-capture and carrier-tag events rather than the aggregated sparse total.
    """

    enabled: bool = False
    win_weight: float = 1.0
    flag_cap_weight: float = 0.5
    sparse_weight: float = 0.2
    scale: float = 1.0
    normalize: bool = True


@dataclass(init=False)
class GPUFieldConfig:
    n_envs: int = 64
    n_agents_per_team: Optional[int] = None
    # Aquaticus standard setup is 2v2.
    max_blue_agents: int = 2
    max_red_agents: int = 2
    map_set: str = "train"
    map_layout: str = MAP_A_OPEN
    map_pool: Tuple[str, ...] = ()
    map_rows: int = 20
    map_cols: int = 20
    obstacle_obs_channel: Optional[bool] = None
    map_b_vertical_mirror_prob: float = 0.5
    map_b_wall_x_min_norm: float = 0.44
    map_b_wall_x_max_norm: float = 0.56
    map_b_wall_y_min_norm: float = 0.25
    map_b_wall_y_max_norm: float = 0.72
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
    # Episode-level domain randomization (training). When True, per-env runtime tensors
    # are resampled on each reset; eval / benchmarking should leave this False.
    train_domain_randomization: bool = False
    # Uniform [0, max] per episode for enemy-position noise in grid obs (map cells).
    dr_sensor_noise_sigma_max: float = 0.12
    # Uniform [0, max] dropout probability for in-range enemy detections.
    dr_sensor_dropout_max: float = 0.08
    # Uniform [1-jitter, 1] blue-team speed cap multiplier vs cfg.max_speed_cps (marine cap still applies).
    dr_blue_speed_jitter: float = 0.12
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

    # --- Tagging rules -------------------------------------------------------
    # Official Aquaticus: a SINGLE eligible defender tags by itself; the NEAREST
    # eligible opponent receives the tag (so a teammate can absorb one to protect
    # a carrier); and a successful tagger must wait a minimum interval before
    # tagging again.
    #
    # RULESET_V1 (superseded, kept reproducible) required two simultaneous
    # taggers with no cooldown. In 2v2 that made a lone defender strictly
    # dominated -- it could neither tag nor suppress -- which removed the
    # opportunity cost of committing both agents forward and collapsed the
    # strategy space onto a single non-dominated policy.
    #
    # Reproduce RULESET_V1 exactly with:
    #     taggers_required=2, tag_nearest_only=False,
    #     tag_min_interval_seconds=0.0, tag_channel_seconds=1.0
    taggers_required: int = 1
    tag_nearest_only: bool = True
    # Minimum interval before the SAME vehicle may tag again. MIT sources differ
    # (game-mechanics page: 30 s; uFldTagManager default: 10 s), so this is a
    # required knob rather than a hardcoded constant -- set it from the exact
    # mission/competition configuration being replicated. Distinct from
    # tag_duration_seconds, which is how long a tagged vehicle stays tagged.
    tag_min_interval_seconds: float = 10.0
    # Sustained-pressure window before a tag lands. The official per-request
    # eligibility model has no group channel, so 0.0 is the faithful value; a
    # small non-zero value is a simulator debounce and must be labeled as an
    # approximation, not a rule.
    tag_channel_seconds: float = 0.0
    # Suppression/kill is a project-specific mechanic, NOT Aquaticus tagging.
    # Kept on its own threshold so correcting tagging cannot silently change it.
    suppression_attackers_required: int = 2
    # Observational tag-event telemetry. OFF by default so training pays nothing.
    # When on, tag successes and cooldown denials are recorded AT THE DECISION
    # POINT, before movement / return-home / flag-drop side effects run. It must
    # be behaviour-neutral: identical states, rewards, and outcomes with it on or
    # off under the same seed (see tests/test_tag_telemetry.py).
    tag_telemetry_enabled: bool = False

    @property
    def ruleset_id(self) -> str:
        """Identity of the tagging ruleset actually in force.

        Stamped into run configs and checkpoints so a RULESET_V1 policy cannot
        silently enter a RULESET_V2 result.
        """
        if (int(self.taggers_required) == 1 and bool(self.tag_nearest_only)
                and float(self.tag_channel_seconds) == 0.0
                and float(self.tag_min_interval_seconds) == 10.0):
            return "RULESET_V2_AQUATICUS_10S"
        if (int(self.taggers_required) == 2 and not bool(self.tag_nearest_only)
                and float(self.tag_min_interval_seconds) == 0.0):
            return "RULESET_V1_TWO_TAGGER"
        return (f"RULESET_CUSTOM_t{int(self.taggers_required)}"
                f"_cd{float(self.tag_min_interval_seconds):g}"
                f"_ch{float(self.tag_channel_seconds):g}"
                f"_near{int(bool(self.tag_nearest_only))}")

    def ruleset_fields(self) -> dict:
        """The full tagging-rule fingerprint for provenance and load checks."""
        return {
            "ruleset_id": self.ruleset_id,
            "taggers_required": int(self.taggers_required),
            "tag_min_interval_seconds": float(self.tag_min_interval_seconds),
            "tag_nearest_only": bool(self.tag_nearest_only),
            "tag_channel_seconds": float(self.tag_channel_seconds),
            "suppression_attackers_required": int(self.suppression_attackers_required),
        }

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
    router_reward_config: Optional[RouterRewardConfig] = None
    device: str = "cpu"
    seed: int = 42

    def __init__(self, **kwargs: Any) -> None:
        reward_kwargs = {
            name: kwargs.pop(name)
            for name in list(kwargs)
            if name in REWARD_FIELD_NAMES
        }
        reward_config = kwargs.pop("reward_config", None)
        router_reward_config = kwargs.pop("router_reward_config", None)
        _excluded = {"reward_config", "router_reward_config"}
        config_fields = [f for f in dataclass_fields(type(self)) if f.name not in _excluded]
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
        object.__setattr__(self, "router_reward_config", router_reward_config)
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
        self.map_layout = normalize_map_layout(self.map_layout)
        pool_raw = tuple(getattr(self, "map_pool", ()) or ())
        if pool_raw:
            self.map_pool = tuple(normalize_map_layout(m) for m in pool_raw)
            if len(self.map_pool) < 1:
                raise ValueError("map_pool must contain at least one layout when set.")
            if self.obstacle_obs_channel is None:
                from ._maps import MAP_A_OPEN as _MAP_A_OPEN

                self.obstacle_obs_channel = any(layout != _MAP_A_OPEN for layout in self.map_pool)
        else:
            self.map_pool = ()
        if self.obstacle_obs_channel is None:
            self.obstacle_obs_channel = self.map_layout != MAP_A_OPEN
        self.map_b_vertical_mirror_prob = max(0.0, min(1.0, float(self.map_b_vertical_mirror_prob)))
        for name in (
            "map_b_wall_x_min_norm",
            "map_b_wall_x_max_norm",
            "map_b_wall_y_min_norm",
            "map_b_wall_y_max_norm",
        ):
            setattr(self, name, max(0.0, min(1.0, float(getattr(self, name)))))

    @property
    def num_cnn_channels(self) -> int:
        return int(NUM_CNN_CHANNELS) + (1 if bool(self.obstacle_obs_channel) else 0)


assert REWARD_FIELD_NAMES <= set(dir(GPUFieldConfig()))


__all__ = ["GPUFieldConfig", "REWARD_FIELD_NAMES", "RewardConfig", "RewardProfile", "RouterRewardConfig"]
