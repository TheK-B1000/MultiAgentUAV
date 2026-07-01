"""
Game state, reward routing, and flag logic. Single source of truth for scoring and
reward/shaping constants.

- game_field_gpu (BatchedCTFCore) imports get_grab_score_delta, get_capture_score_delta,
  sparse event point constants, and DEFAULT_SCORE_LIMIT from here so GPU training and
  the viewer use the same values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# -------------------------
# Reward constants (baseline)
# Tuned to encourage offense, defense, and teamwork (especially for 4v4).
# -------------------------

WIN_TEAM_REWARD = 1.0
LOSE_TEAM_PUNISH = -1.0  # symmetric terminal signal for the losing team
DRAW_TEAM_PENALTY = -0.5

# Offense: capturing and carrying flag (closer to Jacob et al. values)
FLAG_PICKUP_REWARD = 0.1
FLAG_CARRY_HOME_REWARD = 0.5
ENEMY_MAV_KILL_REWARD = 0.5
ACTION_FAILED_PUNISHMENT = -0.2

FLAG_RETURN_DELAY = 10.0

# PBRS (potential based reward shaping): F = coef * (gamma * Phi(s') - Phi(s))
FLAG_PROXIMITY_COEF = 0.45
DEFAULT_SHAPING_GAMMA = 0.99  # IMPORTANT: set this from PPO gamma via env binding
DEFENSE_SHAPING_MULT = 3.5
DEFENSE_CARRIER_PROGRESS_COEF = 0.45
DEFENSE_PRESENCE_RADIUS = 6.0
DEFENSE_PRESENCE_REWARD = 0.06
ESCORT_CARRIER_RADIUS = 5.0
ESCORT_CARRIER_REWARD = 0.08

# Sprint A: Minimal shaping rewards (progress-to-flag/home)
PROGRESS_TO_FLAG_COEF = 0.08
PROGRESS_TO_HOME_COEF = 0.08
PROGRESS_REWARD_THRESHOLD = 0.1  # Minimum distance change to trigger reward

STALL_PENALTY = -0.1
STALL_INTERVAL_SECONDS = 30.0
TEAM_FLAG_TAKEN_PENALTY = -0.2
TEAM_FLAG_SCORED_PENALTY = -0.5
TEAM_FLAG_RECOVER_REWARD = 0.2

# Optional draw penalty by phase (default 0, research-safe)
PHASE_DRAW_TIMEOUT_PENALTY: Dict[str, float] = {
    "OP1": -0.5,
    "OP2": -1.0,
    "OP3": -1.5,
    "OP4": -1.5,   # held-out eval only; same as OP3
    "SELF": -1.5,
}

# Sparse reward "points" used by game_field_gpu (BatchedCTFCore) before /100 normalization.
SPARSE_TAG_NO_FLAG_POINTS = 100.0
SPARSE_TAG_WITH_FLAG_POINTS = 50.0
SPARSE_FLAG_CAPTURE_POINTS = 100.0
SPARSE_OOB_POINTS = -100.0
SPARSE_MINE_TAG_POINTS = 100.0

# Default score limit for CTF scoring.
DEFAULT_SCORE_LIMIT = 3


def get_grab_score_delta(rules_profile: str) -> int:
    """Score added to the grabbing team when they pick up the enemy flag. Used by GPU core and GameManager."""
    return 0


def get_capture_score_delta(rules_profile: str) -> int:
    """Score added when a carrier scores (brings flag home). Used by GPU core and GameManager."""
    return 1


Cell = Tuple[int, int]
FloatPos = Tuple[float, float]
RewardEvent = Tuple[float, str, float]  # (t, agent_id, value)


@dataclass
class FlagState:
    """Encapsulates the state and common operations for a single flag to follow SRP and DRY."""
    home: Cell = (0, 0)
    position: Cell = (0, 0)
    taken: bool = False
    carrier: Optional[Any] = None
    drop_time: Optional[float] = None

    def reset(self, home_pos: Cell) -> None:
        self.home = home_pos
        self.position = home_pos
        self.taken = False
        self.carrier = None
        self.drop_time = None

    def pickup(self, agent: Any, cell: Cell) -> None:
        self.taken = True
        self.carrier = agent
        self.position = cell
        self.drop_time = None

    def drop(self, cell: Cell, time: float) -> None:
        self.taken = False
        self.carrier = None
        self.position = cell
        self.drop_time = time

    def sanity_check(self, time: float, drop_fn: Callable[[Any], None], agent_cell_fn: Callable[[Any], Cell]) -> None:
        if self.taken:
            if self.carrier is None:
                self.taken = False
                self.position = self.home
                self.drop_time = None
            else:
                carrier_enabled = getattr(self.carrier, "isEnabled", lambda: True)()
                if not carrier_enabled:
                    drop_fn(self.carrier)
                else:
                    self.position = agent_cell_fn(self.carrier)
                    self.drop_time = None

    def check_auto_return(self, time: float, return_delay: float) -> None:
        if (
            (not self.taken)
            and self.position != self.home
            and self.drop_time is not None
        ):
            if time - self.drop_time >= return_delay:
                self.position = self.home
                self.drop_time = None


class DynamicsManager:
    """Manages parsing of dynamics configurations and agent speed scaling to separate concerns."""
    def __init__(self) -> None:
        self.config: Optional[Dict[str, Any]] = None

    def set_config(self, cfg: Optional[Dict[str, Any]]) -> None:
        if cfg is None:
            self.config = None
            return
        if not isinstance(cfg, dict):
            raise TypeError(f"dynamics config must be dict or None, got {type(cfg)}")
        self.config = dict(cfg)

    def get_config(self) -> Optional[Dict[str, Any]]:
        return None if self.config is None else dict(self.config)

    def get_team_speed_multiplier(self, side: str) -> float:
        side = str(side).lower().strip()
        cfg = self.config or {}
        raw = cfg.get("blue_speed_mult", 1.0) if side == "blue" else cfg.get("red_speed_mult", 1.0)
        try:
            v = float(raw)
        except Exception:
            return 1.0
        if not math.isfinite(v) or v <= 0.0:
            return 1.0
        return float(v)

    def get_agent_speed_multiplier(self, agent: Any) -> float:
        if agent is None:
            return 1.0
        side = str(getattr(agent, "side", "")).lower().strip()
        if side not in ("blue", "red"):
            return 1.0
        base = self.get_team_speed_multiplier(side)
        if side != "red":
            return base

        cfg = self.config or {}
        try:
            opp_mult = float(cfg.get("opponent_speed_mult", 1.0))
            if math.isfinite(opp_mult) and opp_mult > 0.0:
                base *= opp_mult
        except Exception:
            pass

        kind = str(cfg.get("opponent_kind", "")).lower().strip()
        key = str(cfg.get("opponent_key", "")).upper().strip()
        if not kind:
            kind = str(getattr(agent, "opponent_kind", "")).lower().strip()
        if not key:
            key = str(getattr(agent, "opponent_tag", "")).upper().strip()

        if kind == "scripted":
            table = cfg.get("scripted_speed_mult")
            if isinstance(table, dict):
                try:
                    v = float(table.get(key, 1.0))
                    if math.isfinite(v) and v > 0.0:
                        base *= v
                except Exception:
                    pass
        elif kind == "species":
            species_tag = key or str(cfg.get("species_tag", "BALANCED")).upper().strip()
            table = cfg.get("species_speed_mult")
            if isinstance(table, dict):
                try:
                    v = float(table.get(species_tag, 1.0))
                    if math.isfinite(v) and v > 0.0:
                        base *= v
                except Exception:
                    pass
        elif kind == "snapshot":
            try:
                v = float(cfg.get("snapshot_speed_mult", 1.0))
                if math.isfinite(v) and v > 0.0:
                    base *= v
            except Exception:
                pass

        if not math.isfinite(base) or base <= 0.0:
            return 1.0
        return float(base)

    def get_summary(self) -> Dict[str, Any]:
        cfg = self.config or {}
        return {
            "blue_speed_mult": cfg.get("blue_speed_mult", 1.0),
            "red_speed_mult": cfg.get("red_speed_mult", 1.0),
            "opponent_kind": cfg.get("opponent_kind", None),
            "opponent_key": cfg.get("opponent_key", None),
        }


class MetricTracker:
    """Manages IROS and evaluation telemetry collection to separate concerns."""
    def __init__(self) -> None:
        self.time_to_first_score: Optional[float] = None
        self.time_to_game_over: Optional[float] = None
        self.collision_count: int = 0
        self.collision_events: int = 0
        self.near_miss_count: int = 0
        self.inter_robot_distances: List[float] = []
        self.zone_visited_cells: Set[Cell] = set()
        self.total_zone_cells: int = 0

    def reset(self) -> None:
        self.time_to_first_score = None
        self.time_to_game_over = None
        self.collision_count = 0
        self.collision_events = 0
        self.near_miss_count = 0
        self.inter_robot_distances.clear()
        self.zone_visited_cells.clear()
        self.total_zone_cells = 0

    def record_tick(
        self,
        collision_delta: int = 0,
        near_miss_delta: int = 0,
        collision_events_delta: int = 0,
        blue_inter_robot_dist: Optional[float] = None,
        blue_zone_cells_this_tick: Optional[Set[Cell]] = None,
    ) -> None:
        self.collision_count += int(collision_delta)
        self.near_miss_count += int(near_miss_delta)
        self.collision_events += int(collision_events_delta)
        if blue_inter_robot_dist is not None and math.isfinite(blue_inter_robot_dist):
            self.inter_robot_distances.append(float(blue_inter_robot_dist))
        if blue_zone_cells_this_tick:
            self.zone_visited_cells.update(blue_zone_cells_this_tick)


class RewardRouter:
    """Manages per-agent reward routing and transaction buffering to separate concerns."""
    def __init__(self, shaping_gamma: float = DEFAULT_SHAPING_GAMMA) -> None:
        self.reward_events: List[RewardEvent] = []
        self.blue_episode_reward: float = 0.0
        self.shaping_gamma: float = shaping_gamma

    def reset(self) -> None:
        self.reward_events.clear()
        self.blue_episode_reward = 0.0

    def add_reward_event(self, value: float, agent_id: str, timestamp: float) -> None:
        if agent_id is None or str(agent_id).strip() == "":
            raise ValueError("agent_id must be a non-empty string.")
        try:
            v = float(value)
        except Exception:
            return
        if not math.isfinite(v):
            return
        self.reward_events.append((float(timestamp), str(agent_id), float(v)))

    def add_agent_reward(
        self,
        agent: Any,
        value: float,
        timestamp: float,
        agent_uid_fn: Callable[[Any], str],
        remember_agent_fn: Callable[[Any], None]
    ) -> None:
        if agent is None:
            return
        remember_agent_fn(agent)
        uid = agent_uid_fn(agent)
        self.add_reward_event(value, agent_id=uid, timestamp=timestamp)
        if str(getattr(agent, "side", "")).lower() == "blue":
            self.blue_episode_reward += float(value)

    def add_team_reward(
        self,
        side: str,
        value: float,
        timestamp: float,
        game_field: Any,
        blue_agent_ids_seen: Set[str],
        red_agent_ids_seen: Set[str],
        agent_uid_fn: Callable[[Any], str],
        remember_agent_fn: Callable[[Any], None],
        exclude_agent: Optional[Any] = None,
        include_disabled: bool = False,
    ) -> None:
        side = str(side).lower().strip()
        if side not in ("blue", "red"):
            return
        ex_uid = agent_uid_fn(exclude_agent) if exclude_agent is not None else None

        if game_field is not None:
            team = game_field.blue_agents if side == "blue" else game_field.red_agents
            for a in team:
                if a is None:
                    continue
                if (not include_disabled) and hasattr(a, "isEnabled") and (not a.isEnabled()):
                    continue
                uid = agent_uid_fn(a)
                if ex_uid is not None and uid == ex_uid:
                    continue
                remember_agent_fn(a)
                self.add_reward_event(value, agent_id=uid, timestamp=timestamp)
            return

        ids = blue_agent_ids_seen if side == "blue" else red_agent_ids_seen
        for uid in ids:
            if ex_uid is not None and uid == ex_uid:
                continue
            self.add_reward_event(value, agent_id=uid, timestamp=timestamp)

    def pop_reward_events(self) -> List[RewardEvent]:
        events = self.reward_events
        self.reward_events = []
        return events


class GameManager:
    """
    Game state + reward routing for the non-GPU game field (not used by PPO/GPU training).

    Refactored to act as a coordinator delegating specific tasks to subcomponents (FlagState, 
    DynamicsManager, MetricTracker, RewardRouter) to comply with SOLID and DRY, while exposing
    fully backward-compatible property facades.
    """

    def __init__(
        self,
        cols: int,
        rows: int,
        blue_score: int = 0,
        red_score: int = 0,
        score_limit: int = 3,
        max_time: float = 300.0,
        current_time: float = 300.0,
        sim_time: float = 0.0,
        game_over: bool = False,
        phase_name: str = "OP1",
        timeout_blue_wins_defense_held: bool = False,
        rules_profile: str = "OURS",
        **kwargs: Any
    ) -> None:
        self.cols = int(cols)
        self.rows = int(rows)
        self.blue_score = int(blue_score)
        self.red_score = int(red_score)
        self.score_limit = int(score_limit)
        self.max_time = float(max_time)
        self.current_time = float(current_time)
        self.sim_time = float(sim_time)
        self.game_over = bool(game_over)
        self.phase_name = str(phase_name)
        self.timeout_blue_wins_defense_held = bool(timeout_blue_wins_defense_held)
        self.rules_profile = str(rules_profile)

        # Delegated objects
        self.blue_flag = FlagState()
        self.red_flag = FlagState()
        self.dynamics_manager = DynamicsManager()
        self.metric_tracker = MetricTracker()
        self.reward_router = RewardRouter(shaping_gamma=DEFAULT_SHAPING_GAMMA)

        # Routing memory (ids seen)
        self.blue_agent_ids_seen: Set[str] = set()
        self.red_agent_ids_seen: Set[str] = set()

        # Optional env binding for precise team membership
        self.game_field: Optional[Any] = None

        # Episode telemetry
        self.blue_mine_kills_this_episode: int = 0
        self.red_mine_kills_this_episode: int = 0
        self.blue_captures_this_episode: List[int] = []
        self.mines_placed_in_enemy_half_this_episode: int = 0
        self.mines_triggered_by_red_this_episode: int = 0

        self.last_score_time: float = 0.0

        # Initialize game state parameters
        self.reset_game(reset_scores=False)

    # -------------------------
    # Backward-Compatible Properties (Facades to Delegated Objects)
    # -------------------------

    @property
    def blue_flag_home(self) -> Cell:
        return self.blue_flag.home

    @blue_flag_home.setter
    def blue_flag_home(self, val: Cell) -> None:
        self.blue_flag.home = val

    @property
    def red_flag_home(self) -> Cell:
        return self.red_flag.home

    @red_flag_home.setter
    def red_flag_home(self, val: Cell) -> None:
        self.red_flag.home = val

    @property
    def blue_flag_position(self) -> Cell:
        return self.blue_flag.position

    @blue_flag_position.setter
    def blue_flag_position(self, val: Cell) -> None:
        self.blue_flag.position = val

    @property
    def red_flag_position(self) -> Cell:
        return self.red_flag.position

    @red_flag_position.setter
    def red_flag_position(self, val: Cell) -> None:
        self.red_flag.position = val

    @property
    def blue_flag_taken(self) -> bool:
        return self.blue_flag.taken

    @blue_flag_taken.setter
    def blue_flag_taken(self, val: bool) -> None:
        self.blue_flag.taken = val

    @property
    def red_flag_taken(self) -> bool:
        return self.red_flag.taken

    @red_flag_taken.setter
    def red_flag_taken(self, val: bool) -> None:
        self.red_flag.taken = val

    @property
    def blue_flag_carrier(self) -> Optional[Any]:
        return self.blue_flag.carrier

    @blue_flag_carrier.setter
    def blue_flag_carrier(self, val: Optional[Any]) -> None:
        self.blue_flag.carrier = val

    @property
    def red_flag_carrier(self) -> Optional[Any]:
        return self.red_flag.carrier

    @red_flag_carrier.setter
    def red_flag_carrier(self, val: Optional[Any]) -> None:
        self.red_flag.carrier = val

    @property
    def blue_flag_drop_time(self) -> Optional[float]:
        return self.blue_flag.drop_time

    @blue_flag_drop_time.setter
    def blue_flag_drop_time(self, val: Optional[float]) -> None:
        self.blue_flag.drop_time = val

    @property
    def red_flag_drop_time(self) -> Optional[float]:
        return self.red_flag.drop_time

    @red_flag_drop_time.setter
    def red_flag_drop_time(self, val: Optional[float]) -> None:
        self.red_flag.drop_time = val

    @property
    def dynamics_config(self) -> Optional[Dict[str, Any]]:
        return self.dynamics_manager.config

    @dynamics_config.setter
    def dynamics_config(self, val: Optional[Dict[str, Any]]) -> None:
        self.dynamics_manager.config = val

    @property
    def reward_events(self) -> List[RewardEvent]:
        return self.reward_router.reward_events

    @reward_events.setter
    def reward_events(self, val: List[RewardEvent]) -> None:
        self.reward_router.reward_events = val

    @property
    def blue_episode_reward(self) -> float:
        return self.reward_router.blue_episode_reward

    @blue_episode_reward.setter
    def blue_episode_reward(self, val: float) -> None:
        self.reward_router.blue_episode_reward = val

    @property
    def shaping_gamma(self) -> float:
        return self.reward_router.shaping_gamma

    @shaping_gamma.setter
    def shaping_gamma(self, val: float) -> None:
        self.reward_router.shaping_gamma = val

    # IROS metric redirections
    @property
    def time_to_first_score(self) -> Optional[float]:
        return self.metric_tracker.time_to_first_score

    @time_to_first_score.setter
    def time_to_first_score(self, val: Optional[float]) -> None:
        self.metric_tracker.time_to_first_score = val

    @property
    def time_to_game_over(self) -> Optional[float]:
        return self.metric_tracker.time_to_game_over

    @time_to_game_over.setter
    def time_to_game_over(self, val: Optional[float]) -> None:
        self.metric_tracker.time_to_game_over = val

    @property
    def collision_count_this_episode(self) -> int:
        return self.metric_tracker.collision_count

    @collision_count_this_episode.setter
    def collision_count_this_episode(self, val: int) -> None:
        self.metric_tracker.collision_count = val

    @property
    def collision_events_this_episode(self) -> int:
        return self.metric_tracker.collision_events

    @collision_events_this_episode.setter
    def collision_events_this_episode(self, val: int) -> None:
        self.metric_tracker.collision_events = val

    @property
    def near_miss_count_this_episode(self) -> int:
        return self.metric_tracker.near_miss_count

    @near_miss_count_this_episode.setter
    def near_miss_count_this_episode(self, val: int) -> None:
        self.metric_tracker.near_miss_count = val

    @property
    def blue_inter_robot_distances(self) -> List[float]:
        return self.metric_tracker.inter_robot_distances

    @blue_inter_robot_distances.setter
    def blue_inter_robot_distances(self, val: List[float]) -> None:
        self.metric_tracker.inter_robot_distances = val

    @property
    def blue_zone_visited_cells(self) -> Set[Cell]:
        return self.metric_tracker.zone_visited_cells

    @blue_zone_visited_cells.setter
    def blue_zone_visited_cells(self, val: Set[Cell]) -> None:
        self.metric_tracker.zone_visited_cells = val

    @property
    def total_blue_zone_cells(self) -> int:
        return self.metric_tracker.total_zone_cells

    @total_blue_zone_cells.setter
    def total_blue_zone_cells(self, val: int) -> None:
        self.metric_tracker.total_zone_cells = val

    # -------------------------
    # Binding / config
    # -------------------------

    def bind_game_field(self, game_field: Any) -> None:
        """Bind environment for exact team reward routing."""
        self.game_field = game_field

    def record_tick_metrics(
        self,
        collision_delta: int = 0,
        near_miss_delta: int = 0,
        collision_events_delta: int = 0,
        blue_inter_robot_dist: Optional[float] = None,
        blue_zone_cells_this_tick: Optional[Set[Cell]] = None,
    ) -> None:
        """IROS-style metrics: called by game_field each tick."""
        self.metric_tracker.record_tick(
            collision_delta=collision_delta,
            near_miss_delta=near_miss_delta,
            collision_events_delta=collision_events_delta,
            blue_inter_robot_dist=blue_inter_robot_dist,
            blue_zone_cells_this_tick=blue_zone_cells_this_tick,
        )

    def set_phase(self, phase: str) -> None:
        """Set curriculum phase name (canonical uppercase)."""
        self.phase_name = str(phase).upper().strip()

    def set_rules_profile(self, profile: str) -> None:
        """Kept for backward compatibility; always uses OURS semantics."""
        self.rules_profile = "OURS"

    def _grab_score_delta(self) -> int:
        return get_grab_score_delta(self.rules_profile)

    def _capture_score_delta(self) -> int:
        return get_capture_score_delta(self.rules_profile)

    def _flag_pickup_reward(self) -> float:
        return float(FLAG_PICKUP_REWARD)

    def _flag_capture_reward(self) -> float:
        return float(FLAG_CARRY_HOME_REWARD)

    def set_shaping_gamma(self, gamma: float) -> None:
        """Set shaping gamma; must match PPO gamma for PBRS policy invariance."""
        g = float(gamma)
        if not (0.0 <= g <= 1.0):
            raise ValueError(f"gamma must be in [0,1], got {gamma}")
        self.reward_router.shaping_gamma = g

    # ---- dynamics config (vector-env env_method compatibility) ----

    def set_dynamics_config(self, cfg: Optional[Dict[str, Any]]) -> None:
        self.dynamics_manager.set_config(cfg)
        if cfg is not None:
            rp = cfg.get("rules_profile", None)
            if rp is not None:
                self.set_rules_profile(str(rp))

    def get_dynamics_config(self) -> Optional[Dict[str, Any]]:
        return self.dynamics_manager.get_config()

    def get_team_speed_multiplier(self, side: str) -> float:
        return self.dynamics_manager.get_team_speed_multiplier(side)

    def get_agent_speed_multiplier(self, agent: Any) -> float:
        return self.dynamics_manager.get_agent_speed_multiplier(agent)

    def get_episode_dynamics_summary(self) -> Dict[str, Any]:
        return self.dynamics_manager.get_summary()

    # -------------------------
    # Core helpers
    # -------------------------

    def _clamp_cell(self, x: int, y: int) -> Cell:
        return (max(0, min(self.cols - 1, int(x))), max(0, min(self.rows - 1, int(y))))

    def _agent_cell(self, agent: Any) -> Cell:
        fp = getattr(agent, "float_pos", None)
        if isinstance(fp, (tuple, list)) and len(fp) >= 2:
            return self._clamp_cell(int(round(fp[0])), int(round(fp[1])))
        if hasattr(agent, "get_position"):
            x, y = agent.get_position()
            return self._clamp_cell(int(x), int(y))
        return self._clamp_cell(getattr(agent, "x", 0), getattr(agent, "y", 0))

    def _agent_float(self, agent: Any) -> FloatPos:
        fp = getattr(agent, "float_pos", None)
        if isinstance(fp, (tuple, list)) and len(fp) >= 2:
            return float(fp[0]), float(fp[1])
        return float(getattr(agent, "x", 0.0)), float(getattr(agent, "y", 0.0))

    def _agent_uid(self, agent: Any) -> str:
        uid = getattr(agent, "unique_id", None)
        if uid is None or str(uid).strip() == "":
            return str(id(agent))
        return str(uid)

    def _remember_agent(self, agent: Any) -> None:
        if agent is None:
            return
        side = str(getattr(agent, "side", "")).lower().strip()
        uid = self._agent_uid(agent)
        if side == "blue":
            self.blue_agent_ids_seen.add(uid)
        elif side == "red":
            self.red_agent_ids_seen.add(uid)

    # -------------------------
    # Reward routing (events only)
    # -------------------------

    def add_reward_event(self, value: float, agent_id: str, timestamp: Optional[float] = None) -> None:
        t = self.sim_time if timestamp is None else float(timestamp)
        self.reward_router.add_reward_event(value, agent_id, t)

    def add_agent_reward(self, agent: Any, value: float, timestamp: Optional[float] = None) -> None:
        t = self.sim_time if timestamp is None else float(timestamp)
        self.reward_router.add_agent_reward(agent, value, t, self._agent_uid, self._remember_agent)

    def add_team_reward(
        self,
        side: str,
        value: float,
        timestamp: Optional[float] = None,
        exclude_agent: Optional[Any] = None,
        include_disabled: bool = False,
    ) -> None:
        t = self.sim_time if timestamp is None else float(timestamp)
        self.reward_router.add_team_reward(
            side=side,
            value=value,
            timestamp=t,
            game_field=self.game_field,
            blue_agent_ids_seen=self.blue_agent_ids_seen,
            red_agent_ids_seen=self.red_agent_ids_seen,
            agent_uid_fn=self._agent_uid,
            remember_agent_fn=self._remember_agent,
            exclude_agent=exclude_agent,
            include_disabled=include_disabled,
        )

    def pop_reward_events(self) -> List[RewardEvent]:
        return self.reward_router.pop_reward_events()

    def terminal_outcome_bonus(self, blue_score: int, red_score: int) -> float:
        if blue_score > red_score:
            return float(WIN_TEAM_REWARD)
        if red_score > blue_score:
            return float(LOSE_TEAM_PUNISH)
        return float(DRAW_TEAM_PENALTY)

    # -------------------------
    # Reset
    # -------------------------

    def reset_game(self, reset_scores: bool = True) -> None:
        if reset_scores:
            self.blue_score = 0
            self.red_score = 0

        self.game_over = False
        self.current_time = float(self.max_time)
        self.sim_time = 0.0

        self.reward_router.reset()
        self.metric_tracker.reset()

        self.blue_mine_kills_this_episode = 0
        self.red_mine_kills_this_episode = 0
        self.mines_placed_in_enemy_half_this_episode = 0
        self.mines_triggered_by_red_this_episode = 0

        self.blue_agent_ids_seen.clear()
        self.red_agent_ids_seen.clear()

        self.blue_captures_this_episode.clear()
        self.blue_episode_reward = 0.0
        if self.game_field is not None:
            blue_agents = getattr(self.game_field, "blue_agents", []) or []
            while len(self.blue_captures_this_episode) < len(blue_agents):
                self.blue_captures_this_episode.append(0)

        mid_row = self.rows // 2
        self.blue_flag.reset(self._clamp_cell(2, mid_row))
        self.red_flag.reset(self._clamp_cell(self.cols - 3, mid_row))

        self.last_score_time = 0.0

    # -------------------------
    # Tick / termination
    # -------------------------

    def tick_seconds(self, dt: float) -> Optional[str]:
        if self.game_over or dt <= 0.0:
            return None

        self.sim_time += float(dt)
        self.current_time -= float(dt)

        self.sanity_check_flags()
        self._update_flag_auto_return()

        # Anti-stall: if no score for a while, apply small team penalty
        if (self.sim_time - float(self.last_score_time)) >= float(STALL_INTERVAL_SECONDS):
            self.add_team_reward("blue", STALL_PENALTY)
            self.add_team_reward("red", STALL_PENALTY)
            self.last_score_time = float(self.sim_time)

        # Time over
        if self.current_time <= 0.0 and not self.game_over:
            self.game_over = True
            if bool(getattr(self, "timeout_blue_wins_defense_held", False)):
                # Naval framing: defense held = Blue wins on timeout
                self.add_team_reward("blue", WIN_TEAM_REWARD)
                self.add_team_reward("red", LOSE_TEAM_PUNISH)
                return "BLUE WINS (DEFENSE HELD)"
            if self.blue_score > self.red_score:
                self.add_team_reward("blue", WIN_TEAM_REWARD)
                self.add_team_reward("red", LOSE_TEAM_PUNISH)
                return "BLUE WINS ON TIME"
            if self.red_score > self.blue_score:
                self.add_team_reward("red", WIN_TEAM_REWARD)
                self.add_team_reward("blue", LOSE_TEAM_PUNISH)
                return "RED WINS ON TIME"

            penalty = float(PHASE_DRAW_TIMEOUT_PENALTY.get(self.phase_name, 0.0))
            if penalty != 0.0:
                self.add_team_reward("blue", penalty)
                self.add_team_reward("red", penalty)
                return f"DRAW — PENALTY ({self.phase_name})"
            return f"DRAW — NO PENALTY ({self.phase_name})"

        # Score limit
        if self.blue_score >= self.score_limit:
            self.game_over = True
            self.time_to_game_over = float(self.sim_time)
            self.add_team_reward("blue", WIN_TEAM_REWARD)
            self.add_team_reward("red", LOSE_TEAM_PUNISH)
            return "BLUE WINS BY SCORE!"

        if self.red_score >= self.score_limit:
            self.game_over = True
            self.time_to_game_over = float(self.sim_time)
            self.add_team_reward("red", WIN_TEAM_REWARD)
            self.add_team_reward("blue", LOSE_TEAM_PUNISH)
            return "RED WINS BY SCORE!"

        return None

    def _update_flag_auto_return(self) -> None:
        self.blue_flag.check_auto_return(self.sim_time, FLAG_RETURN_DELAY)
        self.red_flag.check_auto_return(self.sim_time, FLAG_RETURN_DELAY)

    # -------------------------
    # Flag sanity / helpers
    # -------------------------

    def sanity_check_flags(self) -> None:
        """
        Ensures carrier/taken/position is consistent.
        If a carrier is disabled, the flag drops at the carrier cell with a drop_time.
        """
        self.blue_flag.sanity_check(self.sim_time, self.drop_flag_if_carrier_disabled, self._agent_cell)
        self.red_flag.sanity_check(self.sim_time, self.drop_flag_if_carrier_disabled, self._agent_cell)

    def get_enemy_flag_position(self, side: str) -> Cell:
        side = str(side).lower().strip()
        enemy_flag = self.red_flag if side == "blue" else self.blue_flag
        return (
            self._agent_cell(enemy_flag.carrier)
            if (enemy_flag.taken and enemy_flag.carrier is not None)
            else enemy_flag.position
        )

    def get_team_zone_center(self, side: str) -> Cell:
        return self.blue_flag.home if str(side).lower().strip() == "blue" else self.red_flag.home

    def get_sim_time(self) -> float:
        return float(self.sim_time)

    # -------------------------
    # Flag interactions
    # -------------------------

    def try_pickup_enemy_flag(self, agent: Any) -> bool:
        side = str(getattr(agent, "side", "")).lower().strip()
        if side not in ("blue", "red"):
            return False

        self._remember_agent(agent)

        ax, ay = self._agent_float(agent)
        enemy_flag = self.red_flag if side == "blue" else self.blue_flag

        if enemy_flag.taken:
            return False  # already carrying that flag

        if math.hypot(ax - float(enemy_flag.position[0]), ay - float(enemy_flag.position[1])) > 1.0:
            return False

        # Take it
        enemy_flag.pickup(agent, self._agent_cell(agent))

        if hasattr(agent, "setCarryingFlag"):
            agent.setCarryingFlag(True)

        grab_delta = self._grab_score_delta()
        if grab_delta > 0:
            if side == "blue":
                self.blue_score += int(grab_delta)
            else:
                self.red_score += int(grab_delta)
            if self.time_to_first_score is None:
                self.time_to_first_score = float(self.sim_time)
            self.last_score_time = float(self.sim_time)

        self.add_agent_reward(agent, self._flag_pickup_reward())

        if side == "blue":
            self.add_team_reward("red", TEAM_FLAG_TAKEN_PENALTY)
        else:
            self.add_team_reward("blue", TEAM_FLAG_TAKEN_PENALTY)

        return True

    def try_score_if_carrying_and_home(self, agent: Any) -> bool:
        side = str(getattr(agent, "side", "")).lower().strip()
        if side not in ("blue", "red"):
            return False

        self._remember_agent(agent)
        ax, ay = self._agent_float(agent)

        my_flag = self.blue_flag if side == "blue" else self.red_flag
        enemy_flag = self.red_flag if side == "blue" else self.blue_flag

        # Red score check by phase
        if side == "red":
            phase = str(getattr(self, "phase_name", "")).upper()
            if phase in ("OP1", "OP2"):
                return False

        if enemy_flag.taken and (enemy_flag.carrier is agent):
            if math.hypot(ax - float(my_flag.home[0]), ay - float(my_flag.home[1])) <= 2.0:
                if side == "blue":
                    self.blue_score += int(self._capture_score_delta())
                else:
                    self.red_score += int(self._capture_score_delta())

                if self.time_to_first_score is None:
                    self.time_to_first_score = float(self.sim_time)
                enemy_flag.reset(enemy_flag.home)
                self.last_score_time = float(self.sim_time)

                if hasattr(agent, "setCarryingFlag"):
                    agent.setCarryingFlag(False, scored=True)

                cap_reward = self._flag_capture_reward()
                self.add_agent_reward(agent, cap_reward)
                self.add_team_reward(side, cap_reward * 0.5, exclude_agent=agent)
                
                other_side = "red" if side == "blue" else "blue"
                self.add_team_reward(other_side, TEAM_FLAG_SCORED_PENALTY)

                # Per-agent capture count for eval (coordination / variance)
                if side == "blue" and self.game_field is not None:
                    blue_agents = getattr(self.game_field, "blue_agents", []) or []
                    try:
                        idx = blue_agents.index(agent)
                        while len(self.blue_captures_this_episode) <= idx:
                            self.blue_captures_this_episode.append(0)
                        self.blue_captures_this_episode[idx] += 1
                    except (ValueError, AttributeError):
                        pass

                return True

        return False

    def drop_flag_if_carrier_disabled(self, agent: Any, punish: bool = False) -> None:
        """
        Drop the carried flag at the carrier's current cell.

        IMPORTANT:
          - punish=False for normal "death/disabled" drops (called by sanity_check_flags)
          - punish=True only when you explicitly decide a macro/action "failed" and want to penalize
        """
        drop_pos = self._agent_cell(agent)
        self._remember_agent(agent)

        carried_flag = None
        team_flag_side = "blue"
        if self.blue_flag.carrier is agent:
            carried_flag = self.blue_flag
            team_flag_side = "blue"
        elif self.red_flag.carrier is agent:
            carried_flag = self.red_flag
            team_flag_side = "red"

        if carried_flag is not None:
            carried_flag.drop(drop_pos, self.sim_time)

            if punish:
                self.add_agent_reward(agent, ACTION_FAILED_PUNISHMENT)

            if hasattr(agent, "setCarryingFlag"):
                agent.setCarryingFlag(False, scored=False)
            self.add_team_reward(team_flag_side, TEAM_FLAG_RECOVER_REWARD)

    def clear_flag_carrier_if_agent(self, agent: Any) -> None:
        """
        Hard reset carriers if an agent is removed from game unexpectedly.
        Prefer drop_flag_if_carrier_disabled() in normal flow.
        """
        for flag in (self.blue_flag, self.red_flag):
            if flag.carrier is agent:
                flag.reset(flag.home)
                if hasattr(agent, "setCarryingFlag"):
                    agent.setCarryingFlag(False, scored=False)

    # -------------------------
    # Gamma-correct PBRS + progress shaping
    # -------------------------

    def reward_potential_shaping(self, agent: Any, start_pos: FloatPos, end_pos: FloatPos) -> None:
        """
        Potential-Based Reward Shaping:
            F(s,a,s') = coef * (gamma * Phi(s') - Phi(s))

        Uses float positions provided by the env/agent.

        IMPORTANT:
          Ensure set_shaping_gamma() is called from the env wrapper / trainer
          with the same gamma as PPO uses.
        """
        side = str(getattr(agent, "side", "")).lower().strip()
        if side not in ("blue", "red"):
            return

        self._remember_agent(agent)

        my_flag = self.blue_flag if side == "blue" else self.red_flag
        enemy_flag = self.red_flag if side == "blue" else self.blue_flag

        i_am_carrier = enemy_flag.taken and (enemy_flag.carrier is agent)
        teammate_is_carrier = enemy_flag.taken and (enemy_flag.carrier is not None) and (enemy_flag.carrier is not agent)

        # Intercept if enemy has our flag
        enemy_has_our_flag = my_flag.taken and (my_flag.carrier is not None)

        if enemy_has_our_flag and my_flag.carrier is not None and (not i_am_carrier):
            goal_x, goal_y = self._agent_float(my_flag.carrier)
        else:
            goal_x, goal_y = my_flag.home if (i_am_carrier or teammate_is_carrier) else self.get_enemy_flag_position(side)

        max_dist = math.sqrt(float(self.cols * self.cols + self.rows * self.rows))
        if max_dist <= 1e-6:
            return

        sx, sy = float(start_pos[0]), float(start_pos[1])
        ex, ey = float(end_pos[0]), float(end_pos[1])

        prev_d = min(max_dist, math.dist([sx, sy], [float(goal_x), float(goal_y)]))
        cur_d = min(max_dist, math.dist([ex, ey], [float(goal_x), float(goal_y)]))

        phi_before = 1.0 - (prev_d / max_dist)
        phi_after = 1.0 - (cur_d / max_dist)

        shaped = float(FLAG_PROXIMITY_COEF) * (float(self.shaping_gamma) * phi_after - phi_before)
        if enemy_has_our_flag and (not i_am_carrier):
            shaped *= float(DEFENSE_SHAPING_MULT)
        if shaped != 0.0:
            self.add_agent_reward(agent, shaped)

        if enemy_has_our_flag and my_flag.carrier is not None and (not i_am_carrier):
            prev_dc = min(max_dist, math.dist([sx, sy], [float(goal_x), float(goal_y)]))
            cur_dc = min(max_dist, math.dist([ex, ey], [float(goal_x), float(goal_y)]))
            progress = (prev_dc - cur_dc) / max_dist
            if progress > 0.0:
                self.add_agent_reward(agent, float(DEFENSE_CARRIER_PROGRESS_COEF) * float(progress))

        # Sprint A: Minimal shaping rewards for progress-to-flag/home
        self._apply_progress_rewards(agent, start_pos, end_pos, side, i_am_carrier, enemy_has_our_flag, my_flag.carrier)

        ax2, ay2 = float(end_pos[0]), float(end_pos[1])

        # Teamwork: defense presence — reward staying near our flag when enemy has it.
        if enemy_has_our_flag and (not i_am_carrier):
            dist_home = math.hypot(ax2 - float(my_flag.home[0]), ay2 - float(my_flag.home[1]))
            if dist_home <= float(DEFENSE_PRESENCE_RADIUS):
                self.add_agent_reward(agent, DEFENSE_PRESENCE_REWARD)

        # Teamwork: escort — reward being near our carrier when we have the flag.
        if (i_am_carrier or teammate_is_carrier) and (not i_am_carrier) and enemy_flag.carrier is not None:
            cx, cy = self._agent_float(enemy_flag.carrier)
            if math.hypot(ax2 - cx, ay2 - cy) <= float(ESCORT_CARRIER_RADIUS):
                self.add_agent_reward(agent, ESCORT_CARRIER_REWARD)

    # -------------------------
    # Mine/combat hooks (minimal)
    # -------------------------

    def reward_mine_placed(self, agent: Any, mine_pos: Optional[Cell] = None) -> None:
        if mine_pos is None:
            return
        side = str(getattr(agent, "side", "")).lower().strip()
        if side not in ("blue", "red"):
            return

        x, y = mine_pos
        if side == "blue":
            if x > (self.cols * 0.5):
                self.mines_placed_in_enemy_half_this_episode += 1
        else:
            if x < (self.cols * 0.5):
                self.mines_placed_in_enemy_half_this_episode += 1

    def reward_mine_picked_up(self, agent: Any, prev_charges: int = 0) -> None:
        return

    def reward_enemy_killed(
        self,
        killer_agent: Any,
        victim_agent: Optional[Any] = None,
        cause: Optional[str] = None,
    ) -> None:
        if killer_agent is None:
            return

        self._remember_agent(killer_agent)
        kside = str(getattr(killer_agent, "side", "")).lower().strip()

        if cause == "mine":
            if kside == "blue":
                self.blue_mine_kills_this_episode += 1
            elif kside == "red":
                self.red_mine_kills_this_episode += 1

        self.add_agent_reward(killer_agent, ENEMY_MAV_KILL_REWARD)

        if kside in ("blue", "red") and cause == "mine":
            self.add_team_reward(kside, ENEMY_MAV_KILL_REWARD * 0.5, exclude_agent=killer_agent)

    def record_mine_triggered_by_red(self) -> None:
        self.mines_triggered_by_red_this_episode += 1

    def _apply_progress_rewards(
        self,
        agent: Any,
        start_pos: FloatPos,
        end_pos: FloatPos,
        side: str,
        i_am_carrier: bool,
        enemy_has_our_flag: bool,
        carrier: Optional[Any],
    ) -> None:
        """
        Dense rewards for progress toward the right goal (teamwork-aware).
        - When carrying: progress toward home.
        - When enemy has our flag: progress toward carrier (defense), not enemy flag.
        - Otherwise: progress toward enemy flag (offense).
        """
        if agent is None:
            return

        max_dist = math.sqrt(float(self.cols * self.cols + self.rows * self.rows))
        if max_dist <= 1e-6:
            return

        sx, sy = float(start_pos[0]), float(start_pos[1])
        ex, ey = float(end_pos[0]), float(end_pos[1])

        enemy_flag = self.red_flag if side == "blue" else self.blue_flag

        if i_am_carrier:
            goal_x, goal_y = self.get_team_zone_center(side)
            goal = (float(goal_x), float(goal_y))
            coef = float(PROGRESS_TO_HOME_COEF)
        elif enemy_has_our_flag and carrier is not None:
            goal = self._agent_float(carrier)
            coef = float(DEFENSE_CARRIER_PROGRESS_COEF)
        else:
            goal = (float(enemy_flag.position[0]), float(enemy_flag.position[1]))
            coef = float(PROGRESS_TO_FLAG_COEF)

        prev_dist = math.dist([sx, sy], goal)
        curr_dist = math.dist([ex, ey], goal)
        dist_change = prev_dist - curr_dist

        if dist_change > float(PROGRESS_REWARD_THRESHOLD):
            normalized_progress = dist_change / max_dist
            reward = coef * normalized_progress
            if reward > 0.0:
                self.add_agent_reward(agent, reward)

    def punish_failed_action(self, agent: Any) -> None:
        if agent is None:
            return
        self.add_agent_reward(agent, ACTION_FAILED_PUNISHMENT)
