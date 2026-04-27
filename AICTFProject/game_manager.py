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
from typing import Any, Dict, List, Optional, Set, Tuple

# -------------------------
# Reward constants (baseline)
# Tuned to encourage offense, defense, and teamwork (especially for 4v4).
# -------------------------

# Terminal: strong signal so policy learns to win
# Jacob et al.-style magnitudes (Table 3-like, small and symmetric).
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
# Neutral names; values define OURS sparse event scaling.
SPARSE_TAG_NO_FLAG_POINTS = 100.0
SPARSE_TAG_WITH_FLAG_POINTS = 50.0
SPARSE_FLAG_CAPTURE_POINTS = 100.0
SPARSE_OOB_POINTS = -100.0
SPARSE_MINE_TAG_POINTS = 100.0

# Default score limit for CTF scoring.
DEFAULT_SCORE_LIMIT = 3


def get_grab_score_delta(rules_profile: str) -> int:
    """Score added to the grabbing team when they pick up the enemy flag. Used by GPU core and GameManager."""
    # OURS default: only captures change the scoreboard; grabs are rewarded via FLAG_PICKUP_REWARD.
    return 0


def get_capture_score_delta(rules_profile: str) -> int:
    """Score added when a carrier scores (brings flag home). Used by GPU core and GameManager."""
    # OURS default: each capture increments score by 1.
    return 1


Cell = Tuple[int, int]
FloatPos = Tuple[float, float]
RewardEvent = Tuple[float, str, float]  # (t, agent_id, value)


@dataclass
class GameManager:
    """
    Game state + reward routing for the non-GPU game field (not used by PPO/GPU training).

    Research invariants:
      - Rewards are emitted ONLY as per-agent events (no global reward returned).
      - agent_id in reward events is ALWAYS a non-empty string.
      - Flag state is always consistent after each tick (carrier/taken/pos align).
      - PBRS uses gamma-correct shaping: coef * (gamma * Phi(s') - Phi(s)).
      - Supports optional episode-sticky dynamics configuration (speed, drift, etc.)
        via set_dynamics_config() for vector-env compatibility.
    """

    cols: int
    rows: int

    # --- score/time ---
    blue_score: int = 0
    red_score: int = 0
    score_limit: int = 3

    max_time: float = 300.0
    current_time: float = 300.0
    sim_time: float = 0.0
    game_over: bool = False
    phase_name: str = "OP1"
    # Naval framing: if True, on timeout Blue wins (defense held) regardless of score
    timeout_blue_wins_defense_held: bool = False
    # Rules profile (historical; kept for compatibility, but OURS behavior is always used).
    rules_profile: str = "OURS"

    # --- flags ---
    blue_flag_home: Cell = (0, 0)
    red_flag_home: Cell = (0, 0)

    blue_flag_position: Cell = (0, 0)
    red_flag_position: Cell = (0, 0)

    blue_flag_taken: bool = False
    red_flag_taken: bool = False

    blue_flag_carrier: Optional[Any] = None
    red_flag_carrier: Optional[Any] = None

    blue_flag_drop_time: Optional[float] = None
    red_flag_drop_time: Optional[float] = None
    last_score_time: float = 0.0

    # --- reward event buffer ---
    reward_events: List[RewardEvent] = field(default_factory=list)

    # --- episode telemetry (minimal, optional) ---
    blue_mine_kills_this_episode: int = 0
    red_mine_kills_this_episode: int = 0
    # Per-agent flag captures this episode (for eval: coordination / variance)
    blue_captures_this_episode: List[int] = field(default_factory=list)
    # Blue team reward sum this episode (for eval: reward per timestep)
    blue_episode_reward: float = 0.0
    mines_placed_in_enemy_half_this_episode: int = 0
    mines_triggered_by_red_this_episode: int = 0

    # --- IROS-style metrics (Top 5) ---
    time_to_first_score: Optional[float] = None  # sim_time when first score occurred
    time_to_game_over: Optional[float] = None   # sim_time when game ended
    collision_count_this_episode: int = 0   # per-tick contact count (inflates when stuck)
    collision_events_this_episode: int = 0  # unique collision events (enter-collision only)
    near_miss_count_this_episode: int = 0
    blue_inter_robot_distances: List[float] = field(default_factory=list)
    blue_zone_visited_cells: Set[Cell] = field(default_factory=set)
    total_blue_zone_cells: int = 0  # set by game_field for coverage denominator

    # --- routing memory (ids seen) ---
    blue_agent_ids_seen: Set[str] = field(default_factory=set)
    red_agent_ids_seen: Set[str] = field(default_factory=set)

    # --- shaping gamma (should match trainer gamma) ---
    shaping_gamma: float = DEFAULT_SHAPING_GAMMA

    # --- optional env binding for precise team membership ---
    game_field: Optional[Any] = field(default=None, repr=False, compare=False)

    # --- dynamics configuration (episode-sticky knobs: speed, drift, sensors) ---
    dynamics_config: Optional[Dict[str, Any]] = field(default=None, repr=False, compare=False)

    # -------------------------
    # Binding / config
    # -------------------------

    def bind_game_field(self, game_field: Any) -> None:
        """Bind environment for exact team reward routing (recommended)."""
        self.game_field = game_field

    def record_tick_metrics(
        self,
        collision_delta: int = 0,
        near_miss_delta: int = 0,
        collision_events_delta: int = 0,  # only when pair enters collision (recommended for collision_free)
        blue_inter_robot_dist: Optional[float] = None,
        blue_zone_cells_this_tick: Optional[Set[Cell]] = None,
    ) -> None:
        """IROS-style metrics: called by game_field each tick.
        collision_delta: per-tick contact count (legacy, can inflate).
        collision_events_delta: count only when a pair *enters* collision (recommended for collision_free_rate).
        """
        self.collision_count_this_episode += int(collision_delta)
        self.near_miss_count_this_episode += int(near_miss_delta)
        self.collision_events_this_episode += int(collision_events_delta)
        if blue_inter_robot_dist is not None and math.isfinite(blue_inter_robot_dist):
            self.blue_inter_robot_distances.append(float(blue_inter_robot_dist))
        if blue_zone_cells_this_tick:
            self.blue_zone_visited_cells.update(blue_zone_cells_this_tick)

    def set_phase(self, phase: str) -> None:
        """Set curriculum phase name (canonical uppercase)."""
        self.phase_name = str(phase).upper().strip()

    def set_rules_profile(self, profile: str) -> None:
        """
        Kept for backward compatibility; always uses OURS semantics.
        """
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
        self.shaping_gamma = g

    # ---- dynamics config (vector-env env_method compatibility) ----

    def set_dynamics_config(self, cfg: Optional[Dict[str, Any]]) -> None:
        """
        Store episode-sticky dynamics configuration.

        This exists primarily to keep vectorized training calls stable:
            venv.env_method("set_dynamics_config", cfg)

        Config is not applied automatically here; it is a central, consistent
        place to read dynamics knobs in GameField/Manager logic.

        Common keys:
          - "blue_speed_mult": float
          - "red_speed_mult": float
          - "opponent_kind": "scripted"|"species"|"snapshot"
          - "opponent_key": e.g. "OP1"/"BALANCED"/snapshot path label
          - "scripted_speed_mult": {"OP1": 0.9, "OP2": 1.0, "OP3": 1.1}
          - "species_speed_mult": {"BALANCED": 1.0, "FAST": 1.15, ...}
          - "snapshot_speed_mult": float
          - "opponent_speed_mult": float (global red multiplier)
          - (future) "current_drift": (dx,dy), "sensor_noise": {...}, etc.
        """
        if cfg is None:
            self.dynamics_config = None
            return
        if not isinstance(cfg, dict):
            raise TypeError(f"dynamics config must be dict or None, got {type(cfg)}")
        # Shallow copy to avoid external mutation surprises.
        self.dynamics_config = dict(cfg)
        rp = self.dynamics_config.get("rules_profile", None)
        if rp is not None:
            self.set_rules_profile(str(rp))

    def get_dynamics_config(self) -> Optional[Dict[str, Any]]:
        """Get a safe copy of the current dynamics config (or None)."""
        return None if self.dynamics_config is None else dict(self.dynamics_config)

    def _cfg_get(self, key: str, default: Any) -> Any:
        cfg = self.dynamics_config
        if not cfg:
            return default
        return cfg.get(key, default)

    def get_team_speed_multiplier(self, side: str) -> float:
        """
        Team-wide speed multiplier. Defaults to 1.0.
        Supported keys: "blue_speed_mult", "red_speed_mult"
        """
        side = str(side).lower().strip()
        raw = self._cfg_get("blue_speed_mult", 1.0) if side == "blue" else self._cfg_get("red_speed_mult", 1.0)
        try:
            v = float(raw)
        except Exception:
            return 1.0
        if not math.isfinite(v) or v <= 0.0:
            return 1.0
        return float(v)

    def get_agent_speed_multiplier(self, agent: Any) -> float:
        """
        Per-agent speed multiplier. Defaults to team multiplier.
        Applies opponent-specific tables to RED by default (common use-case).

        See set_dynamics_config docstring for supported keys.
        """
        if agent is None:
            return 1.0

        side = str(getattr(agent, "side", "")).lower().strip()
        if side not in ("blue", "red"):
            return 1.0

        base = self.get_team_speed_multiplier(side)

        # Most experiments: only vary opponent (red). Blue remains consistent.
        if side != "red":
            return base

        cfg = self.dynamics_config or {}

        # Global red multiplier
        try:
            opp_mult = float(cfg.get("opponent_speed_mult", 1.0))
            if math.isfinite(opp_mult) and opp_mult > 0.0:
                base *= opp_mult
        except Exception:
            pass

        # Opponent identity
        kind = str(cfg.get("opponent_kind", "")).lower().strip()
        key = str(cfg.get("opponent_key", "")).upper().strip()

        # Fallback inference if you ever tag agents (optional)
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

    def get_episode_dynamics_summary(self) -> Dict[str, Any]:
        """Small, paper-friendly summary for logging."""
        cfg = self.get_dynamics_config() or {}
        return {
            "blue_speed_mult": cfg.get("blue_speed_mult", 1.0),
            "red_speed_mult": cfg.get("red_speed_mult", 1.0),
            "opponent_kind": cfg.get("opponent_kind", None),
            "opponent_key": cfg.get("opponent_key", None),
        }

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
        if agent_id is None or str(agent_id).strip() == "":
            raise ValueError("agent_id must be a non-empty string.")
        try:
            v = float(value)
        except Exception:
            return
        if not math.isfinite(v):
            return
        t = self.sim_time if timestamp is None else float(timestamp)
        self.reward_events.append((float(t), str(agent_id), float(v)))

    def add_agent_reward(self, agent: Any, value: float, timestamp: Optional[float] = None) -> None:
        if agent is None:
            return
        self._remember_agent(agent)
        self.add_reward_event(value, agent_id=self._agent_uid(agent), timestamp=timestamp)
        if str(getattr(agent, "side", "")).lower() == "blue":
            self.blue_episode_reward += float(value)

    def add_team_reward(
        self,
        side: str,
        value: float,
        timestamp: Optional[float] = None,
        exclude_agent: Optional[Any] = None,
        include_disabled: bool = False,
    ) -> None:
        """
        Emit per-agent events for all agents on `side`.
        Uses bound env membership if available; falls back to ids_seen.

        FIX: side is normalized to lowercase to prevent "BLUE"/"Red" routing bugs.
        """
        side = str(side).lower().strip()
        if side not in ("blue", "red"):
            return
        ex_uid = self._agent_uid(exclude_agent) if exclude_agent is not None else None

        gf = self.game_field
        if gf is not None:
            team = gf.blue_agents if side == "blue" else gf.red_agents
            for a in team:
                if a is None:
                    continue
                if (not include_disabled) and hasattr(a, "isEnabled") and (not a.isEnabled()):
                    continue
                uid = self._agent_uid(a)
                if ex_uid is not None and uid == ex_uid:
                    continue
                self._remember_agent(a)
                self.add_reward_event(value, agent_id=uid, timestamp=timestamp)
            return

        ids = self.blue_agent_ids_seen if side == "blue" else self.red_agent_ids_seen
        for uid in ids:
            if ex_uid is not None and uid == ex_uid:
                continue
            self.add_reward_event(value, agent_id=uid, timestamp=timestamp)

    def pop_reward_events(self) -> List[RewardEvent]:
        events = self.reward_events
        self.reward_events = []
        return events

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
    # Legacy hook; no-op for OURS.
        if False:
            self.score_limit = max(self.score_limit, 9)
        if reset_scores:
            self.blue_score = 0
            self.red_score = 0

        self.game_over = False
        self.current_time = float(self.max_time)
        self.sim_time = 0.0
        self.reward_events.clear()

        self.blue_mine_kills_this_episode = 0
        self.red_mine_kills_this_episode = 0
        self.mines_placed_in_enemy_half_this_episode = 0
        self.mines_triggered_by_red_this_episode = 0

        self.collision_count_this_episode = 0
        self.collision_events_this_episode = 0
        self.near_miss_count_this_episode = 0
        self.time_to_first_score = None
        self.time_to_game_over = None
        self.blue_inter_robot_distances.clear()
        self.blue_zone_visited_cells.clear()
        self.total_blue_zone_cells = 0

        self.blue_agent_ids_seen.clear()
        self.red_agent_ids_seen.clear()

        self.blue_captures_this_episode.clear()
        self.blue_episode_reward = 0.0
        if self.game_field is not None:
            blue_agents = getattr(self.game_field, "blue_agents", []) or []
            while len(self.blue_captures_this_episode) < len(blue_agents):
                self.blue_captures_this_episode.append(0)

        mid_row = self.rows // 2
        self.blue_flag_home = self._clamp_cell(2, mid_row)
        self.red_flag_home = self._clamp_cell(self.cols - 3, mid_row)

        self.blue_flag_position = self.blue_flag_home
        self.red_flag_position = self.red_flag_home

        self.blue_flag_taken = False
        self.red_flag_taken = False
        self.blue_flag_carrier = None
        self.red_flag_carrier = None

        self.blue_flag_drop_time = None
        self.red_flag_drop_time = None
        self.last_score_time = 0.0

        # NOTE: dynamics_config is intentionally NOT cleared here by default.
        # If you want it episode-scoped, uncomment:
        # self.dynamics_config = None

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
        if (
            (not self.blue_flag_taken)
            and self.blue_flag_position != self.blue_flag_home
            and self.blue_flag_drop_time is not None
        ):
            if self.sim_time - self.blue_flag_drop_time >= FLAG_RETURN_DELAY:
                self.blue_flag_position = self.blue_flag_home
                self.blue_flag_drop_time = None

        if (
            (not self.red_flag_taken)
            and self.red_flag_position != self.red_flag_home
            and self.red_flag_drop_time is not None
        ):
            if self.sim_time - self.red_flag_drop_time >= FLAG_RETURN_DELAY:
                self.red_flag_position = self.red_flag_home
                self.red_flag_drop_time = None

    # -------------------------
    # Flag sanity / helpers
    # -------------------------

    def sanity_check_flags(self) -> None:
        """
        Ensures carrier/taken/position is consistent.
        If a carrier is disabled, the flag drops at the carrier cell with a drop_time.
        """
        # Blue flag
        if self.blue_flag_taken:
            if self.blue_flag_carrier is None:
                self.blue_flag_taken = False
                self.blue_flag_position = self.blue_flag_home
                self.blue_flag_drop_time = None
            else:
                carrier_enabled = getattr(self.blue_flag_carrier, "isEnabled", lambda: True)()
                if not carrier_enabled:
                    self.drop_flag_if_carrier_disabled(self.blue_flag_carrier)
                else:
                    self.blue_flag_position = self._agent_cell(self.blue_flag_carrier)
                    self.blue_flag_drop_time = None

        # Red flag
        if self.red_flag_taken:
            if self.red_flag_carrier is None:
                self.red_flag_taken = False
                self.red_flag_position = self.red_flag_home
                self.red_flag_drop_time = None
            else:
                carrier_enabled = getattr(self.red_flag_carrier, "isEnabled", lambda: True)()
                if not carrier_enabled:
                    self.drop_flag_if_carrier_disabled(self.red_flag_carrier)
                else:
                    self.red_flag_position = self._agent_cell(self.red_flag_carrier)
                    self.red_flag_drop_time = None

    def get_enemy_flag_position(self, side: str) -> Cell:
        # normalize side
        side = str(side).lower().strip()
        if side == "blue":
            return (
                self._agent_cell(self.red_flag_carrier)
                if (self.red_flag_taken and self.red_flag_carrier is not None)
                else self.red_flag_position
            )
        return (
            self._agent_cell(self.blue_flag_carrier)
            if (self.blue_flag_taken and self.blue_flag_carrier is not None)
            else self.blue_flag_position
        )

    def get_team_zone_center(self, side: str) -> Cell:
        return self.blue_flag_home if str(side).lower().strip() == "blue" else self.red_flag_home

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
        # Which flag we are trying to pick: enemy's flag. Block only if we already hold it.
        # (Our flag being held by the enemy does NOT block us from taking theirs.)
        if side == "blue":
            enemy_taken = self.red_flag_taken  # red's flag taken by blue
            enemy_pos = self.red_flag_position
        else:
            enemy_taken = self.blue_flag_taken  # blue's flag taken by red
            enemy_pos = self.blue_flag_position

        if enemy_taken:
            return False  # already carrying that flag

        if math.hypot(ax - float(enemy_pos[0]), ay - float(enemy_pos[1])) > 1.0:
            return False

        # Take it
        if side == "blue":
            self.red_flag_taken = True
            self.red_flag_carrier = agent
            self.red_flag_position = self._agent_cell(agent)
            self.red_flag_drop_time = None
        else:
            self.blue_flag_taken = True
            self.blue_flag_carrier = agent
            self.blue_flag_position = self._agent_cell(agent)
            self.blue_flag_drop_time = None

        if hasattr(agent, "setCarryingFlag"):
            agent.setCarryingFlag(True)

        # Aquaticus profile: grab contributes to score immediately.
        # Legacy profile: score increments only on capture.
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

        # Blue scores carrying red flag at blue home
        if side == "blue" and self.red_flag_taken and (self.red_flag_carrier is agent):
            if math.hypot(ax - float(self.blue_flag_home[0]), ay - float(self.blue_flag_home[1])) <= 2.0:
                self.blue_score += int(self._capture_score_delta())
                if self.time_to_first_score is None:
                    self.time_to_first_score = float(self.sim_time)
                self._reset_red_flag_to_home()
                self.last_score_time = float(self.sim_time)

                if hasattr(agent, "setCarryingFlag"):
                    agent.setCarryingFlag(False, scored=True)

                cap_reward = self._flag_capture_reward()
                self.add_agent_reward(agent, cap_reward)
                self.add_team_reward("blue", cap_reward * 0.5, exclude_agent=agent)
                self.add_team_reward("red", TEAM_FLAG_SCORED_PENALTY)

                # Per-agent capture count for eval (coordination / variance)
                if self.game_field is not None:
                    blue_agents = getattr(self.game_field, "blue_agents", []) or []
                    try:
                        idx = blue_agents.index(agent)
                        while len(self.blue_captures_this_episode) <= idx:
                            self.blue_captures_this_episode.append(0)
                        self.blue_captures_this_episode[idx] += 1
                    except (ValueError, AttributeError):
                        pass

                return True

        # Red scores carrying blue flag at red home
        phase = str(getattr(self, "phase_name", "")).upper()
        if phase in ("OP1", "OP2") and side == "red":
            return False
        if side == "red" and self.blue_flag_taken and (self.blue_flag_carrier is agent):
            if math.hypot(ax - float(self.red_flag_home[0]), ay - float(self.red_flag_home[1])) <= 2.0:
                self.red_score += int(self._capture_score_delta())
                if self.time_to_first_score is None:
                    self.time_to_first_score = float(self.sim_time)
                self._reset_blue_flag_to_home()
                self.last_score_time = float(self.sim_time)

                if hasattr(agent, "setCarryingFlag"):
                    agent.setCarryingFlag(False, scored=True)

                cap_reward = self._flag_capture_reward()
                self.add_agent_reward(agent, cap_reward)
                self.add_team_reward("red", cap_reward * 0.5, exclude_agent=agent)
                # Concede: team-wide negative so Blue learns to defend (block/intercept/deny)
                self.add_team_reward("blue", TEAM_FLAG_SCORED_PENALTY)

                return True

        return False

    def _reset_blue_flag_to_home(self) -> None:
        self.blue_flag_taken = False
        self.blue_flag_carrier = None
        self.blue_flag_position = self.blue_flag_home
        self.blue_flag_drop_time = None

    def _reset_red_flag_to_home(self) -> None:
        self.red_flag_taken = False
        self.red_flag_carrier = None
        self.red_flag_position = self.red_flag_home
        self.red_flag_drop_time = None

    def drop_flag_if_carrier_disabled(self, agent: Any, punish: bool = False) -> None:
        """
        Drop the carried flag at the carrier's current cell.

        IMPORTANT:
          - punish=False for normal "death/disabled" drops (called by sanity_check_flags)
          - punish=True only when you explicitly decide a macro/action "failed" and want to penalize
        """
        drop_pos = self._agent_cell(agent)
        self._remember_agent(agent)

        if self.blue_flag_carrier is agent:
            self.blue_flag_taken = False
            self.blue_flag_carrier = None
            self.blue_flag_position = drop_pos
            self.blue_flag_drop_time = self.sim_time

            if punish:
                self.add_agent_reward(agent, ACTION_FAILED_PUNISHMENT)

            if hasattr(agent, "setCarryingFlag"):
                agent.setCarryingFlag(False, scored=False)
            self.add_team_reward("blue", TEAM_FLAG_RECOVER_REWARD)

        elif self.red_flag_carrier is agent:
            self.red_flag_taken = False
            self.red_flag_carrier = None
            self.red_flag_position = drop_pos
            self.red_flag_drop_time = self.sim_time

            if punish:
                self.add_agent_reward(agent, ACTION_FAILED_PUNISHMENT)

            if hasattr(agent, "setCarryingFlag"):
                agent.setCarryingFlag(False, scored=False)
            self.add_team_reward("red", TEAM_FLAG_RECOVER_REWARD)

    def clear_flag_carrier_if_agent(self, agent: Any) -> None:
        """
        Hard reset carriers if an agent is removed from game unexpectedly.
        Prefer drop_flag_if_carrier_disabled() in normal flow.
        """
        if self.blue_flag_carrier is agent:
            self._reset_blue_flag_to_home()
            if hasattr(agent, "setCarryingFlag"):
                agent.setCarryingFlag(False, scored=False)

        if self.red_flag_carrier is agent:
            self._reset_red_flag_to_home()
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

        # Define goal for shaping:
        # - if I am carrier: go home
        # - if teammate is carrier: also go home (support)
        # - if enemy is carrying our flag: intercept carrier
        # - else: go to enemy flag position
        if side == "blue":
            enemy_taken = self.red_flag_taken
            enemy_carrier = self.red_flag_carrier
            my_home = self.blue_flag_home
            enemy_goal = self.get_enemy_flag_position("blue")
        else:
            enemy_taken = self.blue_flag_taken
            enemy_carrier = self.blue_flag_carrier
            my_home = self.red_flag_home
            enemy_goal = self.get_enemy_flag_position("red")

        i_am_carrier = enemy_taken and (enemy_carrier is agent)
        teammate_is_carrier = enemy_taken and (enemy_carrier is not None) and (enemy_carrier is not agent)

        # Intercept if enemy has our flag
        if side == "blue":
            enemy_has_our_flag = self.blue_flag_taken and (self.blue_flag_carrier is not None)
            carrier = self.blue_flag_carrier
        else:
            enemy_has_our_flag = self.red_flag_taken and (self.red_flag_carrier is not None)
            carrier = self.red_flag_carrier

        if enemy_has_our_flag and carrier is not None and (not i_am_carrier):
            goal_x, goal_y = self._agent_float(carrier)
        else:
            goal_x, goal_y = my_home if (i_am_carrier or teammate_is_carrier) else enemy_goal

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

        if enemy_has_our_flag and carrier is not None and (not i_am_carrier):
            prev_dc = min(max_dist, math.dist([sx, sy], [float(goal_x), float(goal_y)]))
            cur_dc = min(max_dist, math.dist([ex, ey], [float(goal_x), float(goal_y)]))
            progress = (prev_dc - cur_dc) / max_dist
            if progress > 0.0:
                self.add_agent_reward(agent, float(DEFENSE_CARRIER_PROGRESS_COEF) * float(progress))

        # Sprint A: Minimal shaping rewards for progress-to-flag/home
        self._apply_progress_rewards(agent, start_pos, end_pos, side, i_am_carrier, enemy_has_our_flag, carrier)

        ax2, ay2 = float(end_pos[0]), float(end_pos[1])

        # Teamwork: defense presence — reward staying near our flag when enemy has it.
        if enemy_has_our_flag and (not i_am_carrier):
            home = self.blue_flag_home if side == "blue" else self.red_flag_home
            dist_home = math.hypot(ax2 - float(home[0]), ay2 - float(home[1]))
            if dist_home <= float(DEFENSE_PRESENCE_RADIUS):
                self.add_agent_reward(agent, DEFENSE_PRESENCE_REWARD)

        # Teamwork: escort — reward being near our carrier when we have the flag.
        if (i_am_carrier or teammate_is_carrier) and (not i_am_carrier) and enemy_carrier is not None:
            cx, cy = self._agent_float(enemy_carrier)
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

        # OURS default: single consistent kill reward; mine kills are telemetry only.
        self.add_agent_reward(killer_agent, ENEMY_MAV_KILL_REWARD)

        # Optional team share for mine kills (explicit, excludes killer)
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

        if i_am_carrier:
            goal_x, goal_y = self.get_team_zone_center(side)
            goal = (float(goal_x), float(goal_y))
            coef = float(PROGRESS_TO_HOME_COEF)
        elif enemy_has_our_flag and carrier is not None:
            goal = self._agent_float(carrier)
            coef = float(DEFENSE_CARRIER_PROGRESS_COEF)
        else:
            if side == "blue":
                goal = (float(self.red_flag_position[0]), float(self.red_flag_position[1]))
            else:
                goal = (float(self.blue_flag_position[0]), float(self.blue_flag_position[1]))
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
