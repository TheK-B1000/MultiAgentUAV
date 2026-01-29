from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np  # action masks

from agents import Agent
from game_manager import GameManager
from macro_actions import MacroAction
from pathfinder import Pathfinder
from policies import OP1RedPolicy, OP2RedPolicy, OP3RedPolicy, Policy

# -------------------------
# Constants / configuration
# -------------------------

ARENA_WIDTH_M = 10.0
ARENA_HEIGHT_M = 4.28

CNN_COLS = 20
CNN_ROWS = 20
NUM_CNN_CHANNELS = 7

Cell = Tuple[int, int]
FloatPos = Tuple[float, float]
ExternalAction = Tuple[int, Any]  # (macro_idx, target_param)

Grid = List[List[int]]

# QMIX global state (God View)
GLOBAL_STATE_CHANNELS = 8  # 8 * 20 * 20 = 3200


# ============================================================
# Phase 2 (boat package realism): "physics shift" config
# NOTE: Trainers pass these into the env via env_method (if supported).
# If your env doesn't implement them yet, nothing breaks.
# This file IMPLEMENTS them inside GameField so they DO something.
# ============================================================

@dataclass
class BoatSimConfig:
    # Toggle
    enabled: bool = False
    physics_tag: str = "BASE"

    # Dynamics constraints (grid-cells units per second)
    max_speed_cps: float = 2.2
    max_accel_cps2: float = 2.0
    max_yaw_rate_rps: float = 4.0

    # Disturbances (cells per second)
    current_strength_cps: float = 0.0  # constant +x current
    drift_sigma_cells: float = 0.0     # gaussian drift on position per step

    # Robotics constraints
    action_delay_steps: int = 0
    actuation_noise_sigma: float = 0.0  # gaussian noise on (accel, yawrate)

    # Sensing
    sensor_range_cells: float = 9999.0
    sensor_noise_sigma_cells: float = 0.0   # gaussian noise on observed positions (cells)
    sensor_dropout_prob: float = 0.0        # randomly drop enemy detections

    # Simple collision handling
    bounce_on_wall: bool = False            # if False: stop at wall & clear path
    wall_stop_speed: float = 0.0            # speed after wall collision


# -------------------------
# Simple entities
# -------------------------

@dataclass
class Mine:
    x: int
    y: int
    owner_side: str
    owner_id: Optional[str] = None


@dataclass
class MinePickup:
    x: int
    y: int
    owner_side: str
    charges: int = 1


# -------------------------
# Map registry + loader
# -------------------------
WALL_CHARS = {"#", "1", "X"}
FREE_CHARS = {".", "0", " "}


def _normalize_name(name: str) -> str:
    return str(name).strip().lower()


_MAPS: Dict[str, Grid] = {}


def register_map(name: str, grid: Grid) -> None:
    if not name:
        raise ValueError("map name must be non-empty")
    if not grid or not grid[0]:
        raise ValueError("map grid must be non-empty")

    cols = len(grid[0])
    for row in grid:
        if len(row) != cols:
            raise ValueError("map grid must be rectangular")
        for v in row:
            if int(v) not in (0, 1):
                raise ValueError("map grid values must be 0/1")

    key = _normalize_name(name)
    _MAPS[key] = [list(map(int, r)) for r in grid]


def list_maps() -> List[str]:
    return sorted(_MAPS.keys())


def get_map(name: str) -> Grid:
    key = _normalize_name(name)
    if key not in _MAPS:
        raise KeyError(f"unknown map: {name}")
    return [row[:] for row in _MAPS[key]]


def make_empty_grid(rows: int, cols: int) -> Grid:
    r = max(1, int(rows))
    c = max(1, int(cols))
    return [[0] * c for _ in range(r)]


def parse_ascii_map(lines: Iterable[str]) -> Grid:
    raw = [line.rstrip("\n") for line in lines if line.strip("\n") != ""]
    if not raw:
        raise ValueError("map file is empty")

    width = max(len(line) for line in raw)
    grid: Grid = []
    for line in raw:
        row: List[int] = []
        for ch in line.ljust(width):
            if ch in WALL_CHARS:
                row.append(1)
            elif ch in FREE_CHARS:
                row.append(0)
            else:
                row.append(0)
        grid.append(row)
    return grid


def load_map_from_file(path: str) -> Grid:
    if not path:
        raise ValueError("map path must be non-empty")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return parse_ascii_map(f.readlines())


def register_map_file(name: str, path: str) -> None:
    grid = load_map_from_file(path)
    register_map(name, grid)


def make_game_field(
    *,
    map_name: Optional[str] = None,
    map_path: Optional[str] = None,
    rows: int = 20,
    cols: int = 20,
) -> "GameField":
    if map_path:
        grid = load_map_from_file(map_path)
    elif map_name:
        grid = get_map(map_name)
    else:
        grid = make_empty_grid(rows, cols)
    return GameField(grid)


# Built-in maps (empty baselines to keep training deterministic)
register_map("empty_20x20", make_empty_grid(20, 20))
register_map("empty_30x40", make_empty_grid(30, 40))

_MAP_DIR = os.path.join(os.path.dirname(__file__), "maps")
_EMPTY_20_PATH = os.path.join(_MAP_DIR, "empty_20x20.txt")
if os.path.exists(_EMPTY_20_PATH):
    try:
        register_map_file("empty_20x20_txt", _EMPTY_20_PATH)
    except Exception:
        pass


# -------------------------
# GameField
# -------------------------

class GameField:
    """
    2D Capture-the-Flag (CTF) simulation environment for multi-agent RL.

    Contract for trainers:
      - obs: env.build_observation(agent) -> [7,20,20] (mirrored for red)
      - macro mask: env.get_macro_mask(agent) -> [n_macros] bool
      - macro targets: env.num_macro_targets and env.get_macro_target(i)
      - external control:
          env.set_external_control(side, True/False)
          env.submit_external_actions({key: (macro_idx, target_param)})
      - QMIX:
          env.get_global_state() -> flat float32 [3200]
          env.get_global_state_grid() -> float32 [8,20,20]

    NEW:
      - internal policy wrappers:
          env.set_policy_wrapper("red", wrapper_fn)

    NEW (Phase 2 boat realism):
      - physics shift config & hooks:
          env.set_physics_enabled(True/False)
          env.set_dynamics_config(...)
          env.set_disturbance_config(...)
          env.set_robotics_constraints(...)
          env.set_sensor_config(...)
    """

    # -------- lifecycle --------

    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.row_count = len(grid)
        self.col_count = len(grid[0]) if self.row_count else 0

        self.manager = GameManager(cols=self.col_count, rows=self.row_count)
        self.manager.bind_game_field(self)  # reward routing hook (if implemented)

        self._init_macro_indexing()
        self._init_zones()

        # Core mechanics
        self.mines: List[Mine] = []
        self.mine_pickups: List[MinePickup] = []
        self.mine_radius_cells = 1.5
        self.suppression_range_cells = 2.0
        self.mines_per_team = 4
        self.max_mine_charges_per_agent = 2
        self._default_suppression_range_cells = float(self.suppression_range_cells)
        self._default_mines_per_team = int(self.mines_per_team)
        self._default_max_mine_charges_per_agent = int(self.max_mine_charges_per_agent)

        # Objective-first knobs
        self.mine_detour_disable_radius_cells: float = 8.0
        self.allow_offensive_mine_placing: bool = False
        self.allow_mines_in_op1: bool = False

        # Policy wiring
        self.use_internal_policies: bool = True
        self.policies: Dict[str, Any] = {"blue": None, "red": None}
        self.opponent_mode: str = "OP3"

        # Optional policy wrappers that override internal decision-making
        self.policy_wrappers: Dict[str, Optional[Callable[..., Any]]] = {"blue": None, "red": None}

        # External action routing
        self.external_control_for_side: Dict[str, bool] = {"blue": False, "red": False}
        self.pending_external_actions: Dict[str, ExternalAction] = {}
        self.external_missing_action_mode: str = "idle"  # "idle" or "internal"

        # Agents
        self.agents_per_team: int = 2
        self.blue_agents: List[Agent] = []
        self.red_agents: List[Agent] = []
        self.red_agents_per_team_override: Optional[int] = None
        self.red_speed_scale: float = 1.0
        # Adaptive adversaries: speed variation, deception, coordinated attack, evasion
        self.red_speed_min: Optional[float] = None  # None => use red_speed_scale for both
        self.red_speed_max: Optional[float] = None
        self.red_deception_prob: float = 0.0
        self.red_evasion_prob: float = 0.0
        self.red_sync_attack: bool = False
        self.red_sync_attack_now: bool = False
        self._red_sync_step: int = 0

        # Pathfinding
        self.pathfinder = Pathfinder(
            self.grid,
            self.row_count,
            self.col_count,
            allow_diagonal=True,
            block_corners=True,
        )

        # Spatial mapping
        self.cell_width_m = ARENA_WIDTH_M / max(1, self.col_count)
        self.cell_height_m = ARENA_HEIGHT_M / max(1, self.row_count)

        # Semantic macro targets (stable list)
        self.num_macro_targets: int = 8
        self.macro_targets: List[Cell] = []

        # Episode timing
        self.episode_seed: Optional[int] = None
        self.respawn_seconds: float = 2.0
        self.decision_interval_seconds: float = 0.7
        self.decision_cooldown_seconds_by_agent: Dict[int, float] = {}
        self.max_decisions_per_tick: int = 4

        # Deterministic seeding (optional)
        self.base_seed: Optional[int] = None
        self._rng = random.Random()

        # Agent lookup for fast attribution
        self._agent_by_id: Dict[str, Agent] = {}

        # Optional UI hook
        self.banner_queue: List[Tuple[str, Tuple[int, int, int], float]] = []

        # ==========================
        # Phase 2 boat realism state
        # ==========================
        self.boat_cfg = BoatSimConfig()
        self._phys_rng = np.random.RandomState(0)

        # Per-agent boat state
        # We store:
        #  - heading_rad: float
        #  - speed_cps: float
        # on the agent object (setattr) to avoid changing Agent class.
        #
        # Action delay buffers: unique_id -> ring buffer of actions
        self._action_delay_buffers: Dict[str, List[ExternalAction]] = {}
        self._action_delay_idx: Dict[str, int] = {}

        # Default opponent
        self.set_red_opponent("OP3")
        self.reset_default()

    # -------- macro indexing --------

    def _init_macro_indexing(self) -> None:
        """
        Canonical mapping between network macro indices and MacroAction enum.
        IMPORTANT: Trainers must match this order.
        """
        self.macro_order = [
            MacroAction.GO_TO,       # idx 0
            MacroAction.GRAB_MINE,   # idx 1
            MacroAction.GET_FLAG,    # idx 2
            MacroAction.PLACE_MINE,  # idx 3
            MacroAction.GO_HOME,     # idx 4
        ]
        self.macro_to_index = {m: i for i, m in enumerate(self.macro_order)}
        self.n_macros = len(self.macro_order)

    def macro_idx_to_action(self, idx: int) -> MacroAction:
        return self.macro_order[int(idx) % self.n_macros]

    def macro_action_to_idx(self, action: MacroAction) -> int:
        return int(self.macro_to_index[action])

    def normalize_macro(self, action_any: Any) -> Tuple[int, MacroAction]:
        if isinstance(action_any, MacroAction):
            return self.macro_action_to_idx(action_any), action_any
        try:
            idx = int(action_any)
        except Exception:
            idx = 0
        return idx % self.n_macros, self.macro_idx_to_action(idx)

    # -------- flat action codec (QMIX, optional PPO/MAPPO-flat) --------

    def get_flat_action_dim(self) -> int:
        n_targets = int(getattr(self, "num_macro_targets", 8) or 8)
        return int(self.n_macros * n_targets)

    def encode_flat_action(self, macro_idx: int, target_idx: int) -> int:
        n_targets = int(getattr(self, "num_macro_targets", 8) or 8)
        mi = int(macro_idx) % int(self.n_macros)
        ti = int(target_idx) % int(n_targets)
        return int(mi * n_targets + ti)

    def decode_flat_action(self, flat_action: int) -> Tuple[int, int]:
        n_targets = int(getattr(self, "num_macro_targets", 8) or 8)
        a = int(flat_action)
        macro_idx = int(a // n_targets) % int(self.n_macros)
        target_idx = int(a % n_targets)
        return macro_idx, target_idx

    def get_avail_flat_actions(self, agent: Agent) -> np.ndarray:
        n_targets = int(getattr(self, "num_macro_targets", 8) or 8)
        mm = np.asarray(self.get_macro_mask(agent), dtype=np.bool_).reshape(-1)
        if mm.shape != (self.n_macros,):
            mm = np.ones((self.n_macros,), dtype=np.bool_)
        if not mm.any():
            mm[:] = True
        return np.repeat(mm, int(n_targets)).astype(np.bool_)

    # -------- small utilities --------

    def _is_free_cell(self, x: int, y: int) -> bool:
        return (
            0 <= x < self.col_count
            and 0 <= y < self.row_count
            and int(self.grid[y][x]) == 0
        )

    def _clamp_cell(self, x: int, y: int) -> Cell:
        x = int(max(0, min(self.col_count - 1, x)))
        y = int(max(0, min(self.row_count - 1, y)))
        return x, y

    def _agent_cell_pos(self, agent: Any) -> Cell:
        cp = getattr(agent, "cell_pos", None)
        if isinstance(cp, (tuple, list)) and len(cp) >= 2:
            try:
                return self._clamp_cell(int(cp[0]), int(cp[1]))
            except Exception:
                pass

        for ax, ay in (("x", "y"), ("cell_x", "cell_y"), ("col", "row")):
            if hasattr(agent, ax) and hasattr(agent, ay):
                try:
                    return self._clamp_cell(int(getattr(agent, ax)), int(getattr(agent, ay)))
                except Exception:
                    pass

        fp = getattr(agent, "float_pos", None)
        if isinstance(fp, (tuple, list)) and len(fp) >= 2:
            try:
                return self._clamp_cell(int(round(fp[0])), int(round(fp[1])))
            except Exception:
                pass

        return 0, 0

    def _agent_float_pos(self, agent: Any) -> FloatPos:
        fp = getattr(agent, "float_pos", None)
        if isinstance(fp, (tuple, list)) and len(fp) >= 2:
            try:
                return float(fp[0]), float(fp[1])
            except Exception:
                pass
        try:
            return float(getattr(agent, "x", 0.0)), float(getattr(agent, "y", 0.0))
        except Exception:
            return 0.0, 0.0

    def _wrap_pi(self, ang: float) -> float:
        a = float(ang)
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def _clip(self, x: float, lo: float, hi: float) -> float:
        return float(min(max(float(x), float(lo)), float(hi)))

    # -------- GameManager safe wrappers --------

    def _gm_team_zone_center(self, side: str) -> Cell:
        gm = self.manager
        if hasattr(gm, "get_team_zone_center"):
            try:
                return tuple(gm.get_team_zone_center(side))
            except Exception:
                pass
        if str(side).lower() == "blue":
            lo, hi = self.blue_zone_col_range
        else:
            lo, hi = self.red_zone_col_range
        return (int((lo + hi) // 2), int(self.row_count // 2))

    def _gm_enemy_flag_position(self, side: str) -> Cell:
        gm = self.manager
        if hasattr(gm, "get_enemy_flag_position"):
            try:
                ex, ey = gm.get_enemy_flag_position(side)
                return int(ex), int(ey)
            except Exception:
                pass
        side = str(side).lower()
        if side == "blue":
            p = getattr(gm, "red_flag_position", (self.col_count - 1, self.row_count - 1))
        else:
            p = getattr(gm, "blue_flag_position", (0, 0))
        return int(p[0]), int(p[1])

    # ============================================================
    # Phase 2 boat realism: public API (safe no-op if unused)
    # ============================================================

    def set_physics_enabled(self, enabled: bool) -> None:
        self.boat_cfg.enabled = bool(enabled)

    def set_physics_tag(self, tag: str) -> None:
        self.boat_cfg.physics_tag = str(tag)

    def set_dynamics_config(self, max_speed_cps: float, max_accel_cps2: float, max_yaw_rate_rps: float) -> None:
        self.boat_cfg.max_speed_cps = float(max_speed_cps)
        self.boat_cfg.max_accel_cps2 = float(max_accel_cps2)
        self.boat_cfg.max_yaw_rate_rps = float(max_yaw_rate_rps)

    def set_disturbance_config(self, current_strength_cps: float, drift_sigma_cells: float = 0.0) -> None:
        self.boat_cfg.current_strength_cps = float(current_strength_cps)
        self.boat_cfg.drift_sigma_cells = float(drift_sigma_cells)

    def set_robotics_constraints(self, action_delay_steps: int, actuation_noise_sigma: float = 0.0) -> None:
        self.boat_cfg.action_delay_steps = int(max(0, action_delay_steps))
        self.boat_cfg.actuation_noise_sigma = float(max(0.0, actuation_noise_sigma))
        self._action_delay_buffers.clear()
        self._action_delay_idx.clear()

    def set_sensor_config(self, sensor_range_cells: float, sensor_noise_sigma_cells: float = 0.0, sensor_dropout_prob: float = 0.0) -> None:
        self.boat_cfg.sensor_range_cells = float(sensor_range_cells)
        self.boat_cfg.sensor_noise_sigma_cells = float(max(0.0, sensor_noise_sigma_cells))
        self.boat_cfg.sensor_dropout_prob = float(np.clip(sensor_dropout_prob, 0.0, 1.0))

    # Dict-style setters (handy for env_method payloads)
    def set_dynamics_config_dict(self, cfg: Dict[str, Any]) -> None:
        self.set_dynamics_config(
            cfg.get("max_speed_cps", self.boat_cfg.max_speed_cps),
            cfg.get("max_accel_cps2", self.boat_cfg.max_accel_cps2),
            cfg.get("max_yaw_rate_rps", self.boat_cfg.max_yaw_rate_rps),
        )

    def set_disturbance_config_dict(self, cfg: Dict[str, Any]) -> None:
        self.set_disturbance_config(
            cfg.get("current_strength_cps", self.boat_cfg.current_strength_cps),
            cfg.get("drift_sigma_cells", self.boat_cfg.drift_sigma_cells),
        )

    def set_robotics_constraints_dict(self, cfg: Dict[str, Any]) -> None:
        self.set_robotics_constraints(
            cfg.get("action_delay_steps", self.boat_cfg.action_delay_steps),
            cfg.get("actuation_noise_sigma", self.boat_cfg.actuation_noise_sigma),
        )

    def set_sensor_config_dict(self, cfg: Dict[str, Any]) -> None:
        self.set_sensor_config(
            cfg.get("sensor_range_cells", self.boat_cfg.sensor_range_cells),
            cfg.get("sensor_noise_sigma_cells", self.boat_cfg.sensor_noise_sigma_cells),
            cfg.get("sensor_dropout_prob", self.boat_cfg.sensor_dropout_prob),
        )

    # ============================================================
    # Phase 2 boat realism: internal helpers
    # ============================================================

    def _current_field(self, fx: float, fy: float) -> Tuple[float, float]:
        # Minimal: constant current pushing +x.
        return (float(self.boat_cfg.current_strength_cps), 0.0)

    def _apply_actuation_noise(self, accel: float, yaw_rate: float) -> Tuple[float, float]:
        sig = float(self.boat_cfg.actuation_noise_sigma)
        if sig <= 0.0:
            return float(accel), float(yaw_rate)
        a = float(accel) + float(self._phys_rng.normal(0.0, sig))
        w = float(yaw_rate) + float(self._phys_rng.normal(0.0, sig))
        return a, w

    def _init_agent_boat_state(self, a: Agent) -> None:
        if getattr(a, "heading_rad", None) is None:
            setattr(a, "heading_rad", 0.0)
        if getattr(a, "speed_cps", None) is None:
            setattr(a, "speed_cps", 0.0)

    def _wall_hit(self, x: float, y: float) -> bool:
        cx, cy = self._clamp_cell(int(round(x)), int(round(y)))
        return not self._is_free_cell(cx, cy)

    def _integrate_boat_follow_path(self, agent: Agent, dt: float) -> None:
        """
        Boat kinematics:
          - follow current waypoint (agent.path) using heading + speed
          - constraints: max_speed, accel, yaw_rate
          - disturbances: current + drift
        Writes back:
          agent.float_pos, agent.cell_pos, agent.x, agent.y, agent.heading_rad, agent.speed_cps
        """
        if agent is None or (not agent.isEnabled()) or dt <= 0.0:
            return

        self._init_agent_boat_state(agent)

        fx, fy = self._agent_float_pos(agent)
        heading = float(getattr(agent, "heading_rad", 0.0))
        speed = float(getattr(agent, "speed_cps", 0.0))

        # Choose desired waypoint: first element of path if present
        path = getattr(agent, "path", None)
        desired_fx, desired_fy = fx, fy
        if path and hasattr(path, "__len__") and len(path) > 0:
            # Path cells are (col,row). Move toward the next waypoint.
            try:
                wx, wy = path[0]
                desired_fx = float(wx)
                desired_fy = float(wy)
            except Exception:
                desired_fx, desired_fy = fx, fy

        dx = float(desired_fx) - float(fx)
        dy = float(desired_fy) - float(fy)
        dist = math.hypot(dx, dy)

        # If close to waypoint, pop it (and stop oscillation)
        if path and dist <= 0.25:
            try:
                # remove current waypoint
                path.pop(0)
                setattr(agent, "path", path)
            except Exception:
                pass

        # Recompute after pop
        path = getattr(agent, "path", None)
        desired_fx, desired_fy = fx, fy
        if path and hasattr(path, "__len__") and len(path) > 0:
            try:
                wx, wy = path[0]
                desired_fx = float(wx)
                desired_fy = float(wy)
            except Exception:
                desired_fx, desired_fy = fx, fy

        dx = float(desired_fx) - float(fx)
        dy = float(desired_fy) - float(fy)
        dist = math.hypot(dx, dy)

        # Desired heading
        desired_heading = heading
        if dist > 1e-6:
            desired_heading = math.atan2(dy, dx)

        # Heading control -> yaw rate command
        err = self._wrap_pi(desired_heading - heading)
        # proportional turn, clipped
        yaw_rate_cmd = self._clip(err / max(1e-6, dt), -self.boat_cfg.max_yaw_rate_rps, self.boat_cfg.max_yaw_rate_rps)

        # Speed control -> accel command
        # go faster when far, slow when close
        desired_speed = float(self.boat_cfg.max_speed_cps)
        if dist < 0.75:
            desired_speed = float(self.boat_cfg.max_speed_cps) * (dist / 0.75)
        desired_speed = self._clip(desired_speed, 0.0, self.boat_cfg.max_speed_cps)
        accel_cmd = self._clip((desired_speed - speed) / max(1e-6, dt), -self.boat_cfg.max_accel_cps2, self.boat_cfg.max_accel_cps2)

        accel_cmd, yaw_rate_cmd = self._apply_actuation_noise(accel_cmd, yaw_rate_cmd)

        # Integrate
        speed = self._clip(speed + accel_cmd * dt, 0.0, self.boat_cfg.max_speed_cps)
        heading = self._wrap_pi(heading + yaw_rate_cmd * dt)

        vx = speed * math.cos(heading)
        vy = speed * math.sin(heading)

        cx, cy = self._current_field(fx, fy)
        vx += float(cx)
        vy += float(cy)

        nfx = float(fx) + vx * dt
        nfy = float(fy) + vy * dt

        # Drift (position)
        if float(self.boat_cfg.drift_sigma_cells) > 0.0:
            nfx += float(self._phys_rng.normal(0.0, float(self.boat_cfg.drift_sigma_cells)))
            nfy += float(self._phys_rng.normal(0.0, float(self.boat_cfg.drift_sigma_cells)))

        # Clamp to bounds
        nfx = self._clip(nfx, 0.0, float(max(0, self.col_count - 1)))
        nfy = self._clip(nfy, 0.0, float(max(0, self.row_count - 1)))

        # Wall collision handling
        if self._wall_hit(nfx, nfy):
            if bool(self.boat_cfg.bounce_on_wall):
                # reflect velocity crudely by flipping heading
                heading = self._wrap_pi(heading + math.pi)
                speed = float(self.boat_cfg.wall_stop_speed)
                # keep position (don’t enter wall)
                nfx, nfy = fx, fy
            else:
                # stop and clear path so it replans next decision
                speed = float(self.boat_cfg.wall_stop_speed)
                nfx, nfy = fx, fy
                try:
                    if hasattr(agent, "clearPath"):
                        agent.clearPath()
                    else:
                        agent.path = []
                except Exception:
                    pass

        # Write back
        setattr(agent, "heading_rad", float(heading))
        setattr(agent, "speed_cps", float(speed))

        agent.float_pos = (float(nfx), float(nfy))
        agent.cell_pos = self._clamp_cell(int(round(nfx)), int(round(nfy)))
        # keep legacy fields consistent for other code paths
        agent.x = int(agent.cell_pos[0])
        agent.y = int(agent.cell_pos[1])

    # ============================================================
    # Phase 2 robotics constraints: action delay on decisions
    # ============================================================

    def _delay_key(self, agent: Agent) -> str:
        uid = getattr(agent, "unique_id", None)
        if uid is None:
            uid = f"{getattr(agent, 'side', 'x')}_{getattr(agent, 'agent_id', 0)}"
        return str(uid)

    def _apply_action_delay(self, agent: Agent, act: ExternalAction) -> ExternalAction:
        """
        Delay macro decisions by N steps (per agent).
        If N=0 => passthrough.
        If buffer not yet full, returns an idle action.
        """
        d = int(getattr(self.boat_cfg, "action_delay_steps", 0) or 0)
        if d <= 0:
            return act

        key = self._delay_key(agent)
        buf = self._action_delay_buffers.get(key)
        if buf is None or len(buf) != (d + 1):
            buf = [(self.macro_action_to_idx(MacroAction.GO_TO), self._agent_cell_pos(agent)) for _ in range(d + 1)]
            self._action_delay_buffers[key] = buf
            self._action_delay_idx[key] = 0

        idx = int(self._action_delay_idx.get(key, 0)) % (d + 1)
        # write newest
        buf[idx] = (int(act[0]), act[1])
        # read next
        read_idx = (idx + 1) % (d + 1)
        self._action_delay_idx[key] = read_idx
        return buf[read_idx]

    # -------- external action routing --------

    def _external_key_candidates(self, agent: Agent) -> List[str]:
        keys: List[str] = []
        for attr in ("slot_id", "unique_id"):
            if hasattr(agent, attr):
                try:
                    keys.append(str(getattr(agent, attr)))
                except Exception:
                    pass
        keys.append(f"{agent.side}_{getattr(agent, 'agent_id', 0)}")

        out: List[str] = []
        seen = set()
        for k in keys:
            if k and k not in seen:
                out.append(k)
                seen.add(k)
        return out

    def submit_external_actions(self, actions_by_agent: Dict[str, ExternalAction]) -> None:
        if not isinstance(actions_by_agent, dict):
            return
        for k, v in actions_by_agent.items():
            try:
                macro_val, target_param = v
                self.pending_external_actions[str(k)] = (int(macro_val), target_param)
            except Exception:
                continue

    def _consume_external_action_for_agent(self, agent: Agent) -> Optional[ExternalAction]:
        found = None
        for k in self._external_key_candidates(agent):
            if k in self.pending_external_actions:
                found = self.pending_external_actions.pop(k, None)
                break
        if found is not None:
            self._clear_pending_for_agent(agent)
        return found

    def _clear_pending_for_agent(self, agent: Agent) -> None:
        for k in self._external_key_candidates(agent):
            self.pending_external_actions.pop(k, None)

    # -------- zones / targets --------

    def _init_zones(self) -> None:
        total_cols = max(1, self.col_count)
        third = max(1, total_cols // 3)

        blue_min = 0
        blue_max = max(blue_min, third - 1)

        red_max = total_cols - 1
        red_min = min(total_cols - third, red_max)

        self.blue_zone_col_range = (blue_min, blue_max)
        self.red_zone_col_range = (red_min, red_max)

    def _init_macro_targets(self) -> None:
        self.macro_targets.clear()

        def nearest_free(x: int, y: int, radius: int = 8) -> Cell:
            x, y = self._clamp_cell(x, y)
            if self._is_free_cell(x, y):
                return (x, y)

            best: Optional[Cell] = None
            best_d2 = 10**9
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx, ny = x + dx, y + dy
                    if self._is_free_cell(nx, ny):
                        d2 = dx * dx + dy * dy
                        if d2 < best_d2:
                            best_d2 = d2
                            best = (nx, ny)
            return best if best is not None else (x, y)

        gm = self.manager
        own_home = tuple(getattr(gm, "blue_flag_home", (0, self.row_count // 2)))
        enemy_home = tuple(getattr(gm, "red_flag_home", (self.col_count - 1, self.row_count // 2)))

        own_zone_center = self._gm_team_zone_center("blue")
        enemy_zone_center = self._gm_team_zone_center("red")

        mid_col = self.col_count // 2
        mid_row = self.row_count // 2
        top_row = max(0, min(self.row_count - 1, 5))
        bottom_row = max(0, min(self.row_count - 1, self.row_count - 5))

        def_x = int(own_home[0]) + 5
        if def_x >= self.col_count:
            def_x = max(0, int(own_home[0]) - 5)
        defensive_point = (def_x, int(own_home[1]))

        targets_raw: List[Cell] = [
            (int(own_home[0]), int(own_home[1])),                  # 0
            (int(enemy_home[0]), int(enemy_home[1])),              # 1
            (int(own_zone_center[0]), int(own_zone_center[1])),    # 2
            (int(enemy_zone_center[0]), int(enemy_zone_center[1])),# 3
            (mid_col, mid_row),                                    # 4
            (mid_col, top_row),                                    # 5
            (mid_col, bottom_row),                                 # 6
            (int(defensive_point[0]), int(defensive_point[1])),    # 7
        ]

        self.macro_targets = [nearest_free(x, y) for (x, y) in targets_raw]
        self.num_macro_targets = len(self.macro_targets)

    def get_macro_target(self, index: int) -> Cell:
        if not self.macro_targets:
            self._init_macro_targets()
        if self.num_macro_targets <= 0:
            return (self.col_count // 2, self.row_count // 2)
        return self.macro_targets[int(index) % self.num_macro_targets]

    def get_all_macro_targets(self) -> List[Cell]:
        if not self.macro_targets:
            self._init_macro_targets()
        return list(self.macro_targets)

    # -------- public helpers --------

    def getGameManager(self) -> GameManager:
        return self.manager

    # -------- QMIX "God View" global state (8ch grid + flat) --------

    def get_global_state_dim(self) -> int:
        return int(GLOBAL_STATE_CHANNELS * CNN_ROWS * CNN_COLS)

    def build_global_state_grid(self) -> np.ndarray:
        grid = np.zeros((GLOBAL_STATE_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32)

        def set_chan(c: int, col: int, row: int) -> None:
            if 0 <= c < GLOBAL_STATE_CHANNELS and 0 <= col < CNN_COLS and 0 <= row < CNN_ROWS:
                grid[c, row, col] = 1.0

        mirror_x = False

        for a in getattr(self, "blue_agents", []):
            if a is None or (hasattr(a, "isEnabled") and not a.isEnabled()):
                continue
            fx, fy = self._agent_float_pos(a)
            c, r = self.float_grid_to_cnn_cell(fx, fy, mirror_x=mirror_x)
            set_chan(0, c, r)

        for a in getattr(self, "red_agents", []):
            if a is None or (hasattr(a, "isEnabled") and not a.isEnabled()):
                continue
            fx, fy = self._agent_float_pos(a)
            c, r = self.float_grid_to_cnn_cell(fx, fy, mirror_x=mirror_x)
            set_chan(1, c, r)

        for m in getattr(self, "mines", []):
            if m is None:
                continue
            c, r = self.float_grid_to_cnn_cell(float(getattr(m, "x", 0.0)), float(getattr(m, "y", 0.0)), mirror_x=mirror_x)
            if str(getattr(m, "owner_side", "")).lower() == "blue":
                set_chan(2, c, r)
            else:
                set_chan(3, c, r)

        for p in getattr(self, "mine_pickups", []):
            if p is None:
                continue
            c, r = self.float_grid_to_cnn_cell(float(getattr(p, "x", 0.0)), float(getattr(p, "y", 0.0)), mirror_x=mirror_x)
            if str(getattr(p, "owner_side", "")).lower() == "blue":
                set_chan(4, c, r)
            else:
                set_chan(5, c, r)

        gm = getattr(self, "manager", None)
        if gm is not None:
            bpos = getattr(gm, "blue_flag_position", (0, 0))
            rpos = getattr(gm, "red_flag_position", (self.col_count - 1, self.row_count - 1))
            bc, br = self.float_grid_to_cnn_cell(float(bpos[0]), float(bpos[1]), mirror_x=mirror_x)
            rc, rr = self.float_grid_to_cnn_cell(float(rpos[0]), float(rpos[1]), mirror_x=mirror_x)
            set_chan(6, bc, br)
            set_chan(7, rc, rr)

        return grid

    def get_global_state_grid(self) -> np.ndarray:
        return self.build_global_state_grid()

    def get_global_state(self) -> np.ndarray:
        g = self.build_global_state_grid()
        flat = g.reshape(-1).astype(np.float32, copy=False)
        expected = self.get_global_state_dim()
        if flat.shape[0] != expected:
            raise RuntimeError(f"get_global_state dim mismatch: got {flat.shape[0]}, expected {expected}")
        return flat

    # -------- action masking --------

    def get_macro_mask(self, agent: Agent) -> np.ndarray:
        n = self.n_macros
        mask = np.ones((n,), dtype=np.bool_)

        if agent is None or (not agent.isEnabled()):
            mask[:] = False
            return mask

        side = getattr(agent, "side", None)
        if side not in ("blue", "red"):
            mask[self.macro_action_to_idx(MacroAction.GRAB_MINE)] = False
            mask[self.macro_action_to_idx(MacroAction.PLACE_MINE)] = False
            return mask

        gm = self.manager
        phase = str(getattr(gm, "phase_name", getattr(gm, "phase", "OP1"))).upper()

        idx_grab = self.macro_action_to_idx(MacroAction.GRAB_MINE)
        idx_get = self.macro_action_to_idx(MacroAction.GET_FLAG)
        idx_place = self.macro_action_to_idx(MacroAction.PLACE_MINE)
        idx_home = self.macro_action_to_idx(MacroAction.GO_HOME)

        if agent.isCarryingFlag():
            mask[idx_get] = False
            mask[idx_grab] = False
            mask[idx_place] = False
            mask[idx_home] = True
            return mask

        if phase == "OP1" and (not getattr(self, "allow_mines_in_op1", False)):
            mask[idx_grab] = False
            mask[idx_place] = False

        charges = int(getattr(agent, "mine_charges", 0))
        if charges > 0:
            mask[idx_grab] = False

        ax, ay = self._agent_float_pos(agent)
        ex, ey = self._gm_enemy_flag_position(side)
        if math.hypot(ax - float(ex), ay - float(ey)) <= float(self.mine_detour_disable_radius_cells):
            mask[idx_grab] = False

        if charges <= 0:
            mask[idx_place] = False
        else:
            if not self.allow_offensive_mine_placing:
                own_min, own_max = self.blue_zone_col_range if side == "blue" else self.red_zone_col_range
                if not (own_min - 1 <= ax <= own_max + 1):
                    mask[idx_place] = False

        return mask

    def get_target_mask(self, agent: Agent) -> np.ndarray:
        n = int(self.num_macro_targets or 0)
        if n <= 0:
            return np.ones((0,), dtype=np.bool_)
        mask = np.ones((n,), dtype=np.bool_)

        if agent is None or (not agent.isEnabled()):
            mask[:] = False
            return mask

        ax, ay = self._agent_cell_pos(agent)
        try:
            idx_here = None
            for i in range(n):
                tx, ty = self.get_macro_target(i)
                if int(tx) == int(ax) and int(ty) == int(ay):
                    idx_here = i
                    break
            if idx_here is not None:
                mask[int(idx_here)] = False
        except Exception:
            pass

        return mask

    # -------- policy wiring --------

    def set_policies(self, blue: Any, red: Any) -> None:
        self.policies["blue"] = blue
        self.policies["red"] = red

    def set_red_opponent(self, mode: str) -> None:
        mode = str(mode).upper()
        self.opponent_mode = mode
        if mode == "OP1":
            self.red_agents_per_team_override = None
            self.red_speed_scale = 1.0
            self.red_speed_min = None
            self.red_speed_max = None
            self.red_deception_prob = 0.0
            self.red_evasion_prob = 0.0
            self.red_sync_attack = False
            self.suppression_range_cells = float(self._default_suppression_range_cells)
            self.mines_per_team = int(self._default_mines_per_team)
            self.max_mine_charges_per_agent = int(self._default_max_mine_charges_per_agent)
            self.policies["red"] = OP1RedPolicy("red")
        elif mode == "OP2":
            self.red_agents_per_team_override = None
            self.red_speed_scale = 1.0
            self.red_speed_min = None
            self.red_speed_max = None
            self.red_deception_prob = 0.0
            self.red_evasion_prob = 0.0
            self.red_sync_attack = False
            self.suppression_range_cells = float(self._default_suppression_range_cells)
            self.mines_per_team = int(self._default_mines_per_team)
            self.max_mine_charges_per_agent = int(self._default_max_mine_charges_per_agent)
            self.policies["red"] = OP2RedPolicy("red")
        elif mode in ("OP3_EASY", "OP3EASY"):
            self.red_agents_per_team_override = None
            self.red_speed_scale = 0.9
            self.red_speed_min = None
            self.red_speed_max = None
            self.red_deception_prob = 0.0
            self.red_evasion_prob = 0.0
            self.red_sync_attack = False
            self.suppression_range_cells = 1.5
            self.mines_per_team = 1
            self.max_mine_charges_per_agent = 1
            self.policies["red"] = OP3RedPolicy(
                "red",
                mine_radius_check=1.0,
                defense_radius_cells=3.0,
                patrol_radius_cells=2,
                assist_radius_mult=1.0,
                defense_weight=1.25,
                flag_weight=2.0,
            )
        elif mode in ("OP3_HARD", "OP3HARD"):
            self.red_agents_per_team_override = None
            self.red_speed_scale = 1.0
            self.red_speed_min = None
            self.red_speed_max = None
            self.red_deception_prob = 0.0
            self.red_evasion_prob = 0.0
            self.red_sync_attack = False
            self.suppression_range_cells = float(self._default_suppression_range_cells)
            self.mines_per_team = int(self._default_mines_per_team)
            self.max_mine_charges_per_agent = int(self._default_max_mine_charges_per_agent)
            self.policies["red"] = OP3RedPolicy(
                "red",
                mine_radius_check=2.8,
                defense_radius_cells=6.0,
                patrol_radius_cells=4,
                assist_radius_mult=2.0,
                defense_weight=3.0,
                flag_weight=1.0,
            )
        elif mode in ("OP3_ADAPTIVE", "OP3ADAPTIVE"):
            # Speed variation + deceptive approaches (curriculum: adaptive adversaries)
            self.red_agents_per_team_override = None
            self.red_speed_scale = 1.0
            self.red_speed_min = 0.85
            self.red_speed_max = 1.15
            self.red_deception_prob = 0.3
            self.red_evasion_prob = 0.0
            self.red_sync_attack = False
            self.suppression_range_cells = float(self._default_suppression_range_cells)
            self.mines_per_team = int(self._default_mines_per_team)
            self.max_mine_charges_per_agent = int(self._default_max_mine_charges_per_agent)
            self.policies["red"] = OP3RedPolicy(
                "red",
                mine_radius_check=2.0,
                defense_radius_cells=5.0,
                patrol_radius_cells=3,
                assist_radius_mult=1.5,
                defense_weight=2.0,
                flag_weight=2.0,
                deception_prob=0.3,
            )
        elif mode in ("OP3_SYNC", "OP3SYNC"):
            # Coordinated multi-agent attack (synchronized rush)
            self.red_agents_per_team_override = None
            self.red_speed_scale = 1.0
            self.red_speed_min = None
            self.red_speed_max = None
            self.red_deception_prob = 0.0
            self.red_evasion_prob = 0.0
            self.red_sync_attack = True
            self.suppression_range_cells = float(self._default_suppression_range_cells)
            self.mines_per_team = int(self._default_mines_per_team)
            self.max_mine_charges_per_agent = int(self._default_max_mine_charges_per_agent)
            self.policies["red"] = OP3RedPolicy(
                "red",
                mine_radius_check=2.0,
                defense_radius_cells=5.0,
                patrol_radius_cells=3,
                assist_radius_mult=1.5,
                defense_weight=2.0,
                flag_weight=2.0,
            )
        elif mode in ("OP3_EVASIVE", "OP3EVASIVE"):
            # Evasion strategies (dodge, retreat when pressured)
            self.red_agents_per_team_override = None
            self.red_speed_scale = 1.05
            self.red_speed_min = None
            self.red_speed_max = None
            self.red_deception_prob = 0.15
            self.red_evasion_prob = 0.4
            self.red_sync_attack = False
            self.suppression_range_cells = float(self._default_suppression_range_cells)
            self.mines_per_team = int(self._default_mines_per_team)
            self.max_mine_charges_per_agent = int(self._default_max_mine_charges_per_agent)
            self.policies["red"] = OP3RedPolicy(
                "red",
                mine_radius_check=2.2,
                defense_radius_cells=5.0,
                patrol_radius_cells=3,
                assist_radius_mult=1.8,
                defense_weight=2.5,
                flag_weight=1.5,
                deception_prob=0.15,
            )
        else:
            self.red_agents_per_team_override = None
            self.red_speed_scale = 1.0
            self.suppression_range_cells = float(self._default_suppression_range_cells)
            self.mines_per_team = int(self._default_mines_per_team)
            self.max_mine_charges_per_agent = int(self._default_max_mine_charges_per_agent)
            self.policies["red"] = OP3RedPolicy("red")

    def set_policy_wrapper(self, side: str, wrapper: Optional[Callable[..., Any]]) -> None:
        side = str(side).lower()
        if side not in ("blue", "red"):
            raise ValueError(f"Unknown side: {side}")
        if wrapper is not None and (not callable(wrapper)):
            raise TypeError("wrapper must be callable or None")
        self.policy_wrappers[side] = wrapper

    def clear_policy_wrapper(self, side: str) -> None:
        self.set_policy_wrapper(side, None)

    def set_red_policy_wrapper(self, wrapper: Optional[Callable[..., Any]]) -> None:
        self.set_policy_wrapper("red", wrapper)

    def set_external_control(self, side: str, external: bool) -> None:
        side = str(side).lower().strip()
        if side not in ("blue", "red"):
            raise ValueError(f"Unknown side: {side}")
        self.external_control_for_side[side] = bool(external)

    def set_all_external_control(self, external: bool) -> None:
        for side in ("blue", "red"):
            self.external_control_for_side[side] = bool(external)

    # -------- coordinate mapping --------

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        return (col + 0.5) * self.cell_width_m, (row + 0.5) * self.cell_height_m

    def world_to_cnn_cell(self, x_m: float, y_m: float, *, mirror_x: bool) -> Cell:
        u = max(0.0, min(1.0, x_m / ARENA_WIDTH_M))
        v = max(0.0, min(1.0, y_m / ARENA_HEIGHT_M))
        if mirror_x:
            u = 1.0 - u
        col = max(0, min(CNN_COLS - 1, int(u * CNN_COLS)))
        row = max(0, min(CNN_ROWS - 1, int(v * CNN_ROWS)))
        return col, row

    def float_grid_to_cnn_cell(self, fx: float, fy: float, *, mirror_x: bool) -> Cell:
        x_m = (float(fx) + 0.5) * self.cell_width_m
        y_m = (float(fy) + 0.5) * self.cell_height_m
        return self.world_to_cnn_cell(x_m, y_m, mirror_x=mirror_x)

    # -------- reset / spawn --------

    def reset_default(self) -> None:
        if self.base_seed is not None:
            self._rng.seed(int(self.base_seed))
            self.episode_seed = self._rng.randint(0, 2**31 - 1)
        else:
            self.episode_seed = random.randint(0, 2**31 - 1)

        # Seed physics RNG too (deterministic per episode if base_seed is set)
        seed_val = int(self.episode_seed or 0)
        self._phys_rng = np.random.RandomState(seed_val ^ 0xA53A1F)

        self._init_zones()
        self.manager.reset_game()

        self.mines.clear()
        self.mine_pickups.clear()
        self.pending_external_actions.clear()
        self.decision_cooldown_seconds_by_agent.clear()

        self._action_delay_buffers.clear()
        self._action_delay_idx.clear()

        self._init_macro_targets()

        self.spawn_agents()
        self.spawn_mine_pickups()

    def set_agent_count_and_reset(self, new_count: int) -> None:
        self.agents_per_team = max(1, int(new_count))
        self.manager.reset_game(reset_scores=True)
        self.reset_default()

    def set_seed(self, seed: Optional[int]) -> None:
        self.base_seed = int(seed) if seed is not None else None
        if self.base_seed is not None:
            self._rng.seed(self.base_seed)

    def _ensure_agent_ids(self, a: Agent) -> None:
        side = str(getattr(a, "side", "blue"))
        aid = int(getattr(a, "agent_id", 0))
        if getattr(a, "unique_id", None) is None:
            a.unique_id = f"{side}_{aid}"
        if getattr(a, "slot_id", None) is None:
            a.slot_id = a.unique_id
        self._agent_by_id[str(a.unique_id)] = a

    def spawn_agents(self) -> None:
        base = int(self.episode_seed or 0)
        rng = random.Random(base + 123)

        self.blue_agents.clear()
        self.red_agents.clear()
        self._agent_by_id.clear()

        blue_min_col, blue_max_col = self.blue_zone_col_range
        red_min_col, red_max_col = self.red_zone_col_range

        blue_cells = [
            (r, c)
            for r in range(self.row_count)
            for c in range(blue_min_col, blue_max_col + 1)
            if self._is_free_cell(c, r)
        ]
        red_cells = [
            (r, c)
            for r in range(self.row_count)
            for c in range(red_min_col, red_max_col + 1)
            if self._is_free_cell(c, r)
        ]
        rng.shuffle(blue_cells)
        rng.shuffle(red_cells)

        n = self.agents_per_team
        red_n = n
        if self.red_agents_per_team_override is not None:
            try:
                red_n = max(1, int(self.red_agents_per_team_override))
            except Exception:
                red_n = n

        for i in range(min(n, len(blue_cells))):
            row, col = blue_cells[i]
            a = Agent(
                x=col,
                y=row,
                side="blue",
                cols=self.col_count,
                rows=self.row_count,
                grid=self.grid,
                move_rate_cps=rng.uniform(2.0, 2.4),
                agent_id=i,
                is_miner=(i == 0),
                game_manager=self.manager,
            )
            a.spawn_xy = (col, row)
            a.game_field = self
            a.mine_charges = 0
            a.decision_count = 0
            a.suppression_timer = 0.0
            a.suppressed_this_tick = False
            self._ensure_agent_ids(a)

            if getattr(a, "float_pos", None) is None:
                a.float_pos = (float(col), float(row))
            if getattr(a, "cell_pos", None) is None:
                a.cell_pos = (int(col), int(row))

            # Boat state
            self._init_agent_boat_state(a)
            setattr(a, "heading_rad", 0.0)
            setattr(a, "speed_cps", 0.0)

            self.blue_agents.append(a)

        for i in range(min(red_n, len(red_cells))):
            row, col = red_cells[i]
            a = Agent(
                x=col,
                y=row,
                side="red",
                cols=self.col_count,
                rows=self.row_count,
                grid=self.grid,
                move_rate_cps=rng.uniform(2.0, 2.4) * float(self.red_speed_scale),
                agent_id=i,
                is_miner=(i == 0),
                game_manager=self.manager,
            )
            a.spawn_xy = (col, row)
            a.game_field = self
            a.mine_charges = 0
            a.decision_count = 0
            a.suppression_timer = 0.0
            a.suppressed_this_tick = False
            self._ensure_agent_ids(a)

            if getattr(a, "float_pos", None) is None:
                a.float_pos = (float(col), float(row))
            if getattr(a, "cell_pos", None) is None:
                a.cell_pos = (int(col), int(row))

            # Boat state
            self._init_agent_boat_state(a)
            setattr(a, "heading_rad", math.pi)  # face left-ish by default
            setattr(a, "speed_cps", 0.0)

            self.red_agents.append(a)

    def spawn_mine_pickups(self) -> None:
        self.mine_pickups.clear()

        base = int(self.episode_seed or 0)
        rng = random.Random(base + 9999)

        blue_min_col, blue_max_col = self.blue_zone_col_range
        red_min_col, red_max_col = self.red_zone_col_range

        occupied_spawns = {self._agent_cell_pos(a) for a in (self.blue_agents + self.red_agents)}

        def spawn_for_band(owner_side: str, col_min: int, col_max: int) -> None:
            all_cells = [(row, col) for row in range(self.row_count) for col in range(col_min, col_max + 1)]
            rng.shuffle(all_cells)

            flag_pos = self.manager.blue_flag_position if owner_side == "blue" else self.manager.red_flag_position
            placed = 0

            for row, col in all_cells:
                if not self._is_free_cell(col, row):
                    continue
                if (col, row) == tuple(flag_pos):
                    continue
                if (col, row) in occupied_spawns:
                    continue
                if any(p.x == col and p.y == row for p in self.mine_pickups):
                    continue

                self.mine_pickups.append(MinePickup(x=col, y=row, owner_side=owner_side, charges=1))
                placed += 1
                if placed >= self.mines_per_team:
                    break

        spawn_for_band("blue", blue_min_col, blue_max_col)
        spawn_for_band("red", red_min_col, red_max_col)

    # -------- main simulation step --------

    def update(self, delta_time: float) -> None:
        if self.manager.game_over or delta_time <= 0.0:
            return

        winner_text = self.manager.tick_seconds(delta_time)
        if winner_text:
            color = (230, 230, 230)
            if "BLUE" in winner_text:
                color = (90, 170, 250)
            elif "RED" in winner_text:
                color = (250, 120, 70)
            self.announce(winner_text, color, 3.0)
            return

        # Movement update
        if bool(self.boat_cfg.enabled):
            for agent in (self.blue_agents + self.red_agents):
                # boat physics drives motion; Agent.update may still handle internal timers/respawn
                try:
                    agent.update(delta_time)
                except Exception:
                    pass
                self._integrate_boat_follow_path(agent, float(delta_time))
        else:
            for agent in (self.blue_agents + self.red_agents):
                agent.update(delta_time)

        occupied = [self._agent_cell_pos(a) for a in (self.blue_agents + self.red_agents) if a.isEnabled()]
        if hasattr(self.pathfinder, "setDynamicObstacles"):
            self.pathfinder.setDynamicObstacles(occupied)

        for friendly_team, enemy_team in (
            (self.blue_agents, self.red_agents),
            (self.red_agents, self.blue_agents),
        ):
            for agent in friendly_team:
                if not agent.isEnabled():
                    continue

                self.apply_mine_damage(agent)
                self.apply_suppression(agent, enemy_team, delta_time)
                if not agent.isEnabled():
                    continue

                agent_key = id(agent)
                cooldown = float(self.decision_cooldown_seconds_by_agent.get(agent_key, 0.0)) - float(delta_time)

                side_external = self.external_control_for_side.get(agent.side, False)

                decisions_done = 0
                while cooldown <= 0.0 and decisions_done < int(self.max_decisions_per_tick):
                    made_decision = False

                    if side_external:
                        act = self._consume_external_action_for_agent(agent)
                        if act is not None:
                            act = self._apply_action_delay(agent, act)
                            macro_val, target_param = act
                            self.apply_macro_action(agent, macro_val, target_param)
                            made_decision = True
                        else:
                            if self.external_missing_action_mode == "internal" and self.use_internal_policies:
                                self.decide(agent)
                                made_decision = True
                            elif self.external_missing_action_mode == "idle":
                                idle = (self.macro_action_to_idx(MacroAction.GO_TO), self._agent_cell_pos(agent))
                                idle = self._apply_action_delay(agent, idle)
                                self.apply_macro_action(agent, idle[0], idle[1])
                                made_decision = True
                            else:
                                break
                    else:
                        if self.use_internal_policies:
                            self.decide(agent)
                            made_decision = True

                    if made_decision:
                        cooldown += float(self.decision_interval_seconds)
                        decisions_done += 1
                    else:
                        break

                self.decision_cooldown_seconds_by_agent[agent_key] = cooldown

                self.handle_mine_pickups(agent)
                self.apply_flag_rules(agent)

        if self.banner_queue:
            text, color, t = self.banner_queue[-1]
            t = max(0.0, t - delta_time)
            self.banner_queue[-1] = (text, color, t)
            if t <= 0.0:
                self.banner_queue.pop()

    # -------- observations --------

    def _enemy_detected(self, agent: Agent, enemy_fx: float, enemy_fy: float) -> Optional[Tuple[float, float]]:
        """
        Phase 2 sensing:
          - range gate
          - dropout
          - gaussian noise on observed position
        Returns noisy (fx,fy) or None (not detected).
        """
        cfg = self.boat_cfg
        if not bool(cfg.enabled):
            return (float(enemy_fx), float(enemy_fy))

        ax, ay = self._agent_float_pos(agent)
        d = math.hypot(float(enemy_fx) - float(ax), float(enemy_fy) - float(ay))
        if d > float(cfg.sensor_range_cells):
            return None

        if float(cfg.sensor_dropout_prob) > 0.0:
            if float(self._phys_rng.rand()) < float(cfg.sensor_dropout_prob):
                return None

        ox, oy = float(enemy_fx), float(enemy_fy)
        sig = float(cfg.sensor_noise_sigma_cells)
        if sig > 0.0:
            ox += float(self._phys_rng.normal(0.0, sig))
            oy += float(self._phys_rng.normal(0.0, sig))

        ox = self._clip(ox, 0.0, float(max(0, self.col_count - 1)))
        oy = self._clip(oy, 0.0, float(max(0, self.row_count - 1)))
        return (ox, oy)

    def build_observation(self, agent: Agent) -> List[List[List[float]]]:
        """
        Version B: "continuous-friendly" CNN observation with optional Phase 2 sensing.
        Returns: [NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS]
        Channel meanings:
          0: self
          1: friendly teammates
          2: enemies (sensor-limited if physics enabled)
          3: friendly mines
          4: enemy mines (sensor-limited if physics enabled)
          5: own flag
          6: enemy flag
        """
        side = str(getattr(agent, "side", "blue")).lower()
        gm = getattr(self, "manager", None)
        mirror_x = (side == "red")

        channels = np.zeros((NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32)

        def set_chan(c: int, col: int, row: int, v: float) -> None:
            if 0 <= col < CNN_COLS and 0 <= row < CNN_ROWS:
                if v > channels[c, row, col]:
                    channels[c, row, col] = float(v)

        def _safe_xy(pos, fallback=(0.0, 0.0)) -> Tuple[float, float]:
            if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                try:
                    return float(pos[0]), float(pos[1])
                except Exception:
                    return float(fallback[0]), float(fallback[1])
            return float(fallback[0]), float(fallback[1])

        def _grid_to_cnn_continuous(fx: float, fy: float, *, mirror: bool) -> Tuple[float, float]:
            gx = float(fx)
            gy = float(fy)
            if mirror and getattr(self, "col_count", 0) and self.col_count > 1:
                gx = float((self.col_count - 1) - gx)

            if getattr(self, "col_count", 0) and self.col_count > 0:
                gx = max(0.0, min(float(self.col_count - 1), gx))
            if getattr(self, "row_count", 0) and self.row_count > 0:
                gy = max(0.0, min(float(self.row_count - 1), gy))

            if getattr(self, "col_count", 0) and self.col_count > 1:
                cx = (gx / float(self.col_count - 1)) * float(CNN_COLS - 1)
            else:
                cx = 0.0

            if getattr(self, "row_count", 0) and self.row_count > 1:
                cy = (gy / float(self.row_count - 1)) * float(CNN_ROWS - 1)
            else:
                cy = 0.0

            return (cx, cy)

        def splat_float(c: int, fx: float, fy: float, strength: float = 1.0) -> None:
            cx, cy = _grid_to_cnn_continuous(fx, fy, mirror=mirror_x)

            x0 = int(math.floor(cx))
            y0 = int(math.floor(cy))
            x1 = min(x0 + 1, CNN_COLS - 1)
            y1 = min(y0 + 1, CNN_ROWS - 1)

            tx = cx - float(x0)
            ty = cy - float(y0)

            w00 = (1.0 - tx) * (1.0 - ty)
            w10 = tx * (1.0 - ty)
            w01 = (1.0 - tx) * ty
            w11 = tx * ty

            s = float(strength)
            set_chan(c, x0, y0, s * w00)
            set_chan(c, x1, y0, s * w10)
            set_chan(c, x0, y1, s * w01)
            set_chan(c, x1, y1, s * w11)

        friendly_team = self.blue_agents if side == "blue" else self.red_agents
        enemy_team = self.red_agents if side == "blue" else self.blue_agents

        # Self (always known)
        sx, sy = self._agent_float_pos(agent)
        splat_float(0, float(sx), float(sy), 1.0)

        # Friendlies (always known)
        for a in friendly_team:
            if a is None or a is agent:
                continue
            try:
                if hasattr(a, "isEnabled") and (not a.isEnabled()):
                    continue
            except Exception:
                pass
            fx, fy = self._agent_float_pos(a)
            splat_float(1, float(fx), float(fy), 1.0)

        # Enemies (sensor-limited if physics enabled)
        for a in enemy_team:
            if a is None:
                continue
            try:
                if hasattr(a, "isEnabled") and (not a.isEnabled()):
                    continue
            except Exception:
                pass
            fx, fy = self._agent_float_pos(a)
            det = self._enemy_detected(agent, fx, fy)
            if det is None:
                continue
            splat_float(2, float(det[0]), float(det[1]), 1.0)

        # Mines: friendly always visible; enemy mines sensor-limited if physics enabled
        for m in getattr(self, "mines", []):
            mx, my = float(getattr(m, "x", 0)), float(getattr(m, "y", 0))
            owner = str(getattr(m, "owner_side", "")).lower()
            if owner == side:
                splat_float(3, mx, my, 1.0)
            else:
                det = self._enemy_detected(agent, mx, my)  # reuse same sensing model
                if det is None:
                    continue
                splat_float(4, float(det[0]), float(det[1]), 1.0)

        # Flags (kept always known for objective stability)
        def _gm_attr(*names, default=None):
            if gm is None:
                return default
            for n in names:
                if hasattr(gm, n):
                    return getattr(gm, n)
            return default

        if side == "blue":
            own_flag_pos = _safe_xy(_gm_attr("blue_flag_position", "blue_flag_home", default=(0, 0)), (0.0, 0.0))
            enemy_flag_pos = _safe_xy(self._gm_enemy_flag_position("blue"),
                                      (float(getattr(self, "col_count", CNN_COLS) - 1),
                                       float(getattr(self, "row_count", CNN_ROWS) // 2)))
        else:
            own_flag_pos = _safe_xy(_gm_attr("red_flag_position", "red_flag_home",
                                             default=(getattr(self, "col_count", CNN_COLS) - 1,
                                                      getattr(self, "row_count", CNN_ROWS) - 1)),
                                    (float(getattr(self, "col_count", CNN_COLS) - 1),
                                     float(getattr(self, "row_count", CNN_ROWS) - 1)))
            enemy_flag_pos = _safe_xy(self._gm_enemy_flag_position("red"),
                                      (0.0, float(getattr(self, "row_count", CNN_ROWS) // 2)))

        splat_float(5, float(own_flag_pos[0]), float(own_flag_pos[1]), 1.0)
        splat_float(6, float(enemy_flag_pos[0]), float(enemy_flag_pos[1]), 1.0)

        return channels.tolist()

    # -------- macro actions --------
    # (UNCHANGED logic, but now path-following can be physics-driven when enabled)

    def apply_macro_action(self, agent: Agent, action: Any, param: Optional[Any] = None) -> MacroAction:
        if agent is None or (not agent.isEnabled()):
            return MacroAction.GO_TO

        macro_idx, action = self.normalize_macro(action)
        side = str(getattr(agent, "side", "blue")).lower()
        gm = self.manager

        if action == MacroAction.GET_FLAG and agent.isCarryingFlag():
            action = MacroAction.GO_HOME

        def record(executed: MacroAction) -> MacroAction:
            agent.last_macro_action = executed
            try:
                agent.last_macro_action_idx = self.macro_action_to_idx(executed)
            except Exception:
                agent.last_macro_action_idx = int(macro_idx)
            return executed

        def resolve_target(default_target: Cell) -> Cell:
            if param is None:
                return default_target
            if isinstance(param, (tuple, list)) and len(param) == 2:
                return self._clamp_cell(int(param[0]), int(param[1]))
            try:
                idx = int(param)
            except (TypeError, ValueError):
                return default_target
            return self.get_macro_target(idx)

        def safe_set_path(target: Cell, *, avoid_enemies: bool, radius: int = 1) -> None:
            start = self._agent_cell_pos(agent)
            tgt = self._clamp_cell(int(target[0]), int(target[1]))

            if start == tgt:
                if hasattr(agent, "clearPath"):
                    agent.clearPath()
                else:
                    agent.path = []
                setattr(agent, "current_goal", None)
                return

            try:
                if getattr(agent, "current_goal", None) == tgt:
                    p = getattr(agent, "path", None)
                    if p is not None and hasattr(p, "__len__") and len(p) > 0:
                        return
            except Exception:
                pass

            setattr(agent, "current_goal", tgt)

            danger_saved = dict(getattr(self.pathfinder, "danger_cost", {}) or {})
            danger: Dict[Cell, float] = {}

            do_avoid = bool(avoid_enemies) and side in ("blue", "red") and int(radius) > 0
            if do_avoid:
                enemy_team = self.red_agents if side == "blue" else self.blue_agents
                base_penalty = 3.0
                max_penalty = 8.0
                rr = int(max(1, radius))

                for e in enemy_team:
                    if e is None:
                        continue
                    try:
                        if hasattr(e, "isEnabled") and (not e.isEnabled()):
                            continue
                    except Exception:
                        pass

                    ex, ey = self._agent_cell_pos(e)

                    for dx in range(-rr, rr + 1):
                        for dy in range(-rr, rr + 1):
                            cx, cy = ex + dx, ey + dy
                            if not (0 <= cx < self.col_count and 0 <= cy < self.row_count):
                                continue
                            d = max(abs(dx), abs(dy))
                            pen = base_penalty * float(rr + 1 - d)
                            if pen <= 0.0:
                                continue
                            prev = float(danger.get((cx, cy), 0.0))
                            danger[(cx, cy)] = min(max_penalty, max(prev, float(pen)))

                danger.pop(start, None)
                danger.pop(tgt, None)

            path: Optional[List[Cell]] = None
            try:
                if do_avoid and hasattr(self.pathfinder, "setDangerCosts"):
                    self.pathfinder.setDangerCosts(danger)
                elif hasattr(self.pathfinder, "clearDangerCosts"):
                    self.pathfinder.clearDangerCosts()
                else:
                    setattr(self.pathfinder, "danger_cost", danger if do_avoid else {})

                path = self.pathfinder.astar(start, tgt)
            except Exception:
                path = None
            finally:
                if hasattr(self.pathfinder, "setDangerCosts"):
                    try:
                        self.pathfinder.setDangerCosts(danger_saved)
                    except Exception:
                        setattr(self.pathfinder, "danger_cost", danger_saved)
                else:
                    setattr(self.pathfinder, "danger_cost", danger_saved)

            def prune_collinear(p: List[Cell]) -> List[Cell]:
                if not p or len(p) < 3:
                    return p
                out: List[Cell] = [p[0], p[1]]
                for cur in p[2:]:
                    a = out[-2]
                    b = out[-1]
                    da = (b[0] - a[0], b[1] - a[1])
                    db = (cur[0] - b[0], cur[1] - b[1])
                    if da == db:
                        out[-1] = cur
                    else:
                        out.append(cur)
                return out

            if path:
                try:
                    path = prune_collinear(path)
                except Exception:
                    pass

            if hasattr(agent, "setPath"):
                agent.setPath(path or [])
            else:
                agent.path = path or []

        if action == MacroAction.GO_TO:
            target = resolve_target(self.get_macro_target(4))
            safe_set_path(target, avoid_enemies=agent.isCarryingFlag(), radius=1)
            return record(MacroAction.GO_TO)

        if action == MacroAction.GET_FLAG:
            ex, ey = self._gm_enemy_flag_position(side)
            safe_set_path((int(ex), int(ey)), avoid_enemies=False, radius=1)
            return record(MacroAction.GET_FLAG)

        if action == MacroAction.GO_HOME:
            home = self._gm_team_zone_center(side)
            safe_set_path(home, avoid_enemies=True, radius=2)
            return record(MacroAction.GO_HOME)

        if action == MacroAction.GRAB_MINE:
            if int(getattr(agent, "mine_charges", 0)) > 0:
                ex, ey = self._gm_enemy_flag_position(side)
                safe_set_path((int(ex), int(ey)), avoid_enemies=False, radius=1)
                return record(MacroAction.GET_FLAG)

            ax, ay = self._agent_float_pos(agent)
            ex, ey = self._gm_enemy_flag_position(side)
            if math.hypot(ax - float(ex), ay - float(ey)) <= float(self.mine_detour_disable_radius_cells):
                safe_set_path((int(ex), int(ey)), avoid_enemies=False, radius=1)
                return record(MacroAction.GET_FLAG)

            my_pickups = [p for p in self.mine_pickups if p.owner_side == side]
            if my_pickups:
                axc, ayc = self._agent_cell_pos(agent)
                nearest = min(my_pickups, key=lambda p: (p.x - axc) ** 2 + (p.y - ayc) ** 2)
                target = (nearest.x, nearest.y)
            else:
                target = self.get_macro_target(2)

            safe_set_path(target, avoid_enemies=False, radius=1)
            return record(MacroAction.GRAB_MINE)

        if action == MacroAction.PLACE_MINE:
            target = resolve_target(self._agent_cell_pos(agent))

            if int(getattr(agent, "mine_charges", 0)) > 0:
                if (not self.allow_offensive_mine_placing) and side in ("blue", "red"):
                    own_min, own_max = self.blue_zone_col_range if side == "blue" else self.red_zone_col_range
                    ax_cell, _ = self._agent_cell_pos(agent)
                    if not (own_min - 1 <= float(ax_cell) <= own_max + 1):
                        safe_set_path(self._gm_team_zone_center(side), avoid_enemies=False, radius=1)
                        return record(MacroAction.GO_TO)

                tx, ty = self._clamp_cell(int(target[0]), int(target[1]))
                own_flag_home = getattr(gm, "blue_flag_home", (0, 0)) if side == "blue" else getattr(
                    gm, "red_flag_home", (self.col_count - 1, self.row_count - 1)
                )

                if self._is_free_cell(tx, ty) and not any(m.x == tx and m.y == ty for m in self.mines):
                    if (tx, ty) != tuple(own_flag_home):
                        self.mines.append(Mine(x=tx, y=ty, owner_side=side, owner_id=getattr(agent, "unique_id", None)))
                        agent.mine_charges -= 1
                        if hasattr(self.manager, "reward_mine_placed"):
                            try:
                                self.manager.reward_mine_placed(agent, mine_pos=(tx, ty))
                            except Exception:
                                pass

            safe_set_path(target, avoid_enemies=False, radius=1)
            return record(MacroAction.PLACE_MINE)

        safe_set_path(self.get_macro_target(4), avoid_enemies=False, radius=1)
        return record(MacroAction.GO_TO)

    # -------- internal decision (scripted or neural policy or wrapper) --------

    def decide(self, agent: Agent) -> None:
        if not agent.isEnabled():
            return

        def _to_int(x: Any, default: int = 0) -> int:
            try:
                import torch
                if torch.is_tensor(x):
                    return int(x.reshape(-1)[0].item())
            except Exception:
                pass
            try:
                return int(x)
            except Exception:
                return int(default)

        def _parse_out(out: Any) -> Tuple[int, Optional[Any]]:
            if out is None:
                return 0, None

            if isinstance(out, dict):
                ma = out.get("macro_action", 0)
                ta = out.get("target_action", None)
                macro_id = _to_int(ma, 0)
                if ta is None:
                    return macro_id, None
                target_param = _to_int(ta, 0)
                return macro_id, target_param

            if isinstance(out, (tuple, list)):
                if len(out) == 0:
                    return 0, None
                if len(out) == 1:
                    return _to_int(out[0], 0), None
                return _to_int(out[0], 0), out[1]

            return _to_int(out, 0), None

        agent.decision_count = getattr(agent, "decision_count", 0) + 1
        obs = self.build_observation(agent)

        base_policy = self.policies.get(agent.side)
        wrapper = self.policy_wrappers.get(agent.side)

        # 1) Wrapper override
        if wrapper is not None and callable(wrapper):
            try:
                out = wrapper(obs, agent, self, base_policy)
            except TypeError:
                out = wrapper(obs, agent, self)

            action_id, param = _parse_out(out)
            act = (int(action_id), param)
            act = self._apply_action_delay(agent, act)
            self.apply_macro_action(agent, int(act[0]), act[1])
            return

        # 2) Neural-like policy interface
        if hasattr(base_policy, "act"):
            import torch
            device = torch.device("cpu")
            if hasattr(base_policy, "parameters"):
                try:
                    p = next(base_policy.parameters())
                    device = p.device
                except StopIteration:
                    pass

            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                out = base_policy.act(obs_tensor, agent=agent, game_field=self, deterministic=True)

            action_id, param = _parse_out(out)
            act = (int(action_id), param)
            act = self._apply_action_delay(agent, act)
            self.apply_macro_action(agent, int(act[0]), act[1])
            return

        # 3) Scripted Policy
        if isinstance(base_policy, Policy):
            action_id, param = base_policy.select_action(obs, agent, self)
            act = (int(action_id), param)
            act = self._apply_action_delay(agent, act)
            self.apply_macro_action(agent, int(act[0]), act[1])
            return

        # 4) Callable fallback
        if callable(base_policy):
            try:
                out = base_policy(obs, agent, self)
            except TypeError:
                out = base_policy(agent, self)

            action_id, param = _parse_out(out)
            act = (int(action_id), param)
            act = self._apply_action_delay(agent, act)
            self.apply_macro_action(agent, int(act[0]), act[1])
            return

        return

    # -------- mechanics: mines / suppression / flags --------

    def handle_mine_pickups(self, agent: Agent) -> None:
        if not agent.isEnabled():
            return

        ax, ay = self._agent_cell_pos(agent)

        for pickup in list(self.mine_pickups):
            if pickup.owner_side != agent.side:
                continue
            if ax == pickup.x and ay == pickup.y:
                prev = int(getattr(agent, "mine_charges", 0))

                if agent.mine_charges < self.max_mine_charges_per_agent:
                    needed = self.max_mine_charges_per_agent - agent.mine_charges
                    taken = min(needed, pickup.charges)
                    agent.mine_charges += taken
                    pickup.charges -= taken

                if hasattr(self.manager, "reward_mine_picked_up"):
                    try:
                        self.manager.reward_mine_picked_up(agent, prev_charges=prev)
                    except Exception:
                        pass

                if pickup.charges <= 0:
                    self.mine_pickups.remove(pickup)

    def apply_mine_damage(self, agent: Agent) -> None:
        if not agent.isEnabled():
            return

        ax, ay = self._agent_float_pos(agent)

        for mine in list(self.mines):
            if mine.owner_side == agent.side:
                continue

            dist = math.hypot(float(mine.x) - ax, float(mine.y) - ay)
            if dist <= float(self.mine_radius_cells):
                killer_agent = None

                if mine.owner_id is not None:
                    killer_agent = self._agent_by_id.get(str(mine.owner_id))

                if killer_agent is not None and getattr(killer_agent, "side", None) == "blue":
                    if hasattr(self.manager, "reward_enemy_killed"):
                        try:
                            self.manager.reward_enemy_killed(killer_agent=killer_agent, victim_agent=agent, cause="mine")
                        except Exception:
                            pass

                if mine.owner_side == "blue" and agent.side == "red":
                    if hasattr(self.manager, "record_mine_triggered_by_red"):
                        try:
                            self.manager.record_mine_triggered_by_red()
                        except Exception:
                            pass

                self._clear_pending_for_agent(agent)
                agent.disable_for_seconds(self.respawn_seconds)
                self.mines.remove(mine)

                # Reset boat state on death so it doesn't "slide" after respawn
                try:
                    setattr(agent, "speed_cps", 0.0)
                except Exception:
                    pass
                break

    def apply_suppression(self, agent: Agent, enemies: List[Agent], delta_time: float) -> None:
        if agent is None or (not agent.isEnabled()):
            return

        ax, ay = self._agent_float_pos(agent)

        close_enemies: List[Agent] = []
        rng = float(getattr(self, "suppression_range_cells", 0.0))
        for e in enemies:
            if e is None or (not e.isEnabled()):
                continue
            ex, ey = self._agent_float_pos(e)
            if math.hypot(ex - ax, ey - ay) <= rng:
                close_enemies.append(e)

        if len(close_enemies) >= 2:
            agent.suppressed_this_tick = True
            agent.suppression_timer = float(getattr(agent, "suppression_timer", 0.0)) + float(delta_time)

            if agent.suppression_timer >= 1.0:
                blue_suppressors = [e for e in close_enemies if getattr(e, "side", None) == "blue"]
                killer_agent = blue_suppressors[0] if blue_suppressors else close_enemies[0]

                mgr = getattr(self, "manager", None)
                if (
                    killer_agent is not None
                    and getattr(killer_agent, "side", None) == "blue"
                    and mgr is not None
                    and hasattr(mgr, "reward_enemy_killed")
                ):
                    try:
                        mgr.reward_enemy_killed(killer_agent=killer_agent, victim_agent=agent, cause="suppression")
                    except Exception:
                        pass

                self._clear_pending_for_agent(agent)

                agent.suppression_timer = 0.0
                agent.disable_for_seconds(float(getattr(self, "respawn_seconds", 0.0)))

                try:
                    setattr(agent, "speed_cps", 0.0)
                except Exception:
                    pass
        else:
            agent.suppressed_this_tick = False
            agent.suppression_timer = 0.0

    def apply_flag_rules(self, agent: Agent) -> None:
        if hasattr(self.manager, "try_pickup_enemy_flag") and self.manager.try_pickup_enemy_flag(agent):
            agent.setCarryingFlag(True)

        if agent.isCarryingFlag():
            if hasattr(self.manager, "try_score_if_carrying_and_home") and self.manager.try_score_if_carrying_and_home(agent):
                agent.setCarryingFlag(False, scored=True)
                self.announce(
                    "BLUE SCORES!" if agent.side == "blue" else "RED SCORES!",
                    (90, 170, 250) if agent.side == "blue" else (250, 120, 70),
                    2.0,
                )

    # -------- minimal UI hook --------

    def announce(self, text: str, color: Tuple[int, int, int] = (255, 255, 255), seconds: float = 2.0) -> None:
        self.banner_queue.append((str(text), color, float(seconds)))


__all__ = ["GameField", "MacroAction", "Mine", "MinePickup", "CNN_COLS", "CNN_ROWS", "NUM_CNN_CHANNELS", "BoatSimConfig"]
