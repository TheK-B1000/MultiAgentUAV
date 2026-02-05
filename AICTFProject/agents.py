from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, ClassVar, Deque, List, Optional, Tuple

TEAM_ZONE_RADIUS_CELLS: int = 3
DIAGONAL_COST: float = math.sqrt(2.0)

Grid = List[List[int]]
Cell = Tuple[int, int]
FloatPos = Tuple[float, float]


@dataclass
class Agent:
    # --- Required by GameField.spawn_agents() ---
    x: int
    y: int
    side: str
    cols: int
    rows: int
    grid: Grid

    # Optional role metadata
    is_miner: bool = False
    agent_id: int = 0

    # Optional backrefs (set by GameField)
    game_manager: Optional[Any] = field(default=None, repr=False, compare=False)
    game_field: Optional[Any] = field(default=None, repr=False, compare=False)

    # Movement/pathing
    move_rate_cps: float = 2.2
    path: Deque[Cell] = field(default_factory=deque, repr=False)
    move_accum: float = 0.0
    waypoint: Optional[Cell] = None

    # Cache: avoids replan jitter when target unchanged
    current_goal: Optional[Cell] = None

    # Continuous position
    _float_x: float = field(init=False, repr=False)
    _float_y: float = field(init=False, repr=False)

    # Sub-pixel awareness additions
    _vel_x: float = field(init=False, repr=False)
    _vel_y: float = field(init=False, repr=False)

    # Status
    enabled: bool = True
    tag_cooldown: float = 0.0

    # Objective payload
    is_carrying_flag: bool = False

    # Mines
    mine_charges: int = 0
    max_mine_charges: int = 2

    # Spawn
    spawn_xy: Cell = (0, 0)

    # One-step flags
    _just_picked_up_flag: bool = field(default=False, init=False, repr=False)
    _just_scored: bool = field(default=False, init=False, repr=False)
    _just_tagged_enemy: bool = field(default=False, init=False, repr=False)

    was_just_disabled: bool = field(default=False, init=False)
    disabled_this_tick: bool = field(default=False, init=False)

    suppression_timer: float = 0.0
    suppressed_last_tick: bool = False
    suppressed_this_tick: bool = False

    decision_count: int = 0
    last_macro_action: Optional[Any] = field(default=None, repr=False, compare=False)
    last_macro_action_idx: int = 0

    instance_id: int = field(init=False)
    _NEXT_INSTANCE_ID: ClassVar[int] = 0

    def __post_init__(self) -> None:
        self.side = str(self.side).lower().strip()
        self.x, self.y = self._clamp_cell(self.x, self.y)
        self.spawn_xy = self._clamp_cell(*self.spawn_xy)

        if not isinstance(self.path, deque):
            self.path = deque(self.path)

        self._float_x = float(self.x)
        self._float_y = float(self.y)

        # init velocity state
        self._vel_x = 0.0
        self._vel_y = 0.0

        self.instance_id = Agent._NEXT_INSTANCE_ID
        Agent._NEXT_INSTANCE_ID += 1

        self.slot_id: str = f"{self.side}_{self.agent_id}"
        self.unique_id: str = f"{self.side}_{self.agent_id}_{self.instance_id}"

    # ----------------------------------------------------------
    # Helpers / properties
    # ----------------------------------------------------------
    def _clamp_cell(self, col: int, row: int) -> Cell:
        c = max(0, min(self.cols - 1, int(col)))
        r = max(0, min(self.rows - 1, int(row)))
        return (c, r)

    @property
    def cell_pos(self) -> Cell:
        return (self.x, self.y)

    @property
    def float_pos(self) -> FloatPos:
        return (self._float_x, self._float_y)

    @property
    def vel(self) -> Tuple[float, float]:
        return (float(self._vel_x), float(self._vel_y))

    @property
    def frac_in_cell(self) -> Tuple[float, float]:
        # offset relative to integer cell anchor (x,y)
        return (float(self._float_x - self.x), float(self._float_y - self.y))

    def get_position(self) -> Cell:
        return (self.x, self.y)

    # Legacy-friendly aliases
    def getSide(self) -> str:
        return self.side

    def isEnabled(self) -> bool:
        return bool(self.enabled)

    def isTagged(self) -> bool:
        return (not self.enabled) and (self.tag_cooldown > 0.0)

    def isCarryingFlag(self) -> bool:
        return bool(self.is_carrying_flag)

    def attach_game_manager(self, gm: Any) -> None:
        self.game_manager = gm

    # ----------------------------------------------------------
    # Flag state + one-step flags
    # ----------------------------------------------------------
    def setCarryingFlag(self, value: bool, *, scored: Optional[bool] = None) -> None:
        if value and not self.is_carrying_flag:
            self._just_picked_up_flag = True

        if self.is_carrying_flag and not value:
            if scored is True:
                self._just_scored = True

        self.is_carrying_flag = bool(value)

    def consume_just_picked_up_flag(self) -> bool:
        v = self._just_picked_up_flag
        self._just_picked_up_flag = False
        return v

    def consume_just_scored(self) -> bool:
        v = self._just_scored
        self._just_scored = False
        return v

    def consume_just_tagged_enemy(self) -> bool:
        v = self._just_tagged_enemy
        self._just_tagged_enemy = False
        return v

    def consume_disabled_this_tick(self) -> bool:
        v = self.disabled_this_tick
        self.disabled_this_tick = False
        return v

    # ----------------------------------------------------------
    # Path handling (UPDATED: do NOT reset move_accum on new path)
    # ----------------------------------------------------------
    def setPath(self, path: Optional[List[Cell]]) -> None:
        if not path:
            self.clearPath()
            self.move_accum = 0.0
            self.current_goal = None
            return

        clamped = [self._clamp_cell(c[0], c[1]) for c in path]
        self.path = deque(clamped)
        self.waypoint = clamped[-1] if clamped else None
        # IMPORTANT: keep move_accum to avoid micro-hesitations

    def clearPath(self) -> None:
        self.path.clear()
        self.waypoint = None

    # ----------------------------------------------------------
    # Disable / respawn
    # ----------------------------------------------------------
    def disable_for_seconds(self, seconds: float) -> None:
        s = max(0.0, float(seconds))
        if s <= 0.0:
            return

        if not self.enabled:
            self.tag_cooldown = max(self.tag_cooldown, s)
            return

        self.was_just_disabled = True
        self.disabled_this_tick = True

        self.enabled = False
        self.tag_cooldown = max(self.tag_cooldown, s)

        self.setCarryingFlag(False, scored=False)

        self.clearPath()
        self.move_accum = 0.0
        self.current_goal = None

        self._float_x = float(self.x)
        self._float_y = float(self.y)
        self._vel_x = 0.0
        self._vel_y = 0.0

    def respawn(self) -> None:
        gm = self.game_manager
        if gm is not None and hasattr(gm, "clear_flag_carrier_if_agent"):
            gm.clear_flag_carrier_if_agent(self)

        self.x, self.y = self.spawn_xy
        self._float_x = float(self.x)
        self._float_y = float(self.y)

        self.enabled = True
        self.tag_cooldown = 0.0
        self.was_just_disabled = False
        self.disabled_this_tick = False

        self.clearPath()
        self.move_accum = 0.0
        self.current_goal = None

        self._just_picked_up_flag = False
        self._just_scored = False
        self._just_tagged_enemy = False
        self.is_carrying_flag = False

        self.suppression_timer = 0.0
        self.suppressed_last_tick = False
        self.suppressed_this_tick = False

        self._vel_x = 0.0
        self._vel_y = 0.0

    # ----------------------------------------------------------
    # Per-timestep update (UPDATED: velocity estimation)
    # ----------------------------------------------------------
    def update(self, dt: float) -> None:
        if dt <= 0.0:
            return

        prev_float_pos = self.float_pos
        prev_fx, prev_fy = self._float_x, self._float_y

        # Disabled: tick respawn timer only
        if not self.enabled:
            if self.tag_cooldown > 0.0:
                self.tag_cooldown = max(0.0, self.tag_cooldown - float(dt))
                if self.tag_cooldown <= 0.0:
                    self.respawn()
            return

        self.suppressed_last_tick = self.suppressed_this_tick
        self.suppressed_this_tick = False

        if not self.path:
            self._float_x = float(self.x)
            self._float_y = float(self.y)
            self._vel_x = 0.0
            self._vel_y = 0.0
            return

        # Speed multiplier from GameManager (red_speed_mult / blue_speed_mult from dynamics_config)
        speed_mult = 1.0
        gm = getattr(self, "game_manager", None)
        if gm is not None and hasattr(gm, "get_agent_speed_multiplier"):
            try:
                speed_mult = float(gm.get_agent_speed_multiplier(self))
            except Exception:
                speed_mult = 1.0
        if not math.isfinite(speed_mult) or speed_mult <= 0.0:
            speed_mult = 1.0

        # Continuous movement budget in "cells"
        remaining = float(self.move_rate_cps) * speed_mult * float(dt)

        # Move along path continuously, possibly consuming multiple segments
        while remaining > 1e-8 and self.path:
            nx, ny = self.path[0]

            # Target is center of next cell (in cell coordinates)
            tx = float(nx)
            ty = float(ny)

            dx = tx - self._float_x
            dy = ty - self._float_y
            dist = math.hypot(dx, dy)

            # If we're basically at the next cell already, snap to it and pop
            if dist < 1e-6:
                self._float_x = tx
                self._float_y = ty
                self.x, self.y = nx, ny
                self.path.popleft()
                self.waypoint = self.path[-1] if self.path else None
                continue

            # Move toward (tx,ty) by at most `remaining`
            step = min(remaining, dist)
            ux = dx / dist
            uy = dy / dist

            self._float_x += ux * step
            self._float_y += uy * step
            remaining -= step

            # If we reached the next cell, commit integer cell + pop
            if (dist - step) <= 1e-6:
                self._float_x = tx
                self._float_y = ty
                self.x, self.y = nx, ny
                self.path.popleft()
                self.waypoint = self.path[-1] if self.path else None

        # Velocity estimate (cells/sec)
        self._vel_x = (self._float_x - prev_fx) / float(dt)
        self._vel_y = (self._float_y - prev_fy) / float(dt)

        gm = self.game_manager
        if gm is not None and hasattr(gm, "reward_potential_shaping"):
            gm.reward_potential_shaping(self, prev_float_pos, self.float_pos)


__all__ = ["Agent", "TEAM_ZONE_RADIUS_CELLS", "DIAGONAL_COST"]
