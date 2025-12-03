# agents.py
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Deque
from collections import deque

# Constants
TEAM_ZONE_RADIUS_CELLS = 3
DIAGONAL_COST = math.sqrt(2)

# Type aliases
Grid = List[List[int]]
Cell = Tuple[int, int]                    # Discrete grid cell (col, row)
FloatPos = Tuple[float, float]            # Continuous world position
Path = Deque[FloatPos]                    # List of float waypoints


@dataclass
class Agent:
    # --- Construction arguments ---
    x: float                                 # Initial spawn (now float!)
    y: float
    side: str                                # "blue" or "red"
    cols: int
    rows: int
    grid: Grid

    # --- Identity / role ---
    is_miner: bool = False
    agent_id: int = 0                        # 0 or 1 within team

    # --- Movement ---
    move_rate_cps: float = 2.2               # cells per second (world units/sec)
    path: Path = field(default_factory=deque)  # Float waypoints: [(x, y), ...]
    move_accum: float = 0.0                  # Sub-cell movement accumulator

    # --- Position (continuous is primary now) ---
    _float_x: float = field(init=False)
    _float_y: float = field(init=False)

    # --- Status ---
    enabled: bool = True
    tag_cooldown: float = 0.0
    is_carrying_flag: bool = False
    was_just_disabled: bool = False

    # --- Mines ---
    mine_charges: int = 0
    max_mine_charges: int = 2

    # --- Spawn & navigation ---
    spawn_xy: FloatPos = (0.0, 0.0)
    waypoint: Optional[FloatPos] = None      # Final target (for UI)

    # --- RL / Reward one-shot events ---
    _just_picked_up_flag: bool = False
    _just_scored: bool = False
    _just_tagged_enemy: bool = False

    # --- Unique ID ---
    unique_id: str = field(init=False)

    def __post_init__(self) -> None:
        # Initialize continuous position
        self._float_x = float(self.x)
        self._float_y = float(self.y)
        self.spawn_xy = (self._float_x, self._float_y)

        # Unique ID for RL/reward tracking
        spawn_str = f"{self._float_x:.1f}_{self._float_y:.1f}"
        self.unique_id = f"{self.side}_{self.agent_id}_{spawn_str}"

        # Ensure clean event flags
        self._just_picked_up_flag = False
        self._just_scored = False
        self._just_tagged_enemy = False

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def float_pos(self) -> FloatPos:
        return self._float_x, self._float_y

    @property
    def cell_pos(self) -> Cell:
        """Current discrete grid cell (for pathfinding, collision, etc.)"""
        return int(self._float_x), int(self._float_y)

    def getSide(self) -> str:
        return self.side

    def isEnabled(self) -> bool:
        return self.enabled

    def isTagged(self) -> bool:
        return not self.enabled and self.tag_cooldown > 0.0

    def isCarryingFlag(self) -> bool:
        return self.is_carrying_flag

    # ------------------------------------------------------------------ #
    # Flag event handling
    # ------------------------------------------------------------------ #
    def setCarryingFlag(self, carrying: bool) -> None:
        if not self.is_carrying_flag and carrying:
            self._just_picked_up_flag = True
        elif self.is_carrying_flag and not carrying:
            self._just_scored = True
        self.is_carrying_flag = carrying

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

    # ------------------------------------------------------------------ #
    # Path management (now uses float waypoints)
    # ------------------------------------------------------------------ #
    def setPath(self, waypoints: List[FloatPos]) -> None:
        """Set a new path of float (x, y) waypoints (typically cell centers + 0.5)"""
        if not waypoints:
            self.clearPath()
            return

        # Clamp all waypoints to bounds
        clamped = [
            (
                max(0.0, min(self.cols - 1.999, x)),
                max(0.0, min(self.rows - 1.999, y))
            )
            for x, y in waypoints
        ]

        self.path = deque(clamped)
        self.waypoint = clamped[-1] if clamped else None
        self.move_accum = 0.0

    def clearPath(self) -> None:
        self.path.clear()
        self.waypoint = None

    # ------------------------------------------------------------------ #
    # Disable / respawn
    # ------------------------------------------------------------------ #
    def disable_for_seconds(self, seconds: float) -> None:
        if self.enabled:
            self.was_just_disabled = True
            self._just_tagged_enemy = True
            self.enabled = False
            self.tag_cooldown = max(self.tag_cooldown, seconds)

            # Drop flag
            if self.is_carrying_flag:
                self.is_carrying_flag = False

            # Stop moving
            self.clearPath()
            self.move_accum = 0.0

    def respawn(self) -> None:
        self._float_x, self._float_y = self.spawn_xy
        self.enabled = True
        self.tag_cooldown = 0.0
        self.was_just_disabled = False
        self.clearPath()
        self.move_accum = 0.0

        # Clear one-shot flags
        self._just_picked_up_flag = False
        self._just_scored = False
        self._just_tagged_enemy = False

    # ------------------------------------------------------------------ #
    # Main per-frame update
    # ------------------------------------------------------------------ #
    def update(self, dt: float) -> None:
        if dt <= 0.0:
            return

        # Handle respawn timer
        if not self.enabled:
            self.tag_cooldown = max(0.0, self.tag_cooldown - dt)
            if self.tag_cooldown <= 0.0:
                self.respawn()
            return

        # No path → stay still
        if not self.path:
            cx, cy = self.cell_pos
            self._float_x = cx + 0.5
            self._float_y = cy + 0.5
            return

        # Accumulate movement
        self.move_accum += self.move_rate_cps * dt

        # Follow path
        while self.move_accum >= 1.0 and self.path:
            next_wp = self.path[0]
            dx = next_wp[0] - self._float_x
            dy = next_wp[1] - self._float_y
            dist = math.hypot(dx, dy)

            if dist <= 0.01:  # Already at waypoint
                self.path.popleft()
                self.move_accum -= 1.0
                if self.path:
                    self.waypoint = self.path[-1]
                else:
                    self.waypoint = None
                continue

            # Determine cost to reach next waypoint
            step_cost = dist  # True Euclidean distance

            if self.move_accum >= step_cost:
                # Reach and pass waypoint
                self._float_x, self._float_y = next_wp
                self.path.popleft()
                self.move_accum -= step_cost
                if self.path:
                    self.waypoint = self.path[-1]
                else:
                    self.waypoint = None
            else:
                # Move partially toward waypoint
                progress = self.move_accum / step_cost
                self._float_x += dx * progress
                self._float_y += dy * progress
                self.move_accum = 0.0
                break

        # Final snap if path ended
        if not self.path:
            if self.waypoint:
                self._float_x, self._float_y = self.waypoint
            self.waypoint = None