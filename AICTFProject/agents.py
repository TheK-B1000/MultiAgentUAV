import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from collections import deque

# Radius used elsewhere (e.g., "team zone" checks)
TEAM_ZONE_RADIUS_CELLS = 3

# Diagonal movement cost (for grid movement with 8 directions)
DIAGONAL_COST = math.sqrt(2)


# Type aliases for clarity
Grid = List[List[int]]
Cell = Tuple[int, int]


@dataclass
class Agent:
    # --- Core spatial data ---
    x: int
    y: int
    side: str           # "blue" or "red"
    cols: int           # grid width
    rows: int           # grid height
    grid: Grid          # underlying grid (walls, free cells, etc.)

    # --- Identity / role ---
    is_miner: bool = False
    agent_id: int = 0   # 0 or 1 within team

    # --- Movement along a path ---
    # "cells per second" – macro actions or GameField set the path; Agent just walks it.
    move_rate_cps: float = 2.2
    path: deque[Cell] = field(default_factory=deque)
    move_accum: float = 0.0

    # Smooth position for rendering / interpolation
    _float_x: float = field(init=False)
    _float_y: float = field(init=False)

    # --- Status flags ---
    enabled: bool = True
    tag_cooldown: float = 0.0      # remaining time until respawn if disabled
    is_carrying_flag: bool = False

    # --- Mine capacity (if used by game logic) ---
    mine_charges: int = 0
    max_mine_charges: int = 2

    # --- Spawn & navigation helpers ---
    spawn_xy: Cell = (0, 0)
    waypoint: Optional[Cell] = None  # final target of current path

    # One-step event flags – should be consumed by higher-level logic
    _just_picked_up_flag: bool = False
    _just_scored: bool = False
    _just_tagged_enemy: bool = False

    def __post_init__(self) -> None:
        # Clamp initial position and spawn to map bounds
        self.x, self.y = self._clamp(self.x, self.y)
        self.spawn_xy = self._clamp(*self.spawn_xy)

        self.path = deque()
        self._float_x = float(self.x)
        self._float_y = float(self.y)

        # Unique identifier used by GameManager reward_events.
        # IMPORTANT: keep this in sync with GameManager.get_step_rewards()
        # so global rewards (win/draw/loss) are assigned correctly.
        self.unique_id = f"{self.side}_{self.agent_id}"

        # Ensure event flags are clean
        self._just_picked_up_flag = False
        self._just_scored = False
        self._just_tagged_enemy = False
        self.decision_count: int = 0

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def _clamp(self, col: int, row: int) -> Cell:
        """Clamp a cell into the arena bounds."""
        c = max(0, min(self.cols - 1, col))
        r = max(0, min(self.rows - 1, row))
        return (c, r)

    @property
    def float_pos(self) -> Tuple[float, float]:
        """Smooth position, useful for visualization."""
        return self._float_x, self._float_y

    def get_position(self) -> Cell:
        """Discrete grid position."""
        return (self.x, self.y)

    def getSide(self) -> str:
        return self.side

    # ------------------------------------------------------------------
       # Status checks
    # ------------------------------------------------------------------
    def isEnabled(self) -> bool:
        """True if the agent is active in the game."""
        return self.enabled

    def isTagged(self) -> bool:
        """
        True if the agent is currently disabled and waiting for respawn.
        """
        return (not self.enabled) and self.tag_cooldown > 0.0

    def isCarryingFlag(self) -> bool:
        return self.is_carrying_flag

    # ------------------------------------------------------------------
    # Flag handling + event flags
    # ------------------------------------------------------------------
    def setCarryingFlag(self, value: bool) -> None:
        if not self.is_carrying_flag and value:
            self._just_picked_up_flag = True
        elif self.is_carrying_flag and not value:
            self._just_scored = True
        self.is_carrying_flag = value

    def consume_just_picked_up_flag(self) -> bool:
        """Return True once when the agent has just picked up the flag."""
        v = self._just_picked_up_flag
        self._just_picked_up_flag = False
        return v

    def consume_just_scored(self) -> bool:
        """Return True once when the agent has just delivered the flag."""
        v = self._just_scored
        self._just_scored = False
        return v

    def consume_just_tagged_enemy(self) -> bool:
        """
        Return True once when this agent was just disabled.
        (Name kept for backwards compatibility; semantics = 'just got tagged').
        """
        v = self._just_tagged_enemy
        self._just_tagged_enemy = False
        return v

    # ------------------------------------------------------------------
    # Path handling
    # ------------------------------------------------------------------
    def setPath(self, path: List[Cell]) -> None:
        if not path:
            self.clearPath()
            return

        clamped = [self._clamp(*c) for c in path]
        self.path = deque(clamped)
        self.waypoint = clamped[-1] if clamped else None
        # Reset movement accumulator for fresh path
        self.move_accum = 0.0

    def clearPath(self) -> None:
        """Stop the agent from moving anywhere."""
        self.path.clear()
        self.waypoint = None

    # ------------------------------------------------------------------
    # Disable / respawn logic
    # ------------------------------------------------------------------
    def disable_for_seconds(self, seconds: float) -> None:
        if self.enabled:
            self.was_just_disabled = True
            self._just_tagged_enemy = True   # backward-compatible event flag
            self.enabled = False
            self.tag_cooldown = max(self.tag_cooldown, seconds)

            # Drop the flag if we were carrying it
            if self.is_carrying_flag:
                self.is_carrying_flag = False

            # Stop moving
            self.clearPath()
            self.move_accum = 0.0

    def respawn(self) -> None:
        """
        Reset agent to its spawn point and clear transient state.
        Called automatically when tag_cooldown reaches 0.
        """
        self.x, self.y = self.spawn_xy
        self._float_x = float(self.x)
        self._float_y = float(self.y)

        self.enabled = True
        self.tag_cooldown = 0.0
        self.was_just_disabled = False

        self.clearPath()
        self.move_accum = 0.0

        # Clear one-step flags
        self._just_picked_up_flag = False
        self._just_scored = False
        self._just_tagged_enemy = False

    # ------------------------------------------------------------------
    # Per-timestep update
    # ------------------------------------------------------------------
    def update(self, dt: float) -> None:
        if dt <= 0.0:
            return

        # Handle disabled state / respawn timer
        if not self.enabled:
            if self.tag_cooldown > 0.0:
                self.tag_cooldown = max(0.0, self.tag_cooldown - dt)
                if self.tag_cooldown <= 0.0:
                    self.respawn()
            return

        # If enabled but no path, nothing to do
        if not self.path:
            # Keep float position aligned with discrete cell
            self._float_x = float(self.x)
            self._float_y = float(self.y)
            return

        # Accumulate movement in "cells" along current path
        self.move_accum += self.move_rate_cps * dt

        # Consume as many whole cells as we can
        while self.move_accum >= 1.0 and self.path:
            next_cell = self.path[0]
            dx = next_cell[0] - self.x
            dy = next_cell[1] - self.y
            step_cost = DIAGONAL_COST if dx != 0 and dy != 0 else 1.0

            if self.move_accum >= step_cost:
                # Move into the next cell
                self.x, self.y = next_cell
                self._float_x = float(self.x)
                self._float_y = float(self.y)

                self.path.popleft()
                self.move_accum -= step_cost
                self.waypoint = self.path[-1] if self.path else None
            else:
                break

        # Interpolate within the next cell if we're mid-step
        if self.path:
            next_cell = self.path[0]
            dx = next_cell[0] - self.x
            dy = next_cell[1] - self.y
            step_cost = DIAGONAL_COST if dx != 0 and dy != 0 else 1.0

            progress = min(self.move_accum / step_cost, 1.0)
            self._float_x = self.x + dx * progress
            self._float_y = self.y + dy * progress
        else:
            # End of path; snap float position to cell
            self._float_x = float(self.x)
            self._float_y = float(self.y)
