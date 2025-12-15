import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Deque, ClassVar
from collections import deque

# Radius used elsewhere (e.g., "team zone" checks and flag drawing)
TEAM_ZONE_RADIUS_CELLS = 3

# Diagonal movement cost (for grid movement with 8 directions)
DIAGONAL_COST = math.sqrt(2)

# Type aliases for clarity
Grid = List[List[int]]
Cell = Tuple[int, int]


@dataclass
class Agent:
    # --- Core spatial data (grid coordinates) ---
    x: int
    y: int
    side: str  # "blue" or "red"
    cols: int  # grid width
    rows: int  # grid height
    grid: Grid  # underlying grid (walls, free cells, etc.)

    # --- Identity / role ---
    is_miner: bool = False
    agent_id: int = 0  # 0 or 1 within team

    # --- Movement along a path ---
    # "cells per second" – GameField sets the path; Agent just walks it.
    move_rate_cps: float = 2.2
    path: Deque[Cell] = field(default_factory=deque)
    move_accum: float = 0.0

    # Smooth position for rendering / interpolation
    _float_x: float = field(init=False)
    _float_y: float = field(init=False)

    # --- Status flags ---
    enabled: bool = True
    tag_cooldown: float = 0.0  # remaining time until respawn if disabled
    is_carrying_flag: bool = False

    # --- Mine capacity (if used by game logic) ---
    mine_charges: int = 0
    max_mine_charges: int = 2

    # --- Spawn & navigation helpers ---
    spawn_xy: Cell = (0, 0)
    waypoint: Optional[Cell] = None  # final target of current path

    # One-step event flags – can be consumed by higher-level logic
    _just_picked_up_flag: bool = False
    _just_scored: bool = False
    _just_tagged_enemy: bool = False

    # Reference to GameManager (optional, injected by env or setup)
    game_manager: Optional[Any] = field(default=None, repr=False)

    # --- Instance identity (unique per Agent object) ---
    instance_id: int = field(init=False)

    # Class-level counter for instance_id
    _NEXT_INSTANCE_ID: ClassVar[int] = 0

    # ⚙️ SOFT SUPPRESSION STATE (FIX 2/3)
    suppression_timer: float = 0.0
    suppressed_last_tick: bool = False
    suppressed_this_tick: bool = False

    def __post_init__(self) -> None:
        # Clamp initial position and spawn to map bounds
        self.x, self.y = self._clamp(self.x, self.y)
        self.spawn_xy = self._clamp(*self.spawn_xy)

        # Ensure path is a deque
        if not isinstance(self.path, deque):
            self.path = deque(self.path)

        self._float_x = float(self.x)
        self._float_y = float(self.y)

        # Stable unique instance id (avoids collisions across episodes/runs)
        self.instance_id = Agent._NEXT_INSTANCE_ID
        Agent._NEXT_INSTANCE_ID += 1

        # A stable "slot id" (human-readable) and a truly-unique id (for buffers/events)
        self.slot_id = f"{self.side}_{self.agent_id}"
        self.unique_id = f"{self.side}_{self.agent_id}_{self.instance_id}"

        # Per-episode decision counter used by GameField.build_observation
        self.decision_count: int = 0

        # Ensure event flags are clean
        self._just_picked_up_flag = False
        self._just_scored = False
        self._just_tagged_enemy = False

        # Track disable events
        self.was_just_disabled: bool = False
        self.disabled_this_tick: bool = False

        # Init suppression state
        self.suppression_timer = 0.0
        self.suppressed_last_tick = False
        self.suppressed_this_tick = False

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def _clamp(self, col: int, row: int) -> Cell:
        c = max(0, min(self.cols - 1, int(col)))
        r = max(0, min(self.rows - 1, int(row)))
        return (c, r)

    @property
    def float_pos(self) -> Tuple[float, float]:
        """Continuous position used for rendering / continuous-distance logic."""
        return self._float_x, self._float_y

    def get_position(self) -> Cell:
        return (self.x, self.y)

    def getSide(self) -> str:
        return self.side

    # Status checks
    def isEnabled(self) -> bool:
        return self.enabled

    def isTagged(self) -> bool:
        return (not self.enabled) and self.tag_cooldown > 0.0

    def isCarryingFlag(self) -> bool:
        return self.is_carrying_flag

    # Attach / wiring to GameManager
    def attach_game_manager(self, gm: Any) -> None:
        self.game_manager = gm

    # ------------------------------------------------------------------
    # Flag handling + event flags (No Changes)
    # ------------------------------------------------------------------
    def setCarryingFlag(self, value: bool, *, scored: Optional[bool] = None) -> None:
        """
        Set local flag-carry state.
        ...
        """
        if value and not self.is_carrying_flag:
            self._just_picked_up_flag = True

        if self.is_carrying_flag and not value:
            # Only mark "scored" if caller says so, OR preserve legacy behavior if None.
            if scored is None or scored is True:
                self._just_scored = True

        self.is_carrying_flag = value

    def mark_scored(self) -> None:
        """Explicit scoring event, if you prefer to drive scoring from GameManager/GameField."""
        self._just_scored = True

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

    # ------------------------------------------------------------------
    # Path handling (No Changes)
    # ------------------------------------------------------------------
    def setPath(self, path: Optional[List[Cell]]) -> None:
        """
        Assign a new path.
        ...
        """
        if not path:
            self.clearPath()
            self.move_accum = 0.0
            return

        clamped = [self._clamp(*c) for c in path]
        self.path = deque(clamped)
        self.waypoint = clamped[-1] if clamped else None
        self.move_accum = 0.0

    def clearPath(self) -> None:
        self.path.clear()
        self.waypoint = None

    # ------------------------------------------------------------------
    # Disable / respawn logic (No Changes)
    # ------------------------------------------------------------------
    def disable_for_seconds(self, seconds: float) -> None:
        """
        Disable/tag this agent for a given duration.
        ...
        """
        if not self.enabled:
            return

        self.was_just_disabled = True
        self.disabled_this_tick = True
        self.enabled = False
        self.tag_cooldown = max(self.tag_cooldown, float(seconds))

        # Let GameManager handle dropping the flag, rewards, etc.
        if self.game_manager is not None:
            self.game_manager.handle_agent_death(self)

        # Local view: we are no longer carrying a flag
        # ✅ don't accidentally mark this as a "score"
        self.setCarryingFlag(False, scored=False)

        # Stop moving
        self.clearPath()
        self.move_accum = 0.0

    def respawn(self) -> None:
        # Extra safety: make absolutely sure GameManager does not
        # still think this agent is a flag carrier.
        if self.game_manager is not None:
            self.game_manager.clear_flag_carrier_if_agent(self)

        self.x, self.y = self.spawn_xy
        self._float_x = float(self.x)
        self._float_y = float(self.y)

        self.enabled = True
        self.tag_cooldown = 0.0
        self.was_just_disabled = False
        self.disabled_this_tick = False

        self.clearPath()
        self.move_accum = 0.0

        # Clear one-step flags
        self._just_picked_up_flag = False
        self._just_scored = False
        self._just_tagged_enemy = False

        # On respawn we definitely are not carrying a flag
        self.is_carrying_flag = False

    # ------------------------------------------------------------------
    # Per-timestep update
    # ------------------------------------------------------------------
    def update(self, dt: float) -> None:
        if dt <= 0.0:
            return

        # ⚙️ Capture position before movement for execution-based shaping (FIX 2/2)
        prev_float_pos = self.float_pos

        # Handle disabled state / respawn timer
        if not self.enabled:
            if self.tag_cooldown > 0.0:
                self.tag_cooldown = max(0.0, self.tag_cooldown - dt)
                if self.tag_cooldown <= 0.0:
                    self.respawn()
            return

        # Reset suppressed_this_tick flag for GameField to set next tick
        self.suppressed_last_tick = self.suppressed_this_tick
        self.suppressed_this_tick = False

        # If enabled but no path, nothing to do
        if not self.path:
            self._float_x = float(self.x)
            self._float_y = float(self.y)
            return

        # Accumulate movement in "cells" along current path
        self.move_accum += self.move_rate_cps * dt

        # Consume as many whole cells as we can
        while self.path:
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
            self._float_x = float(self.x)
            self._float_y = float(self.y)

        # ⚙️ Call shaping with executed movement (FIX 2/2)
        if self.game_manager is not None:
            self.game_manager.reward_potential_shaping(self, prev_float_pos, self.float_pos)