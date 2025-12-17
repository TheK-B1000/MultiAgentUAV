# =========================
# agents.py (CLEAN, RL-SAFE, MATCHES GameField.spawn_agents)
# =========================

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Deque, ClassVar
from collections import deque

# Public constants used by env/viewer
TEAM_ZONE_RADIUS_CELLS: int = 3
DIAGONAL_COST: float = math.sqrt(2.0)

Grid = List[List[int]]
Cell = Tuple[int, int]


@dataclass
class Agent:
    # --- Required by GameField.spawn_agents() ---
    x: int
    y: int
    side: str                  # "blue" or "red"
    cols: int                  # grid width
    rows: int                  # grid height
    grid: Grid                 # environment grid reference (walls/free)

    # Optional role metadata (GameField may pass this)
    is_miner: bool = False
    agent_id: int = 0          # 0..N-1 within side/team

    # Movement/pathing (env sets path, Agent executes it)
    move_rate_cps: float = 2.2
    path: Deque[Cell] = field(default_factory=deque, repr=False)
    move_accum: float = 0.0
    waypoint: Optional[Cell] = None

    # Continuous pos (rendering + continuous-distance logic)
    _float_x: float = field(init=False, repr=False)
    _float_y: float = field(init=False, repr=False)

    # Status
    enabled: bool = True
    tag_cooldown: float = 0.0
    is_carrying_flag: bool = False

    # Mines (if used)
    mine_charges: int = 0
    max_mine_charges: int = 2

    # Spawn
    spawn_xy: Cell = (0, 0)

    # One-step event flags (consumed by higher-level logic)
    _just_picked_up_flag: bool = field(default=False, init=False, repr=False)
    _just_scored: bool = field(default=False, init=False, repr=False)
    _just_tagged_enemy: bool = field(default=False, init=False, repr=False)
    was_just_disabled: bool = field(default=False, init=False)
    disabled_this_tick: bool = field(default=False, init=False)

    # Optional backref to GameManager
    game_manager: Optional[Any] = field(default=None, repr=False, compare=False)

    # Stable unique identity (important for reward routing/logging)
    instance_id: int = field(init=False)
    _NEXT_INSTANCE_ID: ClassVar[int] = 0

    # Soft suppression flags (if your GameField uses them)
    suppression_timer: float = 0.0
    suppressed_last_tick: bool = False
    suppressed_this_tick: bool = False

    def __post_init__(self) -> None:
        self.x, self.y = self._clamp(self.x, self.y)
        self.spawn_xy = self._clamp(*self.spawn_xy)

        if not isinstance(self.path, deque):
            self.path = deque(self.path)

        self._float_x = float(self.x)
        self._float_y = float(self.y)

        self.instance_id = Agent._NEXT_INSTANCE_ID
        Agent._NEXT_INSTANCE_ID += 1

        # Human-readable slot id + truly-unique id for buffers/rewards
        self.slot_id: str = f"{self.side}_{self.agent_id}"
        self.unique_id: str = f"{self.side}_{self.agent_id}_{self.instance_id}"

        # Optional counter some obs builders like to use
        self.decision_count: int = 0

    # ----------------------------------------------------------
    # Core helpers
    # ----------------------------------------------------------
    def _clamp(self, col: int, row: int) -> Cell:
        c = max(0, min(self.cols - 1, int(col)))
        r = max(0, min(self.rows - 1, int(row)))
        return (c, r)

    @property
    def float_pos(self) -> Tuple[float, float]:
        return (self._float_x, self._float_y)

    def get_position(self) -> Cell:
        return (self.x, self.y)

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
            # Only mark scored when caller says so
            if scored is True:
                self._just_scored = True

        self.is_carrying_flag = bool(value)

    def mark_scored(self) -> None:
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

    # ----------------------------------------------------------
    # Path handling
    # ----------------------------------------------------------
    def setPath(self, path: Optional[List[Cell]]) -> None:
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

    # ----------------------------------------------------------
    # Disable / respawn
    # ----------------------------------------------------------
    def disable_for_seconds(self, seconds: float) -> None:
        if not self.enabled:
            return

        self.was_just_disabled = True
        self.disabled_this_tick = True
        self.enabled = False
        self.tag_cooldown = max(self.tag_cooldown, float(seconds))

        # Let GM clean up carriers / rewards
        if self.game_manager is not None:
            self.game_manager.handle_agent_death(self)

        # Force local carry off (do NOT mark scored)
        self.setCarryingFlag(False, scored=False)

        self.clearPath()
        self.move_accum = 0.0

    def respawn(self) -> None:
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

        self._just_picked_up_flag = False
        self._just_scored = False
        self._just_tagged_enemy = False
        self.is_carrying_flag = False

        self.suppression_timer = 0.0
        self.suppressed_last_tick = False
        self.suppressed_this_tick = False

    # ----------------------------------------------------------
    # Per-timestep update (movement + executed PBRS hook)
    # ----------------------------------------------------------
    def update(self, dt: float) -> None:
        if dt <= 0.0:
            return

        prev_float_pos = self.float_pos

        # Disabled: tick respawn timer only
        if not self.enabled:
            if self.tag_cooldown > 0.0:
                self.tag_cooldown = max(0.0, self.tag_cooldown - dt)
                if self.tag_cooldown <= 0.0:
                    self.respawn()
            return

        # Suppression flags are controlled by GameField per tick
        self.suppressed_last_tick = self.suppressed_this_tick
        self.suppressed_this_tick = False

        if not self.path:
            self._float_x = float(self.x)
            self._float_y = float(self.y)
            return

        self.move_accum += self.move_rate_cps * dt

        # Consume whole-cell moves
        while self.path:
            nx, ny = self.path[0]
            dx = nx - self.x
            dy = ny - self.y
            step_cost = DIAGONAL_COST if (dx != 0 and dy != 0) else 1.0

            if self.move_accum >= step_cost:
                self.x, self.y = nx, ny
                self._float_x = float(self.x)
                self._float_y = float(self.y)
                self.path.popleft()
                self.move_accum -= step_cost
                self.waypoint = self.path[-1] if self.path else None
            else:
                break

        # Fractional interpolation
        if self.path:
            nx, ny = self.path[0]
            dx = nx - self.x
            dy = ny - self.y
            step_cost = DIAGONAL_COST if (dx != 0 and dy != 0) else 1.0
            prog = min(self.move_accum / step_cost, 1.0)
            self._float_x = self.x + dx * prog
            self._float_y = self.y + dy * prog
        else:
            self._float_x = float(self.x)
            self._float_y = float(self.y)

        # Executed PBRS hook (if GM provides it)
        if self.game_manager is not None and hasattr(self.game_manager, "reward_potential_shaping"):
            self.game_manager.reward_potential_shaping(self, prev_float_pos, self.float_pos)
