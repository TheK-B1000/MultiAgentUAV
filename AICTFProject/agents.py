# =========================
# agents.py (REFACTORED, MARL-READY, RL-SAFE, MATCHES GameField.spawn_agents)
# =========================

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, ClassVar, Deque, List, Optional, Tuple

# Public constants used by env/viewer
TEAM_ZONE_RADIUS_CELLS: int = 3
DIAGONAL_COST: float = math.sqrt(2.0)

Grid = List[List[int]]
Cell = Tuple[int, int]
FloatPos = Tuple[float, float]


@dataclass
class Agent:
    """
    Grid-based agent with continuous float position for:
      - distance-based game logic (mines/suppression/shaping)
      - smooth rendering

    Research invariants:
      - deterministic state transitions given (dt, path)
      - stable per-episode identity for reward routing/logging
      - no hidden randomness inside the Agent

    Design invariant (critical):
      - Agent does NOT call GameManager rule methods (death, scoring, rewards).
        Those are handled by GameField/GameManager during the env tick.
    """

    # --- Required by GameField.spawn_agents() ---
    x: int
    y: int
    side: str                  # "blue" or "red"
    cols: int                  # grid width
    rows: int                  # grid height
    grid: Grid                 # environment grid reference

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

    # Continuous position (derived from x,y and move_accum interpolation)
    _float_x: float = field(init=False, repr=False)
    _float_y: float = field(init=False, repr=False)

    # Status
    enabled: bool = True
    tag_cooldown: float = 0.0

    # Objective payload
    is_carrying_flag: bool = False

    # Mines (optional; env manages meaning)
    mine_charges: int = 0
    max_mine_charges: int = 2

    # Spawn
    spawn_xy: Cell = (0, 0)

    # One-step flags (consumed by trainer/UI if desired)
    _just_picked_up_flag: bool = field(default=False, init=False, repr=False)
    _just_scored: bool = field(default=False, init=False, repr=False)
    _just_tagged_enemy: bool = field(default=False, init=False, repr=False)

    was_just_disabled: bool = field(default=False, init=False)
    disabled_this_tick: bool = field(default=False, init=False)

    # Soft suppression flags (GameField controls)
    suppression_timer: float = 0.0
    suppressed_last_tick: bool = False
    suppressed_this_tick: bool = False

    # RL/debug hooks set by env (optional, but used in your logs)
    decision_count: int = 0
    last_macro_action: Optional[Any] = field(default=None, repr=False, compare=False)
    last_macro_action_idx: int = 0

    # Stable unique identity (for reward routing/logging)
    instance_id: int = field(init=False)
    _NEXT_INSTANCE_ID: ClassVar[int] = 0

    def __post_init__(self) -> None:
        self.x, self.y = self._clamp_cell(self.x, self.y)
        self.spawn_xy = self._clamp_cell(*self.spawn_xy)

        if not isinstance(self.path, deque):
            self.path = deque(self.path)

        self._float_x = float(self.x)
        self._float_y = float(self.y)

        self.instance_id = Agent._NEXT_INSTANCE_ID
        Agent._NEXT_INSTANCE_ID += 1

        # Stable keys used by GameField external action routing and GameManager reward routing
        self.slot_id: str = f"{self.side}_{self.agent_id}"
        self.unique_id: str = f"{self.side}_{self.agent_id}_{self.instance_id}"

    # ----------------------------------------------------------
    # Core helpers / properties
    # ----------------------------------------------------------
    def _clamp_cell(self, col: int, row: int) -> Cell:
        c = max(0, min(self.cols - 1, int(col)))
        r = max(0, min(self.rows - 1, int(row)))
        return (c, r)

    @property
    def cell_pos(self) -> Cell:
        """Convenience alias used by some env utilities."""
        return (self.x, self.y)

    @property
    def float_pos(self) -> FloatPos:
        return (self._float_x, self._float_y)

    def get_position(self) -> Cell:
        return (self.x, self.y)

    # Legacy-friendly aliases (kept intentionally because your env calls these)
    def getSide(self) -> str:
        return self.side

    def isEnabled(self) -> bool:
        return bool(self.enabled)

    def isTagged(self) -> bool:
        return (not self.enabled) and (self.tag_cooldown > 0.0)

    def isCarryingFlag(self) -> bool:
        return bool(self.is_carrying_flag)

    def attach_game_manager(self, gm: Any) -> None:
        """
        Attach GameManager reference.
        Agent will NOT call core rule methods on GM during disable/kill (RL-safe),
        but may call optional PBRS if present.
        """
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
    # Path handling
    # ----------------------------------------------------------
    def setPath(self, path: Optional[List[Cell]]) -> None:
        """
        Path is a list of (x,y) cell waypoints. Agent executes it deterministically.
        Empty/None clears path.
        """
        if not path:
            self.clearPath()
            self.move_accum = 0.0
            return

        clamped = [self._clamp_cell(c[0], c[1]) for c in path]
        self.path = deque(clamped)
        self.waypoint = clamped[-1] if clamped else None
        self.move_accum = 0.0

    def clearPath(self) -> None:
        self.path.clear()
        self.waypoint = None

    # ----------------------------------------------------------
    # Disable / respawn (RL-safe: NO GameManager rule calls)
    # ----------------------------------------------------------
    def disable_for_seconds(self, seconds: float) -> None:
        """
        Disables the agent and starts/extends respawn timer.

        IMPORTANT:
          - Does NOT call GameManager methods (no handle_agent_death, no rewards, no flag logic).
          - Flag dropping/scoring consistency is handled by GameManager.sanity_check_flags()
            inside the main env tick.
        """
        s = max(0.0, float(seconds))

        # If already disabled, just extend the cooldown and exit.
        if not self.enabled:
            self.tag_cooldown = max(self.tag_cooldown, s)
            return

        self.was_just_disabled = True
        self.disabled_this_tick = True

        self.enabled = False
        self.tag_cooldown = max(self.tag_cooldown, s)

        # Force local carry off (do NOT mark scored)
        # NOTE: GM still owns authoritative flag state; it will reconcile on tick via sanity_check_flags.
        self.setCarryingFlag(False, scored=False)

        # Clear movement state deterministically
        self.clearPath()
        self.move_accum = 0.0
        self._float_x = float(self.x)
        self._float_y = float(self.y)

    def respawn(self) -> None:
        """
        Resets to spawn location and clears transient state.

        If GameManager provides clear_flag_carrier_if_agent, we call it defensively
        (it exists in your refactored GM). If not present, we skip to avoid crashes.
        """
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

        self._just_picked_up_flag = False
        self._just_scored = False
        self._just_tagged_enemy = False
        self.is_carrying_flag = False

        self.suppression_timer = 0.0
        self.suppressed_last_tick = False
        self.suppressed_this_tick = False

    # ----------------------------------------------------------
    # Per-timestep update (movement + optional PBRS hook)
    # ----------------------------------------------------------
    def update(self, dt: float) -> None:
        if dt <= 0.0:
            return

        prev_float_pos = self.float_pos

        # Disabled: tick respawn timer only
        if not self.enabled:
            if self.tag_cooldown > 0.0:
                self.tag_cooldown = max(0.0, self.tag_cooldown - float(dt))
                if self.tag_cooldown <= 0.0:
                    self.respawn()
            return

        # Suppression flags are controlled by GameField; we preserve last_tick for convenience
        self.suppressed_last_tick = self.suppressed_this_tick
        self.suppressed_this_tick = False

        # No path: snap float to cell
        if not self.path:
            self._float_x = float(self.x)
            self._float_y = float(self.y)
            return

        # Accumulate movement budget in "cell cost" units
        self.move_accum += float(self.move_rate_cps) * float(dt)

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

        # Fractional interpolation toward next cell
        if self.path:
            nx, ny = self.path[0]
            dx = nx - self.x
            dy = ny - self.y
            step_cost = DIAGONAL_COST if (dx != 0 and dy != 0) else 1.0
            prog = min(max(self.move_accum / step_cost, 0.0), 1.0)
            self._float_x = float(self.x) + float(dx) * prog
            self._float_y = float(self.y) + float(dy) * prog
        else:
            self._float_x = float(self.x)
            self._float_y = float(self.y)

        # Optional PBRS hook (exists in your refactored GM; guard keeps Agent crash-proof)
        gm = self.game_manager
        if gm is not None and hasattr(gm, "reward_potential_shaping"):
            gm.reward_potential_shaping(self, prev_float_pos, self.float_pos)


__all__ = ["Agent", "TEAM_ZONE_RADIUS_CELLS", "DIAGONAL_COST"]
