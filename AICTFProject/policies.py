import math
import random
from typing import Any, Tuple, Optional, List

from agents import Agent
from game_manager import GameManager
from macro_actions import MacroAction


def _macro_to_int(m: Any) -> int:
    """Enum-safe conversion."""
    return int(getattr(m, "value", m))


def _agent_xy(agent: Any) -> Tuple[float, float]:
    """Prefer float_pos if available; fall back to x/y."""
    fp = getattr(agent, "float_pos", None)
    if isinstance(fp, (tuple, list)) and len(fp) >= 2:
        return float(fp[0]), float(fp[1])
    return float(getattr(agent, "x", 0.0)), float(getattr(agent, "y", 0.0))


class Policy:
    """
    Base class for scripted policies in the CTF environment.
    """

    def select_action(
        self,
        obs: Any,
        agent: Agent,
        game_field: "GameField",
    ) -> Tuple[int, Optional[Tuple[int, int]]]:
        raise NotImplementedError

    def reset(self) -> None:
        return None

    # ----------------- bulletproof helpers -----------------
    def _clamp(self, x: int, y: int, game_field: "GameField") -> Tuple[int, int]:
        x = max(0, min(game_field.col_count - 1, int(x)))
        y = max(0, min(game_field.row_count - 1, int(y)))
        return x, y

    def _is_free(self, x: int, y: int, game_field: "GameField") -> bool:
        if not (0 <= x < game_field.col_count and 0 <= y < game_field.row_count):
            return False
        return game_field.grid[y][x] == 0

    def _nearest_free(self, x: int, y: int, game_field: "GameField", radius: int = 6) -> Tuple[int, int]:
        x, y = self._clamp(x, y, game_field)
        if self._is_free(x, y, game_field):
            return x, y

        best = None
        best_d2 = 10**9
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if self._is_free(nx, ny, game_field):
                    d2 = dx * dx + dy * dy
                    if d2 < best_d2:
                        best_d2 = d2
                        best = (nx, ny)
        return best if best is not None else (x, y)

    def _cell_has_mine(self, game_field: "GameField", x: int, y: int) -> bool:
        return any(int(m.x) == int(x) and int(m.y) == int(y) for m in getattr(game_field, "mines", []))

    def _safe_target(self, game_field: "GameField", x: int, y: int) -> Tuple[int, int]:
        """Clamp then nearest-free to guarantee validity."""
        x, y = self._clamp(x, y, game_field)
        return self._nearest_free(x, y, game_field)


class OP1RedPolicy(Policy):
    """
    OP1: Naive defensive policy.
    """

    def __init__(self, side: str = "red"):
        self.side = side

    def select_action(self, obs: Any, agent: Agent, game_field: "GameField") -> Tuple[int, Optional[Tuple[int, int]]]:
        gm: GameManager = game_field.manager
        side = getattr(agent, "side", self.side)

        home_x, home_y = gm.get_team_zone_center(side)

        # "Back" of end zone
        if side == "red":
            tx, ty = home_x + 1, home_y
        else:
            tx, ty = home_x - 1, home_y

        tx, ty = self._safe_target(game_field, tx, ty)

        if agent.isCarryingFlag():
            return _macro_to_int(MacroAction.GO_HOME), None

        return _macro_to_int(MacroAction.GO_TO), (tx, ty)


class OP2RedPolicy(Policy):
    """
    OP2: Defensive mine-layer.
    """

    def __init__(self, side: str = "red", defense_band_radius: int = 1):
        self.side = side
        self.defense_band_radius = max(0, int(defense_band_radius))

    def _zone_col_range(self, game_field: "GameField", side: str) -> Tuple[int, int]:
        # Prefer explicit ranges if present
        if side == "red" and hasattr(game_field, "red_zone_col_range"):
            return tuple(game_field.red_zone_col_range)
        if side != "red" and hasattr(game_field, "blue_zone_col_range"):
            return tuple(game_field.blue_zone_col_range)

        # Fallback: just use full width (still safe, just less “zoney”)
        return (0, game_field.col_count - 1)

    def select_action(self, obs: Any, agent: Agent, game_field: "GameField") -> Tuple[int, Optional[Tuple[int, int]]]:
        gm: GameManager = game_field.manager
        side = getattr(agent, "side", self.side)

        if agent.isCarryingFlag():
            return _macro_to_int(MacroAction.GO_HOME), None

        own_min_col, own_max_col = self._zone_col_range(game_field, side)
        flag_x, flag_y = gm.get_team_zone_center(side)

        defense_band: List[Tuple[int, int]] = []
        for dy in range(-self.defense_band_radius, self.defense_band_radius + 1):
            row = int(flag_y + dy)
            if 0 <= row < game_field.row_count:
                for c in range(int(own_min_col), int(own_max_col) + 1):
                    if self._is_free(c, row, game_field):
                        defense_band.append((c, row))

        ax, ay = _agent_xy(agent)

        if getattr(agent, "mine_charges", 0) > 0:
            # If near an unmined band cell, place
            near: List[Tuple[int, int]] = []
            for cx, cy in defense_band:
                if self._cell_has_mine(game_field, cx, cy):
                    continue
                if math.hypot(ax - cx, ay - cy) <= 1.5:
                    near.append((cx, cy))

            if near:
                tx, ty = random.choice(near)
                tx, ty = self._safe_target(game_field, tx, ty)
                return _macro_to_int(MacroAction.PLACE_MINE), (tx, ty)

            # Otherwise walk to an unmined band cell
            candidates = [(cx, cy) for (cx, cy) in defense_band if not self._cell_has_mine(game_field, cx, cy)]
            if candidates:
                tx, ty = random.choice(candidates)
                tx, ty = self._safe_target(game_field, tx, ty)
                return _macro_to_int(MacroAction.GO_TO), (tx, ty)

            # Everything mined: sit near flag
            tx, ty = self._safe_target(game_field, flag_x, flag_y)
            return _macro_to_int(MacroAction.GO_TO), (tx, ty)

        # No mines: patrol in defense band
        if defense_band:
            tx, ty = random.choice(defense_band)
        else:
            tx, ty = self._safe_target(game_field, flag_x, flag_y)
        tx, ty = self._safe_target(game_field, tx, ty)
        return _macro_to_int(MacroAction.GO_TO), (tx, ty)


class OP3RedPolicy(Policy):
    """
    OP3: Mixed defender-attacker.
    """

    def __init__(self, side: str = "red", mine_radius_check: float = 1.5):
        self.side = side
        self.mine_radius_check = float(mine_radius_check)

    def reset(self) -> None:
        return None

    def _defender_action(self, agent: Agent, gm: GameManager, game_field: "GameField") -> Tuple[int, Optional[Tuple[int, int]]]:
        side = getattr(agent, "side", self.side)
        flag_x, flag_y = gm.get_team_zone_center(side)
        fx, fy = self._safe_target(game_field, flag_x, flag_y)

        if agent.isCarryingFlag():
            return _macro_to_int(MacroAction.GO_HOME), None

        ax, ay = _agent_xy(agent)

        has_flag_mine = any(
            (getattr(m, "owner_side", None) == side) and (math.hypot(float(m.x) - fx, float(m.y) - fy) <= self.mine_radius_check)
            for m in getattr(game_field, "mines", [])
        )

        if (not has_flag_mine) and getattr(agent, "mine_charges", 0) > 0:
            # if close, place (avoid stacking same cell)
            if math.hypot(ax - fx, ay - fy) <= self.mine_radius_check and not self._cell_has_mine(game_field, fx, fy):
                return _macro_to_int(MacroAction.PLACE_MINE), (fx, fy)
            return _macro_to_int(MacroAction.GO_TO), (fx, fy)

        return _macro_to_int(MacroAction.GO_TO), (fx, fy)

    def _attacker_action(self, agent: Agent, gm: GameManager, game_field: "GameField") -> Tuple[int, Optional[Tuple[int, int]]]:
        if agent.isCarryingFlag():
            return _macro_to_int(MacroAction.GO_HOME), None
        return _macro_to_int(MacroAction.GET_FLAG), None

    def select_action(self, obs: Any, agent: Agent, game_field: "GameField") -> Tuple[int, Optional[Tuple[int, int]]]:
        gm: GameManager = game_field.manager
        if getattr(agent, "agent_id", 0) == 0:
            return self._defender_action(agent, gm, game_field)
        else:
            return self._attacker_action(agent, gm, game_field)
