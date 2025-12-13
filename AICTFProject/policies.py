import math
import random
from typing import Any, Tuple, Optional, List

from agents import Agent
from game_manager import GameManager
from macro_actions import MacroAction


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
        # grid[row][col] == 0 is passable in your env
        if not (0 <= x < game_field.col_count and 0 <= y < game_field.row_count):
            return False
        return game_field.grid[y][x] == 0

    def _nearest_free(self, x: int, y: int, game_field: "GameField", radius: int = 6) -> Tuple[int, int]:
        """
        Find a nearby passable cell. If none found, fallback to current.
        """
        x, y = self._clamp(x, y, game_field)
        if self._is_free(x, y, game_field):
            return x, y

        # Spiral-ish search
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

    def _random_free_in_cols(self, game_field: "GameField", col_min: int, col_max: int) -> Tuple[int, int]:
        col_min = max(0, min(game_field.col_count - 1, col_min))
        col_max = max(0, min(game_field.col_count - 1, col_max))
        if col_min > col_max:
            col_min, col_max = col_max, col_min

        # Try a few random samples first
        for _ in range(25):
            x = random.randint(col_min, col_max)
            y = random.randint(0, game_field.row_count - 1)
            if self._is_free(x, y, game_field):
                return (x, y)

        # Fallback: scan
        for y in range(game_field.row_count):
            for x in range(col_min, col_max + 1):
                if self._is_free(x, y, game_field):
                    return (x, y)

        # Last resort
        return (game_field.col_count // 2, game_field.row_count // 2)

    def _cell_has_mine(self, game_field: "GameField", x: int, y: int) -> bool:
        return any(int(m.x) == int(x) and int(m.y) == int(y) for m in getattr(game_field, "mines", []))


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

        tx, ty = self._nearest_free(tx, ty, game_field)

        if agent.isCarryingFlag():
            return int(MacroAction.GO_HOME), None

        return int(MacroAction.GO_TO), (tx, ty)


class OP2RedPolicy(Policy):
    """
    OP2: Defensive mine-layer.
    """

    def __init__(self, side: str = "red", defense_band_width: int = 1):
        self.side = side
        self.defense_band_width = max(1, int(defense_band_width))

    def select_action(self, obs: Any, agent: Agent, game_field: "GameField") -> Tuple[int, Optional[Tuple[int, int]]]:
        gm: GameManager = game_field.manager
        side = getattr(agent, "side", self.side)

        if agent.isCarryingFlag():
            return int(MacroAction.GO_HOME), None

        if side == "red":
            own_min_col, own_max_col = game_field.red_zone_col_range
        else:
            own_min_col, own_max_col = game_field.blue_zone_col_range

        flag_x, flag_y = gm.get_team_zone_center(side)

        defense_band: List[Tuple[int, int]] = []
        half_w = self.defense_band_width // 2
        for dy in range(-half_w, half_w + 1):
            row = flag_y + dy
            if 0 <= row < game_field.row_count:
                for c in range(own_min_col, own_max_col + 1):
                    if self._is_free(c, row, game_field):
                        defense_band.append((c, row))

        # Prefer unminded cells for placing mines
        if agent.mine_charges > 0:
            # If near an unminded band cell, place
            near = []
            for cx, cy in defense_band:
                if self._cell_has_mine(game_field, cx, cy):
                    continue
                if math.hypot(agent.x - cx, agent.y - cy) <= 1.5:
                    near.append((cx, cy))

            if near:
                return int(MacroAction.PLACE_MINE), random.choice(near)

            # Otherwise walk to an unminded band cell
            candidates = [(cx, cy) for (cx, cy) in defense_band if not self._cell_has_mine(game_field, cx, cy)]
            if candidates:
                tx, ty = random.choice(candidates)
                return int(MacroAction.GO_TO), (tx, ty)

            # If every cell already mined, just defend around flag
            tx, ty = self._nearest_free(flag_x, flag_y, game_field)
            return int(MacroAction.GO_TO), (tx, ty)

        # No mines -> grab pickups
        return int(MacroAction.GRAB_MINE), None


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
        fx, fy = self._nearest_free(flag_x, flag_y, game_field)

        if agent.isCarryingFlag():
            return int(MacroAction.GO_HOME), None

        has_flag_mine = any(
            (m.owner_side == side) and (math.hypot(m.x - fx, m.y - fy) <= self.mine_radius_check)
            for m in getattr(game_field, "mines", [])
        )

        if (not has_flag_mine) and agent.mine_charges > 0:
            # if close, place (but avoid stacking the same cell)
            if math.hypot(agent.x - fx, agent.y - fy) <= self.mine_radius_check and not self._cell_has_mine(game_field, fx, fy):
                return int(MacroAction.PLACE_MINE), (fx, fy)
            return int(MacroAction.GO_TO), (fx, fy)

        return int(MacroAction.GO_TO), (fx, fy)

    def _attacker_action(self, agent: Agent, gm: GameManager, game_field: "GameField") -> Tuple[int, Optional[Tuple[int, int]]]:
        if agent.isCarryingFlag():
            return int(MacroAction.GO_HOME), None
        return int(MacroAction.GET_FLAG), None

    def select_action(self, obs: Any, agent: Agent, game_field: "GameField") -> Tuple[int, Optional[Tuple[int, int]]]:
        gm: GameManager = game_field.manager
        if getattr(agent, "agent_id", 0) == 0:
            return self._defender_action(agent, gm, game_field)
        else:
            return self._attacker_action(agent, gm, game_field)
