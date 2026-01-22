# =========================
# policies.py
#   - Scripted Policies (OP1, OP2, OP3)
#   - SelfPlayRedPolicy (Neural Wrapper)
# =========================

import math
import random
from typing import Any, Tuple, Optional, List, Dict

import torch  # Top-level import as requested

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
        best_d2 = 10 ** 9
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
        if side == "red" and hasattr(game_field, "red_zone_col_range"):
            return tuple(game_field.red_zone_col_range)
        if side != "red" and hasattr(game_field, "blue_zone_col_range"):
            return tuple(game_field.blue_zone_col_range)
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

            candidates = [(cx, cy) for (cx, cy) in defense_band if not self._cell_has_mine(game_field, cx, cy)]
            if candidates:
                tx, ty = random.choice(candidates)
                tx, ty = self._safe_target(game_field, tx, ty)
                return _macro_to_int(MacroAction.GO_TO), (tx, ty)

            tx, ty = self._safe_target(game_field, flag_x, flag_y)
            return _macro_to_int(MacroAction.GO_TO), (tx, ty)

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

    def __init__(
        self,
        side: str = "red",
        mine_radius_check: float = 1.5,
        defense_radius_cells: float = 4.0,
        patrol_radius_cells: int = 3,
        assist_radius_mult: float = 1.5,
        defense_weight: Optional[float] = None,
        flag_weight: Optional[float] = None,
    ):
        self.side = side
        self.mine_radius_check = float(mine_radius_check)
        self.defense_radius_cells = float(defense_radius_cells)
        self.patrol_radius_cells = int(max(1, patrol_radius_cells))
        self.assist_radius_mult = float(assist_radius_mult)
        self.defense_weight = None if defense_weight is None else float(defense_weight)
        self.flag_weight = None if flag_weight is None else float(flag_weight)

    def reset(self) -> None:
        return None

    def _defender_action(self, agent: Agent, gm: GameManager, game_field: "GameField") -> Tuple[
        int, Optional[Tuple[int, int]]]:
        side = getattr(agent, "side", self.side)
        flag_x, flag_y = gm.get_team_zone_center(side)
        fx, fy = self._safe_target(game_field, flag_x, flag_y)

        if agent.isCarryingFlag():
            return _macro_to_int(MacroAction.GO_HOME), None

        ax, ay = _agent_xy(agent)

        # If enemy is carrying our flag, intercept to prevent scoring
        enemy_carrier = None
        if side == "blue" and getattr(gm, "blue_flag_taken", False):
            enemy_carrier = getattr(gm, "blue_flag_carrier", None)
        if side == "red" and getattr(gm, "red_flag_taken", False):
            enemy_carrier = getattr(gm, "red_flag_carrier", None)
        if enemy_carrier is not None:
            ex, ey = _agent_xy(enemy_carrier)
            return _macro_to_int(MacroAction.GO_TO), self._safe_target(game_field, int(ex), int(ey))

        # If low on charges and pickups exist, prioritize grabbing mines
        max_charges = int(getattr(game_field, "max_mine_charges_per_agent", 2))
        charges = int(getattr(agent, "mine_charges", 0))
        if charges < max_charges:
            my_pickups = [p for p in getattr(game_field, "mine_pickups", []) if getattr(p, "owner_side", None) == side]
            if my_pickups:
                return _macro_to_int(MacroAction.GRAB_MINE), None

        # Support suppression: move toward enemy near teammate / flag
        enemies = game_field.red_agents if side == "blue" else game_field.blue_agents
        teammates = game_field.blue_agents if side == "blue" else game_field.red_agents
        teammates = [t for t in teammates if t is not agent]
        sup_range = float(getattr(game_field, "suppression_range_cells", 2.0))
        assist_r = sup_range * self.assist_radius_mult

        def _nearest_enemy(refx: float, refy: float) -> Optional[Any]:
            best = None
            best_d2 = 1e9
            for e in enemies:
                if e is None or (hasattr(e, "isEnabled") and not e.isEnabled()):
                    continue
                ex, ey = _agent_xy(e)
                d2 = (ex - refx) ** 2 + (ey - refy) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best = e
            return best

        # If enemy is near our flag, converge for defense
        near_flag_enemy = _nearest_enemy(float(flag_x), float(flag_y))
        if near_flag_enemy is not None:
            ex, ey = _agent_xy(near_flag_enemy)
            if math.hypot(ex - float(flag_x), ey - float(flag_y)) <= self.defense_radius_cells:
                return _macro_to_int(MacroAction.GO_TO), self._safe_target(game_field, int(ex), int(ey))

        # If teammate is close to an enemy, move to support for suppression
        for t in teammates:
            tx, ty = _agent_xy(t)
            e = _nearest_enemy(tx, ty)
            if e is None:
                continue
            ex, ey = _agent_xy(e)
            if math.hypot(ex - tx, ey - ty) <= assist_r:
                return _macro_to_int(MacroAction.GO_TO), self._safe_target(game_field, int(ex), int(ey))

        # Strategic mine placement in our defensive zone
        has_flag_mine = any(
            (getattr(m, "owner_side", None) == side) and (
                        math.hypot(float(m.x) - fx, float(m.y) - fy) <= self.mine_radius_check)
            for m in getattr(game_field, "mines", [])
        )

        if (not has_flag_mine) and getattr(agent, "mine_charges", 0) > 0:
            if math.hypot(ax - fx, ay - fy) <= self.mine_radius_check and not self._cell_has_mine(game_field, fx, fy):
                return _macro_to_int(MacroAction.PLACE_MINE), (fx, fy)
            return _macro_to_int(MacroAction.GO_TO), (fx, fy)

        # Patrol + mine placement around the flag within our zone
        patrol: List[Tuple[int, int]] = []
        blue_zone = getattr(game_field, "blue_zone_col_range", None)
        red_zone = getattr(game_field, "red_zone_col_range", None)
        for dy in range(-self.patrol_radius_cells, self.patrol_radius_cells + 1):
            for dx in range(-self.patrol_radius_cells, self.patrol_radius_cells + 1):
                if dx * dx + dy * dy > self.patrol_radius_cells * self.patrol_radius_cells:
                    continue
                px, py = int(flag_x + dx), int(flag_y + dy)
                if blue_zone is not None and side == "blue":
                    if not (int(blue_zone[0]) <= px <= int(blue_zone[1])):
                        continue
                if red_zone is not None and side == "red":
                    if not (int(red_zone[0]) <= px <= int(red_zone[1])):
                        continue
                if self._is_free(px, py, game_field):
                    patrol.append((px, py))

        if patrol:
            # If we have charges, place mines on a free patrol cell without a mine
            if int(getattr(agent, "mine_charges", 0)) > 0:
                mine_targets = [p for p in patrol if not self._cell_has_mine(game_field, p[0], p[1])]
                if mine_targets:
                    mine_targets.sort(key=lambda p: (p[0] - ax) ** 2 + (p[1] - ay) ** 2)
                    tx, ty = mine_targets[0]
                    if math.hypot(ax - tx, ay - ty) <= self.mine_radius_check:
                        return _macro_to_int(MacroAction.PLACE_MINE), (tx, ty)
                    return _macro_to_int(MacroAction.GO_TO), self._safe_target(game_field, tx, ty)

            # Prefer a patrol point that's not right under the agent
            patrol.sort(key=lambda p: (p[0] - ax) ** 2 + (p[1] - ay) ** 2, reverse=True)
            tx, ty = patrol[0]
            return _macro_to_int(MacroAction.GO_TO), self._safe_target(game_field, tx, ty)

        return _macro_to_int(MacroAction.GO_TO), (fx, fy)

    def _attacker_action(self, agent: Agent, gm: GameManager, game_field: "GameField") -> Tuple[
        int, Optional[Tuple[int, int]]]:
        if agent.isCarryingFlag():
            return _macro_to_int(MacroAction.GO_HOME), None
        return _macro_to_int(MacroAction.GET_FLAG), None

    def select_action(self, obs: Any, agent: Agent, game_field: "GameField") -> Tuple[int, Optional[Tuple[int, int]]]:
        gm: GameManager = game_field.manager
        if self.defense_weight is not None and self.flag_weight is not None:
            total = float(self.defense_weight + self.flag_weight)
            if total > 0.0:
                p_attack = float(self.flag_weight) / total
                if random.random() < p_attack:
                    return self._attacker_action(agent, gm, game_field)
                return self._defender_action(agent, gm, game_field)
        if getattr(agent, "agent_id", 0) == 0:
            return self._defender_action(agent, gm, game_field)
        return self._attacker_action(agent, gm, game_field)


class SelfPlayRedPolicy(Policy):
    """
    Neural self-play opponent wrapper for RED.
    Calls opp_policy.act(...) internally. No external control required.
    """

    def __init__(self, opp_policy: Any, deterministic: bool = True):
        self.opp_policy = opp_policy
        self.deterministic = bool(deterministic)

    def reset(self) -> None:
        if hasattr(self.opp_policy, "reset") and callable(getattr(self.opp_policy, "reset")):
            try:
                self.opp_policy.reset()
            except Exception:
                pass

    def select_action(
            self,
            obs: Any,
            agent: Agent,
            game_field: "GameField",
    ) -> Tuple[int, Optional[Tuple[int, int]]]:

        # Defensive fallback
        if self.opp_policy is None or (not hasattr(self.opp_policy, "act")):
            return _macro_to_int(MacroAction.GO_TO), None

        # Determine device from policy
        device = torch.device("cpu")
        if hasattr(self.opp_policy, "parameters"):
            try:
                p = next(self.opp_policy.parameters())
                device = p.device
            except Exception:
                pass

        # Prepare Tensor [1, C, H, W]
        # Note: obs is typically list-of-lists here. torch.tensor handles it, 
        # but float32 cast is crucial.
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            out = self.opp_policy.act(
                obs_tensor,
                agent=agent,
                game_field=game_field,
                deterministic=self.deterministic,
            )

        # Parse output (can be dict, tuple, or scalar)
        if isinstance(out, dict):
            ma = out.get("macro_action", 0)
            ta = out.get("target_action", None)

            if torch.is_tensor(ma):
                macro_idx = int(ma.reshape(-1)[0].item())
            else:
                macro_idx = int(ma)

            if ta is None:
                return macro_idx, None

            if torch.is_tensor(ta):
                target_idx = int(ta.reshape(-1)[0].item())
            else:
                target_idx = int(ta)

            return macro_idx, target_idx

        if isinstance(out, (tuple, list)):
            if len(out) == 0:
                return 0, None
            macro_idx = int(out[0])
            target_idx = int(out[1]) if len(out) > 1 else None
            return macro_idx, target_idx

        return int(out), None


__all__ = [
    "Policy",
    "OP1RedPolicy",
    "OP2RedPolicy",
    "OP3RedPolicy",
    "SelfPlayRedPolicy",
]