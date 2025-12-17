# =========================
# game_field.py (FULL UPDATED, OBJECTIVE-FIRST, CRASH-PROOF)
# =========================

import random
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import torch
import numpy as np  # ✅ boolean action masks

from game_manager import GameManager
from agents import Agent
from pathfinder import Pathfinder

from macro_actions import MacroAction
from policies import Policy, OP1RedPolicy, OP2RedPolicy, OP3RedPolicy

ARENA_WIDTH_M = 10.0
ARENA_HEIGHT_M = 4.28

CNN_COLS = 20
CNN_ROWS = 20
NUM_CNN_CHANNELS = 7


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


class GameField:
    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.row_count = len(grid)
        self.col_count = len(grid[0]) if self.row_count else 0

        self.manager = GameManager(cols=self.col_count, rows=self.row_count)
        self.manager.bind_game_field(self)  # ✅ enables exact team reward routing
        self._init_macro_indexing()

        self.mines: List[Mine] = []
        self.mine_pickups: List[MinePickup] = []
        self.mine_radius_cells = 1.5
        self.suppression_range_cells = 2.0
        self.mines_per_team = 4
        self.max_mine_charges_per_agent = 2

        # ✅ Objective-first knobs (tuneable)
        self.mine_detour_disable_radius_cells: float = 8.0   # if near enemy flag, GRAB_MINE disabled
        self.allow_offensive_mine_placing: bool = False      # keep mines mostly defensive

        self.use_internal_policies: bool = True
        self.external_control_for_side: Dict[str, bool] = {"blue": False, "red": False}

        self.pending_external_actions: Dict[str, Tuple[int, Any]] = {}
        self.external_missing_action_mode: str = "idle"  # "idle" or "internal"

        self.policies: Dict[str, Any] = {
            "blue": OP3RedPolicy("blue"),
            "red": OP3RedPolicy("red"),
        }
        self.opponent_mode: str = "OP3"

        self._init_zones()
        self.agents_per_team: int = 2
        self.blue_agents: List[Agent] = []
        self.red_agents: List[Agent] = []

        self.pathfinder = Pathfinder(
            self.grid,
            self.row_count,
            self.col_count,
            allow_diagonal=True,
            block_corners=True,
        )

        self.cell_width_m = ARENA_WIDTH_M / max(1, self.col_count)
        self.cell_height_m = ARENA_HEIGHT_M / max(1, self.row_count)

        # Stage-0 semantic targets
        self.num_macro_targets: int = 8
        self.macro_targets: List[Tuple[int, int]] = []

        # ✅ Per-episode seed (set on reset_default)
        self.episode_seed: Optional[int] = None

        self.respawn_seconds: float = 2.0
        self.decision_interval_seconds: float = 0.7
        self.decision_cooldown_seconds_by_agent: Dict[int, float] = {}

        self.banner_queue: List[Tuple[str, Tuple[int, int, int], float]] = []
        self.debug_draw_ranges: bool = True
        self.debug_draw_mine_ranges: bool = True

        self.reset_default()

    # ---------------- Small utilities ----------------
    def _is_free_cell(self, x: int, y: int) -> bool:
        return (
            0 <= x < self.col_count and 0 <= y < self.row_count and
            int(self.grid[y][x]) == 0
        )

    def _clamp_cell(self, x: int, y: int) -> Tuple[int, int]:
        x = int(max(0, min(self.col_count - 1, x)))
        y = int(max(0, min(self.row_count - 1, y)))
        return x, y

    def _agent_cell_pos(self, agent: Any) -> Tuple[int, int]:
        cp = getattr(agent, "cell_pos", None)
        if isinstance(cp, (tuple, list)) and len(cp) >= 2:
            try:
                return self._clamp_cell(int(cp[0]), int(cp[1]))
            except Exception:
                pass

        # common fallbacks
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

        return (0, 0)

    def _agent_float_pos(self, agent: Any) -> Tuple[float, float]:
        fp = getattr(agent, "float_pos", None)
        if isinstance(fp, (tuple, list)) and len(fp) >= 2:
            try:
                return float(fp[0]), float(fp[1])
            except Exception:
                pass
        # fallback: assume x/y are cell coords
        try:
            return float(getattr(agent, "x", 0.0)), float(getattr(agent, "y", 0.0))
        except Exception:
            return 0.0, 0.0

    # ---------------- External actions ----------------
    def _external_key_for_agent(self, agent: Agent) -> str:
        if hasattr(agent, "slot_id"):
            try:
                return str(getattr(agent, "slot_id"))
            except Exception:
                pass
        return f"{agent.side}_{getattr(agent, 'agent_id', 0)}"

    def _external_key_candidates(self, agent: Agent) -> List[str]:
        keys: List[str] = []
        keys.append(self._external_key_for_agent(agent))
        if hasattr(agent, "unique_id"):
            try:
                keys.append(str(getattr(agent, "unique_id")))
            except Exception:
                pass
        if hasattr(agent, "slot_id"):
            try:
                keys.append(str(getattr(agent, "slot_id")))
            except Exception:
                pass
        keys.append(f"{agent.side}_{getattr(agent, 'agent_id', 0)}")

        seen = set()
        out: List[str] = []
        for k in keys:
            if k and k not in seen:
                out.append(k)
                seen.add(k)
        return out

    def _init_macro_indexing(self) -> None:
        """
        Canonical mapping between network action indices (0..4) and MacroAction enum.
        Never assume enum values are 0..4.
        """
        self.macro_order = [
            MacroAction.GO_TO,  # idx 0
            MacroAction.GRAB_MINE,  # idx 1
            MacroAction.GET_FLAG,  # idx 2
            MacroAction.PLACE_MINE,  # idx 3
            MacroAction.GO_HOME,  # idx 4
        ]
        self.macro_to_index = {m: i for i, m in enumerate(self.macro_order)}
        self.n_macros = len(self.macro_order)

    def macro_idx_to_action(self, idx: int) -> MacroAction:
        return self.macro_order[int(idx) % self.n_macros]

    def macro_action_to_idx(self, action: MacroAction) -> int:
        return int(self.macro_to_index[action])

    def normalize_macro(self, action_any: Any) -> Tuple[int, MacroAction]:
        """
        Accepts:
          - network index (0..4)
          - MacroAction enum
          - raw int enum value (we treat as index ONLY if it's 0..n-1)
        Returns:
          (macro_idx, MacroAction)
        """
        if isinstance(action_any, MacroAction):
            a = action_any
            return self.macro_action_to_idx(a), a

        # If someone passed an int, treat it as a network index (0..4).
        try:
            idx = int(action_any)
        except Exception:
            idx = 0

        # idx is always interpreted as network index
        a = self.macro_idx_to_action(idx)
        return idx % self.n_macros, a

    def submit_external_actions(self, actions_by_agent: Dict[str, Tuple[int, Any]]) -> None:
        """
        Trainer-facing API: submit actions for this decision boundary.

        actions_by_agent:
          key -> (macro_val, target_param)

        target_param may be int index OR (x,y) tuple/list.
        """
        if not isinstance(actions_by_agent, dict):
            return

        for k, v in actions_by_agent.items():
            try:
                macro_val, target_param = v
                self.pending_external_actions[str(k)] = (int(macro_val), target_param)
            except Exception:
                continue

    def _clear_pending_for_agent(self, agent: Agent) -> None:
        for k in self._external_key_candidates(agent):
            if k in self.pending_external_actions:
                self.pending_external_actions.pop(k, None)

    def _consume_external_action_for_agent(self, agent: Agent) -> Optional[Tuple[int, Any]]:
        """
        Returns (macro_val, target_param) if present, and consumes it.
        target_param may be int index or (x,y).
        """
        for k in self._external_key_candidates(agent):
            if k in self.pending_external_actions:
                act = self.pending_external_actions.pop(k, None)
                if act is not None:
                    return act
        return None

    # ---------------- Setup ----------------
    def _init_zones(self) -> None:
        total_cols = max(1, self.col_count)
        third = max(1, total_cols // 3)

        blue_min = 0
        blue_max = max(blue_min, third - 1)

        red_max = total_cols - 1
        red_min = min(total_cols - third, red_max)

        self.blue_zone_col_range = (blue_min, blue_max)
        self.red_zone_col_range = (red_min, red_max)

    def _init_macro_targets(self, seed: Optional[int] = None) -> None:
        """
        Fixed semantic waypoints (stable list). Seed accepted for API compatibility only.
        """
        self.macro_targets.clear()

        def clamp_cell(x: int, y: int) -> Tuple[int, int]:
            return self._clamp_cell(int(x), int(y))

        def nearest_free(x: int, y: int, radius: int = 8) -> Tuple[int, int]:
            x, y = clamp_cell(x, y)
            if self._is_free_cell(x, y):
                return (x, y)

            best = None
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

        # these should exist after reset_game()
        own_home = gm.blue_flag_home
        enemy_home = gm.red_flag_home
        own_zone_center = gm.get_team_zone_center("blue")
        enemy_zone_center = gm.get_team_zone_center("red")

        mid_col = self.col_count // 2
        mid_row = self.row_count // 2

        top_row = max(0, min(self.row_count - 1, 5))
        bottom_row = max(0, min(self.row_count - 1, self.row_count - 5))

        def_x = own_home[0] + 5
        if def_x >= self.col_count:
            def_x = max(0, own_home[0] - 5)
        defensive_point = (def_x, own_home[1])

        targets_raw: List[Tuple[int, int]] = [
            own_home,                  # 0
            enemy_home,                # 1
            own_zone_center,           # 2
            enemy_zone_center,         # 3
            (mid_col, mid_row),        # 4
            (mid_col, top_row),        # 5
            (mid_col, bottom_row),     # 6
            defensive_point,           # 7
        ]

        self.macro_targets = [nearest_free(x, y) for (x, y) in targets_raw]
        self.num_macro_targets = 8

    def get_macro_target(self, index: int) -> Tuple[int, int]:
        if not self.macro_targets:
            self._init_macro_targets(seed=self.episode_seed)
        if self.num_macro_targets <= 0:
            return self.col_count // 2, self.row_count // 2
        return self.macro_targets[int(index) % self.num_macro_targets]

    def get_all_macro_targets(self) -> List[Tuple[int, int]]:
        if not self.macro_targets:
            self._init_macro_targets(seed=self.episode_seed)
        return list(self.macro_targets)

    def getGameManager(self) -> GameManager:
        return self.manager

    # ----------------------------------------------------------
    # ✅ ACTION MASK (OBJECTIVE-FIRST)
    # Trainer should apply this at logits-time.
    # ----------------------------------------------------------
    def get_macro_mask(self, agent: Agent) -> np.ndarray:
        """
        Returns a boolean mask over *network macro indices* (0..n_macros-1),
        matching self.macro_order.
        """
        n = getattr(self, "n_macros", 5)
        mask = np.ones((n,), dtype=np.bool_)

        if agent is None or (not agent.isEnabled()):
            mask[:] = False
            return mask

        side = getattr(agent, "side", None)
        if side not in ("blue", "red"):
            # Unknown side: disable mine-related macros by index
            mask[self.macro_action_to_idx(MacroAction.GRAB_MINE)] = False
            mask[self.macro_action_to_idx(MacroAction.PLACE_MINE)] = False
            return mask

        gm = self.manager
        phase = str(getattr(gm, "phase_name", "OP1")).upper()

        idx_go_to = self.macro_action_to_idx(MacroAction.GO_TO)
        idx_grab_mine = self.macro_action_to_idx(MacroAction.GRAB_MINE)
        idx_get_flag = self.macro_action_to_idx(MacroAction.GET_FLAG)
        idx_place_mine = self.macro_action_to_idx(MacroAction.PLACE_MINE)
        idx_go_home = self.macro_action_to_idx(MacroAction.GO_HOME)

        # Carrying flag: only GO_HOME + GO_TO (and GO_HOME should be preferred by reward, not mask)
        if agent.isCarryingFlag():
            mask[idx_get_flag] = False
            mask[idx_grab_mine] = False
            mask[idx_place_mine] = False
            mask[idx_go_home] = True
            return mask

        # OP1: hard-disable mines so learning locks onto objective
        if phase == "OP1":
            mask[idx_grab_mine] = False
            mask[idx_place_mine] = False

        charges = int(getattr(agent, "mine_charges", 0))

        # Rule A: never GRAB_MINE if you already have charges
        if charges > 0:
            mask[idx_grab_mine] = False

        # Rule B: never detour for mines if near enemy flag
        ax, ay = self._agent_float_pos(agent)
        ex, ey = gm.get_enemy_flag_position(side)
        d_enemy_flag = math.hypot(float(ax) - float(ex), float(ay) - float(ey))
        if d_enemy_flag <= float(self.mine_detour_disable_radius_cells):
            mask[idx_grab_mine] = False

        # Rule C: PLACE_MINE only if you have charges, and (optionally) only in your own zone
        if charges <= 0:
            mask[idx_place_mine] = False
        else:
            if not self.allow_offensive_mine_placing:
                own_min, own_max = self.blue_zone_col_range if side == "blue" else self.red_zone_col_range
                if not (own_min - 1 <= float(ax) <= own_max + 1):
                    mask[idx_place_mine] = False

        return mask

    # Policy wiring
    def set_policies(self, blue: Any, red: Any) -> None:
        self.policies["blue"] = blue
        self.policies["red"] = red

    def set_red_opponent(self, mode: str) -> None:
        mode = mode.upper()
        self.opponent_mode = mode
        if mode == "OP1":
            self.policies["red"] = OP1RedPolicy("red")
        elif mode == "OP2":
            self.policies["red"] = OP2RedPolicy("red")
        else:
            self.policies["red"] = OP3RedPolicy("red")

    def set_external_control(self, side: str, external: bool) -> None:
        if side not in ("blue", "red"):
            raise ValueError(f"Unknown side: {side}")
        self.external_control_for_side[side] = external

    def set_all_external_control(self, external: bool) -> None:
        for side in ("blue", "red"):
            self.external_control_for_side[side] = external

    def set_red_policy_neural(self, policy_net: Any) -> None:
        self.policies["red"] = policy_net
        self.opponent_mode = "NEURAL"

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------
    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        x = (col + 0.5) * self.cell_width_m
        y = (row + 0.5) * self.cell_height_m
        return x, y

    def world_to_cnn_cell(self, x_m: float, y_m: float) -> Tuple[int, int]:
        u = max(0.0, min(1.0, x_m / ARENA_WIDTH_M))
        v = max(0.0, min(1.0, y_m / ARENA_HEIGHT_M))

        col = int(u * CNN_COLS)
        row = int(v * CNN_ROWS)

        col = max(0, min(CNN_COLS - 1, col))
        row = max(0, min(CNN_ROWS - 1, row))
        return col, row

    def grid_to_cnn_cell(self, col: int, row: int) -> Tuple[int, int]:
        x_m, y_m = self.grid_to_world(col, row)
        return self.world_to_cnn_cell(x_m, y_m)

    # ---------------- Reset ----------------
    def reset_default(self) -> None:
        self.episode_seed = random.randint(0, 2**31 - 1)

        self._init_zones()
        self.manager.reset_game()
        self.mines.clear()
        self.mine_pickups.clear()

        self._init_macro_targets(seed=self.episode_seed)

        self.spawn_agents()
        self.spawn_mine_pickups()
        self.decision_cooldown_seconds_by_agent.clear()

        self.pending_external_actions.clear()

    def set_agent_count_and_reset(self, new_count: int) -> None:
        if new_count < 1:
            new_count = 1
        self.agents_per_team = new_count
        self.manager.reset_game(reset_scores=True)
        self.reset_default()

    # ---------------- Simulation step ----------------
    def update(self, delta_time: float) -> None:
        if self.manager.game_over:
            return
        if delta_time <= 0.0:
            return

        winner_text = self.manager.tick_seconds(delta_time)
        if winner_text:
            color = (
                (90, 170, 250) if "BLUE" in winner_text else
                (250, 120, 70) if "RED" in winner_text else
                (230, 230, 230)
            )
            self.announce(winner_text, color, 3.0)
            return

        for agent in (self.blue_agents + self.red_agents):
            agent.update(delta_time)

        # ✅ dynamic obstacles should be cell coords (NOT raw floats)
        occupied = [self._agent_cell_pos(a) for a in (self.blue_agents + self.red_agents) if a.isEnabled()]
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
                cooldown = float(self.decision_cooldown_seconds_by_agent.get(agent_key, 0.0))
                cooldown -= float(delta_time)

                side_external = self.external_control_for_side.get(agent.side, False)

                while cooldown <= 0.0:
                    made_decision = False

                    if side_external:
                        act = self._consume_external_action_for_agent(agent)
                        if act is not None:
                            macro_val, target_param = act
                            self.apply_macro_action(agent, int(macro_val), target_param)
                            made_decision = True
                        else:
                            if self.external_missing_action_mode == "internal" and self.use_internal_policies:
                                self.decide(agent)
                                made_decision = True
                            else:
                                break
                    else:
                        if self.use_internal_policies:
                            self.decide(agent)
                            made_decision = True

                    if made_decision:
                        cooldown += float(self.decision_interval_seconds)
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

    # ---------------- Observations ----------------
    def build_observation(self, agent: Agent) -> List[List[List[float]]]:
        side = agent.side
        gm = self.manager

        channels: List[List[List[float]]] = [
            [[0.0 for _ in range(CNN_COLS)] for _ in range(CNN_ROWS)]
            for _ in range(NUM_CNN_CHANNELS)
        ]

        def float_grid_to_world(fx: float, fy: float) -> Tuple[float, float]:
            x_m = (float(fx) + 0.5) * self.cell_width_m
            y_m = (float(fy) + 0.5) * self.cell_height_m
            return x_m, y_m

        def cnn_cell_from_float(fx: float, fy: float) -> Tuple[int, int]:
            x_m, y_m = float_grid_to_world(fx, fy)

            u = max(0.0, min(1.0, x_m / ARENA_WIDTH_M))
            v = max(0.0, min(1.0, y_m / ARENA_HEIGHT_M))

            # ✅ mirror for red so both teams see "enemy is to the right"
            if side == "red":
                u = 1.0 - u

            col_cnn = max(0, min(CNN_COLS - 1, int(u * CNN_COLS)))
            row_cnn = max(0, min(CNN_ROWS - 1, int(v * CNN_ROWS)))
            return col_cnn, row_cnn

        def set_chan(c: int, col: int, row: int) -> None:
            if 0 <= col < CNN_COLS and 0 <= row < CNN_ROWS:
                channels[c][row][col] = 1.0

        friendly_team = self.blue_agents if side == "blue" else self.red_agents
        enemy_team = self.red_agents if side == "blue" else self.blue_agents

        sx, sy = self._agent_float_pos(agent)
        sc, sr = cnn_cell_from_float(sx, sy)
        set_chan(0, sc, sr)

        for a in friendly_team:
            if a is agent or not a.isEnabled():
                continue
            fx, fy = self._agent_float_pos(a)
            c, r = cnn_cell_from_float(fx, fy)
            set_chan(1, c, r)

        for a in enemy_team:
            if not a.isEnabled():
                continue
            fx, fy = self._agent_float_pos(a)
            c, r = cnn_cell_from_float(fx, fy)
            set_chan(2, c, r)

        for m in self.mines:
            c, r = cnn_cell_from_float(float(m.x), float(m.y))
            if m.owner_side == side:
                set_chan(3, c, r)
            else:
                set_chan(4, c, r)

        if side == "blue":
            own_flag_pos = gm.blue_flag_position
            enemy_flag_pos = gm.get_enemy_flag_position("blue")
        else:
            own_flag_pos = gm.red_flag_position
            enemy_flag_pos = gm.get_enemy_flag_position("red")

        c1, r1 = cnn_cell_from_float(float(own_flag_pos[0]), float(own_flag_pos[1]))
        set_chan(5, c1, r1)

        c2, r2 = cnn_cell_from_float(float(enemy_flag_pos[0]), float(enemy_flag_pos[1]))
        set_chan(6, c2, r2)

        return channels

    # ---------------- Macro actions ----------------
    def apply_macro_action(self, agent: Agent, action: Any, param: Optional[Any] = None) -> MacroAction:
        """
        Executes a macro action. Accepts network macro index or MacroAction enum.
        Returns the MacroAction that was ACTUALLY executed (post safety/overrides).
        """
        if agent is None or (not agent.isEnabled()):
            return MacroAction.GO_TO

        macro_idx, action = self.normalize_macro(action)

        side = getattr(agent, "side", None)
        gm = self.manager

        # Safety: GET_FLAG while carrying becomes GO_HOME
        if action == MacroAction.GET_FLAG and agent.isCarryingFlag():
            action = MacroAction.GO_HOME

        # Record what we ACTUALLY executed (post safety/overrides)
        agent.last_macro_action = action
        try:
            agent.last_macro_action_idx = self.macro_action_to_idx(action)
        except Exception:
            agent.last_macro_action_idx = int(macro_idx)

        def _record_and_return(executed: MacroAction) -> MacroAction:
            agent.last_macro_action = executed
            try:
                agent.last_macro_action_idx = self.macro_action_to_idx(executed)
            except Exception:
                agent.last_macro_action_idx = int(self.normalize_macro(executed)[0])
            return executed

        def resolve_target_from_param(default_target: Tuple[int, int]) -> Tuple[int, int]:
            if param is None:
                return default_target
            if isinstance(param, (tuple, list)) and len(param) == 2:
                return self._clamp_cell(int(param[0]), int(param[1]))
            try:
                idx = int(param)
            except (TypeError, ValueError):
                return default_target
            return self.get_macro_target(idx)

        def safe_set_path(target: Tuple[int, int], avoid_enemies: bool = False, radius: int = 1) -> None:
            start = self._agent_cell_pos(agent)
            tgt = self._clamp_cell(int(target[0]), int(target[1]))

            danger_saved = dict(getattr(self.pathfinder, "danger_cost", {}))
            danger: Dict[Tuple[int, int], float] = {}

            if avoid_enemies:
                enemy_team = self.red_agents if side == "blue" else self.blue_agents
                base_penalty = 3.0
                max_penalty = 8.0

                for e in enemy_team:
                    if not e.isEnabled():
                        continue
                    ex, ey = self._agent_cell_pos(e)
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            cx, cy = ex + dx, ey + dy
                            if 0 <= cx < self.col_count and 0 <= cy < self.row_count:
                                d = max(abs(dx), abs(dy))
                                pen = base_penalty * float(radius + 1 - d)
                                if pen > 0:
                                    danger[(cx, cy)] = min(
                                        max_penalty,
                                        max(danger.get((cx, cy), 0.0), pen),
                                    )

                danger.pop(start, None)
                danger.pop(tgt, None)

            path = None  # prevents UnboundLocalError
            try:
                if avoid_enemies:
                    self.pathfinder.setDangerCosts(danger)
                else:
                    self.pathfinder.clearDangerCosts()
                path = self.pathfinder.astar(start, tgt)
            except Exception:
                path = None
            finally:
                self.pathfinder.setDangerCosts(danger_saved)

            if path is None:
                path = []
            agent.setPath(path)

        # -----------------------------
        # Macro execution
        # -----------------------------
        if action == MacroAction.GO_TO:
            default_target = self.get_macro_target(4)  # mid
            target = resolve_target_from_param(default_target)
            safe_set_path(target, avoid_enemies=agent.isCarryingFlag())
            return _record_and_return(MacroAction.GO_TO)

        if action == MacroAction.GRAB_MINE:
            # Objective-first guard: if you already have charges, do NOT grab more; go flag.
            if int(getattr(agent, "mine_charges", 0)) > 0:
                ex, ey = gm.get_enemy_flag_position(side)
                safe_set_path((ex, ey), avoid_enemies=False)
                return _record_and_return(MacroAction.GET_FLAG)

            # If close to enemy flag, do NOT detour for mines; go flag.
            ax, ay = self._agent_float_pos(agent)
            ex, ey = gm.get_enemy_flag_position(side)
            if math.hypot(float(ax) - float(ex), float(ay) - float(ey)) <= float(self.mine_detour_disable_radius_cells):
                safe_set_path((ex, ey), avoid_enemies=False)
                return _record_and_return(MacroAction.GET_FLAG)

            # Go to nearest friendly pickup (use cell coords, NOT agent.x/agent.y)
            my_pickups = [p for p in self.mine_pickups if p.owner_side == side]
            if my_pickups:
                axc, ayc = self._agent_cell_pos(agent)
                nearest = min(my_pickups, key=lambda p: (p.x - axc) ** 2 + (p.y - ayc) ** 2)
                target = (nearest.x, nearest.y)
            else:
                target = self.get_macro_target(2)  # own zone center as fallback

            safe_set_path(target, avoid_enemies=False)
            return _record_and_return(MacroAction.GRAB_MINE)

        if action == MacroAction.GET_FLAG:
            ex, ey = gm.get_enemy_flag_position(side)
            safe_set_path((ex, ey), avoid_enemies=False)
            return _record_and_return(MacroAction.GET_FLAG)

        if action == MacroAction.PLACE_MINE:
            default_target = self._agent_cell_pos(agent)
            target = resolve_target_from_param(default_target)

            if int(getattr(agent, "mine_charges", 0)) > 0:
                # Optional: keep mine placing defensive
                if not self.allow_offensive_mine_placing:
                    own_min, own_max = self.blue_zone_col_range if side == "blue" else self.red_zone_col_range
                    ax_cell, _ = self._agent_cell_pos(agent)
                    if not (own_min - 1 <= float(ax_cell) <= own_max + 1):
                        home = gm.get_team_zone_center(side)
                        safe_set_path(home, avoid_enemies=False)
                        return _record_and_return(MacroAction.GO_TO)

                own_flag_pos = gm.blue_flag_home if side == "blue" else gm.red_flag_home
                tx, ty = self._clamp_cell(int(target[0]), int(target[1]))

                if self._is_free_cell(tx, ty) and not any(m.x == tx and m.y == ty for m in self.mines):
                    if (tx, ty) != tuple(own_flag_pos):
                        self.mines.append(
                            Mine(
                                x=tx,
                                y=ty,
                                owner_side=side,
                                owner_id=getattr(agent, "unique_id", None),
                            )
                        )
                        agent.mine_charges -= 1
                        if hasattr(self.manager, "reward_mine_placed"):
                            try:
                                self.manager.reward_mine_placed(agent, mine_pos=(tx, ty))
                            except Exception:
                                pass

            # Move to target regardless (helps if you decided to place at a spot)
            safe_set_path(target, avoid_enemies=False)
            return _record_and_return(MacroAction.PLACE_MINE)

        if action == MacroAction.GO_HOME:
            home = gm.get_team_zone_center(side)
            safe_set_path(home, avoid_enemies=True, radius=2)
            return _record_and_return(MacroAction.GO_HOME)

        # Fallback: treat unknown macro as GO_TO mid
        target = self.get_macro_target(4)
        safe_set_path(target, avoid_enemies=False)
        return _record_and_return(MacroAction.GO_TO)

    # ---------------- Internal decision ----------------
    def decide(self, agent: Agent) -> None:
        if not agent.isEnabled():
            return

        agent.decision_count = getattr(agent, "decision_count", 0) + 1

        obs = self.build_observation(agent)
        policy = self.policies.get(agent.side)

        if hasattr(policy, "act"):
            device = torch.device("cpu")
            if hasattr(policy, "parameters"):
                try:
                    p = next(policy.parameters())
                    device = p.device
                except StopIteration:
                    device = torch.device("cpu")

            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                out = policy.act(obs_tensor, agent=agent, game_field=self, deterministic=True)

            if isinstance(out, dict):
                action_id = int(out["macro_action"][0].item())
                param = int(out["target_action"][0].item())
            else:
                action_id = int(out[0])
                param = int(out[1]) if len(out) > 1 else None

        elif isinstance(policy, Policy):
            action_id, param = policy.select_action(obs, agent, self)
        else:
            action_id = policy(agent, self)
            param = None

        self.apply_macro_action(agent, int(action_id), param)

    # ---------------- Mines / suppression / flags ----------------
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
                    except TypeError:
                        try:
                            self.manager.reward_mine_picked_up(agent, prev)
                        except Exception:
                            pass
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
                    for a in (self.blue_agents + self.red_agents):
                        if getattr(a, "unique_id", None) == mine.owner_id:
                            killer_agent = a
                            break

                if killer_agent is None and mine.owner_side == "blue":
                    killer_agent = next((a for a in self.blue_agents if getattr(a, "is_miner", False)), None)
                    if killer_agent is None and self.blue_agents:
                        killer_agent = self.blue_agents[0]

                if killer_agent is not None and getattr(killer_agent, "side", None) == "blue":
                    if hasattr(self.manager, "reward_enemy_killed"):
                        self.manager.reward_enemy_killed(killer_agent=killer_agent, victim_agent=agent, cause="mine")

                if mine.owner_side == "blue" and agent.side == "red":
                    if hasattr(self.manager, "record_mine_triggered_by_red"):
                        self.manager.record_mine_triggered_by_red()

                self._clear_pending_for_agent(agent)

                agent.disable_for_seconds(self.respawn_seconds)
                self.mines.remove(mine)
                break

    def apply_suppression(self, agent: Agent, enemies: List[Agent], delta_time: float) -> None:
        if not agent.isEnabled():
            return

        ax, ay = self._agent_float_pos(agent)

        close_enemies = []
        for e in enemies:
            if not e.isEnabled():
                continue
            ex, ey = self._agent_float_pos(e)
            if math.hypot(ex - ax, ey - ay) <= float(self.suppression_range_cells):
                close_enemies.append(e)

        if len(close_enemies) >= 2:
            agent.suppressed_this_tick = True
            agent.suppression_timer = float(getattr(agent, "suppression_timer", 0.0)) + float(delta_time)

            if agent.suppression_timer >= 1.0:
                blue_suppressors = [e for e in close_enemies if getattr(e, "side", None) == "blue"]
                killer_agent = blue_suppressors[0] if blue_suppressors else close_enemies[0]

                if killer_agent is not None and getattr(killer_agent, "side", None) == "blue":
                    if hasattr(self.manager, "reward_enemy_killed"):
                        self.manager.reward_enemy_killed(killer_agent=killer_agent, victim_agent=agent, cause="suppression")

                self._clear_pending_for_agent(agent)
                agent.suppression_timer = 0.0
                agent.disable_for_seconds(self.respawn_seconds)
        else:
            # ✅ keep flags consistent for Agent.update() consumers
            agent.suppressed_this_tick = False

    def apply_flag_rules(self, agent: Agent) -> None:
        if self.manager.try_pickup_enemy_flag(agent):
            if not agent.isCarryingFlag():
                agent.setCarryingFlag(True)

        if agent.isCarryingFlag():
            if self.manager.try_score_if_carrying_and_home(agent):
                agent.setCarryingFlag(False, scored=True)
                self.announce(
                    "BLUE SCORES!" if agent.side == "blue" else "RED SCORES!",
                    (90, 170, 250) if agent.side == "blue" else (250, 120, 70),
                    2.0,
                )

    # ---------------- UI helpers ----------------
    def announce(self, text: str, color=(255, 255, 255), seconds: float = 2.0) -> None:
        self.banner_queue.append((text, color, seconds))

    # ---------------- Spawning ----------------
    def spawn_agents(self) -> None:
        base = int(self.episode_seed or 0)
        rng = random.Random(base + 123)

        self.blue_agents.clear()
        self.red_agents.clear()
        self.mines.clear()
        self.mine_pickups.clear()
        self.decision_cooldown_seconds_by_agent.clear()
        self.pending_external_actions.clear()

        blue_min_col, blue_max_col = self.blue_zone_col_range
        red_min_col, red_max_col = self.red_zone_col_range

        # ✅ only free cells
        blue_cells = [(r, c) for r in range(self.row_count) for c in range(blue_min_col, blue_max_col + 1) if self._is_free_cell(c, r)]
        red_cells = [(r, c) for r in range(self.row_count) for c in range(red_min_col, red_max_col + 1) if self._is_free_cell(c, r)]
        rng.shuffle(blue_cells)
        rng.shuffle(red_cells)

        n = self.agents_per_team

        for i in range(min(n, len(blue_cells))):
            row, col = blue_cells[i]
            agent = Agent(
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
            agent.spawn_xy = (col, row)
            agent.game_field = self
            agent.mine_charges = 0
            agent.decision_count = 0
            agent.suppression_timer = 0.0
            agent.suppressed_last_tick = False
            agent.suppressed_this_tick = False
            self.blue_agents.append(agent)

        for i in range(min(n, len(red_cells))):
            row, col = red_cells[i]
            agent = Agent(
                x=col,
                y=row,
                side="red",
                cols=self.col_count,
                rows=self.row_count,
                grid=self.grid,
                move_rate_cps=rng.uniform(2.0, 2.4),
                agent_id=i,
                is_miner=(i == 0),
                game_manager=self.manager,
            )
            agent.spawn_xy = (col, row)
            agent.game_field = self
            agent.mine_charges = 0
            agent.decision_count = 0
            agent.suppression_timer = 0.0
            agent.suppressed_last_tick = False
            agent.suppressed_this_tick = False
            self.red_agents.append(agent)

    def spawn_mine_pickups(self) -> None:
        self.mine_pickups.clear()
        base = int(self.episode_seed or 0)
        rng = random.Random(base + 9999)

        blue_min_col, blue_max_col = self.blue_zone_col_range
        red_min_col, red_max_col = self.red_zone_col_range

        occupied_spawns = set()
        for a in (self.blue_agents + self.red_agents):
            occupied_spawns.add(self._agent_cell_pos(a))

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


__all__ = ["GameField", "MacroAction", "Mine", "MinePickup"]
