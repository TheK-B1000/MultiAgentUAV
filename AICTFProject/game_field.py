# =========================
# game_field.py (REFACTORED, MARL-READY, OBJECTIVE-FIRST, CRASH-PROOF)
# =========================

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np  # action masks

from agents import Agent
from game_manager import GameManager
from macro_actions import MacroAction
from pathfinder import Pathfinder
from policies import OP1RedPolicy, OP2RedPolicy, OP3RedPolicy, Policy

# -------------------------
# Constants / configuration
# -------------------------

ARENA_WIDTH_M = 10.0
ARENA_HEIGHT_M = 4.28

CNN_COLS = 20
CNN_ROWS = 20
NUM_CNN_CHANNELS = 7

Cell = Tuple[int, int]
FloatPos = Tuple[float, float]
ExternalAction = Tuple[int, Any]  # (macro_idx, target_param)


# -------------------------
# Simple entities
# -------------------------

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


# -------------------------
# GameField
# -------------------------

class GameField:
    """
    2D Capture-the-Flag (CTF) simulation environment for multi-agent RL.

    Design principles:
      - Macro-actions + safety constraints (crash-proof, no undefined behavior).
      - Float positions for distance-based logic (mines, suppression).
      - Cell positions for pathfinding and occupancy/dynamic obstacle logic.
      - Trainer-facing action masking (logits-time masking recommended).
      - External control supported via submit_external_actions().
    """

    # -------- lifecycle --------

    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.row_count = len(grid)
        self.col_count = len(grid[0]) if self.row_count else 0

        self.manager = GameManager(cols=self.col_count, rows=self.row_count)
        self.manager.bind_game_field(self)  # enables exact reward routing

        self._init_macro_indexing()
        self._init_zones()

        # Core game mechanics
        self.mines: List[Mine] = []
        self.mine_pickups: List[MinePickup] = []
        self.mine_radius_cells = 1.5
        self.suppression_range_cells = 2.0
        self.mines_per_team = 4
        self.max_mine_charges_per_agent = 2

        # Objective-first knobs
        self.mine_detour_disable_radius_cells: float = 8.0
        self.allow_offensive_mine_placing: bool = False

        # Policy wiring
        self.use_internal_policies: bool = True
        self.policies: Dict[str, Any] = {"blue": None, "red": None}
        self.opponent_mode: str = "OP3"

        # External action routing (trainer-controlled)
        self.external_control_for_side: Dict[str, bool] = {"blue": False, "red": False}
        self.pending_external_actions: Dict[str, ExternalAction] = {}
        self.external_missing_action_mode: str = "idle"  # "idle" or "internal"

        # Agents
        self.agents_per_team: int = 2
        self.blue_agents: List[Agent] = []
        self.red_agents: List[Agent] = []

        # Pathfinding
        self.pathfinder = Pathfinder(
            self.grid,
            self.row_count,
            self.col_count,
            allow_diagonal=True,
            block_corners=True,
        )

        # Spatial mapping
        self.cell_width_m = ARENA_WIDTH_M / max(1, self.col_count)
        self.cell_height_m = ARENA_HEIGHT_M / max(1, self.row_count)

        # Semantic macro targets (stable list)
        self.num_macro_targets: int = 8
        self.macro_targets: List[Cell] = []

        # Episode timing
        self.episode_seed: Optional[int] = None
        self.respawn_seconds: float = 2.0
        self.decision_interval_seconds: float = 0.7
        self.decision_cooldown_seconds_by_agent: Dict[int, float] = {}

        # Optional minimal UI hook (safe to ignore for training)
        self.banner_queue: List[Tuple[str, Tuple[int, int, int], float]] = []

        # Default opponent (safe, explicit)
        self.set_red_opponent("OP3")
        self.reset_default()

    # -------- macro indexing --------

    def _init_macro_indexing(self) -> None:
        """
        Canonical mapping between network macro indices and MacroAction enum.
        Never assumes enum values are contiguous or 0..N-1.
        """
        self.macro_order = [
            MacroAction.GO_TO,       # idx 0
            MacroAction.GRAB_MINE,   # idx 1
            MacroAction.GET_FLAG,    # idx 2
            MacroAction.PLACE_MINE,  # idx 3
            MacroAction.GO_HOME,     # idx 4
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
          - network index (0..n-1)
          - MacroAction enum
          - anything int-castable (treated as network index)
        Returns:
          (macro_idx, MacroAction)
        """
        if isinstance(action_any, MacroAction):
            return self.macro_action_to_idx(action_any), action_any

        try:
            idx = int(action_any)
        except Exception:
            idx = 0
        return idx % self.n_macros, self.macro_idx_to_action(idx)

    # -------- small utilities --------

    def _is_free_cell(self, x: int, y: int) -> bool:
        return (
            0 <= x < self.col_count
            and 0 <= y < self.row_count
            and int(self.grid[y][x]) == 0
        )

    def _clamp_cell(self, x: int, y: int) -> Cell:
        x = int(max(0, min(self.col_count - 1, x)))
        y = int(max(0, min(self.row_count - 1, y)))
        return x, y

    def _agent_cell_pos(self, agent: Any) -> Cell:
        cp = getattr(agent, "cell_pos", None)
        if isinstance(cp, (tuple, list)) and len(cp) >= 2:
            try:
                return self._clamp_cell(int(cp[0]), int(cp[1]))
            except Exception:
                pass

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

        return 0, 0

    def _agent_float_pos(self, agent: Any) -> FloatPos:
        fp = getattr(agent, "float_pos", None)
        if isinstance(fp, (tuple, list)) and len(fp) >= 2:
            try:
                return float(fp[0]), float(fp[1])
            except Exception:
                pass
        try:
            return float(getattr(agent, "x", 0.0)), float(getattr(agent, "y", 0.0))
        except Exception:
            return 0.0, 0.0

    # -------- external action routing --------

    def _external_key_candidates(self, agent: Agent) -> List[str]:
        """
        Generates multiple stable keys to match trainer submissions robustly.
        """
        keys: List[str] = []
        # Prefer explicit slots/unique ids if present
        for attr in ("slot_id", "unique_id"):
            if hasattr(agent, attr):
                try:
                    keys.append(str(getattr(agent, attr)))
                except Exception:
                    pass
        # Always include side+agent_id canonical form
        keys.append(f"{agent.side}_{getattr(agent, 'agent_id', 0)}")

        out: List[str] = []
        seen = set()
        for k in keys:
            if k and k not in seen:
                out.append(k)
                seen.add(k)
        return out

    def submit_external_actions(self, actions_by_agent: Dict[str, ExternalAction]) -> None:
        """
        Trainer-facing API: submit actions for the next decision boundary.

        actions_by_agent:
          key -> (macro_idx, target_param)
        where target_param may be:
          - int target index (semantic macro target list)
          - (x,y) cell tuple/list
        """
        if not isinstance(actions_by_agent, dict):
            return

        for k, v in actions_by_agent.items():
            try:
                macro_val, target_param = v
                self.pending_external_actions[str(k)] = (int(macro_val), target_param)
            except Exception:
                continue

    def _consume_external_action_for_agent(self, agent: Agent) -> Optional[ExternalAction]:
        for k in self._external_key_candidates(agent):
            if k in self.pending_external_actions:
                return self.pending_external_actions.pop(k, None)
        return None

    def _clear_pending_for_agent(self, agent: Agent) -> None:
        for k in self._external_key_candidates(agent):
            self.pending_external_actions.pop(k, None)

    # -------- zones / targets --------

    def _init_zones(self) -> None:
        total_cols = max(1, self.col_count)
        third = max(1, total_cols // 3)

        blue_min = 0
        blue_max = max(blue_min, third - 1)

        red_max = total_cols - 1
        red_min = min(total_cols - third, red_max)

        self.blue_zone_col_range = (blue_min, blue_max)
        self.red_zone_col_range = (red_min, red_max)

    def _init_macro_targets(self) -> None:
        """
        Fixed semantic waypoints. Kept stable across episodes for reproducibility.
        """
        self.macro_targets.clear()

        def nearest_free(x: int, y: int, radius: int = 8) -> Cell:
            x, y = self._clamp_cell(x, y)
            if self._is_free_cell(x, y):
                return (x, y)

            best: Optional[Cell] = None
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

        targets_raw: List[Cell] = [
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
        self.num_macro_targets = len(self.macro_targets)

    def get_macro_target(self, index: int) -> Cell:
        if not self.macro_targets:
            self._init_macro_targets()
        if self.num_macro_targets <= 0:
            return (self.col_count // 2, self.row_count // 2)
        return self.macro_targets[int(index) % self.num_macro_targets]

    def get_all_macro_targets(self) -> List[Cell]:
        if not self.macro_targets:
            self._init_macro_targets()
        return list(self.macro_targets)

    # -------- public helpers --------

    def getGameManager(self) -> GameManager:
        return self.manager

    # -------- action masking --------

    def get_macro_mask(self, agent: Agent) -> np.ndarray:
        """
        Boolean mask over network macro indices (0..n_macros-1) matching self.macro_order.

        Trainer should apply this at logits-time (set invalid logits to -inf / very negative).
        """
        n = self.n_macros
        mask = np.ones((n,), dtype=np.bool_)

        if agent is None or (not agent.isEnabled()):
            mask[:] = False
            return mask

        side = getattr(agent, "side", None)
        if side not in ("blue", "red"):
            mask[self.macro_action_to_idx(MacroAction.GRAB_MINE)] = False
            mask[self.macro_action_to_idx(MacroAction.PLACE_MINE)] = False
            return mask

        gm = self.manager
        phase = str(getattr(gm, "phase_name", "OP1")).upper()

        idx_grab = self.macro_action_to_idx(MacroAction.GRAB_MINE)
        idx_get = self.macro_action_to_idx(MacroAction.GET_FLAG)
        idx_place = self.macro_action_to_idx(MacroAction.PLACE_MINE)
        idx_home = self.macro_action_to_idx(MacroAction.GO_HOME)

        # Carrying flag: only GO_HOME / GO_TO allowed
        if agent.isCarryingFlag():
            mask[idx_get] = False
            mask[idx_grab] = False
            mask[idx_place] = False
            mask[idx_home] = True
            return mask

        # OP1: hard-disable mines to lock onto objective
        if phase == "OP1":
            mask[idx_grab] = False
            mask[idx_place] = False

        charges = int(getattr(agent, "mine_charges", 0))

        # Never grab mine if already holding charges
        if charges > 0:
            mask[idx_grab] = False

        # Never detour for mines near enemy flag
        ax, ay = self._agent_float_pos(agent)
        ex, ey = gm.get_enemy_flag_position(side)
        if math.hypot(ax - float(ex), ay - float(ey)) <= float(self.mine_detour_disable_radius_cells):
            mask[idx_grab] = False

        # Place mine only if have charges, and optionally only in own zone
        if charges <= 0:
            mask[idx_place] = False
        else:
            if not self.allow_offensive_mine_placing:
                own_min, own_max = self.blue_zone_col_range if side == "blue" else self.red_zone_col_range
                if not (own_min - 1 <= ax <= own_max + 1):
                    mask[idx_place] = False

        return mask

    # -------- policy wiring --------

    def set_policies(self, blue: Any, red: Any) -> None:
        self.policies["blue"] = blue
        self.policies["red"] = red

    def set_red_opponent(self, mode: str) -> None:
        mode = str(mode).upper()
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
        self.external_control_for_side[side] = bool(external)

    def set_all_external_control(self, external: bool) -> None:
        for side in ("blue", "red"):
            self.external_control_for_side[side] = bool(external)

    # -------- coordinate mapping --------

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        return (col + 0.5) * self.cell_width_m, (row + 0.5) * self.cell_height_m

    def world_to_cnn_cell(self, x_m: float, y_m: float, *, mirror_x: bool) -> Cell:
        u = max(0.0, min(1.0, x_m / ARENA_WIDTH_M))
        v = max(0.0, min(1.0, y_m / ARENA_HEIGHT_M))
        if mirror_x:
            u = 1.0 - u

        col = max(0, min(CNN_COLS - 1, int(u * CNN_COLS)))
        row = max(0, min(CNN_ROWS - 1, int(v * CNN_ROWS)))
        return col, row

    def float_grid_to_cnn_cell(self, fx: float, fy: float, *, mirror_x: bool) -> Cell:
        x_m = (float(fx) + 0.5) * self.cell_width_m
        y_m = (float(fy) + 0.5) * self.cell_height_m
        return self.world_to_cnn_cell(x_m, y_m, mirror_x=mirror_x)

    # -------- reset / spawn --------

    def reset_default(self) -> None:
        self.episode_seed = random.randint(0, 2**31 - 1)

        self._init_zones()
        self.manager.reset_game()

        self.mines.clear()
        self.mine_pickups.clear()
        self.pending_external_actions.clear()
        self.decision_cooldown_seconds_by_agent.clear()

        self._init_macro_targets()

        self.spawn_agents()
        self.spawn_mine_pickups()

    def set_agent_count_and_reset(self, new_count: int) -> None:
        self.agents_per_team = max(1, int(new_count))
        self.manager.reset_game(reset_scores=True)
        self.reset_default()

    def spawn_agents(self) -> None:
        base = int(self.episode_seed or 0)
        rng = random.Random(base + 123)

        self.blue_agents.clear()
        self.red_agents.clear()

        blue_min_col, blue_max_col = self.blue_zone_col_range
        red_min_col, red_max_col = self.red_zone_col_range

        blue_cells = [
            (r, c)
            for r in range(self.row_count)
            for c in range(blue_min_col, blue_max_col + 1)
            if self._is_free_cell(c, r)
        ]
        red_cells = [
            (r, c)
            for r in range(self.row_count)
            for c in range(red_min_col, red_max_col + 1)
            if self._is_free_cell(c, r)
        ]
        rng.shuffle(blue_cells)
        rng.shuffle(red_cells)

        n = self.agents_per_team

        for i in range(min(n, len(blue_cells))):
            row, col = blue_cells[i]
            a = Agent(
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
            a.spawn_xy = (col, row)
            a.game_field = self
            a.mine_charges = 0
            a.decision_count = 0
            a.suppression_timer = 0.0
            a.suppressed_this_tick = False
            self.blue_agents.append(a)

        for i in range(min(n, len(red_cells))):
            row, col = red_cells[i]
            a = Agent(
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
            a.spawn_xy = (col, row)
            a.game_field = self
            a.mine_charges = 0
            a.decision_count = 0
            a.suppression_timer = 0.0
            a.suppressed_this_tick = False
            self.red_agents.append(a)

    def spawn_mine_pickups(self) -> None:
        self.mine_pickups.clear()

        base = int(self.episode_seed or 0)
        rng = random.Random(base + 9999)

        blue_min_col, blue_max_col = self.blue_zone_col_range
        red_min_col, red_max_col = self.red_zone_col_range

        occupied_spawns = {self._agent_cell_pos(a) for a in (self.blue_agents + self.red_agents)}

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

    # -------- main simulation step --------

    def update(self, delta_time: float) -> None:
        if self.manager.game_over or delta_time <= 0.0:
            return

        winner_text = self.manager.tick_seconds(delta_time)
        if winner_text:
            color = (230, 230, 230)
            if "BLUE" in winner_text:
                color = (90, 170, 250)
            elif "RED" in winner_text:
                color = (250, 120, 70)
            self.announce(winner_text, color, 3.0)
            return

        # Update continuous motion
        for agent in (self.blue_agents + self.red_agents):
            agent.update(delta_time)

        # Dynamic obstacles from *cell* positions
        occupied = [self._agent_cell_pos(a) for a in (self.blue_agents + self.red_agents) if a.isEnabled()]
        self.pathfinder.setDynamicObstacles(occupied)

        # Apply mechanics + decisions
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
                cooldown = float(self.decision_cooldown_seconds_by_agent.get(agent_key, 0.0)) - float(delta_time)

                side_external = self.external_control_for_side.get(agent.side, False)

                while cooldown <= 0.0:
                    made_decision = False

                    if side_external:
                        act = self._consume_external_action_for_agent(agent)
                        if act is not None:
                            macro_val, target_param = act
                            self.apply_macro_action(agent, macro_val, target_param)
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

        # Minimal banner decay (safe to ignore)
        if self.banner_queue:
            text, color, t = self.banner_queue[-1]
            t = max(0.0, t - delta_time)
            self.banner_queue[-1] = (text, color, t)
            if t <= 0.0:
                self.banner_queue.pop()

    # -------- observations --------

    def build_observation(self, agent: Agent) -> List[List[List[float]]]:
        """
        Returns a [C][H][W] nested list for CNN-style policies.
        Mirrors X for red so both teams see "enemy direction" consistently.
        """
        side = agent.side
        gm = self.manager
        mirror_x = (side == "red")

        channels = [
            [[0.0 for _ in range(CNN_COLS)] for _ in range(CNN_ROWS)]
            for _ in range(NUM_CNN_CHANNELS)
        ]

        def set_chan(c: int, col: int, row: int) -> None:
            if 0 <= col < CNN_COLS and 0 <= row < CNN_ROWS:
                channels[c][row][col] = 1.0

        friendly_team = self.blue_agents if side == "blue" else self.red_agents
        enemy_team = self.red_agents if side == "blue" else self.blue_agents

        sx, sy = self._agent_float_pos(agent)
        sc, sr = self.float_grid_to_cnn_cell(sx, sy, mirror_x=mirror_x)
        set_chan(0, sc, sr)

        for a in friendly_team:
            if a is agent or not a.isEnabled():
                continue
            fx, fy = self._agent_float_pos(a)
            c, r = self.float_grid_to_cnn_cell(fx, fy, mirror_x=mirror_x)
            set_chan(1, c, r)

        for a in enemy_team:
            if not a.isEnabled():
                continue
            fx, fy = self._agent_float_pos(a)
            c, r = self.float_grid_to_cnn_cell(fx, fy, mirror_x=mirror_x)
            set_chan(2, c, r)

        for m in self.mines:
            c, r = self.float_grid_to_cnn_cell(float(m.x), float(m.y), mirror_x=mirror_x)
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

        c1, r1 = self.float_grid_to_cnn_cell(float(own_flag_pos[0]), float(own_flag_pos[1]), mirror_x=mirror_x)
        set_chan(5, c1, r1)

        c2, r2 = self.float_grid_to_cnn_cell(float(enemy_flag_pos[0]), float(enemy_flag_pos[1]), mirror_x=mirror_x)
        set_chan(6, c2, r2)

        return channels

    # -------- macro actions --------

    def apply_macro_action(self, agent: Agent, action: Any, param: Optional[Any] = None) -> MacroAction:
        """
        Executes a macro action. Accepts network macro index or MacroAction enum.
        Returns the MacroAction actually executed (post-safety overrides).
        """
        if agent is None or (not agent.isEnabled()):
            return MacroAction.GO_TO

        macro_idx, action = self.normalize_macro(action)
        gm = self.manager
        side = getattr(agent, "side", None)

        # Safety override: GET_FLAG while carrying becomes GO_HOME
        if action == MacroAction.GET_FLAG and agent.isCarryingFlag():
            action = MacroAction.GO_HOME

        def record(executed: MacroAction) -> MacroAction:
            agent.last_macro_action = executed
            try:
                agent.last_macro_action_idx = self.macro_action_to_idx(executed)
            except Exception:
                agent.last_macro_action_idx = int(macro_idx)
            return executed

        def resolve_target(default_target: Cell) -> Cell:
            if param is None:
                return default_target
            if isinstance(param, (tuple, list)) and len(param) == 2:
                return self._clamp_cell(int(param[0]), int(param[1]))
            try:
                idx = int(param)
            except (TypeError, ValueError):
                return default_target
            return self.get_macro_target(idx)

        def safe_set_path(target: Cell, *, avoid_enemies: bool, radius: int = 1) -> None:
            start = self._agent_cell_pos(agent)
            tgt = self._clamp_cell(int(target[0]), int(target[1]))

            danger_saved = dict(getattr(self.pathfinder, "danger_cost", {}))
            danger: Dict[Cell, float] = {}

            if avoid_enemies and side in ("blue", "red"):
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
                                    danger[(cx, cy)] = min(max_penalty, max(danger.get((cx, cy), 0.0), pen))

                danger.pop(start, None)
                danger.pop(tgt, None)

            path: Optional[List[Cell]] = None
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

            agent.setPath(path or [])

        # ---- dispatch ----

        if action == MacroAction.GO_TO:
            target = resolve_target(self.get_macro_target(4))  # mid
            safe_set_path(target, avoid_enemies=agent.isCarryingFlag())
            return record(MacroAction.GO_TO)

        if action == MacroAction.GET_FLAG:
            ex, ey = gm.get_enemy_flag_position(side)
            safe_set_path((int(ex), int(ey)), avoid_enemies=False)
            return record(MacroAction.GET_FLAG)

        if action == MacroAction.GO_HOME:
            home = gm.get_team_zone_center(side)
            safe_set_path(home, avoid_enemies=True, radius=2)
            return record(MacroAction.GO_HOME)

        if action == MacroAction.GRAB_MINE:
            # Objective-first guards
            if int(getattr(agent, "mine_charges", 0)) > 0:
                ex, ey = gm.get_enemy_flag_position(side)
                safe_set_path((int(ex), int(ey)), avoid_enemies=False)
                return record(MacroAction.GET_FLAG)

            ax, ay = self._agent_float_pos(agent)
            ex, ey = gm.get_enemy_flag_position(side)
            if math.hypot(ax - float(ex), ay - float(ey)) <= float(self.mine_detour_disable_radius_cells):
                safe_set_path((int(ex), int(ey)), avoid_enemies=False)
                return record(MacroAction.GET_FLAG)

            my_pickups = [p for p in self.mine_pickups if p.owner_side == side]
            if my_pickups:
                axc, ayc = self._agent_cell_pos(agent)
                nearest = min(my_pickups, key=lambda p: (p.x - axc) ** 2 + (p.y - ayc) ** 2)
                target = (nearest.x, nearest.y)
            else:
                target = self.get_macro_target(2)  # own zone center fallback

            safe_set_path(target, avoid_enemies=False)
            return record(MacroAction.GRAB_MINE)

        if action == MacroAction.PLACE_MINE:
            target = resolve_target(self._agent_cell_pos(agent))

            if int(getattr(agent, "mine_charges", 0)) > 0:
                if not self.allow_offensive_mine_placing and side in ("blue", "red"):
                    own_min, own_max = self.blue_zone_col_range if side == "blue" else self.red_zone_col_range
                    ax_cell, _ = self._agent_cell_pos(agent)
                    if not (own_min - 1 <= float(ax_cell) <= own_max + 1):
                        safe_set_path(gm.get_team_zone_center(side), avoid_enemies=False)
                        return record(MacroAction.GO_TO)

                tx, ty = self._clamp_cell(int(target[0]), int(target[1]))
                own_flag_home = gm.blue_flag_home if side == "blue" else gm.red_flag_home

                if self._is_free_cell(tx, ty) and not any(m.x == tx and m.y == ty for m in self.mines):
                    if (tx, ty) != tuple(own_flag_home):
                        self.mines.append(Mine(x=tx, y=ty, owner_side=side, owner_id=getattr(agent, "unique_id", None)))
                        agent.mine_charges -= 1
                        if hasattr(self.manager, "reward_mine_placed"):
                            try:
                                self.manager.reward_mine_placed(agent, mine_pos=(tx, ty))
                            except Exception:
                                pass

            safe_set_path(target, avoid_enemies=False)
            return record(MacroAction.PLACE_MINE)

        # Fallback: treat unknown macro as GO_TO mid
        safe_set_path(self.get_macro_target(4), avoid_enemies=False)
        return record(MacroAction.GO_TO)

    # -------- internal decision (scripted or neural policy) --------

    def decide(self, agent: Agent) -> None:
        if not agent.isEnabled():
            return

        agent.decision_count = getattr(agent, "decision_count", 0) + 1
        obs = self.build_observation(agent)
        policy = self.policies.get(agent.side)

        # Neural-like policy interface: policy.act(obs_tensor, ...)
        if hasattr(policy, "act"):
            # Local import so game_field stays light when not using torch-based policies
            import torch  # noqa: PLC0415

            device = torch.device("cpu")
            if hasattr(policy, "parameters"):
                try:
                    p = next(policy.parameters())
                    device = p.device
                except StopIteration:
                    pass

            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                out = policy.act(obs_tensor, agent=agent, game_field=self, deterministic=True)

            if isinstance(out, dict):
                action_id = int(out["macro_action"][0].item())
                param = int(out["target_action"][0].item())
            else:
                action_id = int(out[0])
                param = int(out[1]) if len(out) > 1 else None

            self.apply_macro_action(agent, action_id, param)
            return

        # Scripted Policy type
        if isinstance(policy, Policy):
            action_id, param = policy.select_action(obs, agent, self)
            self.apply_macro_action(agent, int(action_id), param)
            return

        # Callable fallback (rare)
        if callable(policy):
            action_id = policy(agent, self)
            self.apply_macro_action(agent, int(action_id), None)
            return

        # If no policy wired: do nothing (stable training behavior)
        return

    # -------- mechanics: mines / suppression / flags --------

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
        # Never operate on disabled agents
        if agent is None or (not agent.isEnabled()):
            return

        ax, ay = self._agent_float_pos(agent)

        close_enemies: List[Agent] = []
        rng = float(getattr(self, "suppression_range_cells", 0.0))
        for e in enemies:
            if e is None or (not e.isEnabled()):
                continue
            ex, ey = self._agent_float_pos(e)
            if math.hypot(ex - ax, ey - ay) <= rng:
                close_enemies.append(e)

        if len(close_enemies) >= 2:
            agent.suppressed_this_tick = True
            agent.suppression_timer = float(getattr(agent, "suppression_timer", 0.0)) + float(delta_time)

            # Suppression "kill" threshold
            if agent.suppression_timer >= 1.0:
                # Pick a deterministic killer (prefer blue if present, else first)
                killer_agent = None
                blue_suppressors = [e for e in close_enemies if getattr(e, "side", None) == "blue"]
                killer_agent = blue_suppressors[0] if blue_suppressors else close_enemies[0]

                # Reward only if the method exists (no missing-method crashes)
                mgr = getattr(self, "manager", None)
                if (
                        killer_agent is not None
                        and getattr(killer_agent, "side", None) == "blue"
                        and mgr is not None
                        and hasattr(mgr, "reward_enemy_killed")
                ):
                    mgr.reward_enemy_killed(killer_agent=killer_agent, victim_agent=agent, cause="suppression")

                # Clear any pending action state safely
                if hasattr(self, "_clear_pending_for_agent"):
                    self._clear_pending_for_agent(agent)

                agent.suppression_timer = 0.0
                agent.disable_for_seconds(float(getattr(self, "respawn_seconds", 0.0)))
        else:
            # Not currently suppressed: reset cleanly to avoid "stutter kills"
            agent.suppressed_this_tick = False
            agent.suppression_timer = 0.0

    def apply_flag_rules(self, agent: Agent) -> None:
        """
        Flag pickup and scoring.
        Uses manager as source of truth.
        """
        if self.manager.try_pickup_enemy_flag(agent):
            agent.setCarryingFlag(True)

        if agent.isCarryingFlag():
            if self.manager.try_score_if_carrying_and_home(agent):
                agent.setCarryingFlag(False, scored=True)
                self.announce(
                    "BLUE SCORES!" if agent.side == "blue" else "RED SCORES!",
                    (90, 170, 250) if agent.side == "blue" else (250, 120, 70),
                    2.0,
                )

    # -------- minimal UI hook --------

    def announce(self, text: str, color: Tuple[int, int, int] = (255, 255, 255), seconds: float = 2.0) -> None:
        self.banner_queue.append((str(text), color, float(seconds)))


__all__ = ["GameField", "MacroAction", "Mine", "MinePickup"]
