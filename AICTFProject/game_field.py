import random
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import pygame as pg
import numpy as np

from game_manager import GameManager
from agents import Agent, TEAM_ZONE_RADIUS_CELLS
from pathfinder import Pathfinder

from macro_actions import MacroAction
from policies import Policy, OP1RedPolicy, OP2RedPolicy, OP3RedPolicy


# Mines & pickups
@dataclass
class Mine:
    x: int
    y: int
    owner_side: str  # "blue" or "red"
    owner_id: Optional[str] = None  # unique_id of placing agent (if known)


@dataclass
class MinePickup:
    x: int
    y: int
    owner_side: str  # "blue" or "red"
    charges: int = 1


# GameField — main 2D simulation environment
class GameField:
    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.row_count = len(grid)
        self.col_count = len(grid[0]) if self.row_count else 0

        self.manager = GameManager(cols=self.col_count, rows=self.row_count)

        # Mines / pickups
        self.mines: List[Mine] = []
        self.mine_pickups: List[MinePickup] = []
        self.mine_radius_cells = 1.5
        self.suppression_range_cells = 2.0
        self.mines_per_team = 4
        self.max_mine_charges_per_agent = 2

        # Policy wiring
        self.use_internal_policies = True
        self.external_control_for_side = {"blue": False, "red": False}

        # By default, BOTH teams use the paper's OP3-style scripted policy
        self.policies: Dict[str, Any] = {
            "blue": OP3RedPolicy("blue"),  # baseline scripted OP3 on BLUE side
            "red": OP3RedPolicy("red"),    # default opponent: OP3
        }
        self.opponent_mode: str = "OP3"

        # Zones & agents
        self._init_zones()
        self.agents_per_team = 2
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

        # ------------------------------------------------------------------
        # Paper-style: categorical 2D targets (50 random predetermined cells)
        # These are fixed per reset and can be used by RL instead of continuous coords
        # TODO - Convert into Continuous for GRE
        # ------------------------------------------------------------------
        self.num_macro_targets: int = 50
        self.macro_targets: List[Tuple[int, int]] = []  # list of (x, y) grid cells

        # Timers
        self.respawn_seconds = 2.0
        self.decision_interval_seconds = 0.7
        self.decision_cooldown_seconds_by_agent: Dict[int, float] = {}

        # UI / rewards bookkeeping
        self.banner_queue: List[Tuple[str, Tuple[int, int, int], float]] = []

        self.debug_draw_ranges = True
        self.debug_draw_mine_ranges = True

        self.reset_default()

    # Setup & configuration
    def _init_zones(self) -> None:
        total_cols = max(1, self.col_count)
        third = max(1, total_cols // 3)

        blue_min = 0
        blue_max = third - 1

        red_min = total_cols - third
        red_max = total_cols - 1

        blue_max = max(blue_min, blue_max)
        red_min = min(red_min, red_max)

        self.blue_zone_col_range = (blue_min, blue_max)
        self.red_zone_col_range = (red_min, red_max)

    def _init_macro_targets(self, seed: Optional[int] = None) -> None:
        rng = random.Random(seed if seed is not None else 1337)
        self.macro_targets.clear()

        # Prefer free cells (grid == 0); if none, use all cells
        free_cells = [
            (x, y)
            for y in range(self.row_count)
            for x in range(self.col_count)
            if self.grid[y][x] == 0
        ]
        if not free_cells:
            free_cells = [
                (x, y)
                for y in range(self.row_count)
                for x in range(self.col_count)
            ]

        # Sample with replacement to get exactly num_macro_targets positions
        for _ in range(self.num_macro_targets):
            self.macro_targets.append(rng.choice(free_cells))

    def get_macro_target(self, index: int) -> Tuple[int, int]:
        if not self.macro_targets:
            self._init_macro_targets()
        if self.num_macro_targets <= 0:
            # Fallback: just pick center if something goes wrong
            return self.col_count // 2, self.row_count // 2
        idx = int(index) % self.num_macro_targets
        return self.macro_targets[idx]

    def get_all_macro_targets(self) -> List[Tuple[int, int]]:
        if not self.macro_targets:
            self._init_macro_targets()
        return list(self.macro_targets)

    def getGameManager(self) -> GameManager:
        return self.manager

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


    # Test cases / reset
    def reset_default(self) -> None:
        self._init_zones()
        self.manager.reset_game()
        self.agents_per_team = 2
        self.mines.clear()
        self.mine_pickups.clear()
        self._init_macro_targets(seed=self.agents_per_team)
        self.spawn_agents()
        self.spawn_mine_pickups()
        self.decision_cooldown_seconds_by_agent.clear()

    def runTestCase1(self) -> None:
        self.reset_default()

    def runTestCase2(self, agent_count: int, manager: GameManager) -> None:
        self.agents_per_team = int(max(1, min(100, agent_count)))
        self.manager.reset_game()
        self.mines.clear()
        self.mine_pickups.clear()

        # New macro target set when agent count changes
        self._init_macro_targets(seed=self.agents_per_team)

        self.spawn_agents()
        self.spawn_mine_pickups()

    def runTestCase3(self) -> None:
        self.manager.reset_game(reset_scores=True)

        # Swap zone definitions
        self.blue_zone_col_range, self.red_zone_col_range = (
            self.red_zone_col_range,
            self.blue_zone_col_range,
        )

        middle_row = self.row_count // 2

        self.manager.blue_flag_home = (self.col_count - 2, middle_row)
        self.manager.red_flag_home = (1, middle_row)
        self.manager.blue_flag_position = self.manager.blue_flag_home
        self.manager.red_flag_position = self.manager.red_flag_home
        self.manager.blue_flag_taken = False
        self.manager.red_flag_taken = False
        self.manager.blue_flag_carrier = None
        self.manager.red_flag_carrier = None

        self.mines.clear()
        self.mine_pickups.clear()

        # New macro targets for this swapped layout
        self._init_macro_targets(seed=self.agents_per_team + 1234)

        self.spawn_agents()
        self.spawn_mine_pickups()

    def update(self, delta_time: float) -> None:
        if self.manager.game_over:
            return

        # Advance global sim time / check for terminal condition
        winner_text = self.manager.tick_seconds(delta_time)
        if winner_text:
            color = (
                (90, 170, 250) if "BLUE" in winner_text else
                (250, 120, 70) if "RED" in winner_text else
                (230, 230, 230)
            )
            self.announce(winner_text, color, 3.0)
            return

        # ---------- small per-step time penalty for BLUE ----------
        for agent in self.blue_agents:
            if agent.isEnabled():
                self.manager.add_reward_event(-0.001, agent_id=agent.unique_id)

        # 1) Always tick ALL agents (disabled ones respawn)
        for agent in self.blue_agents + self.red_agents:
            agent.update(delta_time)

        # 2) Pathfinder dynamic obstacles: only ENABLED agents
        occupied = [(int(a.x), int(a.y)) for a in self.blue_agents + self.red_agents if a.isEnabled()]
        self.pathfinder.setDynamicObstacles(occupied)

        # 3) Process both teams
        for friendly_team, enemy_team in (
                (self.blue_agents, self.red_agents),
                (self.red_agents, self.blue_agents),
        ):
            for agent in friendly_team:
                if not agent.isEnabled():
                    continue

                # 1. Hazards — THIS IS WHERE MINE KILLS ARE TRACKED
                self.apply_mine_damage(agent)  # ← now tracks kills!
                self.apply_suppression(agent, enemy_team)

                # 2. Decision-making
                agent_key = id(agent)
                cooldown = self.decision_cooldown_seconds_by_agent.get(agent_key, 0.0)
                cooldown -= delta_time
                self.decision_cooldown_seconds_by_agent[agent_key] = cooldown

                side_external = self.external_control_for_side.get(agent.side, False)
                if self.use_internal_policies and not side_external and cooldown <= 0.0:
                    self.decide(agent)
                    self.decision_cooldown_seconds_by_agent[agent_key] = self.decision_interval_seconds

                # 3. Pickups & flag logic
                self.handle_mine_pickups(agent)
                self.apply_flag_rules(agent)

        # 4) Banner fade-out
        if self.banner_queue:
            text, color, t = self.banner_queue[-1]
            t = max(0.0, t - delta_time)
            self.banner_queue[-1] = (text, color, t)
            if t <= 0.0:
                self.banner_queue.pop()

    # Observation builder for RL / policies
    def build_observation(self, agent: Agent):
        """
        Build a paper-style CNN observation:

            obs_map: [7, H, W]  float32 in {0,1}
                ch 0: own UAV
                ch 1: teammate UAVs
                ch 2: enemy UAVs
                ch 3: friendly mines
                ch 4: enemy mines
                ch 5: own flag
                ch 6: enemy flag

            extra_vec: [7] float32
                [0] payload_none
                [1] payload_mine
                [2] payload_flag
                [3] time_norm in [0,1]
                [4] decision_norm in [0,1] (0..13 / 13)
                [5] is_id_0 (one-hot)
                [6] is_id_1 (one-hot)

        Positions are mirrored horizontally for RED so that each side
        always sees "its" base on the left.
        """

        side = agent.side  # "blue" or "red"
        rows, cols = self.row_count, self.col_count

        # We’ll use CNN size == grid size: [7, rows, cols]
        obs_map = np.zeros((7, rows, cols), dtype=np.float32)

        # ---------- helper: mirror coords for RED ----------
        def to_local_coords(x: int, y: int) -> Tuple[int, int]:
            """
            For BLUE: (x, y) stays as is.
            For RED:  mirror horizontally so RED also sees its home on the left.
            """
            if side == "blue":
                return x, y
            else:
                # mirror horizontally: col i -> cols-1-i
                return cols - 1 - x, y

        # =============== 0) OWN UAV =================
        if agent.isEnabled():
            ax, ay = int(agent.x), int(agent.y)
            lx, ly = to_local_coords(ax, ay)
            if 0 <= lx < cols and 0 <= ly < rows:
                obs_map[0, ly, lx] = 1.0

        # =============== 1) TEAMMATE UAVs ============
        team_agents = self.blue_agents if side == "blue" else self.red_agents
        for a in team_agents:
            if a is agent:
                continue
            if not a.isEnabled():
                continue
            x, y = int(a.x), int(a.y)
            lx, ly = to_local_coords(x, y)
            if 0 <= lx < cols and 0 <= ly < rows:
                obs_map[1, ly, lx] = 1.0

        # =============== 2) ENEMY UAVs ===============
        enemy_agents = self.red_agents if side == "blue" else self.blue_agents
        for e in enemy_agents:
            if not e.isEnabled():
                continue
            x, y = int(e.x), int(e.y)
            lx, ly = to_local_coords(x, y)
            if 0 <= lx < cols and 0 <= ly < rows:
                obs_map[2, ly, lx] = 1.0

        # =============== 3/4) MINES ==================
        for mine in self.mines:
            x, y = int(mine.x), int(mine.y)
            lx, ly = to_local_coords(x, y)
            if not (0 <= lx < cols and 0 <= ly < rows):
                continue
            if mine.owner_side == side:
                obs_map[3, ly, lx] = 1.0  # friendly mines
            else:
                obs_map[4, ly, lx] = 1.0  # enemy mines

        # =============== 5/6) FLAGS ==================
        # own flag position
        if side == "blue":
            own_flag_pos = self.manager.blue_flag_position
            enemy_flag_pos = self.manager.red_flag_position
        else:
            own_flag_pos = self.manager.red_flag_position
            enemy_flag_pos = self.manager.blue_flag_position

        ofx, ofy = own_flag_pos
        ofx, ofy = int(ofx), int(ofy)
        lofx, lofy = to_local_coords(ofx, ofy)
        if 0 <= lofx < cols and 0 <= lofy < rows:
            obs_map[5, lofy, lofx] = 1.0

        efx, efy = enemy_flag_pos
        efx, efy = int(efx), int(efy)
        lefx, lefy = to_local_coords(efx, efy)
        if 0 <= lefx < cols and 0 <= lefy < rows:
            obs_map[6, lefy, lefx] = 1.0

        # Payload one-hot: none / mine / flag
        payload_none = 0.0
        payload_mine = 0.0
        payload_flag = 0.0

        if agent.isCarryingFlag():
            payload_flag = 1.0
        elif getattr(agent, "mine_charges", 0) > 0:
            payload_mine = 1.0
        else:
            payload_none = 1.0

        # Time normalized [0,1]
        current_t = float(self.manager.current_time)
        max_t = max(float(self.manager.max_time), 1e-6)
        time_norm = current_t / max_t

        # Decision counter normalized [0,1] over [0..13]
        if not hasattr(agent, "decision_count"):
            agent.decision_count = 0
        decision_clamped = min(int(agent.decision_count), 13)
        decision_norm = decision_clamped / 13.0

        # Agent ID one-hot in team (0 or 1)
        id0 = 0.0
        id1 = 0.0
        if hasattr(agent, "agent_id") and agent.agent_id in (0, 1):
            if agent.agent_id == 0:
                id0 = 1.0
            else:
                id1 = 1.0

        extra_vec = np.array(
            [
                payload_none,
                payload_mine,
                payload_flag,
                time_norm,
                decision_norm,
                id0,
                id1,
            ],
            dtype=np.float32,
        )

        return obs_map, extra_vec

    # Macro-action executor
    def random_point_in_enemy_half(self, side: str) -> Tuple[int, int]:
        blue_min_col, blue_max_col = self.blue_zone_col_range
        red_min_col, red_max_col = self.red_zone_col_range

        if side == "blue":
            enemy_min_col, enemy_max_col = red_min_col, red_max_col
        else:
            enemy_min_col, enemy_max_col = blue_min_col, blue_max_col

        return (
            random.randint(enemy_min_col, enemy_max_col),
            random.randint(0, self.row_count - 1),
        )

    def random_point_in_own_half(self, side: str) -> Tuple[int, int]:
        blue_min_col, blue_max_col = self.blue_zone_col_range
        red_min_col, red_max_col = self.red_zone_col_range

        if side == "blue":
            own_min_col, own_max_col = blue_min_col, blue_max_col
        else:
            own_min_col, own_max_col = red_min_col, red_max_col

        return (
            random.randint(own_min_col, own_max_col),
            random.randint(0, self.row_count - 1),
        )

    def apply_macro_action( self, agent: Agent, action: MacroAction, param: Optional[Any] = None ) -> None:
        if not agent.isEnabled():
            return

        agent.last_macro_action = action
        side = agent.side
        gm = self.manager
        start = (agent.x, agent.y)

        def resolve_target_from_param(default_target: Tuple[int, int]) -> Tuple[int, int]:
            if param is None:
                return default_target

            # Coordinate param
            if isinstance(param, (tuple, list)) and len(param) == 2:
                return int(param[0]), int(param[1])

            # Treat anything else numeric-like as an index
            try:
                idx = int(param)
            except (TypeError, ValueError):
                return default_target

            return self.get_macro_target(idx)

        def safe_set_path(
            target: Tuple[int, int],
            avoid_enemies: bool = False,
            radius: int = 1,
        ):
            # Inflate dynamic obstacles near enemies if avoid_enemies=True
            original_blocked = set(self.pathfinder.blocked)
            blocked = set(original_blocked)
            if avoid_enemies:
                enemy_team = self.red_agents if side == "blue" else self.blue_agents
                for e in enemy_team:
                    if e.isEnabled():
                        for dx in range(-radius, radius + 1):
                            for dy in range(-radius, radius + 1):
                                blocked.add((e.x + dx, e.y + dy))
            self.pathfinder.blocked = blocked
            path = self.pathfinder.astar(start, target)
            self.pathfinder.blocked = original_blocked

            # Fallback to own half if no path
            if not path:
                fallback = self.random_point_in_own_half(side)
                path = self.pathfinder.astar(start, fallback) or []
            agent.setPath(path)

        # Core actions
        if action == MacroAction.GO_TO:
            # Default: random point in enemy half
            default_target = self.random_point_in_enemy_half(side)
            target = resolve_target_from_param(default_target)
            safe_set_path(target, avoid_enemies=agent.isCarryingFlag())

        elif action == MacroAction.GRAB_MINE:
            my_pickups = [p for p in self.mine_pickups if p.owner_side == side]
            if my_pickups:
                nearest = min(
                    my_pickups,
                    key=lambda p: (p.x - agent.x) ** 2 + (p.y - agent.y) ** 2,
                )
                safe_set_path((nearest.x, nearest.y))
            else:
                safe_set_path(self.random_point_in_own_half(side))

        elif action == MacroAction.GET_FLAG:
            ex, ey = gm.get_enemy_flag_position(side)
            safe_set_path((ex, ey), avoid_enemies=agent.isCarryingFlag())

        elif action == MacroAction.PLACE_MINE:
            # Default target: current tile
            default_target = (agent.x, agent.y)
            target = resolve_target_from_param(default_target)

            if agent.mine_charges > 0:
                own_flag_pos = (
                    gm.blue_flag_position if side == "blue" else gm.red_flag_position
                )
                # No duplicate mines on the same tile
                if not any(m.x == target[0] and m.y == target[1] for m in self.mines):
                    # Don't let them stack directly under their own flag
                    if target != own_flag_pos:
                        # Create mine with owner_id so we can credit the correct agent
                        self.mines.append(
                            Mine(
                                x=target[0],
                                y=target[1],
                                owner_side=side,
                                owner_id=getattr(agent, "unique_id", None),
                            )
                        )
                        agent.mine_charges -= 1
                        # Reward placement (blue gets phase-scaled mine reward)
                        self.manager.reward_mine_placed(agent, mine_pos=target)

            # Optionally move after placing (fallback path)
            safe_set_path(target)

        elif action == MacroAction.GO_HOME:
            home = gm.get_team_zone_center(side)
            safe_set_path(home, avoid_enemies=True, radius=2)


    # Policy-driven decision
    def decide(self, agent: Agent) -> None:
        if not agent.isEnabled():
            return

        # Increment decision counter for state feature
        if not hasattr(agent, "decision_count"):
            agent.decision_count = 0
        agent.decision_count += 1

        obs = self.build_observation(agent)
        policy = self.policies.get(agent.side)

        # Scripted policy (Policy subclass) or simple callable
        if isinstance(policy, Policy):
            action_id, param = policy.select_action(obs, agent, self)
        else:
            # Fallback: treat as a callable(agent, game_field) -> action_id
            action_id = policy(agent, self)
            param = None

        action = MacroAction(action_id)
        self.apply_macro_action(agent, action, param)

    # Mines / pickups / suppression / flags
    def handle_mine_pickups(self, agent: Agent) -> None:
        if not agent.isEnabled():
            return
        if not hasattr(agent, "mine_charges"):
            return

        for pickup in list(self.mine_pickups):
            if pickup.owner_side != agent.side:
                continue
            if agent.x == pickup.x and agent.y == pickup.y:
                if agent.mine_charges < self.max_mine_charges_per_agent:
                    needed = self.max_mine_charges_per_agent - agent.mine_charges
                    taken = min(needed, pickup.charges)
                    agent.mine_charges += taken
                    pickup.charges -= taken
                if pickup.charges <= 0:
                    self.mine_pickups.remove(pickup)

    def apply_mine_damage(self, agent: Agent) -> None:
        if not agent.isEnabled():
            return

        for mine in list(self.mines):
            # Skip friendly mines
            if mine.owner_side == agent.side:
                continue

            dist = math.hypot(mine.x - agent.x, mine.y - agent.y)
            if dist <= self.mine_radius_cells:
                # Who do we credit for this kill?
                killer_agent = None

                # 1) If we have an owner_id, try to find that agent
                if mine.owner_id is not None:
                    for a in (self.blue_agents + self.red_agents):
                        if getattr(a, "unique_id", None) == mine.owner_id:
                            killer_agent = a
                            break

                # 2) Fallback: for blue mines with missing owner_id, credit a miner
                if killer_agent is None and mine.owner_side == "blue":
                    killer_agent = next(
                        (a for a in self.blue_agents if getattr(a, "is_miner", False)),
                        None,
                    )
                    if killer_agent is None and self.blue_agents:
                        killer_agent = self.blue_agents[0]

                # Reward kill only if the killer is BLUE (learning team)
                if killer_agent is not None and getattr(killer_agent, "side", None) == "blue":
                    self.manager.reward_enemy_killed(
                        killer_agent=killer_agent,
                        victim_agent=agent,
                        cause="mine",
                    )

                if mine.owner_side == "blue" and agent.side == "red":
                    self.manager.record_mine_triggered_by_red()
                agent.disable_for_seconds(self.respawn_seconds)

                self.mines.remove(mine)
                break

    def apply_suppression(self, agent: Agent, enemies: List[Agent]) -> None:
        if not agent.isEnabled():
            return

        close_enemies = [
            e
            for e in enemies
            if e.isEnabled()
               and math.hypot(e.x - agent.x, e.y - agent.y) <= self.suppression_range_cells
        ]

        if len(close_enemies) >= 2:
            killer_agent = None
            blue_suppressors = [e for e in close_enemies if getattr(e, "side", None) == "blue"]
            if blue_suppressors:
                killer_agent = blue_suppressors[0]
            elif close_enemies:
                killer_agent = close_enemies[0]

            # Reward suppression kill only if the suppressor is BLUE
            if killer_agent is not None and getattr(killer_agent, "side", None) == "blue":
                self.manager.reward_enemy_killed(
                    killer_agent=killer_agent,
                    victim_agent=agent,
                    cause="suppression",
                )

            # Just disable; Agent.disable_for_seconds will:
            # - Drop the flag via GameManager.handle_agent_death
            # - Clear local carrying state
            agent.disable_for_seconds(self.respawn_seconds)

    def apply_flag_rules(self, agent: Agent) -> None:
        if self.manager.try_pickup_enemy_flag(agent):
            if not agent.isCarryingFlag():
                agent.setCarryingFlag(True)

        if agent.isCarryingFlag():
            if self.manager.try_score_if_carrying_and_home(agent):
                agent.setCarryingFlag(False)
                self.announce(
                    "BLUE SCORES!" if agent.side == "blue" else "RED SCORES!",
                    (90, 170, 250)
                    if agent.side == "blue"
                    else (250, 120, 70),
                    2.0,
                )

    # Rendering
    def draw(self, surface: pg.Surface, board_rect: pg.Rect) -> None:
        rect_width, rect_height = board_rect.width, board_rect.height
        cell_width = rect_width / max(1, self.col_count)
        cell_height = rect_height / max(1, self.row_count)

        surface.fill((20, 22, 30), board_rect)
        self.draw_halves_and_center_line(surface, board_rect)

        grid_color = (70, 70, 85)
        for row in range(self.row_count + 1):
            y = int(board_rect.top + row * cell_height)
            pg.draw.line(surface, grid_color, (board_rect.left, y), (board_rect.right, y), 1)

        for col in range(self.col_count + 1):
            x = int(board_rect.left + col * cell_width)
            pg.draw.line(surface, grid_color, (x, board_rect.top), (x, board_rect.bottom), 1)

        # Debug ranges
        if self.debug_draw_ranges or self.debug_draw_mine_ranges:
            range_surface = pg.Surface((board_rect.width, board_rect.height), pg.SRCALPHA)

            if self.debug_draw_ranges:
                sup_radius_px = self.suppression_range_cells * min(cell_width, cell_height)

                def draw_sup_range(agent: Agent, rgba: Tuple[int, int, int, int]) -> None:
                    cx = (agent.x + 0.5) * cell_width
                    cy = (agent.y + 0.5) * cell_height
                    local_center = (int(cx), int(cy))
                    pg.draw.circle(range_surface, rgba, local_center, int(sup_radius_px), width=2)

                for a in self.blue_agents:
                    if a.isEnabled():
                        draw_sup_range(a, (50, 130, 255, 190))
                for a in self.red_agents:
                    if a.isEnabled():
                        draw_sup_range(a, (255, 110, 70, 190))

            if self.debug_draw_mine_ranges and self.mines:
                mine_radius_px = self.mine_radius_cells * min(cell_width, cell_height)
                for mine in self.mines:
                    cx = (mine.x + 0.5) * cell_width
                    cy = (mine.y + 0.5) * cell_height
                    local_center = (int(cx), int(cy))
                    rgba = (
                        (40, 170, 230, 170)
                        if mine.owner_side == "blue"
                        else (230, 120, 80, 170)
                    )
                    pg.draw.circle(range_surface, rgba, local_center, int(mine_radius_px), width=1)

            surface.blit(range_surface, board_rect.topleft)

        # Flags
        blue_base = getattr(self.manager, "blue_flag_home", self.manager.blue_flag_position)
        red_base = getattr(self.manager, "red_flag_home", self.manager.red_flag_position)

        self.draw_flag(
            surface,
            board_rect,
            cell_width,
            cell_height,
            blue_base,  # <-- stays fixed
            (90, 170, 250),
            self.manager.blue_flag_taken,
        )
        self.draw_flag(
            surface,
            board_rect,
            cell_width,
            cell_height,
            red_base,  # <-- stays fixed
            (250, 120, 70),
            self.manager.red_flag_taken,
        )

        # Mine pickups
        def draw_mine_pickup(pickup: MinePickup) -> None:
            cx = board_rect.left + (pickup.x + 0.5) * cell_width
            cy = board_rect.top + (pickup.y + 0.5) * cell_height
            r_outer = int(0.3 * min(cell_width, cell_height))
            r_inner = int(0.16 * min(cell_width, cell_height))
            color = (80, 210, 255) if pickup.owner_side == "blue" else (255, 160, 110)
            pg.draw.circle(surface, (10, 10, 14), (int(cx), int(cy)), r_outer)
            pg.draw.circle(surface, color, (int(cx), int(cy)), r_outer, width=2)
            pg.draw.circle(surface, color, (int(cx), int(cy)), r_inner)

        for pickup in self.mine_pickups:
            draw_mine_pickup(pickup)

        # Armed mines
        def draw_mine(mine: Mine) -> None:
            cx = board_rect.left + (mine.x + 0.5) * cell_width
            cy = board_rect.top + (mine.y + 0.5) * cell_height
            r = int(0.35 * min(cell_width, cell_height))
            color = (40, 170, 230) if mine.owner_side == "blue" else (230, 120, 80)
            pg.draw.circle(surface, color, (int(cx), int(cy)), r)
            pg.draw.circle(surface, (5, 5, 8), (int(cx), int(cy)), r, width=1)

        for mine in self.mines:
            draw_mine(mine)

        # Agents
        def draw_agent(agent: Agent, body_rgb, enemy_flag_rgb):
            center_x = board_rect.left + (agent.x + 0.5) * cell_width
            center_y = board_rect.top + (agent.y + 0.5) * cell_height
            tri_size = 0.45 * min(cell_width, cell_height)

            target_x, target_y = self.manager.get_enemy_flag_position(agent.getSide())
            if agent.path:
                target_x, target_y = agent.path[0]

            to_target_x = target_x - agent.x
            to_target_y = target_y - agent.y
            magnitude = max(
                (to_target_x * to_target_x + to_target_y * to_target_y) ** 0.5, 1e-6
            )
            unit_x, unit_y = to_target_x / magnitude, to_target_y / magnitude
            left_x, left_y = -unit_y, unit_x

            tip = (
                int(center_x + unit_x * tri_size),
                int(center_y + unit_y * tri_size),
            )
            left = (
                int(center_x - unit_x * tri_size * 0.6 + left_x * tri_size * 0.6),
                int(center_y - unit_y * tri_size * 0.6 + left_y * tri_size * 0.6),
            )
            right = (
                int(center_x - unit_x * tri_size * 0.6 - left_x * tri_size * 0.6),
                int(center_y - unit_y * tri_size * 0.6 - left_y * tri_size * 0.6),
            )

            body_color = body_rgb if agent.isEnabled() else (50, 50, 55)
            pg.draw.polygon(surface, body_color, (tip, left, right))

            if agent.isCarryingFlag():
                flag_size = int(tri_size * 0.5)
                flag_rect = pg.Rect(
                    tip[0] - flag_size // 2,
                    tip[1] - flag_size // 2,
                    flag_size,
                    flag_size,
                )
                pg.draw.rect(surface, enemy_flag_rgb, flag_rect)

            if agent.isTagged():
                pg.draw.polygon(surface, (245, 245, 245), (tip, left, right), width=2)

        for agent in self.blue_agents:
            draw_agent(agent, (0, 180, 255), (250, 120, 70))
        for agent in self.red_agents:
            draw_agent(agent, (255, 120, 40), (90, 170, 250))

        # Banner
        if self.banner_queue:
            text, color, time_left = self.banner_queue[-1]
            fade_factor = max(0.3, min(1.0, time_left / 2.0))
            font = pg.font.SysFont(None, 48)
            faded_color = tuple(int(channel * fade_factor) for channel in color)
            img = font.render(text, True, faded_color)
            surface.blit(
                img,
                (
                    board_rect.centerx - img.get_width() // 2,
                    board_rect.top + 12,
                ),
            )

    def draw_halves_and_center_line(self, surface: pg.Surface, board_rect: pg.Rect) -> None:
        rect_width, rect_height = board_rect.width, board_rect.height
        cell_width = rect_width / max(1, self.col_count)

        def fill_cols(col_start: int, col_end: int, rgba: Tuple[int, int, int, int]) -> None:
            if col_start > col_end:
                return
            x0 = int(board_rect.left + col_start * cell_width)
            x1 = int(board_rect.left + (col_end + 1) * cell_width)
            band_width = max(1, x1 - x0)
            band_surface = pg.Surface((band_width, rect_height), pg.SRCALPHA)
            band_surface.fill(rgba)
            surface.blit(band_surface, (x0, board_rect.top))

        total_cols = max(1, self.col_count)
        blue_min_col, blue_max_col = self.blue_zone_col_range
        red_min_col, red_max_col = self.red_zone_col_range

        mid_start = blue_max_col + 1
        mid_end = red_min_col - 1

        fill_cols(blue_min_col, blue_max_col, (15, 45, 120, 140))

        if mid_start <= mid_end:
            fill_cols(mid_start, mid_end, (40, 40, 55, 90))

        fill_cols(red_min_col, red_max_col, (120, 45, 15, 140))

        mid_col_index = total_cols // 2
        mid_x = int(board_rect.left + mid_col_index * cell_width)
        pg.draw.line(
            surface,
            (190, 190, 210),
            (mid_x, board_rect.top),
            (mid_x, board_rect.bottom),
            2,
        )

    def draw_flag(
        self,
        surface: pg.Surface,
        board_rect: pg.Rect,
        cell_width: float,
        cell_height: float,
        grid_pos: Tuple[int, int],
        color: Tuple[int, int, int],
        is_taken: bool,
    ) -> None:
        grid_x, grid_y = grid_pos
        center_x = board_rect.left + (grid_x + 0.5) * cell_width
        center_y = board_rect.top + (grid_y + 0.5) * cell_height
        radius_px = int(TEAM_ZONE_RADIUS_CELLS * min(cell_width, cell_height))

        zone_surface = pg.Surface((board_rect.width, board_rect.height), pg.SRCALPHA)
        local_center = (
            int(center_x - board_rect.left),
            int(center_y - board_rect.top),
        )

        pg.draw.circle(zone_surface, (*color, 40), local_center, radius_px, width=0)
        pg.draw.circle(zone_surface, (*color, 110), local_center, radius_px, width=2)
        surface.blit(zone_surface, board_rect.topleft)

        if not is_taken:
            flag_size = int(0.5 * min(cell_width, cell_height))
            flag_rect = pg.Rect(
                int(center_x - flag_size / 2),
                int(center_y - flag_size / 2),
                flag_size,
                flag_size,
            )
            pg.draw.rect(surface, color, flag_rect)

    def announce(self, text: str, color=(255, 255, 255), seconds: float = 2.0) -> None:
        self.banner_queue.append((text, color, seconds))

    # Spawning
    def spawn_agents(self) -> None:
        rng = random.Random(self.agents_per_team)

        # Clear everything
        for a in self.blue_agents + self.red_agents:
            a.mine_charges = 0
        self.blue_agents.clear()
        self.red_agents.clear()
        self.mines.clear()
        self.mine_pickups.clear()
        self.decision_cooldown_seconds_by_agent.clear()

        blue_min_col, blue_max_col = self.blue_zone_col_range
        red_min_col, red_max_col = self.red_zone_col_range

        blue_cells = [
            (r, c)
            for r in range(self.row_count)
            for c in range(blue_min_col, blue_max_col + 1)
        ]
        red_cells = [
            (r, c)
            for r in range(self.row_count)
            for c in range(red_min_col, red_max_col + 1)
        ]
        rng.shuffle(blue_cells)
        rng.shuffle(red_cells)

        n = self.agents_per_team

        # Blue
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
                game_manager=self.manager,  # <-- IMPORTANT
            )
            agent.spawn_xy = (col, row)
            agent.mine_charges = 0
            agent.decision_count = 0
            self.blue_agents.append(agent)

        # Red
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
            agent.mine_charges = 0
            agent.decision_count = 0
            self.red_agents.append(agent)

    def spawn_mine_pickups(self) -> None:
        self.mine_pickups.clear()
        rng = random.Random(self.agents_per_team + 9999)
        blue_min_col, blue_max_col = self.blue_zone_col_range
        red_min_col, red_max_col = self.red_zone_col_range

        def spawn_for_band(owner_side: str, col_min: int, col_max: int) -> None:
            all_cells = [
                (row, col)
                for row in range(self.row_count)
                for col in range(col_min, col_max + 1)
            ]
            rng.shuffle(all_cells)
            placed = 0
            max_pickups = self.mines_per_team

            flag_pos = (
                self.manager.blue_flag_position
                if owner_side == "blue"
                else self.manager.red_flag_position
            )

            for row, col in all_cells:
                if (col, row) == flag_pos:
                    continue
                self.mine_pickups.append(
                    MinePickup(x=col, y=row, owner_side=owner_side, charges=1)
                )
                placed += 1
                if placed >= max_pickups:
                    break

        spawn_for_band("blue", blue_min_col, blue_max_col)
        spawn_for_band("red", red_min_col, red_max_col)


__all__ = ["GameField", "MacroAction"]
