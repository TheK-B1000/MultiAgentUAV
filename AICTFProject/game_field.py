"""
game_field.py

2D Capture-the-Flag (CTF) simulation environment for multi-agent RL.

This environment models a simplified 2-vs-2 UAV CTF game:

  - Physical arena: 10.0m (X) × 4.28m (Y), mapped to a discrete grid.
  - Two teams (BLUE, RED) with a configurable number of agents per side.
  - Flags, mines, suppression kills, respawns, and team zones.
  - Macro-actions (GO_TO, GRAB_MINE, GET_FLAG, PLACE_MINE, GO_HOME) executed
    via a pathfinder over the discrete grid.
  - CNN observations: 7-channel 30×40 map with ego-centric mirroring so both
    BLUE and RED see the game in a common canonical frame.
  - Scripted opponents (OP1/OP2/OP3) and an optional neural RED policy for
    self-play.

The focus is to provide a stable, research-friendly interface for MARL training
(PPO, MAPPO, QMIX, self-play, etc.) while preserving a clean mapping to a
real-world UAV arena (sim-to-real).

NOTE: This file has **no pygame dependency**. Rendering lives in viewer_game_field.py.
"""

import random
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import torch

from game_manager import GameManager
from agents import Agent, TEAM_ZONE_RADIUS_CELLS
from pathfinder import Pathfinder

from macro_actions import MacroAction
from policies import Policy, OP1RedPolicy, OP2RedPolicy, OP3RedPolicy

# ----------------------------------------------------------------------
# Physical arena dimensions (meters)
# ----------------------------------------------------------------------
ARENA_WIDTH_M = 10.0   # X direction
ARENA_HEIGHT_M = 4.28  # Y direction

# ----------------------------------------------------------------------
# CNN map resolution for observations
# ----------------------------------------------------------------------
CNN_COLS = 30
CNN_ROWS = 40
NUM_CNN_CHANNELS = 7


# ----------------------------------------------------------------------
# Mines & pickups
# ----------------------------------------------------------------------
@dataclass
class Mine:
    x: int
    y: int
    owner_side: str           # "blue" or "red"
    owner_id: Optional[str] = None  # unique_id of placing agent (if known)


@dataclass
class MinePickup:
    x: int
    y: int
    owner_side: str           # "blue" or "red"
    charges: int = 1


# ----------------------------------------------------------------------
# GameField — main 2D simulation environment
# ----------------------------------------------------------------------
class GameField:
    def __init__(self, grid: List[List[int]]):
        """
        Construct a CTF environment over a discrete grid.

        Args:
            grid: 2D list of ints representing walkable/blocked cells (0=free).
        """
        self.grid = grid
        self.row_count = len(grid)
        self.col_count = len(grid[0]) if self.row_count else 0

        # Core game state
        self.manager = GameManager(cols=self.col_count, rows=self.row_count)

        # Mines / pickups configuration
        self.mines: List[Mine] = []
        self.mine_pickups: List[MinePickup] = []
        self.mine_radius_cells = 1.5
        self.suppression_range_cells = 2.0
        self.mines_per_team = 4
        self.max_mine_charges_per_agent = 2

        # Policy wiring (internal vs external control)
        self.use_internal_policies: bool = True
        self.external_control_for_side: Dict[str, bool] = {"blue": False, "red": False}

        # By default, BOTH teams use the paper's OP3-style scripted policy.
        # The MARL trainer typically sets BLUE to external control.
        self.policies: Dict[str, Any] = {
            "blue": OP3RedPolicy("blue"),  # baseline scripted OP3 on BLUE side
            "red": OP3RedPolicy("red"),    # default opponent: OP3
        }
        self.opponent_mode: str = "OP3"

        # Zones & agents
        self._init_zones()
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

        # Physical cell size in meters (mapping grid -> 10m x 4.28m)
        self.cell_width_m = ARENA_WIDTH_M / max(1, self.col_count)
        self.cell_height_m = ARENA_HEIGHT_M / max(1, self.row_count)

        # ------------------------------------------------------------------
        # Categorical 2D macro-targets over random grid positions.
        # The policy outputs an index (0..num_macro_targets-1), which is
        # mapped to a grid cell via self.macro_targets.
        # ------------------------------------------------------------------
        self.num_macro_targets: int = 50
        self.macro_targets: List[Tuple[int, int]] = []  # list of (x, y) grid cells

        # Timers
        self.respawn_seconds: float = 2.0
        self.decision_interval_seconds: float = 0.7
        self.decision_cooldown_seconds_by_agent: Dict[int, float] = {}

        # UI / rewards bookkeeping (for viewer / debugging only)
        # Safe in headless training; no pygame required.
        self.banner_queue: List[Tuple[str, Tuple[int, int, int], float]] = []

        # Debug flags (used only by viewer; harmless here)
        self.debug_draw_ranges: bool = True
        self.debug_draw_mine_ranges: bool = True

        # Initialize to default scenario
        self.reset_default()

    # ------------------------------------------------------------------
    # Setup & configuration
    # ------------------------------------------------------------------
    def _init_zones(self) -> None:
        """
        Initialize team zones (blue base, red base) as left/right thirds of the field.
        """
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
        """
        Initialize a fixed set of discrete macro-targets over the grid.

        These targets let the policy output a categorical index and still
        map to meaningful 2D waypoints in the arena.
        """
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
        """
        Map a categorical macro-target index into a grid cell (x, y).
        """
        if not self.macro_targets:
            self._init_macro_targets()
        if self.num_macro_targets <= 0:
            # Fallback: center of the arena
            return self.col_count // 2, self.row_count // 2
        idx = int(index) % self.num_macro_targets
        return self.macro_targets[idx]

    def get_all_macro_targets(self) -> List[Tuple[int, int]]:
        """
        Return a copy of all macro-target positions.
        """
        if not self.macro_targets:
            self._init_macro_targets()
        return list(self.macro_targets)

    def getGameManager(self) -> GameManager:
        """
        Convenience accessor for the underlying GameManager.
        """
        return self.manager

    # ------------------------------------------------------------------
    # Policy wiring (scripted / neural / external control)
    # ------------------------------------------------------------------
    def set_policies(self, blue: Any, red: Any) -> None:
        self.policies["blue"] = blue
        self.policies["red"] = red

    def set_red_opponent(self, mode: str) -> None:
        """
        Select scripted RED opponent variant (OP1 / OP2 / OP3).
        """
        mode = mode.upper()
        self.opponent_mode = mode
        if mode == "OP1":
            self.policies["red"] = OP1RedPolicy("red")
        elif mode == "OP2":
            self.policies["red"] = OP2RedPolicy("red")
        else:
            self.policies["red"] = OP3RedPolicy("red")

    def set_external_control(self, side: str, external: bool) -> None:
        """
        Set whether a side ("blue" or "red") is controlled externally (RL)
        or internally via self.policies.
        """
        if side not in ("blue", "red"):
            raise ValueError(f"Unknown side: {side}")
        self.external_control_for_side[side] = external

    def set_all_external_control(self, external: bool) -> None:
        """
        Convenience: set external-control flag for both sides.
        """
        for side in ("blue", "red"):
            self.external_control_for_side[side] = external

    def set_red_policy_neural(self, policy_net: Any) -> None:
        """
        Attach a neural policy network as the RED opponent (self-play).
        The network must implement .act(obs, agent, game_field, deterministic=...).
        """
        self.policies["red"] = policy_net
        self.opponent_mode = "NEURAL"

    # ------------------------------------------------------------------
    # Coordinate mapping (grid ↔ world ↔ CNN)
    # ------------------------------------------------------------------
    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """
        Map a grid cell (col, row) to physical coordinates (x, y) in meters
        at the center of the cell.
        """
        x = (col + 0.5) * self.cell_width_m
        y = (row + 0.5) * self.cell_height_m
        return x, y

    def world_to_cnn_cell(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """
        Map physical coordinates (x, y) in meters to a 20×20 CNN cell index.
        """
        # Normalize to [0,1]
        u = max(0.0, min(1.0, x_m / ARENA_WIDTH_M))
        v = max(0.0, min(1.0, y_m / ARENA_HEIGHT_M))

        col = int(u * CNN_COLS)
        row = int(v * CNN_ROWS)

        if col >= CNN_COLS:
            col = CNN_COLS - 1
        if row >= CNN_ROWS:
            row = CNN_ROWS - 1

        return col, row

    def grid_to_cnn_cell(self, col: int, row: int) -> Tuple[int, int]:
        """
        Convenience: map a grid cell (col,row) directly into CNN coordinates.
        """
        x_m, y_m = self.grid_to_world(col, row)
        return self.world_to_cnn_cell(x_m, y_m)

    # ------------------------------------------------------------------
    # Reset / scenario configuration
    # ------------------------------------------------------------------
    def reset_default(self) -> None:
        """
        Reset to the standard 2-vs-2 scenario used for training:
          - Blue and Red zones left/right.
          - Flags at fixed homes.
          - Fresh macro-targets, agents, and mine pickups.
        """
        self._init_zones()
        self.manager.reset_game()
        self.mines.clear()
        self.mine_pickups.clear()
        self._init_macro_targets(seed=self.agents_per_team)
        self.spawn_agents()
        self.spawn_mine_pickups()
        self.decision_cooldown_seconds_by_agent.clear()

    def set_agent_count_and_reset(self, new_count: int) -> None:
        """
        Public method to dynamically change the number of agents per team
        and reset the game environment accordingly, including scores.

        Args:
            new_count: The new number of agents per team (must be >= 1).
        """
        # 1. Ensure count is valid
        if new_count < 1:
            new_count = 1

        # 2. Update the stored agent count
        self.agents_per_team = new_count

        # 3. Reset the scores explicitly (ensures a fresh game)
        self.manager.reset_game(reset_scores=True)

        # 4. Reset the scenario with the new count (spawns agents, etc.)
        self.reset_default()

    # ------------------------------------------------------------------
    # Core simulation step
    # ------------------------------------------------------------------
    def update(self, delta_time: float) -> None:
        """
        Advance the simulation by delta_time (seconds).

        This:
          - Advances game time and checks for win conditions.
          - Updates all agents (motion / respawn timers).
          - Updates pathfinder dynamic obstacles.
          - Applies hazards (mines, suppression).
          - Triggers internal policy decisions for scripted/neural agents.
          - Handles pickups and flag rules.
        """
        if self.manager.game_over:
            return

        # Advance global sim time / check for terminal condition
        winner_text = self.manager.tick_seconds(delta_time)
        if winner_text:
            # In headless mode this just populates banner_queue; viewer decides what to do.
            color = (
                (90, 170, 250) if "BLUE" in winner_text else
                (250, 120, 70) if "RED" in winner_text else
                (230, 230, 230)
            )
            self.announce(winner_text, color, 3.0)
            return

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

                # 1. Hazards — mine kills & suppression
                self.apply_mine_damage(agent)
                self.apply_suppression(agent, enemy_team)

                # 2. Decision-making (only internal, external handled by RL)
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

        # 4) Banner fade-out (visual feedback only; harmless headless)
        if self.banner_queue:
            text, color, t = self.banner_queue[-1]
            t = max(0.0, t - delta_time)
            self.banner_queue[-1] = (text, color, t)
            if t <= 0.0:
                self.banner_queue.pop()

    # ------------------------------------------------------------------
    # Observation builder for RL / policies (7-channel 20×20 CNN)
    # ------------------------------------------------------------------
    def build_observation(self, agent: Agent) -> List[List[List[float]]]:
        """
        Builds a 7-channel 20×20 spatial observation for the given agent.

        Channels:
          0: Own UAV position
          1: Teammate UAVs (same side, excluding self)
          2: Enemy UAVs
          3: Friendly mines
          4: Enemy mines
          5: Own flag
          6: Enemy flag

        The observation is point-mirrored to the agent's side, meaning BLUE's
        view is the canonical frame, and RED's view is mirrored across the
        arena so they see a symmetric layout. This helps with policy transfer.
        """
        side = agent.side  # "blue" or "red"
        game_state = self.manager

        # Initialize C×H×W with zeros
        channels: List[List[List[float]]] = [
            [[0.0 for _ in range(CNN_COLS)] for _ in range(CNN_ROWS)]
            for _ in range(NUM_CNN_CHANNELS)
        ]

        # Helper to map grid (col, row) to CNN (col, row), applying mirroring for Red team
        def get_cnn_cell(col: int, row: int) -> Tuple[int, int]:
            x_m, y_m = self.grid_to_world(col, row)
            # Normalize to [0, 1]
            u = max(0.0, min(1.0, x_m / ARENA_WIDTH_M))
            v = max(0.0, min(1.0, y_m / ARENA_HEIGHT_M))

            # Apply point mirroring if agent is on the Red side
            if side == "red":
                u = 1.0 - u  # Mirror X
                v = 1.0 - v  # Mirror Y

            col_cnn = int(u * CNN_COLS)
            row_cnn = int(v * CNN_ROWS)

            # Clamp to CNN boundaries
            col_cnn = max(0, min(CNN_COLS - 1, col_cnn))
            row_cnn = max(0, min(CNN_ROWS - 1, row_cnn))

            return col_cnn, row_cnn

        # Helper to set a cell in a given channel
        def set_chan(c: int, col: int, row: int) -> None:
            if 0 <= col < CNN_COLS and 0 <= row < CNN_ROWS:
                channels[c][row][col] = 1.0

        # --- 0: own UAV ---
        own_col, own_row = int(agent.x), int(agent.y)
        own_cnn_col, own_cnn_row = get_cnn_cell(own_col, own_row)
        set_chan(0, own_cnn_col, own_cnn_row)

        # Decide friendly/enemy sets
        friendly_team = self.blue_agents if side == "blue" else self.red_agents
        enemy_team = self.red_agents if side == "blue" else self.blue_agents

        # --- 1: teammate UAVs (same side, excluding self) ---
        for a in friendly_team:
            if a is agent or not a.isEnabled():
                continue
            c, r = get_cnn_cell(int(a.x), int(a.y))
            set_chan(1, c, r)

        # --- 2: enemy UAVs ---
        for a in enemy_team:
            if not a.isEnabled():
                continue
            c, r = get_cnn_cell(int(a.x), int(a.y))
            set_chan(2, c, r)

        # --- 3 & 4: mines (friendly vs enemy) ---
        for m in self.mines:
            c, r = get_cnn_cell(int(m.x), int(m.y))
            if m.owner_side == side:
                set_chan(3, c, r)  # friendly mines
            else:
                set_chan(4, c, r)  # enemy mines

        # --- 5 & 6: flags (own vs enemy) ---
        if side == "blue":
            own_flag_pos = game_state.blue_flag_position
            enemy_flag_pos = game_state.red_flag_position
        else:
            own_flag_pos = game_state.red_flag_position
            enemy_flag_pos = game_state.blue_flag_position

        # own flag
        c_own_flag, r_own_flag = get_cnn_cell(int(own_flag_pos[0]), int(own_flag_pos[1]))
        set_chan(5, c_own_flag, r_own_flag)

        # enemy flag
        c_enemy_flag, r_enemy_flag = get_cnn_cell(int(enemy_flag_pos[0]), int(enemy_flag_pos[1]))
        set_chan(6, c_enemy_flag, r_enemy_flag)

        return channels

    # ------------------------------------------------------------------
    # Macro-action executor
    # ------------------------------------------------------------------
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

    def apply_macro_action(
        self,
        agent: Agent,
        action: MacroAction,
        param: Optional[Any] = None,
    ) -> None:
        """
        Execute a high-level macro-action by converting it into a path over the grid.

        For RL, `param` is typically a categorical macro-target index that gets
        resolved into a (x,y) grid coordinate.
        """
        if not agent.isEnabled():
            return

        agent.last_macro_action = action
        side = agent.side
        gm = self.manager
        start = (agent.x, agent.y)

        def resolve_target_from_param(default_target: Tuple[int, int]) -> Tuple[int, int]:
            if param is None:
                return default_target

            # Coordinate param (x,y)
            if isinstance(param, (tuple, list)) and len(param) == 2:
                return int(param[0]), int(param[1])

            # Treat anything else numeric-like as an index into macro_targets
            try:
                idx = int(param)
            except (TypeError, ValueError):
                return default_target

            return self.get_macro_target(idx)

        def safe_set_path(
            target: Tuple[int, int],
            avoid_enemies: bool = False,
            radius: int = 1,
        ) -> None:
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

        # ================= MACRO-ACTION HANDLERS =================
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
                # If no pickups, wander in own half
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

    # ------------------------------------------------------------------
    # Policy-driven decision for internal agents (scripted / neural RED)
    # ------------------------------------------------------------------
    def decide(self, agent: Agent) -> None:
        """
        Let the current internal policy for agent.side choose a macro-action.

        For RL training, BLUE is usually externally controlled; RED can be
        scripted (OP1/OP2/OP3) or a neural ghost policy.
        """
        if not agent.isEnabled():
            return

        # Increment decision counter for potential state features / diagnostics
        if not hasattr(agent, "decision_count"):
            agent.decision_count = 0
        agent.decision_count += 1

        obs = self.build_observation(agent)
        policy = self.policies.get(agent.side)

        # 1. Handle Neural Network Policy (Self-Play)
        if hasattr(policy, "act"):
            # Convert obs list to Tensor [1, C, H, W]
            device = next(policy.parameters()).device
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                # deterministic=False → more variety; deterministic=True → "best" move
                out = policy.act(
                    obs_tensor,
                    agent=agent,
                    game_field=self,
                    deterministic=True,
                )

            # Extract integers from tensors
            action_id = int(out["macro_action"][0].item())
            param = int(out["target_action"][0].item())

        # 2. Handle Scripted Policy (OP1, OP2, OP3)
        elif isinstance(policy, Policy):
            action_id, param = policy.select_action(obs, agent, self)

        # 3. Fallback (simple callable)
        else:
            action_id = policy(agent, self)
            param = None

        action = MacroAction(action_id)
        self.apply_macro_action(agent, action, param)

    # ------------------------------------------------------------------
    # Mines / pickups / suppression / flags
    # ------------------------------------------------------------------
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
                    (90, 170, 250) if agent.side == "blue" else (250, 120, 70),
                    2.0,
                )

    # ------------------------------------------------------------------
    # Utility / UI helpers (no pygame; viewer decides how to render)
    # ------------------------------------------------------------------
    def announce(self, text: str, color=(255, 255, 255), seconds: float = 2.0) -> None:
        """
        Queue a banner message (for visual feedback only).

        Headless training can ignore this; the viewer can consume banner_queue.
        """
        self.banner_queue.append((text, color, seconds))

    # ------------------------------------------------------------------
    # Spawning
    # ------------------------------------------------------------------
    def spawn_agents(self) -> None:
        """
        Spawn agents for both teams into their respective zones.
        """
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
                game_manager=self.manager,
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
        """
        Spawn mine pickups for each team in their home zones, excluding the flag tile.
        """
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


__all__ = ["GameField", "MacroAction", "Mine", "MinePickup"]
