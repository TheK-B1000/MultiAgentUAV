"""
game_field.py

2D Capture-the-Flag (CTF) simulation environment for multi-agent RL.

This environment models a simplified 2-vs-2 UAV CTF game:

  - Physical arena: 10.0m (X) × 4.28m (Y), mapped to a discrete grid.
  - Two teams (BLUE, RED) with a configurable number of agents per side.
  - Flags, mines, suppression kills, respawns, and team zones.
  - Macro-actions (GO_TO, GRAB_MINE, GET_FLAG, PLACE_MINE, GO_HOME) executed
    via a pathfinder over the discrete grid.
  - CNN observations: 7-channel 40×30 map (rows×cols) with ego-centric mirroring.
  - Scripted opponents (OP1/OP2/OP3) and an optional neural RED policy for self-play.

External control support (for PPO/MAPPO/QMIX):
  - submit_external_actions(actions_by_agent)
  - external_missing_action_mode: "idle" (default) or "internal"
  - pending_external_actions cleared on reset to prevent cross-episode contamination
"""

import random
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import torch

from game_manager import GameManager
from agents import Agent
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
    owner_side: str                 # "blue" or "red"
    owner_id: Optional[str] = None  # Agent.unique_id of placing agent (if known)


@dataclass
class MinePickup:
    x: int
    y: int
    owner_side: str                 # "blue" or "red"
    charges: int = 1


class GameField:
    def __init__(self, grid: List[List[int]]):
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

        # Policy wiring
        self.use_internal_policies: bool = True
        self.external_control_for_side: Dict[str, bool] = {"blue": False, "red": False}

        # External action API (trainer submits actions, env consumes at decision boundaries)
        # Keys are arbitrary strings (slot_id/unique_id/side_agentId); resolution is robust.
        self.pending_external_actions: Dict[str, Tuple[int, int]] = {}
        # "idle": if missing external action, do nothing (keep current path)
        # "internal": if missing, fall back to internal decide()
        self.external_missing_action_mode: str = "idle"

        # Defaults: scripted OP3 on both sides (trainer usually sets BLUE external)
        self.policies: Dict[str, Any] = {
            "blue": OP3RedPolicy("blue"),
            "red": OP3RedPolicy("red"),
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

        # Physical cell size in meters (grid -> 10m x 4.28m)
        self.cell_width_m = ARENA_WIDTH_M / max(1, self.col_count)
        self.cell_height_m = ARENA_HEIGHT_M / max(1, self.row_count)

        # Macro-targets
        self.num_macro_targets: int = 50
        self.macro_targets: List[Tuple[int, int]] = []

        # Timers
        self.respawn_seconds: float = 2.0
        self.decision_interval_seconds: float = 0.7
        self.decision_cooldown_seconds_by_agent: Dict[int, float] = {}

        # UI / debug
        self.banner_queue: List[Tuple[str, Tuple[int, int, int], float]] = []
        self.debug_draw_ranges: bool = True
        self.debug_draw_mine_ranges: bool = True

        self.reset_default()

    # ------------------------------------------------------------------
    # External actions (trainer API)
    # ------------------------------------------------------------------
    def _external_key_for_agent(self, agent: Agent) -> str:
        """
        Canonical external-action key for this agent.
        If agent has slot_id, we prefer it. Otherwise: "{side}_{agent_id}".
        """
        if hasattr(agent, "slot_id"):
            try:
                return str(getattr(agent, "slot_id"))
            except Exception:
                pass
        return f"{agent.side}_{getattr(agent, 'agent_id', 0)}"

    def _external_key_candidates(self, agent: Agent) -> List[str]:
        """
        Accepts multiple possible keys so trainers can't get bricked by naming mismatches.
        """
        keys: List[str] = []
        # canonical
        keys.append(self._external_key_for_agent(agent))
        # common aliases
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

        # unique-ify (preserve order)
        seen = set()
        out: List[str] = []
        for k in keys:
            if k and k not in seen:
                out.append(k)
                seen.add(k)
        return out

    def submit_external_actions(self, actions_by_agent: Dict[str, Tuple[int, int]]) -> None:
        """
        Trainer-facing API: submit actions for this decision boundary.

        actions_by_agent:
          key -> (macro_val, target_idx)

        macro_val should be MacroAction.value (int) or int(MacroAction).
        target_idx is either macro target index or (x,y) tuple (handled in apply_macro_action).
        """
        if not isinstance(actions_by_agent, dict):
            return

        # Replace semantics: new submissions overwrite older ones for provided keys.
        for k, v in actions_by_agent.items():
            try:
                macro_val, target_idx = v
                self.pending_external_actions[str(k)] = (int(macro_val), int(target_idx))
            except Exception:
                # ignore malformed entries
                continue

    def _clear_pending_for_agent(self, agent: Agent) -> None:
        for k in self._external_key_candidates(agent):
            if k in self.pending_external_actions:
                self.pending_external_actions.pop(k, None)

    def _consume_external_action_for_agent(self, agent: Agent) -> Optional[Tuple[int, int]]:
        """
        Returns (macro_val, target_idx) if present, and consumes it.
        """
        for k in self._external_key_candidates(agent):
            if k in self.pending_external_actions:
                act = self.pending_external_actions.pop(k, None)
                if act is not None:
                    return act
        return None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
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
        rng = random.Random(seed if seed is not None else 1337)
        self.macro_targets.clear()

        free_cells = [
            (x, y)
            for y in range(self.row_count)
            for x in range(self.col_count)
            if self.grid[y][x] == 0
        ]
        if not free_cells:
            free_cells = [(x, y) for y in range(self.row_count) for x in range(self.col_count)]

        for _ in range(self.num_macro_targets):
            self.macro_targets.append(rng.choice(free_cells))

    def get_macro_target(self, index: int) -> Tuple[int, int]:
        if not self.macro_targets:
            self._init_macro_targets()
        if self.num_macro_targets <= 0:
            return self.col_count // 2, self.row_count // 2
        return self.macro_targets[int(index) % self.num_macro_targets]

    def get_all_macro_targets(self) -> List[Tuple[int, int]]:
        if not self.macro_targets:
            self._init_macro_targets()
        return list(self.macro_targets)

    def getGameManager(self) -> GameManager:
        return self.manager

    # ------------------------------------------------------------------
    # Policy wiring
    # ------------------------------------------------------------------
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
        """
        Map physical coordinates (x, y) in meters to CNN cell indices
        for a CNN_COLS × CNN_ROWS map.
        """
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

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset_default(self) -> None:
        self._init_zones()
        self.manager.reset_game()
        self.mines.clear()
        self.mine_pickups.clear()
        self._init_macro_targets(seed=self.agents_per_team)
        self.spawn_agents()
        self.spawn_mine_pickups()
        self.decision_cooldown_seconds_by_agent.clear()

        # 🛡️ guardrail: prevent inter-episode contamination
        self.pending_external_actions.clear()

    def set_agent_count_and_reset(self, new_count: int) -> None:
        if new_count < 1:
            new_count = 1
        self.agents_per_team = new_count
        self.manager.reset_game(reset_scores=True)
        self.reset_default()

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------
    def update(self, delta_time: float) -> None:
        if self.manager.game_over:
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

        # Tick ALL agents
        for agent in self.blue_agents + self.red_agents:
            agent.update(delta_time)

        # Dynamic obstacles: enabled agents only (mines NOT blocked, so they can be triggered)
        occupied = [(int(a.x), int(a.y)) for a in (self.blue_agents + self.red_agents) if a.isEnabled()]
        self.pathfinder.setDynamicObstacles(occupied)

        # Process both teams
        for friendly_team, enemy_team in (
            (self.blue_agents, self.red_agents),
            (self.red_agents, self.blue_agents),
        ):
            for agent in friendly_team:
                if not agent.isEnabled():
                    continue

                # Hazards
                self.apply_mine_damage(agent)
                self.apply_suppression(agent, enemy_team)

                # ✅ If we died, stop processing this agent this tick
                if not agent.isEnabled():
                    continue

                # Decision-making
                agent_key = id(agent)
                cooldown = self.decision_cooldown_seconds_by_agent.get(agent_key, 0.0) - delta_time
                self.decision_cooldown_seconds_by_agent[agent_key] = cooldown

                side_external = self.external_control_for_side.get(agent.side, False)

                if cooldown <= 0.0:
                    if side_external:
                        # External control path
                        act = self._consume_external_action_for_agent(agent)
                        if act is not None:
                            macro_val, target_idx = act
                            self.apply_macro_action(agent, MacroAction(int(macro_val)), int(target_idx))
                        else:
                            # Missing action behavior
                            if self.external_missing_action_mode == "internal" and self.use_internal_policies:
                                self.decide(agent)
                            # "idle" => do nothing (keep current path)
                        self.decision_cooldown_seconds_by_agent[agent_key] = self.decision_interval_seconds
                    else:
                        # Internal policy path
                        if self.use_internal_policies:
                            self.decide(agent)
                            self.decision_cooldown_seconds_by_agent[agent_key] = self.decision_interval_seconds

                # Pickups & flag rules
                self.handle_mine_pickups(agent)
                self.apply_flag_rules(agent)

        # Banner fade-out
        if self.banner_queue:
            text, color, t = self.banner_queue[-1]
            t = max(0.0, t - delta_time)
            self.banner_queue[-1] = (text, color, t)
            if t <= 0.0:
                self.banner_queue.pop()

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def build_observation(self, agent: Agent) -> List[List[List[float]]]:
        side = agent.side
        game_state = self.manager

        channels: List[List[List[float]]] = [
            [[0.0 for _ in range(CNN_COLS)] for _ in range(CNN_ROWS)]
            for _ in range(NUM_CNN_CHANNELS)
        ]

        def get_cnn_cell(col: int, row: int) -> Tuple[int, int]:
            x_m, y_m = self.grid_to_world(col, row)
            u = max(0.0, min(1.0, x_m / ARENA_WIDTH_M))
            v = max(0.0, min(1.0, y_m / ARENA_HEIGHT_M))

            if side == "red":
                u = 1.0 - u  # mirror across vertical axis only

            col_cnn = max(0, min(CNN_COLS - 1, int(u * CNN_COLS)))
            row_cnn = max(0, min(CNN_ROWS - 1, int(v * CNN_ROWS)))
            return col_cnn, row_cnn

        def set_chan(c: int, col: int, row: int) -> None:
            if 0 <= col < CNN_COLS and 0 <= row < CNN_ROWS:
                channels[c][row][col] = 1.0

        # Channel 0: self + identity encoding
        own_c, own_r = get_cnn_cell(int(agent.x), int(agent.y))
        agent_id = getattr(agent, "agent_id", 0)
        channels[0][own_r][own_c] = 1.0 if agent_id == 0 else 0.5

        friendly_team = self.blue_agents if side == "blue" else self.red_agents
        enemy_team = self.red_agents if side == "blue" else self.blue_agents

        # Channel 1: teammates
        for a in friendly_team:
            if a is agent or not a.isEnabled():
                continue
            c, r = get_cnn_cell(int(a.x), int(a.y))
            set_chan(1, c, r)

        # Channel 2: enemies
        for a in enemy_team:
            if not a.isEnabled():
                continue
            c, r = get_cnn_cell(int(a.x), int(a.y))
            set_chan(2, c, r)

        # Channel 3/4: mines
        for m in self.mines:
            c, r = get_cnn_cell(int(m.x), int(m.y))
            if m.owner_side == side:
                set_chan(3, c, r)
            else:
                set_chan(4, c, r)

        # Channel 5/6: flags (robust to carried/dropped states)
        if side == "blue":
            own_flag_pos = game_state.blue_flag_position
            enemy_flag_pos = game_state.get_enemy_flag_position("blue")
        else:
            own_flag_pos = game_state.red_flag_position
            enemy_flag_pos = game_state.get_enemy_flag_position("red")

        # Clamp just in case
        own_fx = max(0, min(self.col_count - 1, int(own_flag_pos[0])))
        own_fy = max(0, min(self.row_count - 1, int(own_flag_pos[1])))
        enm_fx = max(0, min(self.col_count - 1, int(enemy_flag_pos[0])))
        enm_fy = max(0, min(self.row_count - 1, int(enemy_flag_pos[1])))

        c1, r1 = get_cnn_cell(own_fx, own_fy)
        set_chan(5, c1, r1)

        c2, r2 = get_cnn_cell(enm_fx, enm_fy)
        set_chan(6, c2, r2)

        return channels

    # ------------------------------------------------------------------
    # Macro actions
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

    def apply_macro_action(self, agent: Agent, action: Any, param: Optional[Any] = None) -> None:
        if not agent.isEnabled():
            return

        # normalize action to MacroAction
        if not isinstance(action, MacroAction):
            action = MacroAction(int(action))

        agent.last_macro_action = action
        side = agent.side
        gm = self.manager

        start_pos = (int(agent.x), int(agent.y))
        new_pos = start_pos

        def resolve_target_from_param(default_target: Tuple[int, int]) -> Tuple[int, int]:
            if param is None:
                return default_target

            if isinstance(param, (tuple, list)) and len(param) == 2:
                return int(param[0]), int(param[1])

            try:
                idx = int(param)
            except (TypeError, ValueError):
                return default_target

            return self.get_macro_target(idx)

        def safe_set_path(target: Tuple[int, int], avoid_enemies: bool = False, radius: int = 1) -> Tuple[int, int]:
            original_blocked = set(self.pathfinder.blocked)
            blocked = set(original_blocked)

            start = (int(agent.x), int(agent.y))
            tgt = (int(target[0]), int(target[1]))

            # Bounds clamp for safety
            tgt = (max(0, min(self.col_count - 1, tgt[0])), max(0, min(self.row_count - 1, tgt[1])))

            # Avoid enemy "danger bubble"
            if avoid_enemies:
                enemy_team = self.red_agents if side == "blue" else self.blue_agents
                for e in enemy_team:
                    if e.isEnabled():
                        ex, ey = int(e.x), int(e.y)
                        for dx in range(-radius, radius + 1):
                            for dy in range(-radius, radius + 1):
                                cx, cy = ex + dx, ey + dy
                                if 0 <= cx < self.col_count and 0 <= cy < self.row_count:
                                    blocked.add((cx, cy))

            # CRITICAL: never block your own start or your target
            blocked.discard(start)
            blocked.discard(tgt)

            try:
                self.pathfinder.blocked = blocked
                path = self.pathfinder.astar(start, tgt)
            finally:
                self.pathfinder.blocked = original_blocked

            # None means "no path", [] is valid ("already at goal")
            if path is None:
                fallback = self.random_point_in_own_half(side)
                fallback = (int(fallback[0]), int(fallback[1]))
                try:
                    self.pathfinder.blocked = blocked
                    path = self.pathfinder.astar(start, fallback)
                finally:
                    self.pathfinder.blocked = original_blocked

                if path is None:
                    path = []

            agent.setPath(path)

            if len(path) > 0:
                end_cell = path[-1]
                return (int(end_cell[0]), int(end_cell[1]))
            return start_pos

        if action == MacroAction.GO_TO:
            default_target = self.random_point_in_enemy_half(side)
            target = resolve_target_from_param(default_target)
            new_pos = safe_set_path(target, avoid_enemies=agent.isCarryingFlag())

        elif action == MacroAction.GRAB_MINE:
            my_pickups = [p for p in self.mine_pickups if p.owner_side == side]
            if my_pickups:
                nearest = min(my_pickups, key=lambda p: (p.x - agent.x) ** 2 + (p.y - agent.y) ** 2)
                new_pos = safe_set_path((nearest.x, nearest.y))
            else:
                new_pos = safe_set_path(self.random_point_in_own_half(side))

        elif action == MacroAction.GET_FLAG:
            ex, ey = gm.get_enemy_flag_position(side)
            new_pos = safe_set_path((ex, ey), avoid_enemies=agent.isCarryingFlag())

        elif action == MacroAction.PLACE_MINE:
            default_target = (int(agent.x), int(agent.y))
            target = resolve_target_from_param(default_target)

            if agent.mine_charges > 0:
                own_flag_pos = gm.blue_flag_home if side == "blue" else gm.red_flag_home

                if not any(m.x == target[0] and m.y == target[1] for m in self.mines):
                    if target != own_flag_pos:
                        self.mines.append(
                            Mine(
                                x=int(target[0]),
                                y=int(target[1]),
                                owner_side=side,
                                owner_id=getattr(agent, "unique_id", None),
                            )
                        )
                        agent.mine_charges -= 1
                        self.manager.reward_mine_placed(agent, mine_pos=target)

            new_pos = safe_set_path(target)

        elif action == MacroAction.GO_HOME:
            home = gm.get_team_zone_center(side)
            new_pos = safe_set_path(home, avoid_enemies=True, radius=2)

        if hasattr(gm, "reward_potential_shaping"):
            gm.reward_potential_shaping(agent, start_pos, new_pos)

    # ------------------------------------------------------------------
    # Internal decision
    # ------------------------------------------------------------------
    def decide(self, agent: Agent) -> None:
        if not agent.isEnabled():
            return

        agent.decision_count = getattr(agent, "decision_count", 0) + 1

        obs = self.build_observation(agent)
        policy = self.policies.get(agent.side)

        # NN-style policy
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

        # Scripted baseline Policy class
        elif isinstance(policy, Policy):
            action_id, param = policy.select_action(obs, agent, self)

        # Callable fallback
        else:
            action_id = policy(agent, self)
            param = None

        self.apply_macro_action(agent, MacroAction(int(action_id)), param)

    # ------------------------------------------------------------------
    # Mines / suppression / flags
    # ------------------------------------------------------------------
    def handle_mine_pickups(self, agent: Agent) -> None:
        if not agent.isEnabled():
            return

        for pickup in list(self.mine_pickups):
            if pickup.owner_side != agent.side:
                continue

            if int(agent.x) == pickup.x and int(agent.y) == pickup.y:
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
            if mine.owner_side == agent.side:
                continue

            dist = math.hypot(mine.x - agent.x, mine.y - agent.y)
            if dist <= self.mine_radius_cells:
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
                    self.manager.reward_enemy_killed(killer_agent=killer_agent, victim_agent=agent, cause="mine")

                if mine.owner_side == "blue" and agent.side == "red":
                    self.manager.record_mine_triggered_by_red()

                # clear any queued external action so respawn doesn't execute stale commands
                self._clear_pending_for_agent(agent)

                agent.disable_for_seconds(self.respawn_seconds)
                self.mines.remove(mine)
                break

    def apply_suppression(self, agent: Agent, enemies: List[Agent]) -> None:
        if not agent.isEnabled():
            return

        close_enemies = [
            e for e in enemies
            if e.isEnabled() and math.hypot(e.x - agent.x, e.y - agent.y) <= self.suppression_range_cells
        ]

        if len(close_enemies) >= 2:
            blue_suppressors = [e for e in close_enemies if getattr(e, "side", None) == "blue"]
            killer_agent = blue_suppressors[0] if blue_suppressors else close_enemies[0]

            if killer_agent is not None and getattr(killer_agent, "side", None) == "blue":
                self.manager.reward_enemy_killed(killer_agent=killer_agent, victim_agent=agent, cause="suppression")

            # clear any queued external action so respawn doesn't execute stale commands
            self._clear_pending_for_agent(agent)

            agent.disable_for_seconds(self.respawn_seconds)

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

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def announce(self, text: str, color=(255, 255, 255), seconds: float = 2.0) -> None:
        self.banner_queue.append((text, color, seconds))

    # ------------------------------------------------------------------
    # Spawning
    # ------------------------------------------------------------------
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

        # Also clear pending actions on respawn cycle
        self.pending_external_actions.clear()

        blue_min_col, blue_max_col = self.blue_zone_col_range
        red_min_col, red_max_col = self.red_zone_col_range

        blue_cells = [(r, c) for r in range(self.row_count) for c in range(blue_min_col, blue_max_col + 1)]
        red_cells = [(r, c) for r in range(self.row_count) for c in range(red_min_col, red_max_col + 1)]
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
        self.mine_pickups.clear()
        rng = random.Random(self.agents_per_team + 9999)

        blue_min_col, blue_max_col = self.blue_zone_col_range
        red_min_col, red_max_col = self.red_zone_col_range

        def spawn_for_band(owner_side: str, col_min: int, col_max: int) -> None:
            all_cells = [(row, col) for row in range(self.row_count) for col in range(col_min, col_max + 1)]
            rng.shuffle(all_cells)

            flag_pos = self.manager.blue_flag_position if owner_side == "blue" else self.manager.red_flag_position
            placed = 0

            for row, col in all_cells:
                if (col, row) == flag_pos:
                    continue
                self.mine_pickups.append(MinePickup(x=col, y=row, owner_side=owner_side, charges=1))
                placed += 1
                if placed >= self.mines_per_team:
                    break

        spawn_for_band("blue", blue_min_col, blue_max_col)
        spawn_for_band("red", red_min_col, red_max_col)


__all__ = ["GameField", "MacroAction", "Mine", "MinePickup"]
