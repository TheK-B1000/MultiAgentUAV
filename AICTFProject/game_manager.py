# =========================
# game_manager.py (FULL UPDATED)
# =========================

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any, Set

# ==========================================================
# ⚙️ REWARD / GAME CONSTANTS
# ==========================================================
WIN_TEAM_REWARD = 5.0
FLAG_PICKUP_REWARD = 0.5
FLAG_CARRY_HOME_REWARD = 3.0
ENEMY_MAV_KILL_REWARD = 2.0
ACTION_FAILED_PUNISHMENT = -0.5

FLAG_RETURN_DELAY = 10.0
FLAG_PROXIMITY_COEF = 0.2  # dense shaping coefficient

# Optional extras (safe defaults)
EXPLORATION_REWARD = 0.01
COORDINATION_BONUS = 0.3

PHASE_DRAW_TIMEOUT_PENALTY = {
    "OP1": 0.0,
    "OP2": 0.0,
    "OP3": 0.0,
    "SELF": 0.0,
}


@dataclass
class GameManager:
    cols: int
    rows: int

    # Scores / timers
    blue_score: int = 0
    red_score: int = 0
    max_time: float = 200.0
    current_time: float = 200.0
    game_over: bool = False
    score_limit: int = 3

    # Flags
    blue_flag_position: Tuple[int, int] = (0, 0)
    red_flag_position: Tuple[int, int] = (0, 0)
    blue_flag_taken: bool = False
    red_flag_taken: bool = False
    blue_flag_carrier: Optional[Any] = None
    red_flag_carrier: Optional[Any] = None

    blue_flag_home: Tuple[int, int] = (0, 0)
    red_flag_home: Tuple[int, int] = (0, 0)

    blue_flag_drop_time: Optional[float] = None
    red_flag_drop_time: Optional[float] = None

    # Time / reward events
    sim_time: float = 0.0
    reward_events: List[Tuple[float, Optional[str], float]] = field(default_factory=list)

    phase_name: str = "OP1"

    # Minimal mine telemetry
    blue_mine_kills_this_episode: int = 0
    red_mine_kills_this_episode: int = 0
    mines_placed_in_enemy_half_this_episode: int = 0
    mines_triggered_by_red_this_episode: int = 0

    # ✅ NEW: visited-set for exploration bonus (cell-quantized)
    blue_visited_cells: Set[Tuple[int, int]] = field(default_factory=set)

    # ----------------------------------------------------------
    # Phase / reset
    # ----------------------------------------------------------
    def set_phase(self, phase: str) -> None:
        self.phase_name = str(phase)

    def _clamp_cell(self, x: int, y: int) -> Tuple[int, int]:
        return (max(0, min(self.cols - 1, int(x))), max(0, min(self.rows - 1, int(y))))

    def reset_game(self, reset_scores: bool = True) -> None:
        if reset_scores:
            self.blue_score = 0
            self.red_score = 0

        self.game_over = False
        self.current_time = self.max_time
        self.sim_time = 0.0
        self.reward_events.clear()

        # Reset mine telemetry
        self.blue_mine_kills_this_episode = 0
        self.red_mine_kills_this_episode = 0
        self.mines_placed_in_enemy_half_this_episode = 0
        self.mines_triggered_by_red_this_episode = 0

        # ✅ reset exploration memory
        self.blue_visited_cells.clear()

        mid_row = self.rows // 2
        self.blue_flag_home = self._clamp_cell(2, mid_row)
        self.red_flag_home = self._clamp_cell(self.cols - 3, mid_row)

        self.blue_flag_position = self.blue_flag_home
        self.red_flag_position = self.red_flag_home

        self.blue_flag_taken = False
        self.red_flag_taken = False
        self.blue_flag_carrier = None
        self.red_flag_carrier = None

        self.blue_flag_drop_time = None
        self.red_flag_drop_time = None

    # ----------------------------------------------------------
    # Tick / termination logic
    # ----------------------------------------------------------
    def tick_seconds(self, dt: float) -> Optional[str]:
        if self.game_over or dt <= 0.0:
            return None

        self.sim_time += float(dt)
        self.current_time -= float(dt)

        # Keep flag state consistent every tick
        self.sanity_check_flags()
        self._update_flag_auto_return()

        # Time over
        if self.current_time <= 0.0 and not self.game_over:
            self.game_over = True
            if self.blue_score > self.red_score:
                self.add_reward_event(WIN_TEAM_REWARD)
                return "BLUE WINS ON TIME"
            elif self.red_score > self.blue_score:
                return "RED WINS ON TIME"
            else:
                penalty = PHASE_DRAW_TIMEOUT_PENALTY.get(self.phase_name, 0.0)
                if penalty != 0.0:
                    self.add_reward_event(penalty)
                    return f"DRAW — BLUE PENALIZED FOR STALL ({self.phase_name})"
                return f"DRAW — NO PENALTY ({self.phase_name})"

        # Score limit
        if self.blue_score >= self.score_limit:
            self.game_over = True
            self.add_reward_event(WIN_TEAM_REWARD)
            return "BLUE WINS BY SCORE!"

        if self.red_score >= self.score_limit:
            self.game_over = True
            return "RED WINS BY SCORE!"

        return None

    def _update_flag_auto_return(self) -> None:
        if (not self.blue_flag_taken
                and self.blue_flag_position != self.blue_flag_home
                and self.blue_flag_drop_time is not None):
            if self.sim_time - self.blue_flag_drop_time >= FLAG_RETURN_DELAY:
                self.blue_flag_position = self.blue_flag_home
                self.blue_flag_drop_time = None

        if (not self.red_flag_taken
                and self.red_flag_position != self.red_flag_home
                and self.red_flag_drop_time is not None):
            if self.sim_time - self.red_flag_drop_time >= FLAG_RETURN_DELAY:
                self.red_flag_position = self.red_flag_home
                self.red_flag_drop_time = None

    # ----------------------------------------------------------
    # Flag sanity / helpers
    # ----------------------------------------------------------
    def sanity_check_flags(self) -> None:
        # Blue flag
        if self.blue_flag_taken:
            if self.blue_flag_carrier is None:
                self.blue_flag_taken = False
                self.blue_flag_position = self.blue_flag_home
                self.blue_flag_drop_time = None
            else:
                if not getattr(self.blue_flag_carrier, "isEnabled", lambda: True)():
                    self.drop_flag_if_carrier_disabled(self.blue_flag_carrier)
                else:
                    self.blue_flag_position = self._agent_cell(self.blue_flag_carrier)
                    self.blue_flag_drop_time = None

        # Red flag
        if self.red_flag_taken:
            if self.red_flag_carrier is None:
                self.red_flag_taken = False
                self.red_flag_position = self.red_flag_home
                self.red_flag_drop_time = None
            else:
                if not getattr(self.red_flag_carrier, "isEnabled", lambda: True)():
                    self.drop_flag_if_carrier_disabled(self.red_flag_carrier)
                else:
                    self.red_flag_position = self._agent_cell(self.red_flag_carrier)
                    self.red_flag_drop_time = None

    def _agent_cell(self, agent: Any) -> Tuple[int, int]:
        if hasattr(agent, "float_pos"):
            fx, fy = agent.float_pos
            return self._clamp_cell(int(round(fx)), int(round(fy)))
        if hasattr(agent, "get_position"):
            x, y = agent.get_position()
            return self._clamp_cell(int(x), int(y))
        return self._clamp_cell(getattr(agent, "x", 0), getattr(agent, "y", 0))

    def get_enemy_flag_position(self, side: str) -> Tuple[int, int]:
        side = str(side)
        if side == "blue":
            if self.red_flag_taken and self.red_flag_carrier is not None:
                return self._agent_cell(self.red_flag_carrier)
            return self.red_flag_position
        else:
            if self.blue_flag_taken and self.blue_flag_carrier is not None:
                return self._agent_cell(self.blue_flag_carrier)
            return self.blue_flag_position

    def get_team_zone_center(self, side: str) -> Tuple[int, int]:
        return self.blue_flag_home if str(side) == "blue" else self.red_flag_home

    def get_sim_time(self) -> float:
        return float(self.sim_time)

    # ----------------------------------------------------------
    # Flag interactions
    # ----------------------------------------------------------
    def try_pickup_enemy_flag(self, agent: Any) -> bool:
        side = getattr(agent, "side", None)
        if side not in ("blue", "red"):
            return False

        pos_x, pos_y = getattr(agent, "float_pos", (float(getattr(agent, "x", 0)), float(getattr(agent, "y", 0))))

        if side == "blue":
            enemy_taken = self.red_flag_taken
            enemy_pos = self.red_flag_position
        else:
            enemy_taken = self.blue_flag_taken
            enemy_pos = self.blue_flag_position

        if enemy_taken:
            return False

        if math.hypot(float(pos_x) - float(enemy_pos[0]), float(pos_y) - float(enemy_pos[1])) > 1.0:
            return False

        # Take it
        if side == "blue":
            self.red_flag_taken = True
            self.red_flag_carrier = agent
            self.red_flag_position = self._agent_cell(agent)
            self.red_flag_drop_time = None
        else:
            self.blue_flag_taken = True
            self.blue_flag_carrier = agent
            self.blue_flag_position = self._agent_cell(agent)
            self.blue_flag_drop_time = None

        if hasattr(agent, "setCarryingFlag"):
            agent.setCarryingFlag(True)

        uid = getattr(agent, "unique_id", None)
        self.add_reward_event(FLAG_PICKUP_REWARD, agent_id=uid)

        # Optional coordination bonus (only meaningful if wired)
        if self._teammate_near(agent):
            self.add_reward_event(COORDINATION_BONUS, agent_id=uid)

        return True

    def try_score_if_carrying_and_home(self, agent: Any) -> bool:
        side = getattr(agent, "side", None)
        if side not in ("blue", "red"):
            return False

        pos_x, pos_y = getattr(agent, "float_pos", (float(getattr(agent, "x", 0)), float(getattr(agent, "y", 0))))

        # Blue scores with red flag at blue home
        if side == "blue" and self.red_flag_taken and (self.red_flag_carrier is agent):
            if math.hypot(float(pos_x) - float(self.blue_flag_home[0]), float(pos_y) - float(self.blue_flag_home[1])) <= 2.0:
                self.blue_score += 1

                self.red_flag_taken = False
                self.red_flag_carrier = None
                self.red_flag_position = self.red_flag_home
                self.red_flag_drop_time = None

                if hasattr(agent, "setCarryingFlag"):
                    agent.setCarryingFlag(False, scored=True)

                uid = getattr(agent, "unique_id", None)
                self.add_reward_event(FLAG_CARRY_HOME_REWARD, agent_id=uid)
                self.add_reward_event(FLAG_CARRY_HOME_REWARD * 0.5)

                if self._teammate_near(agent):
                    self.add_reward_event(COORDINATION_BONUS)

                return True

        # Red scores with blue flag at red home (no penalty to BLUE by default)
        if side == "red" and self.blue_flag_taken and (self.blue_flag_carrier is agent):
            if math.hypot(float(pos_x) - float(self.red_flag_home[0]), float(pos_y) - float(self.red_flag_home[1])) <= 2.0:
                self.red_score += 1

                self.blue_flag_taken = False
                self.blue_flag_carrier = None
                self.blue_flag_position = self.blue_flag_home
                self.blue_flag_drop_time = None

                if hasattr(agent, "setCarryingFlag"):
                    agent.setCarryingFlag(False, scored=True)

                return True

        return False

    def drop_flag_if_carrier_disabled(self, agent: Any) -> None:
        drop_pos = self._agent_cell(agent)
        uid = getattr(agent, "unique_id", None)

        if self.blue_flag_carrier is agent:
            self.blue_flag_taken = False
            self.blue_flag_carrier = None
            self.blue_flag_position = drop_pos
            self.blue_flag_drop_time = self.sim_time
            self.add_reward_event(ACTION_FAILED_PUNISHMENT, agent_id=uid)
            if hasattr(agent, "setCarryingFlag"):
                agent.setCarryingFlag(False, scored=False)

        elif self.red_flag_carrier is agent:
            self.red_flag_taken = False
            self.red_flag_carrier = None
            self.red_flag_position = drop_pos
            self.red_flag_drop_time = self.sim_time
            self.add_reward_event(ACTION_FAILED_PUNISHMENT, agent_id=uid)
            if hasattr(agent, "setCarryingFlag"):
                agent.setCarryingFlag(False, scored=False)

    def handle_agent_death(self, agent: Any) -> None:
        self.drop_flag_if_carrier_disabled(agent)

    def clear_flag_carrier_if_agent(self, agent: Any) -> None:
        if self.blue_flag_carrier is agent:
            self.blue_flag_carrier = None
            self.blue_flag_taken = False
            self.blue_flag_position = self.blue_flag_home
            self.blue_flag_drop_time = None
            if hasattr(agent, "setCarryingFlag"):
                agent.setCarryingFlag(False, scored=False)

        if self.red_flag_carrier is agent:
            self.red_flag_carrier = None
            self.red_flag_taken = False
            self.red_flag_position = self.red_flag_home
            self.red_flag_drop_time = None
            if hasattr(agent, "setCarryingFlag"):
                agent.setCarryingFlag(False, scored=False)

    # ----------------------------------------------------------
    # ✅ TEAMMATE NEAR (WORKS if agent.game_field is set)
    # ----------------------------------------------------------
    def _teammate_near(self, agent: Any, radius_cells: float = 5.0) -> bool:
        gf = getattr(agent, "game_field", None)
        if gf is None:
            return False

        side = getattr(agent, "side", None)
        if side not in ("blue", "red"):
            return False

        team = gf.blue_agents if side == "blue" else gf.red_agents
        ax, ay = getattr(agent, "float_pos", (float(getattr(agent, "x", 0)), float(getattr(agent, "y", 0))))

        for other in team:
            if other is agent or (not other.isEnabled()):
                continue
            ox, oy = getattr(other, "float_pos", (float(getattr(other, "x", 0)), float(getattr(other, "y", 0))))
            if math.hypot(float(ox) - float(ax), float(oy) - float(ay)) <= float(radius_cells):
                return True

        return False

    # ----------------------------------------------------------
    # ✅ POTENTIAL-BASED SHAPING (EXECUTED MOTION + LEAKAGE FIX + EXPLORATION)
    # ----------------------------------------------------------
    def reward_potential_shaping(self, agent: Any, start_pos: Tuple[float, float], end_pos: Tuple[float, float]) -> None:
        """
        Movement-based shaping called from Agent.update() using executed float positions.

        Includes:
          - goal leakage fix (non-carrier doesn't chase teammate-carrier)
          - exploration bonus (quantized + set, not float tuples)
        """
        if getattr(agent, "side", None) != "blue":
            return

        # Goal selection (fix clumping/leakage)
        i_am_carrier = (self.red_flag_taken and (self.red_flag_carrier is agent))
        teammate_is_carrier = (self.red_flag_taken and (self.red_flag_carrier is not None) and (self.red_flag_carrier is not agent))

        if i_am_carrier:
            goal_x, goal_y = self.blue_flag_home
        elif teammate_is_carrier:
            # escort/defense prep instead of clumping onto carrier
            goal_x, goal_y = self.blue_flag_home
        else:
            goal_x, goal_y = self.get_enemy_flag_position("blue")

        max_dist = math.sqrt(self.cols ** 2 + self.rows ** 2)
        if max_dist <= 0.0:
            return

        sx, sy = float(start_pos[0]), float(start_pos[1])
        ex, ey = float(end_pos[0]), float(end_pos[1])

        prev_d = max(0.0, min(math.dist([sx, sy], [goal_x, goal_y]), max_dist))
        cur_d  = max(0.0, min(math.dist([ex, ey], [goal_x, goal_y]), max_dist))

        phi_before = 1.0 - (prev_d / max_dist)
        phi_after  = 1.0 - (cur_d  / max_dist)

        shaped = float(FLAG_PROXIMITY_COEF) * (phi_after - phi_before)
        if shaped != 0.0:
            self.add_reward_event(shaped, agent_id=getattr(agent, "unique_id", None))

        # Exploration bonus (quantize to cell + set)
        cell = (int(round(ex)), int(round(ey)))
        cell = self._clamp_cell(cell[0], cell[1])
        if cell not in self.blue_visited_cells:
            self.blue_visited_cells.add(cell)
            self.add_reward_event(EXPLORATION_REWARD, agent_id=getattr(agent, "unique_id", None))

    # ----------------------------------------------------------
    # Mine / combat rewards (minimal)
    # ----------------------------------------------------------
    def reward_mine_placed(self, agent: Any, mine_pos: Optional[Tuple[int, int]] = None) -> None:
        # By default: no immediate reward for placement (prevents spam).
        # Keep telemetry.
        if mine_pos is not None and getattr(agent, "side", None) == "blue":
            x, _ = mine_pos
            if x > (self.cols * 0.5):
                self.mines_placed_in_enemy_half_this_episode += 1

    def reward_enemy_killed(self, killer_agent: Any, victim_agent: Optional[Any] = None, cause: Optional[str] = None) -> None:
        if cause == "mine":
            if getattr(killer_agent, "side", None) == "blue":
                self.blue_mine_kills_this_episode += 1
            else:
                self.red_mine_kills_this_episode += 1

        self.add_reward_event(ENEMY_MAV_KILL_REWARD, agent_id=getattr(killer_agent, "unique_id", None))

        # Optional small team bonus for mine kills
        if getattr(killer_agent, "side", None) == "blue" and cause == "mine":
            self.add_reward_event(ENEMY_MAV_KILL_REWARD * 0.5)

    def record_mine_triggered_by_red(self) -> None:
        self.mines_triggered_by_red_this_episode += 1

    def punish_failed_action(self, agent: Any) -> None:
        self.add_reward_event(ACTION_FAILED_PUNISHMENT, agent_id=getattr(agent, "unique_id", None))

    # ----------------------------------------------------------
    # Reward events API
    # ----------------------------------------------------------
    def add_reward_event(self, value: float, timestamp: Optional[float] = None, agent_id: Optional[str] = None) -> None:
        t = self.sim_time if timestamp is None else float(timestamp)
        self.reward_events.append((t, agent_id, float(value)))

    def pop_reward_events(self) -> List[Tuple[float, Optional[str], float]]:
        events = self.reward_events
        self.reward_events = []
        return events
