# =========================
# game_manager.py (RESEARCH-GRADE ROUTING + GAMMA-PBRS)
# =========================

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Any, Set, Dict

# ==========================================================
# ⚙️ REWARD / GAME CONSTANTS
# ==========================================================
WIN_TEAM_REWARD = 5.0
FLAG_PICKUP_REWARD = 0.5
FLAG_CARRY_HOME_REWARD = 3.0
ENEMY_MAV_KILL_REWARD = 2.0
ACTION_FAILED_PUNISHMENT = -0.5

FLAG_RETURN_DELAY = 10.0

# PBRS parameters
FLAG_PROXIMITY_COEF = 0.2  # shaping coefficient (scale)
DEFAULT_SHAPING_GAMMA = 0.99  # ✅ gamma-correct PBRS term

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
    reward_events: List[Tuple[float, str, float]] = field(default_factory=list)
    #                (time, agent_id, reward_value)

    phase_name: str = "OP1"

    # Minimal mine telemetry
    blue_mine_kills_this_episode: int = 0
    red_mine_kills_this_episode: int = 0
    mines_placed_in_enemy_half_this_episode: int = 0
    mines_triggered_by_red_this_episode: int = 0

    # Exploration memory (team-level)
    blue_visited_cells: Set[Tuple[int, int]] = field(default_factory=set)

    # ✅ ROUTING: remember agent ids seen this episode (so terminal rewards can route)
    blue_agent_ids_seen: Set[str] = field(default_factory=set)
    red_agent_ids_seen: Set[str] = field(default_factory=set)

    # ✅ PBRS gamma (trainer can set this to match PPO/MAPPO gamma)
    shaping_gamma: float = DEFAULT_SHAPING_GAMMA

    # Optional: bind back to env for teammate routing (not required if ids_seen filled)
    game_field: Optional[Any] = field(default=None, repr=False, compare=False)

    # ----------------------------------------------------------
    # Binding / configuration
    # ----------------------------------------------------------
    def bind_game_field(self, game_field: Any) -> None:
        """Call once from GameField.__init__ (recommended)."""
        self.game_field = game_field

    def set_phase(self, phase: str) -> None:
        self.phase_name = str(phase)

    def set_shaping_gamma(self, gamma: float) -> None:
        g = float(gamma)
        if not (0.0 <= g <= 1.0):
            raise ValueError(f"gamma must be in [0,1], got {gamma}")
        self.shaping_gamma = g

    # ----------------------------------------------------------
    # Core helpers
    # ----------------------------------------------------------
    def _clamp_cell(self, x: int, y: int) -> Tuple[int, int]:
        return (max(0, min(self.cols - 1, int(x))), max(0, min(self.rows - 1, int(y))))

    def _agent_cell(self, agent: Any) -> Tuple[int, int]:
        if hasattr(agent, "float_pos"):
            fx, fy = agent.float_pos
            return self._clamp_cell(int(round(fx)), int(round(fy)))
        if hasattr(agent, "get_position"):
            x, y = agent.get_position()
            return self._clamp_cell(int(x), int(y))
        return self._clamp_cell(getattr(agent, "x", 0), getattr(agent, "y", 0))

    def _agent_uid(self, agent: Any) -> str:
        """
        Always returns a non-empty string id.
        Prefer agent.unique_id (stable across episode), else fall back to python id().
        """
        uid = getattr(agent, "unique_id", None)
        if uid is None or str(uid).strip() == "":
            return str(id(agent))
        return str(uid)

    def _remember_agent(self, agent: Any) -> None:
        side = getattr(agent, "side", None)
        uid = self._agent_uid(agent)
        if side == "blue":
            self.blue_agent_ids_seen.add(uid)
        elif side == "red":
            self.red_agent_ids_seen.add(uid)

    # ----------------------------------------------------------
    # ✅ Reward routing (NO agent_id=None EVER)
    # ----------------------------------------------------------
    def add_reward_event(self, value: float, agent_id: str, timestamp: Optional[float] = None) -> None:
        """
        Low-level event append. agent_id MUST be a string.
        """
        if agent_id is None or str(agent_id).strip() == "":
            raise ValueError("agent_id must be a non-empty string (no globals allowed).")
        t = self.sim_time if timestamp is None else float(timestamp)
        self.reward_events.append((t, str(agent_id), float(value)))

    def add_agent_reward(self, agent: Any, value: float, timestamp: Optional[float] = None) -> None:
        """
        Route reward to exactly one agent.
        """
        if agent is None:
            return
        self._remember_agent(agent)
        self.add_reward_event(value, agent_id=self._agent_uid(agent), timestamp=timestamp)

    def add_team_reward(
        self,
        side: str,
        value: float,
        timestamp: Optional[float] = None,
        exclude_agent: Optional[Any] = None,
        include_disabled: bool = False,
    ) -> None:
        """
        Route reward to all agents on a side by emitting per-agent events.
        This keeps trainer logic simple and eliminates ambiguous "global" rewards.
        """
        side = str(side)
        ex_uid = self._agent_uid(exclude_agent) if exclude_agent is not None else None

        # Best: use bound game_field for exact membership
        gf = self.game_field
        if gf is not None:
            team = gf.blue_agents if side == "blue" else gf.red_agents
            for a in team:
                if a is None:
                    continue
                if (not include_disabled) and hasattr(a, "isEnabled") and (not a.isEnabled()):
                    continue
                uid = self._agent_uid(a)
                if ex_uid is not None and uid == ex_uid:
                    continue
                self._remember_agent(a)
                self.add_reward_event(value, agent_id=uid, timestamp=timestamp)
            return

        # Fallback: route to ids we've seen this episode (still deterministic & trainer-safe)
        ids = self.blue_agent_ids_seen if side == "blue" else self.red_agent_ids_seen
        for uid in ids:
            if ex_uid is not None and uid == ex_uid:
                continue
            self.add_reward_event(value, agent_id=uid, timestamp=timestamp)

    # ----------------------------------------------------------
    # Reset
    # ----------------------------------------------------------
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

        # Exploration memory
        self.blue_visited_cells.clear()

        # ✅ routing memory
        self.blue_agent_ids_seen.clear()
        self.red_agent_ids_seen.clear()

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
                # ✅ team-routed (no globals)
                self.add_team_reward("blue", WIN_TEAM_REWARD)
                return "BLUE WINS ON TIME"
            elif self.red_score > self.blue_score:
                return "RED WINS ON TIME"
            else:
                penalty = PHASE_DRAW_TIMEOUT_PENALTY.get(self.phase_name, 0.0)
                if penalty != 0.0:
                    # If you ever use draw penalties, route them explicitly too
                    self.add_team_reward("blue", penalty)
                    return f"DRAW — BLUE PENALIZED FOR STALL ({self.phase_name})"
                return f"DRAW — NO PENALTY ({self.phase_name})"

        # Score limit
        if self.blue_score >= self.score_limit:
            self.game_over = True
            self.add_team_reward("blue", WIN_TEAM_REWARD)
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

        self._remember_agent(agent)

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

        # ✅ pickup reward is for the picker
        self.add_agent_reward(agent, FLAG_PICKUP_REWARD)

        # Coordination bonus: keep it per-agent (picker), not ambiguous
        if self._teammate_near(agent):
            self.add_agent_reward(agent, COORDINATION_BONUS)

        return True

    def try_score_if_carrying_and_home(self, agent: Any) -> bool:
        side = getattr(agent, "side", None)
        if side not in ("blue", "red"):
            return False

        self._remember_agent(agent)

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

                # ✅ carrier gets main score reward
                self.add_agent_reward(agent, FLAG_CARRY_HOME_REWARD)

                # ✅ team share is explicit, and EXCLUDES the carrier (matches your old intent)
                self.add_team_reward("blue", FLAG_CARRY_HOME_REWARD * 0.5, exclude_agent=agent)

                # Optional coordination: keep explicit
                if self._teammate_near(agent):
                    self.add_agent_reward(agent, COORDINATION_BONUS)

                return True

        # Red scores with blue flag at red home
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
        self._remember_agent(agent)

        if self.blue_flag_carrier is agent:
            self.blue_flag_taken = False
            self.blue_flag_carrier = None
            self.blue_flag_position = drop_pos
            self.blue_flag_drop_time = self.sim_time
            self.add_agent_reward(agent, ACTION_FAILED_PUNISHMENT)
            if hasattr(agent, "setCarryingFlag"):
                agent.setCarryingFlag(False, scored=False)

        elif self.red_flag_carrier is agent:
            self.red_flag_taken = False
            self.red_flag_carrier = None
            self.red_flag_position = drop_pos
            self.red_flag_drop_time = self.sim_time
            self.add_agent_reward(agent, ACTION_FAILED_PUNISHMENT)
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
    # teammate near
    # ----------------------------------------------------------
    def _teammate_near(self, agent: Any, radius_cells: float = 5.0) -> bool:
        gf = getattr(agent, "game_field", None) or self.game_field
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
    # ✅ GAMMA-CORRECT PBRS + exploration
    # ----------------------------------------------------------
    def reward_potential_shaping(self, agent: Any, start_pos: Tuple[float, float], end_pos: Tuple[float, float]) -> None:
        """
        Potential-Based Reward Shaping:
            F(s,a,s') = coef * (gamma * Phi(s') - Phi(s))

        Called using executed float positions from Agent.update().
        """
        if getattr(agent, "side", None) != "blue":
            return

        self._remember_agent(agent)

        i_am_carrier = (self.red_flag_taken and (self.red_flag_carrier is agent))
        teammate_is_carrier = (self.red_flag_taken and (self.red_flag_carrier is not None) and (self.red_flag_carrier is not agent))

        if i_am_carrier:
            goal_x, goal_y = self.blue_flag_home
        elif teammate_is_carrier:
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

        shaped = float(FLAG_PROXIMITY_COEF) * (float(self.shaping_gamma) * phi_after - phi_before)
        if shaped != 0.0:
            self.add_agent_reward(agent, shaped)

        # Exploration bonus (cell-quantized, team memory)
        cell = self._clamp_cell(int(round(ex)), int(round(ey)))
        if cell not in self.blue_visited_cells:
            self.blue_visited_cells.add(cell)
            self.add_agent_reward(agent, EXPLORATION_REWARD)

    # ----------------------------------------------------------
    # Mine / combat rewards (minimal)
    # ----------------------------------------------------------
    def reward_mine_placed(self, agent: Any, mine_pos: Optional[Tuple[int, int]] = None) -> None:
        if mine_pos is not None and getattr(agent, "side", None) == "blue":
            x, _ = mine_pos
            if x > (self.cols * 0.5):
                self.mines_placed_in_enemy_half_this_episode += 1

    def reward_enemy_killed(self, killer_agent: Any, victim_agent: Optional[Any] = None, cause: Optional[str] = None) -> None:
        if killer_agent is None:
            return

        self._remember_agent(killer_agent)
        kside = getattr(killer_agent, "side", None)

        if cause == "mine":
            if kside == "blue":
                self.blue_mine_kills_this_episode += 1
            else:
                self.red_mine_kills_this_episode += 1

        # ✅ killer gets the kill reward
        self.add_agent_reward(killer_agent, ENEMY_MAV_KILL_REWARD)

        # ✅ optional team share for mine kills (explicit, excludes killer)
        if kside in ("blue", "red") and cause == "mine":
            self.add_team_reward(kside, ENEMY_MAV_KILL_REWARD * 0.5, exclude_agent=killer_agent)

    def record_mine_triggered_by_red(self) -> None:
        self.mines_triggered_by_red_this_episode += 1

    def punish_failed_action(self, agent: Any) -> None:
        if agent is None:
            return
        self.add_agent_reward(agent, ACTION_FAILED_PUNISHMENT)

    # ----------------------------------------------------------
    # Reward events API
    # ----------------------------------------------------------
    def pop_reward_events(self) -> List[Tuple[float, str, float]]:
        """
        Returns list of (t, agent_id, value). NOTE: agent_id is ALWAYS a string.
        """
        events = self.reward_events
        self.reward_events = []
        return events
