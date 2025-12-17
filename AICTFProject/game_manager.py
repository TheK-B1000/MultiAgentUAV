# =========================
# game_manager.py (REFACTORED, MARL-READY, EVENT REWARD ROUTING + GAMMA-PBRS)
# =========================

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# -------------------------
# Reward constants (baseline)
# -------------------------

WIN_TEAM_REWARD = 5.0
FLAG_PICKUP_REWARD = 0.5
FLAG_CARRY_HOME_REWARD = 3.0
ENEMY_MAV_KILL_REWARD = 2.0
ACTION_FAILED_PUNISHMENT = -0.5

FLAG_RETURN_DELAY = 10.0

# PBRS (potential based reward shaping): F = coef * (gamma * Phi(s') - Phi(s))
FLAG_PROXIMITY_COEF = 0.2
DEFAULT_SHAPING_GAMMA = 0.99

# Optional low-magnitude extras (safe defaults)
EXPLORATION_REWARD = 0.01
COORDINATION_BONUS = 0.3

# Optional draw penalty by phase (default 0, research-safe)
PHASE_DRAW_TIMEOUT_PENALTY: Dict[str, float] = {
    "OP1": 0.0,
    "OP2": 0.0,
    "OP3": 0.0,
    "SELF": 0.0,
}

Cell = Tuple[int, int]
FloatPos = Tuple[float, float]
RewardEvent = Tuple[float, str, float]  # (t, agent_id, value)


@dataclass
class GameManager:
    """
    Game state + reward routing.

    Key research invariants:
      - Rewards are emitted ONLY as per-agent events (no global rewards).
      - agent_id in reward events is ALWAYS a non-empty string.
      - Flag state is always consistent after each tick (carrier & taken flags align).
      - PBRS uses gamma-correct shaping: coef * (gamma * Phi(s') - Phi(s)).
    """

    cols: int
    rows: int

    # --- score/time ---
    blue_score: int = 0
    red_score: int = 0
    score_limit: int = 3

    max_time: float = 200.0
    current_time: float = 200.0
    sim_time: float = 0.0
    game_over: bool = False
    phase_name: str = "OP1"

    # --- flags ---
    blue_flag_home: Cell = (0, 0)
    red_flag_home: Cell = (0, 0)

    blue_flag_position: Cell = (0, 0)
    red_flag_position: Cell = (0, 0)

    blue_flag_taken: bool = False
    red_flag_taken: bool = False

    blue_flag_carrier: Optional[Any] = None
    red_flag_carrier: Optional[Any] = None

    blue_flag_drop_time: Optional[float] = None
    red_flag_drop_time: Optional[float] = None

    # --- reward event buffer ---
    reward_events: List[RewardEvent] = field(default_factory=list)

    # --- episode telemetry (minimal, optional) ---
    blue_mine_kills_this_episode: int = 0
    red_mine_kills_this_episode: int = 0
    mines_placed_in_enemy_half_this_episode: int = 0
    mines_triggered_by_red_this_episode: int = 0

    # --- exploration memory (team-level) ---
    blue_visited_cells: Set[Cell] = field(default_factory=set)
    red_visited_cells: Set[Cell] = field(default_factory=set)

    # --- routing memory (ids seen) ---
    blue_agent_ids_seen: Set[str] = field(default_factory=set)
    red_agent_ids_seen: Set[str] = field(default_factory=set)

    # --- shaping gamma (should match trainer gamma) ---
    shaping_gamma: float = DEFAULT_SHAPING_GAMMA

    # --- optional env binding for precise team membership ---
    game_field: Optional[Any] = field(default=None, repr=False, compare=False)

    # -------------------------
    # Binding / config
    # -------------------------

    def bind_game_field(self, game_field: Any) -> None:
        """Bind environment for exact team reward routing (recommended)."""
        self.game_field = game_field

    def set_phase(self, phase: str) -> None:
        self.phase_name = str(phase).upper()

    def set_shaping_gamma(self, gamma: float) -> None:
        g = float(gamma)
        if not (0.0 <= g <= 1.0):
            raise ValueError(f"gamma must be in [0,1], got {gamma}")
        self.shaping_gamma = g

    # -------------------------
    # Core helpers
    # -------------------------

    def _clamp_cell(self, x: int, y: int) -> Cell:
        return (max(0, min(self.cols - 1, int(x))), max(0, min(self.rows - 1, int(y))))

    def _agent_cell(self, agent: Any) -> Cell:
        fp = getattr(agent, "float_pos", None)
        if isinstance(fp, (tuple, list)) and len(fp) >= 2:
            return self._clamp_cell(int(round(fp[0])), int(round(fp[1])))
        if hasattr(agent, "get_position"):
            x, y = agent.get_position()
            return self._clamp_cell(int(x), int(y))
        return self._clamp_cell(getattr(agent, "x", 0), getattr(agent, "y", 0))

    def _agent_float(self, agent: Any) -> FloatPos:
        fp = getattr(agent, "float_pos", None)
        if isinstance(fp, (tuple, list)) and len(fp) >= 2:
            return float(fp[0]), float(fp[1])
        return float(getattr(agent, "x", 0.0)), float(getattr(agent, "y", 0.0))

    def _agent_uid(self, agent: Any) -> str:
        uid = getattr(agent, "unique_id", None)
        if uid is None or str(uid).strip() == "":
            return str(id(agent))
        return str(uid)

    def _remember_agent(self, agent: Any) -> None:
        if agent is None:
            return
        side = getattr(agent, "side", None)
        uid = self._agent_uid(agent)
        if side == "blue":
            self.blue_agent_ids_seen.add(uid)
        elif side == "red":
            self.red_agent_ids_seen.add(uid)

    # -------------------------
    # Reward routing (events only)
    # -------------------------

    def add_reward_event(self, value: float, agent_id: str, timestamp: Optional[float] = None) -> None:
        if agent_id is None or str(agent_id).strip() == "":
            raise ValueError("agent_id must be a non-empty string.")
        t = self.sim_time if timestamp is None else float(timestamp)
        self.reward_events.append((float(t), str(agent_id), float(value)))

    def add_agent_reward(self, agent: Any, value: float, timestamp: Optional[float] = None) -> None:
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
        Emit per-agent events for all agents on `side`.
        Uses bound env membership if available; falls back to ids_seen.
        """
        side = str(side)
        ex_uid = self._agent_uid(exclude_agent) if exclude_agent is not None else None

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

        ids = self.blue_agent_ids_seen if side == "blue" else self.red_agent_ids_seen
        for uid in ids:
            if ex_uid is not None and uid == ex_uid:
                continue
            self.add_reward_event(value, agent_id=uid, timestamp=timestamp)

    def pop_reward_events(self) -> List[RewardEvent]:
        events = self.reward_events
        self.reward_events = []
        return events

    # -------------------------
    # Reset
    # -------------------------

    def reset_game(self, reset_scores: bool = True) -> None:
        if reset_scores:
            self.blue_score = 0
            self.red_score = 0

        self.game_over = False
        self.current_time = float(self.max_time)
        self.sim_time = 0.0
        self.reward_events.clear()

        self.blue_mine_kills_this_episode = 0
        self.red_mine_kills_this_episode = 0
        self.mines_placed_in_enemy_half_this_episode = 0
        self.mines_triggered_by_red_this_episode = 0

        self.blue_visited_cells.clear()
        self.red_visited_cells.clear()

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

    # -------------------------
    # Tick / termination
    # -------------------------

    def tick_seconds(self, dt: float) -> Optional[str]:
        if self.game_over or dt <= 0.0:
            return None

        self.sim_time += float(dt)
        self.current_time -= float(dt)

        self.sanity_check_flags()
        self._update_flag_auto_return()

        # Time over
        if self.current_time <= 0.0 and not self.game_over:
            self.game_over = True
            if self.blue_score > self.red_score:
                self.add_team_reward("blue", WIN_TEAM_REWARD)
                return "BLUE WINS ON TIME"
            if self.red_score > self.blue_score:
                self.add_team_reward("red", WIN_TEAM_REWARD)
                return "RED WINS ON TIME"

            penalty = float(PHASE_DRAW_TIMEOUT_PENALTY.get(self.phase_name, 0.0))
            if penalty != 0.0:
                # If you ever use draw penalties, route them explicitly
                self.add_team_reward("blue", penalty)
                self.add_team_reward("red", penalty)
                return f"DRAW — PENALTY ({self.phase_name})"
            return f"DRAW — NO PENALTY ({self.phase_name})"

        # Score limit
        if self.blue_score >= self.score_limit:
            self.game_over = True
            self.add_team_reward("blue", WIN_TEAM_REWARD)
            return "BLUE WINS BY SCORE!"

        if self.red_score >= self.score_limit:
            self.game_over = True
            self.add_team_reward("red", WIN_TEAM_REWARD)
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

    # -------------------------
    # Flag sanity / helpers
    # -------------------------

    def sanity_check_flags(self) -> None:
        """
        Ensures carrier/taken/position is consistent.
        If a carrier is disabled, the flag drops at the carrier cell with a drop_time.
        """
        # Blue flag
        if self.blue_flag_taken:
            if self.blue_flag_carrier is None:
                self.blue_flag_taken = False
                self.blue_flag_position = self.blue_flag_home
                self.blue_flag_drop_time = None
            else:
                carrier_enabled = getattr(self.blue_flag_carrier, "isEnabled", lambda: True)()
                if not carrier_enabled:
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
                carrier_enabled = getattr(self.red_flag_carrier, "isEnabled", lambda: True)()
                if not carrier_enabled:
                    self.drop_flag_if_carrier_disabled(self.red_flag_carrier)
                else:
                    self.red_flag_position = self._agent_cell(self.red_flag_carrier)
                    self.red_flag_drop_time = None

    def get_enemy_flag_position(self, side: str) -> Cell:
        side = str(side)
        if side == "blue":
            return self._agent_cell(self.red_flag_carrier) if (self.red_flag_taken and self.red_flag_carrier is not None) else self.red_flag_position
        return self._agent_cell(self.blue_flag_carrier) if (self.blue_flag_taken and self.blue_flag_carrier is not None) else self.blue_flag_position

    def get_team_zone_center(self, side: str) -> Cell:
        return self.blue_flag_home if str(side) == "blue" else self.red_flag_home

    def get_sim_time(self) -> float:
        return float(self.sim_time)

    # -------------------------
    # Flag interactions
    # -------------------------

    def try_pickup_enemy_flag(self, agent: Any) -> bool:
        side = getattr(agent, "side", None)
        if side not in ("blue", "red"):
            return False

        self._remember_agent(agent)

        ax, ay = self._agent_float(agent)
        if side == "blue":
            enemy_taken = self.red_flag_taken
            enemy_pos = self.red_flag_position
        else:
            enemy_taken = self.blue_flag_taken
            enemy_pos = self.blue_flag_position

        if enemy_taken:
            return False

        if math.hypot(ax - float(enemy_pos[0]), ay - float(enemy_pos[1])) > 1.0:
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

        self.add_agent_reward(agent, FLAG_PICKUP_REWARD)

        if self._teammate_near(agent):
            self.add_agent_reward(agent, COORDINATION_BONUS)

        return True

    def try_score_if_carrying_and_home(self, agent: Any) -> bool:
        side = getattr(agent, "side", None)
        if side not in ("blue", "red"):
            return False

        self._remember_agent(agent)
        ax, ay = self._agent_float(agent)

        # Blue scores carrying red flag at blue home
        if side == "blue" and self.red_flag_taken and (self.red_flag_carrier is agent):
            if math.hypot(ax - float(self.blue_flag_home[0]), ay - float(self.blue_flag_home[1])) <= 2.0:
                self.blue_score += 1
                self._reset_red_flag_to_home()

                if hasattr(agent, "setCarryingFlag"):
                    agent.setCarryingFlag(False, scored=True)

                self.add_agent_reward(agent, FLAG_CARRY_HOME_REWARD)
                self.add_team_reward("blue", FLAG_CARRY_HOME_REWARD * 0.5, exclude_agent=agent)

                if self._teammate_near(agent):
                    self.add_agent_reward(agent, COORDINATION_BONUS)

                return True

        # Red scores carrying blue flag at red home
        if side == "red" and self.blue_flag_taken and (self.blue_flag_carrier is agent):
            if math.hypot(ax - float(self.red_flag_home[0]), ay - float(self.red_flag_home[1])) <= 2.0:
                self.red_score += 1
                self._reset_blue_flag_to_home()

                if hasattr(agent, "setCarryingFlag"):
                    agent.setCarryingFlag(False, scored=True)

                self.add_agent_reward(agent, FLAG_CARRY_HOME_REWARD)
                self.add_team_reward("red", FLAG_CARRY_HOME_REWARD * 0.5, exclude_agent=agent)

                if self._teammate_near(agent):
                    self.add_agent_reward(agent, COORDINATION_BONUS)

                return True

        return False

    def _reset_blue_flag_to_home(self) -> None:
        self.blue_flag_taken = False
        self.blue_flag_carrier = None
        self.blue_flag_position = self.blue_flag_home
        self.blue_flag_drop_time = None

    def _reset_red_flag_to_home(self) -> None:
        self.red_flag_taken = False
        self.red_flag_carrier = None
        self.red_flag_position = self.red_flag_home
        self.red_flag_drop_time = None

    # game_manager.py
    def drop_flag_if_carrier_disabled(self, agent: Any, punish: bool = False) -> None:
        """
        Drop the carried flag at the carrier's current cell.

        IMPORTANT:
          - punish=False for normal "death/disabled" drops (called by sanity_check_flags)
          - punish=True only when you explicitly decide a macro/action "failed" and want to penalize
        """
        drop_pos = self._agent_cell(agent)
        self._remember_agent(agent)

        if self.blue_flag_carrier is agent:
            self.blue_flag_taken = False
            self.blue_flag_carrier = None
            self.blue_flag_position = drop_pos
            self.blue_flag_drop_time = self.sim_time

            if punish:
                self.add_agent_reward(agent, ACTION_FAILED_PUNISHMENT)

            if hasattr(agent, "setCarryingFlag"):
                agent.setCarryingFlag(False, scored=False)

        elif self.red_flag_carrier is agent:
            self.red_flag_taken = False
            self.red_flag_carrier = None
            self.red_flag_position = drop_pos
            self.red_flag_drop_time = self.sim_time

            if punish:
                self.add_agent_reward(agent, ACTION_FAILED_PUNISHMENT)

            if hasattr(agent, "setCarryingFlag"):
                agent.setCarryingFlag(False, scored=False)

    def clear_flag_carrier_if_agent(self, agent: Any) -> None:
        """
        Hard reset carriers if an agent is removed from game unexpectedly.
        Prefer drop_flag_if_carrier_disabled() in normal flow.
        """
        if self.blue_flag_carrier is agent:
            self._reset_blue_flag_to_home()
            if hasattr(agent, "setCarryingFlag"):
                agent.setCarryingFlag(False, scored=False)

        if self.red_flag_carrier is agent:
            self._reset_red_flag_to_home()
            if hasattr(agent, "setCarryingFlag"):
                agent.setCarryingFlag(False, scored=False)

    # -------------------------
    # teammate proximity
    # -------------------------

    def _teammate_near(self, agent: Any, radius_cells: float = 5.0) -> bool:
        gf = getattr(agent, "game_field", None) or self.game_field
        if gf is None:
            return False

        side = getattr(agent, "side", None)
        if side not in ("blue", "red"):
            return False

        team = gf.blue_agents if side == "blue" else gf.red_agents
        ax, ay = self._agent_float(agent)

        for other in team:
            if other is agent or (not other.isEnabled()):
                continue
            ox, oy = self._agent_float(other)
            if math.hypot(ox - ax, oy - ay) <= float(radius_cells):
                return True
        return False

    # -------------------------
    # Gamma-correct PBRS + exploration
    # -------------------------

    def reward_potential_shaping(self, agent: Any, start_pos: FloatPos, end_pos: FloatPos) -> None:
        """
        Potential-Based Reward Shaping:
            F(s,a,s') = coef * (gamma * Phi(s') - Phi(s))

        Uses float positions provided by the env/agent.
        By default, shaping is applied to BOTH teams symmetrically.
        If you want blue-only shaping for curriculum, gate by agent.side externally.
        """
        side = getattr(agent, "side", None)
        if side not in ("blue", "red"):
            return

        self._remember_agent(agent)

        # Define goal for shaping:
        # - if I am carrier: go home
        # - if teammate is carrier: also go home (support)
        # - else: go to enemy flag position
        if side == "blue":
            enemy_taken = self.red_flag_taken
            enemy_carrier = self.red_flag_carrier
            my_home = self.blue_flag_home
            enemy_goal = self.get_enemy_flag_position("blue")
        else:
            enemy_taken = self.blue_flag_taken
            enemy_carrier = self.blue_flag_carrier
            my_home = self.red_flag_home
            enemy_goal = self.get_enemy_flag_position("red")

        i_am_carrier = enemy_taken and (enemy_carrier is agent)
        teammate_is_carrier = enemy_taken and (enemy_carrier is not None) and (enemy_carrier is not agent)

        goal_x, goal_y = my_home if (i_am_carrier or teammate_is_carrier) else enemy_goal

        max_dist = math.sqrt(float(self.cols * self.cols + self.rows * self.rows))
        if max_dist <= 1e-6:
            return

        sx, sy = float(start_pos[0]), float(start_pos[1])
        ex, ey = float(end_pos[0]), float(end_pos[1])

        prev_d = min(max_dist, math.dist([sx, sy], [float(goal_x), float(goal_y)]))
        cur_d = min(max_dist, math.dist([ex, ey], [float(goal_x), float(goal_y)]))

        phi_before = 1.0 - (prev_d / max_dist)
        phi_after = 1.0 - (cur_d / max_dist)

        shaped = float(FLAG_PROXIMITY_COEF) * (float(self.shaping_gamma) * phi_after - phi_before)
        if shaped != 0.0:
            self.add_agent_reward(agent, shaped)

        # Exploration: per-team visited set
        cell = self._clamp_cell(int(round(ex)), int(round(ey)))
        if side == "blue":
            visited = self.blue_visited_cells
        else:
            visited = self.red_visited_cells

        if cell not in visited:
            visited.add(cell)
            self.add_agent_reward(agent, EXPLORATION_REWARD)

    # -------------------------
    # Mine/combat hooks (minimal)
    # -------------------------

    def reward_mine_placed(self, agent: Any, mine_pos: Optional[Cell] = None) -> None:
        if mine_pos is None:
            return
        side = getattr(agent, "side", None)
        if side not in ("blue", "red"):
            return

        x, _ = mine_pos
        if side == "blue":
            if x > (self.cols * 0.5):
                self.mines_placed_in_enemy_half_this_episode += 1
        else:
            if x < (self.cols * 0.5):
                self.mines_placed_in_enemy_half_this_episode += 1

    def reward_enemy_killed(self, killer_agent: Any, victim_agent: Optional[Any] = None, cause: Optional[str] = None) -> None:
        if killer_agent is None:
            return

        self._remember_agent(killer_agent)
        kside = getattr(killer_agent, "side", None)

        if cause == "mine":
            if kside == "blue":
                self.blue_mine_kills_this_episode += 1
            elif kside == "red":
                self.red_mine_kills_this_episode += 1

        self.add_agent_reward(killer_agent, ENEMY_MAV_KILL_REWARD)

        # Optional team share for mine kills (explicit, excludes killer)
        if kside in ("blue", "red") and cause == "mine":
            self.add_team_reward(kside, ENEMY_MAV_KILL_REWARD * 0.5, exclude_agent=killer_agent)

    def record_mine_triggered_by_red(self) -> None:
        self.mines_triggered_by_red_this_episode += 1

    def punish_failed_action(self, agent: Any) -> None:
        if agent is None:
            return
        self.add_agent_reward(agent, ACTION_FAILED_PUNISHMENT)
