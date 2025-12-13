from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import math

WIN_TEAM_REWARD = 5.0
FLAG_PICKUP_REWARD = 0.5
FLAG_CARRY_HOME_REWARD = 3.0
ENABLED_MINE_REWARD = 0.1
ENEMY_MAV_KILL_REWARD = 0.8
ACTION_FAILED_PUNISHMENT = -0.5
FLAG_PROXIMITY_COEF = 0.02
FLAG_RETURN_DELAY = 10.0

PHASE_DRAW_TIMEOUT_PENALTY = {
    "OP1": -2.0,
    "OP2": -2.0,
    "OP3":  0.0,
    "SELF": 0.0,
}


@dataclass
class GameManager:
    cols: int
    rows: int

    blue_score: int = 0
    red_score: int = 0
    max_time: float = 200.0
    current_time: float = 200.0
    game_over: bool = False
    score_limit: int = 3

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

    sim_time: float = 0.0
    reward_events: List[Tuple[float, Optional[str], float]] = field(default_factory=list)

    phase_name: str = "OP1"

    blue_mine_kills_this_episode: int = 0
    red_mine_kills_this_episode: int = 0
    mines_placed_in_enemy_half_this_episode: int = 0
    mines_triggered_by_red_this_episode: int = 0
    mines_rewarded_by_agent: Dict[str, bool] = field(default_factory=dict)

    team_mines_rewarded: Dict[str, int] = field(default_factory=lambda: {"blue": 0, "red": 0})

    def set_phase(self, phase: str) -> None:
        self.phase_name = phase

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

        self.team_mines_rewarded = {"blue": 0, "red": 0}
        self.blue_mine_kills_this_episode = 0
        self.red_mine_kills_this_episode = 0
        self.mines_placed_in_enemy_half_this_episode = 0
        self.mines_triggered_by_red_this_episode = 0
        self.mines_rewarded_by_agent.clear()

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

    def tick_seconds(self, dt: float) -> Optional[str]:
        if self.game_over or dt <= 0.0:
            return None

        self.sim_time += dt
        self.current_time -= dt

        # Keep flag state consistent every tick
        self.sanity_check_flags()
        self._update_flag_auto_return()

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

    # ---------- Drift guard ----------
    def sanity_check_flags(self) -> None:
        # If taken, carrier must exist and be enabled
        if self.blue_flag_taken:
            if self.blue_flag_carrier is None:
                self.blue_flag_taken = False
                self.blue_flag_position = self.blue_flag_home
                self.blue_flag_drop_time = None
            else:
                if not getattr(self.blue_flag_carrier, "isEnabled", lambda: True)():
                    self.drop_flag_if_carrier_disabled(self.blue_flag_carrier)
                else:
                    # Keep position valid and on-map while carried
                    self.blue_flag_position = self._agent_cell(self.blue_flag_carrier)
                    self.blue_flag_drop_time = None

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

    # ---------- Helpers ----------
    def get_enemy_flag_position(self, side: str) -> Tuple[int, int]:
        if side == "blue":
            # enemy is red
            if self.red_flag_taken and self.red_flag_carrier is not None:
                return self._agent_cell(self.red_flag_carrier)
            return self.red_flag_position
        else:
            # enemy is blue
            if self.blue_flag_taken and self.blue_flag_carrier is not None:
                return self._agent_cell(self.blue_flag_carrier)
            return self.blue_flag_position

    def get_team_zone_center(self, side: str) -> Tuple[int, int]:
        return self.blue_flag_home if side == "blue" else self.red_flag_home

    def get_sim_time(self) -> float:
        return self.sim_time

    def _agent_cell(self, agent) -> Tuple[int, int]:
        if hasattr(agent, "float_pos"):
            fx, fy = agent.float_pos
            return self._clamp_cell(int(round(fx)), int(round(fy)))
        if hasattr(agent, "get_position"):
            x, y = agent.get_position()
            return self._clamp_cell(int(x), int(y))
        # last resort
        return self._clamp_cell(getattr(agent, "x", 0), getattr(agent, "y", 0))

    def try_pickup_enemy_flag(self, agent) -> bool:
        side = agent.getSide()
        pos_x, pos_y = agent.float_pos

        if side == "blue":
            enemy_taken = self.red_flag_taken
            enemy_pos = self.red_flag_position
        else:
            enemy_taken = self.blue_flag_taken
            enemy_pos = self.blue_flag_position

        if enemy_taken:
            return False

        if math.hypot(pos_x - enemy_pos[0], pos_y - enemy_pos[1]) > 1.0:
            return False

        if side == "blue":
            self.red_flag_taken = True
            self.red_flag_carrier = agent
            self.red_flag_position = self._agent_cell(agent)  # stay valid
            self.red_flag_drop_time = None
        else:
            self.blue_flag_taken = True
            self.blue_flag_carrier = agent
            self.blue_flag_position = self._agent_cell(agent)
            self.blue_flag_drop_time = None

        # Sync agent carry flag if available
        if hasattr(agent, "setCarryingFlag"):
            agent.setCarryingFlag(True)

        self.add_reward_event(FLAG_PICKUP_REWARD, agent_id=getattr(agent, "unique_id", None))
        return True

    def try_score_if_carrying_and_home(self, agent) -> bool:
        side = agent.getSide()
        pos_x, pos_y = agent.float_pos

        if side == "blue" and self.red_flag_taken and self.red_flag_carrier is agent:
            if math.hypot(pos_x - self.blue_flag_home[0], pos_y - self.blue_flag_home[1]) <= 2.0:
                self.blue_score += 1
                self.red_flag_taken = False
                self.red_flag_carrier = None
                self.red_flag_position = self.red_flag_home
                self.red_flag_drop_time = None

                if hasattr(agent, "setCarryingFlag"):
                    agent.setCarryingFlag(False, scored=True)

                self.add_reward_event(FLAG_CARRY_HOME_REWARD, agent_id=getattr(agent, "unique_id", None))
                self.add_reward_event(FLAG_CARRY_HOME_REWARD * 0.5)
                return True

        if side == "red" and self.blue_flag_taken and self.blue_flag_carrier is agent:
            if math.hypot(pos_x - self.red_flag_home[0], pos_y - self.red_flag_home[1]) <= 2.0:
                self.red_score += 1
                self.blue_flag_taken = False
                self.blue_flag_carrier = None
                self.blue_flag_position = self.blue_flag_home
                self.blue_flag_drop_time = None

                if hasattr(agent, "setCarryingFlag"):
                    agent.setCarryingFlag(False, scored=True)

                self.add_reward_event(-FLAG_CARRY_HOME_REWARD * 0.5)
                return True

        return False

    def drop_flag_if_carrier_disabled(self, agent) -> None:
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

    def handle_agent_death(self, agent) -> None:
        self.drop_flag_if_carrier_disabled(agent)

    def clear_flag_carrier_if_agent(self, agent) -> None:
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

    def reward_potential_shaping(self, agent, start_pos, end_pos) -> None:
        if agent.getSide() != "blue":
            return

        if self.red_flag_taken and self.red_flag_carrier is agent:
            goal_x, goal_y = self.blue_flag_home
        else:
            goal_x, goal_y = self.red_flag_position

        max_dist = math.sqrt(self.cols ** 2 + self.rows ** 2)
        if max_dist <= 0.0:
            return

        sx, sy = start_pos
        ex, ey = end_pos

        prev_d = max(0.0, min(math.dist([sx, sy], [goal_x, goal_y]), max_dist))
        cur_d  = max(0.0, min(math.dist([ex, ey], [goal_x, goal_y]), max_dist))

        phi_before = 1.0 - (prev_d / max_dist)
        phi_after  = 1.0 - (cur_d / max_dist)

        shaped = FLAG_PROXIMITY_COEF * (phi_after - phi_before)
        if shaped != 0.0:
            self.add_reward_event(shaped, agent_id=getattr(agent, "unique_id", None))

    def reward_mine_placed(self, agent, mine_pos=None) -> None:
        uid = getattr(agent, "unique_id", None)
        if uid is not None and not self.mines_rewarded_by_agent.get(uid, False):
            self.add_reward_event(ENABLED_MINE_REWARD, agent_id=uid)
            self.mines_rewarded_by_agent[uid] = True

        if mine_pos is not None and agent.getSide() == "blue":
            x, _ = mine_pos
            if x > (self.cols * 0.5):
                self.mines_placed_in_enemy_half_this_episode += 1

    def reward_enemy_killed(self, killer_agent, victim_agent=None, cause=None) -> None:
        if cause == "mine":
            if killer_agent.getSide() == "blue":
                self.blue_mine_kills_this_episode += 1
            else:
                self.red_mine_kills_this_episode += 1

        self.add_reward_event(ENEMY_MAV_KILL_REWARD, agent_id=killer_agent.unique_id)

        if killer_agent.getSide() == "blue" and cause == "mine":
            self.add_reward_event(ENEMY_MAV_KILL_REWARD * 0.5)

    def record_mine_triggered_by_red(self) -> None:
        self.mines_triggered_by_red_this_episode += 1

    def punish_failed_action(self, agent) -> None:
        self.add_reward_event(ACTION_FAILED_PUNISHMENT, agent_id=agent.unique_id)

    def add_reward_event(self, value: float, timestamp=None, agent_id=None) -> None:
        t = self.sim_time if timestamp is None else float(timestamp)
        self.reward_events.append((t, agent_id, float(value)))

    def pop_reward_events(self):
        events = self.reward_events
        self.reward_events = []
        return events
