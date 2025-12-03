# game_manager.py
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import math


# ================= PAPER-INSPIRED REWARDS (TUNED FOR CONTINUOUS) =================
WIN_TEAM_REWARD = 1.0
FLAG_PICKUP_REWARD = 0.20
FLAG_CARRY_HOME_REWARD = 0.80
ENABLED_MINE_REWARD = 0.30
ENEMY_MAV_KILL_REWARD = 0.80

ACTION_FAILED_PUNISHMENT = -0.25

# Terminal rewards
DRAW_PENALTY = -4.0
LOSS_PENALTY = -1.0

# Time pressure
TIME_PENALTY_PER_SECOND = -0.0008

# Shaping rewards (per unit distance moved closer)
DISTANCE_TOWARD_FLAG_COEF = 0.01
CARRIER_HOME_COEF = 0.03


@dataclass
class GameManager:
    cols: int
    rows: int

    # --- Scores & timing ---
    blue_score: int = 0
    red_score: int = 0
    max_time: float = 200.0
    current_time: float = field(init=False)
    game_over: bool = False
    score_limit: int = 3

    # --- Flag state ---
    blue_flag_home: Tuple[int, int] = field(init=False)
    red_flag_home: Tuple[int, int] = field(init=False)
    blue_flag_position: Tuple[float, float] = field(init=False)  # now float when carried
    red_flag_position: Tuple[float, float] = field(init=False)

    blue_flag_taken: bool = False
    red_flag_taken: bool = False
    blue_flag_carrier: Optional[Any] = None   # Agent reference
    red_flag_carrier: Optional[Any] = None

    # --- Simulation & rewards ---
    sim_time: float = 0.0
    reward_events: List[Tuple[float, Optional[str], float, str]] = field(default_factory=list)

    # Debug / stats
    debug_rewards: bool = False
    episode_event_counts: Dict[str, int] = field(default_factory=dict)
    episode_reward_totals: Dict[str, float] = field(default_factory=dict)

    # Shaping state
    last_flag_distance: Dict[str, float] = field(default_factory=dict)
    carrier_last_home_distance: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.reset_game(reset_scores=True)

    # ------------------------------------------------------------------ #
    # Game reset
    # ------------------------------------------------------------------ #
    def reset_game(self, reset_scores: bool = True) -> None:
        if reset_scores:
            self.blue_score = self.red_score = 0
            self.game_over = False

        self.current_time = self.max_time
        self.sim_time = 0.0
        self.reward_events.clear()
        self.episode_event_counts.clear()
        self.episode_reward_totals.clear()
        self.last_flag_distance.clear()
        self.carrier_last_home_distance.clear()

        # Flag home bases (left/right side, middle row)
        mid_row = self.rows // 2
        self.blue_flag_home = (2, mid_row)
        self.red_flag_home = (self.cols - 3, mid_row)

        # Flags start at home
        self.blue_flag_position = (self.blue_flag_home[0] + 0.5, self.blue_flag_home[1] + 0.5)
        self.red_flag_position = (self.red_flag_home[0] + 0.5, self.red_flag_home[1] + 0.5)

        self.blue_flag_taken = self.red_flag_taken = False
        self.blue_flag_carrier = self.red_flag_carrier = None

    # ------------------------------------------------------------------ #
    # Time tick & win condition
    # ------------------------------------------------------------------ #
    def tick_seconds(self, dt: float) -> Optional[str]:
        if self.game_over or dt <= 0.0:
            return None

        self.sim_time += dt
        self.current_time -= dt

        # Global time penalty
        if TIME_PENALTY_PER_SECOND != 0.0:
            self.add_reward_event(TIME_PENALTY_PER_SECOND * dt, agent_id=None, tag="time_penalty")

        # Time expired
        if self.current_time <= 0.0 and not self.game_over:
            self.game_over = True
            if self.blue_score > self.red_score:
                self.add_reward_event(WIN_TEAM_REWARD, agent_id=None, tag="blue_win_time")
                return "BLUE WINS ON TIME"
            elif self.red_score > self.blue_score:
                self.add_reward_event(LOSS_PENALTY, agent_id=None, tag="blue_loss_time")
                return "RED WINS ON TIME"
            else:
                self.add_reward_event(DRAW_PENALTY, agent_id=None, tag="draw_penalty")
                return "DRAW — TIME EXPIRED"

        # Score limit reached
        if self.blue_score >= self.score_limit:
            self.game_over = True
            self.add_reward_event(WIN_TEAM_REWARD, agent_id=None, tag="blue_win_score")
            return "BLUE WINS BY SCORE!"

        if self.red_score >= self.score_limit:
            self.game_over = True
            self.add_reward_event(LOSS_PENALTY, agent_id=None, tag="blue_loss_score")
            return "RED WINS BY SCORE!"

        return None

    # ------------------------------------------------------------------ #
    # Flag helpers (continuous positions!)
    # ------------------------------------------------------------------ #
    def get_enemy_flag_position(self, side: str) -> Tuple[float, float]:
        return self.red_flag_position if side == "blue" else self.blue_flag_position

    def get_team_zone_center(self, side: str) -> Tuple[float, float]:
        home = self.blue_flag_home if side == "blue" else self.red_flag_home
        return (home[0] + 0.5, home[1] + 0.5)

    # ------------------------------------------------------------------ #
    # Flag pickup (continuous distance check)
    # ------------------------------------------------------------------ #
    def try_pickup_enemy_flag(self, agent) -> bool:
        if not agent.isEnabled():
            return False

        side = agent.side
        ax, ay = agent._float_x, agent._float_y

        if side == "blue" and not self.red_flag_taken:
            fx, fy = self.red_flag_position
            if math.hypot(ax - fx, ay - fy) <= 0.8:  # pickup radius
                self.red_flag_taken = True
                self.red_flag_carrier = agent
                self.red_flag_position = (ax, ay)  # flag now moves with carrier
                self.add_reward_event(FLAG_PICKUP_REWARD, agent_id=agent.unique_id, tag="flag_pickup")
                return True

        elif side == "red" and not self.blue_flag_taken:
            fx, fy = self.blue_flag_position
            if math.hypot(ax - fx, ay - fy) <= 0.8:
                self.blue_flag_taken = True
                self.blue_flag_carrier = agent
                self.blue_flag_position = (ax, ay)
                # Red gets no reward here (only blue is trained)
                return True

        return False

    # ------------------------------------------------------------------ #
    # Scoring when carrier returns home
    # ------------------------------------------------------------------ #
    def try_score_if_carrying_and_home(self, agent) -> bool:
        if not agent.isEnabled():
            return False

        ax, ay = agent._float_x, agent._float_y

        if agent.side == "blue" and self.red_flag_taken and self.red_flag_carrier is agent:
            hx, hy = self.get_team_zone_center("blue")
            if math.hypot(ax - hx, ay - hy) <= 2.5:  # scoring radius
                self.blue_score += 1
                self.red_flag_taken = False
                self.red_flag_position = (self.red_flag_home[0] + 0.5, self.red_flag_home[1] + 0.5)
                self.red_flag_carrier = None
                self.add_reward_event(FLAG_CARRY_HOME_REWARD, agent_id=agent.unique_id, tag="flag_score")
                return True

        elif agent.side == "red" and self.blue_flag_taken and self.blue_flag_carrier is agent:
            hx, hy = self.get_team_zone_center("red")
            if math.hypot(ax - hx, ay - hy) <= 2.5:
                self.red_score += 1
                self.blue_flag_taken = False
                self.blue_flag_position = (self.blue_flag_home[0] + 0.5, self.blue_flag_home[1] + 0.5)
                self.blue_flag_carrier = None
                return True

        return False

    # ------------------------------------------------------------------ #
    # Drop flag on death/stun
    # ------------------------------------------------------------------ #
    def drop_flag_if_carrier_disabled(self, agent) -> None:
        if self.red_flag_carrier is agent:
            self.red_flag_taken = False
            self.red_flag_position = (agent._float_x, agent._float_y)
            self.red_flag_carrier = None
            self.add_reward_event(ACTION_FAILED_PUNISHMENT, agent_id=agent.unique_id, tag="carrier_disabled")

        elif self.blue_flag_carrier is agent:
            self.blue_flag_taken = False
            self.blue_flag_position = (agent._float_x, agent._float_y)
            self.blue_flag_carrier = None

    # ------------------------------------------------------------------ #
    # Distance shaping: toward enemy flag (BLUE only)
    # ------------------------------------------------------------------ #
    def distance_shaping_for_blue(self, blue_agents: List[Any]) -> None:
        if not blue_agents:
            return
        fx, fy = self.get_enemy_flag_position("blue")
        for agent in blue_agents:
            if not agent.isEnabled():
                continue
            d = math.hypot(agent._float_x - fx, agent._float_y - fy)
            prev = self.last_flag_distance.get(agent.unique_id)
            self.last_flag_distance[agent.unique_id] = d
            if prev is not None:
                delta = prev - d
                if delta > 0:
                    r = DISTANCE_TOWARD_FLAG_COEF * delta
                    self.add_reward_event(r, agent_id=agent.unique_id, tag="toward_flag")

    # ------------------------------------------------------------------ #
    # Distance shaping: carrier returning home with flag
    # ------------------------------------------------------------------ #
    def distance_shaping_for_blue_carrier(self, blue_agents: List[Any]) -> None:
        if not self.red_flag_taken or self.red_flag_carrier is None:
            return
        carrier = self.red_flag_carrier
        if carrier not in blue_agents or not carrier.isEnabled():
            return

        hx, hy = self.get_team_zone_center("blue")
        d = math.hypot(carrier._float_x - hx, carrier._float_y - hy)
        prev = self.carrier_last_home_distance.get(carrier.unique_id)
        self.carrier_last_home_distance[carrier.unique_id] = d

        if prev is not None:
            delta = prev - d
            if delta > 0:
                r = CARRIER_HOME_COEF * delta
                self.add_reward_event(r, agent_id=carrier.unique_id, tag="carrier_toward_home")
            elif delta < 0:
                r = 0.5 * CARRIER_HOME_COEF * (-delta)
                self.add_reward_event(-r, agent_id=carrier.unique_id, tag="carrier_away_home")

    # ------------------------------------------------------------------ #
    # Reward hooks
    # ------------------------------------------------------------------ #
    def reward_mine_placed(self, agent) -> None:
        self.add_reward_event(ENABLED_MINE_REWARD, agent_id=agent.unique_id, tag="mine_enabled")

    def reward_enemy_killed(self, killer_agent) -> None:
        self.add_reward_event(ENEMY_MAV_KILL_REWARD, agent_id=killer_agent.unique_id, tag="enemy_kill")

    def punish_failed_action(self, agent) -> None:
        self.add_reward_event(ACTION_FAILED_PUNISHMENT, agent_id=agent.unique_id, tag="failed_action")

    # ------------------------------------------------------------------ #
    # Reward event system
    # ------------------------------------------------------------------ #
    def add_reward_event(
        self,
        value: float,
        agent_id: Optional[str] = None,
        tag: str = "generic",
        timestamp: Optional[float] = None,
    ) -> None:
        t = self.sim_time if timestamp is None else timestamp
        value = float(value)

        self.reward_events.append((t, agent_id, value, tag))

        self.episode_event_counts[tag] = self.episode_event_counts.get(tag, 0) + 1
        self.episode_reward_totals[tag] = self.episode_reward_totals.get(tag, 0.0) + value

        if self.debug_rewards:
            who = agent_id if agent_id else "GLOBAL"
            print(f"[REWARD] t={t:6.2f} | {tag:20s} | {who:12s} | {value:+.4f}")

    def get_step_rewards(self, debug: bool = False) -> Dict[Optional[str], float]:
        rewards: Dict[Optional[str], float] = {}
        for _, agent_id, r, tag in self.reward_events:
            rewards.setdefault(agent_id, 0.0)
            rewards[agent_id] += r

        if debug or self.debug_rewards:
            pretty = {k if k else "GLOBAL": v for k, v in rewards.items()}
            print(f"[STEP REWARDS] t={self.sim_time:6.2f} → {pretty}")

        self.reward_events.clear()
        return rewards