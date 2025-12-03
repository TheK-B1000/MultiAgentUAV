from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import math

# ================= PAPER-INSPIRED REWARDS =================
WIN_TEAM_REWARD = 1.0
FLAG_PICKUP_REWARD = 0.1
FLAG_CARRY_HOME_REWARD = 0.5
ENABLED_MINE_REWARD = 0.2
ENEMY_MAV_KILL_REWARD = 0.5
ACTION_FAILED_PUNISHMENT = -0.2

# These are *shaped* penalties you're using for blue-only training:
DRAW_PENALTY = -2.0
LOSS_PENALTY = -1.0


@dataclass
class GameManager:
    cols: int
    rows: int

    # --- Score and timing ---
    blue_score: int = 0
    red_score: int = 0
    max_time: float = 200.0
    current_time: float = 200.0
    game_over: bool = False
    score_limit: int = 3

    # --- Flag state ---
    blue_flag_position: Tuple[int, int] = (0, 0)
    red_flag_position: Tuple[int, int] = (0, 0)
    blue_flag_taken: bool = False
    red_flag_taken: bool = False
    blue_flag_carrier: Optional[Any] = None  # Agent or None
    red_flag_carrier: Optional[Any] = None

    # Flag home positions (where they start and where you score)
    blue_flag_home: Tuple[int, int] = (0, 0)
    red_flag_home: Tuple[int, int] = (0, 0)

    # --- Simulation time and reward buffer ---
    sim_time: float = 0.0
    # List of (timestamp, agent_id, reward_value)
    reward_events: List[Tuple[float, Optional[str], float]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Game reset
    # ------------------------------------------------------------------
    def reset_game(self, reset_scores: bool = True) -> None:
        if reset_scores:
            self.blue_score = 0
            self.red_score = 0

        self.game_over = False
        self.current_time = self.max_time
        self.sim_time = 0.0
        self.reward_events.clear()

        # Place flags at home positions (symmetric mid-row bases)
        mid_row = self.rows // 2
        self.blue_flag_home = (2, mid_row)
        self.red_flag_home = (self.cols - 3, mid_row)

        self.blue_flag_position = self.blue_flag_home
        self.red_flag_position = self.red_flag_home

        self.blue_flag_taken = False
        self.red_flag_taken = False
        self.blue_flag_carrier = None
        self.red_flag_carrier = None

    # ------------------------------------------------------------------
    # Time + win condition
    # ------------------------------------------------------------------
    def tick_seconds(self, dt: float) -> Optional[str]:
        if self.game_over or dt <= 0.0:
            return None

        self.sim_time += dt
        self.current_time -= dt

        # Time-based termination
        if self.current_time <= 0.0 and not self.game_over:
            self.game_over = True
            if self.blue_score > self.red_score:
                # Blue wins on time
                self.add_reward_event(WIN_TEAM_REWARD)  # global => blue agents
                return "BLUE WINS ON TIME"
            elif self.red_score > self.blue_score:
                # Blue loses on time
                self.add_reward_event(LOSS_PENALTY)
                return "RED WINS ON TIME"
            else:
                # Draw: penalize blue to encourage decisive play
                self.add_reward_event(DRAW_PENALTY)
                return "DRAW — BOTH TEAMS LOSE"

        # Score-based termination
        if self.blue_score >= self.score_limit:
            self.game_over = True
            self.add_reward_event(WIN_TEAM_REWARD)
            return "BLUE WINS BY SCORE!"
        if self.red_score >= self.score_limit:
            self.game_over = True
            self.add_reward_event(LOSS_PENALTY)
            return "RED WINS BY SCORE!"

        return None

    # ------------------------------------------------------------------
    # Flag helpers
    # ------------------------------------------------------------------
    def get_enemy_flag_position(self, side: str) -> Tuple[int, int]:
        """Return the *current* position of the enemy flag for a given side."""
        return self.red_flag_position if side == "blue" else self.blue_flag_position

    def get_team_zone_center(self, side: str) -> Tuple[int, int]:
        """Return a reasonable 'team zone' center (flag home) for a given side."""
        return self.blue_flag_home if side == "blue" else self.red_flag_home

    def get_sim_time(self) -> float:
        """Expose simulation time for event-driven RL."""
        return self.sim_time

    def try_pickup_enemy_flag(self, agent) -> bool:
        side = agent.getSide()
        pos_x, pos_y = agent.float_pos

        if side == "blue":
            enemy_flag = self.red_flag_position
            taken_flag = self.red_flag_taken
        else:
            enemy_flag = self.blue_flag_position
            taken_flag = self.blue_flag_taken

        if taken_flag:
            return False

        # Require proximity <= 1 cell in Euclidean distance
        if math.hypot(pos_x - enemy_flag[0], pos_y - enemy_flag[1]) > 1.0:
            return False

        if side == "blue":
            self.red_flag_taken = True
            self.red_flag_carrier = agent
            self.red_flag_position = (-10, -10)  # off-map while carried
        else:
            self.blue_flag_taken = True
            self.blue_flag_carrier = agent
            self.blue_flag_position = (-10, -10)

        self.add_reward_event(FLAG_PICKUP_REWARD, agent_id=agent.unique_id)
        return True

    def try_score_if_carrying_and_home(self, agent) -> bool:
        side = agent.getSide()
        pos_x, pos_y = agent.float_pos

        if side == "blue" and self.red_flag_taken and self.red_flag_carrier is agent:
            if math.hypot(pos_x - self.blue_flag_home[0], pos_y - self.blue_flag_home[1]) <= 2.0:
                self.blue_score += 1
                self.red_flag_taken = False
                self.red_flag_position = self.red_flag_home
                self.red_flag_carrier = None
                self.add_reward_event(FLAG_CARRY_HOME_REWARD, agent_id=agent.unique_id)
                return True

        elif side == "red" and self.blue_flag_taken and self.blue_flag_carrier is agent:
            if math.hypot(pos_x - self.red_flag_home[0], pos_y - self.red_flag_home[1]) <= 2.0:
                self.red_score += 1
                self.blue_flag_taken = False
                self.blue_flag_position = self.blue_flag_home
                self.blue_flag_carrier = None
                self.add_reward_event(FLAG_CARRY_HOME_REWARD, agent_id=agent.unique_id)
                return True

        return False

    def drop_flag_if_carrier_disabled(self, agent) -> None:
        if self.blue_flag_carrier is agent:
            self.blue_flag_taken = False
            self.blue_flag_position = agent.get_position()
            self.blue_flag_carrier = None
            self.add_reward_event(ACTION_FAILED_PUNISHMENT, agent_id=agent.unique_id)

        elif self.red_flag_carrier is agent:
            self.red_flag_taken = False
            self.red_flag_position = agent.get_position()
            self.red_flag_carrier = None
            self.add_reward_event(ACTION_FAILED_PUNISHMENT, agent_id=agent.unique_id)

    # ------------------------------------------------------------------
    # Reward helpers for mines / kills / failed actions
    # (Call these from game_field.py when events happen.)
    # ------------------------------------------------------------------
    def reward_mine_placed(self, agent) -> None:
        """Reward for enabling a mine for the first time."""
        self.add_reward_event(ENABLED_MINE_REWARD, agent_id=agent.unique_id)

    def reward_enemy_killed(self, killer_agent) -> None:
        """Reward for killing an enemy (mine or suppression)."""
        self.add_reward_event(ENEMY_MAV_KILL_REWARD, agent_id=killer_agent.unique_id)

    def punish_failed_action(self, agent) -> None:
        """Penalty for a failed macro-action (e.g., invalid GetFlag / GoTo)."""
        self.add_reward_event(ACTION_FAILED_PUNISHMENT, agent_id=agent.unique_id)

    # ------------------------------------------------------------------
    # Reward event buffer
    # ------------------------------------------------------------------
    def add_reward_event(
        self,
        value: float,
        timestamp: Optional[float] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        t = self.sim_time if timestamp is None else float(timestamp)
        self.reward_events.append((t, agent_id, float(value)))

    def get_step_rewards(self) -> Dict[str, float]:
        rewards: Dict[str, float] = {}
        for _, agent_id, r in self.reward_events:
            if agent_id is None:
                # Global reward for the learning team (blue)
                for aid in ["blue_0", "blue_1"]:
                    rewards.setdefault(aid, 0.0)
                    rewards[aid] += r
            else:
                rewards.setdefault(agent_id, 0.0)
                rewards[agent_id] += r

        self.reward_events.clear()
        return rewards
