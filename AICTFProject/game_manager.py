from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import math

# ================= BASE REWARDS (MATCHING THE 2023 UAV PAPER) =================

WIN_TEAM_REWARD = 2.0
FLAG_PICKUP_REWARD = 0.05
FLAG_CARRY_HOME_REWARD = 1.5
ENABLED_MINE_REWARD = 0.05
ENEMY_MAV_KILL_REWARD = 0.2
ACTION_FAILED_PUNISHMENT = -0.2

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

    # --- Curriculum / shaping state ---
    phase_name: str = "OP1"  # "OP1", "OP2", "OP3", "SELF", etc.

    # --- Per-episode mine effectiveness stats (for HUD) ---
    blue_mine_kills_this_episode: int = 0
    red_mine_kills_this_episode: int = 0
    mines_placed_in_enemy_half_this_episode: int = 0
    mines_triggered_by_red_this_episode: int = 0
    mines_rewarded_by_agent: Dict[str, bool] = field(default_factory=dict)

    # ==================================================================
    # Phase utilities
    # ==================================================================
    def set_phase(self, phase: str) -> None:
        self.phase_name = phase

    # ==================================================================
    # Game reset
    # ==================================================================
    def reset_game(self, reset_scores: bool = True) -> None:
        if reset_scores:
            self.blue_score = 0
            self.red_score = 0

        self.game_over = False
        self.current_time = self.max_time
        self.sim_time = 0.0
        self.reward_events.clear()

        # reset mine reward caps each episode
        self.team_mines_rewarded = {"blue": 0, "red": 0}

        # reset per-episode mine stats
        self.blue_mine_kills_this_episode = 0
        self.red_mine_kills_this_episode = 0
        self.mines_placed_in_enemy_half_this_episode = 0
        self.mines_triggered_by_red_this_episode = 0
        self.mines_rewarded_by_agent.clear()

        mid_row = self.rows // 2
        self.blue_flag_home = (2, mid_row)
        self.red_flag_home = (self.cols - 3, mid_row)

        self.blue_flag_position = self.blue_flag_home
        self.red_flag_position = self.red_flag_home

        self.blue_flag_taken = False
        self.red_flag_taken = False
        self.blue_flag_carrier = None
        self.red_flag_carrier = None

    # ==================================================================
    # Time + win condition
    # ==================================================================
    def tick_seconds(self, dt: float) -> Optional[str]:
        if self.game_over or dt <= 0.0:
            return None

        self.sim_time += dt
        self.current_time -= dt

        # -------------------------------------
        # TIMEOUT CONDITION
        # -------------------------------------
        if self.current_time <= 0.0 and not self.game_over:
            self.game_over = True
            if self.blue_score > self.red_score:
                self.add_reward_event(WIN_TEAM_REWARD)
                return "BLUE WINS ON TIME"
            elif self.red_score > self.blue_score:
                # No LOSS_TEAM_REWARD needed — trainer adds final -1
                return "RED WINS ON TIME"
            else:
                # No draw reward needed — trainer handles terminal bonus
                return "DRAW — NO TEAM REWARD"

        # -------------------------------------
        # SCORE LIMIT CONDITION
        # -------------------------------------
        if self.blue_score >= self.score_limit:
            self.game_over = True
            self.add_reward_event(WIN_TEAM_REWARD)
            return "BLUE WINS BY SCORE!"

        if self.red_score >= self.score_limit:
            self.game_over = True
            # No LOSS_TEAM_REWARD — let trainer apply -1 at terminal
            return "RED WINS BY SCORE!"

        return None

    # ==================================================================
    # Flag helpers
    # ==================================================================
    def get_enemy_flag_position(self, side: str) -> Tuple[int, int]:
        return self.red_flag_position if side == "blue" else self.blue_flag_position

    def get_team_zone_center(self, side: str) -> Tuple[int, int]:
        return self.blue_flag_home if side == "blue" else self.red_flag_home

    def get_sim_time(self) -> float:
        return self.sim_time

    # ------------------------------------------------------------------
    # Flag pickup and scoring
    # ------------------------------------------------------------------
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

        if math.hypot(pos_x - enemy_flag[0], pos_y - enemy_flag[1]) > 1.0:
            return False

        if side == "blue":
            self.red_flag_taken = True
            self.red_flag_carrier = agent
            self.red_flag_position = (-10, -10)
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

    # ==================================================================
    # Reward helpers for mines / kills / failed actions
    # ==================================================================
    def reward_mine_placed(self, agent, mine_pos: Optional[Tuple[int, int]] = None) -> None:
        side = agent.getSide()
        uid = getattr(agent, "unique_id", None)

        # ---------- MATCH PAPER: enabledMineReward only once per UAV per episode ----------
        if uid is not None:
            already_rewarded = self.mines_rewarded_by_agent.get(uid, False)
            if not already_rewarded:
                self.add_reward_event(ENABLED_MINE_REWARD, agent_id=uid)
                self.mines_rewarded_by_agent[uid] = True
        # ---------------------------------------------------------------------

        # Track HUD stats (mines in enemy half) – this can still run every time
        if mine_pos is not None and side == "blue":
            x, y = mine_pos
            mid_x = self.cols * 0.5
            if x > mid_x:
                self.mines_placed_in_enemy_half_this_episode += 1

    def reward_enemy_killed(
            self,
            killer_agent,
            victim_agent: Optional[Any] = None,
            cause: Optional[str] = None,
    ) -> None:
        if cause == "mine":
            if killer_agent.getSide() == "blue":
                self.blue_mine_kills_this_episode += 1
            else:
                self.red_mine_kills_this_episode += 1

        self.add_reward_event(ENEMY_MAV_KILL_REWARD, agent_id=killer_agent.unique_id)

    def record_mine_triggered_by_red(self) -> None:
        """
        Call this from your mine-step / explosion logic when a RED
        agent steps on a BLUE mine and triggers it.
        """
        self.mines_triggered_by_red_this_episode += 1

    def punish_failed_action(self, agent) -> None:
        self.add_reward_event(ACTION_FAILED_PUNISHMENT, agent_id=agent.unique_id)

    # ==================================================================
    # Reward event buffer
    # ==================================================================
    def add_reward_event(
        self,
        value: float,
        timestamp: Optional[float] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        t = self.sim_time if timestamp is None else float(timestamp)
        self.reward_events.append((t, agent_id, float(value)))

    def get_step_rewards(self) -> Dict[Optional[str], float]:
        rewards: Dict[Optional[str], float] = {}
        for _, agent_id, r in self.reward_events:
            rewards.setdefault(agent_id, 0.0)
            rewards[agent_id] += r

        self.reward_events.clear()
        return rewards
