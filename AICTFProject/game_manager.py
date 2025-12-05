# game_manager.py
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import math

# ================= BASE REWARDS (MATCHING THE 2023 UAV PAPER) =================
# Table 3: winTeamReward, flagPickupReward, flagCarryHomeReward,
# enabledLandMineReward, enemyMAVKillReward, actionFailedPunishment
WIN_TEAM_REWARD = 1.0              # winTeamReward
FLAG_PICKUP_REWARD = 0.1           # flagPickupReward
FLAG_CARRY_HOME_REWARD = 0.5       # flagCarryHomeReward
ENABLED_MINE_REWARD = 0.2          # enabledLandMineReward
ENEMY_MAV_KILL_REWARD = 0.5        # enemyMAVKillReward
ACTION_FAILED_PUNISHMENT = -0.2    # actionFailedPunishment
MAX_REWARDED_MINES_PER_TEAM = 3

# For losses and draws (paper only specifies winTeamReward; loss uses -winTeamReward,
# and ties effectively get 0 team reward).
LOSS_TEAM_REWARD = -WIN_TEAM_REWARD
DRAW_TEAM_REWARD = 0.0  # no explicit draw penalty in the paper


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

    team_mines_rewarded: Dict[str, int] = field(
        default_factory=lambda: {"blue": 0, "red": 0}
    )

    # --- Curriculum / shaping state (phase only used for logging if you want) ---
    phase_name: str = "OP1"  # "OP1", "OP2", "OP3", "SELF", etc.

    # (We *do not* use phase-dependent scaling in the pure paper reward.)

    # ==================================================================
    # Phase utilities
    # ==================================================================
    def set_phase(self, phase: str) -> None:
        """
        Called by the trainer whenever the curriculum phase changes,
        e.g. "OP1", "OP2", "OP3".
        Note: phase does NOT affect the reward values in the paper.
        """
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

        # Time-based termination
        if self.current_time <= 0.0 and not self.game_over:
            self.game_over = True
            if self.blue_score > self.red_score:
                # Blue wins on time: +1 team reward
                self.add_reward_event(WIN_TEAM_REWARD)  # global => split across blue agents
                return "BLUE WINS ON TIME"
            elif self.red_score > self.blue_score:
                # Blue loses on time: -1 team reward
                self.add_reward_event(LOSS_TEAM_REWARD)
                return "RED WINS ON TIME"
            else:
                # Draw: 0 team reward in the pure paper scheme
                if abs(DRAW_TEAM_REWARD) > 0.0:
                    self.add_reward_event(DRAW_TEAM_REWARD)
                return "DRAW — NO TEAM REWARD"

        # Score-based termination
        if self.blue_score >= self.score_limit:
            self.game_over = True
            self.add_reward_event(WIN_TEAM_REWARD)
            return "BLUE WINS BY SCORE!"
        if self.red_score >= self.score_limit:
            self.game_over = True
            self.add_reward_event(LOSS_TEAM_REWARD)
            return "RED WINS BY SCORE!"

        return None

    # ==================================================================
    # Flag helpers
    # ==================================================================
    def get_enemy_flag_position(self, side: str) -> Tuple[int, int]:
        """Return the *current* position of the enemy flag for a given side."""
        return self.red_flag_position if side == "blue" else self.blue_flag_position

    def get_team_zone_center(self, side: str) -> Tuple[int, int]:
        """Return a reasonable 'team zone' center (flag home) for a given side."""
        return self.blue_flag_home if side == "blue" else self.red_flag_home

    def get_sim_time(self) -> float:
        """Expose simulation time for event-driven RL."""
        return self.sim_time

    # ------------------------------------------------------------------
    # Flag pickup and scoring (exact paper values; no phase scaling)
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

        # Require proximity <= 1 cell in Euclidean distance
        if math.hypot(pos_x - enemy_flag[0], pos_y - enemy_flag[1]) > 1.0:
            return False

        # Attach flag to carrier
        if side == "blue":
            self.red_flag_taken = True
            self.red_flag_carrier = agent
            self.red_flag_position = (-10, -10)  # off-map while carried
        else:
            self.blue_flag_taken = True
            self.blue_flag_carrier = agent
            self.blue_flag_position = (-10, -10)

        # flagPickupReward: same for both teams in the paper
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

                # flagCarryHomeReward: same value for both teams in the paper
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
            # In the pure paper reward, there is no extra drop penalty beyond failed actions,
            # but you *can* treat this as a failed action if you want.
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

        # Only reward the first few mines per team
        if self.team_mines_rewarded.get(side, 0) >= MAX_REWARDED_MINES_PER_TEAM:
            return

        self.team_mines_rewarded[side] = self.team_mines_rewarded.get(side, 0) + 1
        self.add_reward_event(ENABLED_MINE_REWARD, agent_id=agent.unique_id)

    def reward_enemy_killed(
        self,
        killer_agent,
        victim_agent: Optional[Any] = None,
        cause: Optional[str] = None,
    ) -> None:
        """
        enemyMAVKillReward: reward for killing an opponent with a mine
        or with suppression. The paper uses a single constant value.
        """
        _ = victim_agent
        _ = cause
        self.add_reward_event(ENEMY_MAV_KILL_REWARD, agent_id=killer_agent.unique_id)

    def punish_failed_action(self, agent) -> None:
        """Penalty for a failed macro-action (actionFailedPunishment)."""
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
        """
        Aggregate reward events into a dict keyed by agent_id.

        - agent_id is a string (e.g., "blue_0", "blue_1") for
          per-agent events like flag pickup, mine enabled, etc.
        - agent_id is None for *global* team rewards (win/loss/draw).

        The trainer's collect_blue_rewards_for_step() is responsible
        for turning the global (None) reward into per-agent values,
        so we do NOT split it here. This removes any dependency on
        hard-coded IDs like "blue_0"/"blue_1".
        """
        rewards: Dict[Optional[str], float] = {}
        for _, agent_id, r in self.reward_events:
            rewards.setdefault(agent_id, 0.0)
            rewards[agent_id] += r

        self.reward_events.clear()
        return rewards

