# game_manager.py
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import math

# ================= PAPER-INSPIRED REWARDS (TUNED FOR "WIN OR SUFFER") =================
WIN_TEAM_REWARD           = 1.0

# Make *scoring* the main jackpot; mid-game candy is small
FLAG_PICKUP_REWARD        = 0.05   # was 0.20 / 0.10
FLAG_CARRY_HOME_REWARD    = 1.00   # was 0.80 / 0.50
ENABLED_MINE_REWARD       = 0.05   # was 0.30 / 0.20
ENEMY_MAV_KILL_REWARD     = 0.20   # was 0.80 / 0.50

# Failed actions still feel bad
ACTION_FAILED_PUNISHMENT  = -0.25  # was -0.20

# Draw is *much* worse than winning, and notably worse than losing
DRAW_PENALTY              = -4.0   # was -2.0
LOSS_PENALTY              = -1.0   # unchanged

# Base global time penalty per second (will be ramped up as the match goes on)
TIME_PENALTY_PER_SECOND   = -0.0008  # was -0.0005

# Shaping coefficients
DISTANCE_TOWARD_FLAG_COEF = 0.005   # was 0.02 → 0.01; now softer
CARRIER_HOME_COEF         = 0.03    # reward for moving home with flag


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
    # List of (timestamp, agent_id, reward_value, tag)
    reward_events: List[Tuple[float, Optional[str], float, str]] = field(default_factory=list)

    # --- Debugging / tracking ---
    debug_rewards: bool = False  # flip from trainer for spam logs
    episode_event_counts: Dict[str, int] = field(default_factory=dict)
    episode_reward_totals: Dict[str, float] = field(default_factory=dict)

    # Track previous distance-to-flag per blue agent for shaping
    last_flag_distance: Dict[str, float] = field(default_factory=dict)

    # Track distance from carrier to blue home while carrying enemy flag
    carrier_last_home_distance: Dict[str, float] = field(default_factory=dict)

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
        self.episode_event_counts.clear()
        self.episode_reward_totals.clear()
        self.last_flag_distance.clear()
        self.carrier_last_home_distance.clear()

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

        # Global "living" time penalty to discourage long stalemates.
        # This ramps up as the episode drags on so late-game stalling hurts more.
        if TIME_PENALTY_PER_SECOND != 0.0:
            progress = 0.0
            if self.max_time > 0.0:
                progress = max(0.0, min(1.0, self.sim_time / self.max_time))
            # Multiplier goes from 1.0 → 5.0 over the episode
            time_mult = 1.0 + 4.0 * progress
            self.add_reward_event(
                TIME_PENALTY_PER_SECOND * time_mult * dt,
                agent_id=None,
                tag="time_penalty",
            )

        # Time-based termination
        if self.current_time <= 0.0 and not self.game_over:
            self.game_over = True
            if self.blue_score > self.red_score:
                # Blue wins on time
                self.add_reward_event(
                    WIN_TEAM_REWARD,
                    agent_id=None,
                    tag="blue_win_time",
                )
                return "BLUE WINS ON TIME"
            elif self.red_score > self.blue_score:
                # Blue loses on time
                self.add_reward_event(
                    LOSS_PENALTY,
                    agent_id=None,
                    tag="blue_loss_time",
                )
                return "RED WINS ON TIME"
            else:
                # Draw: heavily penalize to discourage farming high-reward stalemates
                self.add_reward_event(
                    DRAW_PENALTY,
                    agent_id=None,
                    tag="draw_penalty",
                )
                return "DRAW — BOTH TEAMS LOSE"

        # Score-based termination
        if self.blue_score >= self.score_limit:
            self.game_over = True
            self.add_reward_event(
                WIN_TEAM_REWARD,
                agent_id=None,
                tag="blue_win_score",
            )
            return "BLUE WINS BY SCORE!"
        if self.red_score >= self.score_limit:
            self.game_over = True
            self.add_reward_event(
                LOSS_PENALTY,
                agent_id=None,
                tag="blue_loss_score",
            )
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

        self.add_reward_event(
            FLAG_PICKUP_REWARD,
            agent_id=agent.unique_id,
            tag="flag_pickup",
        )
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
                self.add_reward_event(
                    FLAG_CARRY_HOME_REWARD,
                    agent_id=agent.unique_id,
                    tag="flag_score",
                )
                return True

        elif side == "red" and self.blue_flag_taken and self.blue_flag_carrier is agent:
            if math.hypot(pos_x - self.red_flag_home[0], pos_y - self.red_flag_home[1]) <= 2.0:
                self.red_score += 1
                self.blue_flag_taken = False
                self.blue_flag_position = self.blue_flag_home
                self.blue_flag_carrier = None
                self.add_reward_event(
                    FLAG_CARRY_HOME_REWARD,
                    agent_id=agent.unique_id,
                    tag="flag_score",
                )
                return True

        return False

    def drop_flag_if_carrier_disabled(self, agent) -> None:
        if self.blue_flag_carrier is agent:
            self.blue_flag_taken = False
            self.blue_flag_position = agent.get_position()
            self.blue_flag_carrier = None
            self.add_reward_event(
                ACTION_FAILED_PUNISHMENT,
                agent_id=agent.unique_id,
                tag="carrier_disabled",
            )

        elif self.red_flag_carrier is agent:
            self.red_flag_taken = False
            self.red_flag_position = agent.get_position()
            self.red_flag_carrier = None
            self.add_reward_event(
                ACTION_FAILED_PUNISHMENT,
                agent_id=agent.unique_id,
                tag="carrier_disabled",
            )

    # ------------------------------------------------------------------
    # Distance-based shaping for BLUE (move toward enemy flag)
    # ------------------------------------------------------------------
    def distance_shaping_for_blue(self, blue_agents: List[Any]) -> None:
        """
        For each blue agent, reward reductions in distance to the *current*
        enemy flag position since last step.
        """
        if not blue_agents:
            return

        enemy_flag = self.get_enemy_flag_position("blue")

        for agent in blue_agents:
            aid = agent.unique_id
            x, y = agent.float_pos
            d = math.hypot(x - enemy_flag[0], y - enemy_flag[1])

            prev = self.last_flag_distance.get(aid)
            self.last_flag_distance[aid] = d

            # First time we see this agent: just initialize, no reward
            if prev is None:
                continue

            delta = prev - d  # positive if moved closer
            if delta > 0.0:
                reward = DISTANCE_TOWARD_FLAG_COEF * float(delta)
                self.add_reward_event(
                    reward,
                    agent_id=aid,
                    tag="toward_flag",
                )

    # ------------------------------------------------------------------
    # Distance-based shaping for BLUE FLAG CARRIER (move toward home)
    # ------------------------------------------------------------------
    def distance_shaping_for_blue_carrier(self, blue_agents: List[Any]) -> None:
        """
        If a BLUE agent is carrying the RED flag, reward reductions in
        distance to BLUE home (blue_flag_home) since last step.

        This strongly encourages *finishing* the capture instead of
        just dancing around with the flag.
        """
        if not self.red_flag_taken or self.red_flag_carrier is None:
            return

        carrier = self.red_flag_carrier

        # Safety: carrier must still be a live blue agent
        if carrier not in blue_agents:
            return

        aid = carrier.unique_id
        x, y = carrier.float_pos

        # Distance from carrier to BLUE home base
        d = math.hypot(x - self.blue_flag_home[0], y - self.blue_flag_home[1])

        prev = self.carrier_last_home_distance.get(aid)
        self.carrier_last_home_distance[aid] = d

        # First time we see this carrier – just initialize
        if prev is None:
            return

        delta = prev - d  # positive if moved closer to home

        if delta > 0.0:
            # Reward moving closer to home
            reward = CARRIER_HOME_COEF * float(delta)
            self.add_reward_event(
                reward,
                agent_id=aid,
                tag="carrier_toward_home",
            )
        elif delta < 0.0:
            # Small penalty for moving away from home (don't farm time with flag)
            away = -delta
            penalty = 0.5 * CARRIER_HOME_COEF * float(away)
            self.add_reward_event(
                -penalty,
                agent_id=aid,
                tag="carrier_away_home",
            )

    # ------------------------------------------------------------------
    # Reward helpers for mines / kills / failed actions
    # ------------------------------------------------------------------
    def reward_mine_placed(self, agent) -> None:
        """Reward for enabling a mine for the first time."""
        self.add_reward_event(
            ENABLED_MINE_REWARD,
            agent_id=agent.unique_id,
            tag="mine_enabled",
        )

    def reward_enemy_killed(self, killer_agent) -> None:
        """Reward for killing an enemy (mine or suppression)."""
        self.add_reward_event(
            ENEMY_MAV_KILL_REWARD,
            agent_id=killer_agent.unique_id,
            tag="enemy_kill",
        )

    def punish_failed_action(self, agent) -> None:
        """Penalty for a failed macro-action (e.g., invalid GetFlag / GoTo)."""
        self.add_reward_event(
            ACTION_FAILED_PUNISHMENT,
            agent_id=agent.unique_id,
            tag="failed_action",
        )

    # ------------------------------------------------------------------
    # Reward event buffer
    # ------------------------------------------------------------------
    def add_reward_event(
        self,
        value: float,
        timestamp: Optional[float] = None,
        agent_id: Optional[str] = None,
        tag: str = "generic",
    ) -> None:
        """
        Store the raw event. IMPORTANT: global team rewards keep agent_id=None
        so they can be split over actual blue agents later in the trainer.
        """
        t = self.sim_time if timestamp is None else float(timestamp)
        value = float(value)

        # Store raw event (agent_id may be None for globals)
        self.reward_events.append((t, agent_id, value, tag))

        # Aggregate per-episode stats
        self.episode_event_counts[tag] = self.episode_event_counts.get(tag, 0) + 1
        self.episode_reward_totals[tag] = self.episode_reward_totals.get(tag, 0.0) + value

        if self.debug_rewards:
            who = agent_id if agent_id is not None else "GLOBAL_BLUE_TEAM"
            print(
                f"[REWARD-EVENT] t={t:6.2f} tag={tag:16s} "
                f"agent={who:16s} value={value:+.4f}"
            )

    def get_step_rewards(self, debug: bool = False) -> Dict[Optional[str], float]:
        """
        Aggregate all reward_events since the last call into a dict mapping:
        agent_id (or None) -> total_reward_for_this_step

        - Global events are kept under key None.
        - Per-agent events use that agent's unique_id.
        """
        rewards: Dict[Optional[str], float] = {}

        if not self.reward_events:
            return rewards

        for _, agent_id, r, tag in self.reward_events:
            key = agent_id  # may be None for globals
            rewards.setdefault(key, 0.0)
            rewards[key] += r

        if debug or self.debug_rewards:
            pretty = { (k if k is not None else "GLOBAL"): v for k, v in rewards.items() }
            print(f"[STEP-REWARDS] t={self.sim_time:6.2f} raw={pretty}")

        # Clear for next step
        self.reward_events.clear()
        return rewards
