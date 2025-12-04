# game_manager.py
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import math

# ================= BASE REWARDS (PAPER-INSPIRED) =================
WIN_TEAM_REWARD = 1.0

# Base progress rewards (OP1). OP2/OP3 will scale these for BLUE.
FLAG_PICKUP_REWARD_BASE = 0.1
FLAG_CARRY_HOME_REWARD_BASE = 0.5

# Mines & kills (base values)
ENABLED_MINE_REWARD = 0.2
ENEMY_MAV_KILL_REWARD = 0.5

# Gentle step-wise punishment
ACTION_FAILED_PUNISHMENT = -0.2

# Terminal shaped penalties (small but directional)
DRAW_PENALTY = -0.3
LOSS_PENALTY = -0.7

# Extra shaping constants for mines / suppression (blue-focused)
FLAG_CARRIER_KILL_BONUS = 0.5      # extra when killing carrier
DEFENSIVE_KILL_BONUS = 0.2         # extra when kill is near our own flag
MINE_PHASE_SCALE_OP2_OP3 = 1.3     # mines slightly more valuable in OP2/OP3
KILL_PHASE_SCALE_OP2_OP3 = 1.2     # kills slightly more valuable in OP2/OP3

# Radius (in grid cells) to consider a kill "defensive" around home flag
DEFENSIVE_RADIUS_CELLS = 3.0

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

    # Distance-to-home shaping for carriers (per-team)
    blue_carrier_prev_dist: Optional[float] = None
    red_carrier_prev_dist: Optional[float] = None

    # ==================================================================
    # Phase utilities
    # ==================================================================
    def set_phase(self, phase: str) -> None:
        """
        Called by the trainer whenever the curriculum phase changes,
        e.g. "OP1", "OP2", "OP3".
        """
        self.phase_name = phase

    def _progress_scale_for_blue(self) -> float:
        """
        Scale BLUE's progress rewards depending on curriculum phase.

        OP1: 1.0  → pickup=0.5, carry=1.0
        OP2: 1.5  → pickup=0.75, carry=1.5
        OP3: 1.5  → pickup=0.75, carry=1.5
        other: 1.0
        """
        if self.phase_name in ("OP2", "OP3"):
            return 1.5
        return 1.0

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

        # Reset shaping state
        self.blue_carrier_prev_dist = None
        self.red_carrier_prev_dist = None

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
                # Blue wins on time
                self.add_reward_event(WIN_TEAM_REWARD)  # global => blue agents
                return "BLUE WINS ON TIME"
            elif self.red_score > self.blue_score:
                # Blue loses on time
                self.add_reward_event(LOSS_PENALTY)
                return "RED WINS ON TIME"
            else:
                # Draw: mild penalty to encourage decisive play
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
    # Flag pickup and scoring (with BLUE-only phase scaling)
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
            # Reset progress shaping baseline for blue
            self.blue_carrier_prev_dist = None
        else:
            self.blue_flag_taken = True
            self.blue_flag_carrier = agent
            self.blue_flag_position = (-10, -10)
            # Reset progress shaping baseline for red
            self.red_carrier_prev_dist = None

        # Phase-scaled progress reward (BLUE only)
        if side == "blue":
            pickup_reward = FLAG_PICKUP_REWARD_BASE * self._progress_scale_for_blue()
        else:
            pickup_reward = FLAG_PICKUP_REWARD_BASE

        self.add_reward_event(pickup_reward, agent_id=agent.unique_id)
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

                # Phase-scaled carry-home reward (BLUE only)
                carry_reward = FLAG_CARRY_HOME_REWARD_BASE * self._progress_scale_for_blue()
                self.add_reward_event(carry_reward, agent_id=agent.unique_id)

                # Reset shaping state
                self.blue_carrier_prev_dist = None
                return True

        elif side == "red" and self.blue_flag_taken and self.blue_flag_carrier is agent:
            if math.hypot(pos_x - self.red_flag_home[0], pos_y - self.red_flag_home[1]) <= 2.0:
                self.red_score += 1
                self.blue_flag_taken = False
                self.blue_flag_position = self.blue_flag_home
                self.blue_flag_carrier = None

                # Red uses base reward (no scaling; red isn't learning)
                self.add_reward_event(FLAG_CARRY_HOME_REWARD_BASE, agent_id=agent.unique_id)

                # Reset shaping state
                self.red_carrier_prev_dist = None
                return True

        return False

    def drop_flag_if_carrier_disabled(self, agent) -> None:
        if self.blue_flag_carrier is agent:
            self.blue_flag_taken = False
            self.blue_flag_position = agent.get_position()
            self.blue_flag_carrier = None
            self.add_reward_event(ACTION_FAILED_PUNISHMENT, agent_id=agent.unique_id)
            self.blue_carrier_prev_dist = None

        elif self.red_flag_carrier is agent:
            self.red_flag_taken = False
            self.red_flag_position = agent.get_position()
            self.red_flag_carrier = None
            self.add_reward_event(ACTION_FAILED_PUNISHMENT, agent_id=agent.unique_id)
            self.red_carrier_prev_dist = None

    # ==================================================================
    # Distance-to-home shaping while carrying (BLUE-focused)
    # ==================================================================
    def reward_flag_carry_progress(self, agent, shaping_scale: float = 0.02) -> None:
        """
        Potential-based shaping: when BLUE is carrying the red flag,
        give small positive reward for moving closer to home, and
        small negative reward for moving away.

        Call this once per sim step for each agent (GameField.update).
        """
        side = agent.getSide()
        pos_x, pos_y = agent.float_pos

        # ---- BLUE carrier shaping (this is what helps PPO) ----
        if side == "blue" and self.red_flag_carrier is agent:
            # Distance from blue base
            dist_now = math.hypot(pos_x - self.blue_flag_home[0],
                                  pos_y - self.blue_flag_home[1])
            prev = self.blue_carrier_prev_dist
            if prev is not None:
                delta = prev - dist_now  # >0: moved closer, <0: moved away
                if abs(delta) > 1e-6:
                    # reward ~ +0.02 * distance-improvement
                    self.add_reward_event(shaping_scale * delta, agent_id=agent.unique_id)
            self.blue_carrier_prev_dist = dist_now

        # ---- Optional: Red carrier shaping (kept symmetric but smaller) ----
        elif side == "red" and self.blue_flag_carrier is agent:
            dist_now = math.hypot(pos_x - self.red_flag_home[0],
                                  pos_y - self.red_flag_home[1])
            prev = self.red_carrier_prev_dist
            if prev is not None:
                delta = prev - dist_now
                if abs(delta) > 1e-6:
                    # You can set shaping_scale_red = 0.0 to disable if you want.
                    shaping_scale_red = 0.0  # effectively off for red
                    self.add_reward_event(shaping_scale_red * delta, agent_id=agent.unique_id)
            self.red_carrier_prev_dist = dist_now

    # ==================================================================
    # Reward helpers for mines / kills / failed actions
    # ==================================================================
    def reward_mine_placed(
        self,
        agent,
        mine_pos: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Reward for enabling a mine.

        - BLUE gets slightly higher mine reward in OP2/OP3.
        - If mine_pos is provided, we give a tiny bonus when the mine is
          placed "forward" (toward enemy side) to encourage aggressive,
          suppressive mines rather than random spam near spawn.
        """
        side = getattr(agent, "side", agent.getSide())
        reward = ENABLED_MINE_REWARD

        # Phase scaling for BLUE only
        if side == "blue" and self.phase_name in ("OP2", "OP3"):
            reward *= MINE_PHASE_SCALE_OP2_OP3

        # Positional shaping if we know the mine position
        if mine_pos is not None:
            mx, my = mine_pos
            # Simple heuristic: normalized x position, 0=blue left, 1=red right
            norm_x = mx / max(1.0, float(self.cols - 1))
            if side == "blue":
                # Reward a bit if mine is placed toward enemy half
                reward *= (1.0 + 0.15 * norm_x)
            else:
                # Red mines toward blue side (mirrored)
                reward *= (1.0 + 0.15 * (1.0 - norm_x))

        self.add_reward_event(reward, agent_id=agent.unique_id)

    def reward_enemy_killed(
        self,
        killer_agent,
        victim_agent: Optional[Any] = None,
        cause: Optional[str] = None,
    ) -> None:
        """
        Reward for killing an enemy (mine or suppression).

        Parameters
        ----------
        killer_agent : Agent
            The friendly agent credited with the kill (usually BLUE).
        victim_agent : Agent, optional
            The enemy agent that died (used for defensive / carrier bonuses).
        cause : str, optional
            "mine" or "suppression" or None, used for slight scaling.
        """
        side = getattr(killer_agent, "side", killer_agent.getSide())
        reward = ENEMY_MAV_KILL_REWARD

        # Slightly prefer mine-based kills for BLUE (area denial / traps)
        if cause == "mine":
            reward *= 1.2
        elif cause == "suppression":
            reward *= 1.0  # explicit, just for clarity

        # Bonus if we killed the enemy flag carrier
        if victim_agent is not None and hasattr(victim_agent, "isCarryingFlag"):
            try:
                if victim_agent.isCarryingFlag():
                    reward += FLAG_CARRIER_KILL_BONUS
            except TypeError:
                # In case isCarryingFlag isn't callable in some variant
                pass

        # Defensive bonus: kill near our own flag
        if victim_agent is not None and hasattr(victim_agent, "float_pos"):
            vx, vy = victim_agent.float_pos
            if side == "blue":
                hx, hy = self.blue_flag_home
            else:
                hx, hy = self.red_flag_home

            dist_home = math.hypot(vx - hx, vy - hy)
            if dist_home <= DEFENSIVE_RADIUS_CELLS:
                reward += DEFENSIVE_KILL_BONUS

        # Phase scaling for BLUE: in later curriculum, emphasize good trades
        if side == "blue" and self.phase_name in ("OP2", "OP3"):
            reward *= KILL_PHASE_SCALE_OP2_OP3

        self.add_reward_event(reward, agent_id=killer_agent.unique_id)

    def punish_failed_action(self, agent) -> None:
        """Penalty for a failed macro-action (e.g., invalid GetFlag / GoTo)."""
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

    def get_step_rewards(self) -> Dict[str, float]:
        """
        Aggregate reward events into per-agent rewards.

        Global events (agent_id is None) are interpreted as team-level reward
        and split evenly between the two blue agents, so DRAW/LOSS/WIN values
        are not accidentally doubled.
        """
        rewards: Dict[str, float] = {}
        for _, agent_id, r in self.reward_events:
            if agent_id is None:
                # Global reward for the learning team (blue): split evenly.
                per_agent = r / 2.0
                for aid in ["blue_0", "blue_1"]:
                    rewards.setdefault(aid, 0.0)
                    rewards[aid] += per_agent
            else:
                rewards.setdefault(agent_id, 0.0)
                rewards[agent_id] += r

        self.reward_events.clear()
        return rewards
