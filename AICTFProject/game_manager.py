from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import math

# ==============================
# REWARD CONSTANTS (30x40 ARENA)
# ==============================
WIN_TEAM_REWARD = 5.0            # team reward when BLUE wins the game
FLAG_PICKUP_REWARD = 0.5         # shaping: "you found the enemy flag"
FLAG_CARRY_HOME_REWARD = 3.0     # shaping: "you actually scored"
ENABLED_MINE_REWARD = 0.1        # reward once per UAV per episode for enabling mine use
ENEMY_MAV_KILL_REWARD = 0.8      # reward for removing an enemy (mine or suppression)
ACTION_FAILED_PUNISHMENT = -0.5  # e.g. dying while holding a flag, bad macro, etc.
FLAG_RETURN_DELAY = 10.0
DRAW_TIMEOUT_PENALTY = 0.0      # applied when game times out at 0–0
SCORE_PROGRESS_BONUS = 0.2    # per Blue score (team-level)
SCORE_MARGIN_BONUS   = 0.5    # extra bonus per goal of margin above 1
FLAG_PROXIMITY_COEF = 0.02
ATTACKER_FLAG_PROX_COEF = 0.04   # stronger shaping toward attack/scoring
DEFENDER_FLAG_PROX_COEF = 0.02   # shaping toward defending own flag

PHASE_DRAW_TIMEOUT_PENALTY = {
    "OP1": -2.0,
    "OP2": -1.0,
    "OP3": 0.0,   # no penalty for stalling vs hardest scripted opponent
}

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

    # --- Flag drop timers (for auto-return) ---
    blue_flag_drop_time: Optional[float] = None
    red_flag_drop_time: Optional[float] = None

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

    # Track per-team mine rewards (if you cap them anywhere else)
    team_mines_rewarded: Dict[str, int] = field(
        default_factory=lambda: {"blue": 0, "red": 0}
    )

    def set_phase(self, phase: str) -> None:
        self.phase_name = phase

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

        # clear drop timers
        self.blue_flag_drop_time = None
        self.red_flag_drop_time = None

    # Time + win condition + flag auto-return
    def tick_seconds(self, dt: float) -> Optional[str]:
        if self.game_over or dt <= 0.0:
            return None

        self.sim_time += dt
        self.current_time -= dt
        # AUTO-RETURN DROPPED FLAGS
        self._update_flag_auto_return()

        # TIMEOUT CONDITION
        if self.current_time <= 0.0 and not self.game_over:
            self.game_over = True

            if self.blue_score > self.red_score:
                # Margin-based bonus: 3–0 > 2–1 > 1–0
                margin = self.blue_score - self.red_score
                extra = SCORE_MARGIN_BONUS * max(0, margin - 1)
                self.add_reward_event(WIN_TEAM_REWARD + extra)
                return "BLUE WINS ON TIME"

            elif self.red_score > self.blue_score:
                return "RED WINS ON TIME"

            else:
                # Phase-dependent draw penalty for BLUE
                penalty = PHASE_DRAW_TIMEOUT_PENALTY.get(
                    self.phase_name,
                    DRAW_TIMEOUT_PENALTY,
                )
                if penalty != 0.0:
                    self.add_reward_event(penalty)

                return "DRAW — BLUE PENALIZED FOR STALL"

        # SCORE LIMIT CONDITION
        if self.blue_score >= self.score_limit:
            self.game_over = True
            margin = self.blue_score - self.red_score
            extra = SCORE_MARGIN_BONUS * max(0, margin - 1)
            self.add_reward_event(WIN_TEAM_REWARD + extra)
            return "BLUE WINS BY SCORE!"

        if self.red_score >= self.score_limit:
            self.game_over = True
            return "RED WINS BY SCORE!"

        return None

    def _update_flag_auto_return(self) -> None:
        # BLUE flag auto-return
        if (not self.blue_flag_taken
                and self.blue_flag_position != self.blue_flag_home
                and self.blue_flag_drop_time is not None):
            if self.sim_time - self.blue_flag_drop_time >= FLAG_RETURN_DELAY:
                self.blue_flag_position = self.blue_flag_home
                self.blue_flag_drop_time = None  # timer consumed

        # RED flag auto-return
        if (not self.red_flag_taken
                and self.red_flag_position != self.red_flag_home
                and self.red_flag_drop_time is not None):
            if self.sim_time - self.red_flag_drop_time >= FLAG_RETURN_DELAY:
                self.red_flag_position = self.red_flag_home
                self.red_flag_drop_time = None  # timer consumed

    # Flag helpers
    def get_enemy_flag_position(self, side: str) -> Tuple[int, int]:
        return self.red_flag_position if side == "blue" else self.blue_flag_position

    def get_team_zone_center(self, side: str) -> Tuple[int, int]:
        return self.blue_flag_home if side == "blue" else self.red_flag_home

    def get_sim_time(self) -> float:
        return self.sim_time

    # Flag pickup and scoring
    def try_pickup_enemy_flag(self, agent) -> bool:
        side = agent.getSide()
        pos_x, pos_y = agent.float_pos

        if side == "blue":
            enemy_flag = self.red_flag_position
            taken_flag = self.red_flag_taken
        else:
            enemy_flag = self.blue_flag_position
            taken_flag = self.blue_flag_taken

        # Already taken by someone else
        if taken_flag:
            return False

        # Must be close enough to pick up
        if math.hypot(pos_x - enemy_flag[0], pos_y - enemy_flag[1]) > 1.0:
            return False

        # Take the flag and hide it off-map while carried
        if side == "blue":
            self.red_flag_taken = True
            self.red_flag_carrier = agent
            self.red_flag_position = (-10, -10)
            self.red_flag_drop_time = None  # cancel any drop timer
        else:
            self.blue_flag_taken = True
            self.blue_flag_carrier = agent
            self.blue_flag_position = (-10, -10)
            self.blue_flag_drop_time = None

        # Shaping: picking up the flag is already a significant event
        self.add_reward_event(FLAG_PICKUP_REWARD, agent_id=agent.unique_id)
        return True

    def try_score_if_carrying_and_home(self, agent) -> bool:
        side = agent.getSide()
        pos_x, pos_y = agent.float_pos

        # BLUE scoring with RED flag
        if side == "blue" and self.red_flag_taken and self.red_flag_carrier is agent:
            if math.hypot(pos_x - self.blue_flag_home[0], pos_y - self.blue_flag_home[1]) <= 2.0:
                # increment score *first*
                self.blue_score += 1

                # reset flag state
                self.red_flag_taken = False
                self.red_flag_position = self.red_flag_home
                self.red_flag_carrier = None
                self.red_flag_drop_time = None

                # Big shaping reward for the scoring carrier.
                self.add_reward_event(FLAG_CARRY_HOME_REWARD, agent_id=agent.unique_id)

                # Team-wide bonus so defenders/supports also feel the score.
                self.add_reward_event(FLAG_CARRY_HOME_REWARD * 0.5)

                # This is team-level, so both MAPPO agents are pushed to keep attacking.
                self.add_reward_event(SCORE_PROGRESS_BONUS * float(self.blue_score))

                return True

        # RED scoring with BLUE flag (no RL reward, scripted opponent)
        elif side == "red" and self.blue_flag_taken and self.blue_flag_carrier is agent:
            if math.hypot(pos_x - self.red_flag_home[0], pos_y - self.red_flag_home[1]) <= 2.0:
                self.red_score += 1
                self.blue_flag_taken = False
                self.blue_flag_position = self.blue_flag_home
                self.blue_flag_carrier = None
                self.blue_flag_drop_time = None

                # Team-level penalty for BLUE when RED scores.
                self.add_reward_event(-FLAG_CARRY_HOME_REWARD * 0.5)

                return True

        return False

    # Flag drop on death / disable
    def drop_flag_if_carrier_disabled(self, agent) -> None:
        # Use the agent's current float_pos to decide where the flag lands
        if hasattr(agent, "float_pos"):
            fx, fy = agent.float_pos
            drop_pos = (int(round(fx)), int(round(fy)))
        else:
            drop_pos = agent.get_position()

        if self.blue_flag_carrier is agent:
            self.blue_flag_taken = False
            self.blue_flag_position = drop_pos
            self.blue_flag_carrier = None
            self.blue_flag_drop_time = self.sim_time
            # Punish dropping the flag (usually due to death / bad play)
            self.add_reward_event(ACTION_FAILED_PUNISHMENT, agent_id=agent.unique_id)

        elif self.red_flag_carrier is agent:
            self.red_flag_taken = False
            self.red_flag_position = drop_pos
            self.red_flag_carrier = None
            self.red_flag_drop_time = self.sim_time
            self.add_reward_event(ACTION_FAILED_PUNISHMENT, agent_id=agent.unique_id)

    # Agent lifecycle helpers (for death / respawn)
    def handle_agent_death(self, agent) -> None:
        self.drop_flag_if_carrier_disabled(agent)

    def clear_flag_carrier_if_agent(self, agent) -> None:
        if self.blue_flag_carrier is agent:
            self.blue_flag_carrier = None
            self.blue_flag_taken = False
            self.blue_flag_position = self.blue_flag_home
            self.blue_flag_drop_time = None

        if self.red_flag_carrier is agent:
            self.red_flag_carrier = None
            self.red_flag_taken = False
            self.red_flag_position = self.red_flag_home
            self.red_flag_drop_time = None

    def reward_flag_proximity(self, agent, prev_dist_to_goal: float) -> None:
        """
        Potential-Based Shaping (PBS) reward based on distance to a role-specific goal.

        For BLUE:
          - Agent 0 (attacker): goal = enemy flag (or home if carrying enemy flag)
          - Agent 1 (defender): goal = own flag home (defensive anchor)

        Φ(s) = 1 - dist_to_goal / max_dist      (in [0, 1])
        r_shaping = coef(role) * (Φ(s') - Φ(s))

        This preserves optimal policies while giving dense, role-aware guidance.
        """

        side = agent.getSide()
        if side not in ("blue", "red"):
            return

        # For now we only shape BLUE (learning) agents.
        if side != "blue":
            return

        # Role: attacker = agent_id 0, defender = agent_id 1 (default to attacker if missing)
        agent_id = getattr(agent, "agent_id", 0)
        is_attacker = (agent_id == 0)
        is_defender = (agent_id == 1)

        # Agent position (prefer float_pos if available)
        if hasattr(agent, "float_pos"):
            ax, ay = agent.float_pos
        else:
            ax, ay = getattr(agent, "x", 0.0), getattr(agent, "y", 0.0)

        # ----- Choose role-specific goal and coef -----
        if is_attacker:
            # Attacker: move toward enemy flag; if carrying, move toward own home to score
            if self.red_flag_taken and self.red_flag_carrier is agent:
                goal_x, goal_y = self.blue_flag_home
            else:
                goal_x, goal_y = self.red_flag_position
            coef = ATTACKER_FLAG_PROX_COEF

        elif is_defender:
            # Defender: always shape toward own flag home (hold defensive posture)
            goal_x, goal_y = self.blue_flag_home
            coef = DEFENDER_FLAG_PROX_COEF

        else:
            # Unknown role => no shaping (keeps things clean if you add more agents later)
            return

        # Current distance to the role-specific goal
        cur_dist = math.dist([ax, ay], [goal_x, goal_y])

        # Normalize by maximum possible distance in this grid (diagonal)
        max_dist = math.sqrt(self.cols ** 2 + self.rows ** 2)
        if max_dist <= 0.0:
            return

        # Clamp distances
        prev_d = max(0.0, min(prev_dist_to_goal, max_dist))
        cur_d = max(0.0, min(cur_dist, max_dist))

        # Potential function Φ(s) = 1 - d / max_d (closer = higher potential)
        phi_before = 1.0 - (prev_d / max_dist)
        phi_after = 1.0 - (cur_d / max_dist)

        delta_phi = phi_after - phi_before
        shaped_reward = coef * delta_phi

        if abs(shaped_reward) > 0.0:
            uid = getattr(agent, "unique_id", None)
            self.add_reward_event(shaped_reward, agent_id=uid)

    def reward_potential_shaping(
            self,
            agent,
            start_pos: Tuple[int, int],
            end_pos: Tuple[int, int],
    ) -> None:
        """
        Potential-based shaping reward using the *planned* macro movement:

            Φ(s) = 1 - dist_to_goal / max_dist
            r_shaping = coef(role) * (Φ(s') - Φ(s))

        Role-conditional for BLUE:
          - Agent 0 (attacker): goal = enemy flag (or home if carrying enemy flag)
          - Agent 1 (defender): goal = own flag home
        """

        side = agent.getSide()

        # Only shape BLUE (learning) agents
        if side != "blue":
            return

        agent_id = getattr(agent, "agent_id", 0)
        is_attacker = (agent_id == 0)
        is_defender = (agent_id == 1)

        # ----- Choose role-specific goal and coef -----
        if is_attacker:
            # Attacker: toward enemy flag, or toward home if carrying
            if self.red_flag_taken and self.red_flag_carrier is agent:
                goal_x, goal_y = self.blue_flag_home
            else:
                goal_x, goal_y = self.red_flag_position
            coef = ATTACKER_FLAG_PROX_COEF

        elif is_defender:
            # Defender: guard own flag home
            goal_x, goal_y = self.blue_flag_home
            coef = DEFENDER_FLAG_PROX_COEF

        else:
            # Unknown role => no shaping
            return

        max_dist = math.sqrt(self.cols ** 2 + self.rows ** 2)
        if max_dist <= 0.0:
            return

        sx, sy = start_pos
        ex, ey = end_pos

        prev_d = math.dist([sx, sy], [goal_x, goal_y])
        cur_d = math.dist([ex, ey], [goal_x, goal_y])

        # Clamp
        prev_d = max(0.0, min(prev_d, max_dist))
        cur_d = max(0.0, min(cur_d, max_dist))

        phi_before = 1.0 - (prev_d / max_dist)
        phi_after = 1.0 - (cur_d / max_dist)
        delta_phi = phi_after - phi_before

        shaped_reward = coef * delta_phi
        if abs(shaped_reward) > 0.0:
            uid = getattr(agent, "unique_id", None)
            self.add_reward_event(shaped_reward, agent_id=uid)

    # Reward helpers for mines / kills / failed actions
    def reward_mine_placed(self, agent, mine_pos: Optional[Tuple[int, int]] = None) -> None:
        side = agent.getSide()
        uid = getattr(agent, "unique_id", None)

        # enabledMineReward only once per UAV per episode
        if uid is not None:
            already_rewarded = self.mines_rewarded_by_agent.get(uid, False)
            if not already_rewarded:
                self.add_reward_event(ENABLED_MINE_REWARD, agent_id=uid)
                self.mines_rewarded_by_agent[uid] = True

        # Track HUD stats
        if mine_pos is not None and side == "blue":
            x, _ = mine_pos
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

        # Individual reward for whoever got the kill
        self.add_reward_event(ENEMY_MAV_KILL_REWARD, agent_id=killer_agent.unique_id)

        # NEW: if BLUE gets a mine kill, also give the whole team a small bonus
        if killer_agent.getSide() == "blue" and cause == "mine":
            self.add_reward_event(ENEMY_MAV_KILL_REWARD * 0.5)

    def record_mine_triggered_by_red(self) -> None:
        self.mines_triggered_by_red_this_episode += 1

    def punish_failed_action(self, agent) -> None:
        self.add_reward_event(ACTION_FAILED_PUNISHMENT, agent_id=agent.unique_id)

    # Reward event buffer (for event-driven RL)
    def add_reward_event(
        self,
        value: float,
        timestamp: Optional[float] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        t = self.sim_time if timestamp is None else float(timestamp)
        # The reward_events list stores (timestamp, agent_id, reward_value)
        self.reward_events.append((t, agent_id, float(value)))

    def pop_reward_events(self) -> List[Tuple[float, Optional[str], float]]:
        """
        Returns all collected reward events (t, agent_id, reward) and clears
        the internal buffer.

        This is the preferred method for the continuous-time TD/GAE trainer.
        """
        events = self.reward_events
        self.reward_events = []
        return events
