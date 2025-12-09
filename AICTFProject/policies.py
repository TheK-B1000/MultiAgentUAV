"""
policies.py

Scripted baseline policies for the 2-vs-2 UAV Capture-the-Flag (CTF) environment.

These policies are used as fixed opponents (OP1, OP2, OP3) and as baselines
for benchmarking multi-agent RL algorithms (e.g., PPO, MAPPO, QMIX, self-play).
They expose a simple macro-action interface:

    select_action(obs, agent, game_field) -> (macro_action_id: int,
                                              param: Optional[(x, y)])

Macro-actions (see macro_actions.MacroAction):
    - GO_TO      : Move towards a target grid cell (x, y).
    - GRAB_MINE  : Go to own mine pickups.
    - GET_FLAG   : Go to enemy flag.
    - PLACE_MINE : Place a mine at a target grid cell (x, y).
    - GO_HOME    : Return to own flag/home zone.
"""

import math
import random
from typing import Any, Tuple, Optional

from agents import Agent
from game_manager import GameManager
from macro_actions import MacroAction


class Policy:
    """
    Base class for scripted policies in the CTF environment.

    This abstract class defines the interface for agent decision-making. Subclasses
    implement `select_action` to return a macro-action ID and optional parameter
    (e.g., a target cell).

    Research context:
        Scripted policies serve as baselines for benchmarking MARL algorithms
        (PPO, MAPPO, QMIX, self-play). They enable controlled evaluation of
        emergent behaviors, coordination, and robustness to fixed opponents.
    """

    def select_action(
        self,
        obs: Any,
        agent: Agent,
        game_field: "GameField",
    ) -> Tuple[int, Optional[Tuple[int, int]]]:
        """
        Select an action for the agent based on observation, state, and environment.

        Args:
            obs: Agent's observation (e.g., 7×30×40 CNN tensor).
            agent: The acting agent instance.
            game_field: The GameField environment.

        Returns:
            (macro_action_id: int, param: Optional[(x, y)] for target).
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Optional hook for per-episode state reset.

        Scripted baselines are currently stateless, but this method allows the
        environment to treat scripted and learned policies uniformly.
        """
        # Default: nothing to reset.
        return None


class OP1RedPolicy(Policy):
    """
    OP1: Naive, purely defensive baseline.

    Behavior (for the configured `side`):
        - If not carrying a flag: move to the back of its own end zone and stay there.
        - If somehow carrying the enemy flag: GO_HOME.

    Research utility:
        - Extremely weak opponent used as an early curriculum stage.
        - RL policies should quickly learn to outperform OP1 and achieve near-perfect win rate.
    """

    def __init__(self, side: str = "red"):
        self.side = side

    def select_action(
        self,
        obs: Any,
        agent: Agent,
        game_field: "GameField",
    ) -> Tuple[int, Optional[Tuple[int, int]]]:
        gm: GameManager = game_field.manager
        home_x, home_y = gm.get_team_zone_center(self.side)

        # "Back" of the end zone: one cell deeper in own zone along X.
        if self.side == "red":
            target = (min(game_field.col_count - 1, home_x + 1), home_y)
        else:
            target = (max(0, home_x - 1), home_y)

        # Edge case: if carrying enemy flag, go home.
        if agent.isCarryingFlag():
            return int(MacroAction.GO_HOME), None

        return int(MacroAction.GO_TO), target


class OP2RedPolicy(Policy):
    """
    OP2: Defensive-only mine-layer.

    Behavior (for the configured `side`):
        - Places mines in front of its own end zone along a horizontal band.
        - If out of mines, GRAB_MINE from own pickups.
        - Never attacks the enemy flag.

    Research utility:
        - Evaluates RL agents' ability to attack into static defenses.
        - Tests robustness to minefields and suppression without opponent offense.
    """

    def __init__(self, side: str = "red", defense_band_width: int = 1):
        self.side = side
        # Vertical thickness (in rows) of the defensive band around the flag row.
        self.defense_band_width = max(1, defense_band_width)

    def select_action(
        self,
        obs: Any,
        agent: Agent,
        game_field: "GameField",
    ) -> Tuple[int, Optional[Tuple[int, int]]]:
        gm: GameManager = game_field.manager
        side = self.side

        # If somehow carrying a flag, go home.
        if agent.isCarryingFlag():
            return int(MacroAction.GO_HOME), None

        # Own zone column range.
        if side == "red":
            own_min_col, own_max_col = game_field.red_zone_col_range
        else:
            own_min_col, own_max_col = game_field.blue_zone_col_range

        # Center row of end zone (flag home).
        flag_x, flag_y = gm.get_team_zone_center(side)

        # Horizontal defensive band around the flag row (clamped to grid).
        defense_band = []
        half_w = self.defense_band_width // 2
        for dy in range(-half_w, half_w + 1):
            row = flag_y + dy
            if 0 <= row < game_field.row_count:
                for c in range(own_min_col, own_max_col + 1):
                    defense_band.append((c, row))

        # If we have mines: build/extend a mine line along this band.
        if agent.mine_charges > 0:
            # If already near a band cell, place a mine there.
            for cell_x, cell_y in defense_band:
                if math.hypot(agent.x - cell_x, agent.y - cell_y) <= 1.5:
                    return int(MacroAction.PLACE_MINE), (cell_x, cell_y)

            # Otherwise, walk to a random defensive cell in our band.
            if defense_band:
                target = random.choice(defense_band)
                return int(MacroAction.GO_TO), target
            else:
                # Rare fallback: no band constructed (e.g., tiny grid config).
                return int(MacroAction.GO_TO), (flag_x, flag_y)

        # No mines → grab more from own pickups.
        return int(MacroAction.GRAB_MINE), None


class OP3RedPolicy(Policy):
    """
    OP3: Mixed defender–attacker opponent.

    Assumes two agents on the configured `side` (agent_id 0 and 1):
        - Agent 0: Defender
            * Places one mine near own flag (if possible).
            * Then repeatedly GO_TO own flag to defend.
        - Agent 1: Attacker
            * Always GET_FLAG when not carrying.
            * GO_HOME when carrying the enemy flag.

    Research utility:
        - Provides a simple role-specialized baseline.
        - Useful for evaluating RL agents' offensive/defensive coordination
          and adaptation to mixed strategies (one defender, one attacker).
    """

    def __init__(self, side: str = "red", mine_radius_check: float = 1.5):
        self.side = side
        # Distance (in grid cells) used to decide "mine near flag".
        self.mine_radius_check = mine_radius_check

    def reset(self) -> None:
        """
        Reset any internal state between episodes.

        OP3 is currently stateless, but this method keeps the interface
        symmetric with potential stateful scripted policies.
        """
        # Nothing to reset yet.
        return None

    # --- Defender logic -------------------------------------------------
    def _defender_action(
        self,
        agent: Agent,
        gm: GameManager,
        game_field: "GameField",
    ) -> Tuple[int, Optional[Tuple[int, int]]]:
        side = self.side
        flag_x, flag_y = gm.get_team_zone_center(side)

        # If somehow carrying a flag, go home.
        if agent.isCarryingFlag():
            return int(MacroAction.GO_HOME), None

        # Check if a friendly mine is near flag (stateless).
        has_flag_mine = any(
            (m.owner_side == side)
            and (math.hypot(m.x - flag_x, m.y - flag_y) <= self.mine_radius_check)
            for m in game_field.mines
        )

        # If we still haven't placed that mine and we have charges:
        if (not has_flag_mine) and agent.mine_charges > 0:
            # If close to flag, place mine.
            if math.hypot(agent.x - flag_x, agent.y - flag_y) <= self.mine_radius_check:
                return int(MacroAction.PLACE_MINE), (flag_x, flag_y)

            # Otherwise, walk directly to flag area.
            return int(MacroAction.GO_TO), (flag_x, flag_y)

        # After we have a mine near the flag (or no charges), "defend"
        # by repeatedly going to the flag position.
        return int(MacroAction.GO_TO), (flag_x, flag_y)

    # --- Attacker logic -------------------------------------------------
    def _attacker_action(
        self,
        agent: Agent,
        gm: GameManager,
        game_field: "GameField",
    ) -> Tuple[int, Optional[Tuple[int, int]]]:
        # If already carrying the enemy flag, go home.
        if agent.isCarryingFlag():
            return int(MacroAction.GO_HOME), None

        # Otherwise, always try to get the flag.
        return int(MacroAction.GET_FLAG), None

    # --- Main dispatch --------------------------------------------------
    def select_action(
        self,
        obs: Any,
        agent: Agent,
        game_field: "GameField",
    ) -> Tuple[int, Optional[Tuple[int, int]]]:
        gm: GameManager = game_field.manager  # kept for symmetry / future tweaks

        if getattr(agent, "agent_id", 0) == 0:
            # Defender
            return self._defender_action(agent, gm, game_field)
        else:
            # Attacker (any non-zero agent_id)
            return self._attacker_action(agent, gm, game_field)
