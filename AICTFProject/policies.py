# policies.py
import math
import random
from typing import List

from agents import Agent
from game_manager import GameManager
from macro_actions import MacroAction


class Policy:
    def select_action(self, obs: List[float], agent: Agent, game_field: "GameField"):
        raise NotImplementedError


class HeuristicPolicy(Policy):
    """
    Optional: generic heuristic you can use for debugging / non-paper tests.
    This is *not* part of the IHMC opponents, but it's harmless to keep.
    """
    def select_action(self, obs: List[float], agent: Agent, game_field: "GameField"):
        dx_enemy_flag = obs[0]
        dy_enemy_flag = obs[1]
        dx_own_flag   = obs[2]
        dy_own_flag   = obs[3]
        is_carrying   = bool(round(obs[4]))
        own_flag_taken   = bool(round(obs[5]))
        enemy_flag_taken = bool(round(obs[6]))
        side_blue     = bool(round(obs[7]))
        ammo_norm     = obs[8] if len(obs) > 8 else 0.0
        is_miner      = bool(round(obs[9])) if len(obs) > 9 else False

        norm_pos = -dx_own_flag if side_blue else dx_own_flag
        norm_pos = max(0.0, min(1.0, norm_pos))
        dist_to_base = math.hypot(dx_own_flag, dy_own_flag)

        if is_carrying:
            return int(MacroAction.GO_HOME), None

        if is_miner:
            if ammo_norm <= 0.1 and norm_pos < 0.5:
                return int(MacroAction.GRAB_MINE), None
            if ammo_norm > 0.0 and dist_to_base < 0.4 and not own_flag_taken:
                if random.random() < 0.75:
                    return int(MacroAction.PLACE_MINE), None

        if own_flag_taken:
            return int(MacroAction.GO_TO), None

        if norm_pos >= 0.7:
            return int(MacroAction.GET_FLAG), None
        elif norm_pos <= 0.3:
            if is_miner and ammo_norm < 0.8:
                return int(MacroAction.GRAB_MINE), None
            else:
                return int(MacroAction.GO_TO), None
        else:
            r = random.random()
            if r < 0.6:
                return int(MacroAction.GET_FLAG), None
            else:
                return int(MacroAction.GO_TO), None


class OP1RedPolicy(Policy):
    """
    OP1 (paper): naive opponent that simply moves to the back of its own end zone.
    No mines, no offense.
    """
    def __init__(self, side: str = "red"):
        self.side = side

    def select_action(self, obs: List[float], agent: Agent, game_field: "GameField"):
        gm: GameManager = game_field.manager
        home_x, home_y = gm.get_team_zone_center(self.side)

        # "Back" of the endzone: just one cell deeper in your own zone.
        if self.side == "red":
            target = (min(game_field.col_count - 1, home_x + 1), home_y)
        else:
            target = (max(0, home_x - 1), home_y)

        # If for some reason it ever gets the flag, send it home.
        if agent.isCarryingFlag():
            return int(MacroAction.GO_HOME), None

        return int(MacroAction.GO_TO), target


class OP2RedPolicy(Policy):
    """
    OP2 (paper): defensive-only opponent.
    It places mines in front of its own end zone and does not attack.
    """
    def __init__(self, side: str = "red"):
        self.side = side

    def select_action(self, obs: List[float], agent: Agent, game_field: "GameField"):
        gm: GameManager = game_field.manager
        side = self.side

        # If somehow carrying a flag, still go home.
        if agent.isCarryingFlag():
            return int(MacroAction.GO_HOME), None

        # Own zone column range
        if side == "red":
            own_min_col, own_max_col = game_field.red_zone_col_range
        else:
            own_min_col, own_max_col = game_field.blue_zone_col_range

        # Center row of our end zone (flag home)
        flag_x, flag_y = gm.get_team_zone_center(side)

        # Horizontal band in front of our end zone
        defense_band = [
            (c, flag_y)
            for c in range(own_min_col, own_max_col + 1)
        ]

        # If we have mines: try to build a mine line along this band
        if agent.mine_charges > 0:
            # If already near a band cell, place a mine there
            for cell in defense_band:
                if math.hypot(agent.x - cell[0], agent.y - cell[1]) <= 1.0:
                    return int(MacroAction.PLACE_MINE), cell

            # Otherwise, walk to a random defensive cell in our band
            target = random.choice(defense_band)
            return int(MacroAction.GO_TO), target

        # No mines left → go grab more from our own pickups
        return int(MacroAction.GRAB_MINE), None


class OP3RedPolicy(Policy):
    """
    OP3 (paper-style, intentionally weak / scripted):

    - Two agents (assume agent_id 0 and 1):
        * Agent 0: Defender
            - Tries to place ONE mine near its own flag.
            - Then just defends near the flag.
        * Agent 1: Attacker
            - Always goes for the enemy flag.
    - No clever intercepts, no long-term memory. Behavior is purely based
      on current environment state, like in the paper.
    """

    def __init__(self, side: str = "red"):
        self.side = side

    # --- Defender logic -------------------------------------------------
    def _defender_action(self, agent: Agent, gm: GameManager, game_field: "GameField"):
        side = self.side
        flag_x, flag_y = gm.get_team_zone_center(side)

        # If somehow carrying a flag, just go home.
        if agent.isCarryingFlag():
            return int(MacroAction.GO_HOME), None

        # ---- STATLESS "has placed mine" check ----
        # Consider our key mine "placed" if there is ANY friendly mine
        # within ~1.5 cells of our flag.
        has_flag_mine = any(
            (m.owner_side == side) and
            (math.hypot(m.x - flag_x, m.y - flag_y) <= 1.5)
            for m in game_field.mines
        )

        # If we still haven't placed that mine and we have charges:
        if (not has_flag_mine) and agent.mine_charges > 0:
            # If close to flag, drop a mine there
            if math.hypot(agent.x - flag_x, agent.y - flag_y) <= 1.5:
                return int(MacroAction.PLACE_MINE), (flag_x, flag_y)

            # Otherwise walk directly to flag area
            return int(MacroAction.GO_TO), (flag_x, flag_y)

        # After we have a mine near the flag (or no charges), just defend.
        # This is still very dumb: just loiters around midline.
        return int(MacroAction.DEFEND_ZONE), (flag_x, flag_y)

    # --- Attacker logic -------------------------------------------------
    def _attacker_action(self, agent: Agent, gm: GameManager, game_field: "GameField"):
        # If already carrying enemy flag, go home.
        if agent.isCarryingFlag():
            return int(MacroAction.GO_HOME), None

        # Otherwise, always try to get the flag.
        return int(MacroAction.GET_FLAG), None

    # --- Main dispatch --------------------------------------------------
    def select_action(self, obs: List[float], agent: Agent, game_field: "GameField"):
        gm: GameManager = game_field.manager  # kept for symmetry / future tweaks

        # Hard-assign roles by agent_id:
        #   agent_id == 0 → defender
        #   agent_id != 0 → attacker
        if getattr(agent, "agent_id", 0) == 0:
            return self._defender_action(agent, gm, game_field)
        else:
            return self._attacker_action(agent, gm, game_field)
