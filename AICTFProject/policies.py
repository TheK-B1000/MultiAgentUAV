import math
import random
from typing import List, Optional
import pygame as pg

from agents import Agent
from game_manager import GameManager
from macro_actions import MacroAction


class Policy:
    def select_action(self, obs: List[float], agent: Agent, game_field: "GameField"):
        raise NotImplementedError


class HeuristicPolicy(Policy):
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


import math
import random
from typing import List, Optional

import pygame as pg  # kept in case you use it elsewhere

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

        # Center row of our end zone
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
    OP3 (paper): two-role opponent.
      - One attacker that goes for the enemy flag and returns home.
      - One defender that plays like OP2 (mine line in front of own end zone).

    No extra heuristics, no time-based logic, no hold-position action.
    """
    def __init__(self, side: str = "red"):
        self.side = side

    def select_action(self, obs: List[float], agent: Agent, game_field: "GameField"):
        gm: GameManager = game_field.manager
        side = self.side

        # If this agent is carrying a flag, always go home.
        if agent.isCarryingFlag():
            return int(MacroAction.GO_HOME), None

        # -------------- Agent 0: ATTACKER --------------
        if agent.agent_id == 0:
            # Simple: always go for the enemy flag.
            return int(MacroAction.GET_FLAG), None

        # -------------- Agent 1+: DEFENDER(S) ----------
        # Behavior: same as OP2, purely defensive.

        # Own zone column range
        if side == "red":
            own_min_col, own_max_col = game_field.red_zone_col_range
        else:
            own_min_col, own_max_col = game_field.blue_zone_col_range

        # Center row of our end zone
        flag_x, flag_y = gm.get_team_zone_center(side)

        # Horizontal band in front of our end zone
        defense_band = [
            (c, flag_y)
            for c in range(own_min_col, own_max_col + 1)
        ]

        # If we have mines: place them along this band
        if agent.mine_charges > 0:
            for cell in defense_band:
                if math.hypot(agent.x - cell[0], agent.y - cell[1]) <= 1.0:
                    return int(MacroAction.PLACE_MINE), cell

            target = random.choice(defense_band)
            return int(MacroAction.GO_TO), target

        # No mines left → go grab more
        return int(MacroAction.GRAB_MINE), None
