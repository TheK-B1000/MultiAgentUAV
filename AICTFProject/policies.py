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


class OP1RedPolicy(Policy):
    def __init__(self, side: str = "red"):
        self.side = side

    def select_action(self, obs: List[float], agent: Agent, game_field: "GameField"):
        gm: GameManager = game_field.manager
        home_x, home_y = gm.get_team_zone_center(self.side)

        if self.side == "red":
            target = (min(game_field.col_count - 1, home_x + 1), home_y)
        else:
            target = (max(0, home_x - 1), home_y)

        if agent.isCarryingFlag():
            return int(MacroAction.GO_HOME), None

        return int(MacroAction.GO_TO), target


class OP2RedPolicy(Policy):
    def __init__(self, side: str = "red"):
        self.side = side

    def select_action(self, obs: List[float], agent: Agent, game_field: "GameField"):
        gm: GameManager = game_field.manager
        side = self.side

        if agent.isCarryingFlag():
            return int(MacroAction.GO_HOME), None

        if side == "red":
            own_min_col, own_max_col = game_field.red_zone_col_range
        else:
            own_min_col, own_max_col = game_field.blue_zone_col_range

        flag_x, flag_y = gm.get_team_zone_center(side)

        defense_band = [
            (c, flag_y)
            for c in range(own_min_col, own_max_col + 1)
        ]

        if agent.mine_charges > 0:
            for cell in defense_band:
                if math.hypot(agent.x - cell[0], agent.y - cell[1]) <= 1.0:
                    return int(MacroAction.PLACE_MINE), cell

            target = random.choice(defense_band)
            return int(MacroAction.GO_TO), target

        return int(MacroAction.GRAB_MINE), None


class OP3RedPolicy(Policy):
    def __init__(self, side: str = "red"):
        self.side = side

    def select_action(self, obs: List[float], agent: Agent, game_field: "GameField"):
        gm: GameManager = game_field.manager
        side = self.side

        enemy_flag_pos = gm.get_enemy_flag_position(side)
        my_flag_taken      = gm.red_flag_taken if side == "red" else gm.blue_flag_taken
        enemy_flag_taken   = gm.blue_flag_taken if side == "red" else gm.red_flag_taken

        enemy_team = game_field.blue_agents if side == "red" else game_field.red_agents
        our_team   = game_field.red_agents if side == "red" else game_field.blue_agents

        enemy_carrier = next((a for a in enemy_team if a.isCarryingFlag() and a.isEnabled()), None)
        our_carrier   = next((a for a in our_team   if a.isCarryingFlag() and a.isEnabled()), None) if enemy_flag_taken else None

        def dist(a1: Agent, a2: Optional[Agent]) -> float:
            if a2 is None:
                return 999.0
            return math.hypot(a1.x - a2.x, a1.y - a2.y)

        if agent.agent_id == 0:
            if agent.isCarryingFlag():
                return int(MacroAction.GO_HOME), None

            if my_flag_taken and enemy_carrier is not None and dist(agent, enemy_carrier) < 10.0:
                return int(MacroAction.INTERCEPT_CARRIER), None

            return int(MacroAction.GET_FLAG), None

        else:
            if agent.isCarryingFlag():
                return int(MacroAction.GO_HOME), None

            if our_carrier is not None:
                threat = min(
                    (e for e in enemy_team if e.isEnabled()),
                    key=lambda e: dist(e, our_carrier),
                    default=None,
                )
                if threat and dist(threat, our_carrier) < 8.0:
                    return int(MacroAction.INTERCEPT_CARRIER), None

            if agent.mine_charges > 0:
                chokepoints = [
                    (game_field.col_count // 2, game_field.row_count // 4),
                    (game_field.col_count // 2, 3 * game_field.row_count // 4),
                    (game_field.col_count // 3, game_field.row_count // 2),
                    (2 * game_field.col_count // 3, game_field.row_count // 2),
                ]
                agent_v = pg.Vector2(agent.x, agent.y)
                for spot in chokepoints:
                    if (agent_v - pg.Vector2(spot)).length() < 2.0:
                        return int(MacroAction.PLACE_MINE), spot

                return int(MacroAction.GO_TO), random.choice(chokepoints)

            if enemy_carrier is not None:
                d = dist(agent, enemy_carrier)
                if d < 10.0:
                    return int(MacroAction.SUPPRESS_CARRIER), None
                elif d < 18.0:
                    return int(MacroAction.INTERCEPT_CARRIER), None

            return int(MacroAction.DEFEND_ZONE), None
