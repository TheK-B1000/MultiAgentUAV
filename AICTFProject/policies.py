# policies.py
import math
import random
from typing import List, Optional, Tuple
import pygame as pg

from agents import Agent
from game_manager import GameManager
from macro_actions import MacroAction


class Policy:
    """Base interface for all policies."""
    def select_action(
        self,
        obs: List[float],
        agent: Agent,
        game_field: "GameField",
    ) -> Tuple[int, Optional[Tuple[float, float]]]:
        raise NotImplementedError


class HeuristicPolicy(Policy):
    """Simple but effective baseline for blue team."""
    def select_action(
        self,
        obs: List[float],
        agent: Agent,
        game_field: "GameField",
    ) -> Tuple[int, Optional[Tuple[float, float]]]:
        (
            dx_enemy_flag, dy_enemy_flag,
            dx_own_flag, dy_own_flag,
            is_carrying, own_flag_taken, enemy_flag_taken, side_blue,
            ammo_norm, is_miner,
            dx_mine, dy_mine,
        ) = obs[:12]

        is_carrying = bool(round(is_carrying))
        own_flag_taken = bool(round(own_flag_taken))
        is_miner = bool(round(is_miner))

        # Normalized position in own half (0.0 = at home, 1.0 = mid-line+)
        norm_pos = -dx_own_flag if side_blue else dx_own_flag
        norm_pos = max(0.0, min(1.0, norm_pos))

        if is_carrying:
            return int(MacroAction.GO_HOME), None

        if is_miner:
            if ammo_norm < 0.2 and norm_pos < 0.5:
                return int(MacroAction.GRAB_MINE), None
            if ammo_norm > 0.0 and norm_pos < 0.4 and not own_flag_taken:
                if random.random() < 0.8:
                    return int(MacroAction.PLACE_MINE), None

        if own_flag_taken:
            return int(MacroAction.GO_TO), None

        if norm_pos >= 0.7:
            return int(MacroAction.GET_FLAG), None
        elif norm_pos <= 0.3:
            if is_miner and ammo_norm < 0.8:
                return int(MacroAction.GRAB_MINE), None
        else:
            if random.random() < 0.65:
                return int(MacroAction.GET_FLAG), None

        return int(MacroAction.GO_TO), None


class OP1RedPolicy(Policy):
    """Simple defensive red bot — guard base."""
    def __init__(self, side: str = "red"):
        self.side = side

    def select_action(
        self,
        obs: List[float],
        agent: Agent,
        game_field: "GameField",
    ) -> Tuple[int, Optional[Tuple[float, float]]]:
        gm: GameManager = game_field.manager
        hx, hy = gm.get_team_zone_center(self.side)
        hx += 1.5 if self.side == "red" else -1.5
        target = (hx, hy + 0.5)

        if agent.isCarryingFlag():
            return int(MacroAction.GO_HOME), None

        return int(MacroAction.GO_TO), target


class OP2RedPolicy(Policy):
    """Mine-layer defense bot."""
    def __init__(self, side: str = "red"):
        self.side = side

    def select_action(
        self,
        obs: List[float],
        agent: Agent,
        game_field: "GameField",
    ) -> Tuple[int, Optional[Tuple[float, float]]]:
        if agent.isCarryingFlag():
            return int(MacroAction.GO_HOME), None

        min_col, max_col = (
            game_field.red_zone_col_range if self.side == "red"
            else game_field.blue_zone_col_range
        )
        mid_row = game_field.row_count // 2
        defense_line = [(c + 0.5, mid_row + 0.5) for c in range(min_col, max_col + 1)]

        if agent.mine_charges > 0:
            for spot in defense_line:
                if math.hypot(agent._float_x - spot[0], agent._float_y - spot[1]) < 1.2:
                    return int(MacroAction.PLACE_MINE), spot

        target = random.choice(defense_line)
        return int(MacroAction.GO_TO), target


class OP3RedPolicy(Policy):
    """Advanced coordinated red opponent — miner + interceptor."""
    def __init__(self, side: str = "red"):
        self.side = side

    def select_action(
        self,
        obs: List[float],
        agent: Agent,
        game_field: "GameField",
    ) -> Tuple[int, Optional[Tuple[float, float]]]:
        gm: GameManager = game_field.manager
        side = self.side

        enemy_team = game_field.blue_agents if side == "red" else game_field.red_agents
        our_team = game_field.red_agents if side == "red" else game_field.blue_agents

        enemy_carrier = next((a for a in enemy_team if a.isCarryingFlag() and a.isEnabled()), None)
        our_carrier = next((a for a in our_team if a.isCarryingFlag() and a.isEnabled()), None)

        def dist_to(a: Agent, pos: Tuple[float, float]) -> float:
            return math.hypot(a._float_x - pos[0], a._float_y - pos[1])

        # Miner (agent_id == 0)
        if agent.agent_id == 0:
            if agent.isCarryingFlag():
                return int(MacroAction.GO_HOME), None

            if enemy_carrier and dist_to(agent, enemy_carrier.float_pos) < 10.0:
                return int(MacroAction.INTERCEPT_CARRIER), None

            return int(MacroAction.GET_FLAG), None

        # Interceptor / defender (agent_id == 1)
        else:
            if agent.isCarryingFlag():
                return int(MacroAction.GO_HOME), None

            # Protect our carrier
            if our_carrier:
                closest_threat = min(
                    (e for e in enemy_team if e.isEnabled()),
                    key=lambda e: dist_to(e, our_carrier.float_pos),
                    default=None
                )
                if closest_threat and dist_to(closest_threat, our_carrier.float_pos) < 8.0:
                    return int(MacroAction.INTERCEPT_CARRIER), None

            # Mine chokepoints
            if agent.mine_charges > 0:
                chokepoints = [
                    (game_field.col_count * 0.5, game_field.row_count * 0.25),
                    (game_field.col_count * 0.5, game_field.row_count * 0.75),
                    (game_field.col_count * 0.33, game_field.row_count * 0.5),
                    (game_field.col_count * 0.66, game_field.row_count * 0.5),
                ]
                for spot in chokepoints:
                    if dist_to(agent, spot) < 2.0:
                        return int(MacroAction.PLACE_MINE), spot

                return int(MacroAction.GO_TO), random.choice(chokepoints)

            # Suppress or intercept enemy carrier
            if enemy_carrier:
                d = dist_to(agent, enemy_carrier.float_pos)
                if d < 8.0:
                    return int(MacroAction.SUPPRESS_CARRIER), None
                if d < 16.0:
                    return int(MacroAction.INTERCEPT_CARRIER), None

            # Default: defend mid-line
            mid_x = game_field.col_count * 0.5
            target_y = agent._float_y if abs(agent._float_y - game_field.row_count * 0.5) < 5 else game_field.row_count * 0.5
            return int(MacroAction.GO_TO), (mid_x, target_y)