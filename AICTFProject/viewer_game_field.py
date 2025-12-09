"""
viewer_game_field.py

Pygame-based visualization layer for the CTF environment.

This module provides ViewerGameField, a subclass of GameField that adds
all rendering / UI drawing using pygame. The core dynamics and RL interface
live in game_field.GameField.

Use this in your viewer (ctfviewer) so that training code never imports pygame.
"""

from typing import Tuple

import pygame as pg

from game_field import GameField, Mine, MinePickup, ARENA_WIDTH_M, ARENA_HEIGHT_M
from agents import Agent, TEAM_ZONE_RADIUS_CELLS


class ViewerGameField(GameField):
    """
    Extension of GameField with pygame rendering utilities.
    """

    def draw(self, surface: pg.Surface, board_rect: pg.Rect) -> None:
        """
        Render the current game state into a Pygame surface.

        This is for visualization / debugging only and is not required
        for RL training.
        """
        rect_width, rect_height = board_rect.width, board_rect.height
        cell_width = rect_width / max(1, self.col_count)
        cell_height = rect_height / max(1, self.row_count)

        surface.fill((20, 22, 30), board_rect)
        self.draw_halves_and_center_line(surface, board_rect)

        grid_color = (70, 70, 85)
        for row in range(self.row_count + 1):
            y = int(board_rect.top + row * cell_height)
            pg.draw.line(surface, grid_color, (board_rect.left, y), (board_rect.right, y), 1)

        for col in range(self.col_count + 1):
            x = int(board_rect.left + col * cell_width)
            pg.draw.line(surface, grid_color, (x, board_rect.top), (x, board_rect.bottom), 1)

        # Debug ranges
        if self.debug_draw_ranges or self.debug_draw_mine_ranges:
            range_surface = pg.Surface((board_rect.width, board_rect.height), pg.SRCALPHA)

            if self.debug_draw_ranges:
                sup_radius_px = self.suppression_range_cells * min(cell_width, cell_height)

                def draw_sup_range(agent: Agent, rgba: Tuple[int, int, int, int]) -> None:
                    cx = (agent.x + 0.5) * cell_width
                    cy = (agent.y + 0.5) * cell_height
                    local_center = (int(cx), int(cy))
                    pg.draw.circle(range_surface, rgba, local_center, int(sup_radius_px), width=2)

                for a in self.blue_agents:
                    if a.isEnabled():
                        draw_sup_range(a, (50, 130, 255, 190))
                for a in self.red_agents:
                    if a.isEnabled():
                        draw_sup_range(a, (255, 110, 70, 190))

            if self.debug_draw_mine_ranges and self.mines:
                mine_radius_px = self.mine_radius_cells * min(cell_width, cell_height)
                for mine in self.mines:
                    cx = (mine.x + 0.5) * cell_width
                    cy = (mine.y + 0.5) * cell_height
                    local_center = (int(cx), int(cy))
                    rgba = (
                        (40, 170, 230, 170)
                        if mine.owner_side == "blue"
                        else (230, 120, 80, 170)
                    )
                    pg.draw.circle(range_surface, rgba, local_center, int(mine_radius_px), width=1)

            surface.blit(range_surface, board_rect.topleft)

        # Flags
        blue_base = getattr(self.manager, "blue_flag_home", self.manager.blue_flag_position)
        red_base = getattr(self.manager, "red_flag_home", self.manager.red_flag_position)

        self.draw_flag(
            surface,
            board_rect,
            cell_width,
            cell_height,
            blue_base,
            (90, 170, 250),
            self.manager.blue_flag_taken,
        )
        self.draw_flag(
            surface,
            board_rect,
            cell_width,
            cell_height,
            red_base,
            (250, 120, 70),
            self.manager.red_flag_taken,
        )

        # Mine pickups
        def draw_mine_pickup(pickup: MinePickup) -> None:
            cx = board_rect.left + (pickup.x + 0.5) * cell_width
            cy = board_rect.top + (pickup.y + 0.5) * cell_height
            r_outer = int(0.3 * min(cell_width, cell_height))
            r_inner = int(0.16 * min(cell_width, cell_height))
            color = (80, 210, 255) if pickup.owner_side == "blue" else (255, 160, 110)
            pg.draw.circle(surface, (10, 10, 14), (int(cx), int(cy)), r_outer)
            pg.draw.circle(surface, color, (int(cx), int(cy)), r_outer, width=2)
            pg.draw.circle(surface, color, (int(cx), int(cy)), r_inner)

        for pickup in self.mine_pickups:
            draw_mine_pickup(pickup)

        # Armed mines
        def draw_mine(mine: Mine) -> None:
            cx = board_rect.left + (mine.x + 0.5) * cell_width
            cy = board_rect.top + (mine.y + 0.5) * cell_height
            r = int(0.35 * min(cell_width, cell_height))
            color = (40, 170, 230) if mine.owner_side == "blue" else (230, 120, 80)
            pg.draw.circle(surface, color, (int(cx), int(cy)), r)
            pg.draw.circle(surface, (5, 5, 8), (int(cx), int(cy)), r, width=1)

        for mine in self.mines:
            draw_mine(mine)

        # Agents
        def draw_agent(agent: Agent, body_rgb, enemy_flag_rgb):
            center_x = board_rect.left + (agent.x + 0.5) * cell_width
            center_y = board_rect.top + (agent.y + 0.5) * cell_height
            tri_size = 0.45 * min(cell_width, cell_height)

            target_x, target_y = self.manager.get_enemy_flag_position(agent.getSide())
            if agent.path:
                target_x, target_y = agent.path[0]

            to_target_x = target_x - agent.x
            to_target_y = target_y - agent.y
            magnitude = max(
                (to_target_x * to_target_x + to_target_y * to_target_y) ** 0.5, 1e-6
            )
            unit_x, unit_y = to_target_x / magnitude, to_target_y / magnitude
            left_x, left_y = -unit_y, unit_x

            tip = (
                int(center_x + unit_x * tri_size),
                int(center_y + unit_y * tri_size),
            )
            left = (
                int(center_x - unit_x * tri_size * 0.6 + left_x * tri_size * 0.6),
                int(center_y - unit_y * tri_size * 0.6 + left_y * tri_size * 0.6),
            )
            right = (
                int(center_x - unit_x * tri_size * 0.6 - left_x * tri_size * 0.6),
                int(center_y - unit_y * tri_size * 0.6 - left_y * tri_size * 0.6),
            )

            body_color = body_rgb if agent.isEnabled() else (50, 50, 55)
            pg.draw.polygon(surface, body_color, (tip, left, right))

            if agent.isCarryingFlag():
                flag_size = int(tri_size * 0.5)
                flag_rect = pg.Rect(
                    tip[0] - flag_size // 2,
                    tip[1] - flag_size // 2,
                    flag_size,
                    flag_size,
                )
                pg.draw.rect(surface, enemy_flag_rgb, flag_rect)

            if agent.isTagged():
                pg.draw.polygon(surface, (245, 245, 245), (tip, left, right), width=2)

        for agent in self.blue_agents:
            draw_agent(agent, (0, 180, 255), (250, 120, 70))
        for agent in self.red_agents:
            draw_agent(agent, (255, 120, 40), (90, 170, 250))

        # Banner
        if self.banner_queue:
            text, color, time_left = self.banner_queue[-1]
            fade_factor = max(0.3, min(1.0, time_left / 2.0))
            font = pg.font.SysFont(None, 48)
            faded_color = tuple(int(channel * fade_factor) for channel in color)
            img = font.render(text, True, faded_color)
            surface.blit(
                img,
                (
                    board_rect.centerx - img.get_width() // 2,
                    board_rect.top + 12,
                ),
            )

    def draw_halves_and_center_line(self, surface: pg.Surface, board_rect: pg.Rect) -> None:
        rect_width, rect_height = board_rect.width, board_rect.height
        cell_width = rect_width / max(1, self.col_count)

        def fill_cols(col_start: int, col_end: int, rgba: Tuple[int, int, int, int]) -> None:
            if col_start > col_end:
                return
            x0 = int(board_rect.left + col_start * cell_width)
            x1 = int(board_rect.left + (col_end + 1) * cell_width)
            band_width = max(1, x1 - x0)
            band_surface = pg.Surface((band_width, rect_height), pg.SRCALPHA)
            band_surface.fill(rgba)
            surface.blit(band_surface, (x0, board_rect.top))

        total_cols = max(1, self.col_count)
        blue_min_col, blue_max_col = self.blue_zone_col_range
        red_min_col, red_max_col = self.red_zone_col_range

        mid_start = blue_max_col + 1
        mid_end = red_min_col - 1

        fill_cols(blue_min_col, blue_max_col, (15, 45, 120, 140))

        if mid_start <= mid_end:
            fill_cols(mid_start, mid_end, (40, 40, 55, 90))

        fill_cols(red_min_col, red_max_col, (120, 45, 15, 140))

        mid_col_index = total_cols // 2
        mid_x = int(board_rect.left + mid_col_index * cell_width)
        pg.draw.line(
            surface,
            (190, 190, 210),
            (mid_x, board_rect.top),
            (mid_x, board_rect.bottom),
            2,
        )

    def draw_flag(
        self,
        surface: pg.Surface,
        board_rect: pg.Rect,
        cell_width: float,
        cell_height: float,
        grid_pos: Tuple[int, int],
        color: Tuple[int, int, int],
        is_taken: bool,
    ) -> None:
        grid_x, grid_y = grid_pos
        center_x = board_rect.left + (grid_x + 0.5) * cell_width
        center_y = board_rect.top + (grid_y + 0.5) * cell_height
        radius_px = int(TEAM_ZONE_RADIUS_CELLS * min(cell_width, cell_height))

        zone_surface = pg.Surface((board_rect.width, board_rect.height), pg.SRCALPHA)
        local_center = (
            int(center_x - board_rect.left),
            int(center_y - board_rect.top),
        )

        pg.draw.circle(zone_surface, (*color, 40), local_center, radius_px, width=0)
        pg.draw.circle(zone_surface, (*color, 110), local_center, radius_px, width=2)
        surface.blit(zone_surface, board_rect.topleft)

        if not is_taken:
            flag_size = int(0.5 * min(cell_width, cell_height))
            flag_rect = pg.Rect(
                int(center_x - flag_size / 2),
                int(center_y - flag_size / 2),
                flag_size,
                flag_size,
            )
            pg.draw.rect(surface, color, flag_rect)


__all__ = ["ViewerGameField"]
