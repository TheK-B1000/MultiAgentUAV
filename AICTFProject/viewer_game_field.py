"""
viewer_game_field.py

Pygame-based visualization layer for the CTF environment.

ViewerGameField subclasses GameField and adds rendering utilities only.
Core dynamics + RL interface live in game_field.GameField.

Crash-proof goals:
  - Never assume GameField has helper methods like _enabled/_agent_float_pos.
  - Use float_pos when available for smooth rendering.
  - Fall back gracefully to cell position / x,y.
"""

from __future__ import annotations

from typing import Tuple, List, Any, Optional

import pygame as pg

from game_field import GameField, Mine, MinePickup
from agents import Agent, TEAM_ZONE_RADIUS_CELLS


class ViewerGameField(GameField):
    def __init__(self, grid: List[List[int]]):
        super().__init__(grid)

    # ------------------------------------------------------------------
    # Safe wrappers (do NOT rely on GameField private helpers)
    # ------------------------------------------------------------------
    def _enabled(self, agent: Any) -> bool:
        fn = getattr(agent, "isEnabled", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                return True
        # common fallbacks
        for attr in ("enabled", "is_enabled", "alive"):
            if hasattr(agent, attr):
                try:
                    return bool(getattr(agent, attr))
                except Exception:
                    pass
        return True

    def _agent_cell_pos(self, agent: Any) -> Tuple[int, int]:
        # Prefer explicit cell_pos
        v = getattr(agent, "cell_pos", None)
        if isinstance(v, (tuple, list)) and len(v) >= 2:
            try:
                return int(v[0]), int(v[1])
            except Exception:
                pass

        # Common x/y style
        for ax, ay in (("x", "y"), ("cell_x", "cell_y"), ("col", "row")):
            if hasattr(agent, ax) and hasattr(agent, ay):
                try:
                    return int(getattr(agent, ax)), int(getattr(agent, ay))
                except Exception:
                    pass

        # Fall back to float_pos
        fp = getattr(agent, "float_pos", None)
        if isinstance(fp, (tuple, list)) and len(fp) >= 2:
            try:
                return int(round(fp[0])), int(round(fp[1]))
            except Exception:
                pass

        return (0, 0)

    def _agent_float_pos(self, agent: Any) -> Tuple[float, float]:
        fp = getattr(agent, "float_pos", None)
        if isinstance(fp, (tuple, list)) and len(fp) >= 2:
            try:
                return float(fp[0]), float(fp[1])
            except Exception:
                pass
        cx, cy = self._agent_cell_pos(agent)
        return float(cx), float(cy)

    def _agent_is_tagged(self, agent: Any) -> bool:
        fn = getattr(agent, "isTagged", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                return False
        return bool(getattr(agent, "tagged", False))

    def _agent_is_carrying_flag(self, agent: Any) -> bool:
        fn = getattr(agent, "isCarryingFlag", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                return False
        return bool(getattr(agent, "is_carrying_flag", False) or getattr(agent, "carrying_flag", False))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def draw(self, surface: pg.Surface, board_rect: pg.Rect) -> None:
        rect_width, rect_height = board_rect.width, board_rect.height
        cell_w = rect_width / max(1, self.col_count)
        cell_h = rect_height / max(1, self.row_count)

        surface.fill((20, 22, 30), board_rect)
        self.draw_halves_and_center_line(surface, board_rect)

        # grid
        grid_color = (70, 70, 85)
        for row in range(self.row_count + 1):
            y = int(board_rect.top + row * cell_h)
            pg.draw.line(surface, grid_color, (board_rect.left, y), (board_rect.right, y), 1)

        for col in range(self.col_count + 1):
            x = int(board_rect.left + col * cell_w)
            pg.draw.line(surface, grid_color, (x, board_rect.top), (x, board_rect.bottom), 1)

        # ranges overlay
        if getattr(self, "debug_draw_ranges", False) or getattr(self, "debug_draw_mine_ranges", False):
            range_surface = pg.Surface((board_rect.width, board_rect.height), pg.SRCALPHA)

            if getattr(self, "debug_draw_ranges", False):
                sup_radius_px = float(getattr(self, "suppression_range_cells", 0)) * float(min(cell_w, cell_h))

                def draw_sup(agent: Agent, rgba: Tuple[int, int, int, int]) -> None:
                    fx, fy = self._agent_float_pos(agent)
                    cx = (fx + 0.5) * cell_w
                    cy = (fy + 0.5) * cell_h
                    pg.draw.circle(range_surface, rgba, (int(cx), int(cy)), int(sup_radius_px), width=2)

                for a in self.blue_agents:
                    if self._enabled(a):
                        draw_sup(a, (50, 130, 255, 190))
                for a in self.red_agents:
                    if self._enabled(a):
                        draw_sup(a, (255, 110, 70, 190))

            if getattr(self, "debug_draw_mine_ranges", False) and getattr(self, "mines", None):
                mine_radius_px = float(getattr(self, "mine_radius_cells", 0)) * float(min(cell_w, cell_h))
                for mine in self.mines:
                    cx = (float(mine.x) + 0.5) * cell_w
                    cy = (float(mine.y) + 0.5) * cell_h
                    rgba = (40, 170, 230, 170) if str(mine.owner_side).lower() == "blue" else (230, 120, 80, 170)
                    pg.draw.circle(range_surface, rgba, (int(cx), int(cy)), int(mine_radius_px), width=1)

            surface.blit(range_surface, board_rect.topleft)

        # flags
        blue_base = getattr(self.manager, "blue_flag_home", self.manager.blue_flag_position)
        red_base = getattr(self.manager, "red_flag_home", self.manager.red_flag_position)

        self.draw_flag(surface, board_rect, cell_w, cell_h, (int(blue_base[0]), int(blue_base[1])),
                       (90, 170, 250), bool(getattr(self.manager, "blue_flag_taken", False)))
        self.draw_flag(surface, board_rect, cell_w, cell_h, (int(red_base[0]), int(red_base[1])),
                       (250, 120, 70), bool(getattr(self.manager, "red_flag_taken", False)))

        # mine pickups
        for pickup in getattr(self, "mine_pickups", []):
            cx = board_rect.left + (float(pickup.x) + 0.5) * cell_w
            cy = board_rect.top + (float(pickup.y) + 0.5) * cell_h
            r_outer = int(0.30 * min(cell_w, cell_h))
            r_inner = int(0.16 * min(cell_w, cell_h))
            color = (80, 210, 255) if str(pickup.owner_side).lower() == "blue" else (255, 160, 110)
            pg.draw.circle(surface, (10, 10, 14), (int(cx), int(cy)), r_outer)
            pg.draw.circle(surface, color, (int(cx), int(cy)), r_outer, width=2)
            pg.draw.circle(surface, color, (int(cx), int(cy)), r_inner)

        # mines
        for mine in getattr(self, "mines", []):
            cx = board_rect.left + (float(mine.x) + 0.5) * cell_w
            cy = board_rect.top + (float(mine.y) + 0.5) * cell_h
            r = int(0.35 * min(cell_w, cell_h))
            color = (40, 170, 230) if str(mine.owner_side).lower() == "blue" else (230, 120, 80)
            pg.draw.circle(surface, color, (int(cx), int(cy)), r)
            pg.draw.circle(surface, (5, 5, 8), (int(cx), int(cy)), r, width=1)

        # agents
        def draw_agent(agent: Agent, body_rgb: Tuple[int, int, int], enemy_flag_rgb: Tuple[int, int, int]) -> None:
            fx, fy = self._agent_float_pos(agent)
            center_x = board_rect.left + (fx + 0.5) * cell_w
            center_y = board_rect.top + (fy + 0.5) * cell_h
            tri_size = 0.45 * min(cell_w, cell_h)

            # aim: first waypoint, else enemy flag
            if getattr(agent, "path", None):
                try:
                    tx, ty = agent.path[0]
                    target_x, target_y = float(tx), float(ty)
                except Exception:
                    target_x, target_y = fx, fy
            else:
                side = getattr(agent, "side", None)
                if side is None and callable(getattr(agent, "getSide", None)):
                    try:
                        side = agent.getSide()
                    except Exception:
                        side = "blue"
                try:
                    ex, ey = self.manager.get_enemy_flag_position(str(side).lower())
                    target_x, target_y = float(ex), float(ey)
                except Exception:
                    target_x, target_y = fx, fy

            dx = target_x - fx
            dy = target_y - fy
            mag = max((dx * dx + dy * dy) ** 0.5, 1e-6)
            ux, uy = dx / mag, dy / mag
            lx, ly = -uy, ux

            tip = (int(center_x + ux * tri_size), int(center_y + uy * tri_size))
            left = (int(center_x - ux * tri_size * 0.6 + lx * tri_size * 0.6),
                    int(center_y - uy * tri_size * 0.6 + ly * tri_size * 0.6))
            right = (int(center_x - ux * tri_size * 0.6 - lx * tri_size * 0.6),
                     int(center_y - uy * tri_size * 0.6 - ly * tri_size * 0.6))

            body_color = body_rgb if self._enabled(agent) else (50, 50, 55)
            pg.draw.polygon(surface, body_color, (tip, left, right))

            if self._agent_is_carrying_flag(agent):
                flag_size = int(tri_size * 0.5)
                flag_rect = pg.Rect(tip[0] - flag_size // 2, tip[1] - flag_size // 2, flag_size, flag_size)
                pg.draw.rect(surface, enemy_flag_rgb, flag_rect)

            if self._agent_is_tagged(agent):
                pg.draw.polygon(surface, (245, 245, 245), (tip, left, right), width=2)

        for a in self.blue_agents:
            draw_agent(a, (0, 180, 255), (250, 120, 70))
        for a in self.red_agents:
            draw_agent(a, (255, 120, 40), (90, 170, 250))

        # banner
        if getattr(self, "banner_queue", None):
            text, color, time_left = self.banner_queue[-1]
            fade = max(0.3, min(1.0, float(time_left) / 2.0))
            font = pg.font.SysFont(None, 48)
            faded_color = tuple(int(c * fade) for c in color)
            img = font.render(text, True, faded_color)
            surface.blit(img, (board_rect.centerx - img.get_width() // 2, board_rect.top + 12))

    def draw_halves_and_center_line(self, surface: pg.Surface, board_rect: pg.Rect) -> None:
        rect_width, rect_height = board_rect.width, board_rect.height
        cell_width = rect_width / max(1, self.col_count)

        def fill_cols(col_start: int, col_end: int, rgba: Tuple[int, int, int, int]) -> None:
            if col_start > col_end:
                return
            x0 = int(board_rect.left + col_start * cell_width)
            x1 = int(board_rect.left + (col_end + 1) * cell_width)
            band_w = max(1, x1 - x0)
            band = pg.Surface((band_w, rect_height), pg.SRCALPHA)
            band.fill(rgba)
            surface.blit(band, (x0, board_rect.top))

        total_cols = max(1, self.col_count)
        blue_min_col, blue_max_col = self.blue_zone_col_range
        red_min_col, red_max_col = self.red_zone_col_range

        mid_start = blue_max_col + 1
        mid_end = red_min_col - 1

        fill_cols(blue_min_col, blue_max_col, (15, 45, 120, 140))
        if mid_start <= mid_end:
            fill_cols(mid_start, mid_end, (40, 40, 55, 90))
        fill_cols(red_min_col, red_max_col, (120, 45, 15, 140))

        mid_col = total_cols // 2
        mid_x = int(board_rect.left + mid_col * cell_width)
        pg.draw.line(surface, (190, 190, 210), (mid_x, board_rect.top), (mid_x, board_rect.bottom), 2)

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
        gx, gy = int(grid_pos[0]), int(grid_pos[1])
        center_x = board_rect.left + (gx + 0.5) * cell_width
        center_y = board_rect.top + (gy + 0.5) * cell_height
        radius_px = int(float(TEAM_ZONE_RADIUS_CELLS) * float(min(cell_width, cell_height)))

        zone = pg.Surface((board_rect.width, board_rect.height), pg.SRCALPHA)
        local_center = (int(center_x - board_rect.left), int(center_y - board_rect.top))
        pg.draw.circle(zone, (*color, 40), local_center, radius_px, width=0)
        pg.draw.circle(zone, (*color, 110), local_center, radius_px, width=2)
        surface.blit(zone, board_rect.topleft)

        if not bool(is_taken):
            flag_size = int(0.5 * min(cell_width, cell_height))
            flag_rect = pg.Rect(
                int(center_x - flag_size / 2),
                int(center_y - flag_size / 2),
                flag_size,
                flag_size,
            )
            pg.draw.rect(surface, color, flag_rect)


__all__ = ["ViewerGameField"]
