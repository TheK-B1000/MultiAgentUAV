"""
viewer_game_field.py

Pygame visualization layer for the CTF environment.

Key fix for jitter:
  - Store per-agent previous float positions BEFORE each fixed sim update
  - Render interpolated positions using alpha: lerp(prev, curr, alpha)

Extra robustness (NEW):
  - Handles multiple sim substeps per render frame correctly (no "prev = curr" drift)
  - Handles respawns/teleports (snap instead of lerp across map)
  - Clamps alpha and supports missing buffers safely
  - Optional: clamp dt in viewer layer (prevents large dt spikes from exploding motion)
"""

from __future__ import annotations

from typing import Tuple, List, Any, Optional, Dict

import pygame as pg

from game_field import GameField
from agents import TEAM_ZONE_RADIUS_CELLS


FloatPos = Tuple[float, float]
Cell = Tuple[int, int]


class ViewerGameField(GameField):
    def __init__(self, grid: List[List[int]]):
        # Render buffers keyed by python id(agent)
        self._render_prev_fp: Dict[int, FloatPos] = {}
        self._render_curr_fp: Dict[int, FloatPos] = {}

        # Track whether we've ever stepped at least once (to avoid bad lerps on first frame)
        self._has_stepped_once: bool = False

        super().__init__(grid)

    # ------------------------------------------------------------------
    # Dimensions: derive from actual grid
    # ------------------------------------------------------------------
    @property
    def _rows(self) -> int:
        try:
            return int(len(self.grid))
        except Exception:
            return int(getattr(self, "row_count", 0) or 0)

    @property
    def _cols(self) -> int:
        try:
            return int(len(self.grid[0])) if self.grid and len(self.grid) > 0 else 0
        except Exception:
            return int(getattr(self, "col_count", 0) or 0)

    def _safe_row_count(self) -> int:
        r = self._rows
        return r if r > 0 else int(getattr(self, "row_count", 20) or 20)

    def _safe_col_count(self) -> int:
        c = self._cols
        return c if c > 0 else int(getattr(self, "col_count", 20) or 20)

    # ------------------------------------------------------------------
    # Manager access
    # ------------------------------------------------------------------
    @property
    def _gm(self):
        if hasattr(self, "manager"):
            return getattr(self, "manager")
        fn = getattr(self, "getGameManager", None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    # Safe wrappers
    # ------------------------------------------------------------------
    def _enabled(self, agent: Any) -> bool:
        if agent is None:
            return False
        fn = getattr(agent, "isEnabled", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                return True
        for attr in ("enabled", "is_enabled", "alive"):
            if hasattr(agent, attr):
                try:
                    return bool(getattr(agent, attr))
                except Exception:
                    pass
        return True

    def _agent_cell_pos(self, agent: Any) -> Cell:
        if agent is None:
            return (0, 0)

        v = getattr(agent, "cell_pos", None)
        if isinstance(v, (tuple, list)) and len(v) >= 2:
            try:
                return int(v[0]), int(v[1])
            except Exception:
                pass

        for ax, ay in (("x", "y"), ("cell_x", "cell_y"), ("col", "row")):
            if hasattr(agent, ax) and hasattr(agent, ay):
                try:
                    return int(getattr(agent, ax)), int(getattr(agent, ay))
                except Exception:
                    pass

        fp = getattr(agent, "float_pos", None)
        if isinstance(fp, (tuple, list)) and len(fp) >= 2:
            try:
                return int(round(fp[0])), int(round(fp[1]))
            except Exception:
                pass

        return (0, 0)

    def _agent_float_pos_raw(self, agent: Any) -> FloatPos:
        """
        Best-effort float position. If float_pos is missing, fall back to cell.
        """
        if agent is None:
            return (0.0, 0.0)
        fp = getattr(agent, "float_pos", None)
        if isinstance(fp, (tuple, list)) and len(fp) >= 2:
            try:
                return float(fp[0]), float(fp[1])
            except Exception:
                pass
        cx, cy = self._agent_cell_pos(agent)
        return float(cx), float(cy)

    def _agent_is_tagged(self, agent: Any) -> bool:
        if agent is None:
            return False
        fn = getattr(agent, "isTagged", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                return False
        return bool(getattr(agent, "tagged", False))

    def _agent_is_carrying_flag(self, agent: Any) -> bool:
        if agent is None:
            return False
        fn = getattr(agent, "isCarryingFlag", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                return False
        return bool(getattr(agent, "is_carrying_flag", False) or getattr(agent, "carrying_flag", False))

    def _agent_side(self, agent: Any) -> str:
        if agent is None:
            return "blue"
        side = getattr(agent, "side", None)
        if side is None and callable(getattr(agent, "getSide", None)):
            try:
                side = agent.getSide()
            except Exception:
                side = "blue"
        side = str(side or "blue").lower()
        return "red" if side == "red" else "blue"

    # ------------------------------------------------------------------
    # Zones
    # ------------------------------------------------------------------
    def _zone_ranges(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        cols = self._safe_col_count()
        if hasattr(self, "blue_zone_col_range") and hasattr(self, "red_zone_col_range"):
            try:
                bmin, bmax = self.blue_zone_col_range
                rmin, rmax = self.red_zone_col_range
                return (int(bmin), int(bmax)), (int(rmin), int(rmax))
            except Exception:
                pass

        mid = cols // 2
        blue = (0, max(0, mid - 1))
        red = (mid, max(mid, cols - 1))
        return blue, red

    # ------------------------------------------------------------------
    # Render interpolation core
    # ------------------------------------------------------------------
    def update(self, delta_time: float) -> None:
        """
        Override: capture prev/curr for interpolation.

        IMPORTANT nuance for fixed-step viewers:
        When the viewer does multiple substeps per render frame, `update()` is called
        multiple times before a single `draw(alpha)`.

        To make interpolation correct:
          - On each sim step, set prev = curr (last known), then step, then curr = new.
          - That means after N substeps, prev/curr represent the *last* two sim states.
        """
        # Optional defensive clamp to avoid giant dt spikes causing huge motion jumps
        try:
            dt = float(delta_time)
        except Exception:
            dt = 0.0
        dt = max(0.0, min(dt, 0.05))  # tweak if you want (0.033 for stricter)

        agents = list(getattr(self, "blue_agents", [])) + list(getattr(self, "red_agents", []))

        # Ensure curr exists even before the first step
        if not self._has_stepped_once:
            for a in agents:
                aid = id(a)
                fp = self._agent_float_pos_raw(a)
                self._render_prev_fp[aid] = fp
                self._render_curr_fp[aid] = fp

        # prev becomes last curr (so interpolation is between last two sim states)
        for a in agents:
            aid = id(a)
            curr = self._render_curr_fp.get(aid, self._agent_float_pos_raw(a))
            self._render_prev_fp[aid] = curr

        # step sim
        super().update(dt)
        self._has_stepped_once = True

        # snapshot curr after sim step
        for a in agents:
            aid = id(a)
            self._render_curr_fp[aid] = self._agent_float_pos_raw(a)

        # Clean up stale ids (agents respawned/recreated)
        live_ids = {id(a) for a in agents}
        for stale in list(self._render_curr_fp.keys()):
            if stale not in live_ids:
                self._render_curr_fp.pop(stale, None)
                self._render_prev_fp.pop(stale, None)

    def _lerp(self, a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    def _agent_float_pos_interp(self, agent: Any, alpha: float) -> FloatPos:
        if agent is None:
            return (0.0, 0.0)

        aid = id(agent)

        curr = self._render_curr_fp.get(aid, None)
        prev = self._render_prev_fp.get(aid, None)

        if curr is None and prev is None:
            return self._agent_float_pos_raw(agent)
        if curr is None:
            return prev
        if prev is None:
            return curr

        # Clamp alpha defensively
        t = max(0.0, min(1.0, float(alpha)))

        # Teleport/respawn snap: don't lerp across the map
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        if (dx * dx + dy * dy) > (2.0 * 2.0):  # threshold in cells
            self._render_prev_fp[aid] = curr
            return curr

        return (self._lerp(prev[0], curr[0], t), self._lerp(prev[1], curr[1], t))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def reset_default(self) -> None:
        super().reset_default()
        self._render_prev_fp.clear()
        self._render_curr_fp.clear()
        self._has_stepped_once = False

    def draw(self, surface: pg.Surface, board_rect: pg.Rect, alpha: float = 1.0) -> None:
        rows = self._safe_row_count()
        cols = self._safe_col_count()

        rect_width, rect_height = board_rect.width, board_rect.height
        cell_w = rect_width / max(1, cols)
        cell_h = rect_height / max(1, rows)

        # background
        surface.fill((20, 22, 30), board_rect)
        self.draw_halves_and_center_line(surface, board_rect)

        # debug range overlay (use interpolated positions!)
        if getattr(self, "debug_draw_ranges", False) or getattr(self, "debug_draw_mine_ranges", False):
            range_surface = pg.Surface((board_rect.width, board_rect.height), pg.SRCALPHA)

            if getattr(self, "debug_draw_ranges", False):
                sup_cells = float(getattr(self, "suppression_range_cells", 0.0) or 0.0)
                sup_radius_px = sup_cells * float(min(cell_w, cell_h))

                def draw_sup(agent: Any, rgba: Tuple[int, int, int, int]) -> None:
                    fx, fy = self._agent_float_pos_interp(agent, alpha)
                    cx = (fx + 0.5) * cell_w
                    cy = (fy + 0.5) * cell_h
                    pg.draw.circle(range_surface, rgba, (int(cx), int(cy)), int(sup_radius_px), width=2)

                for a in getattr(self, "blue_agents", []):
                    if self._enabled(a):
                        draw_sup(a, (50, 130, 255, 190))
                for a in getattr(self, "red_agents", []):
                    if self._enabled(a):
                        draw_sup(a, (255, 110, 70, 190))

            if getattr(self, "debug_draw_mine_ranges", False) and getattr(self, "mines", None):
                mine_cells = float(getattr(self, "mine_radius_cells", 0.0) or 0.0)
                mine_radius_px = mine_cells * float(min(cell_w, cell_h))
                for mine in getattr(self, "mines", []):
                    cx = (float(getattr(mine, "x", 0)) + 0.5) * cell_w
                    cy = (float(getattr(mine, "y", 0)) + 0.5) * cell_h
                    rgba = (40, 170, 230, 170) if str(getattr(mine, "owner_side", "blue")).lower() == "blue" else (230, 120, 80, 170)
                    pg.draw.circle(range_surface, rgba, (int(cx), int(cy)), int(mine_radius_px), width=1)

            surface.blit(range_surface, board_rect.topleft)

        gm = self._gm

        # flags
        def _gm_attr(*names, default=None):
            if gm is None:
                return default
            for n in names:
                if hasattr(gm, n):
                    return getattr(gm, n)
            return default

        blue_pos = _gm_attr("blue_flag_home", "blue_flag_position", default=(0, rows // 2))
        red_pos = _gm_attr("red_flag_home", "red_flag_position", default=(cols - 1, rows // 2))
        blue_taken = bool(_gm_attr("blue_flag_taken", default=False))
        red_taken = bool(_gm_attr("red_flag_taken", default=False))

        self.draw_flag(surface, board_rect, cell_w, cell_h, (int(blue_pos[0]), int(blue_pos[1])), (90, 170, 250), blue_taken)
        self.draw_flag(surface, board_rect, cell_w, cell_h, (int(red_pos[0]), int(red_pos[1])), (250, 120, 70), red_taken)

        # mine pickups
        for pickup in getattr(self, "mine_pickups", []):
            cx = board_rect.left + (float(getattr(pickup, "x", 0)) + 0.5) * cell_w
            cy = board_rect.top + (float(getattr(pickup, "y", 0)) + 0.5) * cell_h
            r_outer = int(0.30 * min(cell_w, cell_h))
            r_inner = int(0.16 * min(cell_w, cell_h))
            color = (80, 210, 255) if str(getattr(pickup, "owner_side", "blue")).lower() == "blue" else (255, 160, 110)
            pg.draw.circle(surface, (10, 10, 14), (int(cx), int(cy)), r_outer)
            pg.draw.circle(surface, color, (int(cx), int(cy)), r_outer, width=2)
            pg.draw.circle(surface, color, (int(cx), int(cy)), r_inner)

        # mines
        for mine in getattr(self, "mines", []):
            cx = board_rect.left + (float(getattr(mine, "x", 0)) + 0.5) * cell_w
            cy = board_rect.top + (float(getattr(mine, "y", 0)) + 0.5) * cell_h
            r = int(0.35 * min(cell_w, cell_h))
            color = (40, 170, 230) if str(getattr(mine, "owner_side", "blue")).lower() == "blue" else (230, 120, 80)
            pg.draw.circle(surface, color, (int(cx), int(cy)), r)
            pg.draw.circle(surface, (5, 5, 8), (int(cx), int(cy)), r, width=1)

        # agents (use interpolated position)
        def draw_agent(agent: Any, body_rgb: Tuple[int, int, int], enemy_flag_rgb: Tuple[int, int, int]) -> None:
            if agent is None:
                return

            fx, fy = self._agent_float_pos_interp(agent, alpha)
            center_x = board_rect.left + (fx + 0.5) * cell_w
            center_y = board_rect.top + (fy + 0.5) * cell_h
            tri_size = 0.45 * min(cell_w, cell_h)

            # aim: toward next waypoint if exists, else toward enemy flag
            target_x, target_y = fx, fy
            path = getattr(agent, "path", None)
            if path is not None:
                try:
                    if len(path) > 0:
                        tx, ty = path[0]
                        target_x, target_y = float(tx), float(ty)
                except Exception:
                    pass

            if (target_x, target_y) == (fx, fy):
                side = self._agent_side(agent)
                if gm is not None and callable(getattr(gm, "get_enemy_flag_position", None)):
                    try:
                        ex, ey = gm.get_enemy_flag_position(side)
                        target_x, target_y = float(ex), float(ey)
                    except Exception:
                        pass

            dx = target_x - fx
            dy = target_y - fy
            mag = (dx * dx + dy * dy) ** 0.5

            if mag < 1e-4:
                ux, uy = (1.0, 0.0) if self._agent_side(agent) == "blue" else (-1.0, 0.0)
            else:
                ux, uy = (dx / mag, dy / mag)

            lx, ly = -uy, ux

            tip = (int(center_x + ux * tri_size), int(center_y + uy * tri_size))
            left = (
                int(center_x - ux * tri_size * 0.6 + lx * tri_size * 0.6),
                int(center_y - uy * tri_size * 0.6 + ly * tri_size * 0.6),
            )
            right = (
                int(center_x - ux * tri_size * 0.6 - lx * tri_size * 0.6),
                int(center_y - uy * tri_size * 0.6 - ly * tri_size * 0.6),
            )

            body_color = body_rgb if self._enabled(agent) else (50, 50, 55)
            pg.draw.polygon(surface, body_color, (tip, left, right))

            if self._agent_is_carrying_flag(agent):
                flag_size = int(tri_size * 0.5)
                flag_rect = pg.Rect(tip[0] - flag_size // 2, tip[1] - flag_size // 2, flag_size, flag_size)
                pg.draw.rect(surface, enemy_flag_rgb, flag_rect)

            if self._agent_is_tagged(agent):
                pg.draw.polygon(surface, (245, 245, 245), (tip, left, right), width=2)

        for a in getattr(self, "blue_agents", []):
            draw_agent(a, (0, 180, 255), (250, 120, 70))
        for a in getattr(self, "red_agents", []):
            draw_agent(a, (255, 120, 40), (90, 170, 250))

        # banner
        bq = getattr(self, "banner_queue", None)
        if isinstance(bq, list) and len(bq) > 0:
            try:
                text, color, time_left = bq[-1]
                fade = max(0.3, min(1.0, float(time_left) / 2.0))
                font = pg.font.SysFont(None, 48)
                faded_color = tuple(int(c * fade) for c in color)
                img = font.render(str(text), True, faded_color)
                surface.blit(img, (board_rect.centerx - img.get_width() // 2, board_rect.top + 12))
            except Exception:
                pass

    def draw_halves_and_center_line(self, surface: pg.Surface, board_rect: pg.Rect) -> None:
        cols = self._safe_col_count()
        rect_width, rect_height = board_rect.width, board_rect.height
        cell_width = rect_width / max(1, cols)

        def fill_cols(col_start: int, col_end: int, rgba: Tuple[int, int, int, int]) -> None:
            if col_start > col_end:
                return
            x0 = int(board_rect.left + col_start * cell_width)
            x1 = int(board_rect.left + (col_end + 1) * cell_width)
            band_w = max(1, x1 - x0)
            band = pg.Surface((band_w, rect_height), pg.SRCALPHA)
            band.fill(rgba)
            surface.blit(band, (x0, board_rect.top))

        (blue_min_col, blue_max_col), (red_min_col, red_max_col) = self._zone_ranges()

        mid_start = blue_max_col + 1
        mid_end = red_min_col - 1

        fill_cols(blue_min_col, blue_max_col, (15, 45, 120, 140))
        if mid_start <= mid_end:
            fill_cols(mid_start, mid_end, (40, 40, 55, 90))
        fill_cols(red_min_col, red_max_col, (120, 45, 15, 140))

        mid_col = cols // 2
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
