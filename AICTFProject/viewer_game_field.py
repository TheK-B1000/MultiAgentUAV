from __future__ import annotations

import tkinter as tk
from typing import Any, Dict, Optional, Tuple


class ViewerGameField:
    def __init__(
        self,
        game_field: Any,
        *,
        cell_size: int = 24,
        show_grid: bool = False,
    ) -> None:
        self.game_field = game_field
        self.cell_size = int(max(8, cell_size))
        self.show_grid = bool(show_grid)

        self.rows = int(getattr(game_field, "row_count", 0))
        self.cols = int(getattr(game_field, "col_count", 0))
        self.width = self.cols * self.cell_size
        self.height = self.rows * self.cell_size

        self.root = tk.Tk()
        self.root.title("CTF Viewer")
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg="white")
        self.canvas.pack()

        self._closed = False
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._draw_static()

    def _on_close(self) -> None:
        self._closed = True
        try:
            self.root.destroy()
        except Exception:
            pass

    @property
    def is_closed(self) -> bool:
        return bool(self._closed)

    def _cell_to_xy(self, x: float, y: float) -> Tuple[float, float]:
        return (x * self.cell_size, y * self.cell_size)

    def _draw_static(self) -> None:
        grid = getattr(self.game_field, "grid", None)
        if grid is None:
            return

        for r in range(self.rows):
            for c in range(self.cols):
                if int(grid[r][c]) != 0:
                    x0, y0 = self._cell_to_xy(c, r)
                    x1, y1 = x0 + self.cell_size, y0 + self.cell_size
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill="#2b2b2b", outline="")

        if self.show_grid:
            for r in range(self.rows + 1):
                y = r * self.cell_size
                self.canvas.create_line(0, y, self.width, y, fill="#e6e6e6")
            for c in range(self.cols + 1):
                x = c * self.cell_size
                self.canvas.create_line(x, 0, x, self.height, fill="#e6e6e6")

    def _draw_flags(self, gm: Any) -> None:
        blue_pos = getattr(gm, "blue_flag_position", (0, 0))
        red_pos = getattr(gm, "red_flag_position", (self.cols - 1, self.rows - 1))
        for pos, color in [(blue_pos, "#2f80ed"), (red_pos, "#eb5757")]:
            x0, y0 = self._cell_to_xy(float(pos[0]), float(pos[1]))
            x1, y1 = x0 + self.cell_size, y0 + self.cell_size
            self.canvas.create_rectangle(
                x0 + 4,
                y0 + 4,
                x1 - 4,
                y1 - 4,
                fill=color,
                outline="",
                tag="dynamic",
            )

    def _draw_agents(self, agents: list, color: str) -> None:
        for a in agents:
            if a is None or (hasattr(a, "isEnabled") and not a.isEnabled()):
                continue
            fx, fy = getattr(a, "float_pos", (getattr(a, "x", 0), getattr(a, "y", 0)))
            x0, y0 = self._cell_to_xy(float(fx), float(fy))
            x1, y1 = x0 + self.cell_size, y0 + self.cell_size
            radius_pad = 4 if not bool(getattr(a, "isCarryingFlag", lambda: False)()) else 1
            self.canvas.create_oval(
                x0 + radius_pad,
                y0 + radius_pad,
                x1 - radius_pad,
                y1 - radius_pad,
                fill=color,
                outline="",
                tag="dynamic",
            )

    def _draw_mines(self, mines: list, color: str) -> None:
        for m in mines:
            x0, y0 = self._cell_to_xy(float(getattr(m, "x", 0)), float(getattr(m, "y", 0)))
            x1, y1 = x0 + self.cell_size, y0 + self.cell_size
            self.canvas.create_rectangle(
                x0 + 8,
                y0 + 8,
                x1 - 8,
                y1 - 8,
                fill=color,
                outline="",
                tag="dynamic",
            )

    def _draw_pickups(self, pickups: list, color: str) -> None:
        for p in pickups:
            x0, y0 = self._cell_to_xy(float(getattr(p, "x", 0)), float(getattr(p, "y", 0)))
            x1, y1 = x0 + self.cell_size, y0 + self.cell_size
            self.canvas.create_oval(
                x0 + 10,
                y0 + 10,
                x1 - 10,
                y1 - 10,
                fill=color,
                outline="",
                tag="dynamic",
            )

    def update(self) -> None:
        if self._closed:
            return

        self.canvas.delete("dynamic")
        self.canvas.delete("dynamic_text")

        gm = getattr(self.game_field, "manager", None)
        if gm is not None:
            self._draw_flags(gm)

        blue_agents = getattr(self.game_field, "blue_agents", [])
        red_agents = getattr(self.game_field, "red_agents", [])
        self._draw_agents(blue_agents, "#2f80ed")
        self._draw_agents(red_agents, "#eb5757")

        mines = getattr(self.game_field, "mines", [])
        for m in mines:
            color = "#2f80ed" if getattr(m, "owner_side", "") == "blue" else "#eb5757"
            self._draw_mines([m], color)

        pickups = getattr(self.game_field, "mine_pickups", [])
        for p in pickups:
            color = "#2f80ed" if getattr(p, "owner_side", "") == "blue" else "#eb5757"
            self._draw_pickups([p], color)

        if gm is not None:
            score = f"{int(getattr(gm, 'blue_score', 0))}:{int(getattr(gm, 'red_score', 0))}"
            self.canvas.create_text(
                8,
                8,
                anchor="nw",
                text=f"Score {score}",
                fill="#111",
                tag="dynamic_text",
            )

        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self._closed = True
