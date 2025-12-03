# pathfinder.py
import heapq
from typing import List, Tuple, Optional, Set, Dict, Iterable
import math


Grid = List[List[int]]
Coord = Tuple[int, int]  # integer grid cell (col, row)


class Pathfinder:
    """
    Fully continuous-aware A* pathfinder.
    - Static walls from grid
    - Dynamic blocking from live Agent objects (using _float_x/y)
    - Optional enemy inflation radius (used when carrying flag)
    """

    def __init__(
        self,
        grid: Grid,
        rows: int,
        cols: int,
        allow_diagonal: bool = True,
        block_corners: bool = True,
    ):
        self.grid = grid
        self.rows = rows
        self.cols = cols
        self.allow_diagonal = allow_diagonal
        self.block_corners = block_corners

        self.blocked: Set[Coord] = set()  # dynamically blocked cells

    def update_grid(self, grid: Grid, rows: int, cols: int) -> None:
        self.grid = grid
        self.rows = rows
        self.cols = cols
        self.blocked.clear()

    def set_dynamic_obstacles(
        self,
        agents: Iterable["Agent"],
        extra_blocks: Optional[Iterable[Coord]] = None,
        enemy_inflation_radius: int = 0,
        enemy_agents: Optional[Iterable["Agent"]] = None,
    ) -> None:
        """Rebuild blocked cells from current agent positions."""
        self.blocked.clear()

        # Block all enabled agents
        for agent in agents:
            if agent.isEnabled():
                ix, iy = int(agent._float_x), int(agent._float_y)
                self.blocked.add((ix, iy))

        # Inflate enemy positions if requested (e.g. avoid carrier)
        if enemy_inflation_radius > 0 and enemy_agents is not None:
            for agent in enemy_agents:
                if not agent.isEnabled():
                    continue
                cx, cy = int(agent._float_x), int(agent._float_y)
                r = enemy_inflation_radius
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < self.cols and 0 <= ny < self.rows:
                            self.blocked.add((nx, ny))

        # Extra static blocks (e.g. mines)
        if extra_blocks:
            self.blocked.update(extra_blocks)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.cols and 0 <= y < self.rows

    def is_passable(self, x: int, y: int) -> bool:
        if not self.in_bounds(x, y):
            return False
        if self.grid[y][x] != 0:
            return False
        return (x, y) not in self.blocked

    def get_neighbors(self, x: int, y: int, goal: Optional[Coord] = None) -> List[Coord]:
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        if self.allow_diagonal:
            candidates += [(x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1)]

        neighbors = []
        for nx, ny in candidates:
            if not self.in_bounds(nx, ny):
                continue
            if (nx, ny) in self.blocked:
                continue
            if self.grid[ny][nx] != 0:
                continue
            if self.block_corners and abs(nx - x) + abs(ny - y) == 2:
                if not (self.grid[y][nx] == 0 and self.grid[ny][x] == 0):
                    continue
            if goal is not None and (nx, ny) == goal:
                if self.grid[ny][nx] == 0:
                    neighbors.append((nx, ny))
                continue
            neighbors.append((nx, ny))
        return neighbors

    @staticmethod
    def octile_distance(a: Coord, b: Coord) -> float:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (dx + dy) + (1.41421356237 - 2) * min(dx, dy)

    def find_path(self, start: Coord, goal: Coord) -> Optional[List[Coord]]:
        if start == goal:
            return []

        if not (self.in_bounds(*start) and self.in_bounds(*goal)):
            return None
        if self.grid[goal[1]][goal[0]] != 0:
            return None

        open_set = []
        tie = 0
        heapq.heappush(open_set, (self.octile_distance(start, goal), tie, start))

        came_from: Dict[Coord, Coord] = {}
        g_score: Dict[Coord, float] = {start: 0.0}
        SQRT2 = 1.41421356237

        while open_set:
            _, _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path[1:]

            cx, cy = current
            for nx, ny in self.get_neighbors(cx, cy, goal):
                step_cost = SQRT2 if abs(nx - cx) + abs(ny - cy) == 2 else 1.0
                tentative_g = g_score[current] + step_cost

                if tentative_g < g_score.get((nx, ny), float("inf")):
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative_g
                    f = tentative_g + self.octile_distance((nx, ny), goal)
                    tie += 1
                    heapq.heappush(open_set, (f, tie, (nx, ny)))

        return None  # no path