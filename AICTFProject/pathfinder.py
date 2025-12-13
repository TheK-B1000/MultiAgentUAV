import heapq
from typing import List, Tuple, Optional, Set, Dict

Grid = List[List[int]]
Coord = Tuple[int, int]  # (col, row)


class Pathfinder:
    """
    Grid-based A* pathfinder with:

      - Static obstacles from `grid` (grid[row][col] != 0).
      - Dynamic obstacles in `self.blocked` (agents, mines, etc.).
      - Optional diagonal movement with corner-cutting protection.
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

        # Dynamic obstacles (agents + mines, set by GameField before path queries)
        self.blocked: Set[Coord] = set()

        self.original_blocked: Set[Coord] = set()

    # Grid state management
    def update_grid(self, grid: Grid, rows: int, cols: int) -> None:
        self.grid = grid
        self.rows = rows
        self.cols = cols
        self.blocked.clear()

    def setDynamicObstacles(self, blocked_cells: List[Coord]) -> None:
        self.blocked = set(blocked_cells)

    # Basic checks
    def inBounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.cols and 0 <= y < self.rows

    def isStaticPassable(self, x: int, y: int) -> bool:
        if not self.inBounds(x, y):
            return False
        return self.grid[y][x] == 0

    # Neighbor generation
    def getNeighbors(self, cell_x: int, cell_y: int, goal: Optional[Coord] = None, ) -> List[Coord]:
        steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # cardinal
        if self.allow_diagonal:
            steps += [(1, 1), (1, -1), (-1, 1), (-1, -1)]  # diagonals

        neighbors: List[Coord] = []
        for dx, dy in steps:
            nx, ny = cell_x + dx, cell_y + dy
            new_cell = (nx, ny)

            if not self.inBounds(nx, ny):
                continue

            # Always allow stepping ONTO the goal if it's statically passable,
            # even if dynamic obstacles temporarily mark it.
            if goal is not None and new_cell == goal:
                if self.isStaticPassable(nx, ny):
                    neighbors.append(new_cell)
                continue

            # 1) Blocked by dynamic obstacle (agent or mine)?
            if new_cell in self.blocked:
                continue

            # 2) Blocked by static wall?
            if not self.isStaticPassable(nx, ny):
                continue

            # 3) Corner cutting protection for diagonal moves
            if self.block_corners and dx != 0 and dy != 0:
                ax, ay = cell_x + dx, cell_y
                bx, by = cell_x, cell_y + dy

                if not (self.isStaticPassable(ax, ay) and self.isStaticPassable(bx, by)):
                    continue

                # ✅ also prevent cutting corners through dynamic obstacles
                if (ax, ay) in self.blocked or (bx, by) in self.blocked:
                    continue

            neighbors.append(new_cell)

        return neighbors

    # ------------------------------------------------------------------
    # A* core
    # ------------------------------------------------------------------
    def heuristic(self, a: Coord, b: Coord) -> float:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if not self.allow_diagonal:
            # Manhattan distance for 4-neighborhood
            return float(dx + dy)
        # Octile for 8-neighborhood
        SQRT2 = 1.41421356237
        return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)

    def rebuildPath(self, came_from: Dict[Coord, Coord], goal: Coord) -> List[Coord]:
        path = [goal]
        current = goal
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def astar(self, start: Coord, goal: Coord) -> Optional[List[Coord]]:
        if start == goal:
            return []

        if not (self.inBounds(*start) and self.inBounds(*goal)):
            return None

        if not self.isStaticPassable(*goal):
            return None

        # Bulletproof: treat start/goal as passable even if dynamic obstacles include them
        blocked_saved = None
        if start in self.blocked or goal in self.blocked:
            blocked_saved = self.blocked
            self.blocked = set(self.blocked)
            self.blocked.discard(start)
            self.blocked.discard(goal)

        try:
            open_queue: List[Tuple[float, float, Coord]] = []
            heapq.heappush(open_queue, (self.heuristic(start, goal), 0.0, start))

            came_from: Dict[Coord, Coord] = {}
            g_costs: Dict[Coord, float] = {start: 0.0}
            closed: Set[Coord] = set()

            SQRT2 = 1.41421356237

            while open_queue:
                f_score, g_score, current = heapq.heappop(open_queue)

                # Skip stale heap entries
                if g_score > g_costs.get(current, float("inf")):
                    continue

                if current in closed:
                    continue
                closed.add(current)

                if current == goal:
                    path = self.rebuildPath(came_from, current)
                    return path[1:]  # exclude start

                cx, cy = current
                for nx, ny in self.getNeighbors(cx, cy, goal):
                    neighbor = (nx, ny)
                    if neighbor in closed:
                        continue

                    # Cost: 1.0 for cardinal, SQRT2 for diagonal (only if diagonal allowed)
                    if nx == cx or ny == cy:
                        step_cost = 1.0
                    else:
                        if not self.allow_diagonal:
                            continue
                        step_cost = SQRT2

                    g_new = g_score + step_cost
                    if g_new < g_costs.get(neighbor, float("inf")):
                        g_costs[neighbor] = g_new
                        came_from[neighbor] = current
                        f_total = g_new + self.heuristic(neighbor, goal)
                        heapq.heappush(open_queue, (f_total, g_new, neighbor))

            return None

        finally:
            if blocked_saved is not None:
                self.blocked = blocked_saved

