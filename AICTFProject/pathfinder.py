import heapq
from typing import List, Tuple, Optional, Set, Dict

Grid = List[List[int]]
Coord = Tuple[int, int]  # (col, row)


class Pathfinder:
    """
    Grid-based A* with:
      - Static obstacles from `grid` (grid[row][col] != 0)
      - Dynamic obstacles in `blocked`
      - Optional diagonals with corner-cut protection
      - Goal exception: may step onto goal even if dynamically blocked
      - NEW: Soft 'danger' costs (avoid but can pass through)
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
        self.rows = int(rows)
        self.cols = int(cols)
        self.allow_diagonal = bool(allow_diagonal)
        self.block_corners = bool(block_corners)

        self.blocked: Set[Coord] = set()

        # NEW: soft penalty field
        self.danger_cost: Dict[Coord, float] = {}

    def update_grid(self, grid: Grid, rows: int, cols: int) -> None:
        self.grid = grid
        self.rows = int(rows)
        self.cols = int(cols)
        self.blocked.clear()
        self.danger_cost.clear()

    def setDynamicObstacles(self, blocked_cells: List[Coord]) -> None:
        self.blocked = set((int(x), int(y)) for (x, y) in blocked_cells)

    # NEW
    def setDangerCosts(self, danger_cost: Dict[Coord, float]) -> None:
        # expects {(x,y): penalty, ...}
        self.danger_cost = { (int(x), int(y)): float(c) for (x, y), c in danger_cost.items() }

    # NEW
    def clearDangerCosts(self) -> None:
        self.danger_cost = {}

    def inBounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.cols and 0 <= y < self.rows

    def isStaticPassable(self, x: int, y: int) -> bool:
        if not self.inBounds(x, y):
            return False
        try:
            return self.grid[y][x] == 0
        except Exception:
            return False

    def getNeighbors(self, x: int, y: int, goal: Optional[Coord] = None) -> List[Coord]:
        steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if self.allow_diagonal:
            steps += [(1, 1), (1, -1), (-1, 1), (-1, -1)]

        out: List[Coord] = []
        for dx, dy in steps:
            nx, ny = x + dx, y + dy
            nc = (nx, ny)

            if not self.inBounds(nx, ny):
                continue

            # allow stepping onto goal if statically passable, even if dynamically blocked
            if goal is not None and nc == goal:
                if self.isStaticPassable(nx, ny):
                    out.append(nc)
                continue

            if nc in self.blocked:
                continue

            if not self.isStaticPassable(nx, ny):
                continue

            if self.block_corners and dx != 0 and dy != 0:
                ax, ay = x + dx, y
                bx, by = x, y + dy

                if not (self.isStaticPassable(ax, ay) and self.isStaticPassable(bx, by)):
                    continue

                if (ax, ay) in self.blocked or (bx, by) in self.blocked:
                    continue

            out.append(nc)

        return out

    def heuristic(self, a: Coord, b: Coord) -> float:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if not self.allow_diagonal:
            return float(dx + dy)
        SQRT2 = 1.41421356237
        return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)

    def rebuildPath(self, came_from: Dict[Coord, Coord], goal: Coord) -> List[Coord]:
        path = [goal]
        cur = goal
        while cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return path

    def astar(self, start: Coord, goal: Coord) -> Optional[List[Coord]]:
        sx, sy = int(start[0]), int(start[1])
        gx, gy = int(goal[0]), int(goal[1])
        start = (sx, sy)
        goal = (gx, gy)

        if start == goal:
            return []

        if not (self.inBounds(*start) and self.inBounds(*goal)):
            return None

        if not self.isStaticPassable(*goal):
            return None

        # Treat start/goal as passable even if present in dynamic blocked set
        blocked_saved = None
        if start in self.blocked or goal in self.blocked:
            blocked_saved = self.blocked
            self.blocked = set(self.blocked)
            self.blocked.discard(start)
            self.blocked.discard(goal)

        try:
            open_heap: List[Tuple[float, float, Coord]] = []
            heapq.heappush(open_heap, (self.heuristic(start, goal), 0.0, start))

            came_from: Dict[Coord, Coord] = {}
            g_cost: Dict[Coord, float] = {start: 0.0}
            closed: Set[Coord] = set()

            SQRT2 = 1.41421356237

            while open_heap:
                f, g, cur = heapq.heappop(open_heap)

                if g > g_cost.get(cur, float("inf")):
                    continue
                if cur in closed:
                    continue
                closed.add(cur)

                if cur == goal:
                    path = self.rebuildPath(came_from, cur)
                    return path[1:]  # exclude start

                cx, cy = cur
                for nx, ny in self.getNeighbors(cx, cy, goal=goal):
                    nb = (nx, ny)
                    if nb in closed:
                        continue

                    if nx == cx or ny == cy:
                        step = 1.0
                    else:
                        if not self.allow_diagonal:
                            continue
                        step = SQRT2

                    # NEW: add soft danger penalty
                    step += float(self.danger_cost.get(nb, 0.0))

                    g2 = g + step
                    if g2 < g_cost.get(nb, float("inf")):
                        g_cost[nb] = g2
                        came_from[nb] = cur
                        heapq.heappush(open_heap, (g2 + self.heuristic(nb, goal), g2, nb))

            return None
        finally:
            if blocked_saved is not None:
                self.blocked = blocked_saved
