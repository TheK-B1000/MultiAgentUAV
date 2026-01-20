import heapq
import math
from typing import List, Tuple, Optional, Set, Dict, Iterable

Grid = List[List[int]]
Coord = Tuple[int, int]  # (col, row)


class Pathfinder:
    """
    Grid-based A* with:
      - Static obstacles from `grid` (grid[row][col] != 0)
      - Dynamic obstacles in `blocked`
      - Optional diagonals with corner-cut protection
      - Goal exception: may step onto goal even if dynamically blocked
      - Soft 'danger' costs (avoid but can pass through)

    Senior-grade robustness:
      - No mutation of self.blocked during search
      - Heap tie-breaker counter for deterministic ordering
      - Danger costs sanitized (non-finite ignored, negative clamped to 0)
    """

    SQRT2 = 1.41421356237

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
        self.danger_cost: Dict[Coord, float] = {}

        # tie-breaker for heap ordering
        self._push_id = 0

    def update_grid(self, grid: Grid, rows: int, cols: int) -> None:
        self.grid = grid
        self.rows = int(rows)
        self.cols = int(cols)
        self.blocked.clear()
        self.danger_cost.clear()
        self._push_id = 0

    def setDynamicObstacles(self, blocked_cells: List[Coord]) -> None:
        self.blocked = set((int(x), int(y)) for (x, y) in blocked_cells)

    def setDangerCosts(self, danger_cost: Dict[Coord, float]) -> None:
        """
        Expects {(x,y): penalty, ...}
        Penalty is soft cost added when stepping INTO that cell.
        Sanitizes:
          - non-finite -> ignored
          - negative -> clamped to 0
        """
        out: Dict[Coord, float] = {}
        for (x, y), c in danger_cost.items():
            xx, yy = int(x), int(y)
            try:
                cc = float(c)
            except Exception:
                continue
            if not math.isfinite(cc):
                continue
            if cc < 0.0:
                cc = 0.0
            out[(xx, yy)] = cc
        self.danger_cost = out

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

    def _neighbors(self, x: int, y: int) -> Iterable[Tuple[int, int, float]]:
        """
        Yields (nx, ny, base_step_cost) neighbors without blocked logic.
        """
        # cardinal
        yield (x + 1, y, 1.0)
        yield (x - 1, y, 1.0)
        yield (x, y + 1, 1.0)
        yield (x, y - 1, 1.0)

        if self.allow_diagonal:
            yield (x + 1, y + 1, self.SQRT2)
            yield (x + 1, y - 1, self.SQRT2)
            yield (x - 1, y + 1, self.SQRT2)
            yield (x - 1, y - 1, self.SQRT2)

    def getNeighbors(self, x: int, y: int, goal: Optional[Coord] = None, blocked: Optional[Set[Coord]] = None) -> List[Coord]:
        """
        Returns valid neighbors considering static grid, dynamic blocked, diagonal corner rules.
        Goal exception: may step onto goal even if dynamically blocked (but must be statically passable).
        """
        if blocked is None:
            blocked = self.blocked

        out: List[Coord] = []
        for nx, ny, _step in self._neighbors(x, y):
            nc = (nx, ny)

            if not self.inBounds(nx, ny):
                continue

            # allow stepping onto goal if statically passable, even if dynamically blocked
            if goal is not None and nc == goal:
                if self.isStaticPassable(nx, ny):
                    out.append(nc)
                continue

            if nc in blocked:
                continue

            if not self.isStaticPassable(nx, ny):
                continue

            # corner-cut protection
            dx = nx - x
            dy = ny - y
            if self.block_corners and dx != 0 and dy != 0:
                ax, ay = x + dx, y
                bx, by = x, y + dy

                if not (self.isStaticPassable(ax, ay) and self.isStaticPassable(bx, by)):
                    continue
                if (ax, ay) in blocked or (bx, by) in blocked:
                    continue

            out.append(nc)

        return out

    def heuristic(self, a: Coord, b: Coord) -> float:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if not self.allow_diagonal:
            return float(dx + dy)
        # octile distance
        return (dx + dy) + (self.SQRT2 - 2.0) * min(dx, dy)

    def rebuildPath(self, came_from: Dict[Coord, Coord], goal: Coord) -> List[Coord]:
        path = [goal]
        cur = goal
        while cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return path

    def prune_path(self, path: List[Coord]) -> List[Coord]:
        """
        Removes collinear intermediate points so the agent doesn't stutter on long straight lines.
        Safe and deterministic. Works for cardinal + diagonal.
        """
        if not path:
            return path
        out = [path[0]]
        for p in path[1:]:
            if len(out) < 2:
                out.append(p)
                continue
            a = out[-2]
            b = out[-1]
            if (b[0] - a[0], b[1] - a[1]) == (p[0] - b[0], p[1] - b[1]):
                out[-1] = p
            else:
                out.append(p)
        return out

    def astar(self, start: Coord, goal: Coord, max_expansions: Optional[int] = None) -> Optional[List[Coord]]:
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

        # Create a local view of blocked that treats start/goal as passable
        blocked_local = self.blocked
        if start in blocked_local or goal in blocked_local:
            blocked_local = set(blocked_local)
            blocked_local.discard(start)
            blocked_local.discard(goal)

        open_heap: List[Tuple[float, float, int, Coord]] = []
        self._push_id += 1
        heapq.heappush(open_heap, (self.heuristic(start, goal), 0.0, self._push_id, start))

        came_from: Dict[Coord, Coord] = {}
        g_cost: Dict[Coord, float] = {start: 0.0}
        closed: Set[Coord] = set()

        expansions = 0

        while open_heap:
            f, g, _pid, cur = heapq.heappop(open_heap)

            # stale heap entry
            if g > g_cost.get(cur, float("inf")):
                continue
            if cur in closed:
                continue

            closed.add(cur)

            expansions += 1
            if max_expansions is not None and expansions > int(max_expansions):
                return None

            if cur == goal:
                path = self.rebuildPath(came_from, cur)
                path = path[1:]  # exclude start
                return self.prune_path(path)

            cx, cy = cur
            for nb in self.getNeighbors(cx, cy, goal=goal, blocked=blocked_local):
                if nb in closed:
                    continue

                nx, ny = nb
                # determine base step (cardinal vs diagonal)
                if nx == cx or ny == cy:
                    step = 1.0
                else:
                    if not self.allow_diagonal:
                        continue
                    step = self.SQRT2

                # soft danger penalty on entering neighbor cell
                step += float(self.danger_cost.get(nb, 0.0))

                g2 = g + step
                if g2 < g_cost.get(nb, float("inf")):
                    g_cost[nb] = g2
                    came_from[nb] = cur
                    self._push_id += 1
                    heapq.heappush(open_heap, (g2 + self.heuristic(nb, goal), g2, self._push_id, nb))

        return None
