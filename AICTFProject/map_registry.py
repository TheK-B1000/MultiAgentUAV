"""
map_registry.py

Central registry + loader for CTF map configurations.
Supports:
  - in-code registry of named maps
  - ASCII map file loader (".", "0", " " = free; "#", "1", "X" = wall)
  - factory to build GameField from map name/path
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import os

from game_field import GameField

Grid = List[List[int]]


WALL_CHARS = {"#", "1", "X"}
FREE_CHARS = {".", "0", " "}


@dataclass(frozen=True)
class MapSpec:
    name: str
    grid: Grid


_MAPS: Dict[str, MapSpec] = {}


def _normalize_name(name: str) -> str:
    return str(name).strip().lower()


def register_map(name: str, grid: Grid) -> None:
    if not name:
        raise ValueError("map name must be non-empty")
    if not grid or not grid[0]:
        raise ValueError("map grid must be non-empty")

    cols = len(grid[0])
    for row in grid:
        if len(row) != cols:
            raise ValueError("map grid must be rectangular")
        for v in row:
            if int(v) not in (0, 1):
                raise ValueError("map grid values must be 0/1")

    key = _normalize_name(name)
    _MAPS[key] = MapSpec(name=key, grid=[list(map(int, r)) for r in grid])


def list_maps() -> List[str]:
    return sorted(_MAPS.keys())


def get_map(name: str) -> Grid:
    key = _normalize_name(name)
    if key not in _MAPS:
        raise KeyError(f"unknown map: {name}")
    spec = _MAPS[key]
    return [row[:] for row in spec.grid]


def make_empty_grid(rows: int, cols: int) -> Grid:
    r = max(1, int(rows))
    c = max(1, int(cols))
    return [[0] * c for _ in range(r)]


def parse_ascii_map(lines: Iterable[str]) -> Grid:
    raw = [line.rstrip("\n") for line in lines if line.strip("\n") != ""]
    if not raw:
        raise ValueError("map file is empty")

    width = max(len(line) for line in raw)
    grid: Grid = []

    for line in raw:
        row: List[int] = []
        for ch in line.ljust(width):
            if ch in WALL_CHARS:
                row.append(1)
            elif ch in FREE_CHARS:
                row.append(0)
            else:
                # Unknown char: treat as free but keep it explicit for robustness
                row.append(0)
        grid.append(row)

    return grid


def load_map_from_file(path: str) -> Grid:
    if not path:
        raise ValueError("map path must be non-empty")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return parse_ascii_map(f.readlines())


def register_map_file(name: str, path: str) -> None:
    grid = load_map_from_file(path)
    register_map(name, grid)


def make_game_field(
    *,
    map_name: Optional[str] = None,
    map_path: Optional[str] = None,
    rows: int = 20,
    cols: int = 20,
) -> GameField:
    if map_path:
        grid = load_map_from_file(map_path)
    elif map_name:
        grid = get_map(map_name)
    else:
        grid = make_empty_grid(rows, cols)
    return GameField(grid)


# -------------------------------------------------------------------
# Built-in maps (empty baselines to keep training deterministic)
# -------------------------------------------------------------------
register_map("empty_20x20", make_empty_grid(20, 20))
register_map("empty_30x40", make_empty_grid(30, 40))

# Optional: register maps from ./maps directory if present
_MAP_DIR = os.path.join(os.path.dirname(__file__), "maps")
_EMPTY_20_PATH = os.path.join(_MAP_DIR, "empty_20x20.txt")
if os.path.exists(_EMPTY_20_PATH):
    try:
        register_map_file("empty_20x20_txt", _EMPTY_20_PATH)
    except Exception:
        pass

