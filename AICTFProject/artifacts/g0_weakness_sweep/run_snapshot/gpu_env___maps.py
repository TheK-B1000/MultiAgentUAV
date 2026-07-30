from __future__ import annotations

from typing import Tuple

MAP_A_OPEN = "map_a_open"
MAP_B_SPLIT_LANE = "map_b_split_lane"
MAP_B_SPLIT_LANE_V2 = "map_b_split_lane_v2"
MAP_C_HOME_CORRIDOR = "map_c_home_corridor"

MAP_LAYOUT_ALIASES = {
    "a": MAP_A_OPEN,
    "open": MAP_A_OPEN,
    "open_arena": MAP_A_OPEN,
    "map_a": MAP_A_OPEN,
    "map_a_open": MAP_A_OPEN,
    "b": MAP_B_SPLIT_LANE,
    "split_lane": MAP_B_SPLIT_LANE,
    "split_lane_chokepoint": MAP_B_SPLIT_LANE,
    "map_b": MAP_B_SPLIT_LANE,
    "map_b_split_lane": MAP_B_SPLIT_LANE,
    "b2": MAP_B_SPLIT_LANE_V2,
    "split_lane_v2": MAP_B_SPLIT_LANE_V2,
    "split_lane_task_pressure": MAP_B_SPLIT_LANE_V2,
    "map_b_v2": MAP_B_SPLIT_LANE_V2,
    "map_b_split_lane_v2": MAP_B_SPLIT_LANE_V2,
    "c": MAP_C_HOME_CORRIDOR,
    "home_corridor": MAP_C_HOME_CORRIDOR,
    "map_c": MAP_C_HOME_CORRIDOR,
    "map_c_home_corridor": MAP_C_HOME_CORRIDOR,
}

MAP_LAYOUTS = tuple(sorted(set(MAP_LAYOUT_ALIASES.values())))


def normalize_map_layout(value: str) -> str:
    key = str(value or MAP_A_OPEN).strip().lower().replace("-", "_").replace(" ", "_")
    layout = MAP_LAYOUT_ALIASES.get(key)
    if layout is None:
        allowed = ", ".join(MAP_LAYOUTS)
        raise ValueError(f"map_layout must be one of {{{allowed}}}, got {value!r}")
    return layout


def split_lane_rect_norm(
    *,
    x_min: float = 0.44,
    x_max: float = 0.56,
    y_min: float = 0.25,
    y_max: float = 0.72,
    mirror_y: bool = False,
) -> Tuple[float, float, float, float]:
    x0 = max(0.0, min(1.0, float(x_min)))
    x1 = max(0.0, min(1.0, float(x_max)))
    y0 = max(0.0, min(1.0, float(y_min)))
    y1 = max(0.0, min(1.0, float(y_max)))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if mirror_y:
        y0, y1 = 1.0 - y1, 1.0 - y0
    return x0, y0, x1, y1


def split_lane_v2_rect_norm(*, mirror_y: bool = False) -> Tuple[float, float, float, float]:
    """Lower-friction split-lane wall for task-pressure experiments."""
    return split_lane_rect_norm(
        x_min=0.465,
        x_max=0.535,
        y_min=0.32,
        y_max=0.62,
        mirror_y=mirror_y,
    )


def home_corridor_rect_norm(*, mirror_y: bool = False) -> Tuple[float, float, float, float]:
    """Wall positioned near BLUE's home (not centered like Map B) -- creates
    a SINGLE mandatory chokepoint on blue's flag-return leg. Intended
    affordances (per the locked K=4 map-design contract, 2026-07-29):
      - ESCORT: an unescorted blue carrier funnels through this corridor on
        the way home and is exposed to a waiting red interceptor there; a
        nearby teammate can block/redirect that interceptor.
      - TURTLE: a stationary blue defender anchored at the corridor can
        block red's approach to blue's home through the same narrow gap,
        which an open, undefended home (map_a-style) does not offer.
    Reuses the same rectangular-wall/corner-routing mechanism as Map B
    (``_MapStateMixin``); only the position and width differ. Blue's flag
    home sits at x=2 in a 20-wide field (~0.10 normalized) -- this band
    starts just past it, on the corridor blue's carrier must cross.

    Version 2 (revised after Version 1 traced live -- see tracker): V1 used
    a Map-B-style wall with open gaps at BOTH top and bottom (y=0.28-0.70),
    which on a 20x20 field left bypasses above ~y=5.3 and below ~y=13.3
    (~28% of field height each). Attackers/carriers routed around either
    end instead of through a defendable corridor -- TURTLE chase failures
    and ESCORT route confusion both traced to that dual-bypass geometry.
    V2 pins the wall flush against the BOTTOM edge (y_max=1.0), leaving
    exactly ONE gap near the top. Gap height is ~18% of the field
    (y_min=0.18 → ~3.4 cells): narrow enough to be a single choke, wide
    enough for corner-routing clearance (~1.5 cells). Do not shrink below
    ~0.15 or the top corridor becomes impassable under the shared router.

    FROZEN (2026-07-28): Map C V2 — no third wall version. Micro-gate /
    opponent work only; do not retune these numbers.
    """
    return split_lane_rect_norm(
        x_min=0.15,
        x_max=0.24,
        y_min=0.18,
        y_max=1.0,
        mirror_y=mirror_y,
    )


def is_split_lane_layout(value: str) -> bool:
    return normalize_map_layout(value) in {
        MAP_B_SPLIT_LANE,
        MAP_B_SPLIT_LANE_V2,
        MAP_C_HOME_CORRIDOR,
    }


def norm_rect_to_cells(
    rect: Tuple[float, float, float, float],
    *,
    cols: int,
    rows: int,
) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = rect
    max_x = float(max(0, int(cols) - 1))
    max_y = float(max(0, int(rows) - 1))
    return x0 * max_x, y0 * max_y, x1 * max_x, y1 * max_y


__all__ = [
    "MAP_A_OPEN",
    "MAP_B_SPLIT_LANE",
    "MAP_B_SPLIT_LANE_V2",
    "MAP_C_HOME_CORRIDOR",
    "MAP_LAYOUTS",
    "home_corridor_rect_norm",
    "is_split_lane_layout",
    "normalize_map_layout",
    "norm_rect_to_cells",
    "split_lane_rect_norm",
    "split_lane_v2_rect_norm",
]
