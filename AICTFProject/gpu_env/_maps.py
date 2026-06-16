from __future__ import annotations

from typing import Tuple

MAP_A_OPEN = "map_a_open"
MAP_B_SPLIT_LANE = "map_b_split_lane"
MAP_B_SPLIT_LANE_V2 = "map_b_split_lane_v2"

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


def is_split_lane_layout(value: str) -> bool:
    return normalize_map_layout(value) in {MAP_B_SPLIT_LANE, MAP_B_SPLIT_LANE_V2}


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
    "MAP_LAYOUTS",
    "is_split_lane_layout",
    "normalize_map_layout",
    "norm_rect_to_cells",
    "split_lane_rect_norm",
    "split_lane_v2_rect_norm",
]
