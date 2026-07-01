#!/usr/bin/env python3
"""Reproduce / quantify obstacle-stuck behavior on Map B.

Isolates the navigation + collision physics (``_route_targets_around_obstacles``
-> ``_integrate_side`` -> ``_revert_obstacle_hits``) and drives a grid of agents
with fixed cross-wall targets, detecting:

  * agents that end up INSIDE the obstacle,
  * agents that are alive, far from target, yet barely move for many steps
    (i.e. permanently stuck rather than briefly grazing the wall).

Run:
    uv run python tools/probe_obstacle_stuck.py
"""
from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from game_field_gpu import BatchedCTFCore, GPUFieldConfig


def _build_core(map_layout: str = "map_b", n_agents: int = 1) -> BatchedCTFCore:
    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=n_agents,
        max_red_agents=n_agents,
        device="cpu",
        max_decision_steps=400,
        map_layout=map_layout,
    )
    core = BatchedCTFCore(cfg)
    core.reset_all()
    return core


def _make_start_target_grid(core: BatchedCTFCore):
    """Build a batch of (start, target) pairs that all require crossing the wall."""
    rect = core.obstacle_rects[0, 0].tolist()
    x0, y0, x1, y1 = rect
    cols = int(core.cols)
    rows = int(core.rows)
    max_x = float(cols - 1)
    max_y = float(rows - 1)

    starts = []
    targets = []
    # Left-start -> right-target and right-start -> left-target across a sweep of y.
    ys = [r * 0.5 for r in range(0, int(max_y * 2) + 1)]  # 0.0 .. max_y in 0.5 steps
    left_x = max(0.0, x0 - 0.4)   # hugging the near (left) face
    right_x = min(max_x, x1 + 0.4)
    for sy in ys:
        ty = max_y - sy  # opposite y so the straight line cuts through the wall band
        starts.append((left_x, sy))
        targets.append((right_x, ty))
        starts.append((right_x, sy))
        targets.append((left_x, ty))
    # A few starts dead-center against each face, aimed straight through.
    mid_y = (y0 + y1) * 0.5
    for off in (-2.0, -1.0, 0.0, 1.0, 2.0):
        starts.append((left_x, mid_y + off))
        targets.append((right_x, mid_y + off))
        starts.append((right_x, mid_y + off))
        targets.append((left_x, mid_y + off))
    return starts, targets


def run_probe(map_layout: str = "map_b", steps: int = 200) -> dict:
    core = _build_core(map_layout=map_layout, n_agents=1)
    starts, targets = _make_start_target_grid(core)
    n = len(starts)

    dev = core.device
    f32 = core.blue_x.dtype
    sx = torch.tensor([[s[0]] for s in starts], dtype=f32, device=dev)  # (n,1)
    sy = torch.tensor([[s[1]] for s in starts], dtype=f32, device=dev)
    tx = torch.tensor([[t[0]] for t in targets], dtype=f32, device=dev)
    ty = torch.tensor([[t[1]] for t in targets], dtype=f32, device=dev)

    # Broadcast obstacle state from env 0 to all n "envs" by re-allocating with B=n.
    rect = core.obstacle_rects[0:1].clone()  # (1,1,4)
    active = core.obstacle_active[0:1].clone()  # (1,1)
    core.obstacle_rects = rect.expand(n, -1, -1).contiguous()
    core.obstacle_active = active.expand(n, -1).contiguous()
    core.map_layout = core.map_layout
    core.B = n
    core.Nb = 1
    core.Nr = 1
    # Per-env runtime knobs consulted by _integrate_side.
    core.rt_current_strength_cps = torch.zeros((n,), dtype=f32, device=dev)
    core.rt_drift_sigma_cells = torch.zeros((n,), dtype=f32, device=dev)
    core.rt_blue_speed_scale = torch.ones((n,), dtype=f32, device=dev)

    return _faithful_sim(core, sx, sy, tx, ty, steps, window=25)


def _faithful_sim(core, sx, sy, tx, ty, steps, window) -> dict:
    f32 = core.blue_x.dtype
    x = sx.clone()
    y = sy.clone()
    heading = torch.atan2(ty - y, tx - x)
    speed = torch.zeros_like(x)
    alive = torch.ones_like(x, dtype=torch.bool)
    speed_cap = torch.full_like(speed, float(core.cfg.max_speed_cps))

    hist_x: list[torch.Tensor] = []
    hist_y: list[torch.Tensor] = []
    inside_ever = torch.zeros_like(x, dtype=torch.bool)

    for _ in range(steps):
        rtx, rty = core._route_targets_around_obstacles(x, y, tx, ty)
        prev_x, prev_y = x.clone(), y.clone()
        x, y, heading, speed, _oob, _yaw = core._integrate_side(
            prev_x, prev_y, heading, speed, alive, rtx, rty, speed_cap=speed_cap
        )
        x, y, speed, _hit = core._revert_obstacle_hits(prev_x, prev_y, x, y, speed, alive)
        inside_ever = inside_ever | core._points_in_obstacles(x, y)
        hist_x.append(x.clone())
        hist_y.append(y.clone())

    X = torch.stack(hist_x, dim=0)  # (steps, n, 1)
    Y = torch.stack(hist_y, dim=0)
    n = X.shape[1]

    dist_to_target = torch.sqrt((x - tx) ** 2 + (y - ty) ** 2)
    reached = (dist_to_target.squeeze(-1) <= 1.5)

    # Movement over the last `window` steps.
    wx = X[-window:]
    wy = Y[-window:]
    span = torch.sqrt(
        (wx.amax(0) - wx.amin(0)) ** 2 + (wy.amax(0) - wy.amin(0)) ** 2
    ).squeeze(-1)
    far = dist_to_target.squeeze(-1) > 1.5
    stuck = far & (span < 0.5)
    inside = core._points_in_obstacles(x, y).squeeze(-1)

    stuck_idx = torch.where(stuck)[0].tolist()
    inside_idx = torch.where(inside)[0].tolist()
    inside_ever_idx = torch.where(inside_ever.squeeze(-1))[0].tolist()

    details = []
    for i in stuck_idx[:20]:
        details.append(
            {
                "agent": int(i),
                "start": (round(float(sx[i, 0]), 2), round(float(sy[i, 0]), 2)),
                "target": (round(float(tx[i, 0]), 2), round(float(ty[i, 0]), 2)),
                "final": (round(float(x[i, 0]), 2), round(float(y[i, 0]), 2)),
                "last_window_span": round(float(span[i]), 3),
                "dist_to_target": round(float(dist_to_target[i, 0]), 2),
            }
        )

    return {
        "map_layout": core.map_layout,
        "n_agents": int(n),
        "steps": int(steps),
        "reached_target": int(reached.sum().item()),
        "stuck": int(stuck.sum().item()),
        "ended_inside_obstacle": int(inside.sum().item()),
        "inside_obstacle_ever": int(inside_ever.sum().item()),
        "obstacle_rect": [round(v, 2) for v in core.obstacle_rects[0, 0].tolist()],
        "stuck_examples": details,
        "inside_ever_idx": inside_ever_idx[:20],
    }


if __name__ == "__main__":
    import json

    for layout in ("map_b", "map_b_v2"):
        result = run_probe(map_layout=layout, steps=200)
        print(json.dumps(result, indent=2))
        print("-" * 60)
