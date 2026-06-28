#!/usr/bin/env python3
"""Full-environment Map B stuck check.

Drives the real ``core.step()`` loop (scripted blue + scripted red, collision
guard, mirrored walls, mines, combat) across many parallel envs and reports any
alive agent that stays pinned within ~1 cell of a wall face with near-zero net
movement over a long window -- the signature of a collision-stuck agent.
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


def run(map_layout: str = "map_b", n_envs: int = 64, n_agents: int = 4, steps: int = 400) -> dict:
    cfg = GPUFieldConfig(
        n_envs=n_envs,
        max_blue_agents=n_agents,
        max_red_agents=n_agents,
        device="cpu",
        seed=1234,
        max_decision_steps=steps,
        stalemate_max_steps=steps,
        map_layout=map_layout,
        map_b_vertical_mirror_prob=0.5,
    )
    core = BatchedCTFCore(cfg)
    core.blue_scripted = True
    core.reset_all()

    B, Nb = core.B, core.Nb
    window = 40
    hist_bx: list[torch.Tensor] = []
    hist_by: list[torch.Tensor] = []
    near_wall_steps = torch.zeros((B, Nb), dtype=torch.int64)
    inside_events = 0

    act = torch.zeros((B, Nb * 2), dtype=torch.int64, device=core.device)
    for _ in range(steps):
        core.step(act)
        hist_bx.append(core.blue_x.clone())
        hist_by.append(core.blue_y.clone())
        inside_events += int(core._points_in_obstacles(core.blue_x, core.blue_y).sum().item())
        inside_events += int(core._points_in_obstacles(core.red_x, core.red_y).sum().item())

    BX = torch.stack(hist_bx, dim=0)  # (steps, B, Nb)
    BY = torch.stack(hist_by, dim=0)

    rect = core.obstacle_rects[:, 0, :]  # (B,4)
    x0 = rect[:, 0][None, :, None]
    y0 = rect[:, 1][None, :, None]
    x1 = rect[:, 2][None, :, None]
    y1 = rect[:, 3][None, :, None]
    # Distance to the wall rectangle (0 inside) per (step,B,Nb).
    dx = torch.maximum(torch.maximum(x0 - BX, BX - x1), torch.zeros_like(BX))
    dy = torch.maximum(torch.maximum(y0 - BY, BY - y1), torch.zeros_like(BY))
    dist_wall = torch.sqrt(dx * dx + dy * dy)

    # Sliding-window net movement.
    stuck_against_wall = 0
    worst = 0.0
    for t in range(window, BX.shape[0]):
        wx = BX[t - window:t]
        wy = BY[t - window:t]
        span = torch.sqrt((wx.amax(0) - wx.amin(0)) ** 2 + (wy.amax(0) - wy.amin(0)) ** 2)
        near = dist_wall[t] < 1.0
        pinned = near & (span < 0.4)
        stuck_against_wall = max(stuck_against_wall, int(pinned.sum().item()))
        worst = max(worst, float((near & (span < 0.4)).float().mean().item()))

    return {
        "map_layout": core.map_layout,
        "n_envs": int(B),
        "n_agents": int(Nb),
        "steps": int(steps),
        "max_blue_agents_pinned_to_wall_in_any_window": stuck_against_wall,
        "agents_inside_obstacle_events": inside_events,
    }


if __name__ == "__main__":
    import json

    for layout in ("map_b", "map_b_v2"):
        print(json.dumps(run(map_layout=layout), indent=2))
        print("-" * 60)
