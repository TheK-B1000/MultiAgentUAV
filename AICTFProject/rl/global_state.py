"""
Fixed-size global-state features for centralized training (CTDE critic + strategy encoder).

Matches the implementation plan: summarize team geometry so ``state_dim`` is invariant
across 2v2 / 4v4 / 6v6 (no per-agent concatenation in the encoder input).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from game_field_gpu import BatchedCTFCore

# Padded vector; first GLOBAL_STATE_USED entries are meaningful, rest zeros for future use.
GLOBAL_STATE_DIM: int = 32
GLOBAL_STATE_USED: int = 18


def build_global_state_batch(core: "BatchedCTFCore") -> torch.Tensor:
    """
    Return (B, GLOBAL_STATE_DIM) float32 tensor on ``core.device``.

    Features (normalized where noted):
      blue mean x,y; blue std x,y; red mean x,y; red std x,y;
      min alive-blue dist to red flag; min alive-red dist to blue flag;
      any_blue_carrying; any_red_carrying;
      blue_score, red_score (normalized by score_limit);
      mean blue speed, mean red speed (alive-masked);
      L2 norm of (blue_flag - blue_home) and (red_flag - red_home) / field_diag;
      padding to GLOBAL_STATE_DIM.
    """
    B = int(core.B)
    Nb = int(core.Nb)
    Nr = int(core.Nr)
    dev = core.device
    f32 = torch.float32
    eps = 1e-6

    diag = float(core.max_dist)
    if diag <= eps:
        diag = 1.0
    sl = max(1, int(core.score_limit))

    bx, by = core.blue_x, core.blue_y
    rx, ry = core.red_x, core.red_y
    ba, ra = core.blue_alive, core.red_alive
    bf = core.blue_flag_pos
    rf = core.red_flag_pos
    bh = core.blue_flag_home
    rh = core.red_flag_home

    def _masked_mean_std(x: torch.Tensor, y: torch.Tensor, alive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        w = alive.to(f32)
        cnt = torch.clamp(w.sum(dim=1), min=1.0)
        mx = (x * w).sum(dim=1) / cnt
        my = (y * w).sum(dim=1) / cnt
        vx = ((x - mx[:, None]) ** 2 * w).sum(dim=1) / cnt
        vy = ((y - my[:, None]) ** 2 * w).sum(dim=1) / cnt
        sx = torch.sqrt(torch.clamp(vx, min=0.0))
        sy = torch.sqrt(torch.clamp(vy, min=0.0))
        mx = mx / diag
        my = my / diag
        sx = sx / diag
        sy = sy / diag
        return mx, my, sx, sy

    bmx, bmy, bsx, bsy = _masked_mean_std(bx, by, ba)
    rmx, rmy, rsx, rsy = _masked_mean_std(rx, ry, ra)

    # Distances to enemy flags
    rflag_x = rf[:, 0:1].expand(B, Nb)
    rflag_y = rf[:, 1:2].expand(B, Nb)
    bflag_x = bf[:, 0:1].expand(B, Nr)
    bflag_y = bf[:, 1:2].expand(B, Nr)
    db = torch.sqrt(torch.clamp((bx - rflag_x) ** 2 + (by - rflag_y) ** 2, min=0.0))
    dr = torch.sqrt(torch.clamp((rx - bflag_x) ** 2 + (ry - bflag_y) ** 2, min=0.0))
    db = torch.where(ba, db, torch.full_like(db, float("inf")))
    dr = torch.where(ra, dr, torch.full_like(dr, float("inf")))
    min_b_rf = db.min(dim=1).values / diag
    min_r_bf = dr.min(dim=1).values / diag
    min_b_rf = torch.where(torch.isfinite(min_b_rf), min_b_rf, torch.zeros_like(min_b_rf))
    min_r_bf = torch.where(torch.isfinite(min_r_bf), min_r_bf, torch.zeros_like(min_r_bf))

    carry_b = core.blue_carrying.any(dim=1).to(f32)
    carry_r = core.red_carrying.any(dim=1).to(f32)

    bs = core.blue_score.to(f32) / float(sl)
    rs = core.red_score.to(f32) / float(sl)

    w_b = ba.to(f32)
    cnt_b = torch.clamp(w_b.sum(dim=1), min=1.0)
    mean_b_sp = (core.blue_speed * w_b).sum(dim=1) / cnt_b
    w_r = ra.to(f32)
    cnt_r = torch.clamp(w_r.sum(dim=1), min=1.0)
    mean_r_sp = (core.red_speed * w_r).sum(dim=1) / cnt_r
    # Speeds are already in a bounded range in sim; light normalize
    mean_b_sp = mean_b_sp / 3.0
    mean_r_sp = mean_r_sp / 3.0

    d_home_b = torch.sqrt(((bf - bh) ** 2).sum(dim=1)) / diag
    d_home_r = torch.sqrt(((rf - rh) ** 2).sum(dim=1)) / diag

    parts = [
        bmx, bmy, bsx, bsy,
        rmx, rmy, rsx, rsy,
        min_b_rf, min_r_bf,
        carry_b, carry_r,
        bs, rs,
        mean_b_sp, mean_r_sp,
        d_home_b, d_home_r,
    ]
    used = torch.stack(parts, dim=1)
    assert used.shape[1] == GLOBAL_STATE_USED, (used.shape[1], GLOBAL_STATE_USED)
    pad = torch.zeros((B, GLOBAL_STATE_DIM - GLOBAL_STATE_USED), device=dev, dtype=f32)
    return torch.cat([used, pad], dim=1)
