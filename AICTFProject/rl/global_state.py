"""
Fixed-size global-state features for the latent team-strategy CTDE stack.

Feature *order and semantics* follow *Summer Implementation Plan.docx* (IMPLEMENTATION DETAILS §3,
``global_features`` list). The policy does not consume this at execution time.

The CTDE encoder input is a compact, structured summary of:
  - team geometry
  - team dispersion
  - proximity to flags and opponents
  - flag capture status
  - motion statistics
  - score and clock pressure

The policy never consumes these features directly at execution time; they are
only for the centralized critic and the latent strategy encoder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from game_field_gpu import BatchedCTFCore

GLOBAL_STATE_DIM: int = 19
GLOBAL_STATE_USED: int = 19

# Order matches the plan’s “global summary” (team geometry + dispersion, flag proximity, captures, motion).
# The first 14 fields preserve the original plan global-summary order.
# Score and clock pressure are appended for critic/q_phi predictability.
GLOBAL_STATE_FIELD_NAMES: tuple[str, ...] = (
    "blue_mean_x",
    "blue_mean_y",
    "blue_std_x",
    "blue_std_y",
    "red_mean_x",
    "red_mean_y",
    "red_std_x",
    "red_std_y",
    "min_alive_blue_to_red_flag",
    "min_alive_red_to_blue_flag",
    "blue_flag_captured",
    "red_flag_captured",
    "mean_blue_speed",
    "mean_red_speed",
    "blue_score_norm",
    "red_score_norm",
    "score_diff_norm",
    "decision_frac",
    "sim_time_frac",
)
assert len(GLOBAL_STATE_FIELD_NAMES) == GLOBAL_STATE_DIM, len(GLOBAL_STATE_FIELD_NAMES)

# Slices [8:12] are “territory / flag” features for optional event-based resampling (min distances + capture bits).
GLOBAL_STATE_FLAG_TERRITORY_SLICE = slice(8, 12)


def build_global_state_batch(core: "BatchedCTFCore") -> torch.Tensor:
    """
    Return (B, GLOBAL_STATE_DIM) float32 tensor on ``core.device``.

    Features (see ``GLOBAL_STATE_FIELD_NAMES``), normalized where noted:
      the first 14 entries preserve the original global summary; entries 14-18
      append score and clock pressure for critic/q_phi predictability.
      0–3 blue mean/std; 4–7 red mean/std; 8–9 min alive dist to enemy flags;
      10-11 capture indicators; 12-13 mean team speeds; 14-18 score/clock pressure.
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

    blue_flag_captured = (
        core.red_carrying.any(dim=1)
        | (torch.sqrt(torch.clamp(((bf - bh) ** 2).sum(dim=1), min=0.0)) > eps)
    ).to(f32)
    red_flag_captured = (
        core.blue_carrying.any(dim=1)
        | (torch.sqrt(torch.clamp(((rf - rh) ** 2).sum(dim=1), min=0.0)) > eps)
    ).to(f32)

    w_b = ba.to(f32)
    cnt_b = torch.clamp(w_b.sum(dim=1), min=1.0)
    mean_b_sp = (core.blue_speed * w_b).sum(dim=1) / cnt_b
    w_r = ra.to(f32)
    cnt_r = torch.clamp(w_r.sum(dim=1), min=1.0)
    mean_r_sp = (core.red_speed * w_r).sum(dim=1) / cnt_r
    # Speeds are already in a bounded range in sim; light normalize
    mean_b_sp = mean_b_sp / 3.0
    mean_r_sp = mean_r_sp / 3.0

    score_den = max(1.0, float(getattr(core, "score_limit", 1)))
    blue_score_norm = torch.clamp(core.blue_score.to(f32) / score_den, 0.0, 1.0)
    red_score_norm = torch.clamp(core.red_score.to(f32) / score_den, 0.0, 1.0)
    score_diff_norm = torch.clamp(
        (core.blue_score.to(f32) - core.red_score.to(f32)) / score_den,
        -1.0,
        1.0,
    )
    decision_frac = torch.clamp(
        core.step_count.to(f32) / max(1.0, float(getattr(core, "max_steps", 1))),
        0.0,
        1.0,
    )
    sim_time_frac = torch.clamp(
        core.sim_step_count.to(f32) / max(1.0, float(getattr(core, "max_sim_steps", 1))),
        0.0,
        1.0,
    )

    parts = [
        bmx, bmy, bsx, bsy,
        rmx, rmy, rsx, rsy,
        min_b_rf, min_r_bf,
        blue_flag_captured, red_flag_captured,
        mean_b_sp, mean_r_sp,
        blue_score_norm, red_score_norm, score_diff_norm,
        decision_frac, sim_time_frac,
    ]
    used = torch.stack(parts, dim=1)
    assert used.shape[1] == GLOBAL_STATE_USED, (used.shape[1], GLOBAL_STATE_USED)
    return used.to(dtype=f32)


def coarse_game_phase_from_global_state(state: object) -> str:
    """
    Coarse game-phase label from the global state flag bits.

    Shared by E3 step telemetry in ``rl.custom_ppo`` and eval scripts (kept in sync
    with the former ``plot.eval_rollout._strategy_phase_from_global_state``).
    """
    arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if arr.size < 12:
        return "unknown"
    blue_flag_captured = bool(arr[10] > 0.5)
    red_flag_captured = bool(arr[11] > 0.5)
    if blue_flag_captured and red_flag_captured:
        return "both_flags"
    if red_flag_captured:
        return "blue_attack"
    if blue_flag_captured:
        return "blue_defense"
    return "neutral"
