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

GLOBAL_STATE_DIM: int = 34
GLOBAL_STATE_USED: int = 34

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
    "flag_pressure_blue",
    "flag_pressure_red",
    "home_defense_blue",
    "home_defense_red",
    "carrier_dist_home",
    "carrier_enemy_nearest_dist",
    "carrier_teammate_support",
    "mean_blue_red_dist",
    "min_blue_red_dist",
    "blue_near_enemy_flag_count",
    "red_near_enemy_flag_count",
    "blue_near_home_flag_count",
    "red_near_home_flag_count",
    "team_pairwise_distance_mean",
    "team_pairwise_distance_std",
)
assert len(GLOBAL_STATE_FIELD_NAMES) == GLOBAL_STATE_DIM, len(GLOBAL_STATE_FIELD_NAMES)

# Slices [8:12] are “territory / flag” features for optional event-based resampling (min distances + capture bits).
GLOBAL_STATE_FLAG_TERRITORY_SLICE = slice(8, 12)


def build_global_state_batch(core: "BatchedCTFCore") -> torch.Tensor:
    """
    Return (B, GLOBAL_STATE_DIM) float32 tensor on ``core.device``.

    Features (see ``GLOBAL_STATE_FIELD_NAMES``), normalized where noted:
      the first 14 entries preserve the original global summary; entries 14-18
      append score and clock pressure; entries 19-33 append observable
      geometry/pressure signals for q_phi and critic predictability.
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

    near_radius = 0.20 * diag

    def _dist_to_point(x: torch.Tensor, y: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        px = point[:, 0:1].expand_as(x)
        py = point[:, 1:2].expand_as(y)
        return torch.sqrt(torch.clamp((x - px) ** 2 + (y - py) ** 2, min=0.0))

    def _min_alive_dist_to_point(
        x: torch.Tensor,
        y: torch.Tensor,
        alive: torch.Tensor,
        point: torch.Tensor,
    ) -> torch.Tensor:
        dist = _dist_to_point(x, y, point)
        dist = torch.where(alive, dist, torch.full_like(dist, float("inf")))
        out = dist.min(dim=1).values / diag
        return torch.where(torch.isfinite(out), out, torch.ones_like(out))

    def _near_count_frac(
        x: torch.Tensor,
        y: torch.Tensor,
        alive: torch.Tensor,
        point: torch.Tensor,
    ) -> torch.Tensor:
        dist = _dist_to_point(x, y, point)
        near = alive & (dist <= near_radius)
        denom = torch.clamp(alive.to(f32).sum(dim=1), min=1.0)
        return near.to(f32).sum(dim=1) / denom

    min_red_to_blue_home = _min_alive_dist_to_point(rx, ry, ra, bh)
    min_blue_to_red_home = _min_alive_dist_to_point(bx, by, ba, rh)
    min_blue_to_blue_home = _min_alive_dist_to_point(bx, by, ba, bh)
    min_red_to_red_home = _min_alive_dist_to_point(rx, ry, ra, rh)
    flag_pressure_blue = 1.0 - min_red_to_blue_home
    flag_pressure_red = 1.0 - min_blue_to_red_home
    home_defense_blue = 1.0 - min_blue_to_blue_home
    home_defense_red = 1.0 - min_red_to_red_home

    blue_red_dist = torch.sqrt(
        torch.clamp(
            (bx[:, :, None] - rx[:, None, :]) ** 2
            + (by[:, :, None] - ry[:, None, :]) ** 2,
            min=0.0,
        )
    )
    blue_red_alive = ba[:, :, None] & ra[:, None, :]
    blue_red_dist_masked = torch.where(
        blue_red_alive,
        blue_red_dist,
        torch.full_like(blue_red_dist, float("inf")),
    )
    blue_red_count = torch.clamp(blue_red_alive.to(f32).sum(dim=(1, 2)), min=1.0)
    mean_blue_red_dist = (
        torch.where(blue_red_alive, blue_red_dist, torch.zeros_like(blue_red_dist)).sum(dim=(1, 2))
        / blue_red_count
        / diag
    )
    min_blue_red_dist = blue_red_dist_masked.amin(dim=(1, 2)) / diag
    min_blue_red_dist = torch.where(
        torch.isfinite(min_blue_red_dist),
        min_blue_red_dist,
        torch.ones_like(min_blue_red_dist),
    )

    def _carrier_feature_rows(
        team_x: torch.Tensor,
        team_y: torch.Tensor,
        team_alive: torch.Tensor,
        carrying: torch.Tensor,
        home: torch.Tensor,
        enemy_x: torch.Tensor,
        enemy_y: torch.Tensor,
        enemy_alive: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        carry_f = carrying.to(f32)
        has_carrier = carrying.any(dim=1)
        cx = (team_x * carry_f).sum(dim=1)
        cy = (team_y * carry_f).sum(dim=1)
        dist_home = torch.sqrt(torch.clamp((cx - home[:, 0]) ** 2 + (cy - home[:, 1]) ** 2, min=0.0)) / diag

        enemy_dist = torch.sqrt(
            torch.clamp((enemy_x - cx[:, None]) ** 2 + (enemy_y - cy[:, None]) ** 2, min=0.0)
        )
        enemy_dist = torch.where(enemy_alive, enemy_dist, torch.full_like(enemy_dist, float("inf")))
        nearest_enemy = enemy_dist.min(dim=1).values / diag

        teammate_mask = team_alive & (~carrying)
        teammate_dist = torch.sqrt(
            torch.clamp((team_x - cx[:, None]) ** 2 + (team_y - cy[:, None]) ** 2, min=0.0)
        )
        teammate_dist = torch.where(teammate_mask, teammate_dist, torch.full_like(teammate_dist, float("inf")))
        nearest_teammate = teammate_dist.min(dim=1).values / diag
        support = 1.0 - nearest_teammate

        dist_home = torch.where(has_carrier, dist_home, torch.ones_like(dist_home))
        nearest_enemy = torch.where(
            has_carrier & torch.isfinite(nearest_enemy),
            nearest_enemy,
            torch.ones_like(nearest_enemy),
        )
        support = torch.where(
            has_carrier & torch.isfinite(support),
            support,
            torch.zeros_like(support),
        )
        return has_carrier.to(f32), dist_home, nearest_enemy, torch.clamp(support, 0.0, 1.0)

    blue_has_carrier, blue_carrier_home, blue_carrier_enemy, blue_carrier_support = _carrier_feature_rows(
        bx, by, ba, core.blue_carrying, bh, rx, ry, ra
    )
    red_has_carrier, red_carrier_home, red_carrier_enemy, red_carrier_support = _carrier_feature_rows(
        rx, ry, ra, core.red_carrying, rh, bx, by, ba
    )
    any_carrier = (blue_has_carrier + red_has_carrier).clamp(0.0, 1.0)
    carrier_dist_home = torch.where(
        blue_has_carrier > 0.5,
        blue_carrier_home,
        torch.where(red_has_carrier > 0.5, red_carrier_home, torch.ones_like(blue_carrier_home)),
    )
    carrier_enemy_nearest_dist = torch.where(
        blue_has_carrier > 0.5,
        blue_carrier_enemy,
        torch.where(red_has_carrier > 0.5, red_carrier_enemy, torch.ones_like(blue_carrier_enemy)),
    )
    carrier_teammate_support = torch.where(
        blue_has_carrier > 0.5,
        blue_carrier_support,
        torch.where(red_has_carrier > 0.5, red_carrier_support, torch.zeros_like(blue_carrier_support)),
    ) * any_carrier

    blue_near_enemy_flag_count = _near_count_frac(bx, by, ba, rh)
    red_near_enemy_flag_count = _near_count_frac(rx, ry, ra, bh)
    blue_near_home_flag_count = _near_count_frac(bx, by, ba, bh)
    red_near_home_flag_count = _near_count_frac(rx, ry, ra, rh)

    def _within_team_pairwise(x: torch.Tensor, y: torch.Tensor, alive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = int(x.shape[1])
        if n < 2:
            zeros = torch.zeros((B,), dtype=f32, device=dev)
            return zeros, zeros
        dist = torch.sqrt(
            torch.clamp(
                (x[:, :, None] - x[:, None, :]) ** 2
                + (y[:, :, None] - y[:, None, :]) ** 2,
                min=0.0,
            )
        )
        pair_alive = alive[:, :, None] & alive[:, None, :]
        upper = torch.triu(torch.ones((n, n), dtype=torch.bool, device=dev), diagonal=1)
        mask = pair_alive & upper[None, :, :]
        count = mask.to(f32).sum(dim=(1, 2))
        sum_dist = torch.where(mask, dist, torch.zeros_like(dist)).sum(dim=(1, 2))
        mean = sum_dist / torch.clamp(count, min=1.0)
        var = (
            torch.where(mask, (dist - mean[:, None, None]) ** 2, torch.zeros_like(dist)).sum(dim=(1, 2))
            / torch.clamp(count, min=1.0)
        )
        mean = torch.where(count > 0, mean / diag, torch.zeros_like(mean))
        std = torch.where(count > 0, torch.sqrt(torch.clamp(var, min=0.0)) / diag, torch.zeros_like(var))
        return mean, std

    blue_pair_mean, blue_pair_std = _within_team_pairwise(bx, by, ba)
    red_pair_mean, red_pair_std = _within_team_pairwise(rx, ry, ra)
    team_pairwise_distance_mean = 0.5 * (blue_pair_mean + red_pair_mean)
    team_pairwise_distance_std = 0.5 * (blue_pair_std + red_pair_std)

    parts = [
        bmx, bmy, bsx, bsy,
        rmx, rmy, rsx, rsy,
        min_b_rf, min_r_bf,
        blue_flag_captured, red_flag_captured,
        mean_b_sp, mean_r_sp,
        blue_score_norm, red_score_norm, score_diff_norm,
        decision_frac, sim_time_frac,
        flag_pressure_blue, flag_pressure_red,
        home_defense_blue, home_defense_red,
        carrier_dist_home, carrier_enemy_nearest_dist, carrier_teammate_support,
        mean_blue_red_dist, min_blue_red_dist,
        blue_near_enemy_flag_count, red_near_enemy_flag_count,
        blue_near_home_flag_count, red_near_home_flag_count,
        team_pairwise_distance_mean, team_pairwise_distance_std,
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
