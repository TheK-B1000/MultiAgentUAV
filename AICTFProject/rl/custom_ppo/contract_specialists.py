"""V6I14 z-indexed contract rewards for specialist birth.

These rewards are explicit scaffolding. They are intended for the
contract-specialist repertoire stage only, where z is assigned uniformly or
balanced and the actor is trained to make each latent own a different job.
"""

from __future__ import annotations

from typing import Any

import torch


def contract_specialist_reward(
    prev_global_state: torch.Tensor,
    next_global_state: torch.Tensor,
    z_idx: torch.Tensor,
    cfg: Any,
) -> torch.Tensor:
    """Return a bounded per-env contract bonus for the active latent.

    Contract map:
      z0: opening pressure toward the enemy flag.
      z1: home defense and pressure recovery.
      z2: friendly-carrier support.
      z3: carrier conversion progress.

    The helper reads normalized fields from ``rl.global_state``. It does not
    inspect opponent identity, oracle labels, or hindsight best-z targets.
    """
    if not bool(getattr(cfg, "latent_contract_specialist_enabled", False)):
        return torch.zeros_like(z_idx, dtype=torch.float32)

    coef = max(0.0, float(getattr(cfg, "latent_contract_specialist_coef", 0.0) or 0.0))
    if coef <= 0.0:
        return torch.zeros_like(z_idx, dtype=torch.float32)

    prev = prev_global_state.float()
    nxt = next_global_state.float()
    z = z_idx.long().reshape(-1)
    if prev.dim() != 2 or nxt.dim() != 2 or prev.shape != nxt.shape:
        raise ValueError("contract_specialist_reward expects matching (B, D) states")
    if prev.shape[1] < 34:
        raise ValueError("contract_specialist_reward requires the 34-d global state")

    variant = str(getattr(cfg, "latent_contract_specialist_variant", "base") or "base")
    raw = _sharp_contract_raw(prev, nxt) if variant == "sharp" else _base_contract_raw(prev, nxt)
    selected = raw.gather(1, z.clamp(0, raw.shape[1] - 1).unsqueeze(1)).squeeze(1)
    clip = max(0.0, float(getattr(cfg, "latent_contract_specialist_clip", 1.0) or 0.0))
    if clip > 0.0:
        selected = selected.clamp(-clip, clip)
    return selected * coef


def _base_contract_raw(prev: torch.Tensor, nxt: torch.Tensor) -> torch.Tensor:
    blue_flag_captured = nxt[:, 10].clamp(0.0, 1.0)
    red_flag_captured = nxt[:, 11].clamp(0.0, 1.0)
    decision_frac = nxt[:, 17].clamp(0.0, 1.0)
    flag_pressure_blue = nxt[:, 19].clamp(0.0, 1.0)
    flag_pressure_red = nxt[:, 20].clamp(0.0, 1.0)
    home_defense_blue = nxt[:, 21].clamp(0.0, 1.0)
    carrier_dist_home = nxt[:, 23].clamp(0.0, 1.0)
    carrier_enemy_nearest_dist = nxt[:, 24].clamp(0.0, 1.0)
    carrier_teammate_support = nxt[:, 25].clamp(0.0, 1.0)
    blue_near_enemy_flag = nxt[:, 28].clamp(0.0, 1.0)

    prev_pressure_blue = prev[:, 19].clamp(0.0, 1.0)
    prev_carrier_dist_home = prev[:, 23].clamp(0.0, 1.0)

    opening_weight = (1.0 - decision_frac / 0.25).clamp(0.0, 1.0)
    z0_opening_pressure = opening_weight * (
        0.65 * flag_pressure_red + 0.35 * blue_near_enemy_flag
    )

    defensive_context = torch.maximum(blue_flag_captured, flag_pressure_blue)
    z1_defense_recovery = defensive_context * (
        0.50 * home_defense_blue
        + 0.50 * (prev_pressure_blue - flag_pressure_blue).clamp(0.0, 1.0)
    )

    z2_carrier_support = red_flag_captured * (
        0.55 * carrier_teammate_support + 0.45 * carrier_enemy_nearest_dist
    )

    conversion_progress = (prev_carrier_dist_home - carrier_dist_home).clamp(0.0, 1.0)
    z3_conversion = red_flag_captured * (
        0.70 * conversion_progress + 0.30 * (1.0 - carrier_dist_home)
    )

    return torch.stack(
        [z0_opening_pressure, z1_defense_recovery, z2_carrier_support, z3_conversion],
        dim=1,
    )


def _sharp_contract_raw(prev: torch.Tensor, nxt: torch.Tensor) -> torch.Tensor:
    blue_flag_captured = nxt[:, 10].clamp(0.0, 1.0)
    red_flag_captured = nxt[:, 11].clamp(0.0, 1.0)
    decision_frac = nxt[:, 17].clamp(0.0, 1.0)
    flag_pressure_blue = nxt[:, 19].clamp(0.0, 1.0)
    flag_pressure_red = nxt[:, 20].clamp(0.0, 1.0)
    home_defense_blue = nxt[:, 21].clamp(0.0, 1.0)
    carrier_dist_home = nxt[:, 23].clamp(0.0, 1.0)
    carrier_enemy_nearest_dist = nxt[:, 24].clamp(0.0, 1.0)
    carrier_teammate_support = nxt[:, 25].clamp(0.0, 1.0)
    min_blue_red_dist = nxt[:, 27].clamp(0.0, 1.0)
    blue_near_enemy_flag = nxt[:, 28].clamp(0.0, 1.0)
    blue_near_home_flag = nxt[:, 30].clamp(0.0, 1.0)
    team_spread = nxt[:, 32].clamp(0.0, 1.0)
    team_spread_std = nxt[:, 33].clamp(0.0, 1.0)

    prev_pressure_blue = prev[:, 19].clamp(0.0, 1.0)
    prev_pressure_red = prev[:, 20].clamp(0.0, 1.0)
    prev_carrier_dist_home = prev[:, 23].clamp(0.0, 1.0)

    opening_weight = (1.0 - decision_frac / 0.35).clamp(0.0, 1.0)
    enemy_carrier_disruption = blue_flag_captured * (
        0.45 * home_defense_blue
        + 0.35 * (prev_pressure_blue - flag_pressure_blue).clamp(0.0, 1.0)
        + 0.20 * (1.0 - min_blue_red_dist)
    )
    z0_pressure_intercept = torch.maximum(
        opening_weight * (0.60 * flag_pressure_red + 0.40 * blue_near_enemy_flag),
        enemy_carrier_disruption,
    )

    conversion_progress = (prev_carrier_dist_home - carrier_dist_home).clamp(0.0, 1.0)
    z1_escort_conversion = red_flag_captured * (
        0.45 * carrier_teammate_support
        + 0.35 * carrier_enemy_nearest_dist
        + 0.20 * conversion_progress
    )

    z2_home_defense = torch.maximum(blue_flag_captured, flag_pressure_blue) * (
        0.45 * home_defense_blue
        + 0.30 * blue_near_home_flag
        + 0.25 * (prev_pressure_blue - flag_pressure_blue).clamp(0.0, 1.0)
    )

    pressure_gain = (flag_pressure_red - prev_pressure_red).clamp(0.0, 1.0)
    no_carrier_context = (1.0 - torch.maximum(red_flag_captured, blue_flag_captured)).clamp(0.0, 1.0)
    z3_lane_control = no_carrier_context * (
        0.40 * team_spread
        + 0.25 * team_spread_std
        + 0.20 * flag_pressure_red
        + 0.15 * pressure_gain
    )

    return torch.stack(
        [z0_pressure_intercept, z1_escort_conversion, z2_home_defense, z3_lane_control],
        dim=1,
    )


__all__ = ["contract_specialist_reward"]
