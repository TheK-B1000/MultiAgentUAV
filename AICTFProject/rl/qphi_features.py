"""q_phi-only opponent/context feature blocks for staged router experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from game_field_gpu import BatchedCTFCore


C2_QPHI_CONTEXT_FEATURE_NAMES: tuple[str, ...] = (
    "c2_red_mean_speed_recent",
    "c2_red_speed_trend",
    "c2_red_flag_pressure_recent",
    "c2_red_penetration_depth_recent",
    "c2_red_attacker_fraction_recent",
    "c2_red_defender_fraction_recent",
    "c2_red_midline_pressure_recent",
    "c2_red_role_switch_rate_recent",
    "c2_red_flag_pickup_recent",
    "c2_red_capture_recent",
    "c2_red_intercept_recent",
    "c2_red_mine_pickup_recent",
    "c2_blue_flag_threat_depth_recent",
    "c2_red_carrying_current",
    "c2_blue_flag_captured_current",
    "c2_red_score_norm",
)
C2_QPHI_CONTEXT_DIM: int = len(C2_QPHI_CONTEXT_FEATURE_NAMES)


def _valid_ring(core: "BatchedCTFCore") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ring = core._red_behavior_ring
    k = int(core._red_behavior_ring_K)
    valid = (
        torch.arange(k, device=core.device)
        .unsqueeze(0)
        .lt(core._red_behavior_count.unsqueeze(1))
    )
    count = core._red_behavior_count.clamp(min=1).to(torch.float32)
    return ring, valid, count


def _ring_mean(ring: torch.Tensor, valid: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    return (ring * valid.unsqueeze(-1).to(torch.float32)).sum(dim=1) / count.unsqueeze(-1)


def _ring_first_last(ring: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b = int(ring.shape[0])
    last_idx = (valid.long().sum(dim=1) - 1).clamp(min=0)
    first = ring[:, 0, :]
    last = ring[torch.arange(b, device=ring.device), last_idx, :]
    return first, last


def build_c2_qphi_context_batch(core: "BatchedCTFCore") -> torch.Tensor:
    """Return C2 q_phi-only features with shape ``(B, C2_QPHI_CONTEXT_DIM)``.

    This side-channel is intentionally not fed to the actor or centralized critic.
    """
    dev = core.device
    f32 = torch.float32
    diag = max(1e-6, float(getattr(core, "max_dist", 1.0)))
    score_den = max(1.0, float(getattr(core, "score_limit", 1)))
    ring, valid, count = _valid_ring(core)
    mean = _ring_mean(ring, valid, count)
    first, last = _ring_first_last(ring, valid)

    # Ring columns 0..5 preserve the existing global-state behavior features.
    attacker_recent = mean[:, 0].clamp(0.0, 1.0)
    midline_recent = mean[:, 1].clamp(0.0, 1.0)
    speed_recent = (mean[:, 2] / 3.0).clamp(0.0, 2.0)
    defender_recent = mean[:, 3].clamp(0.0, 1.0)
    threat_depth_recent = (mean[:, 4] / diag).clamp(0.0, 1.5)
    role_switch_recent = mean[:, 5].clamp(0.0, 1.0)
    speed_trend = ((last[:, 2] - first[:, 2]) / 3.0).clamp(-1.0, 1.0)
    flag_pressure_recent = (1.0 - threat_depth_recent).clamp(0.0, 1.0)
    penetration_recent = attacker_recent

    def col_mean(idx: int) -> torch.Tensor:
        if int(ring.shape[-1]) <= idx:
            return torch.zeros((int(core.B),), dtype=f32, device=dev)
        return mean[:, idx].to(dtype=f32)

    red_pickup_recent = col_mean(6).clamp(0.0, 1.0)
    red_capture_recent = col_mean(7).clamp(0.0, 1.0)
    red_intercept_recent = col_mean(8).clamp(0.0, 1.0)
    red_mine_pickup_recent = col_mean(9).clamp(0.0, 1.0)

    red_carrying_current = core.red_carrying.any(dim=1).to(f32)
    blue_flag_home = core.blue_flag_home
    blue_flag_pos = core.blue_flag_pos
    blue_flag_captured = (
        red_carrying_current.bool()
        | (torch.sqrt(torch.clamp(((blue_flag_pos - blue_flag_home) ** 2).sum(dim=1), min=0.0)) > 1e-6)
    ).to(f32)
    red_score_norm = torch.clamp(core.red_score.to(f32) / score_den, 0.0, 1.0)

    return torch.stack(
        [
            speed_recent,
            speed_trend,
            flag_pressure_recent,
            penetration_recent,
            attacker_recent,
            defender_recent,
            midline_recent,
            role_switch_recent,
            red_pickup_recent,
            red_capture_recent,
            red_intercept_recent,
            red_mine_pickup_recent,
            threat_depth_recent,
            red_carrying_current,
            blue_flag_captured,
            red_score_norm,
        ],
        dim=1,
    ).to(dtype=f32, device=dev)
