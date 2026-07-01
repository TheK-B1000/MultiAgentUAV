"""Post-hoc context bucketing for latent strategy diagnostics and router losses."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from rl.global_state import GLOBAL_STATE_DIM


def carrier_progress_bucket_ids(global_state: torch.Tensor) -> torch.Tensor:
    """Bucket active carrier progress from the global-state carrier distance.

    Bucket ids:
      0 = no active flag carrier
      1 = carrier far from scoring home
      2 = carrier in midfield
      3 = carrier near scoring home
    """
    if global_state.dim() != 2:
        raise ValueError(f"global_state must be 2-D, got {tuple(global_state.shape)}")
    raw = global_state[:, :GLOBAL_STATE_DIM].float()
    if raw.shape[1] < GLOBAL_STATE_DIM:
        raw = F.pad(raw, (0, GLOBAL_STATE_DIM - int(raw.shape[1])))
    enemy_has_our_flag = raw[:, 10] > 0.5
    we_have_enemy_flag = raw[:, 11] > 0.5
    carrier_active = enemy_has_our_flag | we_have_enemy_flag
    dist_home = raw[:, 23].contiguous()
    far = torch.ones_like(dist_home, dtype=torch.long)
    mid = torch.full_like(dist_home, 2, dtype=torch.long)
    near = torch.full_like(dist_home, 3, dtype=torch.long)
    active_bucket = torch.where(
        dist_home > 0.66,
        far,
        torch.where(dist_home > 0.33, mid, near),
    )
    return torch.where(carrier_active, active_bucket, torch.zeros_like(active_bucket))


def strategy_experience_bucket_ids(context_state: torch.Tensor) -> torch.Tensor:
    """Coarse post-hoc situation buckets for diagnostics only; never used as training labels."""
    if context_state.dim() != 2:
        raise ValueError(f"context_state must be 2-D, got {tuple(context_state.shape)}")
    raw = context_state[:, :GLOBAL_STATE_DIM].float()
    if raw.shape[1] < GLOBAL_STATE_DIM:
        raw = F.pad(raw, (0, GLOBAL_STATE_DIM - int(raw.shape[1])))
    enemy_has_our_flag = (raw[:, 10] > 0.5).long()
    we_have_enemy_flag = (raw[:, 11] > 0.5).long()
    dist_edges = torch.tensor([0.20, 0.50], dtype=torch.float32, device=raw.device)
    closest_ally_to_enemy_flag = torch.bucketize(raw[:, 8].contiguous(), dist_edges).long().clamp(0, 2)
    closest_enemy_to_our_flag = torch.bucketize(raw[:, 9].contiguous(), dist_edges).long().clamp(0, 2)
    spread = torch.sqrt(torch.clamp(raw[:, 2].pow(2) + raw[:, 3].pow(2), min=0.0))
    spread_bin = (spread > 0.15).long()
    score = raw[:, 16]
    score_state = torch.where(
        score < -0.05,
        torch.zeros_like(score, dtype=torch.long),
        torch.where(score > 0.05, torch.full_like(score, 2, dtype=torch.long), torch.ones_like(score, dtype=torch.long)),
    )
    bucket = enemy_has_our_flag
    bucket = bucket * 2 + we_have_enemy_flag
    bucket = bucket * 3 + closest_ally_to_enemy_flag
    bucket = bucket * 3 + closest_enemy_to_our_flag
    bucket = bucket * 2 + spread_bin
    bucket = bucket * 3 + score_state
    return bucket.long()


def team_phase_bucket_ids(raw: torch.Tensor) -> torch.Tensor:
    """Return a coarse five-way team phase from observable global state."""
    enemy_has_our_flag = raw[:, 10] > 0.5
    we_have_enemy_flag = raw[:, 11] > 0.5
    near_enemy_flag = raw[:, 8] < 0.22
    near_own_flag = raw[:, 9] < 0.22
    enemy_pressure = raw[:, 19]
    attack_pressure = raw[:, 20]

    neutral = torch.zeros(raw.shape[0], dtype=torch.long, device=raw.device)
    attacking = torch.ones_like(neutral)
    carrying_home = torch.full_like(neutral, 2)
    defending = torch.full_like(neutral, 3)
    enemy_carrying = torch.full_like(neutral, 4)

    phase = neutral
    phase = torch.where(
        (~enemy_has_our_flag)
        & (~we_have_enemy_flag)
        & ((attack_pressure > enemy_pressure + 0.08) | near_enemy_flag),
        attacking,
        phase,
    )
    phase = torch.where(
        (~enemy_has_our_flag)
        & (~we_have_enemy_flag)
        & ((enemy_pressure > attack_pressure + 0.08) | near_own_flag),
        defending,
        phase,
    )
    phase = torch.where(enemy_has_our_flag & ~we_have_enemy_flag, enemy_carrying, phase)
    phase = torch.where(we_have_enemy_flag & ~enemy_has_our_flag, carrying_home, phase)
    phase = torch.where(enemy_has_our_flag & we_have_enemy_flag, enemy_carrying, phase)
    return phase.long()


def flag_state_bucket_ids(raw: torch.Tensor) -> torch.Tensor:
    """Encode both observable flag possession bits into [0, 3]."""
    enemy_has_our_flag = (raw[:, 10] > 0.5).long()
    we_have_enemy_flag = (raw[:, 11] > 0.5).long()
    return (enemy_has_our_flag * 2 + we_have_enemy_flag).long()


def score_pressure_bucket_ids(raw: torch.Tensor) -> torch.Tensor:
    """Encode trailing, tied, and leading score pressure into [0, 2]."""
    score_diff = raw[:, 16]
    return torch.where(
        score_diff < -0.05,
        torch.zeros_like(score_diff, dtype=torch.long),
        torch.where(
            score_diff > 0.05,
            torch.full_like(score_diff, 2, dtype=torch.long),
            torch.ones_like(score_diff, dtype=torch.long),
        ),
    ).long()


def role_phase_specialist_context_keys(
    global_state: torch.Tensor,
    *,
    include_progress: bool = True,
) -> torch.Tensor:
    """Phase/flag context key for specialist-router grouping.

    This is a battlefield-context bucket, not a role label. It mirrors the
    phase/flag concepts already logged for MI diagnostics so fixed-opponent
    runs can still ask q_phi to become decisive across CTF situations.
    """
    if global_state.dim() != 2:
        raise ValueError(f"global_state must be 2-D, got {tuple(global_state.shape)}")
    raw = global_state[:, :GLOBAL_STATE_DIM].float()
    if raw.shape[1] < GLOBAL_STATE_DIM:
        raw = F.pad(raw, (0, GLOBAL_STATE_DIM - int(raw.shape[1])))

    enemy_has_our_flag = raw[:, 10] > 0.5
    we_have_enemy_flag = raw[:, 11] > 0.5
    near_enemy_flag = raw[:, 8] < 0.22
    near_own_flag = raw[:, 9] < 0.22
    phase = team_phase_bucket_ids(raw)

    flag_state = enemy_has_our_flag.long() * 2 + we_have_enemy_flag.long()
    near_bucket = near_own_flag.long() * 2 + near_enemy_flag.long()
    key = ((phase * 4) + flag_state) * 4 + near_bucket
    if include_progress:
        key = key * 4 + carrier_progress_bucket_ids(raw)
    return key.long()


def tactical_local_context_keys(global_state: torch.Tensor) -> torch.Tensor:
    """Encode phase, both flag states, and score pressure into [0, 59]."""
    if global_state.dim() != 2:
        raise ValueError(f"global_state must be 2-D, got {tuple(global_state.shape)}")
    raw = global_state[:, :GLOBAL_STATE_DIM].float()
    if raw.shape[1] < GLOBAL_STATE_DIM:
        raw = F.pad(raw, (0, GLOBAL_STATE_DIM - int(raw.shape[1])))

    phase = team_phase_bucket_ids(raw)
    our_flag_taken = (raw[:, 10] > 0.5).long()
    enemy_flag_taken = (raw[:, 11] > 0.5).long()
    score_pressure = score_pressure_bucket_ids(raw)

    tactical_key = phase
    tactical_key = tactical_key * 2 + our_flag_taken
    tactical_key = tactical_key * 2 + enemy_flag_taken
    return (tactical_key * 3 + score_pressure).long()


def tactical_specialist_context_keys(
    global_state: torch.Tensor,
    *,
    opponent_ids: Optional[torch.Tensor],
) -> torch.Tensor:
    """Bucket phase, both flag states, score pressure, then opponent.

    The key is used only for router losses, baselines, and diagnostics. The
    decentralized actor never receives it.
    """
    tactical_key = tactical_local_context_keys(global_state)
    if opponent_ids is None:
        return tactical_key.long()
    return (
        tactical_key.long() * 16 + opponent_ids.long().clamp_min(0)
    ).long()


def specialist_context_keys_for_mode(
    *,
    mode: str,
    states: torch.Tensor,
    opponent_ids: Optional[torch.Tensor],
    bucket_ids: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    mode_s = str(mode or "opponent_bucket").strip().lower()
    if mode_s in {"role_phase", "phase_flag"}:
        return role_phase_specialist_context_keys(states, include_progress=False)
    if mode_s in {"role_phase_progress", "phase_flag_progress"}:
        return role_phase_specialist_context_keys(states, include_progress=True)
    if mode_s in {
        "role_phase_opponent",
        "phase_flag_opponent",
        "role_phase_progress_opponent",
        "phase_flag_progress_opponent",
    }:
        include_progress = "progress" in mode_s
        phase_key = role_phase_specialist_context_keys(
            states, include_progress=include_progress
        )
        if opponent_ids is None:
            return phase_key
        return phase_key * 16 + opponent_ids.long().clamp_min(0)
    if mode_s in {
        "tactical_phase_flags_score",
        "tactical_phase_flags_score_opponent",
        "phase_flags_score_opponent",
    }:
        include_opponent = mode_s != "tactical_phase_flags_score"
        return tactical_specialist_context_keys(
            states,
            opponent_ids=opponent_ids if include_opponent else None,
        )
    if opponent_ids is not None and bucket_ids is not None:
        return opponent_ids.long() * 1024 + bucket_ids.long()
    return None


def episode_bucket_baseline_keys(
    *,
    mode: str,
    states: torch.Tensor,
    opponent_ids: torch.Tensor,
    bucket_ids: torch.Tensor,
) -> torch.Tensor:
    mode_s = str(mode or "").strip().lower()
    if mode_s in {
        "tactical_context",
        "tactical_context_opponent",
        "tactical_phase_flags_score_opponent",
    }:
        return (
            bucket_ids.long().clamp(min=0, max=59) * 16
            + opponent_ids.long().clamp_min(0)
        ).long()
    from rl.custom_ppo.latent_bucket_baseline import resolve_bucket_ids

    return resolve_bucket_ids(
        mode=mode_s,
        opponent_ids=opponent_ids,
        bucket_ids=bucket_ids,
    )