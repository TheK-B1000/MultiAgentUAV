"""Owns the latent strategy z-machine for :class:`CustomPPOTrainer`.

This is the SUMMER-plan z state: the per-env current ``z``, when to resample
vs persist, episode-start recording for q_phi PPO credit, and the
episode-strategy update that consumes those records.

Why this module exists
----------------------
Before extraction the trainer mixed five different concerns: reset / per-step
sampling logic, episode-boundary outcome recording, KL-consecutive bookkeeping,
the q_phi grad-norm probe, and the actual episode-strategy PPO update. Reading
``collect_rollout`` required mentally tracking ~15 attribute names that all
started with the same prefix and were mutated from a dozen places.

This class makes the state machine one object you can read top to bottom.
The trainer still owns ``model``, ``optimizer``, ``cfg``, ``env``, and
``device``; this class reads them via ``self.trainer``.

State owned here
----------------
- ``current_z``: ``(N,)`` long, currently in-effect z per env (or ``None``
  before first reset).
- ``strategy_age``: ``(N,)`` long, steps since last z resample.
- ``needs_strategy_sample``: ``(N,)`` bool, True if next step must resample.
- ``z_kl_first_in_ep``: ``(N,)`` bool or ``None``, marks first step in
  episode for KL-consecutive masking.
- ``prev_z_logits``: ``(N, K)`` float or ``None``, previous step's z logits
  for KL-consecutive.
- ``episode_return_accum``: ``(N,)`` float, running sum of rewards within
  the in-progress episode (used as q_phi PPO target).
- ``episode_strategy_state``: ``(N, gs_dim)`` float, global state at the
  start of the current episode (q_phi training input).
- ``episode_strategy_z``, ``episode_strategy_log_prob``,
  ``episode_strategy_probs``, ``episode_strategy_bucket``,
  ``episode_strategy_has_start``: episode-start z record snapshots.
- ``rollout_strategy_episode_records``: list[dict] of completed episode
  records, drained on each rollout.
- ``episode_strategy_recorder``: :class:`EpisodeStrategyRecorder` instance
  that tracks pending/completed episode records by env id.
- ``next_strategy_episode_id``: monotonically increasing id for newly
  started strategy episodes.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Iterable, Optional

import numpy as np
import torch
from torch.distributions import Categorical
from collections import deque
import torch.nn.functional as F

from rl.ppo_core import ppo_policy_loss
from rl.behavior_telemetry import N_TELEMETRY
from rl.global_state import GLOBAL_STATE_DIM
from rl.custom_ppo.latent_value_baselines import compute_z_marginal_strategy_value
from rl.custom_ppo.csv_writers import _opponent_id_int_from_info

if TYPE_CHECKING:
    from rl.custom_ppo.trainer import CustomPPOTrainer


def _v3i3_target_from_items(
    items: list[tuple[int, float]],
    *,
    latent_k: int,
    min_count: int,
    min_distinct_z: int,
    temperature: float,
) -> Optional[np.ndarray]:
    """Compute a softmax target distribution over z from refresh records.

    ``items`` is a list of ``(z, future_return)`` tuples drawn from the
    v3i3 refresh preference buffer for a single bucket key. Returns
    ``None`` when the bucket is undersampled (``< min_count`` total or
    ``< min_distinct_z`` distinct z values observed) so callers can fall
    back to a coarser bucket. Empty z slots within an otherwise valid
    bucket are filled with the minimum sampled mean (so they aren't
    preferred over observed z).
    """
    if len(items) < int(min_count):
        return None
    per_z: list[list[float]] = [[] for _ in range(int(latent_k))]
    for z_val, fr_val in items:
        if 0 <= z_val < int(latent_k):
            per_z[z_val].append(float(fr_val))
    distinct = sum(1 for vs in per_z if vs)
    if distinct < int(min_distinct_z):
        return None
    means = np.zeros(int(latent_k), dtype=np.float32)
    sampled: list[float] = []
    for z_val in range(int(latent_k)):
        if per_z[z_val]:
            m = float(np.mean(per_z[z_val]))
            means[z_val] = m
            sampled.append(m)
    fill = float(min(sampled)) if sampled else 0.0
    for z_val in range(int(latent_k)):
        if not per_z[z_val]:
            means[z_val] = fill
    e = np.exp((means - means.max()) / float(max(1e-6, temperature)))
    return (e / e.sum()).astype(np.float32)


def _advantage_weighted_target_from_records(
    records: list[dict[str, Any]],
    *,
    latent_k: int,
    min_count: int,
    min_distinct_z: int,
    temperature: float,
    margin_threshold: float,
    soft_margin_gating: bool = False,
    use_return: bool = False,
) -> tuple[Optional[np.ndarray], dict[str, float]]:
    """Build a soft z target from per-z win-rate or return advantage evidence."""
    stats = {
        "margin": 0.0,
        "wr_spread": 0.0,
        "best_z": -1.0,
        "count": float(len(records)),
    }
    if len(records) < int(min_count):
        return None, stats
    per_z: list[list[float]] = [[] for _ in range(int(latent_k))]
    for rec in records:
        z_val = int(rec.get("z", -1))
        if 0 <= z_val < int(latent_k):
            val_key = "return" if use_return else "win_loss"
            per_z[z_val].append(float(rec.get(val_key, 0.0)))
    observed = [z for z in range(int(latent_k)) if per_z[z]]
    if len(observed) < int(min_distinct_z):
        return None, stats

    wr = np.zeros(int(latent_k), dtype=np.float32)
    sampled_wr = []
    for z_val in observed:
        mean_wr = float(np.mean(per_z[z_val]))
        wr[z_val] = mean_wr
        sampled_wr.append(mean_wr)
    fallback_wr = float(min(sampled_wr)) if sampled_wr else 0.0
    for z_val in range(int(latent_k)):
        if not per_z[z_val]:
            wr[z_val] = fallback_wr

    baseline = float(np.mean(sampled_wr)) if sampled_wr else 0.0
    advantages = wr - baseline
    observed_advantages = advantages[observed]
    best_pos = int(np.argmax(observed_advantages))
    best_z = int(observed[best_pos])

    temp = float(max(1e-6, temperature))
    e = np.exp((advantages - advantages.max()) / temp)
    target_probs = (e / e.sum()).astype(np.float32)

    if use_return:
        sorted_probs = np.sort(target_probs)
        margin = float(sorted_probs[-1] - sorted_probs[-2])
    else:
        if len(observed_advantages) > 1:
            sorted_adv = np.sort(observed_advantages)
            margin = float(sorted_adv[-1] - sorted_adv[-2])
        else:
            margin = 0.0

    wr_spread = float(max(sampled_wr) - min(sampled_wr)) if sampled_wr else 0.0
    stats.update({"margin": margin, "wr_spread": wr_spread, "best_z": float(best_z)})

    if not soft_margin_gating and margin < float(margin_threshold):
        return None, stats

    return target_probs, stats


def _router_specialist_coef_scale(
    *,
    global_step: int,
    warmup_steps: int,
    ramp_steps: int,
) -> float:
    """0 before warmup, linear ramp after, 1.0 at lock-in."""
    warmup = max(0, int(warmup_steps))
    ramp = max(0, int(ramp_steps))
    step = int(global_step)
    if step < warmup:
        return 0.0
    if ramp <= 0:
        return 1.0
    return float(max(0.0, min(1.0, (step - warmup) / float(ramp))))


def _warmup_ramp_coef_scale(
    *,
    global_step: int,
    warmup_steps: int,
    ramp_steps: int,
) -> float:
    """Default-preserving warmup/ramp scale for opt-in teacher losses."""
    warmup = max(0, int(warmup_steps))
    ramp = max(0, int(ramp_steps))
    if warmup <= 0 and ramp <= 0:
        return 1.0
    step = int(global_step)
    if step < warmup:
        return 0.0
    if ramp <= 0:
        return 1.0
    return float(max(0.0, min(1.0, (step - warmup) / float(ramp))))


def _router_specialist_loss(
    logits: torch.Tensor,
    *,
    context_keys: Optional[torch.Tensor],
    latent_k: int,
    marginal_balance_coef: float,
    conditional_entropy_min_coef: float,
    conditional_entropy_min_coef_start: float = 0.0,
    conditional_entropy_scope: str = "state",
    context_mi_coef: float,
    coef_scale: float,
    min_bucket_count: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Balanced specialization objective for q_phi.

    The loss keeps all latents alive globally with high marginal entropy, while
    pushing q_phi to be decisive within each state/context bucket. Context keys
    are used only for the loss/telemetry grouping, never as policy inputs.
    """
    zero = logits.new_zeros(())
    stats = {
        "latent_specialist_loss": zero,
        "latent_specialist_marginal_entropy": zero,
        "latent_specialist_conditional_entropy": zero,
        "latent_specialist_context_bucket_entropy": zero,
        "latent_specialist_conditional_term": zero,
        "latent_specialist_conditional_coef": zero,
        "latent_specialist_mi": zero,
        "latent_specialist_context_mi": zero,
        "latent_specialist_active_buckets": zero,
        "latent_specialist_coef_scale": logits.new_tensor(float(coef_scale)),
    }
    if logits.dim() != 2 or logits.shape[0] <= 0 or int(latent_k) <= 1:
        return zero, stats
    scale = float(max(0.0, coef_scale))
    marginal_coef = float(max(0.0, marginal_balance_coef))
    conditional_target_coef = float(max(0.0, conditional_entropy_min_coef))
    conditional_start_coef = float(
        max(0.0, conditional_entropy_min_coef_start)
    )
    conditional_coef = conditional_start_coef + scale * (
        conditional_target_coef - conditional_start_coef
    )
    mi_coef = float(max(0.0, context_mi_coef))
    scope = str(conditional_entropy_scope or "state").strip().lower()
    if scope not in {"state", "context_bucket"}:
        raise ValueError(
            "conditional_entropy_scope must be 'state' or 'context_bucket'"
        )
    if (
        scale <= 0.0
        and conditional_coef <= 0.0
    ) or (
        marginal_coef <= 0.0
        and conditional_coef <= 0.0
        and mi_coef <= 0.0
    ):
        return zero, stats

    probs = torch.softmax(logits[:, : int(latent_k)], dim=-1).clamp_min(1e-8)
    marginal_probs = probs.mean(dim=0).clamp_min(1e-8)
    marginal_entropy = -(marginal_probs * torch.log(marginal_probs)).sum()
    per_context_entropy = -(probs * torch.log(probs)).sum(dim=-1)
    conditional_entropy = per_context_entropy.mean()
    router_mi = marginal_entropy - conditional_entropy

    context_mi = zero
    context_bucket_entropy = zero
    active_buckets = zero
    if context_keys is not None and (mi_coef > 0.0 or scope == "context_bucket"):
        keys = context_keys.to(device=logits.device, dtype=torch.long)
        unique_keys_tensor, counts_tensor = torch.unique(keys, return_counts=True)
        unique_keys = unique_keys_tensor.detach().cpu().tolist()
        counts = counts_tensor.detach().cpu().tolist()
        active_masks: list[torch.Tensor] = []
        active_weights: list[torch.Tensor] = []
        bucket_entropies: list[torch.Tensor] = []
        min_count = max(1, int(min_bucket_count))
        for key, count in zip(unique_keys, counts):
            if count < min_count:
                continue
            mask = keys == key
            bucket_probs = probs[mask].mean(dim=0).clamp_min(1e-8)
            bucket_entropies.append(-(bucket_probs * torch.log(bucket_probs)).sum())
            active_masks.append(mask)
            active_weights.append(logits.new_tensor(float(count)))
        if bucket_entropies:
            active_mask = torch.stack(active_masks, dim=0).any(dim=0)
            active_total = torch.stack(active_weights).sum().clamp_min(1.0)
            weighted_entropy = torch.stack(
                [
                    weight / active_total * entropy
                    for weight, entropy in zip(active_weights, bucket_entropies)
                ]
            ).sum()
            active_marginal = probs[active_mask].mean(dim=0).clamp_min(1e-8)
            active_marginal_entropy = -(
                active_marginal * torch.log(active_marginal)
            ).sum()
            context_bucket_entropy = weighted_entropy
            context_mi = (active_marginal_entropy - weighted_entropy).clamp_min(0.0)
            active_buckets = logits.new_tensor(float(len(bucket_entropies)))

    conditional_term = (
        context_bucket_entropy if scope == "context_bucket" else conditional_entropy
    )
    loss = (
        conditional_coef * conditional_term
        - scale * marginal_coef * marginal_entropy
        - scale * mi_coef * context_mi
    )
    stats.update(
        {
            "latent_specialist_loss": loss,
            "latent_specialist_marginal_entropy": marginal_entropy,
            "latent_specialist_conditional_entropy": conditional_entropy,
            "latent_specialist_context_bucket_entropy": context_bucket_entropy,
            "latent_specialist_conditional_term": conditional_term,
            "latent_specialist_conditional_coef": logits.new_tensor(
                float(conditional_coef)
            ),
            "latent_specialist_mi": router_mi,
            "latent_specialist_context_mi": context_mi,
            "latent_specialist_active_buckets": active_buckets,
        }
    )
    return loss, stats


def _v3i3_resolve_target(
    *,
    opponent_id: int,
    event_type: int,
    flag_state_bucket: int,
    by_full: dict,
    by_oe: dict,
    by_o: dict,
    latent_k: int,
    min_count: int,
    min_distinct_z: int,
    temperature: float,
    target_cache: dict,
    carrier_progress_bucket: int = -1,
    by_oef: Optional[dict] = None,
    key_mode: str = "event_flag",
) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Hierarchical fallback target lookup.

    For event_flag_progress:
        (opp, event, flag, progress) -> (opp, event, flag) -> (opp, event) -> (opp)
    For event_flag:
        (opp, event, flag) -> (opp, event) -> (opp)

    Returns ``(target_probs_or_None, level_or_None)`` where ``level`` is
    one of ``"full"``, ``"oef"``, ``"oe"``, ``"o"`` indicating which level produced
    the target (``None`` when nothing matched). Caches resolved targets
    by ``(level, key)`` so subsequent calls with the same key are cheap.
    """
    if key_mode == "event_flag_progress":
        full_key = ("full", (int(opponent_id), int(event_type), int(flag_state_bucket), int(carrier_progress_bucket)))
        oef_key = ("oef", (int(opponent_id), int(event_type), int(flag_state_bucket)))
        oe_key = ("oe", (int(opponent_id), int(event_type)))
        o_key = ("o", (int(opponent_id),))
    else:
        full_key = ("full", (int(opponent_id), int(event_type), int(flag_state_bucket)))
        oef_key = None
        oe_key = ("oe", (int(opponent_id), int(event_type)))
        o_key = ("o", (int(opponent_id),))

    def _get(level_key: tuple, source: dict, raw_key: tuple) -> Optional[np.ndarray]:
        if level_key in target_cache:
            return target_cache[level_key]
        items = source.get(raw_key, [])
        t = _v3i3_target_from_items(
            items,
            latent_k=latent_k,
            min_count=min_count,
            min_distinct_z=min_distinct_z,
            temperature=temperature,
        )
        target_cache[level_key] = t
        return t

    t = _get(full_key, by_full, full_key[1])
    if t is not None:
        return t, "full"

    if oef_key is not None:
        t = _get(oef_key, by_oef, oef_key[1])
        if t is not None:
            return t, "oef"

    t = _get(oe_key, by_oe, oe_key[1])
    if t is not None:
        return t, "oe"

    t = _get(o_key, by_o, o_key[1])
    if t is not None:
        return t, "o"

    return None, None


def _carrier_progress_bucket_ids(global_state: torch.Tensor) -> torch.Tensor:
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


def _strategy_experience_bucket_ids(context_state: torch.Tensor) -> torch.Tensor:
    """Coarse post-hoc situation buckets for diagnostics only; never used as training labels."""
    if context_state.dim() != 2:
        raise ValueError(f"context_state must be 2-D, got {tuple(context_state.shape)}")
    raw = context_state[:, :GLOBAL_STATE_DIM].float()
    if raw.shape[1] < GLOBAL_STATE_DIM:
        raw = torch.nn.functional.pad(raw, (0, GLOBAL_STATE_DIM - int(raw.shape[1])))
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


def _team_phase_bucket_ids(raw: torch.Tensor) -> torch.Tensor:
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


def _role_phase_specialist_context_keys(
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
    phase = _team_phase_bucket_ids(raw)

    flag_state = enemy_has_our_flag.long() * 2 + we_have_enemy_flag.long()
    near_bucket = near_own_flag.long() * 2 + near_enemy_flag.long()
    key = ((phase * 4) + flag_state) * 4 + near_bucket
    if include_progress:
        key = key * 4 + _carrier_progress_bucket_ids(raw)
    return key.long()


def _tactical_local_context_keys(global_state: torch.Tensor) -> torch.Tensor:
    """Encode phase, both flag states, and score pressure into [0, 59]."""
    if global_state.dim() != 2:
        raise ValueError(f"global_state must be 2-D, got {tuple(global_state.shape)}")
    raw = global_state[:, :GLOBAL_STATE_DIM].float()
    if raw.shape[1] < GLOBAL_STATE_DIM:
        raw = F.pad(raw, (0, GLOBAL_STATE_DIM - int(raw.shape[1])))

    phase = _team_phase_bucket_ids(raw)
    our_flag_taken = (raw[:, 10] > 0.5).long()
    enemy_flag_taken = (raw[:, 11] > 0.5).long()
    score_diff = raw[:, 16]
    score_pressure = torch.where(
        score_diff < -0.05,
        torch.zeros_like(score_diff, dtype=torch.long),
        torch.where(
            score_diff > 0.05,
            torch.full_like(score_diff, 2, dtype=torch.long),
            torch.ones_like(score_diff, dtype=torch.long),
        ),
    )

    tactical_key = phase
    tactical_key = tactical_key * 2 + our_flag_taken
    tactical_key = tactical_key * 2 + enemy_flag_taken
    return (tactical_key * 3 + score_pressure).long()


def _tactical_specialist_context_keys(
    global_state: torch.Tensor,
    *,
    opponent_ids: Optional[torch.Tensor],
) -> torch.Tensor:
    """Bucket phase, both flag states, score pressure, then opponent.

    The key is used only for router losses, baselines, and diagnostics. The
    decentralized actor never receives it.
    """
    tactical_key = _tactical_local_context_keys(global_state)
    if opponent_ids is None:
        return tactical_key.long()
    return (
        tactical_key.long() * 16 + opponent_ids.long().clamp_min(0)
    ).long()


def _specialist_context_keys_for_mode(
    *,
    mode: str,
    states: torch.Tensor,
    opponent_ids: Optional[torch.Tensor],
    bucket_ids: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    mode_s = str(mode or "opponent_bucket").strip().lower()
    if mode_s in {"role_phase", "phase_flag"}:
        return _role_phase_specialist_context_keys(states, include_progress=False)
    if mode_s in {"role_phase_progress", "phase_flag_progress"}:
        return _role_phase_specialist_context_keys(states, include_progress=True)
    if mode_s in {
        "role_phase_opponent",
        "phase_flag_opponent",
        "role_phase_progress_opponent",
        "phase_flag_progress_opponent",
    }:
        include_progress = "progress" in mode_s
        phase_key = _role_phase_specialist_context_keys(
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
        return _tactical_specialist_context_keys(
            states,
            opponent_ids=opponent_ids if include_opponent else None,
        )
    if opponent_ids is not None and bucket_ids is not None:
        return opponent_ids.long() * 1024 + bucket_ids.long()
    return None


def _episode_bucket_baseline_keys(
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


class EpisodeStrategyRecorder:
    """Tracks sampled episode-level z actions for task-return PPO credit.

    q_phi is context-rich but opponent-label blind: it sees centralized temporal
    state, not explicit opponent IDs or handcrafted strategy labels. This
    recorder only preserves the exact sampled strategy action and old log-prob
    needed to credit q_phi from completed episode return.
    """

    def __init__(self) -> None:
        self.pending: dict[int, dict[str, Any]] = {}
        self.completed: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.pending.clear()
        self.completed.clear()

    def clear_completed(self) -> None:
        self.completed.clear()

    def record_start(
        self,
        *,
        env_index: int,
        episode_id: int,
        global_state_0: torch.Tensor,
        z: torch.Tensor,
        z_logprob_old: torch.Tensor,
        bucket_id: int,
        q_phi_probs: Iterable[float],
    ) -> None:
        self.pending[int(env_index)] = {
            "episode_id": int(episode_id),
            "global_state_0": global_state_0.detach().clone(),
            "z": int(z.detach().cpu().item()),
            "z_logprob_old": float(z_logprob_old.detach().cpu().item()),
            "episode_return": None,
            "episode_win": None,
            "bucket_id": int(bucket_id),
            "opponent_id": -1,
            "q_phi_probs": [float(x) for x in q_phi_probs],
        }

    def record_outcome(
        self,
        *,
        env_index: int,
        episode_return: float,
        episode_win: int,
        opponent_id: int = -1,
    ) -> Optional[dict[str, Any]]:
        """Finalize a started episode's q_phi record.

        ``opponent_id`` is the scripted-opponent integer id captured at episode
        completion time from the env's info dict. -1 means "unknown / not
        randomized" -- the BucketBaseline path treats these as a single bucket
        and falls back to the global mean when min-count is not met.
        """
        record = self.pending.pop(int(env_index), None)
        if record is None:
            return None
        record["episode_return"] = float(episode_return)
        record["episode_win"] = int(episode_win)
        record["opponent_id"] = int(opponent_id)
        self.completed.append(record)
        return record


class LatentStrategyState:
    """Per-env z-machine + episode-credit machinery for the latent strategy.

    Held by the trainer as ``self.latent_state``. The trainer remains the
    owner of ``model``, ``optimizer``, ``cfg``, ``env``, ``device``, and the
    config-derived flags (``use_latent_strategy``, ``fixed_latent_strategy``,
    ``latent_k``, ``latent_resample_every_n``, etc.).
    """

    def __init__(self, trainer: "CustomPPOTrainer") -> None:
        self.trainer = trainer
        n_envs = int(trainer.env.num_envs)
        device = trainer.device
        strategy_prob_width = max(1, int(trainer.latent_k))

        self.episode_return_accum = torch.zeros((n_envs,), dtype=torch.float32, device=device)
        self.episode_return_baseline_at_commit = torch.zeros((n_envs,), dtype=torch.float32, device=device)
        self.episode_strategy_state = torch.zeros(
            (n_envs, int(trainer.model.global_state_dim)), dtype=torch.float32, device=device
        )
        self.episode_strategy_z = torch.zeros((n_envs,), dtype=torch.long, device=device)
        self.episode_strategy_log_prob = torch.zeros((n_envs,), dtype=torch.float32, device=device)
        self.episode_strategy_probs = torch.zeros(
            (n_envs, strategy_prob_width), dtype=torch.float32, device=device
        )
        self.episode_strategy_bucket = torch.zeros((n_envs,), dtype=torch.long, device=device)
        self.episode_tactical_bucket_counts = torch.zeros(
            (n_envs, 60), dtype=torch.long, device=device
        )
        self.episode_strategy_has_start = torch.zeros((n_envs,), dtype=torch.bool, device=device)
        self.rollout_strategy_episode_records: list[dict[str, Any]] = []
        self.episode_strategy_recorder = EpisodeStrategyRecorder()
        self.next_strategy_episode_id = 0

        self.current_z: Optional[torch.Tensor] = None
        self.strategy_age = torch.zeros((n_envs,), dtype=torch.long, device=device)
        self.needs_strategy_sample = torch.ones((n_envs,), dtype=torch.bool, device=device)
        self.z_kl_first_in_ep: Optional[torch.Tensor] = None
        self.prev_z_logits: Optional[torch.Tensor] = None
        # Per-env state for the episode-credit warmup. Only meaningful when
        # ``latent_episode_strategy_ppo`` is True AND
        # ``latent_episode_strategy_warmup_decision_steps > 0``. ``steps_since_ep_start``
        # counts decision steps elapsed since the most recent episode reset (0 on the
        # step where ``needs_strategy_sample`` first fires). ``episode_strategy_committed``
        # is True once the committed (post-warmup) z + context has been snapshotted, and
        # False between episode reset and that commit moment.
        self.steps_since_ep_start = torch.zeros((n_envs,), dtype=torch.long, device=device)
        self.episode_strategy_committed = torch.zeros((n_envs,), dtype=torch.bool, device=device)
        self.first_z_sample_step = torch.full(
            (n_envs,), -1, dtype=torch.long, device=device
        )
        self.episode_forced_z = torch.zeros((n_envs,), dtype=torch.bool, device=device)
        self.episode_forced_z_id = torch.zeros((n_envs,), dtype=torch.long, device=device)
        self.episode_contrast_bucket = torch.zeros((n_envs,), dtype=torch.long, device=device)
        self.episode_behavior_sum = torch.zeros((n_envs, N_TELEMETRY), dtype=torch.float32, device=device)
        self.episode_behavior_count = torch.zeros((n_envs,), dtype=torch.long, device=device)
        self.rollout_behavior_contrast_bonus_sum = 0.0
        self.rollout_behavior_contrast_distance_sum = 0.0
        self.rollout_behavior_contrast_count = 0
        self.rollout_behavior_contrast_active_count = 0
        self.rollout_forced_z_episode_count = 0
        self.rollout_completed_episode_count = 0
        self.rollout_tactical_bucket_fallback_count = 0
        self.rollout_tactical_bucket_sample_count = 0
        self.latent_preference_buffer = deque(maxlen=20000)

        # Event refresh variables
        self.steps_since_last_refresh = torch.zeros((n_envs,), dtype=torch.long, device=device)
        self.refresh_count_this_episode = torch.zeros((n_envs,), dtype=torch.long, device=device)
        self.prev_global_state = None
        self.rollout_refresh_transitions = np.zeros(
            (max(1, int(trainer.latent_k)), max(1, int(trainer.latent_k))),
            dtype=np.float32,
        )

        # Rollout accumulator stats
        self.rollout_refresh_count = 0
        self.rollout_refresh_z_changed_count = 0
        self.rollout_refresh_reason_enemy_flag = 0
        self.rollout_refresh_reason_friendly_flag = 0
        self.rollout_refresh_reason_score_change = 0
        self.rollout_refresh_reason_near_base = 0
        self.rollout_refresh_total_steps = 0

        # v3i3 event-conditioned preference state.
        #
        # Pending refresh records (per env) accumulate during the rollout as the
        # event-refresh path fires. Each record stores everything needed to
        # finalize the per-refresh datapoint at episode end:
        #   - refresh_state (full context-state row, same input ``strategy_logits``
        #     consumes) so the v3i3 KL loss can re-forward at the refresh moment
        #   - return_at_refresh (the running episode-return accumulator at refresh
        #     time) so ``return_from_now_to_end`` can be computed from the final
        #     episode return at done time
        #   - reason_id / flag_state_bucket / prev_z / next_z / decision_step
        # On env-level done, ``record_episode_strategy_outcome`` finalizes each
        # pending record (attaches opponent_id + future_return) into
        # ``rollout_refresh_records`` (drained per rollout) AND a minimal
        # ``{opp, event, flag, z, future_return}`` entry into
        # ``refresh_preference_buffer`` (cumulative across rollouts; the v3i3
        # teacher's evidence library).
        self.pending_refresh_records: dict[int, list[dict[str, Any]]] = {
            i: [] for i in range(n_envs)
        }
        self.rollout_refresh_records: list[dict[str, Any]] = []
        self.episode_id_per_env = torch.zeros((n_envs,), dtype=torch.long, device=device)
        v3i3_buffer_size = max(
            1, int(getattr(trainer, "latent_v3i3_event_preference_buffer_size", 0) or 50_000)
        )
        self.refresh_preference_buffer: deque = deque(maxlen=v3i3_buffer_size)

    # ------------------------------------------------------------------
    # Reset / per-step sampling
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Re-init z state at the start of a rollout (or after a full env reset)."""
        trainer = self.trainer
        if not trainer.use_latent_strategy:
            return
        n_envs = int(trainer.env.num_envs)
        device = trainer.device
        z0 = trainer.fixed_latent_strategy_id if trainer.fixed_latent_strategy else 0
        self.current_z = torch.full((n_envs,), int(z0), dtype=torch.long, device=device)
        self.strategy_age = torch.zeros((n_envs,), dtype=torch.long, device=device)
        self.needs_strategy_sample = torch.full(
            (n_envs,), not trainer.fixed_latent_strategy, dtype=torch.bool, device=device
        )
        if trainer.latent_kl_consecutive > 0.0:
            self.z_kl_first_in_ep = torch.ones((n_envs,), dtype=torch.bool, device=device)
            self.prev_z_logits = None
        else:
            self.z_kl_first_in_ep = None
            self.prev_z_logits = None
        if trainer.temporal_tracker is not None:
            trainer.temporal_tracker.reset()
        trainer._last_context_state = None
        self.episode_return_accum.zero_()
        self.episode_return_baseline_at_commit.zero_()
        self.episode_strategy_has_start.zero_()
        self.episode_tactical_bucket_counts.zero_()
        self.episode_strategy_recorder.reset()
        self.steps_since_ep_start.zero_()
        self.episode_strategy_committed.zero_()
        self.first_z_sample_step.fill_(-1)
        self.episode_forced_z.zero_()
        self.episode_forced_z_id.zero_()
        self.episode_contrast_bucket.zero_()
        self.episode_behavior_sum.zero_()
        self.episode_behavior_count.zero_()
        self.steps_since_last_refresh.zero_()
        self.refresh_count_this_episode.zero_()
        self.prev_global_state = None
        self.episode_id_per_env.zero_()
        self.pending_refresh_records = {i: [] for i in range(n_envs)}
        self.rollout_refresh_records = []
        self.refresh_preference_buffer.clear()
        self.reset_event_refresh_rollout_stats()
        self.reset_behavior_contrast_rollout_stats()

    def reset_event_refresh_rollout_stats(self) -> None:
        self.rollout_refresh_count = 0
        self.rollout_refresh_z_changed_count = 0
        self.rollout_refresh_reason_enemy_flag = 0
        self.rollout_refresh_reason_friendly_flag = 0
        self.rollout_refresh_reason_score_change = 0
        self.rollout_refresh_reason_near_base = 0
        self.rollout_refresh_total_steps = 0
        self.rollout_refresh_transitions.fill(0.0)

    def clear_rollout_refresh_records(self) -> None:
        """Drain the per-rollout finalized refresh records.

        Called after the v3i3 KL loss + CSV write so the next rollout starts
        fresh. The cumulative ``refresh_preference_buffer`` is intentionally
        NOT cleared here -- it is the teacher's growing evidence library.
        """
        self.rollout_refresh_records = []

    def event_refresh_rollout_stats(self) -> dict[str, float]:
        stats = {}
        latent_k = max(1, int(self.trainer.latent_k))
        if not getattr(self.trainer, "latent_event_refresh_enabled", False):
            stats.update({
                "latent_refresh_count": 0.0,
                "latent_refresh_rate": 0.0,
                "latent_refresh_reason_enemy_flag": 0.0,
                "latent_refresh_reason_friendly_flag": 0.0,
                "latent_refresh_reason_score_change": 0.0,
                "latent_refresh_reason_near_base": 0.0,
                "latent_refresh_z_changed_rate": 0.0,
                "latent_refresh_changed_z_rate": 0.0,
                "latent_refresh_same_z_rate": 0.0,
                "latent_refresh_transition_entropy": 0.0,
            })
            for i in range(latent_k):
                for j in range(latent_k):
                    stats[f"latent_refresh_z{i}_to_z{j}"] = 0.0
            return stats
        
        count = float(self.rollout_refresh_count)
        total_steps = float(max(1, self.rollout_refresh_total_steps))
        z_changed_rate = float(self.rollout_refresh_z_changed_count) / count if count > 0 else 0.0
        same_z_rate = 1.0 - z_changed_rate if count > 0 else 0.0
        
        trans = self.rollout_refresh_transitions
        total_trans = trans.sum()
        if total_trans > 0:
            p = trans.flatten() / total_trans
            p = p[p > 0]
            transition_entropy = -float(np.sum(p * np.log(p)))
        else:
            transition_entropy = 0.0
        
        stats.update({
            "latent_refresh_count": count,
            "latent_refresh_rate": count / total_steps,
            "latent_refresh_reason_enemy_flag": float(self.rollout_refresh_reason_enemy_flag),
            "latent_refresh_reason_friendly_flag": float(self.rollout_refresh_reason_friendly_flag),
            "latent_refresh_reason_score_change": float(self.rollout_refresh_reason_score_change),
            "latent_refresh_reason_near_base": float(self.rollout_refresh_reason_near_base),
            "latent_refresh_z_changed_rate": z_changed_rate,
            "latent_refresh_changed_z_rate": z_changed_rate,
            "latent_refresh_same_z_rate": same_z_rate,
            "latent_refresh_transition_entropy": transition_entropy,
        })
        for i in range(latent_k):
            for j in range(latent_k):
                stats[f"latent_refresh_z{i}_to_z{j}"] = float(self.rollout_refresh_transitions[i, j])
        return stats

    def reset_behavior_contrast_rollout_stats(self) -> None:
        self.rollout_behavior_contrast_bonus_sum = 0.0
        self.rollout_behavior_contrast_distance_sum = 0.0
        self.rollout_behavior_contrast_count = 0
        self.rollout_behavior_contrast_active_count = 0
        self.rollout_forced_z_episode_count = 0
        self.rollout_completed_episode_count = 0
        self.rollout_tactical_bucket_fallback_count = 0
        self.rollout_tactical_bucket_sample_count = 0

    def behavior_contrast_coef(self) -> float:
        trainer = self.trainer
        base = max(0.0, float(getattr(trainer, "latent_behavior_contrast_coef", 0.0) or 0.0))
        after = max(0, int(getattr(trainer, "latent_behavior_contrast_anneal_after_steps", 0) or 0))
        if after <= 0 or int(getattr(trainer, "global_step", 0) or 0) < after:
            return base
        return max(0.0, float(getattr(trainer, "latent_behavior_contrast_anneal_to", 0.0) or 0.0))

    def store_episode_strategy_start(
        self,
        *,
        start_mask: torch.Tensor,
        global_state: torch.Tensor,
        z_idx: torch.Tensor,
        z_log_prob: torch.Tensor,
        z_logits: torch.Tensor,
    ) -> None:
        """Snapshot the exact actor-controlling z at episode start for q_phi PPO credit."""
        trainer = self.trainer
        if not trainer.latent_episode_strategy_ppo or not bool(start_mask.any().item()):
            return
        idx = torch.where(start_mask)[0]
        probs = torch.softmax(z_logits.detach(), dim=-1)
        buckets = _strategy_experience_bucket_ids(global_state.index_select(0, idx)).detach()
        self.episode_strategy_state[idx] = global_state.index_select(0, idx).detach()
        self.episode_strategy_z[idx] = z_idx.index_select(0, idx).detach()
        self.episode_strategy_log_prob[idx] = z_log_prob.index_select(0, idx).detach()
        self.episode_strategy_probs[idx, : trainer.latent_k] = probs.index_select(0, idx)
        self.episode_strategy_bucket[idx] = buckets
        self.episode_strategy_has_start[idx] = True
        for row_i, env_i in enumerate(idx.detach().cpu().tolist()):
            self.episode_strategy_recorder.record_start(
                env_index=int(env_i),
                episode_id=int(self.next_strategy_episode_id),
                global_state_0=global_state[int(env_i)],
                z=z_idx[int(env_i)],
                z_logprob_old=z_log_prob[int(env_i)],
                bucket_id=int(buckets[row_i].detach().cpu().item()),
                q_phi_probs=probs[int(env_i), : trainer.latent_k].detach().cpu().tolist(),
            )
            self.next_strategy_episode_id += 1

    def record_tactical_context_step(self, global_state: torch.Tensor) -> None:
        """Accumulate detached tactical occupancy for each active episode."""
        if global_state.dim() != 2:
            return
        keys = _tactical_local_context_keys(global_state).detach().long()
        env_ids = torch.arange(
            int(keys.shape[0]), dtype=torch.long, device=keys.device
        )
        self.episode_tactical_bucket_counts[env_ids, keys] += 1

    def representative_tactical_bucket(self, env_index: int) -> int:
        """Return the dominant meaningful tactical context for one episode."""
        counts = self.episode_tactical_bucket_counts[int(env_index)]
        if int(counts.sum().item()) <= 0:
            contrast_bucket = int(
                self.episode_contrast_bucket[int(env_index)].item()
            )
            if contrast_bucket != 0:
                return contrast_bucket
            strategy_bucket = int(
                self.episode_strategy_bucket[int(env_index)].item()
            )
            if strategy_bucket != 0:
                return strategy_bucket
            # Neutral phase, no flags taken, tied score is local bucket 1.
            return 1

        candidates = counts.clone()
        # phase=0, flags=(0,0), score=tied encodes to local key 1. Prefer a
        # context where something tactical happened whenever one exists.
        if int(candidates.sum().item() - candidates[1].item()) > 0:
            candidates[1] = 0
        return int(torch.argmax(candidates).detach().cpu().item())

    def strategy_for_step(
        self,
        global_state: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], dict[str, torch.Tensor]]:
        """Return current sparse strategy and sampling metadata for one rollout step."""
        trainer = self.trainer
        if not trainer.use_latent_strategy:
            return None, None, {}
        if self.current_z is None:
            self.reset()
        self.record_tactical_context_step(global_state)
        assert self.current_z is not None

        device = trainer.device
        if trainer.fixed_latent_strategy:
            batch = int(global_state.shape[0])
            z_idx = torch.full(
                (batch,), trainer.fixed_latent_strategy_id, dtype=torch.long, device=device
            )
            prev_z = self.current_z.clone()
            self.current_z = z_idx.clone()
            fixed_logits = torch.full(
                (batch, trainer.latent_k), -1.0e8, dtype=torch.float32, device=device
            )
            fixed_logits[:, trainer.fixed_latent_strategy_id] = 0.0
            false_mask = torch.zeros((batch,), dtype=torch.bool, device=device)
            aux = {
                "z": z_idx,
                "prev_z": prev_z,
                "z_log_prob": torch.zeros((batch,), dtype=torch.float32, device=device),
                "z_entropy": torch.zeros((batch,), dtype=torch.float32, device=device),
                "z_logits": fixed_logits,
                "z_resampled": false_mask,
                "z_forced": false_mask,
                "z_persist_mask": false_mask,
            }
            return z_idx, prev_z, aux

        episode_start_mask = self.needs_strategy_sample.clone()
        resample_mask = episode_start_mask.clone()
        if trainer.latent_resample_every_n > 0:
            resample_mask |= self.strategy_age >= trainer.latent_resample_every_n

        # v3i event refresh
        trigger_enemy_flag = torch.zeros_like(episode_start_mask)
        trigger_friendly_flag = torch.zeros_like(episode_start_mask)
        trigger_score = torch.zeros_like(episode_start_mask)
        trigger_near_base = torch.zeros_like(episode_start_mask)
        trigger_refresh = torch.zeros_like(episode_start_mask)

        curr_gs = global_state[:, :GLOBAL_STATE_DIM].float().detach()

        if getattr(trainer, "latent_event_refresh_enabled", False):
            self.rollout_refresh_total_steps += int(curr_gs.shape[0])
            if self.prev_global_state is not None:
                active_envs = ~episode_start_mask
                if bool(active_envs.any().item()):
                    prev_gs = self.prev_global_state

                    # 1. enemy captures/grabs flag (index 10)
                    trigger_enemy_flag = active_envs & (prev_gs[:, 10] <= 0.5) & (curr_gs[:, 10] > 0.5)
                    # 2. friendly captures/grabs flag (index 11)
                    trigger_friendly_flag = active_envs & (prev_gs[:, 11] <= 0.5) & (curr_gs[:, 11] > 0.5)
                    # 3. score changes (indices 14 and 15)
                    trigger_score = active_envs & ((prev_gs[:, 14] != curr_gs[:, 14]) | (prev_gs[:, 15] != curr_gs[:, 15]))

                    # 4. enemy carrier near base
                    enemy_near = (curr_gs[:, 10] > 0.5) & (curr_gs[:, 23] < 0.20)
                    enemy_near_prev = (prev_gs[:, 10] > 0.5) & (prev_gs[:, 23] < 0.20)
                    trigger_enemy_near = active_envs & enemy_near & ~enemy_near_prev

                    # 5. friendly carrier near base
                    friendly_near = (curr_gs[:, 11] > 0.5) & (curr_gs[:, 23] < 0.20)
                    friendly_near_prev = (prev_gs[:, 11] > 0.5) & (prev_gs[:, 23] < 0.20)
                    trigger_friendly_near = active_envs & friendly_near & ~friendly_near_prev

                    trigger_near_base = trigger_enemy_near | trigger_friendly_near

                    # Guardrails
                    event_refresh_allowed = (
                        (self.steps_since_last_refresh >= trainer.latent_event_refresh_min_gap_steps)
                        & (self.refresh_count_this_episode < trainer.latent_event_refresh_max_per_episode)
                    )

                    trigger_refresh = event_refresh_allowed & (
                        trigger_enemy_flag | trigger_friendly_flag | trigger_score | trigger_near_base
                    )
                    resample_mask |= trigger_refresh

        # Warmup: defer the committed z snapshot until ctx170 EMAs
        # have observed a few decision steps of opponent behavior. The provisional
        # z chosen at step 0 still drives actions during the warmup window, but we
        # force a resample at the commit step and snapshot/train on that committed
        # (context, z) pair instead. Without this guard, q_phi is fed a structurally
        # opponent-blind context (raw initial geometry + zeroed EMAs) at step 0.
        warmup = int(getattr(trainer, "latent_episode_strategy_warmup_decision_steps", 0) or 0)
        commit_now = torch.zeros_like(episode_start_mask)
        if warmup > 0:
            commit_now = (
                (self.steps_since_ep_start == warmup)
                & (~self.episode_strategy_committed)
                & (~episode_start_mask)  # never both on the same call
            )
            if bool(commit_now.any().item()):
                resample_mask = resample_mask | commit_now
                # Fix forced-z bucket alignment at warmup/commit step:
                forced_commit = commit_now & self.episode_forced_z
                if bool(forced_commit.any().item()):
                    f_idx = torch.where(forced_commit)[0]
                    self.episode_contrast_bucket[f_idx] = _strategy_experience_bucket_ids(
                        global_state.index_select(0, f_idx)
                    ).detach()

        prev_z = self.current_z.clone()
        z_idx = self.current_z.clone()
        persist_mask = resample_mask & (~self.needs_strategy_sample) & (~commit_now)

        z_logits = trainer.model.strategy_logits(global_state)
        z_dist = Categorical(logits=z_logits)
        if bool(episode_start_mask.any().item()):
            start_idx = torch.where(episode_start_mask)[0]
            self.episode_forced_z[start_idx] = False
            self.episode_behavior_sum[start_idx] = 0.0
            self.episode_behavior_count[start_idx] = 0
            self.episode_contrast_bucket[start_idx] = _strategy_experience_bucket_ids(
                global_state.index_select(0, start_idx)
            ).detach()
            forced_frac = max(
                0.0,
                min(float(getattr(trainer, "latent_forced_z_episode_frac", 0.0) or 0.0), 1.0),
            )
            contrast_on = (
                getattr(trainer, "latent_behavior_contrast", None) is not None
                and self.behavior_contrast_coef() > 0.0
                and forced_frac > 0.0
            )
            if contrast_on:
                gen = trainer.model._sampling_gen_strategy
                rand_kwargs = {
                    "dtype": torch.float32,
                    "device": device,
                }
                if gen is not None:
                    rand_kwargs["generator"] = gen
                forced_draw = torch.rand((int(start_idx.numel()),), **rand_kwargs)
                forced_mask_local = forced_draw < forced_frac
                if bool(forced_mask_local.any().item()):
                    forced_idx = start_idx[forced_mask_local]
                    uniform_logits = torch.zeros(
                        (int(forced_idx.numel()), trainer.latent_k),
                        dtype=torch.float32,
                        device=device,
                    )
                    uniform_dist = Categorical(logits=uniform_logits)
                    forced_z = trainer.model._categorical_argmax_or_sample(
                        uniform_dist,
                        deterministic=False,
                        generator=trainer.model._sampling_gen_strategy,
                    ).long()
                    self.episode_forced_z[forced_idx] = True
                    self.episode_forced_z_id[forced_idx] = forced_z

        forced_active = self.episode_forced_z.clone()
        resample_mask = resample_mask & (~forced_active)
        if bool(resample_mask.any().item()):
            idx = torch.where(resample_mask)[0]
            sampled_dist = Categorical(logits=z_logits.index_select(0, idx))
            sampled_z = trainer.model._categorical_argmax_or_sample(
                sampled_dist,
                deterministic=False,
                generator=trainer.model._sampling_gen_strategy,
            )
            
            # Telemetry for event refresh
            if getattr(trainer, "latent_event_refresh_enabled", False):
                event_resampled = trigger_refresh & resample_mask
                if bool(event_resampled.any().item()):
                    self.rollout_refresh_count += int(event_resampled.sum().item())
                    self.rollout_refresh_reason_enemy_flag += int(trigger_enemy_flag[event_resampled].sum().item())
                    self.rollout_refresh_reason_friendly_flag += int(trigger_friendly_flag[event_resampled].sum().item())
                    self.rollout_refresh_reason_score_change += int(trigger_score[event_resampled].sum().item())
                    self.rollout_refresh_reason_near_base += int(trigger_near_base[event_resampled].sum().item())

                    self.refresh_count_this_episode[event_resampled] += 1

            # v3i3 per-refresh capture: stash the (state_at_refresh, prev_z,
            # next_z, event_type, flag_state_bucket, decision_step,
            # return_at_refresh) tuple for every event-driven refresh that
            # actually fired this step. Opponent_id + future_return are filled
            # in on episode-done by ``_finalize_v3i3_refresh_records``. The
            # capture is gated on either of v3i3's two consumer features
            # (preference loss OR per-refresh CSV log) being enabled so
            # disabled runs pay zero overhead.
            v3i3_enabled = bool(
                getattr(trainer, "latent_v3i3_event_preference_enabled", False)
                or getattr(trainer, "latent_v3i3_refresh_log_enabled", False)
            )
            if v3i3_enabled and getattr(trainer, "latent_event_refresh_enabled", False):
                event_resampled = trigger_refresh & resample_mask
                if bool(event_resampled.any().item()):
                    # Primary event type per env when multiple triggers fire on
                    # the same step. Priority left-to-right:
                    #   enemy_flag (0) > friendly_flag (1) > score (2) > near_base (3)
                    event_type_t = torch.full(
                        (curr_gs.shape[0],), -1, dtype=torch.long, device=device
                    )
                    event_type_t = torch.where(
                        trigger_near_base, torch.full_like(event_type_t, 3), event_type_t
                    )
                    event_type_t = torch.where(
                        trigger_score, torch.full_like(event_type_t, 2), event_type_t
                    )
                    event_type_t = torch.where(
                        trigger_friendly_flag, torch.full_like(event_type_t, 1), event_type_t
                    )
                    event_type_t = torch.where(
                        trigger_enemy_flag, torch.full_like(event_type_t, 0), event_type_t
                    )
                    # 2*enemy_carries_our_flag + we_carry_enemy_flag, range 0..3.
                    enemy_has = (curr_gs[:, 10] > 0.5).long()
                    we_have = (curr_gs[:, 11] > 0.5).long()
                    flag_state_t = enemy_has * 2 + we_have

                    carrier_progress_bucket_t = _carrier_progress_bucket_ids(curr_gs)

                    # The actual sampled z post-resample for the event-refreshed
                    # envs. ``z_idx`` still holds prev_z at this point in the
                    # method (the bulk ``z_idx[idx] = sampled_z`` happens below);
                    # construct next_z by indexing into sampled_z which is
                    # aligned with ``idx`` rows.
                    idx_to_pos = {int(v.item()): i for i, v in enumerate(idx)}
                    for env_i_t in torch.where(event_resampled)[0]:
                        env_i = int(env_i_t.item())
                        pos = idx_to_pos.get(env_i, None)
                        if pos is None:
                            continue
                        record = {
                            "env_id": env_i,
                            "episode_id": int(self.episode_id_per_env[env_i].item()),
                            "decision_step": int(self.steps_since_ep_start[env_i].item()),
                            "reason_id": int(event_type_t[env_i].item()),
                            "prev_z": int(prev_z[env_i].item()),
                            "next_z": int(sampled_z[pos].item()),
                            "flag_state_bucket": int(flag_state_t[env_i].item()),
                            "carrier_progress_bucket": int(carrier_progress_bucket_t[env_i].item()),
                            "return_at_refresh": float(
                                self.episode_return_accum[env_i].item()
                            ),
                            "refresh_state": global_state[env_i].detach().clone(),
                        }
                        self.pending_refresh_records.setdefault(env_i, []).append(record)

            z_idx[idx] = sampled_z
            self.current_z = z_idx.clone()
            self.strategy_age[idx] = 0
            self.needs_strategy_sample[idx] = False
            self.steps_since_last_refresh[resample_mask] = 0

        if bool(forced_active.any().item()):
            z_idx[forced_active] = self.episode_forced_z_id[forced_active]
            self.current_z = z_idx.clone()
            self.strategy_age[forced_active] = 0
            self.needs_strategy_sample[forced_active] = False
            self.steps_since_last_refresh[forced_active] = 0

        # Check actual z changes for event-refreshed envs
        if getattr(trainer, "latent_event_refresh_enabled", False):
            event_resampled = trigger_refresh & resample_mask
            if bool(event_resampled.any().item()):
                actual_changes = (z_idx != prev_z) & event_resampled
                self.rollout_refresh_z_changed_count += int(actual_changes.sum().item())
                
                # Track transitions
                for env_idx in torch.where(event_resampled)[0]:
                    pz_val = int(prev_z[env_idx].item())
                    nz_val = int(z_idx[env_idx].item())
                    latent_k = int(trainer.latent_k)
                    if 0 <= pz_val < latent_k and 0 <= nz_val < latent_k:
                        self.rollout_refresh_transitions[pz_val, nz_val] += 1.0

        z_log_prob = z_dist.log_prob(z_idx)
        z_entropy = z_dist.entropy()
        # Snapshot the q_phi training (state, z, log_prob) pair:
        # - warmup == 0: legacy behavior, snapshot at episode start (step 0)
        # - warmup  > 0: snapshot at the commit step, after the EMA window
        if warmup > 0:
            snapshot_mask = commit_now
        else:
            snapshot_mask = episode_start_mask
        snapshot_mask = snapshot_mask & (~forced_active)

        # Track the warmup bookkeeping.
        if bool(snapshot_mask.any().item()):
            self.episode_strategy_committed |= snapshot_mask
            self.first_z_sample_step = torch.where(
                snapshot_mask,
                self.steps_since_ep_start,
                self.first_z_sample_step,
            )
            if warmup > 0:
                self.episode_return_baseline_at_commit = torch.where(
                    snapshot_mask,
                    self.episode_return_accum,
                    self.episode_return_baseline_at_commit,
                )
        self.store_episode_strategy_start(
            start_mask=snapshot_mask,
            global_state=global_state,
            z_idx=z_idx,
            z_log_prob=z_log_prob,
            z_logits=z_logits,
        )

        # Exclude step 0 from q_phi PPO training when warmup is active.
        # z_resampled means "eligible for q_phi training", not merely "sampled a latent"
        training_resample_mask = resample_mask.clone()
        if warmup > 0:
            training_resample_mask = training_resample_mask & (~episode_start_mask)
        training_resample_mask = training_resample_mask & (~forced_active)

        self.prev_global_state = curr_gs.clone()

        aux = {
            "z": z_idx,
            "prev_z": prev_z,
            "z_log_prob": z_log_prob,
            "z_entropy": z_entropy,
            "z_logits": z_logits,
            "z_resampled": training_resample_mask,
            "z_resampled_actual": resample_mask,
            "z_persist_mask": persist_mask,
            "z_forced": forced_active,
        }
        return z_idx, prev_z, aux

    def mark_strategy_step_done(self, dones: np.ndarray) -> None:
        """Advance per-env step counter; reset on env-level done."""
        trainer = self.trainer
        if not trainer.use_latent_strategy:
            return
        done_t = torch.as_tensor(dones, dtype=torch.bool, device=trainer.device)
        self.strategy_age += 1
        self.steps_since_ep_start += 1
        self.steps_since_last_refresh += 1
        if bool(done_t.any().item()):
            self.strategy_age[done_t] = 0
            self.needs_strategy_sample[done_t] = not trainer.fixed_latent_strategy
            self.steps_since_ep_start[done_t] = 0
            self.episode_strategy_committed[done_t] = False
            self.episode_tactical_bucket_counts[done_t] = 0
            self.first_z_sample_step[done_t] = -1
            self.episode_return_baseline_at_commit[done_t] = 0.0
            self.episode_forced_z[done_t] = False
            self.episode_forced_z_id[done_t] = 0
            self.episode_contrast_bucket[done_t] = 0
            self.episode_behavior_sum[done_t] = 0.0
            self.episode_behavior_count[done_t] = 0
            self.steps_since_last_refresh[done_t] = 0
            self.refresh_count_this_episode[done_t] = 0
            if self.prev_global_state is not None:
                self.prev_global_state[done_t] = 0.0
            self.episode_id_per_env[done_t] += 1
            # Defensive: drop any v3i3 pending refresh records that weren't
            # finalized by ``_finalize_v3i3_refresh_records`` (shouldn't
            # happen in the normal rollout flow, but avoids leaking state
            # into the next episode if a caller forgets to wire the hook).
            for env_i, done_i in enumerate(dones):
                if bool(done_i) and self.pending_refresh_records.get(env_i):
                    self.pending_refresh_records[env_i] = []

    def record_behavior_contrast_step(
        self,
        *,
        behavior_telemetry: torch.Tensor,
        z_idx: torch.Tensor,
        dones: np.ndarray,
    ) -> torch.Tensor:
        """Accumulate behavior and return a terminal contrast bonus per env."""
        trainer = self.trainer
        n_envs = int(behavior_telemetry.shape[0])
        bonus = torch.zeros((n_envs,), dtype=torch.float32, device=trainer.device)
        memory = getattr(trainer, "latent_behavior_contrast", None)
        if memory is None:
            return bonus

        self.episode_behavior_sum = self.episode_behavior_sum + behavior_telemetry.detach().float()
        self.episode_behavior_count = self.episode_behavior_count + 1
        done_t = torch.as_tensor(dones, dtype=torch.bool, device=trainer.device)
        if not bool(done_t.any().item()):
            return bonus

        team_size = int(getattr(getattr(trainer.env, "core", None), "Nb", 1) or 1)
        coef = self.behavior_contrast_coef()
        for env_i, done_i in enumerate(dones):
            if not bool(done_i):
                continue
            self.rollout_completed_episode_count += 1
            if not bool(self.episode_forced_z[env_i].detach().cpu().item()):
                continue
            self.rollout_forced_z_episode_count += 1
            count = max(1, int(self.episode_behavior_count[env_i].detach().cpu().item()))
            emb = self.episode_behavior_sum[env_i] / float(count)
            emb = memory.normalize(emb, team_size=team_size)
            result = memory.score_and_update(
                bucket_id=int(self.episode_contrast_bucket[env_i].detach().cpu().item()),
                z=int(z_idx[env_i].detach().cpu().item()),
                embedding=emb,
                coef=coef,
            )
            bonus[env_i] = result.bonus.to(device=trainer.device)
            self.rollout_behavior_contrast_bonus_sum += float(result.bonus.detach().cpu().item())
            self.rollout_behavior_contrast_distance_sum += float(result.distance)
            self.rollout_behavior_contrast_count += int(result.count)
            self.rollout_behavior_contrast_active_count += int(result.active)
        return bonus

    def behavior_contrast_rollout_stats(self) -> dict[str, float]:
        count = max(1, int(self.rollout_behavior_contrast_count))
        completed = max(1, int(self.rollout_completed_episode_count))
        forced = max(1, int(self.rollout_forced_z_episode_count))
        return {
            "latent_forced_z_episode_fraction": float(self.rollout_forced_z_episode_count) / float(completed),
            "latent_behavior_contrast_bonus_mean": float(self.rollout_behavior_contrast_bonus_sum) / float(forced),
            "latent_behavior_contrast_distance_mean": float(self.rollout_behavior_contrast_distance_sum) / float(count),
            "latent_behavior_contrast_active_frac": float(self.rollout_behavior_contrast_active_count) / float(count),
            "latent_behavior_contrast_coef": float(self.behavior_contrast_coef()),
            "latent_tactical_bucket_fallback_fraction": (
                float(self.rollout_tactical_bucket_fallback_count)
                / float(max(1, self.rollout_tactical_bucket_sample_count))
            ),
        }

    # ------------------------------------------------------------------
    # Episode outcome → completed-record buffer
    # ------------------------------------------------------------------

    def finalize_v3i3_refresh_records(
        self,
        env_index: int,
        info: dict[str, Any],
        *,
        episode_return: float,
    ) -> None:
        """Finalize all pending v3i3 refresh records for an env on episode-done.

        Each pending record gets ``opponent_id`` (read from completion info)
        and ``future_return = episode_return - return_at_refresh`` (the post-
        refresh credit signal the v3i3 teacher distills into a target z
        distribution). Finalized records flow into two sinks:

        * ``rollout_refresh_records`` -- drained per rollout. Consumed by the
          v3i3 KL loss (provides per-refresh training queries) and by the
          per-refresh CSV log writer.
        * ``refresh_preference_buffer`` -- cumulative across rollouts (capped
          by ``latent_v3i3_event_preference_buffer_size``). The teacher's
          evidence library, keyed by ``(opp, event_type, flag_state)`` with
          hierarchical fallback at lookup time.

        Always-safe to call (no-op when v3i3 is disabled and no pending
        records). Independent of ``latent_episode_strategy_ppo`` so the
        per-refresh log can be enabled even without the episode-credit path.
        """
        trainer = self.trainer
        env_i = int(env_index)
        v3i3_enabled = bool(
            getattr(trainer, "latent_v3i3_event_preference_enabled", False)
            or getattr(trainer, "latent_v3i3_refresh_log_enabled", False)
        )
        if not v3i3_enabled:
            return
        pending = self.pending_refresh_records.get(env_i, [])
        if not pending:
            return
        try:
            opponent_id = int(_opponent_id_int_from_info(trainer.cfg, info))
        except Exception:
            opponent_id = -1
        ep_return = float(episode_return)
        pref_buffer_on = bool(
            getattr(trainer, "latent_v3i3_event_preference_enabled", False)
        )
        for rec in pending:
            future_return = ep_return - float(rec["return_at_refresh"])
            finalized = dict(rec)
            finalized["opponent_id"] = opponent_id
            finalized["future_return"] = future_return
            finalized["return_from_now_to_end"] = future_return
            self.rollout_refresh_records.append(finalized)
            if pref_buffer_on:
                self.refresh_preference_buffer.append(
                    {
                        "opponent_id": opponent_id,
                        "event_type": int(rec["reason_id"]),
                        "flag_state_bucket": int(rec["flag_state_bucket"]),
                        "carrier_progress_bucket": int(rec.get("carrier_progress_bucket", -1)),
                        "z": int(rec["next_z"]),
                        "future_return": future_return,
                    }
                )
        self.pending_refresh_records[env_i] = []

    def record_episode_strategy_outcome(
        self,
        env_index: int,
        info: dict[str, Any],
        *,
        episode_return: float,
    ) -> None:
        """Snapshot a finished episode's q_phi record (state, z, log_prob, return).

        Also captures ``opponent_id`` from the completion info -- needed by the
        bucket-baseline path (v3d) which stratifies the q_phi advantage by
        opponent. Falls back to -1 when opponent info is absent (e.g. fixed-
        opponent runs); the BucketBaseline collapses unknown ids to the global
        mean automatically.
        """
        trainer = self.trainer
        if not trainer.latent_episode_strategy_ppo:
            return
        env_i = int(env_index)
        if env_i < 0 or env_i >= int(self.episode_strategy_has_start.numel()):
            return

        is_forced_z = bool(self.episode_forced_z[env_i].detach().cpu().item())
        if (
            not is_forced_z
            and not bool(
                self.episode_strategy_has_start[env_i].detach().cpu().item()
            )
        ):
            return
        used_tactical_fallback = (
            int(self.episode_tactical_bucket_counts[env_i].sum().item()) <= 0
        )
        self.rollout_tactical_bucket_sample_count += 1
        self.rollout_tactical_bucket_fallback_count += int(
            used_tactical_fallback
        )
        tactical_bucket = self.representative_tactical_bucket(env_i)
        if is_forced_z:
            try:
                opponent_id = int(_opponent_id_int_from_info(self.trainer.cfg, info))
            except Exception:
                opponent_id = -1
            
            er = info.get("episode_result") if isinstance(info.get("episode_result"), dict) else {}
            bs = int(er.get("blue_score", info.get("blue_score", 0)) or 0)
            rs = int(er.get("red_score", info.get("red_score", 0)) or 0)
            episode_win = 1 if bs > rs else 0
            
            z_val = int(self.episode_forced_z_id[env_i].detach().cpu().item())
            count = max(1, int(self.episode_behavior_count[env_i].detach().cpu().item()))
            emb = (self.episode_behavior_sum[env_i] / float(count)).detach().cpu().numpy().tolist()
            
            forced_record = {
                "context_bucket": tactical_bucket,
                "opponent": opponent_id,
                "phase_flag_state": tactical_bucket,
                "z": z_val,
                "return": float(episode_return),
                "behavior_embedding": emb,
                "win_loss": episode_win,
            }
            self.latent_preference_buffer.append(forced_record)
            return

        er = info.get("episode_result") if isinstance(info.get("episode_result"), dict) else {}
        bs = int(er.get("blue_score", info.get("blue_score", 0)) or 0)
        rs = int(er.get("red_score", info.get("red_score", 0)) or 0)
        episode_win = 1 if bs > rs else 0
        warmup = int(getattr(trainer, "latent_episode_strategy_warmup_decision_steps", 0) or 0)
        if warmup > 0:
            baseline = float(self.episode_return_baseline_at_commit[env_i].detach().cpu().item())
            adjusted_return = episode_return - baseline
        else:
            adjusted_return = episode_return

        try:
            opponent_id = int(_opponent_id_int_from_info(trainer.cfg, info))
        except Exception:
            opponent_id = -1

        record = self.episode_strategy_recorder.record_outcome(
            env_index=env_i,
            episode_return=float(adjusted_return),
            episode_win=episode_win,
            opponent_id=opponent_id,
        )
        if record is not None:
            record["bucket_id"] = tactical_bucket
            self.rollout_strategy_episode_records.append(record)
            return
        probs = self.episode_strategy_probs[env_i, : trainer.latent_k].detach().cpu().tolist()
        self.rollout_strategy_episode_records.append(
            {
                "episode_id": int(trainer.episode_stats.episodes_completed),
                "global_state_0": self.episode_strategy_state[env_i].detach().clone(),
                "z": int(self.episode_strategy_z[env_i].detach().cpu().item()),
                "z_logprob_old": float(self.episode_strategy_log_prob[env_i].detach().cpu().item()),
                "episode_return": float(adjusted_return),
                "episode_win": episode_win,
                "bucket_id": tactical_bucket,
                "opponent_id": opponent_id,
                "q_phi_probs": [float(x) for x in probs],
            }
        )

    # ------------------------------------------------------------------
    # Episode-strategy PPO update (consumes the completed-record buffer)
    # ------------------------------------------------------------------

    @staticmethod
    def empty_episode_strategy_stats(latent_k: int = 4) -> dict[str, float]:
        res = {
            "latent_preference_loss": 0.0,
            "latent_preference_active_fraction": 0.0,
            "latent_preference_buffer_size": 0.0,
            "latent_preference_num_active_buckets": 0.0,
            "latent_preference_target_entropy": 0.0,
            "latent_awrd_loss": 0.0,
            "latent_awrd_coef_scale": 0.0,
            "latent_awrd_active_fraction": 0.0,
            "latent_awrd_active_buckets": 0.0,
            "latent_awrd_buffer_size": 0.0,
            "latent_awrd_target_entropy": 0.0,
            "latent_awrd_margin_mean": 0.0,
            "latent_awrd_wr_spread_mean": 0.0,
            "latent_awrd_best_z_mean": -1.0,
            "latent_awrd_effective_coef_mean": 0.0,
            "latent_awrd_best_z_match_rate": 0.0,
            "latent_specialist_loss": 0.0,
            "latent_specialist_marginal_entropy": 0.0,
            "latent_specialist_conditional_entropy": 0.0,
            "latent_specialist_context_bucket_entropy": 0.0,
            "latent_specialist_mi": 0.0,
            "latent_specialist_context_mi": 0.0,
            "latent_specialist_active_buckets": 0.0,
            "latent_specialist_coef_scale": 0.0,
            "latent_specialist_rollout_samples": 0.0,
            "latent_episode_pg_loss": 0.0,
            "latent_episode_v_loss": 0.0,
            "latent_episode_entropy": 0.0,
            "latent_episode_adv_mean": 0.0,
            "latent_episode_adv_std": 0.0,
            "latent_episode_return_mean": 0.0,
            "latent_episode_return_std": 0.0,
            "latent_episode_ratio_mean": 0.0,
            "latent_episode_ratio_max": 0.0,
            "latent_episode_ratio_min": 0.0,
            "latent_episode_ratio_std": 0.0,
            "latent_episode_approx_kl": 0.0,
            "latent_episode_clip_fraction": 0.0,
            "latent_episode_count": 0.0,
            "latent_episode_loss": 0.0,
            "strategy_entropy_resample_mean": 0.0,
            "qphi_margin_resample_mean": 0.0,
            "episode_credit_grad_norm": 0.0,
            "episode_credit_adv_mean": 0.0,
            "episode_credit_adv_std": 0.0,
            # v3d bucket-baseline telemetry. Zero when bucket baseline is OFF.
            "bucket_baseline_count": 0.0,
            "bucket_baseline_fallback_frac": 0.0,
            "bucket_baseline_var_reduction": 1.0,
            "bucket_baseline_global_mean": 0.0,
            "bucket_baseline_raw_return_std": 0.0,
            "bucket_baseline_adv_std": 0.0,
            "latent_usage_balance_loss": 0.0,
            "latent_usage_balance_kl": 0.0,
            "latent_q_phi_train_active": 0.0,
        }
        for opp_name in ["op5", "op6"]:
            res[f"latent_pref_{opp_name}_loss"] = 0.0
            res[f"latent_pref_{opp_name}_active_fraction"] = 0.0
            res[f"latent_pref_{opp_name}_target_entropy"] = 0.0
            res[f"latent_pref_{opp_name}_best_z"] = -1.0
            res[f"latent_pref_{opp_name}_buffer_count"] = 0.0
            res[f"latent_pref_{opp_name}_active_buckets"] = 0.0
            for z in range(latent_k):
                res[f"latent_pref_{opp_name}_target_z{z}"] = 0.0
        # v3i3 event-conditioned preference telemetry. Zero when disabled.
        res.update(
            {
                "latent_v3i3_event_pref_loss": 0.0,
                "latent_v3i3_event_pref_active_fraction": 0.0,
                "latent_v3i3_event_pref_active_buckets": 0.0,
                "latent_v3i3_event_pref_active_records": 0.0,
                "latent_v3i3_event_pref_buffer_size": 0.0,
                "latent_v3i3_event_pref_target_entropy": 0.0,
                "latent_v3i3_event_pref_fallback_full": 0.0,
                "latent_v3i3_event_pref_fallback_oef": 0.0,
                "latent_v3i3_event_pref_fallback_oe": 0.0,
                "latent_v3i3_event_pref_fallback_o": 0.0,
                "latent_v3i3_event_pref_rollout_records": 0.0,
            }
        )
        return res

    def episode_strategy_training_batch(self) -> Optional[dict[str, torch.Tensor]]:
        trainer = self.trainer
        if (
            not trainer.latent_episode_strategy_ppo
            or trainer.fixed_latent_strategy
            or trainer.model.episode_strategy_value_head is None
        ):
            return None
        records = list(self.rollout_strategy_episode_records)
        if not records:
            return None
        device = trainer.device
        states = torch.stack([r["global_state_0"].detach().float() for r in records], dim=0).to(device)
        z = torch.as_tensor([int(r["z"]) for r in records], dtype=torch.long, device=device)
        old_log_prob = torch.as_tensor(
            [float(r["z_logprob_old"]) for r in records], dtype=torch.float32, device=device
        )
        episode_returns = torch.as_tensor(
            [float(r["episode_return"]) for r in records], dtype=torch.float32, device=device
        )
        # Bucket keys for v3d. Each is shape (N_eps,) long, on the trainer
        # device. ``-1`` slots are pre-v3d records or fixed-opponent runs and
        # are handled as a degenerate "unknown" bucket by BucketBaseline.
        opponent_ids = torch.as_tensor(
            [int(r.get("opponent_id", -1)) for r in records],
            dtype=torch.long,
            device=device,
        )
        bucket_ids = torch.as_tensor(
            [int(r.get("bucket_id", -1)) for r in records],
            dtype=torch.long,
            device=device,
        )
        return {
            "states": states,
            "z": z,
            "old_log_prob": old_log_prob,
            "episode_returns": episode_returns,
            "opponent_ids": opponent_ids,
            "bucket_ids": bucket_ids,
        }

    def apply_episode_strategy_ppo(self, *, latent_lam_h: float) -> dict[str, float]:
        """Run inner-epoch PPO update(s) on q_phi using completed episode records.

        With ``latent_episode_strategy_n_epochs == 1`` (legacy v3/v3b behavior),
        this is a single backward step per rollout -- effectively a one-shot
        REINFORCE-style update because the PPO ratio starts at exactly 1.0 (new
        log_prob is computed from the same weights that produced old_log_prob).
        Across a 1M-step run that's only ~15 update cycles, which cannot move
        q_phi off uniform at the shared optimizer's actor-tuned LR.

        With ``n_epochs > 1``, we run N PPO inner epochs over the same completed
        episode batch -- the same pattern the actor's main PPO loop uses. After
        the first epoch's optimizer step, subsequent epochs recompute
        new_log_prob from the *updated* logits, so the PPO ratio drifts away
        from 1.0 and the clipped policy gradient does meaningful work.

        When ``trainer.latent_router_optimizer`` is set (via
        ``latent_episode_strategy_lr``), this dedicated AdamW steps only the
        strategy_encoder + episode_strategy_value_head params -- at a higher
        LR than the shared optimizer can afford for the actor.
        """
        trainer = self.trainer
        stats = self.empty_episode_strategy_stats(trainer.latent_k)
        batch = self.episode_strategy_training_batch()
        if batch is None:
            return stats
        states = batch["states"]
        z = batch["z"]
        old_log_prob = batch["old_log_prob"]
        episode_returns = batch["episode_returns"]
        opponent_ids = batch.get("opponent_ids")
        bucket_ids = batch.get("bucket_ids")
        stats["latent_episode_count"] = float(episode_returns.numel())
        train_after = max(
            0, int(getattr(trainer, "latent_q_phi_train_after_steps", 0) or 0)
        )
        if train_after > 0 and int(getattr(trainer, "global_step", 0) or 0) < train_after:
            return stats
        stats["latent_q_phi_train_active"] = 1.0

        # v3d bucket-baseline path: when ``latent_q_phi_bucket_baseline`` is
        # set, replace the V-marginal baseline with the per-bucket empirical
        # mean of episode returns. Computed ONCE per rollout (the EMA + min-
        # count fallback already smooth across rollouts), then re-used across
        # all inner epochs since the baseline depends only on returns, not on
        # the strategy_encoder being updated.
        bucket_baseline_vector: Optional[torch.Tensor] = None
        bucket_baseline_helper = getattr(trainer, "latent_bucket_baseline", None)
        bucket_mode = getattr(trainer, "latent_q_phi_bucket_baseline", None)
        if (
            bucket_baseline_helper is not None
            and bucket_mode is not None
            and opponent_ids is not None
            and bucket_ids is not None
        ):
            keys = _episode_bucket_baseline_keys(
                mode=str(bucket_mode),
                states=states,
                opponent_ids=opponent_ids,
                bucket_ids=bucket_ids,
            )
            bucket_baseline_vector = bucket_baseline_helper.update_and_compute(
                episode_returns.detach(), keys.detach()
            ).detach()

        # Counterfactual Latent Preference precomputation
        pref_coef = float(getattr(trainer, "latent_preference_coef", 0.0) or 0.0)
        B = states.shape[0]
        batch_target_probs = torch.zeros((B, trainer.latent_k), dtype=torch.float32, device=trainer.device)
        batch_pref_mask = torch.zeros((B,), dtype=torch.bool, device=trainer.device)

        active_buckets_count = 0
        target_entropy_sum = 0.0
        unique_keys = set()
        key_to_target_probs = {}

        if pref_coef > 0.0 and len(self.latent_preference_buffer) > 0 and opponent_ids is not None and bucket_ids is not None:
            batch_keys = (opponent_ids * 256 + bucket_ids).detach().cpu().numpy().tolist()
            unique_keys = set(batch_keys)
            
            # Group buffer records by key
            buffer_by_key = {}
            for r in self.latent_preference_buffer:
                k = int(r["opponent"] * 256 + r["context_bucket"])
                if k not in buffer_by_key:
                    buffer_by_key[k] = []
                buffer_by_key[k].append(r)
            
            min_bucket_count = int(getattr(trainer, "latent_preference_min_bucket_count", 8) or 8)
            min_distinct_z = int(getattr(trainer, "latent_preference_min_distinct_z", 2) or 2)
            temperature = float(getattr(trainer, "latent_preference_temperature", 0.75) or 0.75)
            
            key_to_target_probs = {}
            for k in unique_keys:
                matching = buffer_by_key.get(int(k), [])
                distinct_zs_in_matching = set(r["z"] for r in matching)
                if len(matching) < min_bucket_count or len(distinct_zs_in_matching) < min_distinct_z:
                    key_to_target_probs[k] = None
                else:
                    active_buckets_count += 1
                    returns_for_z = {z_idx: [] for z_idx in range(trainer.latent_k)}
                    for r in matching:
                        returns_for_z[r["z"]].append(r["return"])
                    
                    avg_return_by_z = {}
                    for z_idx in range(trainer.latent_k):
                        if len(returns_for_z[z_idx]) > 0:
                            avg_return_by_z[z_idx] = sum(returns_for_z[z_idx]) / len(returns_for_z[z_idx])
                    
                    sampled_avgs = [avg_return_by_z[z_idx] for z_idx in range(trainer.latent_k) if z_idx in avg_return_by_z]
                    fallback_val = min(sampled_avgs) if len(sampled_avgs) > 0 else 0.0
                    
                    for z_idx in range(trainer.latent_k):
                        if z_idx not in avg_return_by_z:
                            avg_return_by_z[z_idx] = fallback_val
                    
                    avg_returns = np.array([avg_return_by_z[z_idx] for z_idx in range(trainer.latent_k)], dtype=np.float32)
                    exp_returns = np.exp((avg_returns - np.max(avg_returns)) / temperature)
                    target_prob = exp_returns / np.sum(exp_returns)
                    key_to_target_probs[k] = target_prob
            
            for i, k in enumerate(batch_keys):
                target = key_to_target_probs.get(k)
                if target is not None:
                    batch_target_probs[i] = torch.as_tensor(target, dtype=torch.float32, device=trainer.device)
                    batch_pref_mask[i] = True
                    # Target entropy computation: -sum(p * log(p))
                    entropy = -np.sum(target * np.log(target + 1e-12))
                    target_entropy_sum += float(entropy)

        # v3i7 advantage-weighted router distillation. This consumes the same
        # forced-z evidence library as the legacy preference path, but uses
        # win-rate or return advantage by z and only fires when a bucket has a clear best-z
        # margin. It teaches q_phi to trust discovered winning z choices without
        # adding entropy pressure or semantic role labels.
        awrd_enabled = bool(getattr(trainer, "latent_awrd_enabled", False))
        awrd_coef_scale = _warmup_ramp_coef_scale(
            global_step=int(getattr(trainer, "global_step", 0) or 0),
            warmup_steps=int(getattr(trainer, "latent_awrd_warmup_steps", 0) or 0),
            ramp_steps=int(getattr(trainer, "latent_awrd_ramp_steps", 0) or 0),
        )
        awrd_coef = (
            float(getattr(trainer, "latent_awrd_coef", 0.0) or 0.0) * awrd_coef_scale
        )
        awrd_soft_margin = bool(getattr(trainer, "latent_awrd_soft_margin_gating", False))
        awrd_use_return = awrd_soft_margin
        batch_awrd_target_probs = torch.zeros(
            (B, trainer.latent_k), dtype=torch.float32, device=trainer.device
        )
        batch_awrd_mask = torch.zeros((B,), dtype=torch.bool, device=trainer.device)
        batch_awrd_coefs = torch.zeros((B,), dtype=torch.float32, device=trainer.device)
        awrd_active_buckets = 0
        awrd_target_entropy_sum = 0.0
        awrd_margin_sum = 0.0
        awrd_wr_spread_sum = 0.0
        awrd_best_z_sum = 0.0
        awrd_best_z_matches = 0.0
        awrd_effective_coef_sum = 0.0
        awrd_key_stats: dict[int, dict[str, float]] = {}
        if (
            awrd_enabled
            and awrd_coef > 0.0
            and len(self.latent_preference_buffer) > 0
            and opponent_ids is not None
            and bucket_ids is not None
        ):
            batch_awrd_keys = (opponent_ids * 256 + bucket_ids).detach().cpu().numpy().tolist()
            awrd_buffer_by_key: dict[int, list[dict[str, Any]]] = {}
            for rec in self.latent_preference_buffer:
                key = int(rec["opponent"] * 256 + rec["context_bucket"])
                awrd_buffer_by_key.setdefault(key, []).append(rec)
            awrd_min_count = int(getattr(trainer, "latent_awrd_min_bucket_count", 8) or 8)
            awrd_min_distinct = int(getattr(trainer, "latent_awrd_min_distinct_z", 2) or 2)
            awrd_temp = float(getattr(trainer, "latent_awrd_temperature", 0.35) or 0.35)
            awrd_threshold = float(
                getattr(trainer, "latent_awrd_margin_threshold", 0.15) or 0.15
            )
            awrd_key_to_target: dict[int, Optional[np.ndarray]] = {}
            for key in set(batch_awrd_keys):
                target, key_stats = _advantage_weighted_target_from_records(
                    awrd_buffer_by_key.get(int(key), []),
                    latent_k=int(trainer.latent_k),
                    min_count=awrd_min_count,
                    min_distinct_z=awrd_min_distinct,
                    temperature=awrd_temp,
                    margin_threshold=awrd_threshold,
                    soft_margin_gating=awrd_soft_margin,
                    use_return=awrd_use_return,
                )
                awrd_key_to_target[int(key)] = target
                awrd_key_stats[int(key)] = key_stats
                if target is not None:
                    awrd_active_buckets += 1
            for i, key in enumerate(batch_awrd_keys):
                target = awrd_key_to_target.get(int(key))
                if target is None:
                    continue
                batch_awrd_target_probs[i] = torch.as_tensor(
                    target, dtype=torch.float32, device=trainer.device
                )
                batch_awrd_mask[i] = True
                awrd_target_entropy_sum += float(-np.sum(target * np.log(target + 1e-12)))
                key_stats = awrd_key_stats.get(int(key), {})
                awrd_margin_sum += float(key_stats.get("margin", 0.0))
                awrd_wr_spread_sum += float(key_stats.get("wr_spread", 0.0))
                awrd_best_z_sum += float(key_stats.get("best_z", -1.0))
                
                # Match rate telemetry
                z_picked = int(z[i].item())
                best_z = int(key_stats.get("best_z", -1))
                if z_picked == best_z:
                    awrd_best_z_matches += 1.0
                    
                if awrd_soft_margin:
                    cur_awrd_coef = awrd_coef
                    if trainer.global_step >= 700_000:
                        cur_awrd_coef *= 1.5
                    margin = float(key_stats.get("margin", 0.0))
                    scale = float(getattr(trainer, "latent_awrd_margin_scale", 3.0) or 3.0)
                    min_margin = float(getattr(trainer, "latent_awrd_min_margin", 0.08) or 0.08)
                    eff_coef = cur_awrd_coef * (1.0 + scale * margin)
                    if margin < min_margin:
                        eff_coef = cur_awrd_coef * 0.25
                    batch_awrd_coefs[i] = eff_coef
                    awrd_effective_coef_sum += eff_coef

        # v3i3 event-conditioned preference precomputation (once per rollout).
        # Builds a (B_r, K) target table over the rollout's finalized refresh
        # records using hierarchical fallback over the cumulative preference
        # buffer. Independent of the legacy ``latent_preference_*`` path.
        v3i3_coef = float(
            getattr(trainer, "latent_v3i3_event_preference_coef", 0.0) or 0.0
        )
        v3i3_warmup = int(
            getattr(trainer, "latent_v3i3_event_preference_warmup_steps", 0) or 0
        )
        v3i3_enabled = bool(
            getattr(trainer, "latent_v3i3_event_preference_enabled", False)
        )
        v3i3_records = list(self.rollout_refresh_records)
        v3i3_active = (
            v3i3_enabled
            and v3i3_coef > 0.0
            and len(self.refresh_preference_buffer) > 0
            and len(v3i3_records) > 0
            and (
                v3i3_warmup <= 0
                or int(getattr(trainer, "global_step", 0) or 0) >= v3i3_warmup
            )
        )
        v3i3_refresh_states_t: Optional[torch.Tensor] = None
        v3i3_target_probs_t: Optional[torch.Tensor] = None
        v3i3_mask_t: Optional[torch.Tensor] = None
        v3i3_active_buckets = 0
        v3i3_active_records_count = 0
        v3i3_target_entropy_sum = 0.0
        v3i3_fallback_counts = {"full": 0, "oef": 0, "oe": 0, "o": 0}
        if v3i3_active:
            v3i3_refresh_states_t = torch.stack(
                [r["refresh_state"].detach().float() for r in v3i3_records], dim=0
            ).to(trainer.device)
            by_full: dict = {}
            by_oef: dict = {}
            by_oe: dict = {}
            by_o: dict = {}
            normalize = bool(
                getattr(
                    trainer, "latent_v3i3_event_preference_normalize", False
                )
            )
            if normalize:
                baselines: dict = {}
                counts: dict = {}
                for r in self.refresh_preference_buffer:
                    if trainer.latent_event_preference_key_mode == "event_flag_progress":
                        k = (int(r["opponent_id"]), int(r["event_type"]), int(r["flag_state_bucket"]), int(r.get("carrier_progress_bucket", -1)))
                    else:
                        k = (int(r["opponent_id"]), int(r["event_type"]), int(r["flag_state_bucket"]))
                    baselines[k] = baselines.get(k, 0.0) + float(r["future_return"])
                    counts[k] = counts.get(k, 0) + 1
                for k in baselines:
                    baselines[k] /= float(counts[k])
            for r in self.refresh_preference_buffer:
                opp_b = int(r["opponent_id"])
                ev_b = int(r["event_type"])
                fl_b = int(r["flag_state_bucket"])
                pr_b = int(r.get("carrier_progress_bucket", -1))
                ret_val = float(r["future_return"])
                if normalize:
                    if trainer.latent_event_preference_key_mode == "event_flag_progress":
                        k_full = (opp_b, ev_b, fl_b, pr_b)
                    else:
                        k_full = (opp_b, ev_b, fl_b)
                    ret_val -= baselines.get(k_full, 0.0)
                pair = (int(r["z"]), ret_val)
                if trainer.latent_event_preference_key_mode == "event_flag_progress":
                    by_full.setdefault((opp_b, ev_b, fl_b, pr_b), []).append(pair)
                    by_oef.setdefault((opp_b, ev_b, fl_b), []).append(pair)
                else:
                    by_full.setdefault((opp_b, ev_b, fl_b), []).append(pair)
                by_oe.setdefault((opp_b, ev_b), []).append(pair)
                by_o.setdefault((opp_b,), []).append(pair)
            min_count = int(
                getattr(
                    trainer, "latent_v3i3_event_preference_min_bucket_count", 4
                )
                or 4
            )
            min_distinct = int(
                getattr(
                    trainer, "latent_v3i3_event_preference_min_distinct_z", 2
                )
                or 2
            )
            temperature = float(
                getattr(
                    trainer, "latent_v3i3_event_preference_temperature", 0.75
                )
                or 0.75
            )
            K = int(trainer.latent_k)
            target_arr = np.full(
                (len(v3i3_records), K), 1.0 / float(K), dtype=np.float32
            )
            mask_arr = np.zeros((len(v3i3_records),), dtype=bool)
            target_cache: dict = {}
            active_keys: set = set()
            for i, r in enumerate(v3i3_records):
                t, level = _v3i3_resolve_target(
                    opponent_id=int(r["opponent_id"]),
                    event_type=int(r["reason_id"]),
                    flag_state_bucket=int(r["flag_state_bucket"]),
                    carrier_progress_bucket=int(r.get("carrier_progress_bucket", -1)),
                    by_full=by_full,
                    by_oef=by_oef,
                    by_oe=by_oe,
                    by_o=by_o,
                    latent_k=K,
                    min_count=min_count,
                    min_distinct_z=min_distinct,
                    temperature=temperature,
                    target_cache=target_cache,
                    key_mode=trainer.latent_event_preference_key_mode,
                )
                if t is not None and level is not None:
                    target_arr[i] = t
                    mask_arr[i] = True
                    v3i3_active_records_count += 1
                    v3i3_target_entropy_sum += float(
                        -(t * np.log(t + 1e-12)).sum()
                    )
                    v3i3_fallback_counts[level] = (
                        v3i3_fallback_counts[level] + 1
                    )
                    if level == "full":
                        if trainer.latent_event_preference_key_mode == "event_flag_progress":
                            active_keys.add(
                                (
                                    "full",
                                    int(r["opponent_id"]),
                                    int(r["reason_id"]),
                                    int(r["flag_state_bucket"]),
                                    int(r.get("carrier_progress_bucket", -1)),
                                )
                            )
                        else:
                            active_keys.add(
                                (
                                    "full",
                                    int(r["opponent_id"]),
                                    int(r["reason_id"]),
                                    int(r["flag_state_bucket"]),
                                )
                            )
                    elif level == "oef":
                        active_keys.add(
                            (
                                "oef",
                                int(r["opponent_id"]),
                                int(r["reason_id"]),
                                int(r["flag_state_bucket"]),
                            )
                        )
                    elif level == "oe":
                        active_keys.add(
                            ("oe", int(r["opponent_id"]), int(r["reason_id"]))
                        )
                    else:
                        active_keys.add(("o", int(r["opponent_id"])))
            v3i3_target_probs_t = torch.as_tensor(
                target_arr, dtype=torch.float32, device=trainer.device
            )
            v3i3_mask_t = torch.as_tensor(
                mask_arr, dtype=torch.bool, device=trainer.device
            )
            v3i3_active_buckets = len(active_keys)

        n_inner_epochs = max(
            1, int(getattr(trainer, "latent_episode_strategy_n_epochs", 1) or 1)
        )
        router_optimizer = (
            getattr(trainer, "latent_router_optimizer", None) or trainer.optimizer
        )
        # Only clip the router's own params when using the dedicated optimizer;
        # under the shared path the legacy full-model scope is fine because
        # non-router params have zero gradients in this backward.
        if getattr(trainer, "latent_router_optimizer", None) is not None:
            clip_params: list[torch.nn.Parameter] = []
            for group in trainer.latent_router_optimizer.param_groups:
                clip_params.extend(group["params"])
        else:
            clip_params = list(trainer.model.parameters())

        specialist_enabled = bool(
            getattr(trainer, "latent_specialist_router_enabled", False)
        ) and not bool(
            getattr(trainer, "latent_specialist_use_rollout_states", False)
        )
        specialist_warmup_steps = int(
            getattr(trainer, "latent_specialist_warmup_steps", 0) or 0
        )
        specialist_scale = _router_specialist_coef_scale(
            global_step=int(getattr(trainer, "global_step", 0) or 0),
            warmup_steps=specialist_warmup_steps,
            ramp_steps=int(getattr(trainer, "latent_specialist_ramp_steps", 1) or 0),
        )
        specialist_conditional_start = (
            float(
                getattr(
                    trainer,
                    "latent_conditional_entropy_min_coef_start",
                    0.0,
                )
                or 0.0
            )
            if int(getattr(trainer, "global_step", 0) or 0)
            >= specialist_warmup_steps
            else 0.0
        )
        specialist_context_keys: Optional[torch.Tensor] = None
        specialist_context_keys = _specialist_context_keys_for_mode(
            mode=str(
                getattr(
                    trainer,
                    "latent_specialist_context_key_mode",
                    "opponent_bucket",
                )
                or "opponent_bucket"
            ),
            states=states,
            opponent_ids=opponent_ids,
            bucket_ids=bucket_ids,
        )

        pg_loss = torch.zeros((), dtype=torch.float32, device=trainer.device)
        v_loss = torch.zeros((), dtype=torch.float32, device=trainer.device)
        loss = torch.zeros((), dtype=torch.float32, device=trainer.device)
        z_entropy = torch.zeros((), dtype=torch.float32, device=trainer.device)
        adv = torch.zeros((1,), dtype=torch.float32, device=trainer.device)
        ppo_stats: dict[str, torch.Tensor] = {
            "ratio": torch.ones((1,), dtype=torch.float32, device=trainer.device),
            "approx_kl": torch.zeros((), dtype=torch.float32, device=trainer.device),
            "clip_fraction": torch.zeros((), dtype=torch.float32, device=trainer.device),
        }
        logits = trainer.model.strategy_logits(states)
        episode_credit_grad_norm = 0.0
        usage_loss = torch.zeros((), dtype=torch.float32, device=trainer.device)
        usage_kl = torch.zeros((), dtype=torch.float32, device=trainer.device)
        specialist_loss = torch.zeros((), dtype=torch.float32, device=trainer.device)
        specialist_stats_t: dict[str, torch.Tensor] = {
            k: torch.zeros((), dtype=torch.float32, device=trainer.device)
            for k in (
                "latent_specialist_loss",
                "latent_specialist_marginal_entropy",
                "latent_specialist_conditional_entropy",
                "latent_specialist_context_bucket_entropy",
                "latent_specialist_conditional_term",
                "latent_specialist_conditional_coef",
                "latent_specialist_mi",
                "latent_specialist_context_mi",
                "latent_specialist_active_buckets",
                "latent_specialist_coef_scale",
            )
        }

        for _ in range(n_inner_epochs):
            logits = trainer.model.strategy_logits(states)
            dist = Categorical(logits=logits)
            new_log_prob = dist.log_prob(z)
            v_z = trainer.model.episode_strategy_value(states, z)

            # q_phi advantage baseline. Three modes, in priority order:
            #
            #   v3d (bucket_baseline_vector is not None):
            #     adv = R - mean(R | bucket(s)) -- empirical per-bucket mean,
            #     EMA-smoothed across rollouts, min-count fallback to global
            #     mean. Variance-reduction by stratification; bypasses V
            #     entirely, so off-policy z calibration of V no longer
            #     bottlenecks q_phi's gradient.
            #
            #   v3b/v3c (latent_q_phi_marginal_baseline=True, bucket off):
            #     adv = R - mean_k V(s, z_k) -- AAC marginal-over-V baseline.
            #     Detached helper. Removes the "V(s, z_picked) eats the signal"
            #     pathology of legacy mode but still depends on V being well-
            #     calibrated for off-policy z, which it often isn't.
            #
            #   Legacy default (both off):
            #     adv = R - V(s, z_picked) -- the centralized critic absorbs
            #     E[R | s, z] before q_phi sees the gradient. Mostly within-z
            #     noise; documented here for completeness, do not use.
            #
            # All three paths produce detached baselines so the value head's
            # gradient route is exclusively through ``v_loss``.
            if bucket_baseline_vector is not None:
                v_baseline = bucket_baseline_vector
            elif getattr(trainer.cfg, "latent_q_phi_marginal_baseline", False):
                v_baseline = compute_z_marginal_strategy_value(
                    trainer.model, states, trainer.latent_k, policy_weighted=False
                )
            else:
                v_baseline = v_z.detach()

            adv = episode_returns - v_baseline
            if trainer.latent_episode_strategy_return_norm and adv.numel() > 1:
                if bucket_baseline_vector is not None and bucket_mode is not None:
                    keys = _episode_bucket_baseline_keys(
                        mode=str(bucket_mode),
                        states=states,
                        opponent_ids=opponent_ids,
                        bucket_ids=bucket_ids,
                    )
                    normalized_adv = torch.zeros_like(adv)
                    unique_keys_tensor, counts_tensor = torch.unique(keys, return_counts=True)
                    unique_keys = unique_keys_tensor.detach().cpu().tolist()
                    counts = counts_tensor.detach().cpu().tolist()
                    for k, count in zip(unique_keys, counts):
                        mask = (keys == k)
                        if count > 1:
                            sub_adv = adv[mask]
                            normalized_adv[mask] = (sub_adv - sub_adv.mean()) / (sub_adv.std(unbiased=False) + 1e-8)
                        else:
                            normalized_adv[mask] = adv[mask]
                    adv = normalized_adv
                else:
                    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

            pg_loss, ppo_stats = ppo_policy_loss(
                new_log_prob,
                old_log_prob,
                adv.detach(),
                trainer.latent_episode_strategy_clip_eps,
            )
            v_loss = 0.5 * (episode_returns - v_z).pow(2).mean()
            z_entropy = dist.entropy().mean()
            h_goal = str(
                getattr(trainer.cfg, "latent_entropy_objective", "maximize") or "maximize"
            ).lower()
            if h_goal == "none" or latent_lam_h <= 0.0:
                entropy_term = torch.zeros((), dtype=torch.float32, device=trainer.device)
            elif h_goal == "minimize":
                entropy_term = float(latent_lam_h) * z_entropy
            else:
                entropy_term = -float(latent_lam_h) * z_entropy
            usage_coef = max(0.0, float(getattr(trainer, "latent_usage_balance_coef", 0.0) or 0.0))
            if usage_coef > 0.0 and logits.shape[0] > 0:
                p_bar = torch.softmax(logits, dim=-1).mean(dim=0).clamp_min(1e-8)
                usage_kl = (
                    p_bar * (torch.log(p_bar) + torch.log(p_bar.new_tensor(float(trainer.latent_k))))
                ).sum()
                usage_loss = usage_coef * usage_kl
            else:
                usage_kl = torch.zeros((), dtype=torch.float32, device=trainer.device)
                usage_loss = torch.zeros((), dtype=torch.float32, device=trainer.device)
            if specialist_enabled:
                specialist_loss, specialist_stats_t = _router_specialist_loss(
                    logits,
                    context_keys=specialist_context_keys,
                    latent_k=int(trainer.latent_k),
                    marginal_balance_coef=float(
                        getattr(trainer, "latent_marginal_balance_coef", 0.0) or 0.0
                    ),
                    conditional_entropy_min_coef=float(
                        getattr(trainer, "latent_conditional_entropy_min_coef", 0.0)
                        or 0.0
                    ),
                    conditional_entropy_min_coef_start=specialist_conditional_start,
                    conditional_entropy_scope=str(
                        getattr(
                            trainer,
                            "latent_specialist_conditional_entropy_scope",
                            "state",
                        )
                        or "state"
                    ),
                    context_mi_coef=float(
                        getattr(trainer, "latent_context_mi_coef", 0.0) or 0.0
                    ),
                    coef_scale=specialist_scale,
                    min_bucket_count=int(
                        getattr(trainer, "latent_specialist_min_bucket_count", 2) or 2
                    ),
                )
            else:
                specialist_loss = torch.zeros((), dtype=torch.float32, device=trainer.device)
            pref_loss = torch.zeros((), dtype=torch.float32, device=trainer.device)
            pref_loss_scaled = torch.zeros((), dtype=torch.float32, device=trainer.device)
            commit_loss = torch.zeros((), dtype=torch.float32, device=trainer.device)
            awrd_loss = torch.zeros((), dtype=torch.float32, device=trainer.device)
            awrd_loss_scaled = torch.zeros((), dtype=torch.float32, device=trainer.device)
            if pref_coef > 0.0 and bool(batch_pref_mask.any().item()):
                valid_logits = logits[batch_pref_mask]
                valid_targets = batch_target_probs[batch_pref_mask]
                log_probs = torch.log_softmax(valid_logits, dim=-1)
                
                # Compute target confidence: 1.0 - target_entropy / log(K)
                target_probs_clamped = valid_targets.clamp_min(1e-8)
                target_entropy_eps = -(valid_targets * torch.log(target_probs_clamped)).sum(dim=-1)
                target_confidence = 1.0 - target_entropy_eps / math.log(trainer.latent_k)
                target_confidence = target_confidence.clamp(0.0, 1.0)
                
                confidence_scale = float(getattr(trainer, "latent_preference_confidence_scale", 2.0) or 2.0)
                commit_coef = float(getattr(trainer, "latent_preference_commit_coef", 0.0) or 0.0)
                
                # effective preference coefficient per episode: base_pref_coef * (1.0 + confidence_scale * target_confidence)
                effective_coef_eps = pref_coef * (1.0 + confidence_scale * target_confidence)
                
                # Compute KL divergence per episode
                kl_per_episode = F.kl_div(
                    log_probs,
                    valid_targets,
                    reduction="none"
                ).sum(dim=-1)
                
                opponent_balanced = getattr(trainer.cfg, "latent_preference_opponent_balanced", False) and opponent_ids is not None
                if opponent_balanced:
                    valid_opps = opponent_ids[batch_pref_mask]
                    unique_opps = torch.unique(valid_opps).detach().cpu().tolist()
                else:
                    unique_opps = []

                # Raw KL loss for telemetry
                if opponent_balanced:
                    opponent_losses = []
                    for opp_id in unique_opps:
                        opp_mask = (valid_opps == opp_id)
                        opp_kl = kl_per_episode[opp_mask]
                        if opp_kl.numel() > 0:
                            opponent_losses.append(opp_kl.mean())
                    if len(opponent_losses) > 0:
                        pref_loss = torch.stack(opponent_losses).mean()
                else:
                    pref_loss = kl_per_episode.mean()
                
                # Scaled preference loss applied to loss
                weighted_kl_per_episode = effective_coef_eps * kl_per_episode
                if opponent_balanced:
                    opponent_weighted_losses = []
                    for opp_id in unique_opps:
                        opp_mask = (valid_opps == opp_id)
                        opp_weighted_kl = weighted_kl_per_episode[opp_mask]
                        if opp_weighted_kl.numel() > 0:
                            opponent_weighted_losses.append(opp_weighted_kl.mean())
                    if len(opponent_weighted_losses) > 0:
                        pref_loss_scaled = torch.stack(opponent_weighted_losses).mean()
                else:
                    pref_loss_scaled = weighted_kl_per_episode.mean()
                
                # Confidence-weighted entropy commitment loss
                commit_type = str(getattr(trainer.cfg, "commitment_type", "confidence_weighted_entropy") or "confidence_weighted_entropy")
                if commit_type == "confidence_weighted_entropy" and commit_coef > 0.0:
                    valid_q_probs = torch.softmax(valid_logits, dim=-1)
                    q_entropy_eps = -(valid_q_probs * torch.log(valid_q_probs + 1e-8)).sum(dim=-1)
                    commit_loss_eps = target_confidence * q_entropy_eps
                    
                    if opponent_balanced:
                        opponent_commit_losses = []
                        for opp_id in unique_opps:
                            opp_mask = (valid_opps == opp_id)
                            opp_commit = commit_loss_eps[opp_mask]
                            if opp_commit.numel() > 0:
                                opponent_commit_losses.append(opp_commit.mean())
                        if len(opponent_commit_losses) > 0:
                            commit_loss = commit_coef * torch.stack(opponent_commit_losses).mean()
                    else:
                        commit_loss = commit_coef * commit_loss_eps.mean()

            if awrd_coef > 0.0 and bool(batch_awrd_mask.any().item()):
                awrd_logits = logits[batch_awrd_mask]
                awrd_targets = batch_awrd_target_probs[batch_awrd_mask]
                awrd_log_probs = torch.log_softmax(awrd_logits, dim=-1)
                awrd_kl = F.kl_div(
                    awrd_log_probs, awrd_targets, reduction="none"
                ).sum(dim=-1)
                awrd_loss = awrd_kl.mean()
                if awrd_soft_margin:
                    valid_coefs = batch_awrd_coefs[batch_awrd_mask]
                    awrd_loss_scaled = (valid_coefs * awrd_kl).mean()
                else:
                    awrd_scale = float(getattr(trainer, "latent_awrd_margin_scale", 2.0) or 2.0)
                    active_count = max(1, int(batch_awrd_mask.sum().item()))
                    margin_mean = float(awrd_margin_sum / active_count)
                    awrd_loss_scaled = awrd_coef * (1.0 + awrd_scale * margin_mean) * awrd_loss

            # v3i3 event-conditioned preference loss. Re-forwards
            # ``strategy_logits`` at the refresh-moment states and pulls
            # ``q_phi(z | state_at_refresh)`` toward the bucketed target
            # distribution. Gradient flows through the strategy encoder.
            v3i3_pref_loss = torch.zeros((), dtype=torch.float32, device=trainer.device)
            v3i3_pref_loss_scaled = torch.zeros((), dtype=torch.float32, device=trainer.device)
            if (
                v3i3_active
                and v3i3_refresh_states_t is not None
                and v3i3_mask_t is not None
                and v3i3_target_probs_t is not None
                and bool(v3i3_mask_t.any().item())
            ):
                v3i3_logits = trainer.model.strategy_logits(v3i3_refresh_states_t)
                valid_logits_v3i3 = v3i3_logits[v3i3_mask_t]
                valid_targets_v3i3 = v3i3_target_probs_t[v3i3_mask_t]
                v3i3_log_probs = torch.log_softmax(valid_logits_v3i3, dim=-1)
                v3i3_kl = F.kl_div(
                    v3i3_log_probs, valid_targets_v3i3, reduction="none"
                ).sum(dim=-1)
                v3i3_pref_loss = v3i3_kl.mean()
                v3i3_pref_loss_scaled = v3i3_coef * v3i3_pref_loss

            loss = trainer.latent_episode_strategy_coef * (
                pg_loss + trainer.latent_episode_strategy_value_coef * v_loss
            ) + entropy_term + usage_loss + specialist_loss + pref_loss_scaled + commit_loss + awrd_loss_scaled + v3i3_pref_loss_scaled

            router_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            episode_credit_grad_norm = self.strategy_encoder_grad_norm()
            torch.nn.utils.clip_grad_norm_(clip_params, float(trainer.cfg.max_grad_norm))
            router_optimizer.step()

        ratio = ppo_stats["ratio"].detach().float()
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            chosen_probs = probs.gather(dim=-1, index=z.unsqueeze(-1)).squeeze(-1)
            margin_resample = chosen_probs - (1.0 / trainer.latent_k)
            qphi_margin_resample_mean = float(margin_resample.mean().detach().cpu().item())
            strategy_entropy_resample_mean = float(z_entropy.detach().cpu().item())

            stats.update(
                {
                    "latent_episode_pg_loss": float(pg_loss.detach().cpu().item()),
                    "latent_episode_v_loss": float(v_loss.detach().cpu().item()),
                    "latent_episode_entropy": float(z_entropy.detach().cpu().item()),
                    "latent_episode_adv_mean": float(adv.detach().mean().cpu().item()),
                    "latent_episode_adv_std": float(
                        adv.detach().std(unbiased=False).cpu().item()
                    ) if adv.numel() > 1 else 0.0,
                    "latent_episode_return_mean": float(episode_returns.detach().mean().cpu().item()),
                    "latent_episode_return_std": float(
                        episode_returns.detach().std(unbiased=False).cpu().item()
                    ) if episode_returns.numel() > 1 else 0.0,
                    "latent_episode_ratio_mean": float(ratio.mean().cpu().item()),
                    "latent_episode_ratio_max": float(ratio.max().cpu().item()),
                    "latent_episode_ratio_min": float(ratio.min().cpu().item()),
                    "latent_episode_ratio_std": float(ratio.std(unbiased=False).cpu().item()) if ratio.numel() > 1 else 0.0,
                    "latent_episode_approx_kl": float(ppo_stats["approx_kl"].detach().cpu().item()),
                    "latent_episode_clip_fraction": float(ppo_stats["clip_fraction"].detach().cpu().item()),
                    "latent_episode_count": float(episode_returns.numel()),
                    "latent_episode_loss": float(loss.detach().cpu().item()),
                    "strategy_entropy_resample_mean": strategy_entropy_resample_mean,
                    "qphi_margin_resample_mean": qphi_margin_resample_mean,
                    "episode_credit_grad_norm": episode_credit_grad_norm,
                    "episode_credit_adv_mean": float(adv.detach().mean().cpu().item()),
                    "episode_credit_adv_std": float(
                        adv.detach().std(unbiased=False).cpu().item()
                    ) if adv.numel() > 1 else 0.0,
                }
            )

            # v3d bucket-baseline telemetry. ``last_stats`` reflects the SINGLE
            # update_and_compute call made at the top of this rollout (outside
            # the inner-epoch loop) -- the baseline math runs once per rollout,
            # not once per inner epoch.
            if bucket_baseline_vector is not None and bucket_baseline_helper is not None:
                bs = bucket_baseline_helper.last_stats
                stats.update(
                    {
                        "bucket_baseline_count": float(bs.get("bucket_count", 0)),
                        "bucket_baseline_fallback_frac": float(bs.get("fallback_fraction", 0.0)),
                        "bucket_baseline_var_reduction": float(bs.get("variance_reduction_ratio", 1.0)),
                        "bucket_baseline_global_mean": float(bs.get("global_mean", 0.0)),
                        "bucket_baseline_raw_return_std": float(bs.get("raw_return_std", 0.0)),
                        "bucket_baseline_adv_std": float(bs.get("adv_std", 0.0)),
                    }
                )
            stats["latent_usage_balance_loss"] = float(usage_loss.detach().cpu().item())
            stats["latent_usage_balance_kl"] = float(usage_kl.detach().cpu().item())
            for key, value in specialist_stats_t.items():
                stats[key] = float(value.detach().cpu().item())
            # v3i3 event-conditioned preference telemetry. ``last_stats``
            # captures the FINAL inner-epoch's loss tensor; the active
            # masks / bucket counts / fallback breakdown are precomputed
            # once per rollout (above the inner loop).
            stats["latent_v3i3_event_pref_loss"] = float(
                v3i3_pref_loss.detach().cpu().item()
            )
            stats["latent_v3i3_event_pref_active_fraction"] = (
                float(v3i3_mask_t.float().mean().cpu().item())
                if v3i3_mask_t is not None and v3i3_mask_t.numel() > 0
                else 0.0
            )
            stats["latent_v3i3_event_pref_active_buckets"] = float(v3i3_active_buckets)
            stats["latent_v3i3_event_pref_active_records"] = float(
                v3i3_active_records_count
            )
            stats["latent_v3i3_event_pref_buffer_size"] = float(
                len(self.refresh_preference_buffer)
            )
            stats["latent_v3i3_event_pref_target_entropy"] = (
                float(v3i3_target_entropy_sum / max(1, v3i3_active_records_count))
                if v3i3_active_records_count > 0
                else 0.0
            )
            stats["latent_v3i3_event_pref_fallback_full"] = float(
                v3i3_fallback_counts["full"]
            )
            stats["latent_v3i3_event_pref_fallback_oef"] = float(
                v3i3_fallback_counts["oef"]
            )
            stats["latent_v3i3_event_pref_fallback_oe"] = float(
                v3i3_fallback_counts["oe"]
            )
            stats["latent_v3i3_event_pref_fallback_o"] = float(
                v3i3_fallback_counts["o"]
            )
            stats["latent_v3i3_event_pref_rollout_records"] = float(len(v3i3_records))
            stats["latent_preference_loss"] = float(pref_loss.detach().cpu().item())
            stats["latent_preference_active_fraction"] = float(batch_pref_mask.float().mean().cpu().item())
            stats["latent_preference_buffer_size"] = float(len(self.latent_preference_buffer))
            stats["latent_preference_num_active_buckets"] = float(active_buckets_count)
            valid_count = int(batch_pref_mask.sum().item())
            stats["latent_preference_target_entropy"] = float(target_entropy_sum / max(1, valid_count)) if valid_count > 0 else 0.0
            awrd_valid_count = int(batch_awrd_mask.sum().item())
            stats["latent_awrd_loss"] = float(awrd_loss.detach().cpu().item())
            stats["latent_awrd_coef_scale"] = float(awrd_coef_scale)
            stats["latent_awrd_active_fraction"] = (
                float(batch_awrd_mask.float().mean().cpu().item())
                if batch_awrd_mask.numel() > 0
                else 0.0
            )
            stats["latent_awrd_active_buckets"] = float(awrd_active_buckets)
            stats["latent_awrd_buffer_size"] = float(len(self.latent_preference_buffer))
            stats["latent_awrd_target_entropy"] = (
                float(awrd_target_entropy_sum / max(1, awrd_valid_count))
                if awrd_valid_count > 0
                else 0.0
            )
            stats["latent_awrd_margin_mean"] = (
                float(awrd_margin_sum / max(1, awrd_valid_count))
                if awrd_valid_count > 0
                else 0.0
            )
            stats["latent_awrd_wr_spread_mean"] = (
                float(awrd_wr_spread_sum / max(1, awrd_valid_count))
                if awrd_valid_count > 0
                else 0.0
            )
            stats["latent_awrd_best_z_mean"] = (
                float(awrd_best_z_sum / max(1, awrd_valid_count))
                if awrd_valid_count > 0
                else -1.0
            )
            stats["latent_awrd_effective_coef_mean"] = (
                float(awrd_effective_coef_sum / max(1, awrd_valid_count))
                if awrd_valid_count > 0
                else 0.0
            )
            stats["latent_awrd_best_z_match_rate"] = (
                float(awrd_best_z_matches / max(1, awrd_valid_count))
                if awrd_valid_count > 0
                else 0.0
            )

            # --- Opponent specific preference target telemetry ---
            log_opponent_targets = bool(getattr(trainer.cfg, "latent_preference_log_opponent_targets", False))
            
            # Always track buffer counts as requested
            for opp_name, opp_id in [("op5", 4), ("op6", 5)]:
                stats[f"latent_pref_{opp_name}_buffer_count"] = float(sum(1 for r in self.latent_preference_buffer if r["opponent"] == opp_id))
                
            if log_opponent_targets and opponent_ids is not None:
                # 1. Compute elementwise KL values per episode in the batch (for logging)
                if batch_pref_mask.any():
                    valid_logits = logits[batch_pref_mask]
                    valid_targets = batch_target_probs[batch_pref_mask]
                    valid_log_probs = torch.log_softmax(valid_logits, dim=-1)
                    kl_per_episode = F.kl_div(valid_log_probs, valid_targets, reduction="none").sum(dim=-1)
                    valid_opps = opponent_ids[batch_pref_mask]
                else:
                    kl_per_episode = None
                    valid_opps = None
                    
                for opp_name, opp_id in [("op5", 4), ("op6", 5)]:
                    opp_mask = (opponent_ids == opp_id)
                    opp_episodes_count = int(opp_mask.sum().item())
                    opp_active_mask = opp_mask & batch_pref_mask
                    opp_active_count = int(opp_active_mask.sum().item())
                    
                    if opp_episodes_count > 0:
                        stats[f"latent_pref_{opp_name}_active_fraction"] = float(opp_active_count) / opp_episodes_count
                    else:
                        stats[f"latent_pref_{opp_name}_active_fraction"] = 0.0
                        
                    opp_keys_in_batch = [k for k in unique_keys if (k // 256) == opp_id]
                    stats[f"latent_pref_{opp_name}_active_buckets"] = float(sum(1 for k in opp_keys_in_batch if key_to_target_probs.get(k) is not None))
                    
                    if opp_active_count > 0 and kl_per_episode is not None and valid_opps is not None:
                        opp_valid_mask = (valid_opps == opp_id)
                        opp_loss = float(kl_per_episode[opp_valid_mask].mean().item())
                        stats[f"latent_pref_{opp_name}_loss"] = opp_loss
                        
                        opp_valid_targets = valid_targets[opp_valid_mask]
                        entropy_per_episode = -(opp_valid_targets * torch.log(opp_valid_targets + 1e-12)).sum(dim=-1)
                        stats[f"latent_pref_{opp_name}_target_entropy"] = float(entropy_per_episode.mean().item())
                        
                        opp_mean_targets = opp_valid_targets.mean(dim=0)
                        for z_idx in range(trainer.latent_k):
                            stats[f"latent_pref_{opp_name}_target_z{z_idx}"] = float(opp_mean_targets[z_idx].item())
                        stats[f"latent_pref_{opp_name}_best_z"] = float(opp_mean_targets.argmax().item())
                    else:
                        stats[f"latent_pref_{opp_name}_loss"] = 0.0
                        stats[f"latent_pref_{opp_name}_target_entropy"] = 0.0
                        stats[f"latent_pref_{opp_name}_best_z"] = -1.0
                        for z_idx in range(trainer.latent_k):
                            stats[f"latent_pref_{opp_name}_target_z{z_idx}"] = 0.0
        return stats

    def apply_rollout_specialist_router(self, buffer: Any) -> dict[str, float]:
        """Train q_phi specialization on tactical states observed in rollout."""
        trainer = self.trainer
        stats = {
            "latent_specialist_loss": 0.0,
            "latent_specialist_marginal_entropy": 0.0,
            "latent_specialist_conditional_entropy": 0.0,
            "latent_specialist_context_bucket_entropy": 0.0,
            "latent_specialist_conditional_term": 0.0,
            "latent_specialist_conditional_coef": 0.0,
            "latent_specialist_mi": 0.0,
            "latent_specialist_context_mi": 0.0,
            "latent_specialist_active_buckets": 0.0,
            "latent_specialist_coef_scale": 0.0,
            "latent_specialist_rollout_samples": 0.0,
        }
        if (
            not bool(getattr(trainer, "latent_specialist_router_enabled", False))
            or not bool(
                getattr(trainer, "latent_specialist_use_rollout_states", False)
            )
            or bool(getattr(trainer, "fixed_latent_strategy", False))
            or int(getattr(buffer, "pos", 0)) <= 0
            or "global_state" not in buffer.fields
            or "opponent_id" not in buffer.fields
        ):
            return stats

        length = int(buffer.pos)
        states = buffer.fields["global_state"][:length].reshape(
            -1, buffer.fields["global_state"].shape[-1]
        )
        opponent_ids = buffer.fields["opponent_id"][:length].reshape(-1).long()
        total = int(states.shape[0])
        max_samples = max(
            1,
            int(
                getattr(trainer, "latent_specialist_rollout_max_samples", 8192)
                or 8192
            ),
        )
        if total > max_samples:
            sample_idx = torch.linspace(
                0,
                total - 1,
                steps=max_samples,
                device=states.device,
            ).round().long().unique()
            states = states.index_select(0, sample_idx)
            opponent_ids = opponent_ids.index_select(0, sample_idx)

        context_keys = _specialist_context_keys_for_mode(
            mode=str(
                getattr(
                    trainer,
                    "latent_specialist_context_key_mode",
                    "opponent_bucket",
                )
                or "opponent_bucket"
            ),
            states=states,
            opponent_ids=opponent_ids,
            bucket_ids=None,
        )
        if context_keys is None:
            return stats

        warmup_steps = int(
            getattr(trainer, "latent_specialist_warmup_steps", 0) or 0
        )
        global_step = int(getattr(trainer, "global_step", 0) or 0)
        coef_scale = _router_specialist_coef_scale(
            global_step=global_step,
            warmup_steps=warmup_steps,
            ramp_steps=int(
                getattr(trainer, "latent_specialist_ramp_steps", 1) or 0
            ),
        )
        conditional_start = (
            float(
                getattr(
                    trainer,
                    "latent_conditional_entropy_min_coef_start",
                    0.0,
                )
                or 0.0
            )
            if global_step >= warmup_steps
            else 0.0
        )

        logits = trainer.model.strategy_logits(states)
        loss, tensor_stats = _router_specialist_loss(
            logits,
            context_keys=context_keys,
            latent_k=int(trainer.latent_k),
            marginal_balance_coef=float(
                getattr(trainer, "latent_marginal_balance_coef", 0.0) or 0.0
            ),
            conditional_entropy_min_coef=float(
                getattr(trainer, "latent_conditional_entropy_min_coef", 0.0)
                or 0.0
            ),
            conditional_entropy_min_coef_start=conditional_start,
            conditional_entropy_scope=str(
                getattr(
                    trainer,
                    "latent_specialist_conditional_entropy_scope",
                    "state",
                )
                or "state"
            ),
            context_mi_coef=float(
                getattr(trainer, "latent_context_mi_coef", 0.0) or 0.0
            ),
            coef_scale=coef_scale,
            min_bucket_count=int(
                getattr(trainer, "latent_specialist_min_bucket_count", 2) or 2
            ),
        )
        if loss.requires_grad and (
            coef_scale > 0.0
            or float(
                getattr(
                    trainer,
                    "latent_conditional_entropy_min_coef_start",
                    0.0,
                )
                or 0.0
            )
            > 0.0
        ):
            optimizer = (
                getattr(trainer, "latent_router_optimizer", None)
                or trainer.optimizer
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            strategy_module = getattr(trainer.model, "strategy_encoder", None)
            if strategy_module is not None:
                torch.nn.utils.clip_grad_norm_(
                    strategy_module.parameters(),
                    float(trainer.cfg.max_grad_norm),
                )
            optimizer.step()

        for key, value in tensor_stats.items():
            stats[key] = float(value.detach().cpu().item())
        stats["latent_specialist_rollout_samples"] = float(states.shape[0])
        return stats

    def strategy_encoder_grad_norm(self) -> float:
        """Return the current q_phi gradient norm before global clipping.

        Reads ``strategy_encoder`` only — since Step 5 the optional aux-return
        head is a separate module, so the q_phi (z-policy) gradient signal is
        the strategy encoder's parameters, not the auxiliary head's.
        """
        trainer = self.trainer
        strategy_module = getattr(trainer.model, "strategy_encoder", None)
        if strategy_module is None:
            return 0.0
        total = torch.zeros((), dtype=torch.float32, device=trainer.device)
        for param in strategy_module.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            total = total + grad.pow(2).sum()
        return float(torch.sqrt(total).detach().cpu().item())


__all__ = ["EpisodeStrategyRecorder", "LatentStrategyState"]
