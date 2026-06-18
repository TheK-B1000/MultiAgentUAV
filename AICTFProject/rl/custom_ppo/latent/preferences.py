"""v3i3 preference targets and router teacher helpers."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

def v3i3_target_from_items(
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


def advantage_weighted_target_from_records(
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


def router_specialist_coef_scale(
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


def warmup_ramp_coef_scale(
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


def router_specialist_loss(
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


def v3i3_resolve_target(
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

    def get(level_key: tuple, source: dict, raw_key: tuple) -> Optional[np.ndarray]:
        if level_key in target_cache:
            return target_cache[level_key]
        items = source.get(raw_key, [])
        t = v3i3_target_from_items(
            items,
            latent_k=latent_k,
            min_count=min_count,
            min_distinct_z=min_distinct_z,
            temperature=temperature,
        )
        target_cache[level_key] = t
        return t

    t = get(full_key, by_full, full_key[1])
    if t is not None:
        return t, "full"

    if oef_key is not None:
        t = get(oef_key, by_oef, oef_key[1])
        if t is not None:
            return t, "oef"

    t = get(oe_key, by_oe, oe_key[1])
    if t is not None:
        return t, "oe"

    t = get(o_key, by_o, o_key[1])
    if t is not None:
        return t, "o"

    return None, None
