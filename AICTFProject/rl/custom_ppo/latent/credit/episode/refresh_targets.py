"""v3i3 event-conditioned refresh preference targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from rl.custom_ppo.latent.preferences import v3i3_resolve_target
from rl.custom_ppo.latent.records import stack_selector_hidden_records


@dataclass
class RefreshTargets:
    active: bool
    coef: float
    refresh_states: torch.Tensor | None
    refresh_hidden: torch.Tensor | None
    target_probs: torch.Tensor | None
    mask: torch.Tensor | None
    active_buckets: int
    active_records: int
    target_entropy_sum: float
    fallback_counts: dict[str, int]
    rollout_records: int
    recurrent_required: bool
    valid: bool


def build_refresh_targets(*, trainer: Any, host: Any) -> RefreshTargets:
    coef = float(getattr(trainer, "latent_v3i3_event_preference_coef", 0.0) or 0.0)
    warmup = int(getattr(trainer, "latent_v3i3_event_preference_warmup_steps", 0) or 0)
    enabled = bool(getattr(trainer, "latent_v3i3_event_preference_enabled", False))
    records = list(host.rollout_refresh_records)
    recurrent_required = bool(getattr(trainer.model, "use_recurrent_selector", False))
    empty = RefreshTargets(
        active=False,
        coef=coef,
        refresh_states=None,
        refresh_hidden=None,
        target_probs=None,
        mask=None,
        active_buckets=0,
        active_records=0,
        target_entropy_sum=0.0,
        fallback_counts={"full": 0, "oef": 0, "oe": 0, "o": 0},
        rollout_records=len(records),
        recurrent_required=recurrent_required,
        valid=True,
    )
    active = (
        enabled
        and coef > 0.0
        and len(host.refresh_preference_buffer) > 0
        and len(records) > 0
        and (warmup <= 0 or int(getattr(trainer, "global_step", 0) or 0) >= warmup)
    )
    if not active:
        return empty

    device = trainer.device
    refresh_states = torch.stack(
        [r["refresh_state"].detach().float() for r in records], dim=0
    ).to(device)
    refresh_hidden = stack_selector_hidden_records(records, device=device)
    if recurrent_required and refresh_hidden is None:
        return RefreshTargets(
            active=True,
            coef=coef,
            refresh_states=refresh_states,
            refresh_hidden=None,
            target_probs=None,
            mask=None,
            active_buckets=0,
            active_records=0,
            target_entropy_sum=0.0,
            fallback_counts={"full": 0, "oef": 0, "oe": 0, "o": 0},
            rollout_records=len(records),
            recurrent_required=True,
            valid=False,
        )

    by_full: dict = {}
    by_oef: dict = {}
    by_oe: dict = {}
    by_o: dict = {}
    normalize = bool(getattr(trainer, "latent_v3i3_event_preference_normalize", False))
    if normalize:
        baselines: dict = {}
        counts: dict = {}
        for record in host.refresh_preference_buffer:
            if trainer.latent_event_preference_key_mode == "event_flag_progress":
                key = (
                    int(record["opponent_id"]),
                    int(record["event_type"]),
                    int(record["flag_state_bucket"]),
                    int(record.get("carrier_progress_bucket", -1)),
                )
            else:
                key = (
                    int(record["opponent_id"]),
                    int(record["event_type"]),
                    int(record["flag_state_bucket"]),
                )
            baselines[key] = baselines.get(key, 0.0) + float(record["future_return"])
            counts[key] = counts.get(key, 0) + 1
        for key in baselines:
            baselines[key] /= float(counts[key])

    for record in host.refresh_preference_buffer:
        opp_b = int(record["opponent_id"])
        ev_b = int(record["event_type"])
        fl_b = int(record["flag_state_bucket"])
        pr_b = int(record.get("carrier_progress_bucket", -1))
        ret_val = float(record["future_return"])
        if normalize:
            if trainer.latent_event_preference_key_mode == "event_flag_progress":
                k_full = (opp_b, ev_b, fl_b, pr_b)
            else:
                k_full = (opp_b, ev_b, fl_b)
            ret_val -= baselines.get(k_full, 0.0)
        pair = (int(record["z"]), ret_val)
        if trainer.latent_event_preference_key_mode == "event_flag_progress":
            by_full.setdefault((opp_b, ev_b, fl_b, pr_b), []).append(pair)
            by_oef.setdefault((opp_b, ev_b, fl_b), []).append(pair)
        else:
            by_full.setdefault((opp_b, ev_b, fl_b), []).append(pair)
        by_oe.setdefault((opp_b, ev_b), []).append(pair)
        by_o.setdefault((opp_b,), []).append(pair)

    min_count = int(getattr(trainer, "latent_v3i3_event_preference_min_bucket_count", 4) or 4)
    min_distinct = int(getattr(trainer, "latent_v3i3_event_preference_min_distinct_z", 2) or 2)
    temperature = float(getattr(trainer, "latent_v3i3_event_preference_temperature", 0.75) or 0.75)
    latent_k = int(trainer.latent_k)
    target_arr = np.full((len(records), latent_k), 1.0 / float(latent_k), dtype=np.float32)
    mask_arr = np.zeros((len(records),), dtype=bool)
    target_cache: dict = {}
    active_keys: set = set()
    active_records = 0
    target_entropy_sum = 0.0
    fallback_counts = {"full": 0, "oef": 0, "oe": 0, "o": 0}

    for i, record in enumerate(records):
        target, level = v3i3_resolve_target(
            opponent_id=int(record["opponent_id"]),
            event_type=int(record["reason_id"]),
            flag_state_bucket=int(record["flag_state_bucket"]),
            carrier_progress_bucket=int(record.get("carrier_progress_bucket", -1)),
            by_full=by_full,
            by_oef=by_oef,
            by_oe=by_oe,
            by_o=by_o,
            latent_k=latent_k,
            min_count=min_count,
            min_distinct_z=min_distinct,
            temperature=temperature,
            target_cache=target_cache,
            key_mode=trainer.latent_event_preference_key_mode,
        )
        if target is None or level is None:
            continue
        target_arr[i] = target
        mask_arr[i] = True
        active_records += 1
        target_entropy_sum += float(-(target * np.log(target + 1e-12)).sum())
        fallback_counts[level] = fallback_counts[level] + 1
        if level == "full":
            if trainer.latent_event_preference_key_mode == "event_flag_progress":
                active_keys.add(
                    (
                        "full",
                        int(record["opponent_id"]),
                        int(record["reason_id"]),
                        int(record["flag_state_bucket"]),
                        int(record.get("carrier_progress_bucket", -1)),
                    )
                )
            else:
                active_keys.add(
                    (
                        "full",
                        int(record["opponent_id"]),
                        int(record["reason_id"]),
                        int(record["flag_state_bucket"]),
                    )
                )
        elif level == "oef":
            active_keys.add(
                ("oef", int(record["opponent_id"]), int(record["reason_id"]), int(record["flag_state_bucket"]))
            )
        elif level == "oe":
            active_keys.add(("oe", int(record["opponent_id"]), int(record["reason_id"])))
        else:
            active_keys.add(("o", int(record["opponent_id"])))

    return RefreshTargets(
        active=True,
        coef=coef,
        refresh_states=refresh_states,
        refresh_hidden=refresh_hidden,
        target_probs=torch.as_tensor(target_arr, dtype=torch.float32, device=device),
        mask=torch.as_tensor(mask_arr, dtype=torch.bool, device=device),
        active_buckets=len(active_keys),
        active_records=active_records,
        target_entropy_sum=target_entropy_sum,
        fallback_counts=fallback_counts,
        rollout_records=len(records),
        recurrent_required=recurrent_required,
        valid=True,
    )
