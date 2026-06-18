"""Episode-boundary router credit and q_phi PPO update."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from rl.custom_ppo.csv_writers import SCRIPTED_OPPONENT_MI_COUNT, _opponent_id_int_from_info as opponent_id_int_from_info
from rl.custom_ppo.latent_value_baselines import compute_z_marginal_strategy_value
from rl.ppo_core import ppo_policy_loss
from rl.custom_ppo.latent.context_buckets import (
    episode_bucket_baseline_keys,
    specialist_context_keys_for_mode,
    strategy_experience_bucket_ids,
)
from rl.custom_ppo.latent.records import stack_selector_hidden_records
from rl.custom_ppo.latent.types import RouterActionSource
from rl.custom_ppo.latent.preferences import (
    advantage_weighted_target_from_records as _advantage_weighted_target_from_records,
    router_specialist_coef_scale as _router_specialist_coef_scale,
    router_specialist_loss as _router_specialist_loss,
    v3i3_resolve_target as _v3i3_resolve_target,
    v3i3_target_from_items as _v3i3_target_from_items,
    warmup_ramp_coef_scale as _warmup_ramp_coef_scale,
)

if TYPE_CHECKING:
    from rl.custom_ppo.latent.state import LatentStrategyState


class EpisodeCreditManager:
    def __init__(self, host: LatentStrategyState) -> None:
        self.host = host

    def store_episode_strategy_start(
        self,
        *,
        start_mask: torch.Tensor,
        global_state: torch.Tensor,
        z_idx: torch.Tensor,
        z_log_prob: torch.Tensor,
        z_logits: torch.Tensor,
        selector_hidden: torch.Tensor | None = None,
    ) -> None:
        """Snapshot the exact actor-controlling z at episode start for q_phi PPO credit."""
        trainer = self.host.trainer
        if not trainer.latent_episode_strategy_ppo or not bool(start_mask.any().item()):
            return
        idx = torch.where(start_mask)[0]
        probs = torch.softmax(z_logits.detach(), dim=-1)
        buckets = strategy_experience_bucket_ids(global_state.index_select(0, idx)).detach()
        self.host.episode_strategy_state[idx] = global_state.index_select(0, idx).detach()
        self.host.episode_strategy_z[idx] = z_idx.index_select(0, idx).detach()
        self.host.episode_strategy_log_prob[idx] = z_log_prob.index_select(0, idx).detach()
        if selector_hidden is not None and self.host.episode_strategy_selector_hidden is not None:
            self.host.episode_strategy_selector_hidden[idx] = selector_hidden.index_select(0, idx).detach()
        self.host.episode_strategy_probs[idx, : trainer.latent_k] = probs.index_select(0, idx)
        self.host.episode_strategy_bucket[idx] = buckets
        self.host.episode_strategy_has_start[idx] = True
        for row_i, env_i in enumerate(idx.detach().cpu().tolist()):
            hidden_row = None
            if self.host.episode_strategy_selector_hidden is not None:
                hidden_row = self.host.episode_strategy_selector_hidden[int(env_i)]
            router_row_logprob = float(
                Categorical(logits=z_logits[int(env_i) : int(env_i) + 1])
                .log_prob(z_idx[int(env_i) : int(env_i) + 1])
                .detach()
                .cpu()
                .item()
            )
            self.host.episode_strategy_recorder.record_start(
                env_index=int(env_i),
                episode_id=int(self.host.next_strategy_episode_id),
                global_state_0=global_state[int(env_i)],
                proposed_z=int(z_idx[int(env_i)].detach().cpu().item()),
                executed_z=int(z_idx[int(env_i)].detach().cpu().item()),
                behavior_log_prob=float(z_log_prob[int(env_i)].detach().cpu().item()),
                router_log_prob=router_row_logprob,
                action_source=RouterActionSource.ROUTER,
                bucket_id=int(buckets[row_i].detach().cpu().item()),
                q_phi_probs=probs[int(env_i), : trainer.latent_k].detach().cpu().tolist(),
                selector_hidden_0=hidden_row,
            )
            self.host.next_strategy_episode_id += 1

    # ------------------------------------------------------------------
    # v3i19 arc-credit hooks
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
        trainer = self.host.trainer
        if (
            not trainer.latent_episode_strategy_ppo
            or trainer.fixed_latent_strategy
            or trainer.model.episode_strategy_value_head is None
        ):
            return None
        records = list(self.host.rollout_strategy_episode_records)
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
        selector_hidden = stack_selector_hidden_records(records, device=device)
        batch = {
            "states": states,
            "z": z,
            "old_log_prob": old_log_prob,
            "episode_returns": episode_returns,
            "opponent_ids": opponent_ids,
            "bucket_ids": bucket_ids,
        }
        if selector_hidden is not None:
            batch["selector_hidden"] = selector_hidden
        return batch


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
        trainer = self.host.trainer
        stats = EpisodeCreditManager.empty_episode_strategy_stats(trainer.latent_k)
        batch = self.host.episode_strategy_training_batch()
        if batch is None:
            return stats
        states = batch["states"]
        z = batch["z"]
        old_log_prob = batch["old_log_prob"]
        episode_returns = batch["episode_returns"]
        opponent_ids = batch.get("opponent_ids")
        bucket_ids = batch.get("bucket_ids")
        selector_hidden = batch.get("selector_hidden")
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
            keys = episode_bucket_baseline_keys(
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

        if pref_coef > 0.0 and len(self.host.latent_preference_buffer) > 0 and opponent_ids is not None and bucket_ids is not None:
            batch_keys = (opponent_ids * 256 + bucket_ids).detach().cpu().numpy().tolist()
            unique_keys = set(batch_keys)
            
            # Group buffer records by key
            buffer_by_key = {}
            for r in self.host.latent_preference_buffer:
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
            and len(self.host.latent_preference_buffer) > 0
            and opponent_ids is not None
            and bucket_ids is not None
        ):
            batch_awrd_keys = (opponent_ids * 256 + bucket_ids).detach().cpu().numpy().tolist()
            awrd_buffer_by_key: dict[int, list[dict[str, Any]]] = {}
            for rec in self.host.latent_preference_buffer:
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
        v3i3_records = list(self.host.rollout_refresh_records)
        v3i3_active = (
            v3i3_enabled
            and v3i3_coef > 0.0
            and len(self.host.refresh_preference_buffer) > 0
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
                for r in self.host.refresh_preference_buffer:
                    if trainer.latent_event_preference_key_mode == "event_flag_progress":
                        k = (int(r["opponent_id"]), int(r["event_type"]), int(r["flag_state_bucket"]), int(r.get("carrier_progress_bucket", -1)))
                    else:
                        k = (int(r["opponent_id"]), int(r["event_type"]), int(r["flag_state_bucket"]))
                    baselines[k] = baselines.get(k, 0.0) + float(r["future_return"])
                    counts[k] = counts.get(k, 0) + 1
                for k in baselines:
                    baselines[k] /= float(counts[k])
            for r in self.host.refresh_preference_buffer:
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
        specialist_context_keys = specialist_context_keys_for_mode(
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
        logits = trainer.model.strategy_logits(states, selector_hidden=selector_hidden)
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
            logits = trainer.model.strategy_logits(states, selector_hidden=selector_hidden)
            dist = Categorical(logits=logits)
            new_log_prob = dist.log_prob(z)
            v_z = trainer.model.episode_strategy_value(
                states, z, selector_hidden=selector_hidden
            )

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
                    trainer.model,
                    states,
                    trainer.latent_k,
                    policy_weighted=False,
                    selector_hidden=selector_hidden,
                )
            else:
                v_baseline = v_z.detach()

            adv = episode_returns - v_baseline
            if trainer.latent_episode_strategy_return_norm and adv.numel() > 1:
                if bucket_baseline_vector is not None and bucket_mode is not None:
                    keys = episode_bucket_baseline_keys(
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
            from rl.custom_ppo.v6i1_phase_runtime import (
                is_v6i1_staged_trainer,
                resolve_v6i1_rollout_usage_coef,
            )

            if is_v6i1_staged_trainer(trainer):
                usage_coef = float(resolve_v6i1_rollout_usage_coef(trainer))
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
            episode_credit_grad_norm = self.host.strategy_encoder_grad_norm()
            torch.nn.utils.clip_grad_norm_(clip_params, float(trainer.cfg.max_grad_norm))
            router_optimizer.step()
            self.host.router_optimizer_step_count += 1

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

            # v5i3 per-z router telemetry. Lets the post-mortem distinguish
            # "z_i is rarely sampled" (router_sample_count_by_z[i] low) from
            # "z_i is sampled enough but receives noisy/weak credit"
            # (count high, adv_std_by_z[i] high, adv_mean_by_z[i] near 0).
            # All inputs come from the on-policy episode batch -- forced
            # episodes are already filtered out by record_episode_strategy_outcome's
            # is_forced_z branch (see line ~2348).
            K = max(1, int(trainer.latent_k))
            z_cpu = z.detach().cpu()
            adv_cpu = adv.detach().cpu()
            ret_cpu = episode_returns.detach().cpu()
            ratio_cpu = ratio.detach().cpu()
            clip_eps = float(getattr(trainer, "latent_episode_strategy_clip_eps", 0.2) or 0.2)
            clipped_per_record = (torch.abs(ratio_cpu - 1.0) > clip_eps).float()
            for z_i in range(K):
                mask = (z_cpu == z_i)
                count_i = int(mask.sum().item())
                forced_i = int(self.host.rollout_forced_z_episode_count_by_z[z_i])
                stats[f"router_sample_count_by_z_{z_i}"] = float(count_i)
                stats[f"forced_sample_count_by_z_{z_i}"] = float(forced_i)
                stats[f"episode_count_by_z_{z_i}"] = float(count_i + forced_i)
                if count_i == 0:
                    stats[f"mean_episode_advantage_by_z_{z_i}"] = 0.0
                    stats[f"std_episode_advantage_by_z_{z_i}"] = 0.0
                    stats[f"mean_return_by_z_{z_i}"] = 0.0
                    stats[f"mean_logprob_ratio_by_z_{z_i}"] = 1.0
                    stats[f"clip_fraction_by_z_{z_i}"] = 0.0
                    continue
                adv_i = adv_cpu[mask]
                ret_i = ret_cpu[mask]
                ratio_i = ratio_cpu[mask]
                clip_i = clipped_per_record[mask]
                stats[f"mean_episode_advantage_by_z_{z_i}"] = float(adv_i.mean().item())
                stats[f"std_episode_advantage_by_z_{z_i}"] = (
                    float(adv_i.std(unbiased=False).item()) if count_i > 1 else 0.0
                )
                stats[f"mean_return_by_z_{z_i}"] = float(ret_i.mean().item())
                stats[f"mean_logprob_ratio_by_z_{z_i}"] = float(ratio_i.mean().item())
                stats[f"clip_fraction_by_z_{z_i}"] = float(clip_i.mean().item())
            for o_idx in range(int(SCRIPTED_OPPONENT_MI_COUNT)):
                for z_i in range(K):
                    stats[f"forced_episode_opp{o_idx}_z{z_i}_count"] = float(
                        self.host.rollout_forced_episode_count_by_opp_z[o_idx, z_i]
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
                len(self.host.refresh_preference_buffer)
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
            stats["latent_preference_buffer_size"] = float(len(self.host.latent_preference_buffer))
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
            stats["latent_awrd_buffer_size"] = float(len(self.host.latent_preference_buffer))
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
                stats[f"latent_pref_{opp_name}_buffer_count"] = float(sum(1 for r in self.host.latent_preference_buffer if r["opponent"] == opp_id))
                
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
        trainer = self.host.trainer
        env_i = int(env_index)
        if env_i < 0 or env_i >= int(self.host.episode_strategy_has_start.numel()):
            return

        is_forced_z = bool(self.host.episode_forced_z[env_i].detach().cpu().item())
        z_val = int(self.host.episode_forced_z_id[env_i].detach().cpu().item()) if is_forced_z else int(self.host.current_z[env_i].detach().cpu().item())
        self.host.recent_z_history.append(z_val)

        if is_forced_z:
            try:
                opponent_id = int(opponent_id_int_from_info(self.host.trainer.cfg, info))
            except Exception:
                opponent_id = -1

            er = info.get("episode_result") if isinstance(info.get("episode_result"), dict) else {}
            bs = int(er.get("blue_score", info.get("blue_score", 0)) or 0)
            rs = int(er.get("red_score", info.get("red_score", 0)) or 0)
            episode_win = 1 if bs > rs else 0

            count = max(1, int(self.host.episode_behavior_count[env_i].detach().cpu().item()))
            emb = (self.host.episode_behavior_sum[env_i] / float(count)).detach().cpu().numpy().tolist()
            tactical_bucket = self.host.representative_tactical_bucket(env_i)

            if trainer.latent_episode_strategy_ppo:
                forced_record = {
                    "context_bucket": tactical_bucket,
                    "opponent": opponent_id,
                    "phase_flag_state": tactical_bucket,
                    "z": z_val,
                    "return": float(episode_return),
                    "behavior_embedding": emb,
                    "win_loss": episode_win,
                }
                self.host.latent_preference_buffer.append(forced_record)
                used_tactical_fallback = (
                    int(self.host.episode_tactical_bucket_counts[env_i].sum().item()) <= 0
                )
                self.host.rollout_tactical_bucket_sample_count += 1
                self.host.rollout_tactical_bucket_fallback_count += int(
                    used_tactical_fallback
                )

            self.host.rollout_completed_episode_count += 1
            self.host.rollout_forced_z_episode_count += 1
            if 0 <= z_val < int(self.host.rollout_forced_z_episode_count_by_z.shape[0]):
                self.host.rollout_forced_z_episode_count_by_z[z_val] += 1
            if (
                0 <= opponent_id < int(self.host.rollout_forced_episode_count_by_opp_z.shape[0])
                and 0 <= z_val < int(self.host.rollout_forced_episode_count_by_opp_z.shape[1])
            ):
                self.host.rollout_forced_episode_count_by_opp_z[opponent_id, z_val] += 1

            # Update V6I1 competence running metrics
            alpha = float(getattr(trainer.cfg, "latent_cf_competence_ema_alpha", 0.05))
            self.host.cf_J[z_val] = (1.0 - alpha) * self.host.cf_J[z_val] + alpha * float(episode_return)
            self.host.cf_episode_counts[z_val] += 1
            self.host.cf_has_experience[z_val] = True
            
            old_mean = self.host.cf_return_mean
            self.host.cf_return_mean = (1.0 - alpha) * old_mean + alpha * float(episode_return)
            self.host.cf_return_var = (1.0 - alpha) * self.host.cf_return_var + alpha * (float(episode_return) - old_mean) * (float(episode_return) - self.host.cf_return_mean)
            return

        if not trainer.latent_episode_strategy_ppo:
            return

        if not bool(self.host.episode_strategy_has_start[env_i].detach().cpu().item()):
            return
        used_tactical_fallback = (
            int(self.host.episode_tactical_bucket_counts[env_i].sum().item()) <= 0
        )
        self.host.rollout_tactical_bucket_sample_count += 1
        self.host.rollout_tactical_bucket_fallback_count += int(
            used_tactical_fallback
        )
        tactical_bucket = self.host.representative_tactical_bucket(env_i)

        er = info.get("episode_result") if isinstance(info.get("episode_result"), dict) else {}
        bs = int(er.get("blue_score", info.get("blue_score", 0)) or 0)
        rs = int(er.get("red_score", info.get("red_score", 0)) or 0)
        episode_win = 1 if bs > rs else 0
        warmup = int(getattr(trainer, "latent_episode_strategy_warmup_decision_steps", 0) or 0)
        if warmup > 0:
            baseline = float(self.host.episode_return_baseline_at_commit[env_i].detach().cpu().item())
            adjusted_return = episode_return - baseline
        else:
            adjusted_return = episode_return

        try:
            opponent_id = int(opponent_id_int_from_info(trainer.cfg, info))
        except Exception:
            opponent_id = -1

        record = self.host.episode_strategy_recorder.record_outcome(
            env_index=env_i,
            episode_return=float(adjusted_return),
            episode_win=episode_win,
            opponent_id=opponent_id,
        )
        if record is not None:
            record["bucket_id"] = tactical_bucket
            self.host.rollout_strategy_episode_records.append(record)
            return
        self.host.missing_episode_record_count += 1
