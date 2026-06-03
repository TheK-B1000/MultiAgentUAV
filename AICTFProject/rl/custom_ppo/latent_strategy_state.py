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
            })
            for i in range(latent_k):
                for j in range(latent_k):
                    stats[f"latent_refresh_z{i}_to_z{j}"] = 0.0
            return stats
        
        count = float(self.rollout_refresh_count)
        total_steps = float(max(1, self.rollout_refresh_total_steps))
        z_changed_rate = float(self.rollout_refresh_z_changed_count) / count if count > 0 else 0.0
        
        stats.update({
            "latent_refresh_count": count,
            "latent_refresh_rate": count / total_steps,
            "latent_refresh_reason_enemy_flag": float(self.rollout_refresh_reason_enemy_flag),
            "latent_refresh_reason_friendly_flag": float(self.rollout_refresh_reason_friendly_flag),
            "latent_refresh_reason_score_change": float(self.rollout_refresh_reason_score_change),
            "latent_refresh_reason_near_base": float(self.rollout_refresh_reason_near_base),
            "latent_refresh_z_changed_rate": z_changed_rate,
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
        }

    # ------------------------------------------------------------------
    # Episode outcome → completed-record buffer
    # ------------------------------------------------------------------

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
                "context_bucket": int(self.episode_contrast_bucket[env_i].detach().cpu().item()),
                "opponent": opponent_id,
                "phase_flag_state": int(self.episode_contrast_bucket[env_i].detach().cpu().item()),
                "z": z_val,
                "return": float(episode_return),
                "behavior_embedding": emb,
                "win_loss": episode_win,
            }
            self.latent_preference_buffer.append(forced_record)
            return

        if not bool(self.episode_strategy_has_start[env_i].detach().cpu().item()):
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
                "bucket_id": int(self.episode_strategy_bucket[env_i].detach().cpu().item()),
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
            from rl.custom_ppo.latent_bucket_baseline import resolve_bucket_ids
            keys = resolve_bucket_ids(
                mode=str(bucket_mode),
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
                    from rl.custom_ppo.latent_bucket_baseline import resolve_bucket_ids
                    keys = resolve_bucket_ids(
                        mode=str(bucket_mode),
                        opponent_ids=opponent_ids,
                        bucket_ids=bucket_ids,
                    )
                    normalized_adv = torch.zeros_like(adv)
                    unique_keys_tensor = torch.unique(keys)
                    for k in unique_keys_tensor:
                        mask = (keys == k)
                        if mask.sum() > 1:
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
            pref_loss = torch.zeros((), dtype=torch.float32, device=trainer.device)
            pref_loss_scaled = torch.zeros((), dtype=torch.float32, device=trainer.device)
            commit_loss = torch.zeros((), dtype=torch.float32, device=trainer.device)
            if pref_coef > 0.0 and bool(batch_pref_mask.any().item()):
                valid_logits = logits[batch_pref_mask]
                valid_targets = batch_target_probs[batch_pref_mask]
                log_probs = torch.log_softmax(valid_logits, dim=-1)
                
                # Compute target confidence: 1.0 - target_entropy / log(K)
                target_probs_clamped = valid_targets.clamp_min(1e-8)
                target_entropy_eps = -(valid_targets * torch.log(target_probs_clamped)).sum(dim=-1)
                import math
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
                
                # Raw KL loss for telemetry
                if getattr(trainer.cfg, "latent_preference_opponent_balanced", False) and opponent_ids is not None:
                    valid_opps = opponent_ids[batch_pref_mask]
                    unique_opps = torch.unique(valid_opps)
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
                if getattr(trainer.cfg, "latent_preference_opponent_balanced", False) and opponent_ids is not None:
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
                    
                    if getattr(trainer.cfg, "latent_preference_opponent_balanced", False) and opponent_ids is not None:
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

            loss = trainer.latent_episode_strategy_coef * (
                pg_loss + trainer.latent_episode_strategy_value_coef * v_loss
            ) + entropy_term + usage_loss + pref_loss_scaled + commit_loss

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
            stats["latent_preference_loss"] = float(pref_loss.detach().cpu().item())
            stats["latent_preference_active_fraction"] = float(batch_pref_mask.float().mean().cpu().item())
            stats["latent_preference_buffer_size"] = float(len(self.latent_preference_buffer))
            stats["latent_preference_num_active_buckets"] = float(active_buckets_count)
            valid_count = int(batch_pref_mask.sum().item())
            stats["latent_preference_target_entropy"] = float(target_entropy_sum / max(1, valid_count)) if valid_count > 0 else 0.0

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
