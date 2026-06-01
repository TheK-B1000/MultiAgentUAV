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

from rl.ppo_core import ppo_policy_loss
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
                "z_persist_mask": false_mask,
            }
            return z_idx, prev_z, aux

        episode_start_mask = self.needs_strategy_sample.clone()
        resample_mask = episode_start_mask.clone()
        if trainer.latent_resample_every_n > 0:
            resample_mask |= self.strategy_age >= trainer.latent_resample_every_n

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

        prev_z = self.current_z.clone()
        z_idx = self.current_z.clone()
        persist_mask = resample_mask & (~self.needs_strategy_sample) & (~commit_now)

        z_logits = trainer.model.strategy_logits(global_state)
        z_dist = Categorical(logits=z_logits)
        if bool(resample_mask.any().item()):
            idx = torch.where(resample_mask)[0]
            sampled_dist = Categorical(logits=z_logits.index_select(0, idx))
            sampled_z = trainer.model._categorical_argmax_or_sample(
                sampled_dist,
                deterministic=False,
                generator=trainer.model._sampling_gen_strategy,
            )
            z_idx[idx] = sampled_z
            self.current_z = z_idx.clone()
            self.strategy_age[idx] = 0
            self.needs_strategy_sample[idx] = False

        z_log_prob = z_dist.log_prob(z_idx)
        z_entropy = z_dist.entropy()
        # Snapshot the q_phi training (state, z, log_prob) pair:
        # - warmup == 0: legacy behavior, snapshot at episode start (step 0)
        # - warmup  > 0: snapshot at the commit step, after the EMA window
        if warmup > 0:
            snapshot_mask = commit_now
        else:
            snapshot_mask = episode_start_mask

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

        aux = {
            "z": z_idx,
            "prev_z": prev_z,
            "z_log_prob": z_log_prob,
            "z_entropy": z_entropy,
            "z_logits": z_logits,
            "z_resampled": training_resample_mask,
            "z_resampled_actual": resample_mask,
            "z_persist_mask": persist_mask,
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
        if bool(done_t.any().item()):
            self.strategy_age[done_t] = 0
            self.needs_strategy_sample[done_t] = not trainer.fixed_latent_strategy
            self.steps_since_ep_start[done_t] = 0
            self.episode_strategy_committed[done_t] = False
            self.first_z_sample_step[done_t] = -1
            self.episode_return_baseline_at_commit[done_t] = 0.0

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
    def empty_episode_strategy_stats() -> dict[str, float]:
        return {
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
        }

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
        stats = self.empty_episode_strategy_stats()
        batch = self.episode_strategy_training_batch()
        if batch is None:
            return stats
        states = batch["states"]
        z = batch["z"]
        old_log_prob = batch["old_log_prob"]
        episode_returns = batch["episode_returns"]
        opponent_ids = batch.get("opponent_ids")
        bucket_ids = batch.get("bucket_ids")

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
                    unique_keys = torch.unique(keys)
                    for k in unique_keys:
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
            loss = trainer.latent_episode_strategy_coef * (
                pg_loss + trainer.latent_episode_strategy_value_coef * v_loss
            ) + entropy_term

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
