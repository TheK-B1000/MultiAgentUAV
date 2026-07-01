"""Tensor and scalar field allocation for :class:`LatentStrategyStateCore`."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

from rl.behavior_telemetry import N_TELEMETRY
from rl.custom_ppo.csv_writers import SCRIPTED_OPPONENT_MI_COUNT
from rl.custom_ppo.latent.lifecycle import EpisodeLifecycleState
from rl.custom_ppo.latent.records import EpisodeStrategyRecorder
from rl.custom_ppo.latent.selector_memory import SelectorMemory

if TYPE_CHECKING:
    from rl.custom_ppo.trainer import CustomPPOTrainer


def allocate_latent_state_fields(host: Any, trainer: "CustomPPOTrainer") -> None:
    """Attach rollout tensors, buffers, and lifecycle owners to ``host``."""
    n_envs = int(trainer.env.num_envs)
    device = trainer.device
    strategy_prob_width = max(1, int(trainer.latent_k))

    host.episode_return_accum = torch.zeros((n_envs,), dtype=torch.float32, device=device)
    host.episode_return_baseline_at_commit = torch.zeros((n_envs,), dtype=torch.float32, device=device)
    host.episode_strategy_state = torch.zeros(
        (n_envs, int(trainer.model.global_state_dim)), dtype=torch.float32, device=device
    )
    host.episode_strategy_selector_hidden: Optional[torch.Tensor] = None
    host.episode_strategy_z = torch.zeros((n_envs,), dtype=torch.long, device=device)
    host.episode_strategy_log_prob = torch.zeros((n_envs,), dtype=torch.float32, device=device)
    host.episode_strategy_probs = torch.zeros(
        (n_envs, strategy_prob_width), dtype=torch.float32, device=device
    )
    host.episode_strategy_bucket = torch.zeros((n_envs,), dtype=torch.long, device=device)
    host.episode_tactical_bucket_counts = torch.zeros(
        (n_envs, 60), dtype=torch.long, device=device
    )
    host.episode_strategy_has_start = torch.zeros((n_envs,), dtype=torch.bool, device=device)
    host.rollout_strategy_episode_records: list[dict[str, Any]] = []
    host.episode_strategy_recorder = EpisodeStrategyRecorder()
    host.next_strategy_episode_id = 0

    host.current_z: Optional[torch.Tensor] = None
    host.strategy_age = torch.zeros((n_envs,), dtype=torch.long, device=device)
    host.needs_strategy_sample = torch.ones((n_envs,), dtype=torch.bool, device=device)
    host.z_kl_first_in_ep: Optional[torch.Tensor] = None
    host.prev_z_logits: Optional[torch.Tensor] = None
    # Per-env state for the episode-credit warmup. Only meaningful when
    # ``latent_episode_strategy_ppo`` is True AND
    # ``latent_episode_strategy_warmup_decision_steps > 0``. ``steps_since_ep_start``
    # counts decision steps elapsed since the most recent episode reset (0 on the
    # step where ``needs_strategy_sample`` first fires). ``episode_strategy_committed``
    # is True once the committed (post-warmup) z + context has been snapshotted, and
    # False between episode reset and that commit moment.
    host.steps_since_ep_start = torch.zeros((n_envs,), dtype=torch.long, device=device)
    host.episode_strategy_committed = torch.zeros((n_envs,), dtype=torch.bool, device=device)
    host.first_z_sample_step = torch.full(
        (n_envs,), -1, dtype=torch.long, device=device
    )
    host.episode_forced_z = torch.zeros((n_envs,), dtype=torch.bool, device=device)
    host.episode_forced_z_id = torch.zeros((n_envs,), dtype=torch.long, device=device)
    host.episode_contrast_bucket = torch.zeros((n_envs,), dtype=torch.long, device=device)
    host.episode_behavior_sum = torch.zeros((n_envs, N_TELEMETRY), dtype=torch.float32, device=device)
    host.episode_behavior_count = torch.zeros((n_envs,), dtype=torch.long, device=device)
    host.rollout_behavior_contrast_bonus_sum = 0.0
    host.rollout_behavior_contrast_distance_sum = 0.0
    host.rollout_behavior_contrast_count = 0
    host.rollout_behavior_contrast_active_count = 0
    host.rollout_forced_z_episode_count = 0
    host.rollout_completed_episode_count = 0
    host.rollout_tactical_bucket_fallback_count = 0
    host.rollout_tactical_bucket_sample_count = 0
    # v5i3 per-z router telemetry. Counts the forced (uniformly-sampled
    # exploration) episodes by z within the current rollout window. The
    # router-sample count by z is derived from rollout_strategy_episode_records
    # at telemetry time (forced episodes never enter that buffer; see the
    # is_forced_z early-return in record_episode_strategy_outcome).
    host.rollout_forced_z_episode_count_by_z = np.zeros(
        (max(1, int(trainer.latent_k)),), dtype=np.int64
    )
    host.rollout_forced_episode_count_by_opp_z = np.zeros(
        (
            int(SCRIPTED_OPPONENT_MI_COUNT),
            max(1, int(trainer.latent_k)),
        ),
        dtype=np.int64,
    )
    host.latent_preference_buffer = deque(maxlen=20000)

    # Event refresh variables
    host.steps_since_last_refresh = torch.zeros((n_envs,), dtype=torch.long, device=device)
    host.refresh_count_this_episode = torch.zeros((n_envs,), dtype=torch.long, device=device)
    host.steps_since_last_tactical_refresh = torch.zeros(
        (n_envs,), dtype=torch.long, device=device
    )
    host.steps_since_z_change = torch.zeros(
        (n_envs,), dtype=torch.long, device=device
    )
    host.prev_global_state = None
    host.previous_opportunity_features = torch.zeros(
        (n_envs, int(trainer.model.global_state_dim) // 5 if int(trainer.model.global_state_dim) % 5 == 0 else 34),
        dtype=torch.float32,
        device=device,
    )
    host.previous_router_context = torch.zeros(
        (n_envs, 68), dtype=torch.float32, device=device
    )
    host.persistence_valid = torch.zeros((n_envs,), dtype=torch.bool, device=device)
    host.opportunity_index_per_env = torch.zeros((n_envs,), dtype=torch.long, device=device)
    host.rollout_refresh_transitions = np.zeros(
        (max(1, int(trainer.latent_k)), max(1, int(trainer.latent_k))),
        dtype=np.float32,
    )

    # Rollout accumulator stats
    host.rollout_refresh_count = 0
    host.rollout_refresh_z_changed_count = 0
    host.rollout_refresh_reason_enemy_flag = 0
    host.rollout_refresh_reason_friendly_flag = 0
    host.rollout_refresh_reason_score_change = 0
    host.rollout_refresh_reason_near_base = 0
    host.rollout_refresh_total_steps = 0
    host.rollout_sparse_z_change_count = 0
    host.rollout_sparse_z_dwell_sum = 0.0
    host.rollout_sparse_z_dwell_count = 0
    host.rollout_sparse_refresh_attempt_count = 0
    host.rollout_sparse_refresh_accept_count = 0
    host.rollout_sparse_refresh_reject_dwell_count = 0
    host.rollout_sparse_refresh_reason_interval = 0
    host.rollout_sparse_refresh_reason_flag = 0
    host.rollout_sparse_refresh_reason_phase = 0
    host.rollout_sparse_refresh_reason_score_pressure = 0
    host.rollout_q_phi_argmax_executed_agree_count = 0
    host.rollout_q_phi_argmax_executed_total = 0

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
    host.pending_refresh_records: dict[int, list[dict[str, Any]]] = {
        i: [] for i in range(n_envs)
    }
    host.rollout_refresh_records: list[dict[str, Any]] = []
    host.episode_id_per_env = torch.zeros((n_envs,), dtype=torch.long, device=device)
    v3i3_buffer_size = max(
        1, int(getattr(trainer, "latent_v3i3_event_preference_buffer_size", 0) or 50_000)
    )
    host.refresh_preference_buffer: deque = deque(maxlen=v3i3_buffer_size)

    # ------------------------------------------------------------------
    # v3i19 arc-credit channel: per-env state for the currently-open arc.
    #
    # An "arc" begins at every z-sample boundary (episode start, sparse
    # resample, or event refresh) and ends when z is resampled again or
    # the episode terminates. While an arc is open, ``arc_return_accum``
    # grows with the env reward and ``arc_steps_accum`` counts decision
    # steps. On arc end, if ``arc_steps_accum >= latent_arc_credit_min_len``,
    # the snapshot (ctx, z, log_prob, opponent_id, bucket_id, arc_return)
    # is pushed to ``rollout_strategy_arc_records`` for PPO update.
    # ``arc_has_open`` gates the finalize/snapshot side-effects so the
    # very first sample of a rollout (when nothing is open yet) is a
    # no-op for the finalize hook.
    # ------------------------------------------------------------------
    host.arc_open_ctx = torch.zeros(
        (n_envs, int(trainer.model.global_state_dim)), dtype=torch.float32, device=device
    )
    host.arc_open_z = torch.zeros((n_envs,), dtype=torch.long, device=device)
    host.arc_open_log_prob = torch.zeros((n_envs,), dtype=torch.float32, device=device)
    host.arc_open_opponent_id = torch.full((n_envs,), -1, dtype=torch.long, device=device)
    host.arc_open_bucket_id = torch.full((n_envs,), -1, dtype=torch.long, device=device)
    host.arc_return_accum = torch.zeros((n_envs,), dtype=torch.float32, device=device)
    host.arc_steps_accum = torch.zeros((n_envs,), dtype=torch.long, device=device)
    host.arc_has_open = torch.zeros((n_envs,), dtype=torch.bool, device=device)
    # Append-only buffer of finalized arc records consumed by
    # ``apply_arc_strategy_ppo`` at training update time and drained after.
    host.rollout_strategy_arc_records: list[dict[str, Any]] = []
    # Rollout-level telemetry: total arcs finalized, arcs dropped below
    # ``latent_arc_credit_min_len``, mean arc length and return.
    host.rollout_arc_finalized_count = 0
    host.rollout_arc_dropped_short_count = 0
    host.rollout_arc_length_sum = 0
    host.rollout_arc_return_sum = 0.0
    # Running-mean baseline (used when ``latent_arc_credit_baseline ==
    # "running_mean"``). Plain detached EMA, no neural component.
    host.arc_return_running_mean = 0.0
    host.arc_return_running_count = 0

    # V6I1 macro-router segments (64-decision boundaries in Phase B/C).
    host.macro_open_ctx = torch.zeros(
        (n_envs, int(trainer.model.global_state_dim)), dtype=torch.float32, device=device
    )
    host.macro_open_z = torch.zeros((n_envs,), dtype=torch.long, device=device)
    host.macro_open_log_prob = torch.zeros((n_envs,), dtype=torch.float32, device=device)
    host.macro_return_accum = torch.zeros((n_envs,), dtype=torch.float32, device=device)
    host.macro_steps_accum = torch.zeros((n_envs,), dtype=torch.long, device=device)
    host.macro_has_open = torch.zeros((n_envs,), dtype=torch.bool, device=device)
    host.rollout_strategy_macro_records: list[dict[str, Any]] = []
    host.rollout_macro_finalized_count = 0
    host.rollout_macro_dropped_short_count = 0
    host.rollout_macro_length_sum = 0
    host.rollout_macro_return_sum = 0.0
    host.macro_return_running_mean = 0.0
    host.macro_return_running_count = 0
    hidden_dim = int(getattr(trainer.model, "recurrent_selector_hidden_dim", 0) or 0)
    if hidden_dim > 0 and getattr(trainer.model, "selector_gru", None) is not None:
        host.selector_hidden = torch.zeros(
            (n_envs, hidden_dim), dtype=torch.float32, device=device
        )
        host.macro_open_selector_hidden = torch.zeros(
            (n_envs, hidden_dim), dtype=torch.float32, device=device
        )
        host.arc_open_selector_hidden = torch.zeros(
            (n_envs, hidden_dim), dtype=torch.float32, device=device
        )
        host.episode_strategy_selector_hidden = torch.zeros(
            (n_envs, hidden_dim), dtype=torch.float32, device=device
        )
    else:
        host.selector_hidden = None
        host.macro_open_selector_hidden = None
        host.arc_open_selector_hidden = None
        host.episode_strategy_selector_hidden = None
    host.v6i1_episode_rehearsal = torch.zeros((n_envs,), dtype=torch.bool, device=device)

    # V6I1 Staged Curriculum & Competence tracking variables
    host.cf_J = np.zeros((max(1, int(trainer.latent_k)),), dtype=np.float32)
    host.cf_episode_counts = np.zeros((max(1, int(trainer.latent_k)),), dtype=np.int64)
    host.cf_has_experience = np.zeros((max(1, int(trainer.latent_k)),), dtype=np.bool_)
    host.cf_return_mean = 0.0
    host.cf_return_var = 1.0

    host.recent_z_history = deque(maxlen=200)
    # v6i1 macro intervention EMA (legacy single-gate protocol).
    host.pair_jsd_ema = np.zeros((6,), dtype=np.float32)
    host.jsd_gate_consecutive_updates = 0
    host.pairwise_ema_valid_updates = 0
    host.pairwise_ema_last_update_step = -1
    # v6i2 dual-gate protocol: separate actor-CF and macro-rollout EMA tracks.
    host.cf_pair_jsd_ema = np.zeros((6,), dtype=np.float32)
    host.cf_pair_jsd_last_batch = np.zeros((6,), dtype=np.float32)
    host.cf_pair_jsd_valid_updates = 0
    host.cf_pair_jsd_last_update_step = -1
    host.actor_intervention_consecutive_updates = 0
    host.actor_intervention_skipped_gate_count = 0
    host.actor_intervention_last_skipped_gate_step = -1
    host.macro_pair_jsd_ema = np.zeros((6,), dtype=np.float32)
    host.macro_pair_jsd_valid_updates = 0
    host.macro_pair_jsd_last_update_step = -1
    host.router_optimizer_step_count = 0
    hidden_dim = int(getattr(trainer.model, "recurrent_selector_hidden_dim", 0) or 0)
    host.lifecycle = EpisodeLifecycleState(n_envs=n_envs, device=device)
    host.selector_memory = SelectorMemory(n_envs=n_envs, hidden_dim=hidden_dim, device=device)
    host.missing_episode_record_count = 0

    # V6I7 balanced latent assignment counters.
    # balanced_episode_counter: cumulative episode starts per env; used to stagger z across
    #   envs via z = (counter + env_index) % K.
    # arc_step_counter: steps elapsed in current episode; resets to 0 on done.
    # episode_arc_start_z: the starting latent for the current episode in balanced_arc mode;
    #   set on episode start, held until the next episode boundary.
    host.balanced_episode_counter = torch.zeros((n_envs,), dtype=torch.long, device=device)
    host.arc_step_counter = torch.zeros((n_envs,), dtype=torch.long, device=device)
    host.episode_arc_start_z = torch.zeros((n_envs,), dtype=torch.long, device=device)
