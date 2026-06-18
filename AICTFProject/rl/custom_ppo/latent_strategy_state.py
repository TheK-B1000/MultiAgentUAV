"""Core latent z-machine state (resets, telemetry, competence, q_phi param helpers)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from rl.behavior_telemetry import N_TELEMETRY
from rl.custom_ppo.csv_writers import SCRIPTED_OPPONENT_MI_COUNT, _opponent_id_int_from_info
from rl.custom_ppo.latent.optimization.specialist_router import SpecialistRouterManager
from rl.custom_ppo.latent.tensor_state import allocate_latent_state_fields
from rl.custom_ppo.latent.records import stack_selector_hidden_records

if TYPE_CHECKING:
    from rl.custom_ppo.trainer import CustomPPOTrainer

# Backward-compatible re-exports (tests and presets import these names).
from rl.custom_ppo.latent.context_buckets import (  # noqa: F401
    carrier_progress_bucket_ids as _carrier_progress_bucket_ids,
    episode_bucket_baseline_keys as _episode_bucket_baseline_keys,
    flag_state_bucket_ids as _flag_state_bucket_ids,
    role_phase_specialist_context_keys as _role_phase_specialist_context_keys,
    score_pressure_bucket_ids as _score_pressure_bucket_ids,
    specialist_context_keys_for_mode as _specialist_context_keys_for_mode,
    strategy_experience_bucket_ids as _strategy_experience_bucket_ids,
    tactical_local_context_keys as _tactical_local_context_keys,
    tactical_specialist_context_keys as _tactical_specialist_context_keys,
    team_phase_bucket_ids as _team_phase_bucket_ids,
)
from rl.custom_ppo.latent.preferences import (  # noqa: F401
    advantage_weighted_target_from_records as _advantage_weighted_target_from_records,
    router_specialist_coef_scale as _router_specialist_coef_scale,
    router_specialist_loss as _router_specialist_loss,
    v3i3_resolve_target as _v3i3_resolve_target,
    v3i3_target_from_items as _v3i3_target_from_items,
    warmup_ramp_coef_scale as _warmup_ramp_coef_scale,
)

_stack_selector_hidden_records = stack_selector_hidden_records


class LatentStrategyStateCore:
    """Per-env z-machine + episode-credit machinery for the latent strategy.

    Held by the trainer as ``self.latent_state``. The trainer remains the
    owner of ``model``, ``optimizer``, ``cfg``, ``env``, ``device``, and the
    config-derived flags (``use_latent_strategy``, ``fixed_latent_strategy``,
    ``latent_k``, ``latent_resample_every_n``, etc.).
    """

    def __init__(self, trainer: "CustomPPOTrainer") -> None:
        self.trainer = trainer
        allocate_latent_state_fields(self, trainer)
        self._specialist_router = SpecialistRouterManager(self)

    def begin_episodes(self, episode_start_mask: torch.Tensor) -> None:
        """Reset recurrent selector rows and lifecycle before router inference."""
        if not bool(episode_start_mask.any().item()):
            return
        self.selector_memory.reset_rows(episode_start_mask)
        self.lifecycle.begin(episode_start_mask)
        if self.selector_hidden is not None:
            self.selector_hidden[episode_start_mask] = 0.0

    def initialize(self) -> None:
        """Fresh-run initialization (alias for full training-state reset)."""
        self.reset_all_training_state()

    def reset_completed_envs(self, done_mask: torch.Tensor) -> None:
        """Clear per-episode row state for envs that finished an episode."""
        if not bool(done_mask.any().item()):
            return
        self.lifecycle.complete(done_mask)
        self.selector_memory.reset_rows(done_mask)
        self.episode_strategy_has_start[done_mask] = False
        if self.selector_hidden is not None:
            self.selector_hidden[done_mask] = 0.0

    def reset_rollout_statistics(self) -> None:
        """Drain rollout buffers and counters without touching cumulative evidence."""
        self.reset_event_refresh_rollout_stats()
        self.reset_sparse_tactical_refresh_rollout_stats()
        self.reset_behavior_contrast_rollout_stats()
        self.reset_arc_credit_rollout_state()
        self.reset_macro_rollout_state()
        self.rollout_strategy_episode_records = []

    def reset_all_training_state(self) -> None:
        """Full reset for a genuinely fresh run."""
        self.reset()

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
        self.steps_since_last_tactical_refresh.zero_()
        self.steps_since_z_change.zero_()
        self.prev_global_state = None
        self.episode_id_per_env.zero_()
        self.pending_refresh_records = {i: [] for i in range(n_envs)}
        self.rollout_refresh_records = []
        self.refresh_preference_buffer.clear()
        # v3i19 arc-credit per-env state reset: no arc is open at trainer
        # init / rollout-start. The buffer/telemetry reset happens below in
        # ``reset_arc_credit_rollout_state``.
        self.arc_open_ctx.zero_()
        self.arc_open_z.zero_()
        self.arc_open_log_prob.zero_()
        self.arc_open_opponent_id.fill_(-1)
        self.arc_open_bucket_id.fill_(-1)
        self.arc_return_accum.zero_()
        self.arc_steps_accum.zero_()
        self.arc_has_open.zero_()
        self.arc_return_running_mean = 0.0
        self.arc_return_running_count = 0
        self.macro_open_ctx.zero_()
        self.macro_open_z.zero_()
        self.macro_open_log_prob.zero_()
        self.macro_return_accum.zero_()
        self.macro_steps_accum.zero_()
        self.macro_has_open.zero_()
        self.macro_return_running_mean = 0.0
        self.macro_return_running_count = 0
        if self.selector_hidden is not None:
            self.selector_hidden.zero_()
        if self.macro_open_selector_hidden is not None:
            self.macro_open_selector_hidden.zero_()
        if self.arc_open_selector_hidden is not None:
            self.arc_open_selector_hidden.zero_()
        if self.episode_strategy_selector_hidden is not None:
            self.episode_strategy_selector_hidden.zero_()
        self.v6i1_episode_rehearsal.zero_()
        self.reset_event_refresh_rollout_stats()
        self.reset_sparse_tactical_refresh_rollout_stats()
        self.reset_behavior_contrast_rollout_stats()
        self.reset_arc_credit_rollout_state()
        self.reset_macro_rollout_state()

    def reset_arc_credit_rollout_state(self) -> None:
        """Drop the rollout's arc-credit buffer + telemetry counters.

        Does NOT touch the per-env open-arc state (``arc_open_*``,
        ``arc_return_accum``, ``arc_steps_accum``, ``arc_has_open``) because
        those reflect in-flight arcs that span the rollout boundary; only the
        finalized-record buffer + rollout-level counters are drained.
        """
        self.rollout_strategy_arc_records = []
        self.rollout_arc_finalized_count = 0
        self.rollout_arc_dropped_short_count = 0
        self.rollout_arc_length_sum = 0
        self.rollout_arc_return_sum = 0.0

    def _v6i1_macro_enabled(self) -> bool:
        from rl.custom_ppo.v6i1_phase_runtime import v6i1_macro_router_active

        return v6i1_macro_router_active(self.trainer)

    def reset_macro_rollout_state(self) -> None:
        self.rollout_strategy_macro_records = []
        self.rollout_macro_finalized_count = 0
        self.rollout_macro_dropped_short_count = 0
        self.rollout_macro_length_sum = 0
        self.rollout_macro_return_sum = 0.0
    def reset_event_refresh_rollout_stats(self) -> None:
        self.rollout_refresh_count = 0
        self.rollout_refresh_z_changed_count = 0
        self.rollout_refresh_reason_enemy_flag = 0
        self.rollout_refresh_reason_friendly_flag = 0
        self.rollout_refresh_reason_score_change = 0
        self.rollout_refresh_reason_near_base = 0
        self.rollout_refresh_total_steps = 0
        self.rollout_refresh_transitions.fill(0.0)

    def reset_sparse_tactical_refresh_rollout_stats(self) -> None:
        self.rollout_sparse_z_change_count = 0
        self.rollout_sparse_z_dwell_sum = 0.0
        self.rollout_sparse_z_dwell_count = 0
        self.rollout_sparse_refresh_attempt_count = 0
        self.rollout_sparse_refresh_accept_count = 0
        self.rollout_sparse_refresh_reject_dwell_count = 0
        self.rollout_sparse_refresh_reason_interval = 0
        self.rollout_sparse_refresh_reason_flag = 0
        self.rollout_sparse_refresh_reason_phase = 0
        self.rollout_sparse_refresh_reason_score_pressure = 0
        self.rollout_q_phi_argmax_executed_agree_count = 0
        self.rollout_q_phi_argmax_executed_total = 0

    def sparse_tactical_refresh_rollout_stats(self) -> dict[str, float]:
        enabled = bool(
            getattr(
                self.trainer,
                "latent_sparse_tactical_refresh_enabled",
                False,
            )
        )
        if not enabled:
            return {
                "z_change_count": 0.0,
                "z_dwell_mean": 0.0,
                "z_refresh_attempt_count": 0.0,
                "z_refresh_accept_count": 0.0,
                "z_refresh_reject_dwell_count": 0.0,
                "z_refresh_reason_interval": 0.0,
                "z_refresh_reason_flag": 0.0,
                "z_refresh_reason_phase": 0.0,
                "z_refresh_reason_score_pressure": 0.0,
                "q_phi_argmax_vs_executed_z_agreement": 0.0,
            }
        dwell_mean = (
            self.rollout_sparse_z_dwell_sum
            / float(self.rollout_sparse_z_dwell_count)
            if self.rollout_sparse_z_dwell_count > 0
            else 0.0
        )
        agreement = (
            float(self.rollout_q_phi_argmax_executed_agree_count)
            / float(self.rollout_q_phi_argmax_executed_total)
            if self.rollout_q_phi_argmax_executed_total > 0
            else 0.0
        )
        return {
            "z_change_count": float(self.rollout_sparse_z_change_count),
            "z_dwell_mean": float(dwell_mean),
            "z_refresh_attempt_count": float(
                self.rollout_sparse_refresh_attempt_count
            ),
            "z_refresh_accept_count": float(
                self.rollout_sparse_refresh_accept_count
            ),
            "z_refresh_reject_dwell_count": float(
                self.rollout_sparse_refresh_reject_dwell_count
            ),
            "z_refresh_reason_interval": float(
                self.rollout_sparse_refresh_reason_interval
            ),
            "z_refresh_reason_flag": float(
                self.rollout_sparse_refresh_reason_flag
            ),
            "z_refresh_reason_phase": float(
                self.rollout_sparse_refresh_reason_phase
            ),
            "z_refresh_reason_score_pressure": float(
                self.rollout_sparse_refresh_reason_score_pressure
            ),
            "q_phi_argmax_vs_executed_z_agreement": float(agreement),
        }

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
        self.rollout_forced_z_episode_count_by_z[:] = 0
        self.rollout_forced_episode_count_by_opp_z[:] = 0

    def behavior_contrast_coef(self) -> float:
        trainer = self.trainer
        base = max(0.0, float(getattr(trainer, "latent_behavior_contrast_coef", 0.0) or 0.0))
        after = max(0, int(getattr(trainer, "latent_behavior_contrast_anneal_after_steps", 0) or 0))
        if after <= 0 or int(getattr(trainer, "global_step", 0) or 0) < after:
            return base
        return max(0.0, float(getattr(trainer, "latent_behavior_contrast_anneal_to", 0.0) or 0.0))
    def _strategy_encoder_params(self) -> list[torch.nn.Parameter]:
        """Return params of q_phi's routing network (``strategy_encoder``).

        These are the parameters that control ``pi(z | s)`` -- the actual
        routing decision the smoke alarm should measure. Separated from the
        value head so we can answer "where does the arc-credit gradient
        land?" diagnostically.
        """
        trainer = self.trainer
        strategy_encoder = getattr(trainer.model, "strategy_encoder", None)
        selector_gru = getattr(trainer.model, "selector_gru", None)
        params: list[torch.nn.Parameter] = []
        if strategy_encoder is not None:
            params.extend(p for p in strategy_encoder.parameters() if p.requires_grad)
        if selector_gru is not None:
            params.extend(p for p in selector_gru.parameters() if p.requires_grad)
        return params

    def _value_head_params(self) -> list[torch.nn.Parameter]:
        """Return params of ``episode_strategy_value_head`` (V_phi(s, z)).

        These are the baseline params. Updated by the arc-credit ``v_loss``
        but DO NOT influence ``pi(z | s)``. Splitting their grad norm out of
        the combined q_phi norm tells us whether arc credit is training the
        router (good) or just the baseline (decorative).
        """
        trainer = self.trainer
        value_head = getattr(trainer.model, "episode_strategy_value_head", None)
        if value_head is None:
            return []
        return [p for p in value_head.parameters() if p.requires_grad]

    def _q_phi_params(self) -> list[torch.nn.Parameter]:
        """Return the full q_phi parameter list (router + value head).

        Used by ``apply_arc_strategy_ppo`` to compute the combined
        ``q_phi_grad_norm`` AFTER backward and BEFORE optimizer.step().
        The split into ``_strategy_encoder_params`` + ``_value_head_params``
        is exhaustive given current model architecture, but we still emit a
        ``q_phi_other_grad_norm`` field as a defensive drift detector in
        case future model changes add unaccounted parameters.
        """
        return self._strategy_encoder_params() + self._value_head_params()

    @staticmethod
    def _grad_norm_l2(params: list[torch.nn.Parameter]) -> float:
        """L2 grad norm over ``params``. Returns 0.0 when all grads are None."""
        sq_sum = 0.0
        any_grad = False
        for p in params:
            g = p.grad
            if g is None:
                continue
            any_grad = True
            sq_sum += float(g.detach().pow(2).sum().item())
        if not any_grad:
            return 0.0
        return float(sq_sum ** 0.5)
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
        stats: dict[str, float] = {
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
        k = max(1, int(self.trainer.latent_k))
        for z_i in range(k):
            forced_i = int(self.rollout_forced_z_episode_count_by_z[z_i])
            stats[f"forced_sample_count_by_z_{z_i}"] = float(forced_i)
            stats[f"episode_count_by_z_{z_i}"] = float(forced_i)
        for o_idx in range(int(SCRIPTED_OPPONENT_MI_COUNT)):
            for z_i in range(k):
                stats[f"forced_episode_opp{o_idx}_z{z_i}_count"] = float(
                    self.rollout_forced_episode_count_by_opp_z[o_idx, z_i]
                )
        return stats
    def compute_competence_scores(self) -> tuple[np.ndarray, bool]:
        """Compute sigmoid competence scores for each latent."""
        trainer = self.trainer
        min_eps = int(getattr(trainer.cfg, "latent_cf_min_episodes_per_z", 50))
        delta = float(getattr(trainer.cfg, "latent_cf_competence_delta", 5.0))
        T_c = float(getattr(trainer.cfg, "latent_cf_competence_gate_tc", 1.0))
        
        # If any latent has fewer than min_eps completed episodes, they are not ready
        if any(count < min_eps for count in self.cf_episode_counts):
            return np.zeros((self.trainer.latent_k,), dtype=np.float32), False
            
        J_best = float(np.max(self.cf_J))
        sigma_J = float(np.sqrt(max(1e-8, self.cf_return_var)))
        scale = max(T_c, sigma_J, 1e-8)
        
        # c_z = sigmoid( (J_z - J_best + delta) / scale )
        c_z = 1.0 / (1.0 + np.exp(- (self.cf_J - J_best + delta) / scale))
        return c_z, True

    def apply_rollout_specialist_router(self, buffer: Any) -> dict[str, float]:
        return self._specialist_router.apply_rollout_specialist_router(buffer)

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


def __getattr__(name: str):
    if name == "LatentStrategyState":
        from rl.custom_ppo.latent.state import LatentStrategyState as _LS
        return _LS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
