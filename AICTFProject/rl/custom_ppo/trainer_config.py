"""Trainer-side hyperparameter resolution.

Historically :class:`~rl.custom_ppo.trainer.CustomPPOTrainer` resolved its
runtime hyperparameters with ~50 inline ``getattr(cfg, "...", default)``
calls scattered through its constructor. That conflated three things:

1. *Schema*: what fields the trainer actually depends on.
2. *Defaults*: the fallback values used when the legacy ``PPOConfig`` /
   checkpoint cfg-dict omits a field.
3. *Coercion / clamping*: ``max(0.0, float(...))`` style bounds.

This module centralizes all three behind a single immutable
:class:`TrainerHyperparams` frozen dataclass and a
:meth:`TrainerHyperparams.from_ppo_config` factory. The trainer can then
do a single bulk assignment from a typed object instead of dozens of
inline ``getattr`` reads.

Backward compatibility:
    * The trainer still keeps ``self.cfg`` as the original ``PPOConfig`` /
      cfg-like object — many downstream modules (``RolloutCollector``,
      ``PPOUpdater``, ``TrainingTelemetry``, etc.) read fields off
      ``trainer.cfg`` for values that don't need pre-resolution (e.g.
      ``cfg.gamma``, ``cfg.gae_lambda``, ``cfg.max_grad_norm``,
      ``cfg.target_kl``, ``cfg.run_tag``).
    * The trainer also keeps ``self.hparams`` as the single source of truth
      for resolved hyperparameters; legacy ``trainer.<name>`` attribute reads
      are forwarded from :attr:`hparams` via
      :meth:`~rl.custom_ppo.trainer.CustomPPOTrainer.__getattr__`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from rl.custom_ppo.curriculum_gates import is_staged_v6i1_curriculum
from rl.latent_marl import CONTEXT_STATE_DIM


def router_current_plus_delta_enabled(cfg: Any) -> bool:
    return str(getattr(cfg, "router_context_mode", "") or "") == "current_plus_delta"


def router_current_plus_delta_dim(cfg: Any) -> int:
    return int(getattr(cfg, "router_context_dimension", 0) or 0)


@dataclass(frozen=True)
class TrainerHyperparams:
    """Immutable resolved trainer hyperparameters.

    Built once from the raw ``PPOConfig`` (or a cfg-dict) via
    :meth:`from_ppo_config`. The trainer stores the result on
    :attr:`~rl.custom_ppo.trainer.CustomPPOTrainer.hparams` and serves
    legacy ``trainer.<name>`` reads through :meth:`~rl.custom_ppo.trainer.CustomPPOTrainer.__getattr__`.

    All fields are derived; nothing is owned by the trainer that mutates
    them (use ``trainer.global_step`` / ``trainer.last_stats`` / etc. for
    runtime state). Reward-shaping schedule fields (``..._start``,
    ``..._end``, ``..._decay_steps``) stay here because the trainer
    interpolates them per step at runtime.
    """

    # ----- optimization / PPO -----
    learning_rate: float
    clip_range: float
    value_clip_range: float | None
    ent_coef: float
    n_epochs: int
    batch_size: int
    vf_coef: float
    normalize_returns: bool

    # ----- latent strategy gating -----
    use_latent_strategy: bool
    latent_k: int
    latent_resample_every_n: int
    fixed_latent_strategy: bool
    fixed_latent_strategy_id: int
    latent_gae_reset_on_z_change: bool
    latent_bootstrap_z_deterministic: bool
    latent_resample_on_flag: bool
    latent_kl_consecutive: float

    # ----- latent strategy losses / coefficients -----
    latent_strategy_ppo_coef: float
    latent_episode_strategy_ppo: bool
    latent_episode_strategy_coef: float
    latent_episode_strategy_clip_eps: float
    latent_episode_strategy_value_coef: float
    latent_episode_strategy_return_norm: bool
    latent_episode_strategy_warmup_decision_steps: int
    latent_episode_strategy_n_epochs: int
    latent_episode_strategy_lr: Optional[float]
    # v3i19 arc-credit channel (Summer-faithful per-arc q_phi PPO gradient).
    latent_arc_credit_enabled: bool
    latent_arc_credit_coef: float
    latent_arc_credit_n_epochs: int
    latent_arc_credit_clip_eps: float
    latent_arc_credit_return_norm: bool
    latent_arc_credit_baseline: str
    latent_arc_credit_min_len: int
    latent_q_phi_marginal_baseline: bool
    latent_q_phi_bucket_baseline: Optional[str]
    latent_q_phi_bucket_baseline_ema: float
    latent_q_phi_bucket_baseline_min_count: int
    latent_strategy_aux_return_coef: float
    latent_strategy_aux_return_head: bool
    latent_strategy_aux_predict_phase_coef: float
    latent_forced_z_episode_frac: float
    latent_behavior_contrast_coef: float
    latent_behavior_contrast_margin: float
    latent_behavior_contrast_ema: float
    latent_behavior_contrast_anneal_after_steps: int
    latent_behavior_contrast_anneal_to: float
    latent_outcome_diversity_coef: float
    latent_outcome_diversity_margin: float
    latent_outcome_diversity_ema: float
    latent_outcome_diversity_success_only: bool
    latent_actor_z_separation_coef: float
    latent_actor_z_separation_start_coef: float
    latent_actor_z_separation_margin: float
    latent_actor_z_separation_min_abs_advantage: float
    latent_actor_z_separation_min_decision_frac: float
    latent_actor_z_separation_max_entropy_frac: float
    latent_actor_z_adapter_enabled: bool
    latent_actor_z_adapter_scale: float
    latent_actor_z_film_layers: int
    latent_actor_z_adapter_warmup_steps: int
    latent_actor_z_adapter_ramp_steps: int
    latent_actor_z_separation_warmup_steps: int
    latent_actor_z_separation_ramp_steps: int
    latent_usage_balance_coef: float
    latent_q_phi_train_after_steps: int
    latent_preference_coef: float
    latent_preference_temperature: float
    latent_preference_min_bucket_count: int
    latent_preference_min_distinct_z: int
    latent_preference_opponent_balanced: bool
    latent_preference_log_opponent_targets: bool
    latent_preference_confidence_scale: float
    latent_preference_commit_coef: float
    latent_awrd_enabled: bool
    latent_awrd_coef: float
    latent_awrd_temperature: float
    latent_awrd_min_bucket_count: int
    latent_awrd_min_distinct_z: int
    latent_awrd_margin_threshold: float
    latent_awrd_margin_scale: float
    latent_awrd_min_margin: float
    latent_awrd_soft_margin_gating: bool
    latent_awrd_warmup_steps: int
    latent_awrd_ramp_steps: int
    latent_specialist_router_enabled: bool
    latent_marginal_balance_coef: float
    latent_conditional_entropy_min_coef: float
    latent_conditional_entropy_min_coef_start: float
    latent_specialist_conditional_entropy_scope: str
    latent_context_mi_coef: float
    latent_specialist_warmup_steps: int
    latent_specialist_ramp_steps: int
    latent_specialist_min_bucket_count: int
    latent_specialist_context_key_mode: str
    latent_specialist_use_rollout_states: bool
    latent_specialist_rollout_max_samples: int
    late_entropy_floor: float
    commitment_type: str
    latent_event_refresh_enabled: bool
    latent_event_refresh_min_gap_steps: int
    latent_event_refresh_max_per_episode: int
    latent_event_refresh_use_q_phi: bool
    latent_event_refresh_force_roles: bool
    latent_sparse_tactical_refresh_enabled: bool
    latent_sparse_tactical_refresh_interval_steps: int
    latent_sparse_tactical_refresh_min_dwell_steps: int
    latent_v3i3_event_preference_enabled: bool
    latent_v3i3_event_preference_coef: float
    latent_v3i3_event_preference_temperature: float
    latent_v3i3_event_preference_min_bucket_count: int
    latent_v3i3_event_preference_min_distinct_z: int
    latent_v3i3_event_preference_buffer_size: int
    latent_v3i3_event_preference_warmup_steps: int
    latent_v3i3_event_preference_normalize: bool
    latent_v3i3_refresh_log_enabled: bool
    latent_v3i3_refresh_log_path: str
    latent_event_preference_key_mode: str

    # ----- reward composition (env-driven defaults) -----
    reward_dense_weight: float
    reward_scale: float
    reward_clip: float
    reward_stalemate_penalty: float

    # ----- reward shaping schedule -----
    reward_shaping_coef_start: float
    reward_shaping_coef_end: float
    reward_shaping_decay_steps: int

    # ----- checkpointing -----
    periodic_checkpoint_steps: int

    # ----- run identity / telemetry paths -----
    run_id: str
    run_pid: int
    metrics_csv_path: str
    episode_csv_path: str
    strategy_experience_csv_path: str

    # ----- opponent pool training -----
    opponent_randomize_training: bool
    opponent_pool_tags: tuple[str, ...] = field(default_factory=tuple)
    # Empty tuple ⇒ uniform sampling. Non-empty must align positionally with
    # ``opponent_pool_tags`` and sum to 1.0 (normalized upstream).
    opponent_pool_weights: tuple[float, ...] = field(default_factory=tuple)

    @classmethod
    def from_ppo_config(
        cls,
        cfg: Any,
        env: Any,
        *,
        learning_rate: float,
        clip_range: float,
        ent_coef: float,
        n_epochs: int,
        batch_size: int,
        value_clip_range: float | None,
        curriculum: Any | None,
    ) -> "TrainerHyperparams":
        """Resolve a :class:`TrainerHyperparams` from a ``PPOConfig`` (or cfg-like).

        ``env`` is needed because reward-composition defaults live on
        ``env.cfg`` rather than the PPOConfig. ``curriculum`` participates
        in the opponent-pool gate: curriculum-driven runs disable the
        flat opponent-pool path even when ``cfg.mode == OPPONENT_POOL``
        (the curriculum supplies its own opponent each episode).
        """
        use_latent = bool(getattr(cfg, "use_latent_strategy", False))
        latent_k = int(getattr(cfg, "latent_k", 4)) if use_latent else 0
        fixed_latent = use_latent and bool(getattr(cfg, "fixed_latent_strategy", False))
        fixed_latent_id = (
            max(0, min(int(getattr(cfg, "fixed_latent_strategy_id", 0) or 0), latent_k - 1))
            if use_latent
            else 0
        )
        latent_gae_reset = bool(getattr(cfg, "latent_gae_reset_on_z_change", True)) and (
            use_latent and not fixed_latent
        )

        env_cfg = getattr(env, "cfg", None)

        mode_s = str(getattr(cfg, "mode", "") or "").strip().upper()
        opp_randomize = (
            (mode_s == "OPPONENT_POOL" or bool(getattr(cfg, "opponent_randomize", False)))
            and curriculum is None
        )
        opp_tags: tuple[str, ...] = (
            tuple(str(x).strip().upper() for x in getattr(cfg, "opponent_pool", ()))
            if opp_randomize
            else ()
        )
        opp_weights: tuple[float, ...] = (
            tuple(float(w) for w in getattr(cfg, "opponent_pool_weights", ()) or ())
            if opp_randomize
            else ()
        )

        return cls(
            learning_rate=float(learning_rate),
            clip_range=float(clip_range),
            value_clip_range=None if value_clip_range is None else float(value_clip_range),
            ent_coef=float(ent_coef),
            n_epochs=int(n_epochs),
            batch_size=int(batch_size),
            vf_coef=max(0.0, float(getattr(cfg, "vf_coef", 1.0) or 0.0)),
            normalize_returns=bool(getattr(cfg, "normalize_returns", False)),
            use_latent_strategy=use_latent,
            latent_k=latent_k,
            latent_resample_every_n=max(0, int(getattr(cfg, "latent_resample_every_n", 0) or 0)),
            fixed_latent_strategy=fixed_latent,
            fixed_latent_strategy_id=fixed_latent_id,
            latent_gae_reset_on_z_change=latent_gae_reset,
            latent_bootstrap_z_deterministic=bool(
                getattr(cfg, "latent_bootstrap_z_deterministic", True)
            ),
            latent_resample_on_flag=(
                bool(getattr(cfg, "latent_resample_on_flag", False))
                and use_latent
                and not fixed_latent
            ),
            latent_kl_consecutive=(
                max(0.0, float(getattr(cfg, "latent_kl_consecutive", 0.0) or 0.0))
                if use_latent and not fixed_latent
                else 0.0
            ),
            latent_strategy_ppo_coef=max(
                0.0, float(getattr(cfg, "latent_strategy_ppo_coef", 0.1) or 0.0)
            ),
            latent_episode_strategy_ppo=(
                use_latent
                and not fixed_latent
                and bool(getattr(cfg, "latent_episode_strategy_ppo", False))
            ),
            latent_episode_strategy_coef=max(
                0.0, float(getattr(cfg, "latent_episode_strategy_coef", 0.0) or 0.0)
            ),
            latent_episode_strategy_clip_eps=max(
                1e-6, float(getattr(cfg, "latent_episode_strategy_clip_eps", 0.2) or 0.2)
            ),
            latent_episode_strategy_value_coef=max(
                0.0, float(getattr(cfg, "latent_episode_strategy_value_coef", 0.5) or 0.0)
            ),
            latent_episode_strategy_return_norm=bool(
                getattr(cfg, "latent_episode_strategy_return_norm", True)
            ),
            latent_episode_strategy_warmup_decision_steps=max(
                0,
                int(getattr(cfg, "latent_episode_strategy_warmup_decision_steps", 0) or 0),
            ),
            latent_episode_strategy_n_epochs=max(
                1, int(getattr(cfg, "latent_episode_strategy_n_epochs", 1) or 1)
            ),
            latent_episode_strategy_lr=(
                float(getattr(cfg, "latent_episode_strategy_lr", None))
                if getattr(cfg, "latent_episode_strategy_lr", None) is not None
                else None
            ),
            latent_arc_credit_enabled=(
                use_latent
                and not fixed_latent
                and bool(getattr(cfg, "latent_arc_credit_enabled", False))
            ),
            latent_arc_credit_coef=max(
                0.0, float(getattr(cfg, "latent_arc_credit_coef", 1.0) or 0.0)
            ),
            latent_arc_credit_n_epochs=max(
                1, int(getattr(cfg, "latent_arc_credit_n_epochs", 4) or 1)
            ),
            latent_arc_credit_clip_eps=max(
                1e-6, float(getattr(cfg, "latent_arc_credit_clip_eps", 0.2) or 0.2)
            ),
            latent_arc_credit_return_norm=bool(
                getattr(cfg, "latent_arc_credit_return_norm", True)
            ),
            latent_arc_credit_baseline=str(
                getattr(cfg, "latent_arc_credit_baseline", "context_value")
                or "context_value"
            ).lower(),
            latent_arc_credit_min_len=max(
                1, int(getattr(cfg, "latent_arc_credit_min_len", 32) or 1)
            ),
            latent_q_phi_marginal_baseline=bool(
                getattr(cfg, "latent_q_phi_marginal_baseline", False)
            ),
            latent_q_phi_bucket_baseline=(
                str(getattr(cfg, "latent_q_phi_bucket_baseline", None))
                if getattr(cfg, "latent_q_phi_bucket_baseline", None)
                else None
            ),
            latent_q_phi_bucket_baseline_ema=float(
                getattr(cfg, "latent_q_phi_bucket_baseline_ema", 0.9) or 0.0
            ),
            latent_q_phi_bucket_baseline_min_count=max(
                1, int(getattr(cfg, "latent_q_phi_bucket_baseline_min_count", 8) or 1)
            ),
            # Canonical attribute access only — legacy ``latent_strategy_q_*`` keys are
            # folded at the config-load boundary (see
            # ``rl.custom_ppo.inference.canonicalize_latent_strategy_cfg`` and the CLI
            # argparse handler).
            latent_strategy_aux_return_coef=max(
                0.0, float(getattr(cfg, "latent_strategy_aux_return_coef", 0.0) or 0.0)
            ),
            latent_strategy_aux_return_head=(
                use_latent and bool(getattr(cfg, "latent_strategy_aux_return_head", False))
            ),
            latent_strategy_aux_predict_phase_coef=max(
                0.0,
                float(getattr(cfg, "latent_strategy_aux_predict_phase_coef", 0.0) or 0.0),
            ),
            latent_forced_z_episode_frac=(
                min(max(float(getattr(cfg, "latent_forced_z_episode_frac", 0.0) or 0.0), 0.0), 1.0)
                if use_latent and not fixed_latent
                else 0.0
            ),
            latent_behavior_contrast_coef=(
                max(0.0, float(getattr(cfg, "latent_behavior_contrast_coef", 0.0) or 0.0))
                if use_latent and not fixed_latent
                else 0.0
            ),
            latent_behavior_contrast_margin=max(
                1e-6, float(getattr(cfg, "latent_behavior_contrast_margin", 0.25) or 0.25)
            ),
            latent_behavior_contrast_ema=min(
                max(float(getattr(cfg, "latent_behavior_contrast_ema", 0.9) or 0.0), 0.0),
                0.999,
            ),
            latent_behavior_contrast_anneal_after_steps=max(
                0, int(getattr(cfg, "latent_behavior_contrast_anneal_after_steps", 0) or 0)
            ),
            latent_behavior_contrast_anneal_to=max(
                0.0, float(getattr(cfg, "latent_behavior_contrast_anneal_to", 0.0) or 0.0)
            ),
            latent_outcome_diversity_coef=(
                max(0.0, float(getattr(cfg, "latent_outcome_diversity_coef", 0.0) or 0.0))
                if use_latent and not fixed_latent
                else 0.0
            ),
            latent_outcome_diversity_margin=max(
                1e-6, float(getattr(cfg, "latent_outcome_diversity_margin", 1.0) or 1.0)
            ),
            latent_outcome_diversity_ema=min(
                max(float(getattr(cfg, "latent_outcome_diversity_ema", 0.9) or 0.0), 0.0),
                0.999,
            ),
            latent_outcome_diversity_success_only=bool(
                getattr(cfg, "latent_outcome_diversity_success_only", True)
            ),
            latent_actor_z_separation_coef=(
                max(0.0, float(getattr(cfg, "latent_actor_z_separation_coef", 0.0) or 0.0))
                if use_latent and not fixed_latent
                else 0.0
            ),
            latent_actor_z_separation_start_coef=max(
                0.0,
                float(
                    getattr(cfg, "latent_actor_z_separation_start_coef", 0.0)
                    or 0.0
                ),
            ),
            latent_actor_z_separation_margin=max(
                0.0, float(getattr(cfg, "latent_actor_z_separation_margin", 0.02) or 0.0)
            ),
            latent_actor_z_separation_min_abs_advantage=max(
                0.0,
                float(
                    getattr(
                        cfg,
                        "latent_actor_z_separation_min_abs_advantage",
                        0.0,
                    )
                    or 0.0
                ),
            ),
            latent_actor_z_separation_min_decision_frac=min(
                1.0,
                max(
                    0.0,
                    float(
                        getattr(
                            cfg,
                            "latent_actor_z_separation_min_decision_frac",
                            0.0,
                        )
                        or 0.0
                    ),
                ),
            ),
            latent_actor_z_separation_max_entropy_frac=min(
                1.0,
                max(
                    0.0,
                    float(
                        getattr(
                            cfg,
                            "latent_actor_z_separation_max_entropy_frac",
                            1.0,
                        )
                        if getattr(
                            cfg,
                            "latent_actor_z_separation_max_entropy_frac",
                            1.0,
                        )
                        is not None
                        else 1.0
                    ),
                ),
            ),
            latent_actor_z_adapter_enabled=bool(
                getattr(cfg, "latent_actor_z_adapter_enabled", False)
            ),
            latent_actor_z_adapter_scale=max(
                0.0, float(getattr(cfg, "latent_actor_z_adapter_scale", 0.0) or 0.0)
            ),
            latent_actor_z_film_layers=max(
                1,
                min(2, int(getattr(cfg, "latent_actor_z_film_layers", 1) or 1)),
            ),
            latent_actor_z_adapter_warmup_steps=max(
                0, int(getattr(cfg, "latent_actor_z_adapter_warmup_steps", 0) or 0)
            ),
            latent_actor_z_adapter_ramp_steps=max(
                0, int(getattr(cfg, "latent_actor_z_adapter_ramp_steps", 0) or 0)
            ),
            latent_actor_z_separation_warmup_steps=max(
                0, int(getattr(cfg, "latent_actor_z_separation_warmup_steps", 0) or 0)
            ),
            latent_actor_z_separation_ramp_steps=max(
                0, int(getattr(cfg, "latent_actor_z_separation_ramp_steps", 0) or 0)
            ),
            latent_usage_balance_coef=(
                max(0.0, float(getattr(cfg, "latent_usage_balance_coef", 0.0) or 0.0))
                if use_latent and not fixed_latent
                else 0.0
            ),
            latent_q_phi_train_after_steps=max(
                0, int(getattr(cfg, "latent_q_phi_train_after_steps", 0) or 0)
            ),
            latent_preference_coef=max(
                0.0, float(getattr(cfg, "latent_preference_coef", 0.0) or 0.0)
            ),
            latent_preference_temperature=max(
                1e-6, float(getattr(cfg, "latent_preference_temperature", 0.75) or 0.75)
            ),
            latent_preference_min_bucket_count=max(
                1, int(getattr(cfg, "latent_preference_min_bucket_count", 8) or 8)
            ),
            latent_preference_min_distinct_z=max(
                1, int(getattr(cfg, "latent_preference_min_distinct_z", 2) or 2)
            ),
            latent_preference_opponent_balanced=bool(getattr(cfg, "latent_preference_opponent_balanced", False)),
            latent_preference_log_opponent_targets=bool(getattr(cfg, "latent_preference_log_opponent_targets", False)),
            latent_preference_confidence_scale=float(getattr(cfg, "latent_preference_confidence_scale", 2.0) or 2.0),
            latent_preference_commit_coef=float(getattr(cfg, "latent_preference_commit_coef", 0.0) or 0.0),
            latent_awrd_enabled=(
                use_latent
                and not fixed_latent
                and bool(getattr(cfg, "latent_awrd_enabled", False))
            ),
            latent_awrd_coef=max(
                0.0, float(getattr(cfg, "latent_awrd_coef", 0.0))
            ),
            latent_awrd_temperature=max(
                1e-6, float(getattr(cfg, "latent_awrd_temperature", 0.35))
            ),
            latent_awrd_min_bucket_count=max(
                1, int(getattr(cfg, "latent_awrd_min_bucket_count", 8))
            ),
            latent_awrd_min_distinct_z=max(
                1, int(getattr(cfg, "latent_awrd_min_distinct_z", 2))
            ),
            latent_awrd_margin_threshold=max(
                0.0, float(getattr(cfg, "latent_awrd_margin_threshold", 0.15))
            ),
            latent_awrd_margin_scale=max(
                0.0, float(getattr(cfg, "latent_awrd_margin_scale", 2.0))
            ),
            latent_awrd_min_margin=float(getattr(cfg, "latent_awrd_min_margin", 0.08)),
            latent_awrd_soft_margin_gating=bool(getattr(cfg, "latent_awrd_soft_margin_gating", False)),
            latent_awrd_warmup_steps=max(
                0, int(getattr(cfg, "latent_awrd_warmup_steps", 0) or 0)
            ),
            latent_awrd_ramp_steps=max(
                0, int(getattr(cfg, "latent_awrd_ramp_steps", 0) or 0)
            ),
            latent_specialist_router_enabled=(
                use_latent
                and not fixed_latent
                and bool(getattr(cfg, "latent_specialist_router_enabled", False))
            ),
            latent_marginal_balance_coef=max(
                0.0, float(getattr(cfg, "latent_marginal_balance_coef", 0.0) or 0.0)
            ),
            latent_conditional_entropy_min_coef=max(
                0.0,
                float(getattr(cfg, "latent_conditional_entropy_min_coef", 0.0) or 0.0),
            ),
            latent_conditional_entropy_min_coef_start=max(
                0.0,
                float(
                    getattr(
                        cfg,
                        "latent_conditional_entropy_min_coef_start",
                        0.0,
                    )
                    or 0.0
                ),
            ),
            latent_specialist_conditional_entropy_scope=str(
                getattr(
                    cfg,
                    "latent_specialist_conditional_entropy_scope",
                    "state",
                )
                or "state"
            ),
            latent_context_mi_coef=max(
                0.0, float(getattr(cfg, "latent_context_mi_coef", 0.0) or 0.0)
            ),
            latent_specialist_warmup_steps=max(
                0, int(getattr(cfg, "latent_specialist_warmup_steps", 0) or 0)
            ),
            latent_specialist_ramp_steps=max(
                0, int(getattr(cfg, "latent_specialist_ramp_steps", 1) or 0)
            ),
            latent_specialist_min_bucket_count=max(
                1, int(getattr(cfg, "latent_specialist_min_bucket_count", 2) or 2)
            ),
            latent_specialist_context_key_mode=str(
                getattr(cfg, "latent_specialist_context_key_mode", "opponent_bucket")
                or "opponent_bucket"
            ),
            latent_specialist_use_rollout_states=bool(
                getattr(cfg, "latent_specialist_use_rollout_states", False)
            ),
            latent_specialist_rollout_max_samples=max(
                1,
                int(
                    getattr(cfg, "latent_specialist_rollout_max_samples", 8192)
                    or 8192
                ),
            ),
            late_entropy_floor=float(getattr(cfg, "late_entropy_floor", 0.0003) or 0.0003),
            commitment_type=str(getattr(cfg, "commitment_type", "confidence_weighted_entropy") or "confidence_weighted_entropy"),
            latent_event_refresh_enabled=bool(getattr(cfg, "latent_event_refresh_enabled", False)),
            latent_event_refresh_min_gap_steps=int(getattr(cfg, "latent_event_refresh_min_gap_steps", 20)),
            latent_event_refresh_max_per_episode=int(getattr(cfg, "latent_event_refresh_max_per_episode", 3)),
            latent_event_refresh_use_q_phi=bool(getattr(cfg, "latent_event_refresh_use_q_phi", True)),
            latent_event_refresh_force_roles=bool(getattr(cfg, "latent_event_refresh_force_roles", False)),
            latent_sparse_tactical_refresh_enabled=(
                use_latent
                and not fixed_latent
                and bool(
                    getattr(
                        cfg,
                        "latent_sparse_tactical_refresh_enabled",
                        False,
                    )
                )
            ),
            latent_sparse_tactical_refresh_interval_steps=max(
                1,
                int(
                    getattr(
                        cfg,
                        "latent_sparse_tactical_refresh_interval_steps",
                        32,
                    )
                    or 32
                ),
            ),
            latent_sparse_tactical_refresh_min_dwell_steps=max(
                1,
                int(
                    getattr(
                        cfg,
                        "latent_sparse_tactical_refresh_min_dwell_steps",
                        16,
                    )
                    or 16
                ),
            ),
            latent_v3i3_event_preference_enabled=(
                use_latent
                and not fixed_latent
                and bool(getattr(cfg, "latent_v3i3_event_preference_enabled", False))
            ),
            latent_v3i3_event_preference_coef=max(
                0.0, float(getattr(cfg, "latent_v3i3_event_preference_coef", 0.0) or 0.0)
            ),
            latent_v3i3_event_preference_temperature=max(
                1e-6, float(getattr(cfg, "latent_v3i3_event_preference_temperature", 0.75) or 0.75)
            ),
            latent_v3i3_event_preference_min_bucket_count=max(
                1, int(getattr(cfg, "latent_v3i3_event_preference_min_bucket_count", 4) or 4)
            ),
            latent_v3i3_event_preference_min_distinct_z=max(
                1, int(getattr(cfg, "latent_v3i3_event_preference_min_distinct_z", 2) or 2)
            ),
            latent_v3i3_event_preference_buffer_size=max(
                1, int(getattr(cfg, "latent_v3i3_event_preference_buffer_size", 50_000) or 50_000)
            ),
            latent_v3i3_event_preference_warmup_steps=max(
                0, int(getattr(cfg, "latent_v3i3_event_preference_warmup_steps", 0) or 0)
            ),
            latent_v3i3_event_preference_normalize=bool(
                getattr(cfg, "latent_v3i3_event_preference_normalize", False)
            ),
            latent_v3i3_refresh_log_enabled=bool(
                getattr(cfg, "latent_v3i3_refresh_log_enabled", False)
            ),
            latent_v3i3_refresh_log_path=str(
                getattr(cfg, "latent_v3i3_refresh_log_path", "") or ""
            ),
            latent_event_preference_key_mode=str(
                getattr(cfg, "latent_event_preference_key_mode", "event_flag") or "event_flag"
            ),
            reward_dense_weight=max(0.0, float(getattr(env_cfg, "dense_weight", 1.0) or 0.0)),
            reward_scale=max(1e-6, float(getattr(env_cfg, "reward_scale", 1.0) or 1.0)),
            reward_clip=max(1e-6, float(getattr(env_cfg, "reward_clip", 1.0) or 1.0)),
            reward_stalemate_penalty=float(getattr(env_cfg, "stalemate_penalty", 0.0) or 0.0),
            reward_shaping_coef_start=float(getattr(cfg, "reward_shaping_coef_start", 1.0) or 1.0),
            reward_shaping_coef_end=float(
                getattr(
                    cfg,
                    "reward_shaping_coef_end",
                    float(getattr(cfg, "reward_shaping_coef_start", 1.0) or 1.0),
                )
            ),
            reward_shaping_decay_steps=max(
                0, int(getattr(cfg, "reward_shaping_decay_steps", 0) or 0)
            ),
            periodic_checkpoint_steps=max(
                0, int(getattr(cfg, "periodic_checkpoint_steps", 0) or 0)
            ),
            run_id=str(getattr(cfg, "run_id", "") or ""),
            run_pid=int(getattr(cfg, "run_pid", os.getpid()) or os.getpid()),
            metrics_csv_path=str(getattr(cfg, "metrics_csv_path", "") or ""),
            episode_csv_path=str(getattr(cfg, "episode_csv_path", "") or ""),
            strategy_experience_csv_path=str(
                getattr(cfg, "strategy_experience_csv_path", "") or ""
            ),
            opponent_randomize_training=opp_randomize,
            opponent_pool_tags=opp_tags,
            opponent_pool_weights=opp_weights,
        )

def resolve_q_phi_input_dim_from_cfg(cfg: Any) -> int:
    """Resolved q_phi / strategy_encoder input width before the model is built.

    Mirrors ``SharedActorCentralizedCritic.__init__`` + ``build_model_kwargs``:
    - router_context_mode="current" (V6I7): GLOBAL_STATE_V6I7_DIM(35) + GRU hidden
    - router_context_mode="current_plus_delta": router_context_dimension
    - V6I1 staged curriculum with EMA stack: CONTEXT_STATE_DIM(170) + GRU hidden
    - All other latent runs: CONTEXT_STATE_DIM(170)
    """
    if not bool(getattr(cfg, "use_latent_strategy", False)):
        return 0
    if router_current_plus_delta_enabled(cfg):
        dim = router_current_plus_delta_dim(cfg)
        if dim <= 0:
            raise ValueError("router_context_mode=current_plus_delta requires router_context_dimension > 0")
        return dim
    if str(getattr(cfg, "router_context_mode", "") or "") == "current":
        from rl.global_state import GLOBAL_STATE_V6I7_DIM
        recurrent_hidden = int(
            getattr(cfg, "recurrent_selector_hidden_dim", 0)
            or getattr(cfg, "v6i1_recurrent_selector_hidden", 0)
            or 0
        )
        return GLOBAL_STATE_V6I7_DIM + recurrent_hidden
    dim = int(CONTEXT_STATE_DIM)
    if is_staged_v6i1_curriculum(cfg):
        recurrent_hidden = int(
            getattr(cfg, "recurrent_selector_hidden_dim", 0)
            or getattr(cfg, "v6i1_recurrent_selector_hidden", 32)
            or 32
        )
        dim += recurrent_hidden
    return dim


def build_model_kwargs(cfg: Any, hparams: TrainerHyperparams) -> dict[str, Any]:
    """Compose ``SharedActorCentralizedCritic`` kwargs from cfg + hparams.

    Model shape (CNN feature dim, latent network sizes, embed dim) lives
    on ``PPOConfig`` and is read here once. Latent-feature kwargs are
    only included when ``hparams.use_latent_strategy`` is true; this
    matches the historical trainer behavior so checkpoints stay
    compatible.
    """
    model_kwargs: dict[str, Any] = {
        "actor_cnn_feature_dim": int(getattr(cfg, "actor_cnn_feature_dim", 128)),
        "actor_hidden_dim": int(getattr(cfg, "actor_hidden_dim", 256)),
    }
    if hparams.use_latent_strategy:
        v6i1_staged = is_staged_v6i1_curriculum(cfg)
        router_context_enabled = router_current_plus_delta_enabled(cfg)
        model_kwargs.update(
            {
                "experiment_id": str(getattr(cfg, "experiment_id", "") or ""),
                "router_context_mode": str(getattr(cfg, "router_context_mode", "") or ""),
                "router_context_dimension": router_current_plus_delta_dim(cfg),
                "latent_k": hparams.latent_k,
                "strategy_encoder_enabled": bool(
                    getattr(cfg, "latent_strategy_encoder_enabled", True)
                ),
                "z_embed_dim": int(getattr(cfg, "latent_z_embed_dim", 16)),
                "strategy_hidden_dim": int(getattr(cfg, "latent_strategy_hidden", 128)),
                "critic_hidden_dim": int(getattr(cfg, "latent_vf_hidden", 128)),
                # Canonical attribute access. Legacy CLI / cfg-dict keys
                # (``latent_strategy_q_head``) are folded into the canonical
                # name at the load boundary (CLI parsing for PPOConfig,
                # ``inference.canonicalize_latent_strategy_cfg`` for
                # checkpoint dicts), so the trainer reads one name.
                "use_strategy_aux_return_head": bool(
                    getattr(cfg, "latent_strategy_aux_return_head", False)
                ),
                # Model-architecture flag — keep ungated (mirrors historical
                # behavior). The runtime gating (``use_latent and not
                # fixed_latent``) lives on ``hparams.latent_episode_strategy_ppo``.
                "use_episode_strategy_value_head": bool(
                    getattr(cfg, "latent_episode_strategy_ppo", False)
                    or v6i1_staged
                    or (
                        getattr(cfg, "latent_arc_credit_enabled", False)
                        and str(
                            getattr(cfg, "latent_arc_credit_baseline", "context_value")
                            or "context_value"
                        ).lower() == "context_value"
                    )
                ),
                "use_recurrent_selector": bool(
                    int(getattr(cfg, "recurrent_selector_hidden_dim", 0) or 0) > 0
                    and (
                        (v6i1_staged and not router_context_enabled)
                        or str(getattr(cfg, "router_context_mode", "") or "") == "current"
                    )
                ),
                "recurrent_selector_hidden_dim": int(
                    getattr(cfg, "recurrent_selector_hidden_dim", 0) or 0
                ),
                "strategy_tau": max(
                    1e-3, float(getattr(cfg, "latent_strategy_tau", 1.0) or 1.0)
                ),
                "latent_actor_z_onehot_enabled": bool(
                    getattr(cfg, "latent_actor_z_onehot_enabled", False)
                ),
                "latent_actor_z_onehot_scale": max(
                    0.0, float(getattr(cfg, "latent_actor_z_onehot_scale", 1.0) or 0.0)
                ),
                "latent_actor_z_embed_scale": max(
                    0.0, float(getattr(cfg, "latent_actor_z_embed_scale", 1.0) or 0.0)
                ),
                "latent_actor_z_adapter_enabled": bool(
                    getattr(cfg, "latent_actor_z_adapter_enabled", False)
                ),
                "latent_actor_z_adapter_scale": max(
                    0.0, float(getattr(cfg, "latent_actor_z_adapter_scale", 0.0) or 0.0)
                ),
                "latent_actor_z_adapter_init_std": max(
                    0.0,
                    float(getattr(cfg, "latent_actor_z_adapter_init_std", 0.02) or 0.0),
                ),
                "latent_actor_z_film_layers": max(
                    1,
                    min(
                        2,
                        int(getattr(cfg, "latent_actor_z_film_layers", 1) or 1),
                    ),
                ),
                "enable_actor_z_film": bool(
                    getattr(cfg, "enable_actor_z_film", False)
                ),
                "actor_z_film_init_scale": max(
                    0.0,
                    float(getattr(cfg, "actor_z_film_init_scale", 0.0) or 0.0),
                ),
                "actor_z_film_layer": max(
                    1,
                    min(2, int(getattr(cfg, "actor_z_film_layer", 2) or 2)),
                ),
                "latent_actor_conditioning": getattr(cfg, "latent_actor_conditioning", "concat"),
                "enable_latent_z_residual": bool(
                    getattr(cfg, "enable_latent_z_residual", False)
                ),
                "latent_z_gate_init": max(
                    0.0, float(getattr(cfg, "latent_z_gate_init", 0.01) or 0.01)
                ),
                "latent_z_residual_alpha": max(
                    0.0, float(getattr(cfg, "latent_z_residual_alpha", 0.0) or 0.0)
                ),
                "latent_population_birth_active_z_only": bool(
                    getattr(cfg, "latent_population_birth_active_z_only", False)
                ),
                "latent_population_birth_per_z_action_heads": bool(
                    getattr(cfg, "latent_population_birth_per_z_action_heads", False)
                ),
                "exp2c_mode_specific_action_heads": bool(
                    getattr(cfg, "exp2c_mode_specific_action_heads", False)
                ),
                "rasr_private_critic_heads": bool(
                    getattr(cfg, "rasr_private_critic_heads", False)
                ),
                "latent_lro_deep_branches": bool(
                    getattr(cfg, "latent_lro_deep_branches", False)
                ),
            }
        )
    model_kwargs.update(
        {
            "communication_enabled": bool(getattr(cfg, "communication_enabled", False)),
            "comm_num_symbols": int(getattr(cfg, "comm_num_symbols", 4) or 4),
        }
    )
    return model_kwargs


__all__ = ["TrainerHyperparams", "build_model_kwargs", "resolve_q_phi_input_dim_from_cfg"]
