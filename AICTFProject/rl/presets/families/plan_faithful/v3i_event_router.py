"""v3i event router presets — event refresh through v3i6 stronger actor contrast."""
from __future__ import annotations

from rl.config.ppo_config import PPOConfig, TrainMode

from .v3_router import apply_plan_faithful_latent_v3h2_balanced_preference


def apply_plan_faithful_latent_v3i_event_refresh(cfg: PPOConfig) -> PPOConfig:
    """v3i: sparse q_phi refresh on meaningful game events."""
    cfg = apply_plan_faithful_latent_v3h2_balanced_preference(cfg)
    cfg.latent_event_refresh_enabled = True
    cfg.latent_event_refresh_min_gap_steps = 20
    cfg.latent_event_refresh_max_per_episode = 3
    cfg.latent_event_refresh_use_q_phi = True
    cfg.latent_event_refresh_force_roles = False
    cfg.run_tag = "latent_v3i_event_refresh_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3i2_router_signal(cfg: PPOConfig) -> PPOConfig:
    """v3i2: event refresh + stronger 4v4 router signal."""
    cfg = apply_plan_faithful_latent_v3i_event_refresh(cfg)
    cfg.latent_q_phi_train_after_steps = 50_000
    cfg.latent_preference_min_bucket_count = 4
    cfg.latent_preference_min_distinct_z = 2
    cfg.latent_preference_commit_coef = 0.005
    cfg.run_tag = "latent_v3i2_router_signal_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3i3_event_conditioned_preference(cfg: PPOConfig) -> PPOConfig:
    """v3i3: event-conditioned preference distillation on top of v3i2.

    The v3i2 telemetry analysis showed the audible system is mechanically
    working but the audible choice is essentially random:
        changed_z_rate ~= 74.6% (uniform expectation = 75%)
        transition_entropy = 2.748 (max ln(16) = 2.7726)
    The commander hears the horn, turns around, and then rolls a four-sided
    die. The v3i3 fix teaches WHICH horn calls are worth making by
    distilling a per-event teacher: each event refresh becomes a
    per-refresh training datapoint with bucket key
        (opponent_id, event_type, flag_state_bucket)
    and credit signal ``return_from_now_to_end_of_episode``. The teacher
    loss is a KL between ``q_phi(z | state_at_refresh)`` and softmax over
    average future-return per z within the matching bucket. When the
    finest-grained bucket is undersampled the lookup falls through:
        (opp, event, flag) -> (opp, event) -> (opp)

    Plan-faithful (the SUMMER protected rule):
        event_type = context, never command.
    The teacher only re-shapes q_phi's preferences using *observed*
    post-refresh returns; q_phi still learns ``pi(z | state)`` with no
    scripted z roles, no labels, and no opponent-ID gating in the actor.

    On top of v3i2:
      * Mandatory per-refresh proof-layer log
        (env_id, episode_id, decision_step, reason, prev_z, next_z,
         opponent_id, flag_state_bucket, return_from_now_to_end)
      * Event-conditioned preference loss with hierarchical fallback
      * Tighter bucket gates (min_bucket_count=4, min_distinct_z=2) since
        the 4v4 refresh count is plentiful but bucket cardinality is high

    Success gate (per the v3i3 design memo):
      WR >= 67%
      OP5 >= 50%, OP6 > 50%
      transition_entropy drops below random (currently 2.748 nats)
      changed_z_rate no longer matches uniform 75% by accident
      MI_z_phase, MI_z_flag rise above v3i2's noise floor
      event-specific post-refresh return improves (read off the refresh log)
    """
    cfg = apply_plan_faithful_latent_v3i2_router_signal(cfg)
    cfg.latent_v3i3_event_preference_enabled = True
    cfg.latent_v3i3_event_preference_coef = 0.03
    cfg.latent_v3i3_event_preference_temperature = 0.75
    cfg.latent_v3i3_event_preference_min_bucket_count = 4
    cfg.latent_v3i3_event_preference_min_distinct_z = 2
    cfg.latent_v3i3_event_preference_buffer_size = 50_000
    cfg.latent_v3i3_event_preference_warmup_steps = 50_000
    cfg.latent_v3i3_refresh_log_enabled = True
    cfg.run_tag = "latent_v3i3_event_conditioned_preference_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3i4_event_progress_preference(cfg: PPOConfig) -> PPOConfig:
    """v3i4: event + carrier progress preference distillation.

    Inherits from v3i3 and sets key_mode to event_flag_progress.
    """
    cfg = apply_plan_faithful_latent_v3i3_event_conditioned_preference(cfg)
    cfg.latent_event_preference_key_mode = "event_flag_progress"
    cfg.latent_v3i3_event_preference_normalize = True
    cfg.latent_v3i3_event_preference_warmup_steps = 50_000
    cfg.run_tag = "latent_v3i4_event_progress_preference_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3i5_crisp_router(cfg: PPOConfig) -> PPOConfig:
    """v3i5: crisp router (unshackle router).

    Inherits from v3i4, disables entropy pressure on the router,
    reduces usage balance coefficient, and sharpens event preference temperature.
    """
    cfg = apply_plan_faithful_latent_v3i4_event_progress_preference(cfg)
    cfg.latent_entropy_objective = "none"
    cfg.latent_usage_balance_coef = 0.05
    cfg.latent_v3i3_event_preference_temperature = 0.35
    cfg.latent_v3i3_event_preference_coef = 0.05
    cfg.run_tag = "latent_v3i5_crisp_router_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3i6_stronger_actor_contrast(cfg: PPOConfig) -> PPOConfig:
    """v3i6: stronger actor behavior contrast.

    Inherits from v3i4, sets behavior contrast coef to 0.10 and margin to 0.35.
    """
    cfg = apply_plan_faithful_latent_v3i4_event_progress_preference(cfg)
    cfg.latent_behavior_contrast_coef = 0.10
    cfg.latent_behavior_contrast_margin = 0.35
    cfg.run_tag = "latent_v3i6_stronger_actor_contrast_1m_4v4"
    return cfg
