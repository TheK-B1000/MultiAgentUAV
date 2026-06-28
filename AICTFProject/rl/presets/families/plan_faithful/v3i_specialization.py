"""v3i specialization presets — v3i7 through v3i15 specialist/separation variants."""
from __future__ import annotations

from rl.config.ppo_config import PPOConfig, TrainMode

from .v3i_event_router import (
    apply_plan_faithful_latent_v3i4_event_progress_preference,
    apply_plan_faithful_latent_v3i6_stronger_actor_contrast,
)


def apply_plan_faithful_latent_v3i7_advantage_weighted_router_distill(cfg: PPOConfig) -> PPOConfig:
    """v3i7: advantage-weighted router distillation.

    Inherits v3i6's stronger latent separation and adds a label-free bridge
    that pulls q_phi toward discovered winning z choices only when forced-z
    evidence shows a clear per-z win-rate advantage margin.
    """
    cfg = apply_plan_faithful_latent_v3i6_stronger_actor_contrast(cfg)
    cfg.latent_awrd_enabled = True
    cfg.latent_awrd_coef = 0.04
    cfg.latent_awrd_temperature = 0.35
    cfg.latent_awrd_min_bucket_count = 8
    cfg.latent_awrd_min_distinct_z = 2
    cfg.latent_awrd_margin_threshold = 0.15
    cfg.latent_awrd_margin_scale = 2.0
    cfg.run_tag = "latent_v3i7_adv_weighted_router_distill_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3i8_commander_lockin(cfg: PPOConfig) -> PPOConfig:
    """v3i8: commander lock-in preset.

    Combines v3i4 policy strength with v3i7 AWRD routing intelligence.
    """
    cfg = apply_plan_faithful_latent_v3i4_event_progress_preference(cfg)

    cfg.latent_awrd_enabled = True
    cfg.latent_awrd_coef = 0.05
    cfg.latent_awrd_temperature = 0.35
    cfg.latent_awrd_min_margin = 0.08
    cfg.latent_awrd_margin_scale = 3.0
    cfg.latent_awrd_soft_margin_gating = True
    cfg.latent_awrd_boost_after_fraction = 0.7
    cfg.latent_awrd_boost_multiplier = 1.5

    cfg.latent_behavior_contrast_coef = 0.05
    cfg.latent_behavior_contrast_margin = 0.25

    cfg.run_tag = "latent_v3i8_commander_lockin_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3i9_specialist_router(cfg: PPOConfig) -> PPOConfig:
    """v3i9: balanced specialist router.

    Inherits v3i8's event-progress preference and AWRD bridge, then changes
    the router pressure from raw entropy everywhere to balanced
    specialization:

      - high marginal entropy across the full batch, so all K latents stay alive
      - low conditional entropy inside each situation, so q_phi gets decisive
      - high context MI across opponent/context buckets, so different buckets
        can prefer different z choices

    Plan-faithful: context buckets are loss/telemetry groupings only. The code
    never assigns tactical names or role labels to z.
    """
    cfg = apply_plan_faithful_latent_v3i8_commander_lockin(cfg)

    cfg.latent_lam_h = 0.003
    cfg.latent_lam_h_start = 0.003
    cfg.latent_lam_h_end = 0.0003
    cfg.latent_entropy_anneal_start = 100_000
    cfg.latent_entropy_anneal_end = 500_000
    cfg.latent_entropy_objective = "maximize"

    cfg.latent_usage_balance_coef = 0.0
    cfg.latent_specialist_router_enabled = True
    cfg.latent_marginal_balance_coef = 0.02
    cfg.latent_conditional_entropy_min_coef = 0.015
    cfg.latent_context_mi_coef = 0.04
    cfg.latent_specialist_warmup_steps = 100_000
    cfg.latent_specialist_ramp_steps = 400_000
    cfg.latent_specialist_min_bucket_count = 4

    cfg.latent_awrd_coef = 0.06
    cfg.run_tag = "latent_v3i9_specialist_router_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3i10_role_phase_specialist(cfg: PPOConfig) -> PPOConfig:
    """v3i10: fixed-opponent role/phase specialist.

    Gives the team-level latent a clearer job without opponent pooling:
    specialize q_phi by phase/flag/carrier-progress context, keep z more
    persistent after event refreshes, and strengthen behavior separation so the
    actor cannot treat z as decorative.
    """
    cfg = apply_plan_faithful_latent_v3i9_specialist_router(cfg)

    cfg.latent_specialist_context_key_mode = "role_phase_progress_opponent"
    cfg.latent_marginal_balance_coef = 0.015
    cfg.latent_conditional_entropy_min_coef = 0.035
    cfg.latent_context_mi_coef = 0.08
    cfg.latent_specialist_min_bucket_count = 3

    cfg.latent_event_refresh_min_gap_steps = 80
    cfg.latent_event_refresh_max_per_episode = 1
    cfg.latent_v3i3_event_preference_coef = 0.04
    cfg.latent_v3i3_event_preference_temperature = 0.55

    cfg.latent_forced_z_episode_frac = 0.40
    cfg.latent_behavior_contrast_coef = 0.12
    cfg.latent_behavior_contrast_margin = 0.35
    cfg.latent_behavior_contrast_anneal_after_steps = 900_000
    cfg.latent_behavior_contrast_anneal_to = 0.02

    cfg.run_tag = "latent_v3i10_role_phase_specialist_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3i11_z_reactive_actor_adapters(cfg: PPOConfig) -> PPOConfig:
    """v3i11: make z harder for the actor/router to ignore.

    Inherits v3i10's role/phase/opponent specialist router, then adds a small
    per-z actor adapter and delayed stronger AWRD pressure. The adapter is
    still a shared actor with a lightweight z-conditioned residual, not K
    separate policies.
    """
    cfg = apply_plan_faithful_latent_v3i10_role_phase_specialist(cfg)

    cfg.mode = TrainMode.OPPONENT_POOL.value
    cfg.opponent_randomize = True
    cfg.opponent_pool = ("OP3", "OP5", "OP6")

    cfg.latent_actor_z_adapter_enabled = True
    cfg.latent_actor_z_adapter_scale = 0.35
    cfg.latent_actor_z_adapter_init_std = 0.03

    cfg.latent_lam_h = 0.003
    cfg.latent_lam_h_start = 0.003
    cfg.latent_lam_h_end = 0.0001
    cfg.latent_entropy_anneal_start = 100_000
    cfg.latent_entropy_anneal_end = 450_000

    cfg.latent_usage_balance_coef = 0.015
    cfg.latent_marginal_balance_coef = 0.02
    cfg.latent_conditional_entropy_min_coef = 0.05
    cfg.latent_context_mi_coef = 0.12
    cfg.latent_specialist_warmup_steps = 100_000
    cfg.latent_specialist_ramp_steps = 300_000
    cfg.latent_specialist_min_bucket_count = 3
    cfg.latent_awrd_coef = 0.10
    cfg.latent_awrd_temperature = 0.30
    cfg.latent_awrd_min_bucket_count = 6
    cfg.latent_awrd_margin_threshold = 0.08
    cfg.latent_awrd_margin_scale = 4.0
    cfg.latent_awrd_warmup_steps = 100_000
    cfg.latent_awrd_ramp_steps = 250_000

    cfg.latent_forced_z_episode_frac = 0.45
    cfg.latent_behavior_contrast_coef = 0.14
    cfg.latent_behavior_contrast_margin = 0.35

    cfg.run_tag = "latent_v3i11_z_reactive_actor_adapters_pool_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3i12_faithful_z_pressure(cfg: PPOConfig) -> PPOConfig:
    """v3i12: make the existing shared actor listen to z.

    Inherits v3i10's role/phase/opponent specialist router, then adds only
    faithful z pressure: z one-hot columns in the shared actor input, a small
    forced-z logit separation loss, and delayed AWRD router distillation.
    No per-latent adapters, no per-latent actor heads, and no scripted z roles.
    """
    cfg = apply_plan_faithful_latent_v3i10_role_phase_specialist(cfg)

    cfg.mode = TrainMode.OPPONENT_POOL.value
    cfg.opponent_randomize = True
    cfg.opponent_pool = ("OP3", "OP5", "OP6")

    cfg.latent_actor_z_onehot_enabled = True
    cfg.latent_actor_z_onehot_scale = 1.0
    cfg.latent_actor_z_embed_scale = 1.25
    cfg.latent_actor_z_adapter_enabled = False
    cfg.latent_actor_z_adapter_scale = 0.0
    cfg.latent_actor_z_adapter_init_std = 0.02
    cfg.latent_actor_z_separation_coef = 0.015
    cfg.latent_actor_z_separation_margin = 0.02

    cfg.latent_lam_h = 0.003
    cfg.latent_lam_h_start = 0.003
    cfg.latent_lam_h_end = 0.0003
    cfg.latent_entropy_anneal_start = 100_000
    cfg.latent_entropy_anneal_end = 450_000
    cfg.latent_entropy_objective = "maximize"

    cfg.latent_usage_balance_coef = 0.015
    cfg.latent_marginal_balance_coef = 0.02
    cfg.latent_conditional_entropy_min_coef = 0.05
    cfg.latent_context_mi_coef = 0.12
    cfg.latent_specialist_warmup_steps = 100_000
    cfg.latent_specialist_ramp_steps = 300_000
    cfg.latent_specialist_min_bucket_count = 3

    cfg.latent_awrd_enabled = True
    cfg.latent_awrd_coef = 0.10
    cfg.latent_awrd_temperature = 0.30
    cfg.latent_awrd_min_bucket_count = 6
    cfg.latent_awrd_margin_threshold = 0.08
    cfg.latent_awrd_margin_scale = 4.0
    cfg.latent_awrd_warmup_steps = 100_000
    cfg.latent_awrd_ramp_steps = 300_000

    cfg.latent_forced_z_episode_frac = 0.45
    cfg.latent_behavior_contrast_coef = 0.14
    cfg.latent_behavior_contrast_margin = 0.35

    cfg.run_tag = "latent_v3i12_faithful_z_pressure_pool_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3i13_strict_faithful_z(cfg: PPOConfig) -> PPOConfig:
    """v3i13: strict-faithful policy space separation with z-FiLM only.

    Focuses on actor-side z dependence and policy-space separation:
      - z-only FiLM conditioning (no concat: z_onehot_enabled=False, z_embed_dim=0, z_adapter_enabled=True, scale=0.15)
      - Ramped z-adapter scale (warmup 100k, ramp 300k, target 0.15).
      - Strengthened forced-z policy separation using average pairwise JS divergence loss over all K latent options.
      - Conservative JSD separation: coef = 0.02, margin = 0.08, warmup 100k, ramp 300k.
      - Lower entropy pressure after warmup (lam_h starting at 0.003, annealing to 0.0001 by 400k steps).
      - Marginal anti-collapse balance pressure (latent_usage_balance_coef=0.015 and latent_marginal_balance_coef=0.02) to keep all z alive.
      - Behavior telemetry for evaluation only.
    """
    cfg = apply_plan_faithful_latent_v3i12_faithful_z_pressure(cfg)

    # 1. z-only FiLM conditioning (no concat)
    cfg.latent_actor_z_onehot_enabled = False
    cfg.latent_z_embed_dim = 0
    cfg.latent_actor_z_adapter_enabled = True
    cfg.latent_actor_z_adapter_scale = 0.15
    cfg.latent_actor_z_adapter_warmup_steps = 100_000
    cfg.latent_actor_z_adapter_ramp_steps = 300_000
    cfg.latent_actor_z_adapter_init_std = 0.02

    # 2. Strengthen forced-z policy separation using average pairwise JSD
    cfg.latent_actor_z_separation_coef = 0.02
    cfg.latent_actor_z_separation_margin = 0.08
    cfg.latent_actor_z_separation_warmup_steps = 100_000
    cfg.latent_actor_z_separation_ramp_steps = 300_000

    # 3. Lower entropy pressure after warmup
    cfg.latent_lam_h = 0.003
    cfg.latent_lam_h_start = 0.003
    cfg.latent_lam_h_end = 0.0001
    cfg.latent_entropy_anneal_start = 100_000
    cfg.latent_entropy_anneal_end = 400_000
    cfg.latent_entropy_objective = "maximize"

    # 4. Marginal anti-collapse pressure
    cfg.latent_usage_balance_coef = 0.015
    cfg.latent_marginal_balance_coef = 0.02

    cfg.run_tag = "latent_v3i13_strict_faithful_z_pool_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3i14_specialized_faithful_z(
    cfg: PPOConfig,
) -> PPOConfig:
    """v3i14: turn balanced live latents into tactical specialists.

    Keeps the strict-faithful v3i13 architecture: one shared actor, no
    opponent-id actor input, no supervised strategy labels, and no per-z
    actor heads. Specialization comes from tactical router buckets, gated
    forced-z policy separation, stronger shared FiLM, and harder opponents.
    """
    cfg = apply_plan_faithful_latent_v3i13_strict_faithful_z(cfg)

    cfg.mode = TrainMode.OPPONENT_POOL.value
    cfg.opponent_randomize = True
    cfg.opponent_pool = ("OP3", "OP5", "OP6")
    cfg.opponent_pool_weights = (0.15, 0.40, 0.45)

    cfg.latent_q_phi_bucket_baseline = "tactical_context_opponent"
    cfg.latent_specialist_router_enabled = True
    cfg.latent_specialist_context_key_mode = (
        "tactical_phase_flags_score_opponent"
    )
    cfg.latent_specialist_conditional_entropy_scope = "context_bucket"
    cfg.latent_usage_balance_coef = 0.015
    cfg.latent_marginal_balance_coef = 0.02
    cfg.latent_conditional_entropy_min_coef_start = 0.01
    cfg.latent_conditional_entropy_min_coef = 0.05
    cfg.latent_context_mi_coef = 0.05
    cfg.latent_specialist_warmup_steps = 100_000
    cfg.latent_specialist_ramp_steps = 300_000
    cfg.latent_specialist_min_bucket_count = 3
    cfg.latent_specialist_use_rollout_states = True
    cfg.latent_specialist_rollout_max_samples = 8192

    cfg.latent_lam_h = 0.0001
    cfg.latent_lam_h_start = 0.0001
    cfg.latent_lam_h_end = 0.0001
    cfg.latent_entropy_anneal_start = 0
    cfg.latent_entropy_anneal_end = 0
    cfg.latent_entropy_objective = "maximize"
    cfg.latent_episode_strategy_coef = 0.30

    cfg.latent_actor_z_onehot_enabled = False
    cfg.latent_z_embed_dim = 0
    cfg.latent_actor_z_adapter_enabled = True
    cfg.latent_actor_z_adapter_scale = 0.5
    cfg.latent_actor_z_film_layers = 2
    cfg.latent_actor_z_adapter_warmup_steps = 100_000
    cfg.latent_actor_z_adapter_ramp_steps = 300_000

    cfg.latent_actor_z_separation_start_coef = 0.005
    cfg.latent_actor_z_separation_coef = 0.02
    cfg.latent_actor_z_separation_margin = 0.08
    cfg.latent_actor_z_separation_warmup_steps = 100_000
    cfg.latent_actor_z_separation_ramp_steps = 300_000
    cfg.latent_actor_z_separation_min_abs_advantage = 0.5
    cfg.latent_actor_z_separation_min_decision_frac = 0.05
    cfg.latent_actor_z_separation_max_entropy_frac = 0.90

    cfg.run_tag = "latent_v3i14_specialized_faithful_z_pool_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3i14_tuned(cfg: PPOConfig) -> PPOConfig:
    """v3i14 tuned: stronger tactical niches without changing architecture."""
    cfg = apply_plan_faithful_latent_v3i14_specialized_faithful_z(cfg)

    # Preserve the gradual v3i14 schedule, but strengthen the final tactical
    # target once warmup and ramp have completed.
    cfg.latent_conditional_entropy_min_coef = 0.09
    cfg.latent_marginal_balance_coef = 0.015

    # Keep the same gated all-pairs JSD path with a moderately stronger final
    # separation coefficient.
    cfg.latent_actor_z_separation_coef = 0.028

    # Reduce global entropy maximization while usage balance remains active.
    cfg.latent_lam_h = 0.00005
    cfg.latent_lam_h_start = 0.00005
    cfg.latent_lam_h_end = 0.00005

    cfg.run_tag = "latent_v3i14_tuned_tactical_specialist_pool_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3i15_strong_separation(cfg: PPOConfig) -> PPOConfig:
    """v3i15 strong separation: eliminate warmups, strengthen separation JSD and initialization std."""
    cfg = apply_plan_faithful_latent_v3i14_tuned(cfg)

    # Eliminate warmups and use a rapid 20k steps ramp to prevent representation lock-in
    cfg.latent_actor_z_adapter_warmup_steps = 0
    cfg.latent_actor_z_adapter_ramp_steps = 20_000
    cfg.latent_actor_z_adapter_init_std = 0.10

    cfg.latent_actor_z_separation_start_coef = 0.01
    cfg.latent_actor_z_separation_coef = 0.20
    cfg.latent_actor_z_separation_margin = 0.35
    cfg.latent_actor_z_separation_warmup_steps = 0
    cfg.latent_actor_z_separation_ramp_steps = 20_000

    cfg.run_tag = "latent_v3i15_strong_separation_pool_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3i15_sparse_tactical_refresh(
    cfg: PPOConfig,
) -> PPOConfig:
    """v3i15: test sparse tactical routing timescale without actor changes."""
    cfg = apply_plan_faithful_latent_v3i14_tuned(cfg)

    # Isolate the timescale experiment from the older event-refresh path.
    cfg.latent_event_refresh_enabled = False
    cfg.latent_sparse_tactical_refresh_enabled = True
    cfg.latent_sparse_tactical_refresh_interval_steps = 32
    cfg.latent_sparse_tactical_refresh_min_dwell_steps = 16

    cfg.latent_lam_p = 0.02
    cfg.latent_lam_h = 0.000025
    cfg.latent_lam_h_start = 0.000025
    cfg.latent_lam_h_end = 0.000025
    cfg.latent_conditional_entropy_min_coef = 0.09
    cfg.latent_actor_z_separation_coef = 0.028
    cfg.latent_marginal_balance_coef = 0.015
    cfg.latent_gae_reset_on_z_change = True

    cfg.run_tag = "latent_v3i15_sparse_tactical_refresh_pool_1m_4v4"
    return cfg
