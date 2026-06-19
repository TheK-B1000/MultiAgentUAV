"""Plan-faithful training presets aligned with the Summer Implementation Plan and paper tables."""

from __future__ import annotations

from rl.train_ppo import PPOConfig, TrainMode


def apply_plan_faithful_base(cfg: PPOConfig) -> PPOConfig:
    """Shared Summer-plan setup for the clean latent/no-latent ablation family."""
    cfg.total_timesteps = 1_000_000
    cfg.mode = TrainMode.FIXED_OPPONENT.value
    cfg.fixed_opponent_tag = "OP3"
    cfg.normalize_returns = True
    cfg.clip_range = 0.18
    cfg.clip_range_vf = 0.2
    cfg.vf_coef = 1.1
    cfg.learning_rate = 1.8e-4
    cfg.lr_floor_frac = 0.05
    cfg.target_kl = 0.02
    cfg.n_steps = 2048
    cfg.batch_size = 512
    cfg.n_epochs = 8
    cfg.ent_coef = 0.0015
    cfg.latent_k = 4
    cfg.latent_entropy_objective = "maximize"
    cfg.latent_lam_h = 0.003
    cfg.latent_lam_p = 0.025
    cfg.latent_strategy_ppo_coef = 0.30
    cfg.latent_episode_strategy_ppo = False
    cfg.latent_episode_strategy_coef = 0.0
    cfg.latent_episode_strategy_clip_eps = 0.2
    cfg.latent_episode_strategy_value_coef = 0.5
    cfg.latent_episode_strategy_return_norm = True
    cfg.latent_strategy_aux_return_head = False
    cfg.latent_strategy_aux_return_coef = 0.0
    cfg.latent_strategy_tau = 1.0
    cfg.latent_resample_every_n = 20
    cfg.latent_resample_on_flag = False
    cfg.latent_kl_consecutive = 0.0
    cfg.latent_gae_reset_on_z_change = True
    cfg.latent_bootstrap_z_deterministic = True
    cfg.latent_vf_hidden = 128
    cfg.fixed_latent_strategy = False
    cfg.fixed_latent_strategy_id = 0
    cfg.env_win_team_reward = None
    cfg.env_draw_team_penalty = None
    cfg.env_lose_team_punish = None
    cfg.env_action_failed_punishment = None
    cfg.env_dense_weight = None
    cfg.env_sparse_weight = None
    cfg.env_reward_scale = None
    cfg.env_reward_clip = None
    cfg.env_stalemate_penalty = None
    cfg.env_stalemate_max_steps = None
    cfg.reward_shaping_coef_start = 1.0
    cfg.reward_shaping_coef_end = 1.0
    cfg.reward_shaping_decay_steps = 0
    cfg.periodic_checkpoint_steps = 50_000
    cfg.load_path = None
    return cfg


def apply_plan_faithful_latent(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_plan_faithful_base(cfg)
    cfg.use_latent_strategy = True
    cfg.run_tag = "latent_recommended_1m_2v2"
    return cfg


def apply_plan_faithful_latent_no_persistence(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_plan_faithful_latent(cfg)
    cfg.latent_lam_p = 0.0
    cfg.run_tag = "latent_recommended_no_persistence_1m_2v2"
    return cfg


def apply_plan_faithful_latent_episode_strategic(cfg: PPOConfig) -> PPOConfig:
    """Episode-credit answer to the decorative-z problem (variance-reduction fix).

    Switches q_phi's credit assignment from per-segment option-credit to
    per-episode episode-credit. The key insight: per-segment credit gives q_phi
    a gradient dominated by within-segment policy/opponent stochasticity rather
    than by z's marginal contribution. With K=4 and ~60 (state-class, z) cells,
    a 20-step-segment rollout produces only ~13 samples per cell — far too few
    to learn a sharp conditional q_phi(z|state) when each sample is high-variance.

    Per-episode credit collapses this to **one clean datapoint per episode per cell**:
    a 1M-step rollout sees ~10,000 episodes ⇒ ~800 samples per cell ⇒ ~60x
    variance reduction. With that signal-to-noise q_phi can actually sharpen,
    which makes MI(z; opponent/outcome/flag) measurably > 0 instead of stuck
    at the plug-in noise floor.

    Inherits the plan-faithful base (K=4, ctx170 from the upgraded env,
    rebalanced 4v4 opponents) and configures:

      - ``latent_resample_every_n = 0``  → z is sampled at episode start only
      - ``latent_episode_strategy_warmup_decision_steps = 5`` → defer the
        committed z snapshot for 5 decision steps so ctx170 EMAs have observed
        opponent dynamics (red_speed, formation, flag pressure) before q_phi
        chooses. Without this guard the snapshot context is structurally
        opponent-blind (raw initial geometry + zeroed EMAs) and MI(z; opponent)
        is upper-bounded near zero regardless of credit quality.
      - ``latent_lam_p = 0``              → no persistence loss (nothing to persist)
      - ``latent_lam_h = 0.003``          → start with entropy pressure so K=4
                                            stays alive
      - ``latent_lam_h_start/end``        → anneal 0.003 -> 0.0005 from
                                            200k to 700k so q_phi can sharpen
      - ``latent_strategy_ppo_coef = 0``  → no per-step strategy coupling
      - ``latent_episode_strategy_ppo = True``
      - ``latent_episode_strategy_coef = 0.30``  → per-episode credit weight
      - ``latent_q_phi_option_advantage = False`` → unused in episode mode

    Plan-faithful: no labels, no aux heads, no opponent ID. Only changes the
    *credit horizon* for q_phi from 20 steps to the full episode, and defers
    the q_phi decision moment until the context is opponent-informed.
    """
    cfg = apply_plan_faithful_base(cfg)
    cfg.use_latent_strategy = True
    cfg.latent_k = 4
    cfg.latent_resample_every_n = 0
    cfg.latent_resample_on_flag = False
    cfg.latent_lam_p = 0.0
    cfg.latent_lam_h = 0.003
    cfg.latent_lam_h_start = 0.003
    cfg.latent_lam_h_end = 0.0005
    cfg.latent_entropy_anneal_start = 200_000
    cfg.latent_entropy_anneal_end = 700_000
    cfg.latent_entropy_objective = "maximize"
    cfg.latent_strategy_ppo_coef = 0.0
    cfg.latent_episode_strategy_ppo = True
    cfg.latent_episode_strategy_coef = 0.30
    cfg.latent_episode_strategy_clip_eps = 0.2
    cfg.latent_episode_strategy_value_coef = 0.5
    cfg.latent_episode_strategy_return_norm = True
    cfg.latent_episode_strategy_warmup_decision_steps = 5
    cfg.latent_q_phi_option_advantage = False
    cfg.latent_strategy_aux_predict_phase_coef = 0.0
    cfg.latent_strategy_aux_return_head = False
    cfg.latent_strategy_aux_return_coef = 0.0
    cfg.latent_kl_consecutive = 0.0
    cfg.fixed_latent_strategy = False
    cfg.run_tag = "latent_episode_strategic_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3b_marginal(cfg: PPOConfig) -> PPOConfig:
    """Episode-credit strategy PPO with uniform marginal baseline.

    Inherits from ``apply_plan_faithful_latent_episode_strategic`` and configures:
      - ``latent_q_phi_marginal_baseline = True`` (advantage relative to average V over all strategies)
    """
    cfg = apply_plan_faithful_latent_episode_strategic(cfg)
    cfg.latent_q_phi_marginal_baseline = True
    cfg.run_tag = "latent_v3b_marginal_1m_4v4"
    return cfg


def apply_plan_faithful_latent_step6(cfg: PPOConfig) -> PPOConfig:
    """Step 6: option-style q_phi advantage.

    Inherits from apply_plan_faithful_latent_strategic (latent_q_phi_option_advantage = True,
    latent_strategy_ppo_coef = 0.40) and configures:
      - warmup5 commit behavior (warmup_decision_steps = 5)
      - no persistence loss (latent_lam_p = 0.0)
      - lamH entropy annealing (0.003 -> 0.0005 over steps 200k to 700k)
    """
    cfg = apply_plan_faithful_latent_strategic(cfg)

    cfg.latent_resample_every_n = 0
    cfg.latent_episode_strategy_warmup_decision_steps = 5
    cfg.latent_lam_p = 0.0

    cfg.latent_lam_h = 0.003
    cfg.latent_lam_h_start = 0.003
    cfg.latent_lam_h_end = 0.0005
    cfg.latent_entropy_anneal_start = 200_000
    cfg.latent_entropy_anneal_end = 700_000

    cfg.run_tag = "latent_step6_optionadv_warmup5_lamHanneal_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3b_marginal(cfg: PPOConfig) -> PPOConfig:
    """v3b: episode-credit with z-marginal baseline (fixes baseline-eating).

    The v3 episode-credit + warmup + entropy-anneal run held PPO stable
    (~65% WR, EV 0.61) but produced effectively zero MI(z; opponent) and
    kept ``zH`` pinned at maximum entropy even after ``lam_h`` annealed
    to the floor. Code audit identified the math root cause:

        adv_z = R - V(s, z_picked)

    The centralized critic ``V(s, z_picked)`` already absorbs ``E[R | s, z]``,
    so the advantage that arrives at q_phi is "this z vs its own expectation"
    -- i.e. mostly within-z policy noise. The cross-z signal q_phi needs to
    learn from is mathematically subtracted before the gradient is computed.

    v3b applies the variance-optimal AAC fix while keeping everything else
    identical to v3. ``adv`` now uses a z-marginal baseline:

        adv_z = R - E_{z' ~ q_phi(s)}[V(s, z')] = R - sum_k pi_phi(k|s) * V(s, k)

    so the advantage encodes "this z vs the average available z in this
    context" -- the exact signal contextual specialization needs.

    Inherits ``apply_plan_faithful_latent_episode_strategic`` (which already
    has warmup=5, lam_p=0, lam_h anneal 0.003 -> 0.0005 from 200k -> 700k,
    K=4, ctx170, episode-credit on, per-step coupling off). Flips one knob:

      - ``latent_q_phi_marginal_baseline = True``

    Plan-faithful: no labels, no aux heads, no opponent IDs. Only the
    baseline math changes.

    Expected first signs of working (per implementation review):
      ~100k: latent_episode_pg_loss meaningfully nonzero, adv_std > 0.5
      ~300k: latent_episode_approx_kl consistent in 0.001-0.01 range
      ~500k: zH_frac drops below 0.95 (q_phi moves off uniform)
      ~700k: MI(z; opponent) above 0.02 if the fix works
      ~1M:   WR matches the 65% baseline, MI > 0.02

    If MI is still pinned near 0 by 700k under v3b, the problem is deeper
    than the baseline -- next experiment becomes V calibration on off-policy
    z slots (train V(s, z) on all K z per state, not just the picked one)
    or expand the actor's z-conditioning capacity (z_emb 16 -> 32).
    """
    cfg = apply_plan_faithful_latent_episode_strategic(cfg)
    cfg.latent_q_phi_marginal_baseline = True
    cfg.run_tag = "latent_v3b_episodecredit_marginalbaseline_warmup5_lamHanneal_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3d_delayed_anneal(cfg: PPOConfig) -> PPOConfig:
    """v3d variant: same smart-router baseline, delayed entropy anneal.

    Motivating diagnosis from the live v3d run: at ~327k the bucket baseline +
    dedicated router LR are already pushing meaningful router updates
    (``kl=0.0150``, ``grad_norm=0.08``, ``z_occ`` non-uniform) WHILE the
    entropy schedule is still in its 200k-700k decay window (lamH already at
    ~0.0025 by 327k). That means two opposing forces are active at the same
    time:

      (1) The smart router pushing q_phi to commit to whichever z scored best
          in each bucket so far -- with strong updates that early movement
          gets locked in fast.

      (2) The entropy regularizer pulling q_phi back toward uniform -- but
          the leash is already loosening.

    Risk: q_phi latches onto z_k that won during the OPENING ROLLOUTS, where
    the actor hadn't yet had enough updates under all K strategies. Result is
    "louder, not wiser" -- a self-fulfilling z preference where unused z slots
    starve for training because the router stopped sampling them.

    This variant decouples the timing: same start/floor entropy (0.003 -> 0.001
    keeps the curiosity high early and the floor low enough for selection
    late), but the decay starts later and ends later, giving the actor an
    extra 100k steps of fully-uniform-sampled rollouts under all K strategies
    before the entropy leash starts to loosen.

    Plan-faithful: identical to v3d in every other respect. Only changes
    ``latent_entropy_anneal_start`` (200k -> 300k) and
    ``latent_entropy_anneal_end`` (700k -> 800k). Bucket baseline still
    "opponent", router LR still 5e-3, n_epochs still 6.

    When to launch:
      - v3d's MI(z; opponent) stalls below 0.02 by 500k.
      - OR v3d's z_occ shows premature one-z dominance (any slot > 0.5
        before 500k).
      - OR z_wr_spread starts declining (router locking in suboptimally).

    When NOT to launch:
      - v3d's MI is climbing healthily. Don't fix what isn't broken.
      - z_occ stays in the [.15, .40] band across all four z slots through
        the run. That's a router doing its job; no need to give it more
        exploration runway.
    """
    cfg = apply_plan_faithful_latent_v3d_smart_router(cfg)
    cfg.latent_entropy_anneal_start = 300_000
    cfg.latent_entropy_anneal_end = 800_000
    cfg.run_tag = "latent_v3d_delayedanneal_300k_800k_bucketopp_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3e_strong_z_actor(cfg: PPOConfig) -> PPOConfig:
    """v3e: strong z actor preset.

    Inherits from v3d_delayed_anneal and configures:
      - latent_z_embed_dim: 16 -> 32
      - actor_hidden_dim: 256 -> 384
      - run_tag: latent_v3e_strong_z_actor_1m_4v4
    """
    cfg = apply_plan_faithful_latent_v3d_delayed_anneal(cfg)
    cfg.latent_z_embed_dim = 32
    cfg.actor_hidden_dim = 384
    cfg.run_tag = "latent_v3e_strong_z_actor_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3f_behavior_contrast(cfg: PPOConfig) -> PPOConfig:
    """v3f: self-supervised latent behavior contrast.

    Inherits v3e's stronger z-conditioned actor and v3d's episode-credit
    router, then adds label-free option separation:

      - 30% forced-z episodes, uniformly sampled across K
      - completed-episode behavior contrast bonus on forced-z episodes only
      - q_phi episode-credit delayed until the actor has seen forced-z data
      - weak aggregate q_phi usage balance, applied only inside q_phi credit

    Plan-faithful: no role labels, no scripted z meanings, no opponent-ID
    heads, no supervised router targets. The behavior embedding is built from
    existing observable team telemetry and compared inside coarse game-state
    buckets so the pressure is "different modes under similar contexts."
    """
    cfg = apply_plan_faithful_latent_v3e_strong_z_actor(cfg)
    cfg.latent_forced_z_episode_frac = 0.30
    cfg.latent_behavior_contrast_coef = 0.05
    cfg.latent_behavior_contrast_margin = 0.25
    cfg.latent_behavior_contrast_ema = 0.90
    cfg.latent_behavior_contrast_anneal_after_steps = 800_000
    cfg.latent_behavior_contrast_anneal_to = 0.005
    cfg.latent_usage_balance_coef = 0.01
    cfg.latent_q_phi_train_after_steps = 100_000
    cfg.run_tag = "latent_v3f_behavior_contrast_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3g_preference(cfg: PPOConfig) -> PPOConfig:
    """v3g: self-supervised latent preference distillation from forced-z.

    Inherits v3f's contrastive separation (forced-z exploration + actor conditioning)
    and smart router, then distills a soft target probability distribution from
    the returns of forced-z episodes into q_phi.
    """
    cfg = apply_plan_faithful_latent_v3f_behavior_contrast(cfg)
    cfg.latent_preference_coef = 0.03
    cfg.latent_preference_temperature = 0.75
    cfg.latent_preference_min_bucket_count = 8
    cfg.latent_preference_min_distinct_z = 2
    cfg.run_tag = "latent_v3g_preference_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3h_balanced_preference(cfg: PPOConfig) -> PPOConfig:
    """v3h: self-supervised latent preference distillation with opponent-balanced KL loss and target telemetry."""
    cfg = apply_plan_faithful_latent_v3g_preference(cfg)
    cfg.latent_preference_coef = 0.03
    cfg.latent_preference_opponent_balanced = True
    cfg.latent_preference_log_opponent_targets = True
    cfg.run_tag = "latent_v3h_balanced_preference_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3h2_balanced_preference(cfg: PPOConfig) -> PPOConfig:
    """v3h2: self-supervised latent preference distillation with confidence-weighted KL + entropy commitment."""
    cfg = apply_plan_faithful_latent_v3h_balanced_preference(cfg)
    cfg.latent_preference_confidence_scale = 2.0
    cfg.latent_preference_commit_coef = 0.003
    cfg.late_entropy_floor = 0.0003
    cfg.commitment_type = "confidence_weighted_entropy"
    # Entropy schedule:
    # 0 - 300k steps: lam_h = 0.003
    # 300k - 600k steps: linear anneal from 0.003 to 0.001
    # 600k+ steps: linear anneal from 0.001 to late_floor (0.0003) at total_timesteps
    cfg.latent_lam_h = 0.003
    cfg.latent_lam_h_start = 0.003
    cfg.latent_lam_h_end = 0.0003
    cfg.latent_entropy_anneal_start = 300_000
    cfg.latent_entropy_anneal_end = 600_000
    cfg.run_tag = "latent_v3h2_balanced_preference_1m_4v4"
    return cfg


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


def apply_plan_faithful_latent_v3i16_policy_z_embedding(
    cfg: PPOConfig,
) -> PPOConfig:
    """v3i16: strict plan-faithful learned-z actor conditioning.

    The actor receives ``concat(CNN(grid), per_agent_vec, z_embedding)``.
    The critic keeps its existing z one-hot path. All later experimental
    routing teachers, behavior rewards, policy-separation losses, adapters,
    and event-driven refresh paths are disabled for this clean test.
    """
    cfg = apply_plan_faithful_latent(cfg)

    cfg.latent_actor_z_onehot_enabled = False
    cfg.latent_z_embed_dim = 16
    cfg.latent_actor_z_embed_scale = 1.0
    cfg.latent_actor_z_adapter_enabled = False
    cfg.latent_actor_z_adapter_scale = 0.0
    cfg.latent_actor_z_film_layers = 1
    cfg.latent_actor_z_adapter_warmup_steps = 0
    cfg.latent_actor_z_adapter_ramp_steps = 0

    cfg.latent_lam_h = 0.001
    cfg.latent_lam_h_start = 0.001
    cfg.latent_lam_h_end = 0.001
    cfg.latent_entropy_anneal_start = 0
    cfg.latent_entropy_anneal_end = 0
    cfg.latent_entropy_objective = "maximize"
    cfg.latent_lam_p = 0.02

    # Fixed-cadence persistence is the simple Option-B path. It keeps lambda_p
    # active without the event/tactical refresh machinery from v3i15.
    cfg.latent_resample_every_n = 64
    cfg.latent_resample_on_flag = False
    cfg.latent_event_refresh_enabled = False
    cfg.latent_sparse_tactical_refresh_enabled = False
    cfg.latent_kl_consecutive = 0.0
    cfg.latent_gae_reset_on_z_change = True

    cfg.latent_strategy_aux_predict_phase_coef = 0.0
    cfg.latent_strategy_aux_return_head = False
    cfg.latent_strategy_aux_return_coef = 0.0
    cfg.latent_forced_z_episode_frac = 0.0
    cfg.latent_behavior_contrast_coef = 0.0
    cfg.latent_actor_z_separation_start_coef = 0.0
    cfg.latent_actor_z_separation_coef = 0.0
    cfg.latent_usage_balance_coef = 0.0
    cfg.latent_preference_coef = 0.0
    cfg.latent_preference_commit_coef = 0.0
    cfg.latent_awrd_enabled = False
    cfg.latent_awrd_coef = 0.0
    cfg.latent_specialist_router_enabled = False
    cfg.latent_marginal_balance_coef = 0.0
    cfg.latent_conditional_entropy_min_coef_start = 0.0
    cfg.latent_conditional_entropy_min_coef = 0.0
    cfg.latent_context_mi_coef = 0.0
    cfg.latent_v3i3_event_preference_enabled = False
    cfg.latent_v3i3_event_preference_coef = 0.0
    cfg.latent_v3i3_refresh_log_enabled = False

    cfg.run_tag = "v3i16_plan_faithful_z_embed_1m_4v4"
    return cfg


def _apply_v3i17_consequence_only(cfg: PPOConfig) -> PPOConfig:
    """Shared v3i17 configuration: 'reward z consequences, not z existence'.

    Branched into two sibling presets, ``v3i17_episode_arc`` (episode-level z)
    and ``v3i17_long_arc`` (256-step persistence). Both keep v3i16's actor
    z-embedding architecture and disable every "existence" knob, leaving only
    the consequence channel (episode-credit PPO on q_phi) alive after the
    entropy anneal completes.

    Existence pressure removed / annealed away:

    * ``latent_lam_h`` anneals **0.003 -> 0.0** over steps **200k -> 700k**.
      Early phase keeps K=4 alive while q_phi explores; after 700k the
      coefficient is exactly zero so the marginal-entropy reward stops
      contributing gradient.
    * ``latent_actor_z_separation_*``, ``latent_usage_balance_*``,
      ``latent_marginal_balance_*``, ``latent_behavior_contrast_*``,
      ``latent_specialist_*``, ``latent_conditional_entropy_min_*``,
      ``latent_context_mi_*`` all forced to 0 / disabled (inherited from v3i16).

    Consequence channel kept on:

    * ``latent_episode_strategy_ppo = True`` with
      ``latent_episode_strategy_coef = 0.30``. q_phi's gradient is the
      per-episode return advantage. One clean datapoint per episode per
      (context, z) cell; the only signal that pushes q_phi to specialise.
    * ``latent_episode_strategy_warmup_decision_steps = 5`` so q_phi commits
      after the ctx170 EMAs have observed opponent dynamics.
    * ``latent_strategy_ppo_coef = 0.0`` -- no per-step strategy coupling.
    * v3i3 / preference / AWRD distillation channels stay OFF; we want a
      single, audit-able consequence source.
    """
    cfg = apply_plan_faithful_latent_v3i16_policy_z_embedding(cfg)

    cfg.latent_lam_h = 0.003
    cfg.latent_lam_h_start = 0.003
    cfg.latent_lam_h_end = 0.0
    cfg.latent_entropy_anneal_start = 200_000
    cfg.latent_entropy_anneal_end = 700_000
    cfg.latent_entropy_objective = "maximize"

    cfg.latent_strategy_ppo_coef = 0.0
    cfg.latent_episode_strategy_ppo = True
    cfg.latent_episode_strategy_coef = 0.30
    cfg.latent_episode_strategy_clip_eps = 0.2
    cfg.latent_episode_strategy_value_coef = 0.5
    cfg.latent_episode_strategy_return_norm = True
    cfg.latent_episode_strategy_warmup_decision_steps = 5
    cfg.latent_q_phi_option_advantage = False
    return cfg


def apply_plan_faithful_latent_v3i18_v3i16_plus_128(cfg: PPOConfig) -> PPOConfig:
    """v3i18: conservative ``v3i16 + 128`` -- only the resample interval changes.

    Hypothesis: v3i16 had the best-behaved actor-z embedding path of the v3iX
    family but its 64-step strategic horizon may be too short. Doubling the
    persistence window to 128 decision steps gives z a longer arc without
    touching any other dial.

    Inherits ``apply_plan_faithful_latent_v3i16_policy_z_embedding`` verbatim
    and changes exactly **one** runtime knob:

    * ``latent_resample_every_n``: ``64`` -> ``128``

    Everything else from v3i16 is preserved bit-for-bit:

    * actor z embedding path: ``latent_z_embed_dim = 16``,
      ``latent_actor_z_onehot_enabled = False``,
      ``latent_actor_z_adapter_enabled = False``,
      ``latent_actor_z_film_layers = 1``
    * ``latent_strategy_ppo_coef = 0.30`` (per-step PPO strategy gradient,
      inherited from ``apply_plan_faithful_base``)
    * ``latent_lam_p = 0.02`` (persistence loss within the 128-step window)
    * ``latent_lam_h = 0.001`` flat (no anneal)
    * ``latent_episode_strategy_ppo = False`` (no episode-credit channel)
    * no AWRD, no v3i3 event preference, no preference distillation
    * no supervised labels, no phase / flag / outcome heads, no opponent heads
    * no behavior-contrast loss, no actor-z separation loss,
      no usage-balance / marginal-balance / specialist pressure
    * ``latent_event_refresh_enabled = False``,
      ``latent_sparse_tactical_refresh_enabled = False``
    """
    cfg = apply_plan_faithful_latent_v3i16_policy_z_embedding(cfg)

    cfg.latent_resample_every_n = 128

    cfg.run_tag = "v3i18_v3i16_plus_128_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3i17_episode_arc(cfg: PPOConfig) -> PPOConfig:
    """v3i17 episode-arc: one z per episode, consequence-only gradient.

    "Strategy needs a story arc, not a 5-second costume change."

    Differences vs v3i16:

    * ``latent_resample_every_n = 0``  -- z is sampled once at episode start
      and held for the entire episode. No mid-episode refreshes.
    * ``latent_lam_p = 0.0``  -- with z fixed for the episode, persistence
      loss is a no-op; zeroing it out avoids stale telemetry.
    * ``latent_lam_h`` anneals 0.003 -> 0.0 from 200k -> 700k (consequence-
      only past 700k).
    * ``latent_event_refresh_enabled = False`` (already inherited; reaffirmed).
    * Episode-credit PPO is the sole gradient to q_phi.

    Faithful guarantee unchanged: no labels, no aux heads, no opponent ID.
    """
    cfg = _apply_v3i17_consequence_only(cfg)

    cfg.latent_resample_every_n = 0
    cfg.latent_resample_on_flag = False
    cfg.latent_event_refresh_enabled = False
    cfg.latent_sparse_tactical_refresh_enabled = False
    cfg.latent_lam_p = 0.0
    cfg.latent_kl_consecutive = 0.0
    cfg.latent_gae_reset_on_z_change = True

    cfg.run_tag = "v3i17_episode_arc_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3i17_long_arc(cfg: PPOConfig) -> PPOConfig:
    """v3i17 long-arc: 256-step z persistence, consequence-only gradient.

    Sibling to ``v3i17_episode_arc``. Keeps the option of mid-episode z
    refreshes but extends the dwell to a 256-step "story arc" -- 4x longer
    than v3i16's 64-step costume change.

    Differences vs v3i16:

    * ``latent_resample_every_n = 256`` (was 64).
    * ``latent_lam_p = 0.01`` (was 0.02) -- small switch cost preserved so
      within-arc continuity is encouraged, but lighter since 256 steps is
      already a long arc.
    * ``latent_lam_h`` anneals 0.003 -> 0.0 from 200k -> 700k.
    * Episode-credit PPO is the sole consequence channel; per-step strategy
      coupling stays off.
    """
    cfg = _apply_v3i17_consequence_only(cfg)

    cfg.latent_resample_every_n = 256
    cfg.latent_resample_on_flag = False
    cfg.latent_event_refresh_enabled = False
    cfg.latent_sparse_tactical_refresh_enabled = False
    cfg.latent_lam_p = 0.01
    cfg.latent_kl_consecutive = 0.0
    cfg.latent_gae_reset_on_z_change = True

    cfg.run_tag = "v3i17_long_arc_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3i19_summer_consequence(cfg: PPOConfig) -> PPOConfig:
    """v3i19: Summer-faithful per-arc consequence credit.

    Diagnosis driving the design (from v3i18 telemetry):

    * v3i18's z had near-max entropy AND near-zero MI with any context.
    * Per-step ``latent_strategy_ppo_coef = 0.30`` was too noisy a credit
      pathway when ~200 z decisions per episode shared a single env return.
    * Behavior fingerprints across z values were statistically identical;
      the actor learned to ignore z.

    v3i19 changes credit assignment, not the conceptual design. Same K=4,
    same global-state q_phi, same shared actor, same critic-z conditioning,
    same persistence + entropy regularisation, same task-reward-only signal.
    The only change is HOW the task-reward gradient reaches q_phi.

    Recipe (per the locked design):

    1. **Sparse refresh with optional flag-event reactivity** --
       ``latent_resample_every_n = 64`` (vs v3i18's 128). Optional flag-
       event refresh on territory changes via ``latent_resample_on_flag``.
       This gives q_phi more chances to react to flag-state transitions
       while keeping persistence in effect within each arc.
    2. **Persistence**: ``latent_lam_p = 0.03`` (range 0.01-0.05 from plan;
       prevents thrashing without freezing z forever).
    3. **Entropy decays from 0.003 -> 0.0002 over 300k steps.** Early
       training keeps z alive; later training stops paying q_phi to spin
       the roulette wheel. Still Summer-faithful: entropy regularisation
       remains present as a collapse guard, not as the primary objective.
    4. **Per-arc credit replaces per-step PPO.**
       ``latent_strategy_ppo_coef = 0.0``,
       ``latent_episode_strategy_ppo = False``,
       ``latent_arc_credit_enabled = True``,
       ``latent_arc_credit_coef = 1.0``.
       At each z-decision boundary the trainer saves
       (ctx_at_arc_start, z, log_prob(z), V_phi(ctx)). When the arc ends
       (next z-resample or episode end), arc_return = sum env reward over
       the arc, arc_advantage = arc_return - V_phi(ctx). Normalized within
       the rollout batch. q_phi loss = clipped PPO ratio * advantage.
    5. **Stronger actor z conditioning via architecture** (FiLM + onehot
       concat). The actor receives both the FiLM scale/shift modulation
       from the z embedding AND an onehot z appended to the per-agent vector.
       Still inside the Summer policy form pi_i(a_i | o_i, z); no separate
       per-z heads. This makes z harder for the actor to ignore without
       any auxiliary supervision.
    6. **Critic z conditioning** remains on (inherited from v3i16): the
       centralized critic sees ``concat(global_state, joint_actions, z_onehot)``.

    Plan-faithful contract maintained:

    * No labels, no opponent IDs, no phase/flag/outcome heads.
    * No reconstruction loss, no auxiliary prediction heads.
    * No handcrafted strategy rewards, no role-labelled bonuses.
    * Critic-z, persistence, sparse refresh, and entropy regularisation
      all explicitly endorsed by the Summer plan.

    Minimum proof thresholds (analysis tool):

    * ``normalized_MI_z_opponent`` > 0.02 (v3i18 ~= 0.0001)
    * ``normalized_MI_z_phase`` > 0.01-0.02 (v3i18 ~= 0.00006)
    * ``normalized_MI_z_flag`` > 0.02 (v3i18 ~= 0.00024)
    * ``behavior_by_z`` clear spread in >= 3 signals (v3i18: tiny)
    * fixed-z behavior visibly different across z (v3i18: identical)

    The success criterion is "z carries nonzero consequence", NOT WR alone.
    """
    cfg = apply_plan_faithful_latent_v3i16_policy_z_embedding(cfg)

    # 1. Sparse refresh ONLY. ``latent_resample_on_flag`` is disabled because
    #    the current ``_apply_flag_resample_trigger`` fires on continuous
    #    distance-feature changes (>1e-4 in slice [8:12]), which means it
    #    triggers a z resample on essentially every decision step in a 4v4
    #    game with moving agents. The first v3i19 launch confirmed this: arcs
    #    averaged 1.3 steps, 100% dropped by ``min_len=32``, q_phi grad
    #    stayed at 0.0. The user's spec called this "optional if easy"; it's
    #    not easy in the current implementation. Future revisit could enable
    #    the more disciplined ``latent_event_refresh_enabled`` path (which
    #    has min_gap_steps + max_per_episode guardrails and uses discrete
    #    capture-bit transitions instead of distance deltas).
    cfg.latent_resample_every_n = 64
    cfg.latent_resample_on_flag = False
    cfg.latent_event_refresh_enabled = False
    cfg.latent_sparse_tactical_refresh_enabled = False
    cfg.latent_gae_reset_on_z_change = True
    cfg.latent_kl_consecutive = 0.0

    # 2. Persistence (range 0.01-0.05; doubled from v3i16/v3i18's 0.02 to
    #    discourage thrashing now that sparse + flag refresh combine to
    #    propose more resamples per episode).
    cfg.latent_lam_p = 0.03

    # 3. Entropy schedule: 0.003 -> 0.0002 over 300k steps. Collapse guard
    #    early, near-zero late. ``latent_entropy_anneal_start = 0`` so the
    #    decay begins immediately; ``_end`` is the user-spec 300_000 mark.
    cfg.latent_lam_h = 0.003
    cfg.latent_lam_h_start = 0.003
    cfg.latent_lam_h_end = 0.0002
    cfg.latent_entropy_anneal_start = 0
    cfg.latent_entropy_anneal_end = 300_000
    cfg.latent_entropy_objective = "maximize"

    # 4. Per-arc consequence credit. Per-step PPO and episode-credit OFF.
    cfg.latent_strategy_ppo_coef = 0.0
    cfg.latent_episode_strategy_ppo = False
    cfg.latent_episode_strategy_coef = 0.0
    cfg.latent_arc_credit_enabled = True
    cfg.latent_arc_credit_coef = 1.0
    cfg.latent_arc_credit_baseline = "context_value"
    cfg.latent_arc_credit_return_norm = True
    cfg.latent_arc_credit_min_len = 32
    cfg.latent_arc_credit_n_epochs = 4
    cfg.latent_arc_credit_clip_eps = 0.2

    # 5. FiLM + onehot concat actor z conditioning. Stronger than v3i16's
    #    FiLM-only path; still shared-actor form pi_i(a_i | o_i, z).
    cfg.latent_actor_z_onehot_enabled = True
    cfg.latent_actor_z_onehot_scale = 1.0
    cfg.latent_z_embed_dim = 16
    cfg.latent_actor_z_film_layers = 1

    # 6. Defensive zeroing of every "z existence" pressure / labelled head /
    #    aux objective (mostly inherited from v3i16 already; reaffirmed
    #    here for audit clarity).
    cfg.latent_strategy_aux_predict_phase_coef = 0.0
    cfg.latent_strategy_aux_return_head = False
    cfg.latent_strategy_aux_return_coef = 0.0
    cfg.latent_forced_z_episode_frac = 0.0
    cfg.latent_behavior_contrast_coef = 0.0
    cfg.latent_actor_z_separation_coef = 0.0
    cfg.latent_actor_z_separation_start_coef = 0.0
    cfg.latent_usage_balance_coef = 0.0
    cfg.latent_preference_coef = 0.0
    cfg.latent_preference_commit_coef = 0.0
    cfg.latent_awrd_enabled = False
    cfg.latent_awrd_coef = 0.0
    cfg.latent_specialist_router_enabled = False
    cfg.latent_marginal_balance_coef = 0.0
    cfg.latent_conditional_entropy_min_coef_start = 0.0
    cfg.latent_conditional_entropy_min_coef = 0.0
    cfg.latent_context_mi_coef = 0.0
    cfg.latent_v3i3_event_preference_enabled = False
    cfg.latent_v3i3_event_preference_coef = 0.0
    cfg.latent_v3i3_refresh_log_enabled = False

    cfg.run_tag = "v3i19_summer_consequence_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v4i1_strategic_pressure_qprobe(cfg: PPOConfig) -> PPOConfig:
    """v4i1: Strategic Pressure Benchmark + Offline Return Contrast Probe.

    Inherits v3i19 verbatim. The only deltas are:

    1. The opponent pool is restricted to ``{OP5, OP6, OP7}`` so different z
       values have a strategic reason to differ. v3i18/v3i19 trained against
       OP0..OP6 mixtures that included free-win opponents; the agent could
       win without specializing, so the latent had no job. v4i1 removes
       those easy opponents to force the environment to reward distinct
       strategies (OP5 = aggressive flag rush, OP6 = defensive turtle,
       OP7 = switcher / coordination-and-timing).
    2. The run_tag is updated.

    The latent machinery is **intentionally unchanged** -- v4i1 stops
    changing the brain and changes the world instead. Same K=4, same q_phi,
    same arc-credit, same FiLM+onehot actor conditioning, same entropy
    schedule.

    Primary metric for this run is computed OUT-OF-BAND by
    ``tools/q_probe.py``:

        return_contrast = max_z(R) - min_z(R)

    where R is the mean undiscounted episode return per forced z across
    matched probe seeds, per opponent. Failure: contrast < 0.05 means the
    environment does not care about strategy (escalate to Environment v2).
    Success: contrast >= 0.10-0.20 means different z choices create
    different outcomes (proceed to v4i2 = latent regret specialization).

    All existing in-trainer latent diagnostics (MI(z;*), policy_z_sensitivity_KL,
    actor_z_jsd, H(z), behavior_by_z) keep emitting unchanged and are
    demoted to secondary signals.
    """
    cfg = apply_plan_faithful_latent_v3i19_summer_consequence(cfg)

    cfg.opponent_randomize = True
    cfg.opponent_pool = ("OP5", "OP6", "OP7")
    cfg.opponent_pool_weights = ()

    cfg.run_tag = "v4i1_strategic_pressure_qprobe_OP5_OP6_OP7_2m_4v4"
    return cfg


def apply_plan_faithful_latent_v4i3_summer_proof(cfg: PPOConfig) -> PPOConfig:
    """v4i3 (canonical): Summer-Faithful Proof Suite training preset.

    Thesis: under the locked Summer design, does a discrete shared latent
    strategy ``z`` become a meaningful team-level coordination signal?

    v4i3 inherits v4i1 verbatim. The point of v4i3 is **not** to add new
    latent machinery -- it is to *prove or falsify* the Summer plan
    cleanly. No distillation, no auxiliary heads, no labels, no router
    tutoring. The Summer plan's strict claim is that q_phi learns to use
    z end-to-end from reward alone; v4i3 is the experiment that tests
    that claim with proper baselines and proper counterfactual probing
    (see ``tools/q_probe_local_counterfactual.py`` and
    ``tools/summer_proof_report.py``).

    All deltas vs v4i1 are defensive re-assertions (audit clarity). The
    actual config is identical to v4i1 except for the run_tag and the
    explicit guards on post-Summer extensions:

    * ``latent_router_distill_enabled = False`` -- router distillation is
      a v4i4 extension; v4i3 must be a faithful Summer run.
    * ``latent_strategy_aux_predict_phase_coef = 0.0`` and
      ``latent_strategy_aux_return_head = False`` -- no auxiliary
      prediction heads. The Summer plan is strict about z being learned
      end-to-end from task reward.

    Proof artifacts produced after training:

    * Fixed-z q_probe (``tools/q_probe.py``)  -- forced-z return contrast
      per (opp, seed) at matched starts; proves latent modes exist.
    * Local counterfactual probe
      (``tools/q_probe_local_counterfactual.py``) -- at each arc boundary,
      snapshot env state, force each z, roll to completion. Proves
      Q(s, z) contrast at the exact decision points where q_phi acts.
    * No-latent baseline (``apply_plan_faithful_no_latent_v4i3_baseline``)
      run at the same budget; proves the gain from z (if any) over a
      same-everything-except-latent control.
    * Natural q_phi rollout vs. fixed-z oracle and random-z baseline
      (``tools/qualitative_rollout.py``) -- proves q_phi is routing.
    * Summary report (``tools/summer_proof_report.py``) gates 1-5 of the
      Summer-Proof spec.

    If v4i3 passes the gates, the Summer plan is alive. If it fails, the
    honest follow-up is v4i4 (counterfactual router refinement) framed
    as a clearly-labelled post-Summer extension, not a "fix" of the
    Summer plan itself.
    """
    cfg = apply_plan_faithful_latent_v4i1_strategic_pressure_qprobe(cfg)

    # Summer-faithful latent machinery (all already set by v3i19 chain;
    # re-asserted here so any drift in upstream presets is caught by
    # config-diff at PR review time, NOT at run start). Same K=4, same
    # 64-step sparse refresh, same lam_p / lam_h_start / lam_h_end, same
    # arc-credit recipe.
    cfg.latent_k = 4
    cfg.latent_resample_every_n = 64
    cfg.latent_resample_on_flag = False
    cfg.latent_lam_p = 0.03
    cfg.latent_lam_h_start = 0.003
    cfg.latent_lam_h_end = 0.0002
    cfg.latent_strategy_ppo_coef = 0.0
    cfg.latent_arc_credit_enabled = True
    cfg.latent_arc_credit_coef = 1.0

    # Strategic-pressure pool (v4i1 already sets this; re-assert so the
    # CLI guard at training/cli.py is not the only place this is enforced).
    cfg.opponent_randomize = True
    cfg.opponent_pool = ("OP5", "OP6", "OP7")
    cfg.opponent_pool_weights = ()

    # Explicitly OFF -- post-Summer extensions that v4i3 must NOT include.
    # ``latent_router_distill_enabled`` is the v4i4post periodic-distill
    # hook. The aux predict-phase / return heads are forbidden by the
    # Summer plan's "no auxiliary objectives" clause.
    cfg.latent_router_distill_enabled = False
    cfg.latent_strategy_aux_predict_phase_coef = 0.0
    cfg.latent_strategy_aux_return_head = False
    cfg.latent_strategy_aux_return_coef = 0.0

    # Run tag is budget-agnostic on purpose: the preset locks the
    # Summer-faithful machinery, NOT a specific ``--total-steps``. Probes
    # at smaller budgets (e.g. 1M) and the locked 2M proof run share the
    # same artifacts namespace; if you need separate trees, pass
    # ``--run-tag`` to override.
    cfg.run_tag = "v4i3_summer_proof_OP5_OP6_OP7_4v4"
    return cfg


def apply_plan_faithful_no_latent_v4i3_baseline(cfg: PPOConfig) -> PPOConfig:
    """v4i3 no-latent baseline: the same-everything-except-z control.

    The Summer plan calls the no-latent ablation decisive: replace
    ``pi(a | o, z)`` with ``pi(a | o)`` and keep everything else identical.
    To honour "everything else identical", this preset inherits v4i1
    verbatim (same reward, same arc-credit math, same entropy schedule,
    same opponent pool ``{OP5, OP6, OP7}``, same PPO knobs, same map,
    same n_envs, same n_epochs, same n_steps, same total budget) and
    flips ONLY ``use_latent_strategy = False``.

    Important note about the ancestry choice: there is no pre-latent
    v3iN ancestor in the file that mirrors v4i1's reward / opponent
    pool / arc-credit machinery. ``apply_plan_faithful_no_latent`` (the
    legacy 1M-step 2v2 OP3 baseline) does NOT mirror v4i1; using it as
    a control would confound the latent ablation with ~8 other deltas
    (timesteps, team size, opponent pool, reward shaping, arc-credit
    on/off, FiLM scaffolding, ...). Inheriting v4i1-and-flipping is the
    only honest way to do the ablation in this codebase.

    Latent-only coefficients (``latent_arc_credit_*``,
    ``latent_episode_strategy_*``, ``latent_lam_*``, ``latent_actor_z_*``,
    ``latent_router_distill_*``) become no-ops when
    ``use_latent_strategy = False``; we still defensively zero the most
    consequential ones for audit clarity. Anything related to z that
    survives must be either (a) a pure config field with no runtime
    consequence under no-latent, or (b) a bug that needs fixing.
    """
    cfg = apply_plan_faithful_latent_v4i1_strategic_pressure_qprobe(cfg)

    cfg.use_latent_strategy = False
    cfg.fixed_latent_strategy = False
    cfg.latent_arc_credit_enabled = False
    cfg.latent_arc_credit_coef = 0.0
    cfg.latent_strategy_ppo_coef = 0.0
    cfg.latent_episode_strategy_ppo = False
    cfg.latent_episode_strategy_coef = 0.0
    cfg.latent_strategy_aux_return_head = False
    cfg.latent_strategy_aux_return_coef = 0.0
    cfg.latent_strategy_aux_predict_phase_coef = 0.0
    cfg.latent_router_distill_enabled = False
    cfg.latent_actor_z_onehot_enabled = False

    # Same budget-agnostic naming convention as the latent preset above.
    cfg.run_tag = "v4i3_no_latent_baseline_OP5_OP6_OP7_4v4"
    return cfg


def apply_plan_faithful_latent_v5_strict_summer(cfg: PPOConfig) -> PPOConfig:
    """v5 (strict-Summer): the literal docs/algorithm.md objective.

    The Summer plan's locked loss is::

        L = L_PPO + lam_p * L_persist - lam_H * H(q_phi(z | s))

    with the explicit clause "PPO clipped ratio uses action log-probs only;
    q_phi is trained through strategy entropy and persistence, plus optional
    consecutive KL." That excludes every auxiliary q_phi gradient channel
    the post-Summer chain accumulated -- per-step strategy PPO
    (``latent_strategy_ppo_coef``), per-episode credit
    (``latent_episode_strategy_ppo``), per-arc credit
    (``latent_arc_credit_enabled``), aux return prediction
    (``latent_strategy_aux_return_head``), and aux phase prediction
    (``latent_strategy_aux_predict_phase_coef``).

    v4i3 inherited the v3i19 arc-credit channel (``coef = 1.0``,
    ``baseline = context_value``). Useful for proving "credit can pull
    q_phi off uniform when given a per-arc PG signal", but not literally
    Summer-strict. v5 is the experiment that tests *whether the docs/
    algorithm.md loss alone* (entropy + persistence on q_phi, with the
    actor receiving z via a plain ``nn.Embedding(K, d_z)`` concat) is
    enough to differentiate the four latent strategies.

    Recipe (one-variable changes vs v4i3):

    1. **No auxiliary q_phi PG channels.**
       ``latent_arc_credit_enabled = False`` (coef = 0),
       ``latent_episode_strategy_ppo = False`` (coef = 0),
       ``latent_strategy_ppo_coef = 0.0``,
       ``latent_strategy_aux_return_head = False``,
       ``latent_strategy_aux_predict_phase_coef = 0.0``.
    2. **Strict actor-z conditioning per algorithm.md.** Only
       ``nn.Embedding(K, d_z)`` concatenated to per-agent features
       (``latent_z_embed_dim = 16``). FiLM
       (``latent_actor_z_adapter_enabled = False``) and z-onehot concat
       (``latent_actor_z_onehot_enabled = False``) are disabled because
       neither appears in the Summer plan's actor spec.
    3. **Regularizers preserved.** ``latent_lam_p = 0.03``,
       ``latent_lam_h`` schedule 0.003 -> 0.0002 over 300k steps,
       ``latent_resample_every_n = 64``. Matches v4i3 exactly.

    Required gate fix (already in place at this commit): the v5 gate in
    ``ppo_updater.update`` no longer silences the main-loop q_phi loss
    when ``latent_strategy_ppo_coef == 0``. It silences only when a
    dedicated ``latent_router_optimizer`` is active (the v3c safeguard).
    Without that fix, ``lam_p`` and ``lam_h`` would be silently zeroed
    here and q_phi would receive zero gradient. See the comment block at
    ``ppo_updater._gate_q_phi_main_loop`` / the ``MainLoopGatingTests``
    in ``test_marginal_baseline.py``.

    Plan-faithful contract (re-asserted):

    * No labels, no opponent IDs, no phase/flag/outcome heads.
    * No reconstruction loss, no auxiliary prediction heads.
    * No handcrafted strategy rewards, no role-labelled bonuses.
    * Critic-z, persistence, and entropy regularisation are explicitly
      endorsed by the Summer plan.

    Expected outcome (this preset's role in the proof table):

    * If H(q_phi) collapses to a single z and/or stays at ln(K), and the
      WR matches the no-latent v4i3 baseline, then the literal-strict
      reading of docs/algorithm.md does NOT actually train q_phi from
      reward. The arc-credit / episode-credit / per-step PG channels in
      v4i3 / v3c were each an answer to this problem.
    * If H(q_phi) sharpens and WR exceeds the no-latent baseline by a
      paired-bootstrap-significant margin, the Summer plan is alive
      *exactly as written*.

    The same opponent pool / map / budget as v4i3 must be used to make
    the comparison meaningful.
    """
    cfg = apply_plan_faithful_latent_v4i3_summer_proof(cfg)

    # 1. Disable every auxiliary q_phi PG / supervision channel.
    cfg.latent_arc_credit_enabled = False
    cfg.latent_arc_credit_coef = 0.0
    cfg.latent_episode_strategy_ppo = False
    cfg.latent_episode_strategy_coef = 0.0
    cfg.latent_strategy_ppo_coef = 0.0
    cfg.latent_strategy_aux_return_head = False
    cfg.latent_strategy_aux_return_coef = 0.0
    cfg.latent_strategy_aux_predict_phase_coef = 0.0

    # 2. Strict actor-z conditioning per docs/algorithm.md: nn.Embedding only.
    cfg.latent_actor_z_onehot_enabled = False
    cfg.latent_actor_z_onehot_scale = 0.0
    cfg.latent_actor_z_adapter_enabled = False
    cfg.latent_actor_z_adapter_scale = 0.0
    cfg.latent_actor_z_film_layers = 1  # ignored when adapter disabled
    cfg.latent_z_embed_dim = 16
    cfg.latent_actor_z_embed_scale = 1.0

    # 3. Defensive zero on every post-Summer separation / preference / specialist
    #    loss inherited indirectly through the v3i19 chain. Most are already
    #    zero in v4i3; re-asserted here so config-diff at PR time catches drift.
    cfg.latent_forced_z_episode_frac = 0.0
    cfg.latent_behavior_contrast_coef = 0.0
    cfg.latent_actor_z_separation_coef = 0.0
    cfg.latent_actor_z_separation_start_coef = 0.0
    cfg.latent_usage_balance_coef = 0.0
    cfg.latent_preference_coef = 0.0
    cfg.latent_preference_commit_coef = 0.0
    cfg.latent_awrd_enabled = False
    cfg.latent_awrd_coef = 0.0
    cfg.latent_specialist_router_enabled = False
    cfg.latent_marginal_balance_coef = 0.0
    cfg.latent_conditional_entropy_min_coef = 0.0
    cfg.latent_conditional_entropy_min_coef_start = 0.0
    cfg.latent_context_mi_coef = 0.0
    cfg.latent_v3i3_event_preference_enabled = False
    cfg.latent_v3i3_event_preference_coef = 0.0
    cfg.latent_router_distill_enabled = False

    cfg.run_tag = "v5_strict_summer_OP5_OP6_OP7_2m_4v4"
    return cfg


def apply_plan_faithful_latent_v5i1_reward_credit_router(
    cfg: PPOConfig,
) -> PPOConfig:
    """v5i1: reward-credit repair for the collapsed strict-Summer router.

    ``v5_strict_summer`` proved that persistence plus entropy does not provide
    task-return credit to q_phi. Persistence self-reinforces whichever latent
    wins the early sampling race, while the entropy coefficient anneals too
    quickly to keep all four choices alive.

    This additive preset preserves v5's plain ``nn.Embedding`` actor contract
    and all no-label/no-auxiliary-head guards, but makes q_phi trainable from
    task reward:

    * commit one z per episode after five context-building decision steps;
    * optimize that sampled z from completed-episode return;
    * subtract the detached z-marginal value baseline;
    * use six router PPO epochs and a dedicated 5e-3 router learning rate;
    * retain a 1e-3 entropy floor as collapse insurance.

    No opponent ID, phase label, handcrafted strategy reward, preference
    target, or distillation target enters the router.
    """
    cfg = apply_plan_faithful_latent_v5_strict_summer(cfg)

    cfg.latent_resample_every_n = 0
    cfg.latent_resample_on_flag = False
    cfg.latent_lam_p = 0.0

    cfg.latent_lam_h = 0.003
    cfg.latent_lam_h_start = 0.003
    cfg.latent_lam_h_end = 0.001
    cfg.latent_entropy_anneal_start = 200_000
    cfg.latent_entropy_anneal_end = 700_000

    cfg.latent_episode_strategy_ppo = True
    cfg.latent_episode_strategy_coef = 0.30
    cfg.latent_episode_strategy_warmup_decision_steps = 5
    cfg.latent_episode_strategy_n_epochs = 6
    cfg.latent_episode_strategy_lr = 5e-3
    cfg.latent_q_phi_marginal_baseline = True

    cfg.run_tag = "v5i1_reward_credit_router_OP5_OP6_OP7_2m_4v4"
    return cfg


def apply_plan_faithful_latent_v5i2_stronger_z_conditioning(
    cfg: PPOConfig,
) -> PPOConfig:
    """v5i2: strengthen actor controllability with embedding-driven FiLM.

    This experiment inherits the v5i1 episode-level reward-credit router
    unchanged. The only behavioral change is an actor-only FiLM projection
    from the existing learned z embedding into the second hidden layer:

        h' = gamma(z) * h + beta(z)

    The projection starts near identity so the embedding-concat policy is
    preserved at initialization while giving PPO a direct multiplicative and
    additive path from z to the policy head. No specialization loss, diversity
    reward, forced-z balancing, role assignment, critic change, or router
    objective is added.
    """
    cfg = apply_plan_faithful_latent_v5i1_reward_credit_router(cfg)

    cfg.enable_actor_z_film = True
    cfg.actor_z_film_init_scale = 0.02
    cfg.actor_z_film_layer = 2

    cfg.run_tag = "v5i2_stronger_z_conditioning_OP5_OP6_OP7_2m_4v4"
    return cfg


def apply_plan_faithful_latent_v5i4_end_to_end(
    cfg: PPOConfig,
) -> PPOConfig:
    """v5i4: paper-faithful conditional-entropy reference row.

    Built directly on ``v5_strict_summer`` (NOT on v5i1/v5i2/v5i3), with one
    correction: the on-policy categorical PPO term on ``q_phi`` is enabled.

    The Summer-plan claim that ``q_phi`` is "trained end-to-end from task
    reward" requires a score-function gradient on the discrete latent --
    persistence and entropy alone do not transmit task-reward information
    into the router. The categorical strategy PPO term

        L_strategy_PPO = - E[ min( rho(z) * A, clip(rho(z), 1+/-eps) * A ) ]

    where ``A`` is the centralized critic's GAE advantage at each
    resample step and ``rho(z) = pi_phi(z|s) / pi_phi_old(z|s)``, is the
    operational implementation of that claim. It belongs inside

        L_MARL = L_actor_PPO + c_V*L_critic + c_Z*L_strategy_PPO
                 + lam_p*L_persist - lam_H*H(q_phi)

    and is not an auxiliary prediction task, label, preference target,
    role assignment, distillation target, or curriculum. The
    ``latent_strategy_ppo_coef`` coefficient is the ``c_Z`` weight.

    What's ON:
    *  Discrete categorical ``z``, ``K = 4``, ``z_embed_dim = 16``.
    *  Sparse resampling every 64 decisions (``latent_resample_every_n = 64``).
    *  Actor reads ``z`` via a plain ``nn.Embedding(K, d_z)`` concatenated
       to local CNN features + scalar ``vec`` (no FiLM, no adapter,
       no one-hot, no opponent/phase info in the actor).
    *  Centralized critic ``V(s, a, z)`` supplies the baseline.
    *  Strategy persistence (``lam_p = 0.03``) and strategy entropy
       (``lam_h`` 0.003 -> 0.0002 schedule inherited from v4i3).
    *  Main-loop ``q_phi`` PPO with ``c_Z = 0.10`` (the paper's task-reward
       gradient channel for the router).

    What's OFF (and must stay off for the paper-faithful claim):
    *  Episode-credit extension (v5i1's per-episode router PPO + dedicated
       AdamW). Mutually exclusive with the per-step main-loop PG above.
    *  FiLM and any other non-concat actor-z mechanism.
    *  Forced-z exploration curriculum (v5i3) -- no labels, no scheduled
       uniform sampling; the router learns purely from on-policy reward.
    *  Arc-credit (v3i19), preference distillation, AWRD, router distill,
       behavior contrast, specialist router, auxiliary return / phase
       prediction heads, and any other post-Summer channel.
    *  Event-triggered switching (``latent_resample_on_flag = False``).
    *  Sparse-tactical refresh and event refresh disabled (inherited
       from the v4i3 chain).

    Relationship to the rest of the v5 ladder:

    |  Run               | q_phi gradient channels                       |
    |--------------------|-----------------------------------------------|
    | v5_strict_summer   | entropy + persistence (NO task-reward signal) |
    | v5i1               | + per-episode credit (dedicated AdamW)        |
    | v5i2               | v5i1 + FiLM (actor-only, no q_phi change)     |
    | v5i3               | v5i2 + forced-z anneal (actor coverage)       |
    | v5i4 (this preset) | conditional entropy + persistence + per-step main-loop PG |

    v5i4 remains the conditional-entropy comparison row because (a) the
    actor is the embedding-concat one docs/algorithm.md specifies
    literally, (b) the q_phi gradient is the main-loop categorical PPO
    that the paper's "learned end-to-end from task reward" wording
    requires, and (c) no label or auxiliary prediction target enters
    anywhere. v5i6 inherits this contract and becomes the canonical
    Summer interpretation by changing only the entropy reduction to the
    batch marginal.

    The launch-time audit banner is emitted by
    ``rl.training.banner`` when the resolved run_tag
    contains ``v5i4_paper_faithful`` so a reviewer can verify the
    invariants at the top of the log without diffing config snapshots.
    """
    cfg = apply_plan_faithful_latent_v5_strict_summer(cfg)

    # ------------------------------------------------------------------
    # Core paper design.
    # ------------------------------------------------------------------
    cfg.use_latent_strategy = True
    cfg.latent_k = 4
    cfg.fixed_latent_strategy = False
    # Sparse switching every 64 decisions. v5_strict_summer already inherits
    # 64 from v4i3, but re-assert here so a future v4i3 change does not
    # silently shift v5i4's cadence.
    cfg.latent_resample_every_n = 64
    cfg.latent_resample_on_flag = False

    # ------------------------------------------------------------------
    # Literal actor architecture: only nn.Embedding(K, d_z) concat.
    # ------------------------------------------------------------------
    cfg.enable_actor_z_film = False
    cfg.actor_z_film_init_scale = 0.0
    cfg.latent_actor_z_adapter_enabled = False
    cfg.latent_actor_z_adapter_scale = 0.0
    cfg.latent_actor_z_onehot_enabled = False
    cfg.latent_actor_z_onehot_scale = 0.0
    cfg.latent_z_embed_dim = 16
    cfg.latent_actor_z_embed_scale = 1.0

    # ------------------------------------------------------------------
    # Main-loop categorical PPO on q_phi (the paper's task-reward channel
    # for the router). NOT the v5i1 episode-credit channel.
    # ------------------------------------------------------------------
    cfg.latent_strategy_ppo_coef = 0.10
    # Defensive: keep the shared optimizer driving q_phi via the main-loop
    # gate. Setting latent_episode_strategy_lr would create a dedicated
    # AdamW that suppresses the main-loop PG (see
    # ``apply_main_loop_qphi_loss`` in ppo_updater.update and the
    # ``MainLoopGatingTests`` in test_marginal_baseline.py).
    cfg.latent_episode_strategy_lr = None
    cfg.latent_episode_strategy_ppo = False
    cfg.latent_episode_strategy_coef = 0.0
    cfg.latent_episode_strategy_warmup_decision_steps = 0
    cfg.latent_episode_strategy_n_epochs = 1

    # ------------------------------------------------------------------
    # Paper regularizers.
    # ------------------------------------------------------------------
    cfg.latent_lam_p = 0.03
    cfg.latent_lam_h = 0.003
    cfg.latent_kl_consecutive = 0.0
    # Entropy maximization is the default; pin explicitly so a future
    # default flip cannot silently invert the sign in v5i4.
    cfg.latent_entropy_mode = "conditional"
    cfg.latent_entropy_objective = "maximize"

    # ------------------------------------------------------------------
    # Forced-z curriculum OFF (constant zero; resolver short-circuits to
    # the legacy field because all four schedule fields are None).
    # ------------------------------------------------------------------
    cfg.latent_forced_z_episode_frac = 0.0
    cfg.latent_forced_z_episode_frac_start = None
    cfg.latent_forced_z_episode_frac_end = None
    cfg.latent_forced_z_anneal_start = None
    cfg.latent_forced_z_anneal_end = None

    # ------------------------------------------------------------------
    # Explicitly disable every non-paper q_phi channel inherited up the
    # chain. Most are already zero in v5_strict_summer; re-asserted here
    # so config-diff at PR time catches any future drift.
    # ------------------------------------------------------------------
    cfg.latent_arc_credit_enabled = False
    cfg.latent_arc_credit_coef = 0.0
    cfg.latent_strategy_aux_return_head = False
    cfg.latent_strategy_aux_return_coef = 0.0
    cfg.latent_strategy_aux_predict_phase_coef = 0.0
    cfg.latent_router_distill_enabled = False
    cfg.latent_v3i3_event_preference_enabled = False
    cfg.latent_v3i3_event_preference_coef = 0.0
    cfg.latent_behavior_contrast_coef = 0.0
    cfg.latent_actor_z_separation_coef = 0.0
    cfg.latent_actor_z_separation_start_coef = 0.0
    cfg.latent_usage_balance_coef = 0.0
    cfg.latent_preference_coef = 0.0
    cfg.latent_preference_commit_coef = 0.0
    cfg.latent_awrd_enabled = False
    cfg.latent_awrd_coef = 0.0
    cfg.latent_specialist_router_enabled = False
    cfg.latent_marginal_balance_coef = 0.0
    cfg.latent_conditional_entropy_min_coef = 0.0
    cfg.latent_conditional_entropy_min_coef_start = 0.0
    cfg.latent_context_mi_coef = 0.0

    # NOTE: budget tag matches the actual PPOConfig default (1_000_000 timesteps).
    # The v5_strict_summer / v5i1 / v5i2 / v5i3 chain inherited a misleading
    # "_2m_" suffix from v4i1's run_tag even though none of those presets ever
    # overrode total_timesteps from its 1M default. v5i4 corrects the tag so
    # the run-tag and the trainer's reported total_timesteps agree.
    cfg.run_tag = "v5i4_paper_faithful_end_to_end_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor(
    cfg: PPOConfig,
) -> PPOConfig:
    """v5i5: paper-faithful entropy floor.

    Single-axis follow-up to ``v5i4_paper_faithful_end_to_end``. The v5i4
    run shows the actor uses ``z`` and ``q_phi`` receives gradients, but the
    rollout occupancy concentrates heavily on a single latent (~64% on z2
    vs ~7% on z3 at the 150k checkpoint). The smallest intervention aimed
    directly at that failure mode -- without changing the loss objective or
    introducing any new gradient channel -- is to raise the entropy floor
    so the entropy regularizer keeps a stronger pull on under-sampled
    latents late in training.

    Recipe (one-variable change vs v5i4):

    1. ``latent_lam_h_end = 0.001`` (was ``0.0002``). Five times the v5i4
       floor while still inside the documented Summer-plan
       ``lambda_H in [0.001, 0.01]`` range.
    2. Everything else identical to v5i4: concat-only actor (no FiLM /
       adapter / one-hot), ``latent_strategy_ppo_coef = 0.10`` main-loop
       categorical PPO term on ``q_phi``, ``latent_lam_p = 0.03``,
       ``latent_resample_every_n = 64``, no curriculum, no preferences,
       no aux heads, no arc-credit, no episode-credit.

    Classification: PAPER-FAITHFUL. The single change is a hyperparameter
    inside the plan-allowed entropy range; no fidelity rule (R1..R42 in
    ``docs/summer-fidelity-rules.md``) flips state. The run still fires
    the v5i4-family paper-faithful audit banner.

    Decisive comparison: v5i4 vs v5i5 with identical seed, learning rate,
    timesteps, opponent pool {OP5, OP6, OP7}, maps, reward function,
    resampling interval, network architecture, n_envs, and PPO epochs.
    Multiple seeds recommended for headline claims.

    Diagnostics (added in this PR, no new losses):

    * ``effective_num_latents`` = ``exp(strategy_entropy_marginal_nats)``
    * ``latent_occupancy_min`` / ``latent_occupancy_max`` /
      ``latent_occupancy_ratio = max / max(min, eps)``
    * ``mean_strategy_duration`` (rollout-level mean dwell length in
      decisions per latent arc)

    These let a reviewer separate "stronger entropy preserves useful
    diversity" from "stronger entropy makes the router randomly
    uncertain" without needing a new objective term.

    What's deliberately NOT included (would be a different experiment):

    * episode-credit extension (``latent_episode_strategy_ppo``)
    * forced-z curriculum (v5i3-style ``latent_forced_z_episode_frac_*``)
    * supervised phase or opponent labels
    * opponent-ID input to the actor
    * auxiliary return prediction head
    * FiLM / adapter / one-hot actor conditioning
    * behavior diversity rewards
    * handcrafted latent targets
    * marginal-occupancy entropy reward (covered by the separate v5i6
      marginal-entropy interpretation)
    """
    cfg = apply_plan_faithful_latent_v5i4_end_to_end(cfg)

    # Single-variable change: raise the lam_H floor 0.0002 -> 0.001.
    # ``latent_lam_h_start`` (= 0.003), ``latent_entropy_anneal_start``
    # (= 0), and ``latent_entropy_anneal_end`` (= 300_000) are all
    # inherited unchanged from v5i4 -> v5_strict_summer -> v4i3.
    cfg.latent_lam_h_end = 0.001

    # Run tag rolled forward. The audit banner fires when the tag
    # contains ``v5i5_paper_faithful``; the suffix mirrors v5i4's
    # ``_OP5_OP6_OP7_1m_4v4`` (same opponent pool, same total_timesteps
    # of 1_000_000 inherited from v5_strict_summer) so the artifact
    # namespace is parallel and the v5i4-vs-v5i5 comparison is clean.
    cfg.run_tag = "v5i5_paper_faithful_entropy_floor_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v5i6_paper_faithful_marginal_entropy(
    cfg: PPOConfig,
) -> PPOConfig:
    """v5i6: paper-faithful marginal-entropy Summer interpretation.

    Direct child of ``v5i4_paper_faithful_end_to_end``. This preset keeps
    the v5i4 actor, critic, router PPO, persistence, sparse resampling,
    opponent pool, and no-label/no-curriculum contract unchanged. The
    scientific delta is only the entropy reduction:

    * v5i4 / v5i5: maximize mean conditional entropy E_s[H(q_phi(z|s))].
    * v5i6: maximize batch-marginal entropy H(E_s[q_phi(z|s)]) by
      minimizing KL(E_s[q_phi(z|s)] || Uniform).

    The marginal term is driven by the same lambda_H schedule used by v5i5
    (0.003 -> 0.001 over 0..300k), so v5i6 tests the interpretation of
    H(z) as aggregate strategy-repertoire entropy rather than stacking a
    conditional-entropy bonus on top of a usage-balancing extension.
    """
    cfg = apply_plan_faithful_latent_v5i4_end_to_end(cfg)

    cfg.latent_entropy_mode = "marginal"
    cfg.latent_entropy_objective = "maximize"
    cfg.latent_lam_h_end = 0.001
    cfg.latent_usage_balance_coef = 0.0

    cfg.run_tag = "v5i6_paper_faithful_marginal_entropy_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v5i7_entropy_floor_split_lane(
    cfg: PPOConfig,
) -> PPOConfig:
    """v5i7: v5i5 entropy floor on the split-lane map geometry.

    ## Proposed Preset Review

    ### Identity
    - Proposed name: v5i7_entropy_floor_split_lane
    - Parent preset: v5i5_paper_faithful_entropy_floor
    - Classification: PAPER-FAITHFUL
    - Research question: Does the v5i5 entropy-floor fix produce deployed
      latent routing when the task geometry contains lane/chokepoint choices?

    ### Intended delta
    - Fields changed: map_layout, run_tag
    - Why this change is necessary: v5i5's entropy floor can keep a latent
      repertoire alive, but the open map may not create enough return contrast
      for different z choices to matter.
    - Why an existing preset cannot answer the question: v5i5 tests the same
      latent method on the default open arena; it does not test whether explicit
      route geometry makes strategy choice useful.

    ### Fidelity impact
    - Actor architecture changed: NO
    - Router objective changed: NO
    - Exploration schedule changed: NO
    - Reward changed: NO
    - Supervision added: NO
    - Auxiliary task added: NO
    - Resampling changed: NO

    ### Exact deviations from the paper-faithful preset
    - map_layout: map_a_open -> map_b_split_lane; reason: add lane/chokepoint
      structure while preserving the v5i5 latent loss and training contract.
    - run_tag: v5i5... -> v5i7_summer_faithful_entropy_floor_split_lane...; reason:
      artifact namespace must advertise the environment geometry deviation.

    This remains Summer-faithful by inheriting v5i5's actor, critic, q_phi
    losses, entropy schedule, persistence, sparse resampling, opponent pool,
    and no-label/no-curriculum contract. Any comparison against v5i5/no-latent
    must disclose and match map geometry rather than attributing deltas to the
    latent alone.
    """
    cfg = apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor(cfg)

    cfg.map_layout = "map_b_split_lane"
    cfg.run_tag = "v5i7_summer_faithful_entropy_floor_split_lane_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure(
    cfg: PPOConfig,
) -> PPOConfig:
    """v5i8: v5i7 latent contract on the split-lane v2 task-pressure map.

    ## Proposed Preset Review

    ### Identity
    - Proposed name: v5i8_split_lane_v2_task_pressure
    - Parent preset: v5i7_summer_faithful_entropy_floor_split_lane
    - Classification: PAPER-FAITHFUL
    - Research question: Does lower-friction, higher-route-contrast split-lane
      geometry create enough task-return structure for the existing v5i5
      Summer-faithful latent PPO objective to learn deployed strategies?

    ### Intended delta
    - Fields changed: map_layout, run_tag
    - Why this change is necessary: v5i7's first split-lane geometry produced
      high obstacle-collision counts, so navigation friction may drown out the
      strategic route signal.
    - Why an existing preset cannot answer the question: v5i7 tests the first
      split-lane geometry; v5i8 isolates a task-side geometry revision with no
      latent coefficient or objective change.

    ### Fidelity impact
    - Actor architecture changed: NO
    - Router objective changed: NO
    - Exploration schedule changed: NO
    - Reward changed: NO
    - Supervision added: NO
    - Auxiliary task added: NO
    - Resampling changed: NO

    ### Exact deviations from the paper-faithful preset
    - map_layout: map_b_split_lane -> map_b_split_lane_v2; reason: reduce wall
      bump noise and expose clearer route-pressure choices while preserving the
      latent loss and training contract.
    - run_tag: v5i7... -> v5i8_summer_faithful_split_lane_v2_task_pressure...; reason:
      artifact namespace must advertise the environment geometry revision.
    """
    cfg = apply_plan_faithful_latent_v5i7_entropy_floor_split_lane(cfg)

    cfg.map_layout = "map_b_split_lane_v2"
    cfg.run_tag = "v5i8_summer_faithful_split_lane_v2_task_pressure_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v5i8_repertoire_uniform_z(
    cfg: PPOConfig,
) -> PPOConfig:
    """v5i8 repertoire Stage-1 diagnostic: sustained uniform forced-z coverage.

    ## Proposed Preset Review

    ### Identity
    - Proposed name: v5i8_repertoire_uniform_z
    - Parent preset: v5i8_split_lane_v2_task_pressure
    - Classification: DIAGNOSTIC
    - Research question: Is repertoire failure on v5i8 caused mainly by router
      collapse and unequal per-z experience?

    ### Intended delta
    - Fields changed: ``latent_forced_z_episode_frac*``, ``latent_forced_z_anneal_*``,
      ``run_tag`` only.
    - Why this change is necessary: joint router+actor training lets one z
      dominate experience; this ablation removes router choice for the full run
      so every latent receives uniform episode exposure.
    - Why an existing preset cannot answer the question: v5i8 keeps router
      sampling; v5i3 anneals forced coverage back to zero and inherits FiLM.

    ### Fidelity impact
    - Router objective changed: NO (router receives no on-policy episodes while
      forced fraction is 1.0; this is intentional coverage isolation).
    - Exploration schedule changed: YES (100% uniform forced-z episodes).
    - Reward / actor / map / opponents unchanged vs v5i8.
    """
    cfg = apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure(cfg)

    cfg.latent_forced_z_episode_frac_start = 1.0
    cfg.latent_forced_z_episode_frac_end = 1.0
    cfg.latent_forced_z_anneal_start = 0
    cfg.latent_forced_z_anneal_end = int(cfg.total_timesteps)
    cfg.latent_forced_z_episode_frac = 1.0

    cfg.run_tag = "v5i8_repertoire_uniform_z_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v5i9_csia_guided_specialization(
    cfg: PPOConfig,
) -> PPOConfig:
    """v5i9: CSIA-guided latent specialization extension on v5i8.

    ## Proposed Preset Review

    ### Identity
    - Proposed name: v5i9_csia_guided_specialization
    - Parent preset: v5i8_split_lane_v2_task_pressure
    - Classification: SUMMER-COMPATIBLE EXTENSION
    - Research question: Can causal strategic-impact feedback from frozen
      forced-z evaluations improve opponent-adaptive latent specialization?

    ### Intended delta
    - Fields changed: csia_enabled, csia_reward_coef, run_tag. The CSIA
      gate thresholds use the PPOConfig defaults unless overridden by CLI.
    - Why this change is necessary: v5i8 can prove whether forced z causes
      strategy differences, but it does not feed that causal evidence back
      into training when specialization is useful but weak or unstable.
    - Why an existing preset cannot answer the question: v5i8 is a
      task-pressure/evaluation row only. It keeps the original reward path
      unchanged and therefore cannot test causal-impact feedback.

    ### Fidelity impact
    - Actor architecture changed: NO
    - Router objective changed: NO
    - Exploration schedule changed: NO
    - Reward changed: YES, via detached CSIA bonus after gates pass
    - Supervision added: NO
    - Auxiliary task added: NO
    - Resampling changed: NO

    ### Exact deviations from the parent
    - csia_enabled: False -> True; reason: enable the v5i9 extension.
    - csia_reward_coef: 0.0 -> 0.02; reason: add a small detached bonus
      proportional to centered causal strategic-impact advantage S(o,z).
    - run_tag: v5i8... -> v5i9_csia_guided_specialization...; reason:
      artifact namespace must advertise the post-Summer reward extension.

    This preset must not be described as the original Summer plan. It keeps
    v5i8's actor, critic, q_phi loss, entropy floor, persistence, resampling,
    opponent pool, and map geometry, but the trainer-side reward is no longer
    the paper-faithful reward once CSIA gates activate.
    """
    cfg = apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure(cfg)

    cfg.csia_enabled = True
    cfg.csia_reward_coef = 0.02
    cfg.csia_probe_interval = 1
    cfg.csia_min_behavior_spread = 0.10
    cfg.csia_min_interaction_strength = 0.05
    cfg.csia_quality_floor_delta = 0.10
    cfg.csia_require_gates = True
    cfg.csia_min_count_per_cell = 1
    cfg.run_tag = "v5i9_csia_guided_specialization_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v6i1_staged_team_intent_curriculum(
    cfg: PPOConfig,
) -> PPOConfig:
    """v6i1 production staged team-intent curriculum on split-lane v2.

    Inherits v5i8 map/opponent geometry and latent contract, then enables the
    V6I1 phase controller with enforce-mode boundary evaluation and probe.
    Forced-z fraction, CF coefficient, usage KL, and exploration epsilon are
    resolved at runtime from ``resolve_v6i1_*`` schedules — do not set the v5i3
    ``latent_forced_z_anneal_*`` fields on this preset.

    Phase B/C selector learning uses the macro-router path only
    (``apply_macro_strategy_ppo``). Legacy episode-level strategy PPO stays
    off so q_phi is not trained through two overlapping credit channels.
    """
    cfg = apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure(cfg)

    cfg.use_v6i1_curriculum = True
    cfg.training_mode = "staged_team_intent_curriculum"
    cfg.experiment_family = "v6"
    cfg.experiment_id = "v6i1"
    cfg.phase_boundary_gate_mode = "enforce"
    cfg.curriculum_gate_run_boundary_eval = True
    cfg.curriculum_gate_run_probe = True
    cfg.curriculum_nominal_timesteps = int(cfg.total_timesteps)
    cfg.latent_cf_coef_max = 0.01
    cfg.latent_episode_strategy_ppo = False
    cfg.latent_episode_strategy_coef = 0.0
    cfg.latent_episode_strategy_warmup_decision_steps = 0
    cfg.latent_episode_strategy_lr = None
    cfg.latent_usage_balance_coef = 0.0
    cfg.latent_actor_z_separation_coef = 0.0
    cfg.latent_actor_z_separation_start_coef = 0.0
    cfg.run_tag = "v6i1_staged_team_intent_curriculum_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v6i2_staged_team_intent_curriculum(
    cfg: PPOConfig,
) -> PPOConfig:
    """v6i2 staged team-intent curriculum: dual actor/behavioral gate protocol.

    Inherits the v6i1 actor, critic, router, CF objective, and A/B/C phase
    schedule. Only the Phase A evidence and promotion protocol differs: Gate A
    measures CF-batch actor intervention; Gate B composites matched-seed
    behavioral realization (macro pair JSD is diagnostic only).

    Confirmatory v6i2 uses ``latent_cf_coef_max = 1.0`` (calibrated strong CF)
    plus competence-gated pairwise hinge pressure with a worst-pair term and
    persistent weak-pair weighting; v6i1 retains the weak ``0.01`` baseline for
    threshold calibration.
    """
    cfg = apply_plan_faithful_latent_v6i1_staged_team_intent_curriculum(cfg)

    cfg.experiment_id = "v6i2"
    # Confirmatory v6i2: calibrated strong CF ceiling (v6i1 weak baseline keeps 0.01).
    cfg.latent_cf_coef_max = 1.0
    cfg.latent_cf_worst_pair_coef = 0.5
    cfg.latent_cf_weak_pair_boost = 1.0
    cfg.latent_cf_require_competence = True
    cfg.gate_protocol_version = "v6i2_dual_evidence"
    cfg.phase_a_max_end_fraction = 0.70
    # Frozen v6i2 confirmatory gate thresholds; mirrored in docs/v6i2-gate-protocol-freeze.md.
    cfg.actor_jsd_margin = 0.001
    cfg.actor_jsd_floor_fraction = 0.5
    cfg.actor_jsd_min_passing_pairs = 5
    cfg.actor_jsd_consecutive_updates = 3
    cfg.actor_jsd_ema_decay = 0.10
    cfg.macro_jsd_margin = 0.0001
    cfg.macro_jsd_floor_fraction = 0.5
    cfg.macro_jsd_min_passing_pairs = 1
    cfg.macro_jsd_ema_decay = 0.10
    cfg.behavioral_realization_min_opponents_pass = 2
    cfg.behavioral_realization_effect_threshold = 0.02
    cfg.behavioral_realization_adverse_threshold = -0.01
    cfg.behavioral_route_distance_scale = 0.03
    cfg.behavioral_task_behavior_distance_scale = 0.02
    cfg.behavioral_performance_spread_scale = 0.03
    cfg.behavioral_route_distance_weight = 0.25
    cfg.behavioral_task_behavior_distance_weight = 0.50
    cfg.behavioral_performance_spread_weight = 0.25
    cfg.behavioral_aggregate_effect_threshold = 0.75
    cfg.behavioral_min_task_behavior_distance = 0.01
    cfg.behavioral_min_performance_spread = 0.01
    cfg.behavioral_matched_seed_min_seeds_per_opponent = 20
    cfg.curriculum_probe_min_examples = 10
    cfg.run_tag = "v6i2_staged_team_intent_curriculum_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v6i3_strategy_local_comm(
    cfg: PPOConfig,
) -> PPOConfig:
    """v6i3: v6i2 staged curriculum plus local emergent communication.

    Inherits v6i2 dual-evidence gates and adds communication transport as
    Phase A evidence. Listener causal response remains a diagnostic until
    final matched-seed communication-value evaluation.
    """
    cfg = apply_plan_faithful_latent_v6i2_staged_team_intent_curriculum(cfg)

    cfg.experiment_id = "v6i3"
    cfg.gate_protocol_version = "v6i3_strategy_local_comm_v1"
    cfg.communication_enabled = True
    cfg.comm_protocol_version = "v6i3_strategy_local_comm_v1"
    cfg.comm_num_symbols = 5
    cfg.comm_silence_symbol = 0
    cfg.comm_interval_steps = 32
    cfg.comm_delivery_delay_steps = 1
    cfg.comm_radius_cells = 6.0
    cfg.comm_dropout_probability = 0.10
    cfg.comm_entropy_coef = 0.001
    cfg.comm_hold_last_message = True
    cfg.comm_local_only = True
    cfg.comm_include_sender_position = True
    cfg.comm_message_grid_channels = 4
    cfg.comm_cf_include_message_head = False
    # Frozen v6i3 Phase A communication evidence gates; listener response is diagnostic.
    cfg.comm_min_valid_boundaries = 1024
    cfg.comm_min_deliveries = 4096
    cfg.comm_min_symbols_used = 2
    cfg.comm_entropy_floor = 0.0
    cfg.comm_symbol_dominance_ceiling = 1.0
    cfg.comm_listener_jsd_margin = 0.001
    cfg.comm_listener_min_passing_pairs = 3
    cfg.comm_listener_min_states = 64
    cfg.comm_listener_consecutive_updates = 1
    cfg.run_tag = "v6i3_strategy_local_comm_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v6i1_repertoire_only_ablation(
    cfg: PPOConfig,
) -> PPOConfig:
    """v6i1 repertoire-only ablation: uniform forced-z, no staged controller.

    Shares the v6i1 experiment id for artifact grouping but must never mount
    the staged curriculum controller because ``use_v6i1_curriculum=False`` and
    ``training_mode=repertoire_only_ablation``.
    """
    cfg = apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure(cfg)

    cfg.use_v6i1_curriculum = False
    cfg.training_mode = "repertoire_only_ablation"
    cfg.experiment_family = "v6"
    cfg.experiment_id = "v6i1"
    cfg.latent_forced_z_episode_frac_start = 1.0
    cfg.latent_forced_z_episode_frac_end = 1.0
    cfg.latent_forced_z_anneal_start = 0
    cfg.latent_forced_z_anneal_end = int(cfg.total_timesteps)
    cfg.latent_forced_z_episode_frac = 1.0
    cfg.run_tag = "v6i1_repertoire_only_ablation_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v5i3_balanced_warmup(
    cfg: PPOConfig,
) -> PPOConfig:
    """v5i3: forced-z anneal layered on top of v5i2.

    Diagnosis of v5i2: the router collapsed (z2 dominant, z1 near-extinct)
    even though FiLM was wired in. The actor's per-z sensitivity grew but
    only on z values q_phi actually picked, so under-sampled latents stayed
    blind regardless of conditioning bandwidth. v5i3 fixes the *coverage*
    problem the same way exploration noise fixes argmax collapse in plain
    PPO: force a fraction of episodes onto a uniformly-sampled z early in
    training, then anneal that fraction to zero so late training is pure
    router-vs-task-reward.

    Schedule:

    *  ``0 -- 200k``: forced fraction = 0.30. Every latent gets balanced
       actor exposure across roughly the same opponent/phase mix.
    *  ``200k -- 500k``: linearly anneal 0.30 -> 0.00.
    *  ``500k -- 1M``: router-only sampling.

    Forced episodes always route into ``latent_preference_buffer`` and are
    excluded from ``rollout_strategy_episode_records`` (see the
    ``is_forced_z`` branch in ``record_episode_strategy_outcome``), so
    q_phi's PPO update only sees true on-policy episodes; off-policy bias
    on the router is structurally avoided.

    Summer-compatibility: forcing is unlabeled uniform exploration, not
    role assignment. Latent meanings still emerge from task reward via the
    inherited v5i1 episode-credit PPO. The preference-distillation hook
    (``latent_v3i3_event_preference_*``) and router-distill hook stay
    disabled.
    """
    cfg = apply_plan_faithful_latent_v5i2_stronger_z_conditioning(cfg)

    # Anneal schedule. ``resolve_latent_forced_z_frac`` reads these four
    # fields at every episode start; the legacy constant below is set to
    # the start value as a safety so any code that inspects
    # ``cfg.latent_forced_z_episode_frac`` directly (without the resolver)
    # still observes a sane warmup value.
    cfg.latent_forced_z_episode_frac_start = 0.30
    cfg.latent_forced_z_episode_frac_end = 0.00
    cfg.latent_forced_z_anneal_start = 200_000
    cfg.latent_forced_z_anneal_end = 500_000
    cfg.latent_forced_z_episode_frac = 0.30

    # Defensive re-assertions: keep every supervised / preference / distill
    # channel disabled. v5i3 must remain a pure forced-z coverage layer on
    # top of v5i2's router-credit + FiLM stack.
    cfg.latent_behavior_contrast_coef = 0.0
    cfg.latent_actor_z_separation_coef = 0.0
    cfg.latent_actor_z_separation_start_coef = 0.0
    cfg.latent_usage_balance_coef = 0.0
    cfg.latent_preference_coef = 0.0
    cfg.latent_preference_commit_coef = 0.0
    cfg.latent_awrd_enabled = False
    cfg.latent_awrd_coef = 0.0
    cfg.latent_v3i3_event_preference_enabled = False
    cfg.latent_v3i3_event_preference_coef = 0.0
    cfg.latent_router_distill_enabled = False
    cfg.latent_specialist_router_enabled = False

    cfg.run_tag = "v5i3_balanced_warmup_OP5_OP6_OP7_2m_4v4"
    return cfg


def apply_plan_faithful_latent_v4i4post_periodic_router_distill(cfg: PPOConfig) -> PPOConfig:
    """v4i4 (post-Summer extension): Online / Periodic Return-Ranked Router Distillation.

    NOTE: This preset was originally named ``v4i3_periodic_router_distill``,
    but the canonical v4i3 was rescoped to the **Summer Proof Suite**
    (see :func:`apply_plan_faithful_latent_v4i3_summer_proof`). The
    periodic-distill recipe is now explicitly framed as a **post-Summer
    extension** because it introduces counterfactual router supervision,
    which the Summer plan's "no labels, no auxiliary objectives" clause
    forbids. v4i4 is meaningful only AFTER v4i3 has either passed or
    failed its gates; if v4i3 passes, v4i4 is icing; if v4i3 fails, v4i4
    is the next honest experiment.

    Inherits the v4i1 strategic-pressure setup verbatim (same actor, critic,
    reward, opponent pool ``{OP5, OP6, OP7}``, arc-credit math, entropy
    schedule, and PPO loop). The only delta versus v4i1 is that the trainer
    enables :class:`PeriodicRouterDistillHook`: every
    ``latent_router_distill_every_n_steps`` global steps, the trainer pauses
    after saving a checkpoint, spawns

      1. ``tools/q_probe.py``  -- matched-start return contrast + saved
         q_phi contexts on the just-saved checkpoint,
      2. ``tools/router_distill_from_qprobe.py`` -- offline KL distillation
         of ``strategy_encoder`` (q_phi) from those returns,

    then hot-swaps the distilled ``strategy_encoder.*`` weights back into
    the running model and clears the Adam moments for those params on both
    the main optimizer and the dedicated router optimizer.

    Pre-v4i4 story (recap):

    * v4i1: matched-start forced-z probes prove latent modes exist
      (large per-seed return contrasts across OP5/OP6/OP7).
    * v4i2 (offline): ``router_distill_from_qprobe.py`` proves a small
      offline distill round can teach q_phi to route into those modes
      from saved contexts.
    * v4i3 (Summer Proof Suite): proves / falsifies whether pure Summer
      end-to-end routing (no distill, no aux heads) is sufficient.

    v4i4 lifts the offline distill loop into training so q_phi keeps
    catching up to the actor as PPO drifts. The hook is **best-effort**:
    any subprocess or hot-swap failure is logged and training continues
    with the pre-distill weights, so v4i4 cannot deadlock or corrupt PPO
    state.

    Defaults aimed at the 2M-step 4v4 budget:

    * cadence:    250k steps  (8 distill rounds per 2M-step run)
    * probe:      8 seeds x 3 opponents x latent_k=4 = 96 episodes per round
    * distill:    100 epochs, lr 1e-4, temperature 1.0
    * device:     cpu  (so the GPU is not contended; the round runs while
                  the PPO process is paused after the periodic save)
    * artifacts:  ``<checkpoint_dir>/v4i4post_router_distill/step_<N>/``
    """
    cfg = apply_plan_faithful_latent_v4i1_strategic_pressure_qprobe(cfg)

    cfg.latent_router_distill_enabled = True
    cfg.latent_router_distill_every_n_steps = 250_000
    cfg.latent_router_distill_n_seeds = 8
    cfg.latent_router_distill_base_seed = 1000
    cfg.latent_router_distill_opponents = ("OP5", "OP6", "OP7")
    cfg.latent_router_distill_epochs = 100
    cfg.latent_router_distill_lr = 1e-4
    cfg.latent_router_distill_temperature = 1.0
    cfg.latent_router_distill_weight_decay = 0.0
    cfg.latent_router_distill_device = "cpu"
    cfg.latent_router_distill_artifacts_subdir = "v4i4post_router_distill"

    cfg.run_tag = "v4i4post_periodic_router_distill_OP5_OP6_OP7_2m_4v4"
    return cfg


def apply_plan_faithful_latent_v3d_smart_router(cfg: PPOConfig) -> PPOConfig:
    """v3d: context-bucketed marginal baseline ("smart coach router").

    The v3c experimental finding (~851k steps): the dedicated router
    optimizer + 6 inner epochs DID move q_phi off uniform (zH dropped to
    1.335(0.96), z_occ went [.25, .20, .38, .17]). Router gradient is alive
    (episode-credit grad_norm ~0.02). But the movement is "global z preference"
    (z2 is just popular) not "context-conditioned z selection" (different z
    for different situations). MI(z; opponent) still hovering near noise.

    Diagnosis (per analyst): the v3c V-marginal baseline ``mean_k V(s, z_k)``
    depends on V being well-calibrated for off-policy z slots, but V only sees
    value-loss updates for episodes where each z was *actually picked* (~25%
    at uniform). So the marginal baseline subtracts noise approximating the
    right thing -- the cross-z signal q_phi needs is fuzzy.

    v3d replaces the baseline with an *empirical* per-bucket mean of episode
    returns::

        v3c:  adv = R - mean_k V(s, z_k)        # V-marginal
        v3d:  adv = R - mean(R | bucket(s))     # bucket-empirical

    where ``bucket`` defaults to the scripted opponent id (3 buckets for
    OP3/OP5/OP6 in the standard tough pool). q_phi now learns "is this z
    better than the average z WITHIN this opponent's episodes?" rather than
    "better than overall average?". This is variance reduction by
    stratification -- standard Monte Carlo technique, no V noise, no
    architecture changes.

    Plan-faithful guarantee: the bucket id is a GRADIENT-SHAPING signal
    (input to the baseline), NEVER a policy input. q_phi still sees only
    ``s`` and learns ``pi(z|s)``. The bucket only affects the variance of
    the estimator. Two episodes vs OP5 where z=2 won and z=2 lost
    contribute oppositely-signed advantages -- q_phi must still discover
    from ``s`` alone which z to pick under each context.

    Inherits everything from v3c:
      - latent_episode_strategy_n_epochs = 6
      - latent_episode_strategy_lr = 5e-3
      - latent_q_phi_marginal_baseline = True (kept on as a fallback path,
        though v3d's bucket baseline takes priority in apply_episode_strategy_ppo)
      - lamH anneal 0.003 -> 0.001 from 200k -> 700k
      - warmup=5, K=4, ctx170, episode-credit on, coef==0 main-loop gate

    Sets:
      - latent_q_phi_bucket_baseline = "opponent"
      - latent_q_phi_bucket_baseline_ema = 0.9
      - latent_q_phi_bucket_baseline_min_count = 8

    Why "opponent" only at the start (not "opponent_x_bucket")?
      "opponent" gives 3 buckets with ~1000 episodes each per rollout --
      extremely robust per-bucket means. "opponent_x_bucket" splits ~3000
      episodes across ~648 buckets (~5 episodes each), well below min_count,
      so the fallback-to-global path would dominate -- defeating the purpose.
      If v3d works but MI plateaus, the next iteration adds the bucket_id
      composite for sharper context conditioning.

    Hypothesis tested: "q_phi's gradient direction is correct (v3c showed
    movement), but the V-marginal baseline is too noisy for off-policy z to
    produce CONTEXT-CONDITIONED specialization. An empirical bucket mean
    bypasses V entirely and exposes the within-opponent cross-z signal."

    Expected first signs of working (per analyst's success criteria):
      ~50k:   [bucket-baseline] var_reduction < 1.0 (R_std visibly larger
              than adv_std; opponent stratification is removing return
              variance the marginal baseline missed)
      ~200k:  MI(z; opponent) above 0.02 (first real cross-opponent
              differentiation -- this is the metric v3c could not move)
      ~500k:  z_wr_spread > 0.15 (z choices materially affect WR per
              opponent), per-opponent z_occ visibly non-uniform per opponent
      ~700k:  MI(z; opponent) > 0.05, z_wr_spread > 0.20, WR matches or
              beats v3c's 67% baseline

    If MI(z; opponent) stays at noise floor under v3d, the bottleneck is
    no longer signal quality -- next experiment becomes either (a) larger
    z_embedding (16 -> 32 dims) for richer per-z conditioning, or (b)
    auxiliary V-calibration on off-policy z (train V(s, z) on ALL K z per
    state via the bucket-mean as a target, not just the picked one).
    """
    cfg = apply_plan_faithful_latent_v3c_router_lr(cfg)
    cfg.latent_q_phi_bucket_baseline = "opponent"
    cfg.latent_q_phi_bucket_baseline_ema = 0.9
    cfg.latent_q_phi_bucket_baseline_min_count = 8
    cfg.run_tag = "latent_v3d_smartrouter_bucketopp_ema09_min8_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3c_router_lr(cfg: PPOConfig) -> PPOConfig:
    """v3c: amplify the router update strength on top of v3b's marginal baseline.

    v3b's experimental finding: the marginal baseline successfully unblocked
    q_phi's gradient signal (``episode_credit_grad_norm`` was non-zero from
    update 1, ~0.005-0.027 per update) but cumulative logit change over a
    1M-step run was only ~10^-5 -- five orders of magnitude short of the
    ~ln(2) ≈ 0.7 needed to differentiate K=4 strategies. q_phi stayed at
    max entropy (zH_frac=1.0), MI(z; opponent) stayed at noise floor.

    Diagnosis (per implementation review): two compounding constraints on the
    router's effective step size:

      (1) ``apply_episode_strategy_ppo`` runs ONE backward step per rollout
          (vs the actor's 6-8 PPO inner epochs). That alone is a ~7x signal
          reduction.

      (2) The shared optimizer's LR (1.35e-4 for 4v4) is calibrated for the
          noisy actor gradient. q_phi's per-step gradient is clean but small
          (~0.01), and at this LR moves logits by only ~1.35e-6 per update.

    v3c lifts both constraints with config-only changes (no architecture
    surgery, no labels, no aux heads):

      - ``latent_episode_strategy_n_epochs = 6``  → q_phi gets PPO inner epochs
        like the actor. The first epoch is ratio==1 REINFORCE-style; epochs 2-6
        see the ratio drift away from 1 and the clipped PG actually does work.

      - ``latent_episode_strategy_lr = 5e-3``     → dedicated AdamW for the
        strategy_encoder + episode_strategy_value_head at ~37x the shared LR.
        Combined with (1), the effective per-update step grows ~7 × 37 = ~260x.

      - ``latent_lam_h_end = 0.001`` (vs v3b's 0.0005) → slightly higher entropy
        floor as collapse insurance. If MI growth threatens to winner-take-all
        one z (because V is poorly calibrated for the disused z slots), the
        entropy regularizer holds the distribution open. Tighten back to 0.0005
        in a follow-up if collapse doesn't materialize.

    Plan-faithful: no labels, no opponent IDs, no aux heads, no Gumbel tricks,
    no imitation. Only changes router update strength (epochs + LR) and a tiny
    entropy floor adjustment.

    Hypothesis tested: "q_phi has the correct reward-derived gradient, but the
    current number of router updates and shared LR are too small to move logits."
    Falsifiable: if zH_frac stays at 1.0 and MI stays at noise floor under
    v3c, the bottleneck is no longer training strength -- next experiment
    becomes V calibration on off-policy z slots or z_embed capacity.

    Expected first signs of working:
      ~50k:  episode-credit ratio drifts off [1.000, 1.000]; clip_fraction > 0
      ~100k: zH_frac drops below 0.99 (first measurable router movement)
      ~300k: MI(z; opponent) above 0.02 if hypothesis is right
      ~700k: zH_frac in 0.80-0.95, MI(z; opponent) > 0.05 if it sharpens
      ~1M:   WR matches v3b 67% baseline, MI > 0.05, occupancy biased not collapsed
    """
    cfg = apply_plan_faithful_latent_v3b_marginal(cfg)
    cfg.latent_episode_strategy_n_epochs = 6
    cfg.latent_episode_strategy_lr = 5e-3
    cfg.latent_lam_h_end = 0.001
    cfg.run_tag = "latent_v3c_routerlr_epochs6_lr5e3_lamHfloor1e3_1m_4v4"
    return cfg


def apply_plan_faithful_latent_strategic(cfg: PPOConfig) -> PPOConfig:
    """Plan-faithful primary latent + two anti-decorative-z fixes.

    Inherits ``apply_plan_faithful_latent`` (K=4, persistence on, entropy on,
    z_emb=16, resample_every_n=20) and flips:

    1. ``latent_q_phi_option_advantage = True`` -- q_phi learns from the
       sum of rewards across the whole z-segment (until the next resample
       or episode end), instead of only the per-step advantage at the
       resample timestep. Plan-faithful: no new labels, no aux heads --
       just correct temporal credit for the latent router under
       option-style sampling.

    2. ``latent_strategy_ppo_coef`` raised 0.30 -> 0.40 -- stronger pull
       on the actor to honor z, so the policy stops being approximately
       z-invariant. Still well within the plan's allowed range
       (phase1_coupling/phase3b use 0.45).

    Use as the back-pressure run when the primary preset shows
    non-collapsed but decorative z (high z_entropy, low z_wr_spread,
    near-noise-floor MI(z; outcome/role/spread/flag)).
    """
    cfg = apply_plan_faithful_latent(cfg)
    cfg.latent_q_phi_option_advantage = True
    cfg.latent_strategy_ppo_coef = 0.40
    cfg.run_tag = "latent_strategic_1m_2v2"
    return cfg


def apply_plan_faithful_latent_no_entropy(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_plan_faithful_latent(cfg)
    cfg.latent_entropy_objective = "none"
    cfg.latent_lam_h = 0.0
    cfg.run_tag = "latent_recommended_no_entropy_1m_2v2"
    return cfg


def apply_plan_faithful_latent_phase1_coupling(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_plan_faithful_latent_no_entropy(cfg)
    cfg.latent_resample_every_n = 10
    cfg.latent_strategy_ppo_coef = 0.45
    cfg.latent_strategy_aux_predict_phase_coef = 0.0
    cfg.run_tag = "plan_faithful_latent_phase1_coupling_hardpool_1m_2v2"
    return cfg


def apply_plan_faithful_latent_phase2_credit(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_plan_faithful_latent_phase1_coupling(cfg)
    cfg.gae_lambda = 0.97
    cfg.run_tag = "plan_faithful_latent_phase2_credit_hardpool_1m_2v2"
    return cfg


def apply_plan_faithful_latent_phase3_reward_geometry(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_plan_faithful_latent_phase1_coupling(cfg)
    cfg.env_dense_weight = 0.18
    cfg.run_tag = "plan_faithful_latent_phase3_reward_geometry_hardpool_1m_2v2"
    return cfg


def apply_plan_faithful_latent_phase3b_outcome_clean(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_plan_faithful_latent_phase1_coupling(cfg)
    cfg.latent_strategy_aux_return_head = False
    cfg.latent_strategy_aux_return_coef = 0.0
    cfg.latent_strategy_ppo_coef = 0.45
    cfg.latent_lam_p = 0.025
    cfg.latent_resample_every_n = 10
    cfg.env_dense_weight = 0.10
    cfg.env_reward_scale = 4.0
    cfg.env_reward_clip = 1.0
    cfg.env_stalemate_penalty = -0.10
    cfg.run_tag = "plan_faithful_latent_phase3b_outcome_clean_hardpool_1m_2v2"
    return cfg


def apply_plan_faithful_latent_phase3b_ablate_k1(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_plan_faithful_latent_phase3b_outcome_clean(cfg)
    cfg.latent_k = 1
    cfg.run_tag = "plan_faithful_latent_phase3b_ablate_k1_hardpool_1m_2v2"
    return cfg


def apply_plan_faithful_latent_phase3b_ablate_no_persistence(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_plan_faithful_latent_phase3b_outcome_clean(cfg)
    cfg.latent_lam_p = 0.0
    cfg.run_tag = "plan_faithful_latent_phase3b_ablate_no_persistence_hardpool_1m_2v2"
    return cfg


def apply_plan_faithful_latent_phase4a_rescue(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_plan_faithful_latent_phase3b_outcome_clean(cfg)
    cfg.latent_strategy_aux_predict_phase_coef = 0.0
    cfg.run_tag = "plan_faithful_latent_phase4a_rescue_1m_2v2"
    return cfg


def apply_plan_faithful_latent_phase4a_rescue_hardpool(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_plan_faithful_latent_phase4a_rescue(cfg)
    cfg.run_tag = "plan_faithful_latent_phase4a_rescue_hardpool_1m_2v2"
    return cfg


def apply_plan_faithful_latent_episode_z_clean(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_plan_faithful_base(cfg)
    cfg.use_latent_strategy = True
    cfg.latent_k = 4
    cfg.latent_resample_every_n = 0
    cfg.latent_lam_p = 0.0
    cfg.latent_lam_h = 0.001
    cfg.latent_strategy_aux_predict_phase_coef = 0.0
    cfg.latent_strategy_aux_return_head = False
    cfg.latent_strategy_aux_return_coef = 0.0
    cfg.run_tag = "plan_faithful_latent_episode_z_clean_1m_2v2"
    return cfg


def apply_plan_faithful_latent_option_a_episode_credit(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_plan_faithful_latent_option_a(cfg)
    cfg.latent_strategy_ppo_coef = 0.0
    cfg.latent_episode_strategy_ppo = True
    cfg.latent_episode_strategy_coef = 0.25
    cfg.latent_episode_strategy_clip_eps = 0.2
    cfg.latent_episode_strategy_value_coef = 0.5
    cfg.latent_episode_strategy_return_norm = True
    cfg.run_tag = "plan_faithful_latent_option_a_episode_credit_1m_2v2"
    return cfg


def apply_plan_faithful_latent_option_a(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_plan_faithful_base(cfg)
    cfg.use_latent_strategy = True
    cfg.latent_k = 4
    cfg.latent_resample_every_n = 0
    cfg.latent_lam_p = 0.0
    cfg.latent_lam_h = 0.001
    cfg.latent_entropy_objective = "maximize"
    cfg.latent_strategy_aux_predict_phase_coef = 0.0
    cfg.latent_strategy_aux_return_head = False
    cfg.latent_strategy_aux_return_coef = 0.0
    cfg.latent_resample_on_flag = False
    cfg.latent_kl_consecutive = 0.0
    cfg.fixed_latent_strategy = False
    cfg.run_tag = "plan_faithful_latent_option_a_1m_2v2"
    return cfg


def apply_plan_faithful_latent_k1(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_plan_faithful_latent(cfg)
    cfg.latent_k = 1
    cfg.run_tag = "latent_recommended_collapsed_k1_1m_2v2"
    return cfg


def apply_plan_faithful_no_latent(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_plan_faithful_base(cfg)
    cfg.use_latent_strategy = False
    cfg.fixed_latent_strategy = False
    cfg.run_tag = "no_latent_baseline_1m_2v2"
    return cfg


def apply_plan_option_a(cfg: PPOConfig) -> PPOConfig:
    return apply_latent_a1_plan_faithful(cfg)


def apply_plan_option_b_lamp(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_latent_a1_plan_faithful(cfg)
    cfg.latent_resample_every_n = 20
    cfg.latent_lam_p = 0.02
    cfg.run_tag = "plan_option_b_lamp_1m_2v2"
    return cfg


def apply_latent_a1_plan_faithful(cfg: PPOConfig) -> PPOConfig:
    """Plan-faithful A1 recipe (used as the base for option-A/option-B variants)."""
    cfg.use_latent_strategy = True
    cfg.total_timesteps = 1_000_000
    cfg.mode = TrainMode.FIXED_OPPONENT.value
    cfg.fixed_opponent_tag = "OP3"
    cfg.normalize_returns = True
    cfg.clip_range = 0.18
    cfg.clip_range_vf = 0.2
    cfg.vf_coef = 1.1
    cfg.learning_rate = 1.8e-4
    cfg.lr_floor_frac = 0.05
    cfg.target_kl = 0.02
    cfg.n_steps = 2048
    cfg.batch_size = 512
    cfg.n_epochs = 8
    cfg.ent_coef = 0.0015
    cfg.latent_entropy_objective = "maximize"
    cfg.latent_lam_h = 0.005
    cfg.latent_lam_p = 0.02
    cfg.latent_strategy_ppo_coef = 0.30
    cfg.latent_strategy_aux_return_head = False
    cfg.latent_strategy_aux_return_coef = 0.0
    cfg.latent_strategy_tau = 1.0
    cfg.latent_resample_every_n = 0
    cfg.latent_resample_on_flag = False
    cfg.latent_kl_consecutive = 0.0
    cfg.latent_gae_reset_on_z_change = True
    cfg.latent_bootstrap_z_deterministic = True
    cfg.latent_vf_hidden = 128
    cfg.env_win_team_reward = None
    cfg.env_draw_team_penalty = None
    cfg.env_lose_team_punish = None
    cfg.env_action_failed_punishment = None
    cfg.env_dense_weight = None
    cfg.env_sparse_weight = None
    cfg.env_reward_scale = None
    cfg.env_reward_clip = None
    cfg.env_stalemate_penalty = None
    cfg.env_stalemate_max_steps = None
    cfg.reward_shaping_coef_start = 1.0
    cfg.reward_shaping_coef_end = 1.0
    cfg.reward_shaping_decay_steps = 0
    cfg.periodic_checkpoint_steps = 50_000
    cfg.load_path = None
    cfg.run_tag = "latent_a1_plan_faithful_1m_2v2"
    return cfg
