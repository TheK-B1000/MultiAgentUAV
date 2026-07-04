"""v6i7+ router adapter presets — recurrent router through v6i9 nav refinement."""
from __future__ import annotations

from rl.config.ppo_config import PPOConfig, TrainMode


def apply_plan_faithful_latent_v6i7_recurrent_router(cfg: PPOConfig) -> PPOConfig:
    """V6I7-A smoke test: GRU router with BPTT, EMA context disabled.

    Do NOT load from a V6I6 checkpoint — the q_phi input layer dimension
    changed (CONTEXT_STATE_DIM=170 → 34+64=98), making weights incompatible.
    Start fresh or warm-start from a compatible actor-only checkpoint.
    """
    from rl.config_presets import v6i7_recurrent_router_config
    preset = v6i7_recurrent_router_config()

    cfg.use_latent_strategy = preset.use_latent_strategy
    cfg.latent_k = preset.latent_k
    cfg.router_context_mode = preset.router_context_mode
    cfg.recurrent_selector_hidden_dim = preset.recurrent_selector_hidden_dim
    cfg.recurrent_seq_len = preset.recurrent_seq_len
    cfg.recurrent_burn_in = preset.recurrent_burn_in
    cfg.router_chunks_per_batch = preset.router_chunks_per_batch
    cfg.router_ent_coef = preset.router_ent_coef
    cfg.latent_resample_every_n = preset.latent_resample_every_n
    cfg.strategy_interval = preset.strategy_interval
    cfg.latent_resample_on_flag = preset.latent_resample_on_flag
    cfg.latent_event_refresh_enabled = preset.latent_event_refresh_enabled
    cfg.latent_sparse_tactical_refresh_enabled = preset.latent_sparse_tactical_refresh_enabled
    cfg.latent_strategy_ppo_coef = preset.latent_strategy_ppo_coef
    cfg.latent_lam_p = preset.latent_lam_p
    cfg.latent_lam_h = preset.latent_lam_h
    cfg.latent_entropy_objective = preset.latent_entropy_objective
    cfg.h_mode = preset.h_mode
    cfg.latent_strategy_aux_return_head = preset.latent_strategy_aux_return_head
    cfg.latent_strategy_aux_return_coef = preset.latent_strategy_aux_return_coef
    cfg.latent_strategy_aux_predict_phase_coef = preset.latent_strategy_aux_predict_phase_coef
    cfg.latent_cf_separation_coef = preset.latent_cf_separation_coef
    cfg.latent_kl_consecutive = preset.latent_kl_consecutive
    cfg.fixed_latent_strategy = preset.fixed_latent_strategy

    cfg.experiment_id = "v6i7"
    cfg.experiment_family = "v6"
    cfg.total_timesteps = 1_000_000
    cfg.mode = TrainMode.OPPONENT_POOL.value
    cfg.opponent_pool = ["OP5", "OP6", "OP7"]
    cfg.normalize_returns = True
    cfg.clip_range = 0.18
    cfg.clip_range_vf = 0.2
    cfg.periodic_checkpoint_steps = 50_000
    cfg.run_tag = "v6i7_recurrent_router_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v6i7_sparse_router(cfg: PPOConfig) -> PPOConfig:
    """V6I7 with separate sparse router reward (team wins + flag events only)."""
    from rl.config_presets import v6i7_sparse_router_config
    cfg = apply_plan_faithful_latent_v6i7_recurrent_router(cfg)
    preset = v6i7_sparse_router_config()
    cfg.router_reward_enabled = preset.router_reward_enabled
    cfg.router_reward_win_weight = preset.router_reward_win_weight
    cfg.router_reward_flag_cap_weight = preset.router_reward_flag_cap_weight
    cfg.router_reward_sparse_weight = preset.router_reward_sparse_weight
    cfg.router_reward_scale = preset.router_reward_scale
    cfg.router_reward_normalize = preset.router_reward_normalize
    cfg.experiment_id = "v6i7_sparse"
    cfg.run_tag = "v6i7_sparse_router_OP5_OP6_OP7_1m"
    return cfg


def apply_plan_faithful_latent_v6i7_repertoire_balanced_episode(cfg: PPOConfig) -> PPOConfig:
    """V6I7 with balanced-episode forced-latent repertoire training."""
    from rl.config_presets import v6i7_repertoire_balanced_episode_config
    cfg = apply_plan_faithful_latent_v6i7_recurrent_router(cfg)
    preset = v6i7_repertoire_balanced_episode_config()
    cfg.latent_assignment_mode = preset.latent_assignment_mode
    cfg.train_router_when_forced = preset.train_router_when_forced
    cfg.train_router_critic_when_forced = preset.train_router_critic_when_forced
    cfg.experiment_id = "v6i7_balanced_ep"
    cfg.run_tag = "v6i7_balanced_episode_OP5_OP6_OP7_1m"
    return cfg


def apply_plan_faithful_latent_v6i7_router_critic_warmup(cfg: PPOConfig) -> PPOConfig:
    """V6I7 two-phase: sparse router reward + balanced-episode coverage warmup."""
    from rl.config_presets import v6i7_router_critic_warmup_config
    cfg = apply_plan_faithful_latent_v6i7_recurrent_router(cfg)
    preset = v6i7_router_critic_warmup_config()
    cfg.router_reward_enabled = preset.router_reward_enabled
    cfg.router_reward_win_weight = preset.router_reward_win_weight
    cfg.router_reward_flag_cap_weight = preset.router_reward_flag_cap_weight
    cfg.router_reward_sparse_weight = preset.router_reward_sparse_weight
    cfg.router_reward_scale = preset.router_reward_scale
    cfg.router_reward_normalize = preset.router_reward_normalize
    cfg.latent_assignment_mode = preset.latent_assignment_mode
    cfg.train_router_when_forced = preset.train_router_when_forced
    cfg.train_router_critic_when_forced = preset.train_router_critic_when_forced
    cfg.experiment_id = "v6i7_warmup"
    cfg.run_tag = "v6i7_router_critic_warmup_OP5_OP6_OP7_1m"
    return cfg


def apply_plan_faithful_latent_v6i8_adapter_balanced(cfg: PPOConfig) -> PPOConfig:
    """V6I8-balanced: GRU router + residual adapters + balanced-episode warmup.

    Extends V6I7 with per-latent actor adapters h_z=h+g_z*A_z(h) and logit
    biases B_z.  Adapters zero-initialized so V6I7 checkpoints load with exact
    behavioral equivalence.  Router held out of PPO during forced episodes;
    router critic trained throughout for cold-start readiness.

    Compare directly against ``v6i7_warmup`` at identical step budget to
    isolate the effect of stronger latent actor conditioning.
    """
    from rl.config_presets import v6i8_adapter_balanced_config
    cfg = apply_plan_faithful_latent_v6i7_recurrent_router(cfg)
    preset = v6i8_adapter_balanced_config()
    cfg.enable_latent_z_residual = preset.enable_latent_z_residual
    cfg.latent_z_gate_init = preset.latent_z_gate_init
    cfg.latent_assignment_mode = preset.latent_assignment_mode
    cfg.train_router_when_forced = preset.train_router_when_forced
    cfg.train_router_critic_when_forced = preset.train_router_critic_when_forced
    cfg.experiment_id = "v6i8_balanced"
    cfg.run_tag = "v6i8_adapter_balanced_OP5_OP6_OP7_1m"
    return cfg


def apply_plan_faithful_latent_v6i8_adapter_sparse(cfg: PPOConfig) -> PPOConfig:
    """V6I8-sparse: GRU router + residual adapters + sparse router reward.

    Use after adapter differentiation is confirmed with v6i8_adapter_balanced.
    """
    from rl.config_presets import v6i8_adapter_sparse_config
    cfg = apply_plan_faithful_latent_v6i8_adapter_balanced(cfg)
    preset = v6i8_adapter_sparse_config()
    cfg.router_reward_enabled = preset.router_reward_enabled
    cfg.router_reward_win_weight = preset.router_reward_win_weight
    cfg.router_reward_flag_cap_weight = preset.router_reward_flag_cap_weight
    cfg.router_reward_sparse_weight = preset.router_reward_sparse_weight
    cfg.router_reward_scale = preset.router_reward_scale
    cfg.router_reward_normalize = preset.router_reward_normalize
    cfg.latent_assignment_mode = "router"
    cfg.train_router_when_forced = True
    cfg.train_router_critic_when_forced = True
    cfg.experiment_id = "v6i8_sparse"
    cfg.run_tag = "v6i8_adapter_sparse_OP5_OP6_OP7_1m"
    return cfg


def apply_plan_faithful_latent_v6i8_adapter_balanced_hardpool(cfg: PPOConfig) -> PPOConfig:
    """V6I8-balanced + hard opponent pool (OP8/OP9/OP10).

    Inherits V6I8-balanced exactly; only the opponent pool changes.
    OP8 (coordinated interceptor), OP9 (fortress + counterattack), and
    OP10 (active escort carrier) each require a distinct counter-strategy,
    giving the latent router a real optimisation objective.
    """
    from rl.config_presets import v6i8_adapter_balanced_hardpool_config
    cfg = apply_plan_faithful_latent_v6i8_adapter_balanced(cfg)
    preset = v6i8_adapter_balanced_hardpool_config()
    cfg.opponent_pool = preset.opponent_pool
    cfg.opponent_pool_weights = preset.opponent_pool_weights
    cfg.experiment_id = "v6i8_balanced_hardpool"
    cfg.run_tag = "v6i8_adapter_balanced_hardpool_OP8_OP9_OP10_1m"
    return cfg


def apply_plan_faithful_latent_v6i8_adapter_sparse_hardpool(cfg: PPOConfig) -> PPOConfig:
    """V6I8-sparse + hard opponent pool (OP8/OP9/OP10).

    Inherits V6I8-sparse exactly; only the opponent pool changes.
    """
    from rl.config_presets import v6i8_adapter_sparse_hardpool_config
    cfg = apply_plan_faithful_latent_v6i8_adapter_sparse(cfg)
    preset = v6i8_adapter_sparse_hardpool_config()
    cfg.opponent_pool = preset.opponent_pool
    cfg.opponent_pool_weights = preset.opponent_pool_weights
    cfg.experiment_id = "v6i8_sparse_hardpool"
    cfg.run_tag = "v6i8_adapter_sparse_hardpool_OP8_OP9_OP10_1m"
    return cfg


def apply_plan_faithful_latent_v6i9_mapaware_generalist_hardpool(cfg: PPOConfig) -> PPOConfig:
    """V6I9 Stage 1 — map-aware generalist competence on map_b with OP8/9/10.

    Warm-started from V6I8 (750k checkpoint recommended).  The checkpoint
    loader automatically expands the first CNN conv from 7→8 channels: channels
    0-6 are copied verbatim, channel 7 (obstacle) is zero-initialized so the
    policy is behaviourally unchanged at init.

    What is trained: obstacle-aware CNN, shared actor body, shared action head,
    critic.  What is disabled: latent adapters, latent action biases, router.

    Run for 200k steps then check the promotion gate with:
        python experiments/gate_v6i9_map_aware.py --checkpoint <ckpt>
    """
    from rl.config_presets import v6i9_map_aware_config
    cfg = apply_plan_faithful_latent_v6i8_adapter_balanced_hardpool(cfg)
    preset = v6i9_map_aware_config()
    cfg.map_layout = preset.map_layout                              # map_b
    cfg.enable_latent_z_residual = preset.enable_latent_z_residual  # False
    cfg.latent_assignment_mode = preset.latent_assignment_mode      # balanced_episode
    cfg.train_router_when_forced = preset.train_router_when_forced  # False
    cfg.train_router_critic_when_forced = False                     # held out entirely
    cfg.v6i9_training_stage = "generalist"
    cfg.experiment_id = "v6i9"
    cfg.run_tag = "v6i9_mapaware_generalist_hardpool_OP8_OP9_OP10_200k"
    return cfg


def apply_plan_faithful_latent_v6i9_mapaware_generalist_hardpool_split(cfg: PPOConfig) -> PPOConfig:
    """V6I9 Stage 1 variant — same as generalist but on map_b_split_lane_v2.

    Run concurrently with the map_b variant to ensure route diversity before
    Stage 2.  The gate check compares actor logits across both map geometries.
    """
    from rl.config_presets import v6i9_map_aware_split_lane_config
    cfg = apply_plan_faithful_latent_v6i9_mapaware_generalist_hardpool(cfg)
    preset = v6i9_map_aware_split_lane_config()
    cfg.map_layout = preset.map_layout  # map_b_split_lane_v2
    cfg.run_tag = "v6i9_mapaware_generalist_hardpool_split_OP8_OP9_OP10_200k"
    return cfg


def apply_plan_faithful_latent_v6i9_mapaware_repertoire_hardpool(cfg: PPOConfig) -> PPOConfig:
    """V6I9 Stage 2 — TALENTS-inspired repertoire birth.

    Prerequisite: Stage 1 gate must pass (obstacle weights nonzero, map
    sensitivity confirmed, WR in [50%, 90%]).

    What is frozen: CNN, shared actor trunk, shared action head, router.
    What is trained: z-specific adapters, gates, action biases, z-embeddings,
    z-conditioned critic.

    One persistent latent z per episode, balanced round-robin across z0..z3.
    The JSD separation term (latent_cf_separation_coef) fires once all latents
    are competent and provides behavioral diversity pressure without supervised
    strategy labels.
    """
    cfg = apply_plan_faithful_latent_v6i9_mapaware_generalist_hardpool(cfg)
    cfg.enable_latent_z_residual = True          # adapters active
    cfg.latent_z_gate_init = 0.01
    cfg.latent_assignment_mode = "balanced_episode"
    cfg.train_router_when_forced = False
    cfg.train_router_critic_when_forced = False
    cfg.latent_cf_separation_coef = 0.005        # JSD diversity pressure; starts soft
    cfg.v6i9_training_stage = "repertoire"       # freeze shared trunk in optimizer build
    cfg.experiment_id = "v6i9"
    cfg.run_tag = "v6i9_mapaware_repertoire_hardpool_OP8_OP9_OP10"
    return cfg


def apply_plan_faithful_latent_v6i9_mapaware_router_sparse_hardpool(cfg: PPOConfig) -> PPOConfig:
    """V6I9 Stage 3 — RILI-inspired recurrent router training.

    Prerequisite: Stage C forced-z causal validation must pass
    (oracle_wr > best_fixed_wr AND best-z varies across map-opponent cells).

    What is frozen: CNN, shared actor, z-embeddings, adapters, gates, action
    biases.  What is trained: recurrent q_phi GRU router, router critic.

    Sparse task-consequence reward only: flag captures, returns, tag events,
    score changes, episode outcome.  The router is NOT told which opponent it
    faces or what a latent "means" — routing emerges from interaction history.
    """
    cfg = apply_plan_faithful_latent_v6i9_mapaware_repertoire_hardpool(cfg)
    # Freeze actor + z-specific modules via stage flag; router_freeze_actor adds
    # the actor-level freeze on top (belt-and-suspenders for the shared trunk).
    cfg.router_freeze_actor = True
    cfg.v6i9_training_stage = "router"
    # Enable sparse router reward (captures, returns, score, outcome).
    cfg.router_reward_enabled = True
    cfg.router_reward_win_weight = 1.0
    cfg.router_reward_flag_cap_weight = 0.5
    cfg.router_reward_sparse_weight = 0.2
    cfg.router_reward_scale = 1.0
    cfg.router_reward_normalize = True
    # Let the router learn freely; forced-z only for early warm-up, then switch to router mode.
    cfg.latent_assignment_mode = "router"
    cfg.train_router_when_forced = True
    cfg.train_router_critic_when_forced = True
    cfg.latent_cf_separation_coef = 0.0  # repertoire is frozen; no diversity gradient needed
    cfg.experiment_id = "v6i9"
    cfg.run_tag = "v6i9_mapaware_router_sparse_hardpool_OP8_OP9_OP10"
    return cfg


def apply_plan_faithful_latent_v6i9_arc_credit_running_mean_hardpool(cfg: PPOConfig) -> PPOConfig:
    """V6I9 treatment: arc-credit with running_mean baseline, BPTT PPO disabled.

    A/B counterpart to ``v6i9_mapaware_router_sparse_hardpool`` (control).
    Identical in every way EXCEPT the router credit channel:

    Control:
        latent_strategy_ppo_coef=0.10  (actor-GAE BPTT PPO active)
        latent_arc_credit_enabled=False

    Treatment (this preset):
        latent_strategy_ppo_coef=0.0   (actor-GAE BPTT PPO disabled)
        latent_arc_credit_enabled=True
        latent_arc_credit_baseline="running_mean"

    Motivation from credit audit (2026-07-02):
    - Actor-GAE critic overestimates V(s_0) by +2.71 units on average.
    - RouterSequenceUpdater normalization is per-chunk over 1-2 decisions;
      for single-decision chunks (>50% of all chunks with strategy_interval=32)
      normalization is disabled entirely by the numel>1 guard.
    - The +2.705 offset therefore survives into the PPO loss for most updates,
      producing chronically negative advantages and 41% sign flips.
    - Arc credit with running_mean baseline auto-centers advantages (EMA tracks
      actual game returns), removing the absolute bias without changing any
      other hyperparameter.

    BPTT path still contributes entropy and persistence regularization
    (router_ent_coef * ent_loss + latent_lam_p * persist_loss); only the
    PPO term (router_ppo_coef * ppo_loss) is zeroed.
    """
    cfg = apply_plan_faithful_latent_v6i9_mapaware_router_sparse_hardpool(cfg)
    # Disable actor-GAE PPO from BPTT (the source of the biased credit).
    cfg.latent_strategy_ppo_coef = 0.0
    # Enable arc-level consequence credit with auto-centering baseline.
    cfg.latent_arc_credit_enabled = True
    cfg.latent_arc_credit_baseline = "running_mean"
    cfg.latent_arc_credit_coef = 1.0   # default; full arc-credit gradient weight
    cfg.latent_arc_credit_min_len = 8   # accept shorter terminal arcs (vs default 32)
    cfg.run_tag = "v6i9_arc_credit_running_mean_hardpool_OP8_OP9_OP10"
    return cfg


def apply_plan_faithful_latent_v6i9_arc_credit_specialize_hardpool(cfg: PPOConfig) -> PPOConfig:
    """V6I9 specialization treatment: arc-credit + reduced conditional entropy + marginal coverage.

    Builds on ``v6i9_arc_credit_running_mean_hardpool`` (arc credit fixed, BPTT PPO
    disabled) and adds two entropy-balance changes to break near-tie argmax collapse:

    1. ``router_ent_coef`` reduced from 0.005 -> 0.001
       Weaker conditional entropy pressure allows the router to develop
       context-specific preferences rather than staying near-uniform everywhere.

    2. ``latent_lam_h`` enabled at 0.01 (marginal coverage)
       KL(q_bar || Uniform) penalty ensures all four z values remain globally
       used, preventing the router from dropping entire latents even as it
       becomes more confident within individual contexts.

    Diagnosis motivating this preset (2026-07-03):
    - Training softmax entropy: 1.374 / 1.386 max -- near-uniform at all times.
    - Eval argmax collapses to z=3 (85%) + z=1 (15%); z=0 and z=2 never selected.
    - Cross-episode shuffle gate vacuous: can_reassign=False because every
      episode draws from the same two-latent universe.
    - Root cause: router_ent_coef=0.005 keeps H(z|context) near maximum,
      preventing per-context confidence. latent_lam_h=0.0 means no marginal
      coverage gradient.
    """
    cfg = apply_plan_faithful_latent_v6i9_arc_credit_running_mean_hardpool(cfg)
    cfg.router_ent_coef = 0.001
    cfg.latent_lam_h = 0.01
    # ``latent_entropy_mode`` is the field the runtime entropy path
    # (rl/custom_ppo/update/entropy_objectives.py) and the audit banner
    # actually read; ``h_mode`` is a legacy alias with no consumer in the
    # entropy path. Setting only ``h_mode`` leaves the marginal-coverage loss
    # OFF and turns ``latent_lam_h`` into a CONDITIONAL entropy-maximization
    # term (pushes q_phi toward uniform per context) — the exact opposite of
    # this preset's intent. Set the runtime field. The rollout-level marginal
    # aggregation contract (AGENTS.md §"Aggregation contract") is honored
    # because ``latent_entropy_mode == "marginal"`` routes through
    # ``rollout_marginal_entropy_loss``.
    cfg.latent_entropy_mode = "marginal"
    cfg.h_mode = "marginal"
    # The arc-credit parent zeroes ``latent_entropy_objective`` (it disables the
    # main-loop strategy PPO term). The marginal-coverage path additionally
    # requires an active objective (``h_goal != "none"``), so re-enable it to
    # MAXIMIZE marginal entropy H(q_bar) — i.e. keep all four latents globally
    # used — matching the canonical marginal preset v5i6.
    cfg.latent_entropy_objective = "maximize"
    cfg.run_tag = "v6i9_arc_credit_specialize_hardpool_OP8_OP9_OP10"
    return cfg


def apply_plan_faithful_latent_v6i9_arc_credit_running_mean_feedforward_hardpool(
    cfg: PPOConfig,
) -> PPOConfig:
    """V6I9 treatment: feedforward router + running-mean arc credit (A/B vs feedforward control).

    Direct A/B counterpart to ``v6i9_mapaware_router_feedforward_hardpool``
    (the control).  The router architecture, 35-dim context, strategy_interval,
    learning rate, entropy coefficient, opponent/map pool, frozen actor +
    z-specific parameters, seed, and training budget are held IDENTICAL to the
    control.  The ONLY resolved-config deltas (verified by
    ``tests/test_v6i9_arc_credit_feedforward.py``) are:

        latent_arc_credit_enabled  : False -> True
        latent_arc_credit_baseline : context_value -> running_mean
        latent_strategy_ppo_coef   : 0.1   -> 0.0
        run_tag                    : ...   -> arc-credit tag

    Scientific rationale (credit audit, 2026-07-02)
    ----------------------------------------------
    The control routes q_phi credit through the main-loop strategy PPO term
    scaled by ``latent_strategy_ppo_coef`` using ``router_advantages`` =
    ``router_return - V_critic``.  The critic overestimates V(s_0) by ~+2.71
    units, and because most single-decision chunks skip the per-chunk
    advantage normalization (the ``numel > 1`` guard), that constant +2.705
    bias survives into the PPO loss, producing chronically negative advantages
    and ~41% sign flips.

    This treatment REPLACES that channel: it zeroes ``latent_strategy_ppo_coef``
    (removing the biased critic advantage) and enables arc-level consequence
    credit with a detached running-mean baseline (an EMA over completed arc
    returns, no V dependency).  The EMA auto-centers advantages, removing the
    absolute bias without touching any architectural hyperparameter.

    ``latent_arc_credit_min_len`` is left at the control default (32) so only
    full strategy-interval arcs contribute to the PPO batch.  The BPTT/main
    entropy and persistence regularizers are unaffected (only the PPO term is
    zeroed); q_phi's learning signal now flows exclusively through
    ``apply_arc_strategy_ppo``.
    """
    cfg = apply_plan_faithful_latent_v6i9_mapaware_router_feedforward_hardpool(cfg)
    # Remove the biased critic-based router advantage (the "magnet").
    cfg.latent_strategy_ppo_coef = 0.0
    # Enable arc-level consequence credit with an auto-centering EMA baseline.
    cfg.latent_arc_credit_enabled = True
    cfg.latent_arc_credit_baseline = "running_mean"
    cfg.latent_arc_credit_coef = 1.0  # control default; explicit for clarity
    cfg.run_tag = "v6i9_arc_credit_running_mean_feedforward_hardpool_OP8_OP9_OP10"
    return cfg


def apply_plan_faithful_latent_v6i10_episode_router_explore_hardpool(
    cfg: PPOConfig,
) -> PPOConfig:
    """V6I10: feedforward episode router over the frozen v6i9 repertoire.

    Proposed Preset Review
    ----------------------
    Proposed name: v6i10_episode_router_explore_hardpool.
    Parent preset: v6i9_mapaware_router_feedforward_hardpool.
    Classification: SUMMER-COMPATIBLE EXTENSION.
    Research question: can a one-decision-per-episode feedforward router learn
    useful dispatch over the validated frozen repertoire before adding history
    encoders, dynamic switching, or opponent-response models?

    Delta table vs parent:
        Actor conditioning: residual adapters remain frozen, intended change no.
        Router task PPO: on -> off, intended change yes.
        Episode credit: running-mean arc credit on one episode-long arc, intended
        change yes.
        Arc credit: off -> running_mean, intended change yes.
        Forced-z schedule: off, intended change no.
        Persistence: 0.02 -> 0.0, intended change yes.
        Entropy: conditional -> marginal coverage, intended change yes.
        Resampling: interval 32 -> episode-start only, intended change yes.
        Exploration: 0.0 -> 0.20 behavior mixture, intended change yes.

    This is not a paper-faithful row: it changes cadence, credit aggregation,
    entropy strength, and behavior-policy exploration. It remains label-free:
    no opponent IDs, oracle z labels, supervised best-z targets, aux heads, or
    forced-z curriculum are added.
    """
    cfg = apply_plan_faithful_latent_v6i9_mapaware_router_feedforward_hardpool(cfg)
    cfg.latent_resample_every_n = 0
    cfg.strategy_interval = 0
    cfg.latent_lam_p = 0.0
    cfg.latent_strategy_ppo_coef = 0.0
    cfg.latent_arc_credit_enabled = True
    cfg.latent_arc_credit_baseline = "running_mean"
    cfg.latent_arc_credit_coef = 1.0
    cfg.latent_arc_credit_min_len = 1
    cfg.learning_rate = 1e-4
    cfg.router_ent_coef = 0.002
    cfg.router_uniform_exploration_prob = 0.20
    cfg.latent_lam_h = 0.015
    cfg.latent_lam_h_end = 0.015
    cfg.latent_entropy_anneal_start = 0
    cfg.latent_entropy_anneal_end = 0
    cfg.latent_entropy_mode = "marginal"
    cfg.h_mode = "marginal"
    cfg.latent_entropy_objective = "maximize"
    cfg.experiment_id = "v6i10"
    cfg.run_tag = "v6i10_episode_router_explore_hardpool_OP8_OP9_OP10"
    return cfg


def apply_plan_faithful_latent_v6i9_mapaware_router_feedforward_hardpool(cfg: PPOConfig) -> PPOConfig:
    """V6I9 Stage 3 feedforward — state-only MLP router over frozen repertoire.

    Identical to ``v6i9_mapaware_router_sparse_hardpool`` except recurrence is
    disabled (V6I7-B0 pattern).  Tests whether observable current state is
    sufficient to capture oracle complementarity before trying GRU/RILI.
    """
    cfg = apply_plan_faithful_latent_v6i9_mapaware_router_sparse_hardpool(cfg)
    cfg.recurrent_selector_hidden_dim = 0
    cfg.recurrent_seq_len = 0
    cfg.recurrent_burn_in = 0
    cfg.router_chunks_per_batch = 0
    cfg.router_reinitialize_on_load = True
    cfg.run_tag = "v6i9_mapaware_router_feedforward_hardpool_OP8_OP9_OP10"
    return cfg


def apply_plan_faithful_latent_v6i9_mapaware_nav_refinement(cfg: PPOConfig) -> PPOConfig:
    """V6I9.1 — navigation refinement fine-tune from the 1M generalist checkpoint.

    Motivation
    ----------
    Stage A evaluation (100 ep/cell, map_b_split_lane, OP8/OP9/OP10) revealed:

    * Stuck steps INCREASED vs baseline in every split-lane cell, worst for OP9
      (0.77 vs 0.49, +57 %).  Map A (open) showed zero regression — the failure
      is split-lane-specific.
    * OP8 win rate dropped −6 pp in split-lane (0.89 vs 0.95 baseline).
    * Route-lane fractions are statistically identical to baseline — the obstacle
      channel is not yet driving route selection.
    * Gradient and counterfactual gates could not run (get_distribution() not
      exposed); obstacle weights moved (gate PASS) but behavioral impact is weak.

    Fix
    ---
    Fine-tune the 1M Stage A checkpoint for 200k–400k additional steps with:

    1. Training map: map_b_split_lane — the exact geometry where failures occurred.
       Wall-geometry variation via map_b_vertical_mirror_prob=0.5 so the policy
       generalises across both lane configurations.

    2. Opponent pool: OP8 / OP9 / OP10, equal weight.  OP9 is the largest
       regression; keeping all three prevents overfit to a single opponent.

    3. Blocked-movement penalty: env_action_failed_punishment = −0.01 (small).
       The existing telemetry tracks repeated_blocked_movement per episode; this
       coefficient penalises each invalid-move event so the actor learns to prefer
       valid directions near walls instead of repeating blocked ones.

    4. Reduced learning rate: 5e-5 (one-third of the Stage A default).  This is
       a fine-tune — large gradients would destabilise the competent trunk.

    5. Geometry randomization: map_b_vertical_mirror_prob=0.5 (already in the
       GPU env) ensures the policy is exposed to both lane polarities.

    Note on multi-map curriculum
    ----------------------------
    The user prescription calls for 60–70 % obstacle-rich / 30–40 % open maps.
    The current training system passes a single map_layout per run; there is no
    built-in map-pool sampling.  This preset uses split-lane exclusively because
    (a) that is where failures occurred, and (b) map-pool support requires adding
    a map_pool field and updating env_factory to sample per episode.  Add
    map-pool training as a follow-up if route differentiation remains absent after
    this refinement.
    """
    cfg = apply_plan_faithful_latent_v6i9_mapaware_generalist_hardpool(cfg)
    # Switch to the split-lane map — where stuck/OP8-WR regressions occurred.
    cfg.map_layout = "map_b_split_lane"
    # Geometry variation: mirror the wall axis so both lane polarities are seen.
    cfg.map_b_vertical_mirror_prob = 0.5
    # Blocked-movement consequence signal: small per-action penalty for hitting a wall.
    cfg.env_action_failed_punishment = -0.01
    # Fine-tune LR — trunk is already competent; avoid destabilising it.
    cfg.learning_rate = 5e-5
    cfg.experiment_id = "v6i9"
    cfg.run_tag = "v6i9_mapaware_nav_refinement_splitlane_OP8_OP9_OP10_200k"
    return cfg


def apply_plan_faithful_latent_v6i11_q_router_hardpool(cfg: PPOConfig) -> PPOConfig:
    """V6I11 — contextual Q-value return router over the frozen v6i9 repertoire.

    Proposed Preset Review
    ----------------------
    Proposed name: v6i11_q_router_hardpool.
    Parent preset: v6i10_episode_router_explore_hardpool (episode-persistent,
    feedforward, arc == episode, validated one-decision-per-episode contract).
    Classification: SUMMER-COMPATIBLE EXTENSION (context enrichment + off-policy
    value regression; label-free — targets are online experienced returns).

    Research question: can a SEPARATE return-prediction model learn
    ``context + selected z -> expected EPISODE return`` from online exploratory
    data, cleanly separating "estimate which latent has higher value" (Q-router)
    from "execute the selected latent" (frozen actor)?  This side-steps the
    policy-gradient credit problem that repeatedly turned tiny logit biases into
    one-latent argmax collapse (v6i9/v6i10).

    Target-horizon contract (2026-07-03 correction)
    ------------------------------------------------
    The validated complementarity (Probe A, forced-z oracle, +2.37 oracle gap)
    is defined on EPISODE-PERSISTENT forced-z EPISODE return: "which z is best
    for the whole episode?".  An earlier draft of this preset inherited the
    cadence-32 recurrent lineage (``strategy_interval=32``,
    ``latent_resample_every_n=32``, ``latent_arc_credit_min_len=32``), which
    made each arc a ~32-step MID-EPISODE segment and changed the learning target
    to "which z produced the best LOCAL arc return?".  Those two targets need
    not agree (a latent may pay a short-term cost for a better eventual capture),
    so the diagnostic would not have measured the validated task.

    This preset therefore inherits v6i10's episode-persistent contract so that:
        * exactly one routing decision per episode (resample only at episode start),
        * ``global_state_0`` in each arc record IS the episode-start context,
        * ``arc_return`` IS the total episode return (min_len == 1, one arc/episode).
    That matches Probe A and the forced-z oracle exactly.

    Key changes vs v6i10_episode_router_explore_hardpool
    ----------------------------------------------------
    * ``latent_arc_credit_coef = 0.0``     — the internal router is NOT updated
                                              from arc records; arc records are
                                              collected purely as data for the
                                              EXTERNAL Q-regressor.
    * ``router_ent_coef = 0.0``            — no entropy pressure on the router.
    * ``latent_lam_h = 0.0`` / ``latent_lam_h_end = 0.0`` — no marginal-entropy
                                              pressure (v6i10 used 0.015).
    * ``latent_strategy_ppo_coef = 0.0``   — BPTT PPO disabled (already 0).
    * ``router_uniform_exploration_prob = 0.5`` — 50 % uniform z / 50 % router
                                              argmax so every (opponent, z) cell
                                              gets samples.
    * ``latent_arc_credit_enabled = True`` — arc records still collected.
    Inherited unchanged from v6i10: episode-persistent resampling
    (``strategy_interval=0``, ``latent_resample_every_n=0``), feedforward router,
    ``latent_arc_credit_min_len=1`` (arc == episode), frozen actor + adapters.

    The Q-regressor is EXTERNAL to the PPO trainer: it is instantiated in the
    experiment script (``experiments/run_v6i11_q_router.py``) and trained from
    ``trainer.latent_state.rollout_strategy_arc_records`` after each rollout.
    The router context adds a 3-way opponent one-hot to the 35-d geometry; this
    is an observed INPUT feature, not opponent-identity SUPERVISION.

    What FLAT does and does not mean
    --------------------------------
    A flat Q-router does NOT re-open the question of repertoire diversity — that
    was already established by counterfactual actor-logit differences, forced-z
    behavioural separation, and the +2.37 oracle gap.  FLAT here means "no usable
    value separation was learned under THIS dataset, target horizon, context, and
    training budget" — i.e. the current Q-learning formulation failed to resolve
    the latents, not that the latents do not differ.  See the verdict semantics
    in ``experiments/run_v6i11_q_router.py``.
    """
    cfg = apply_plan_faithful_latent_v6i10_episode_router_explore_hardpool(cfg)
    # Internal router receives NO gradient; all routing credit goes to the
    # external Q-regressor.  Arc records are still collected as its training data.
    cfg.latent_arc_credit_coef = 0.0
    cfg.router_ent_coef = 0.0
    cfg.latent_lam_p = 0.0
    cfg.latent_lam_h = 0.0
    cfg.latent_lam_h_end = 0.0
    cfg.latent_strategy_ppo_coef = 0.0
    # Episode-persistent contract (arc == episode) is inherited from v6i10:
    #   latent_resample_every_n = 0, strategy_interval = 0, latent_arc_credit_min_len = 1.
    cfg.latent_arc_credit_enabled = True
    # 50 % uniform exploration so every (opponent, z) cell gets adequate samples.
    cfg.router_uniform_exploration_prob = 0.5
    cfg.experiment_id = "v6i11"
    cfg.run_tag = "v6i11_q_router_hardpool_OP8_OP9_OP10"
    return cfg


def apply_plan_faithful_latent_v6i12_advantage_router_hardpool(cfg: PPOConfig) -> PPOConfig:
    """V6I12 — paired-advantage router: V(context) baseline + A(context, z) residual.

    Proposed Preset Review
    ----------------------
    Proposed name: v6i12_advantage_router_hardpool.
    Parent preset: v6i11_q_router_hardpool (same arc collection infrastructure).
    Classification: SUMMER-COMPATIBLE EXTENSION (same frozen-actor, label-free
    training; only the EXTERNAL regressor changes from Q to V+A pair).

    Research question: does subtracting a context-specific baseline V(context)
    from the normalized episode return reveal reliable latent-advantage signal?
    V6I11 used raw paired Q targets, which still carry ~2.6–3.9 std of
    episode-level variance across opponents and episodes.  V(context) absorbs
    that component; A(context, z) = normalized_return - stopgrad(V(context))
    is the latent residual.

    Double-centering (in the external regressor, not the trainer):
      1. Global: norm_ret = (return - batch_mean) / (batch_std + eps)
      2. Context: a_target = norm_ret - stopgrad(V(context))
    Route: argmax_z A(context, z)

    Verdict pass condition (two requirements):
      * advantage gap CI excludes zero for ≥2 opponents
      * gap ≥ spread_threshold (default 0.05; lower than Q-router because
        advantages are V-centered, compressing the raw return scale)
    Held-out gate (required before promotion):
      * A-router > cross-episode-shuffled-A-router (decisive)
      * then A-router > uniform and approaches/beats fixed_z2

    Key changes vs v6i11_q_router_hardpool
    ---------------------------------------
    * Experiment script: ``experiments/run_v6i12_advantage_router.py``
    * External model: ContextualVBaseline + AdvantageRouter (``rl/router/advantage_router.py``)
    * Trainer-side: IDENTICAL to v6i11 (arc collection unchanged)
    Trainer settings inherited unchanged:
      * latent_arc_credit_enabled = True
      * router_uniform_exploration_prob = 0.5
      * strategy_interval = 0, latent_resample_every_n = 0
      * latent_arc_credit_min_len = 1 (arc == episode)
      * latent_arc_credit_coef = 0.0 (internal router no-op)
      * router_ent_coef = latent_lam_p = latent_lam_h = latent_strategy_ppo_coef = 0.0
    """
    cfg = apply_plan_faithful_latent_v6i11_q_router_hardpool(cfg)
    cfg.experiment_id = "v6i12"
    cfg.run_tag = "v6i12_advantage_router_hardpool_OP8_OP9_OP10"
    return cfg


def apply_plan_faithful_latent_v6i13_opening_window_advantage_router(cfg: PPOConfig) -> PPOConfig:
    """V6I13: delayed-commit opening-window advantage router.

    Proposed Preset Review
    ----------------------
    Proposed name: v6i13_opening_window_advantage_router.
    Parent preset: v6i12_advantage_router_hardpool.
    Classification: SUMMER-COMPATIBLE EXTENSION.
    Research question: does waiting 32 decision steps, then routing on an
    opening-window summary, produce a more learnable latent-advantage signal
    than choosing from the thin episode-start context?

    Key changes vs v6i12
    --------------------
    * ``latent_episode_strategy_warmup_decision_steps = 32``: q_phi commits
      after observing the opening.
    * ``router_warmup_uniform_z = True``: the pre-commit latent is sampled
      uniformly so no z receives a hidden default advantage.
    * ``router_arc_post_commit_only = True``: arc records open at commit, so
      ``arc_return`` is post-commit return, not full episode return.
    * ``router_opening_context_mode = "initial_commit_delta"``: finalized arc
      records carry ``opening_context = [state_0, state_commit, delta]`` for
      the external V/A diagnostic.

    The internal router PPO remains disabled exactly as in v6i12; the external
    diagnostic learns only from online sampled returns. No labels, opponent-ID
    supervision head, forced-z oracle target, hindsight best-z target, auxiliary
    task, or actor training is added.
    """
    cfg = apply_plan_faithful_latent_v6i12_advantage_router_hardpool(cfg)
    cfg.latent_episode_strategy_warmup_decision_steps = 32
    cfg.router_warmup_uniform_z = True
    cfg.router_arc_post_commit_only = True
    cfg.router_opening_context_mode = "initial_commit_delta"
    cfg.experiment_id = "v6i13"
    cfg.run_tag = "v6i13_opening_window_advantage_router_OP8_OP9_OP10"
    return cfg


def apply_plan_faithful_latent_v6i14_contract_specialists(cfg: PPOConfig) -> PPOConfig:
    """V6I14: contract-specialist repertoire birth.

    Proposed Preset Review
    ----------------------
    Proposed name: v6i14_contract_specialists.
    Parent preset: v6i9_mapaware_repertoire_hardpool.
    Classification: DIAGNOSTIC (non-Summer scaffold).
    Research question: can explicit temporary z-indexed behavioral contracts
    create real reusable specialists before router training resumes?

    This intentionally breaks the no-handcrafted-role boundary. It is not a
    paper-faithful row and should not be used for Summer-faithful claims.

    Contract map:
      z0: opening pressure toward enemy flag.
      z1: home defense / enemy-carrier pressure recovery.
      z2: friendly-carrier escort and support.
      z3: carrier conversion / closeout progress.

    Router is off during this phase. z is assigned by balanced episodes,
    shared actor trunk remains frozen by the v6i9 repertoire stage, and the
    z-specific adapters / embeddings / biases learn under normal env reward
    plus a small contract scaffold.
    """
    cfg = apply_plan_faithful_latent_v6i9_mapaware_repertoire_hardpool(cfg)
    cfg.latent_contract_specialist_enabled = True
    cfg.latent_contract_specialist_coef = 0.25
    cfg.latent_contract_specialist_clip = 1.0
    cfg.experiment_id = "v6i14"
    cfg.run_tag = "v6i14_contract_specialists_OP8_OP9_OP10"
    return cfg


def _apply_v6i15_contract_pressure(
    cfg: PPOConfig,
    *,
    multiplier: int,
    suffix: str,
) -> PPOConfig:
    cfg = apply_plan_faithful_latent_v6i14_contract_specialists(cfg)
    cfg.latent_contract_specialist_coef = 0.25 * float(multiplier)
    cfg.experiment_id = "v6i15"
    cfg.run_tag = f"v6i15_contract_pressure_{suffix}_OP8_OP9_OP10"
    return cfg


def apply_plan_faithful_latent_v6i15_contract_pressure_3x(cfg: PPOConfig) -> PPOConfig:
    """V6I15A: contract-pressure diagnostic, 3x contract coefficient.

    Proposed Preset Review
    ----------------------
    Proposed name: v6i15_contract_pressure_3x.
    Parent preset: v6i14_contract_specialists.
    Classification: DIAGNOSTIC (non-Summer scaffold).
    Research question: can the current frozen-shared-trunk, z-specific
    pathway express distinct behavior when the role contract is made loud?

    Delta table vs v6i14:
      Reward changed: yes, contract coefficient 0.25 -> 0.75.
      Actor architecture changed: no.
      Router objective changed: no; router remains off.
      Exploration schedule changed: no; balanced_episode remains active.
      Supervision added: no new labels beyond the inherited handcrafted
      z-role contract scaffold.

    This is the first pressure arm. If behavior fingerprints do not move
    under 3x/6x/10x, the next diagnostic is z-specific capacity or feature
    design, not router training.
    """
    return _apply_v6i15_contract_pressure(cfg, multiplier=3, suffix="3x")


def apply_plan_faithful_latent_v6i15_contract_pressure_6x(cfg: PPOConfig) -> PPOConfig:
    """V6I15A: contract-pressure diagnostic, 6x contract coefficient."""
    return _apply_v6i15_contract_pressure(cfg, multiplier=6, suffix="6x")


def apply_plan_faithful_latent_v6i15_contract_pressure_10x(cfg: PPOConfig) -> PPOConfig:
    """V6I15A: contract-pressure diagnostic, 10x contract coefficient."""
    return _apply_v6i15_contract_pressure(cfg, multiplier=10, suffix="10x")


def _apply_v6i16_capacity_knobs(cfg: PPOConfig) -> None:
    cfg.latent_z_gate_init = 0.08
    cfg.latent_actor_z_adapter_enabled = True
    cfg.latent_actor_z_adapter_scale = 0.10
    cfg.latent_actor_z_adapter_init_std = 0.05
    cfg.latent_actor_z_film_layers = 1


def apply_plan_faithful_latent_v6i16_sharp_contracts(cfg: PPOConfig) -> PPOConfig:
    """V6I16A: sharp contract-feature diagnostic with current z capacity.

    Proposed Preset Review
    ----------------------
    Proposed name: v6i16_sharp_contracts.
    Parent preset: v6i15_contract_pressure_3x.
    Classification: DIAGNOSTIC (non-Summer scaffold).
    Research question: can sharper role contracts force behavior separation when
    the current z pathway is held fixed?

    Delta table vs v6i15 3x:
      Reward changed: yes, contract variant base -> sharp.
      Actor architecture changed: no.
      Router objective changed: no; router remains off.
      Exploration schedule changed: no; balanced_episode remains active.
      Supervision added: no new external labels; the handcrafted contract
      scaffold is sharper and remains diagnostic-only.
    """
    cfg = apply_plan_faithful_latent_v6i15_contract_pressure_3x(cfg)
    cfg.latent_contract_specialist_variant = "sharp"
    cfg.experiment_id = "v6i16"
    cfg.run_tag = "v6i16_sharp_contracts_3x_OP8_OP9_OP10"
    return cfg


def apply_plan_faithful_latent_v6i16_capacity(cfg: PPOConfig) -> PPOConfig:
    """V6I16B: z-pathway capacity diagnostic with base contracts.

    Tests whether stronger z-specific actor leverage helps when the reward
    contract is held at the best v6i15 pressure level.
    """
    cfg = apply_plan_faithful_latent_v6i15_contract_pressure_3x(cfg)
    _apply_v6i16_capacity_knobs(cfg)
    cfg.experiment_id = "v6i16"
    cfg.run_tag = "v6i16_capacity_3x_OP8_OP9_OP10"
    return cfg


def apply_plan_faithful_latent_v6i16_capacity_sharp_contracts(cfg: PPOConfig) -> PPOConfig:
    """V6I16C: z-pathway capacity plus sharp contract features.

    This is the combined diagnostic arm. It should be judged only by forced-z
    behavioral fingerprints and role ownership metrics, not by saturated win
    rate. Router training remains blocked.
    """
    cfg = apply_plan_faithful_latent_v6i15_contract_pressure_3x(cfg)
    _apply_v6i16_capacity_knobs(cfg)
    cfg.latent_contract_specialist_variant = "sharp"
    cfg.experiment_id = "v6i16"
    cfg.run_tag = "v6i16_capacity_sharp_contracts_3x_OP8_OP9_OP10"
    return cfg


def apply_plan_faithful_latent_v6i17_surface_pressure_diagnostic(cfg: PPOConfig) -> PPOConfig:
    """V6I17A: surface-pressure diagnostic over the v6i16 combined scaffold.

    Proposed Preset Review
    ----------------------
    Proposed name: v6i17_surface_pressure_diagnostic.
    Parent preset: v6i16_capacity_sharp_contracts.
    Classification: DIAGNOSTIC (non-Summer scaffold).
    Research question: can specialists separate when the environment surface
    creates role tradeoffs that the current OP8/OP9/OP10 surface did not?

    Delta table vs v6i16 combined:
      Opponent surface changed: yes, OP8/OP9/OP10 -> OP8/OP9/OP10/OP11/OP12.
      Reward contract changed: no, sharp 3x contracts are inherited.
      Actor z capacity changed: no, v6i16 capacity knobs are inherited.
      Router objective changed: no; router remains off.
      Exploration schedule changed: no; balanced_episode remains active.

    This is not a router row. Promotion requires forced-z behavior, margin,
    tempo, or role-fingerprint separation on the harder/asymmetric surface.
    """
    cfg = apply_plan_faithful_latent_v6i16_capacity_sharp_contracts(cfg)
    cfg.opponent_pool = ("OP8", "OP9", "OP10", "OP11", "OP12")
    cfg.opponent_pool_weights = ()
    cfg.experiment_id = "v6i17"
    cfg.run_tag = "v6i17_surface_pressure_diagnostic_OP8_OP9_OP10_OP11_OP12"
    return cfg


def apply_plan_faithful_latent_v6i18_margin_tempo_surface_diagnostic(cfg: PPOConfig) -> PPOConfig:
    """V6I18A: margin/tempo consequence surface over the v6i17 scaffold.

    Proposed Preset Review
    ----------------------
    Proposed name: v6i18_margin_tempo_surface_diagnostic.
    Parent preset: v6i17_surface_pressure_diagnostic.
    Classification: DIAGNOSTIC (non-Summer scaffold).
    Research question: can specialists separate when the task grades margin,
    tempo, enemy pressure allowed, and near-cap conversion instead of only
    binary win/loss?

    Delta table vs v6i17:
      Horizon changed: yes, 400 -> 240 decision steps.
      Consequence surface changed: yes, default-off margin/tempo pressure terms.
      Opponent surface changed: no, OP8..OP12 inherited.
      Reward contract changed: no, sharp 3x contracts are inherited.
      Actor z capacity changed: no, v6i16 capacity knobs are inherited.
      Router objective changed: no; router remains off.
    """
    cfg = apply_plan_faithful_latent_v6i17_surface_pressure_diagnostic(cfg)
    cfg.max_decision_steps = 240
    cfg.env_stalemate_max_steps = 80
    cfg.env_surface_score_margin_coef = 0.15
    cfg.env_surface_blue_capture_tempo_bonus = 0.25
    cfg.env_surface_red_flag_touch_penalty = 0.20
    cfg.env_surface_red_carrier_progress_penalty = 0.025
    cfg.env_surface_blue_near_cap_bonus = 0.015
    cfg.experiment_id = "v6i18"
    cfg.run_tag = "v6i18_margin_tempo_surface_OP8_OP9_OP10_OP11_OP12"
    return cfg
