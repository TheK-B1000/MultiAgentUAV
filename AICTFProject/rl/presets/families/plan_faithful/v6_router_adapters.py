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
