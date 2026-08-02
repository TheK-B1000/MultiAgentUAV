"""``PPOConfig`` dataclass + ``TrainMode`` enum.

Extracted from :mod:`rl.train_ppo` to keep configuration shape independent of
training logic so the trainer, presets, CLI parsing, and validation modules
can all import it without dragging in the full training pipeline.

Reproducibility contract: field names, defaults, and the ``TrainMode`` string
values are part of the on-disk artifact format (``run_config.json`` snapshots,
checkpoint payloads, preset modules). Treat changes here as a breaking change
for archived runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional, Tuple

import torch
from rl.telemetry_mode import TrainingTelemetryMode


class TrainMode(str, Enum):
    FIXED_OPPONENT = "FIXED_OPPONENT"
    # Uniform random scripted opponent each episode (same hook as opponent_randomize; explicit mode).
    OPPONENT_POOL = "OPPONENT_POOL"
    CURRICULUM = "CURRICULUM"
    # Backward-compatible alias for old configs/commands.
    CURRICULUM_NO_LEAGUE = "CURRICULUM"


@dataclass
class PPOConfig:
    seed: int = 42
    total_timesteps: int = 1_000_000
    n_envs: int = 32
    n_steps: int = 2048
    batch_size: int = 1024
    n_epochs: int = 6
    gamma: float = 0.995
    gae_lambda: float = 0.99
    clip_range: float = 0.25
    clip_range_vf: Optional[float] = 0.2
    vf_coef: float = 1.0
    normalize_returns: bool = False
    ent_coef: float = 0.01
    learning_rate: float = 5e-4
    lr_floor_frac: float = 0.1
    max_grad_norm: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir: str = "checkpoints"
    load_path: Optional[str] = None
    allow_active_actor_module_migration: bool = False
    run_tag: str = "ppo_latent_2v2"
    enable_metrics_csv: bool = True
    metrics_csv_path: Optional[str] = None
    episode_csv_path: Optional[str] = None
    strategy_experience_csv_path: Optional[str] = None
    # If True before training, existing non-empty telemetry CSVs are rotated aside so a new run
    # does not append duplicate timesteps under the same --run-tag.
    fresh_metrics_csv: bool = False
    # Set from CLI ``--preset`` only (reproducibility / run_config.json); behavior is already merged into fields below.
    cli_preset: Optional[str] = None
    # E3: optional per-step CSV (z, H(q), argmax, switch, phase). See `rl.custom_ppo.E3_STEP_TELEMETRY_FIELDS`.
    e3_step_telemetry_path: Optional[str] = None

    # Telemetry and monitoring configurations (Phase 6.1)
    training_telemetry_mode: TrainingTelemetryMode = TrainingTelemetryMode.OFF
    training_events_jsonl_path: Optional[str] = None
    telemetry_events_jsonl_path: Optional[str] = None
    performance_summary_path: Optional[str] = None
    performance_samples_path: Optional[str] = None
    gpu_monitor_enabled: bool = False
    gpu_monitor_interval_seconds: float = 1.0
    # SB3-compatible: ``tqdm`` (prefer ``tqdm.rich``) during rollout, ``total=remaining`` timesteps, ``update(n_envs)`` / step.
    enable_progress_bar: bool = True
    verbose_training: bool = False
    # After this many *completed* episodes, print W/L/D and win rate (0 = disabled).
    episode_log_every: int = 1000

    # Observational tag-event telemetry on the LIVE environment: tag successes,
    # cooldown denials, capture events and episode-reset markers, each carrying
    # the authoritative integer event identity. Behaviour-neutral by contract
    # (tests/test_tag_telemetry.py) -- identical states, rewards and outcomes
    # under the same seed with it on or off. Off by default so ordinary runs pay
    # nothing; ``formal_run`` requires it explicitly so a formal result can never
    # be reported without the evidence needed to audit its tagging.
    tag_telemetry_enabled: bool = False
    # Marks a run whose artifacts are intended as a formal result. Turns the
    # audit preconditions into start-time gates rather than after-the-fact hopes.
    formal_run: bool = False

    max_decision_steps: int = 400
    map_set: str = "train"
    map_layout: str = "map_a_open"
    map_pool: tuple[str, ...] = field(default_factory=tuple)
    # None = derive from map_layout / map_pool (GPUFieldConfig). Set True to keep
    # the 8-channel obstacle plane on map_a_open when continuing map_b-lineage
    # checkpoints (LRO / V6I23+). The plane is zeros on open arenas.
    obstacle_obs_channel: Optional[bool] = None
    mode: str = TrainMode.FIXED_OPPONENT.value
    fixed_opponent_tag: str = "OP3"
    # Uniform random scripted opponent per episode: either mode=OPPONENT_POOL or FIXED_OPPONENT + True.
    # Uses GPUCTFVecEnv pre-reset hook so the next episode matches sampled opponents from opponent_pool.
    # Default excludes OP4 (reserved for zero-shot eval). Use ``--allow-op4-in-training-pool`` to train vs OP4.
    opponent_randomize: bool = False
    opponent_pool: tuple[str, ...] = field(default_factory=lambda: ("OP1", "OP2", "OP3"))
    # Per-tag sampling probabilities for opponent_randomize, aligned positionally with
    # opponent_pool. Empty tuple (default) = uniform 1/N over the pool. Non-empty must
    # have the same length as opponent_pool; values are auto-normalized to sum 1.0 by
    # ``normalize_and_validate_training_config``. Plan-faithful — only changes how often
    # each opponent is sampled, not the opponent definitions.
    opponent_pool_weights: tuple[float, ...] = field(default_factory=tuple)
    # Optional joint (opponent, map) episode distribution for diagnostics (e.g. V6I24).
    # Each entry is (opponent_tag, map_layout, weight). When non-empty, the pre-reset
    # hook samples a cell jointly and overrides both opponent and next map layout.
    # Weights are normalized to sum 1.0. Empty = legacy independent opponent/map sampling.
    training_cell_distribution: tuple[tuple[str, str, float], ...] = field(default_factory=tuple)
    # After loading a checkpoint, freeze return-normalization stats (no further updates).
    # Used by V6I24 so member policies share identical frozen normalization.
    freeze_return_norm_after_load: bool = False
    allow_op4_in_training_pool: bool = False
    max_blue_agents: int = 2
    use_deterministic: bool = False
    # Not in *Summer Implementation Plan.docx*; when True, overrides several PPO fields below for a legacy "stable" profile. Default False so explicit config matches the spec numbers.
    use_stable_marl_ppo: bool = False
    target_kl: Optional[float] = 0.02
    strategy_target_kl: Optional[float] = None
    actor_cnn_feature_dim: int = 128
    actor_hidden_dim: int = 256
    router_context_mode: str = ""
    router_context_dimension: int = 0
    router_persistence_mode: str = ""
    router_marginal_entropy_coefficient: float = 0.0
    router_conditional_entropy_coefficient: float = 0.0
    router_allowed_latents: tuple[int, ...] = field(default_factory=tuple)
    router_freeze_actor: bool = False
    router_reinitialize_on_load: bool = False
    # Training-only router behavior mixture:
    #   p_train(z|s) = (1 - eps) * q_phi(z|s) + eps * Uniform(allowed z).
    # Deterministic evaluation still uses q_phi directly.
    router_uniform_exploration_prob: float = 0.0
    # Delayed-commit router controls. Default-off so existing warmup users keep
    # their current behavior. When enabled, the pre-commit latent is sampled
    # uniformly, arc credit starts only at the warmup commit, and finalized arc
    # records carry an opening summary for external V/A router diagnostics.
    router_warmup_uniform_z: bool = False
    router_arc_post_commit_only: bool = False
    router_opening_context_mode: str = ""

    # Summer/ICRA latent team strategy is the default proposed algorithm.
    use_latent_strategy: bool = True
    latent_k: int = 4
    latent_z_embed_dim: int = 16
    latent_actor_conditioning: Literal["concat", "film_v6"] = "concat"
    latent_actor_z_onehot_enabled: bool = False
    latent_actor_z_onehot_scale: float = 1.0
    latent_actor_z_embed_scale: float = 1.0
    latent_actor_z_adapter_enabled: bool = False
    latent_actor_z_adapter_scale: float = 0.0
    latent_actor_z_adapter_init_std: float = 0.02
    latent_actor_z_film_layers: int = 1
    enable_actor_z_film: bool = False
    actor_z_film_init_scale: float = 0.0
    actor_z_film_layer: int = 2
    latent_actor_z_adapter_warmup_steps: int = 0
    latent_actor_z_adapter_ramp_steps: int = 0
    latent_vf_hidden: int = 128
    latent_strategy_hidden: int = 128

    # v6i1 staged curriculum parameters
    curriculum_nominal_timesteps: int = 1_000_000
    phase_a_gate_check_interval: int = 25_000
    latent_cf_min_episodes_per_z: int = 50
    latent_cf_occupancy_min: float = 0.18
    latent_cf_occupancy_max: float = 0.34
    latent_cf_jsd_margin: float = 0.01
    latent_cf_jsd_ema_alpha: float = 0.10
    latent_cf_gate_consecutive_updates: int = 5
    latent_cf_competence_delta: float = 5.0
    latent_cf_competence_gate_tc: float = 1.0
    latent_cf_competence_ema_alpha: float = 0.05
    phase_boundary_gate_mode: str = "enforce"
    phase_a_disable_promotion: bool = False
    latent_cf_coef_max: float = 0.01
    latent_cf_worst_pair_coef: float = 0.0
    latent_cf_weak_pair_boost: float = 0.0
    latent_cf_require_competence: bool = False
    actor_cf_update_mode: Literal["combined", "ppo_then_cf", "cf_then_ppo"] = "combined"
    latent_cf_sequential_update: bool = False
    probe_utility_tie_margin: float = 0.05
    phase_a_gate_max_seconds: int = 900
    phase_a_gate_progress_interval_seconds: int = 60
    curriculum_gate_online_matched_seed_count: int = 5
    curriculum_gate_online_matched_seed_max_steps: int = 64
    curriculum_gate_run_boundary_eval: bool = False
    curriculum_gate_run_probe: bool = False
    curriculum_gate_selector_blocks_phase_a: bool = False
    use_v6i1_curriculum: bool = False
    training_mode: str = "default"
    experiment_family: str = "v6"
    experiment_id: str = "v6i1"
    # Gate promotion protocol (v6i1 = single macro intervention; v6i2 = dual evidence).
    gate_protocol_version: str = "v6i1_single_macro_intervention"
    phase_a_earliest_end_fraction: float = 0.40
    phase_a_max_end_fraction: float = 0.55
    phase_b_fixed_fraction: float = 0.30
    phase_c_fixed_fraction: float = 0.30
    phase_c_start_fraction: float = 0.70
    curriculum_extend_terminal_on_late_promotion: bool = True
    allow_gate_config_mismatch_on_resume: bool = False
    gate_config_mismatch_override_used: bool = False
    gate_config_fingerprint_checkpoint: str = ""
    gate_config_fingerprint_active: str = ""
    confirmatory_gate_lineage_valid: bool = True
    evaluation_only_preset: bool = False
    evaluation_only_runner: str = ""
    evaluation_only_requires_checkpoint: bool = False
    evaluation_only_checkpoint_family: str = ""
    # v6i2 actor-intervention track (CF-batch pair JSD EMA).
    actor_jsd_margin: float = 0.001
    actor_jsd_floor_fraction: float = 0.5
    actor_jsd_min_passing_pairs: int = 5
    actor_jsd_consecutive_updates: int = 3
    actor_jsd_ema_decay: float = 0.10
    actor_jsd_stale_gate_grace: int = 1
    # v6i2 macro-rollout supporting profile (forced-z rollout pair JSD EMA).
    macro_jsd_margin: float = 0.0001
    macro_jsd_floor_fraction: float = 0.5
    macro_jsd_min_passing_pairs: int = 1
    macro_jsd_ema_decay: float = 0.10
    # v6i2 behavioral-realization matched-seed composite.
    behavioral_realization_min_opponents_pass: int = 2
    behavioral_realization_effect_threshold: float = 0.02
    behavioral_realization_adverse_threshold: float = -0.01
    behavioral_route_distance_scale: float = 0.03
    behavioral_task_behavior_distance_scale: float = 0.02
    behavioral_performance_spread_scale: float = 0.03
    behavioral_route_distance_weight: float = 0.25
    behavioral_task_behavior_distance_weight: float = 0.50
    behavioral_performance_spread_weight: float = 0.25
    behavioral_aggregate_effect_threshold: float = 0.75
    behavioral_min_task_behavior_distance: float = 0.01
    behavioral_min_performance_spread: float = 0.01
    behavioral_matched_seed_min_seeds_per_opponent: int = 20
    curriculum_probe_min_examples: int = 10
    # V6I3 local emergent communication (off by default — v6i2 unchanged when False).
    communication_enabled: bool = False
    comm_protocol_version: str = "v6i3_strategy_local_comm_v1"
    comm_num_symbols: int = 4
    comm_silence_symbol: int = -1
    comm_interval_steps: int = 32
    comm_delivery_delay_steps: int = 1
    comm_radius_cells: float = 6.0
    comm_dropout_probability: float = 0.10
    comm_entropy_coef: float = 0.001
    comm_hold_last_message: bool = True
    comm_local_only: bool = True
    comm_include_sender_position: bool = True
    comm_message_grid_channels: int = 4
    comm_cf_include_message_head: bool = False
    # V6I3 evidence gates. Defaults are inert for non-communication rows; v6i3 freezes overrides.
    comm_min_valid_boundaries: int = 0
    comm_min_deliveries: int = 0
    comm_min_symbols_used: int = 3
    comm_entropy_floor: float = 0.0
    comm_symbol_dominance_ceiling: float = 1.0
    comm_listener_jsd_margin: float = 0.0
    comm_listener_min_passing_pairs: int = 0
    comm_listener_min_states: int = 0
    comm_listener_consecutive_updates: int = 0
    # V6I7: per-latent residual actor adapters h_z = h + g_z*A_z(h) and logit biases B_z.
    enable_latent_z_residual: bool = False
    latent_z_gate_init: float = 0.01
    # V6I22E: fixed-alpha gate-free adapters.  When > 0, Kaiming init replaces
    # zero-init and the learned gate is removed: h_z = h + alpha * A_z(h).
    latent_z_residual_alpha: float = 0.0
    # V6I23 population birth (Summer-compatible extension, not paper-faithful):
    # independent per-z specialists under forced balanced_episode assignment.
    # Active-z-only residual forward avoids evaluating unused adapters; per-z
    # action heads are Stage-2 trainable (shared action_head stays frozen).
    latent_population_birth_active_z_only: bool = False
    latent_population_birth_per_z_action_heads: bool = False
    # V6I26 Latent Response-Oracle (DIAGNOSTIC / Claim B):
    # Deep per-z trunks (last two MLP layers) + active-branch-only BR rounds.
    latent_lro_deep_branches: bool = False
    latent_lro_active_branch_only: bool = False
    # V6I26 actor-step ablation (DIAGNOSTIC): repertoire shared Adam with
    # separate z-actor vs critic grad clipping, and optional z-actor LR mult.
    # Defaults preserve the joint-clip / single-LR behavior of the weak 5u pilot.
    latent_lro_separate_actor_critic_clip: bool = False
    latent_lro_z_actor_lr_mult: float = 1.0
    # V6I24 full-policy population diagnostic (DIAGNOSTIC, not PAPER-FAITHFUL):
    # Trains K completely independent policies from the same cloned checkpoint.
    # Each policy has its own actor, critic, optimizer, buffer, and obs-norm.
    # No shared gradients, no router, no latent conditioning.
    population_training_enabled: bool = False
    population_k: int = 4
    population_pressure_rotation_interval: int = 10
    population_round_robin_updates_per_cycle: int = 1
    # V6I1 Phase B/C macro-router and rehearsal controls.
    v6i1_recurrent_selector_hidden: int = 32
    v6i1_macro_strategy_ppo_coef: float = 1.0
    v6i1_macro_strategy_n_epochs: int = 4
    v6i1_macro_strategy_clip_eps: float = 0.2
    v6i1_macro_strategy_value_coef: float = 0.5
    v6i1_macro_strategy_return_norm: bool = True
    v6i1_router_lr: Optional[float] = 5e-3
    v6i1_phase_c_actor_lr_frac: float = 0.05
    v6i1_router_rehearsal_episode_frac: float = 0.25
    # V6I4 router-ablation evaluation protocol. These fields do not change
    # PPO training; they lock the post-training comparison contract for a
    # promoted v6i2-style checkpoint.
    router_ablation_protocol_version: str = ""
    router_ablation_claim_label: str = ""
    router_ablation_classification: str = ""
    router_ablation_conditions: tuple[str, ...] = field(default_factory=tuple)
    router_ablation_oracle_conditions: tuple[str, ...] = field(default_factory=tuple)
    router_ablation_primary_metrics: tuple[str, ...] = field(default_factory=tuple)
    router_ablation_diagnostic_metrics: tuple[str, ...] = field(default_factory=tuple)
    router_ablation_opponents: tuple[str, ...] = field(default_factory=tuple)
    router_ablation_calibration_seed_set: str = ""
    router_ablation_evaluation_seed_set: str = ""
    router_ablation_matched_seeds: bool = True
    router_ablation_identical_initial_states: bool = True
    router_ablation_identical_action_sampling: bool = True
    router_ablation_identical_episode_horizon: bool = True
    router_ablation_episode_oracle_is_deployable: bool = False

    # Plan IMPLEMENTATION §6: typical λ_H ∈ [0.001, 0.01]; λ_p ∈ [0.01, 0.05] (see also §3.3 for a wider λ_p range).
    # ``maximize`` matches the plan (encourage exploratory / diverse q_phi). ``minimize`` adds +λ_H·H to the
    # minimized loss and sharpens q_phi (recommended when telemetry shows strategy_entropy≈ln K with no persistence grad).
    # ``none`` removes the H term (strategy_encoder receives no gradient from λ_H when λ_p/KL are also inactive).
    latent_entropy_mode: Literal["conditional", "marginal"] = "conditional"
    latent_entropy_objective: Literal["maximize", "minimize", "none"] = "maximize"
    latent_lam_h: float = 0.005
    latent_lam_h_start: Optional[float] = None
    latent_lam_h_end: Optional[float] = None
    latent_entropy_anneal_start: Optional[int] = None
    latent_entropy_anneal_end: Optional[int] = None
    latent_lam_p: float = 0.02
    # A1: clipped PPO/REINFORCE-style update for sampled z. Kept low because z operates at episode cadence.
    latent_strategy_ppo_coef: float = 0.1
    # Option A episode-start strategy credit: PPO update on the sampled z using full
    # completed-episode return. Pure task-return credit; no labels or semantic heads.
    latent_episode_strategy_ppo: bool = False
    # Default 0.0 keeps episode-credit OFF by default per the SUMMER plan: latent z is
    # learned end-to-end from task reward via the MARL loss + persistence regularizer,
    # with no auxiliary objectives. Presets that opt into episode-credit (e.g.
    # plan_faithful_latent_episode_credit) must set this explicitly.
    latent_episode_strategy_coef: float = 0.0
    latent_episode_strategy_clip_eps: float = 0.2
    latent_episode_strategy_value_coef: float = 0.5
    latent_episode_strategy_return_norm: bool = True
    # Decision-step warmup before locking the per-episode z under episode-credit mode.
    # 0 = legacy behavior: snapshot the z chosen at step 0 (ctx170 EMAs still at reset,
    # zero opponent fingerprint -- structurally bounds MI(z; opponent) near zero).
    # >0 = let the episode run for N decision steps, then force-resample z and snapshot
    # that (context, z) pair for q_phi's per-episode credit. With alpha_short=0.2 the
    # short EMA reaches ~63%/86% of equilibrium by step 5/10, exposing opponent dynamics
    # (red_speed, formation, flag pressure) that distinguish OP3/OP5/OP6. Only takes
    # effect when ``latent_episode_strategy_ppo == True``.
    latent_episode_strategy_warmup_decision_steps: int = 0
    # Number of inner PPO epochs ``apply_episode_strategy_ppo`` runs per training
    # update. Default 1 reproduces legacy v3/v3b behavior (a single backward step
    # per rollout, ~15 q_phi updates over a 1M-step run). Set higher (e.g. 6-8,
    # matching the actor's ``n_epochs``) when the marginal-baseline gradient is
    # measurably nonzero but cumulative logit change is too small to move q_phi
    # off uniform within the update budget. Only takes effect when
    # ``latent_episode_strategy_ppo == True``.
    latent_episode_strategy_n_epochs: int = 1
    # Dedicated learning rate for the q_phi strategy encoder and the
    # episode_strategy_value_head when running ``apply_episode_strategy_ppo``.
    # ``None`` (default) falls back to the shared optimizer's learning rate
    # (calibrated for the noisy actor). Set higher (e.g. 1e-3 to 1e-2) when
    # the router has the right gradient direction but cumulative step is too
    # small at the actor-tuned LR. A separate AdamW optimizer with this LR is
    # built at trainer init and steps strategy_encoder + episode_strategy_value
    # parameters. To avoid double-stepping the same params from the shared
    # optimizer, ``ppo_updater.update`` detects the dedicated router optimizer
    # (``runtime.latent_router_optimizer is not None``) and suppresses the
    # main-loop q_phi gradient channels (entropy / persistence / KL /
    # strategy-PPO / aux-return) for that update. This replaces the v3c
    # "Fix 5" coef-zero gate; the dedicated-optimizer check is more precise
    # because it doesn't silently zero ``lam_p`` and ``lam_h`` when a preset
    # uses arc-credit only and runs the shared optimizer for everything else
    # (the v3i19 / v4i1 / v4i3 case).
    latent_episode_strategy_lr: Optional[float] = None
    # ------------------------------------------------------------------
    # v3i19 arc-credit channel: per-z-arc PPO update on q_phi.
    # ------------------------------------------------------------------
    # When ``latent_arc_credit_enabled`` is True, q_phi receives a PPO gradient
    # at every z-arc boundary (i.e. every time a sparse resample or event refresh
    # changes z, plus at episode end). Each arc contributes one training record
    # (ctx_at_arc_start, z, log_prob(z), arc_return) to the rollout's arc buffer
    # and the buffer is processed in inner PPO epochs at update time.
    #
    # This is the Summer-faithful "reward z consequences, not z existence" channel
    # specified for v3i19_summer_consequence. It is mutually exclusive with the
    # per-step ``latent_strategy_ppo_coef`` channel (which v3i18 proved too noisy)
    # and orthogonal to the per-episode ``latent_episode_strategy_ppo`` channel
    # (which can be left off when arc-credit is on; arc credit subsumes it for
    # episode-arc presets when ``latent_resample_every_n == 0``).
    #
    # Plan-faithful contract: arc_return is summed env reward over the arc only.
    # No labels, no semantic heads, no opponent ID seen by q_phi. The advantage
    # baseline is either q_phi's own value head (``context_value``) or a global
    # EMA over arc returns (``running_mean``); neither uses external labels.
    latent_arc_credit_enabled: bool = False
    # Loss coefficient applied to the arc-credit PPO loss. Mirrors
    # ``latent_episode_strategy_coef`` semantics. v3i18 confirmed per-step
    # coef=0.3 was insufficient; v3i19's per-arc default is 1.0 for stronger
    # consequence amplitude (one credit signal per arc, not per step).
    latent_arc_credit_coef: float = 1.0
    # Number of PPO inner epochs run over the arc buffer per training update.
    # Mirrors ``latent_episode_strategy_n_epochs``. Default 4 because the arc
    # batch is per-arc (not per-episode), so the batch is ~K times larger than
    # the episode batch with ``resample_every_n>0``; the extra epochs give the
    # arc gradient enough cumulative step to move q_phi.
    latent_arc_credit_n_epochs: int = 4
    # Standard PPO clipping epsilon for the arc-credit ratio.
    latent_arc_credit_clip_eps: float = 0.2
    # Normalize arc advantages within each PPO mini-batch (mean=0, std=1).
    # On by default; essential when WR varies 50-100% across opponents.
    latent_arc_credit_return_norm: bool = True
    # Advantage baseline for the arc-credit PPO loss:
    #   "context_value" -- V_phi(ctx_at_arc_start, z_arc). Reuses the existing
    #                      ``episode_strategy_value_head`` (no new params).
    #   "running_mean"  -- detached EMA of arc returns (no V dependency).
    # The Summer plan only requires that no labels or aux heads be introduced;
    # both baselines satisfy that. Default "context_value" matches v3i19 spec.
    latent_arc_credit_baseline: str = "context_value"
    # Minimum arc length (in env steps) required for the arc to contribute to
    # the PPO loss. Arcs shorter than this (e.g. cut by episode end soon after
    # a z resample) are dropped from the loss but still counted in telemetry.
    # Prevents extremely short noisy arcs from dominating the gradient.
    latent_arc_credit_min_len: int = 32
    # Bucketing strategy for q_phi's advantage baseline (v3d "Smart Coach
    # Router"). When set, the q_phi advantage replaces the V-marginal baseline
    # with an empirical per-bucket mean of episode returns:
    #
    #     v3c:  adv = R - mean_k V(s, z_k)        # marginal-over-V baseline
    #     v3d:  adv = R - mean(R | bucket(s))     # marginal-over-bucket baseline
    #
    # This is variance-reduction by stratification (standard PPO/REINFORCE
    # technique). q_phi learns "is this z better than average WITHIN this
    # bucket?" rather than "better than overall average?". The V head is
    # poorly calibrated for off-policy z (each z slot only sees value-loss
    # updates for episodes where it was actually picked, ~25% at uniform),
    # so the V-marginal baseline subtracts noise; the bucket mean is empirical
    # and gives q_phi the variance-optimal stratified-sampling gradient.
    #
    # Plan-faithful: bucket ids are GRADIENT-shaping signals, never inputs to
    # the policy. q_phi still only sees (s) and learns pi(z|s). Bucket only
    # affects gradient variance, not policy input.
    #
    # Supported values:
    #   None                -- default; v3c V-marginal baseline behavior
    #   "opponent"          -- bucket by scripted opponent (3 buckets for OP3/5/6)
    #   "bucket_id"         -- 216-bucket flag/score/spread/dist composite captured
    #                          at z-commit time (in episode_strategy_bucket).
    #   "opponent_x_bucket" -- cross product (up to ~648 buckets; sharper but noisier)
    #   "tactical_context_opponent" -- phase + both flag states + score pressure
    #                                  + opponent, accumulated across the episode.
    latent_q_phi_bucket_baseline: Optional[str] = None
    # EMA decay for cross-rollout bucket means. 1.0 = no update (pure prior);
    # 0.0 = per-rollout means only (no smoothing). Default 0.9 retains 90% of
    # prior + 10% of new rollout's bucket means. Higher = smoother but slower
    # to react when policy/opponent statistics shift.
    latent_q_phi_bucket_baseline_ema: float = 0.9
    # Per-rollout count fallback. When fewer than this many episodes land in a
    # bucket within the current rollout, use the rollout's GLOBAL mean for those
    # episodes instead of the bucket mean. Avoids huge advantages from
    # singleton-bucket episodes during the early "EMA priming" rollouts.
    latent_q_phi_bucket_baseline_min_count: int = 8
    # Use a z-marginal value baseline for q_phi's policy-gradient advantage.
    # Default False preserves the legacy (z-conditioned) baseline:
    #     adv = R - V(s, z_picked)
    # which mathematically subtracts the cross-z signal q_phi is supposed to learn from
    # (the centralized value head absorbs E[R | s, z] before the router ever sees it).
    # When True, the advantage uses an expectation under q_phi's current policy:
    #     adv = R - E_{z' ~ q_phi(s)}[V(s, z')] = R - sum_k pi_phi(k|s) * V(s, k)
    # which is the variance-optimal AAC baseline and gives q_phi a non-zero gradient
    # encoding "this z vs the average z" instead of "this z vs its own expectation".
    # Plan-faithful: no labels, no aux heads, no opponent IDs -- just the correct
    # baseline math for an option-style policy gradient.
    latent_q_phi_marginal_baseline: bool = False
    # A2 (opt-in): auxiliary MSE on the shared q_phi trunk predicting per-z returns from the **sampled** z only.
    # Not a full Q(s,a,z) critic and not off-policy Q-learning; MAPPO value remains V_phi(s, a, z).
    latent_strategy_aux_return_head: bool = False
    latent_strategy_aux_return_coef: float = 1.0
    latent_strategy_aux_predict_phase_coef: float = 0.0
    latent_strategy_tau: float = 1.0
    # 0 = sample once at episode start (main paper default; plan Option A). N>=2 = sparse refresh (Option B).
    latent_resample_every_n: int = 0
    # Mid-episode z changes make V(s,z) discontinuous; optionally break GAE carry across z[t]!=z[t+1].
    latent_gae_reset_on_z_change: bool = True
    # Use argmax z from q_phi(s') when bootstrapping V(s') so peek matches no duplicate stochastic z draw.
    latent_bootstrap_z_deterministic: bool = True
    # Baseline: keep latent actor/critic plumbing, but clamp every rollout to one strategy ID.
    # This tests whether learned/multiple strategy selection matters beyond a single learned z embedding.
    fixed_latent_strategy: bool = False
    fixed_latent_strategy_id: int = 0
    # **Ablation / plan §12 only** — not combined with the main “episode-start z” story by default.
    # Use ``rl.config_presets.ablation_flag_resample_config`` for an explicit run.
    latent_resample_on_flag: bool = False
    # Optional §12: KL( q_\phi(s_t) || q_\phi(s_{t-1}) ) on consecutive time steps; 0 = off (ablation only).
    latent_kl_consecutive: float = 0.0
    latent_q_phi_option_advantage: bool = False
    latent_q_phi_marginal_baseline: bool = False
    # v3f: self-supervised option separation. A fraction of episodes are forced
    # to a uniformly sampled z, then completed-episode behavior embeddings are
    # rewarded for separating from other z centroids in the same label-free
    # context bucket. This does not assign semantic roles to latents.
    latent_forced_z_episode_frac: float = 0.0
    # v5i3: optional anneal schedule for ``latent_forced_z_episode_frac``.
    # When all four fields are set, ``resolve_latent_forced_z_frac`` linearly
    # interpolates from ``_start`` (before ``anneal_start``) to ``_end`` (after
    # ``anneal_end``); otherwise the legacy constant value above is used.
    # The forced episodes themselves are never routed into ``q_phi``'s PPO
    # update -- they early-return into ``latent_preference_buffer`` -- so the
    # schedule controls actor-side coverage exploration only.
    latent_forced_z_episode_frac_start: Optional[float] = None
    latent_forced_z_episode_frac_end: Optional[float] = None
    latent_forced_z_anneal_start: Optional[int] = None
    latent_forced_z_anneal_end: Optional[int] = None
    latent_behavior_contrast_coef: float = 0.0
    latent_behavior_contrast_margin: float = 0.25
    latent_behavior_contrast_ema: float = 0.9
    latent_behavior_contrast_anneal_after_steps: int = 0
    latent_behavior_contrast_anneal_to: float = 0.0
    # Label-free outcome diversity for repertoire birth. Successful forced-z
    # episodes can receive a bounded terminal bonus when their generic outcome
    # scalar separates from other z outcome centroids in the same context bucket.
    latent_outcome_diversity_coef: float = 0.0
    latent_outcome_diversity_margin: float = 1.0
    latent_outcome_diversity_ema: float = 0.9
    latent_outcome_diversity_success_only: bool = True
    # v5i9 CSIA extension: detached reward feedback from frozen forced-z
    # evaluation evidence. Default-off so every existing preset reproduces
    # its original reward path unless a preset/CLI explicitly enables it.
    csia_enabled: bool = False
    csia_reward_coef: float = 0.0
    # Expected payoff CSV: tools/qualitative_rollout.py
    # ``*_qualitative_rollout_by_z.csv`` with fixed_z rows.
    csia_payoff_csv_path: Optional[str] = None
    # Optional evidence CSV: ``*_strategy_evidence.csv`` for natural router
    # baseline and forced-z behavior spread gates.
    csia_strategy_evidence_csv_path: Optional[str] = None
    # Number of PPO updates between evidence reloads. 0 = load once.
    csia_probe_interval: int = 1
    csia_min_behavior_spread: float = 0.10
    csia_min_interaction_strength: float = 0.05
    csia_quality_floor_delta: float = 0.10
    csia_require_gates: bool = True
    csia_min_count_per_cell: int = 1
    latent_actor_z_separation_coef: float = 0.0
    latent_actor_z_separation_start_coef: float = 0.0
    latent_actor_z_separation_margin: float = 0.02
    latent_actor_z_separation_warmup_steps: int = 0
    latent_actor_z_separation_ramp_steps: int = 0
    latent_actor_z_separation_min_abs_advantage: float = 0.0
    latent_actor_z_separation_min_decision_frac: float = 0.0
    latent_actor_z_separation_max_entropy_frac: float = 1.0
    latent_usage_balance_coef: float = 0.0
    latent_q_phi_train_after_steps: int = 0
    latent_preference_coef: float = 0.0
    latent_preference_temperature: float = 0.75
    latent_preference_min_bucket_count: int = 8
    latent_preference_min_distinct_z: int = 2
    latent_preference_opponent_balanced: bool = False
    latent_preference_log_opponent_targets: bool = False
    latent_preference_confidence_scale: float = 2.0
    latent_preference_commit_coef: float = 0.003
    # v3i7: advantage-weighted router distillation. Uses forced-z outcome
    # evidence to teach q_phi which discovered z wins within the same
    # opponent/context bucket. This is label-free: the target is derived from
    # observed win-rate advantage by z, not role names or scripted tactics.
    latent_awrd_enabled: bool = False
    latent_awrd_coef: float = 0.0
    latent_awrd_temperature: float = 0.35
    latent_awrd_min_bucket_count: int = 8
    latent_awrd_min_distinct_z: int = 2
    latent_awrd_margin_threshold: float = 0.15
    latent_awrd_margin_scale: float = 2.0
    latent_awrd_min_margin: float = 0.08
    latent_awrd_soft_margin_gating: bool = False
    latent_awrd_warmup_steps: int = 0
    latent_awrd_ramp_steps: int = 0
    latent_awrd_boost_after_steps: int = 0
    latent_awrd_boost_after_fraction: float = 0.0
    latent_awrd_boost_multiplier: float = 1.0
    # v3i9: balanced specialist router. Keeps the marginal q_phi usage
    # distribution high-entropy across the batch while reducing conditional
    # entropy inside opponent/context buckets. No role labels or scripted
    # z meanings; buckets shape the router objective and telemetry only.
    latent_specialist_router_enabled: bool = False
    latent_marginal_balance_coef: float = 0.0
    latent_conditional_entropy_min_coef: float = 0.0
    latent_conditional_entropy_min_coef_start: float = 0.0
    # "state" minimizes H(q_phi(z|s)); "context_bucket" minimizes the entropy
    # of the mean router distribution within each active tactical bucket.
    latent_specialist_conditional_entropy_scope: str = "state"
    latent_context_mi_coef: float = 0.0
    latent_specialist_warmup_steps: int = 0
    latent_specialist_ramp_steps: int = 1
    latent_specialist_min_bucket_count: int = 2
    latent_specialist_context_key_mode: str = "opponent_bucket"
    latent_specialist_use_rollout_states: bool = False
    latent_specialist_rollout_max_samples: int = 8192
    late_entropy_floor: float = 0.0003
    commitment_type: str = "confidence_weighted_entropy"

    # v3i event refresh config parameters
    latent_event_refresh_enabled: bool = False
    latent_event_refresh_min_gap_steps: int = 20
    latent_event_refresh_max_per_episode: int = 3
    latent_event_refresh_use_q_phi: bool = True
    latent_event_refresh_force_roles: bool = False

    # v3i15 sparse tactical q_phi refresh. Disabled by default so all earlier
    # presets retain episode-persistent strategy execution.
    latent_sparse_tactical_refresh_enabled: bool = False
    latent_sparse_tactical_refresh_interval_steps: int = 32
    latent_sparse_tactical_refresh_min_dwell_steps: int = 16

    # v3i3 event-conditioned preference. Each event refresh becomes a
    # per-refresh learning datapoint with target bucket key
    # ``(opponent_id, event_type, flag_state_bucket)`` and credit signal
    # ``return_from_now_to_end_of_episode``. The teacher loss is a KL between
    # ``q_phi(z | state_at_refresh)`` and softmax over avg future-return per
    # z within the matching bucket. Hierarchical fallback when the
    # finest-grained bucket is undersampled:
    #     (opp, event, flag) -> (opp, event) -> (opp)
    # Plan-faithful: ``event_type`` is *context* (input only to the
    # teacher's bucket key + future-return credit), never a command. q_phi
    # still learns pi(z | state) -- no scripted z assignments.
    latent_v3i3_event_preference_enabled: bool = False
    latent_v3i3_event_preference_coef: float = 0.0
    latent_v3i3_event_preference_temperature: float = 0.75
    latent_v3i3_event_preference_min_bucket_count: int = 4
    latent_v3i3_event_preference_min_distinct_z: int = 2
    latent_v3i3_event_preference_buffer_size: int = 50_000
    # Number of global steps to wait before applying the v3i3 preference
    # loss. 0 = apply from the first rollout. Useful to let the buffer
    # accumulate evidence before the teacher fires.
    latent_v3i3_event_preference_warmup_steps: int = 0
    # Normalize event preference returns by subtracting the baseline for the specific event key.
    latent_v3i3_event_preference_normalize: bool = False
    # Key mode for event-conditioned preference distillation.
    #   "event_flag"          -- opponent_id, event_type, flag_state (v3i3 mode)
    #   "event_flag_progress" -- opponent_id, event_type, flag_state, carrier_progress (v3i4 mode)
    latent_event_preference_key_mode: str = "event_flag"
    # Per-refresh proof-layer log. One CSV row per finalized refresh event:
    #   env_id, episode_id, decision_step, reason, prev_z, next_z,
    #   opponent_id, flag_state_bucket, return_from_now_to_end
    # Independent of the preference loss -- enable the log alone for an
    # instrumentation-only run.
    latent_v3i3_refresh_log_enabled: bool = False
    latent_v3i3_refresh_log_path: Optional[str] = None

    # Episode-level domain randomization for sim robustness (sensor dropout/noise, blue speed jitter).
    # See ``GPUFieldConfig`` for numeric ranges; eval harnesses should keep this False.
    train_domain_randomization: bool = False
    dr_sensor_noise_sigma_max: float = 0.12
    dr_sensor_dropout_max: float = 0.08
    dr_blue_speed_jitter: float = 0.12

    # Optional overrides forwarded to ``GPUFieldConfig`` reward shaping (None = env defaults).
    # Useful for training-winrate recipes: stronger W/D contrast and less dense dilution of terminals.
    env_win_team_reward: Optional[float] = None
    env_draw_team_penalty: Optional[float] = None
    env_lose_team_punish: Optional[float] = None
    env_action_failed_punishment: Optional[float] = None
    # Points for tagging a non-carrying opponent (symmetric: earned on tag,
    # paid when tagged). Default +100 equals a flag capture; see GPUFieldConfig.
    env_sparse_tag_no_flag_points: Optional[float] = None
    # Points for tagging the enemy flag carrier (default +50).
    env_sparse_tag_with_flag_points: Optional[float] = None
    env_dense_weight: Optional[float] = None
    env_sparse_weight: Optional[float] = None
    env_reward_scale: Optional[float] = None
    env_reward_clip: Optional[float] = None
    env_stalemate_penalty: Optional[float] = None
    env_stalemate_max_steps: Optional[int] = None
    env_surface_score_margin_coef: Optional[float] = None
    env_surface_blue_capture_tempo_bonus: Optional[float] = None
    env_surface_red_flag_touch_penalty: Optional[float] = None
    env_surface_red_carrier_progress_penalty: Optional[float] = None
    env_surface_blue_near_cap_bonus: Optional[float] = None
    # Optional trainer-side reward shaping decay: scales (offense+pbrs+team) contribution seen by PPO.
    reward_shaping_coef_start: float = 1.0
    reward_shaping_coef_end: float = 1.0
    reward_shaping_decay_steps: int = 0
    periodic_checkpoint_steps: int = 50_000

    # --- v4i3: Periodic Return-Ranked Router Distillation during training ---
    # When ``latent_router_distill_enabled`` is True, the trainer spawns
    # ``tools/q_probe.py`` + ``tools/router_distill_from_qprobe.py`` as
    # subprocesses against the just-saved checkpoint every
    # ``latent_router_distill_every_n_steps`` global steps, then hot-swaps
    # the distilled ``strategy_encoder.*`` weights into the running model
    # and resets the corresponding Adam moments. PPO training is unchanged
    # outside this hook: actor, critic, reward, opponents, maps, arc-credit
    # math, entropy schedule, and the PPO loop itself are all untouched.
    # The whole hook is best-effort: any subprocess or hot-swap failure
    # is logged and PPO training continues with the pre-distill weights.
    latent_router_distill_enabled: bool = False
    latent_router_distill_every_n_steps: int = 250_000
    latent_router_distill_n_seeds: int = 8
    latent_router_distill_base_seed: int = 1000
    latent_router_distill_opponents: tuple[str, ...] = ("OP5", "OP6", "OP7")
    latent_router_distill_epochs: int = 100
    latent_router_distill_lr: float = 1e-4
    latent_router_distill_temperature: float = 1.0
    latent_router_distill_weight_decay: float = 0.0
    # The probe + distill runs run as separate subprocesses; ``cpu`` is the
    # safe default so the PPO GPU is not contended.
    latent_router_distill_device: str = "cpu"
    # Subdirectory under ``checkpoint_dir`` for the v4i4post artifact tree.
    # (This recipe was originally introduced as v4i3 but was rescoped: the
    # canonical v4i3 is now the Summer-Faithful Proof Suite, and periodic
    # router distillation lives in v4i4post as a post-Summer extension.)
    latent_router_distill_artifacts_subdir: str = "v4i4post_router_distill"

    # --- v6i6: evidence-gated repertoire Expansion Stage E1 ---
    use_v6i6_expansion: bool = False
    v6i6_require_validated_anchors: bool = True
    v6i6_anchor_validation_manifest: Optional[str] = None
    v6i6_expansion_protocol_version: str = ""
    v6i6_expansion_stage: str = ""
    v6i6_anchor_latents: Tuple[int, ...] = ()
    v6i6_target_latent: int = -1
    v6i6_dormant_latents: Tuple[int, ...] = ()
    v6i6_fixed_z_episode_attribution: bool = True
    v6i6_target_episode_fraction: float = 0.50
    v6i6_anchor_episode_fraction: float = 0.50
    v6i6_trainable_scope: str = ""
    v6i6_use_reference_critic_for_opportunity: bool = True
    v6i6_restore_masked_latent_rows_after_step: bool = True
    v6i6_assert_anchor_bitwise_invariant: bool = True
    v6i6_count_draw_as: float = 0.5
    v6i6_competence_rho: float = 0.85
    v6i6_competence_cfloor: float = 0.35
    v6i6_novelty_coef: float = 0.5
    v6i6_novelty_tau_d: float = 0.1
    # Stopping Gates
    v6i6_e1_max_steps: int = 400000
    v6i6_e1_min_target_episodes_per_opponent: int = 50
    v6i6_e1_min_effective_updates: int = 100
    v6i6_e1_min_competence_ratio: float = 0.85
    v6i6_e1_min_nearest_anchor_jsd: float = 0.02
    v6i6_e1_required_consecutive_checks: int = 5

    # --- V6I7: Summer-Faithful Recurrent Router ---
    # GRU hidden dimension; also used to determine q_phi input size (34 + hidden_dim).
    recurrent_selector_hidden_dim: int = 64
    # Sequence minibatch chunk lengths for truncated BPTT.
    recurrent_seq_len: int = 32
    recurrent_burn_in: int = 8
    # Number of independent sequence chunks per BPTT minibatch.
    router_chunks_per_batch: int = 4
    # Conditional entropy coefficient for router decisions (applied inside BPTT loop).
    router_ent_coef: float = 0.005
    # Entropy mode for the marginal coverage term: "marginal" | "conditional".
    h_mode: str = "marginal"
    # Counterfactual separation coefficient (set 0.0 to disable for V6I7).
    latent_cf_separation_coef: float = 0.0
    # Strategy decision interval; controls how often z is resampled in fixed-cadence mode.
    strategy_interval: int = 32

    # --- V6I7: Reward Separation ---
    # When False (default) the router uses the same shaped actor reward for credit assignment.
    # When True a separate sparse team-consequence reward signal is computed and stored in
    # buffer["router_reward"]; compute_router_returns() uses it instead of buffer["rewards"].
    router_reward_enabled: bool = False
    # Win/loss terminal bonus weight in the router reward.
    router_reward_win_weight: float = 1.0
    # Flag capture event weight (blue cap positive, red cap negative).
    router_reward_flag_cap_weight: float = 0.5
    # Sparse event weight applied to the full sparse_points component as a proxy for
    # carrier stops and tag events (reward_sparse = cfg.sparse_weight * sparse_points/100).
    router_reward_sparse_weight: float = 0.2
    # Scalar multiplier applied after the weighted sum before optional tanh normalisation.
    router_reward_scale: float = 1.0
    # When True applies tanh to keep the router reward in (-1, 1).
    router_reward_normalize: bool = True

    # --- V6I7: Forced-Latent Repertoire Training ---
    # "router"           -- GRU q_phi decides z (default, current V6I7-A behaviour).
    # "fixed"            -- All envs use forced_latent_id for the whole run.
    # "balanced_episode" -- Round-robin through K latents across episode starts.
    # "balanced_arc"     -- Within each episode switch z every forced_latent_arc_steps,
    #                       cycling through K in order.
    latent_assignment_mode: str = "router"
    # Latent ID used when latent_assignment_mode == "fixed".
    forced_latent_id: int = 0
    # Steps between z switches when latent_assignment_mode == "balanced_arc".
    forced_latent_arc_steps: int = 32
    # When True, continue updating the router's PPO objective even during forced episodes.
    train_router_when_forced: bool = False
    # When True, continue updating the router critic target even during forced episodes.
    train_router_critic_when_forced: bool = False
    # --- V6I14: Contract-specialist scaffold rewards ---
    # Default-off z-indexed behavioral contracts used to birth recognizable
    # specialists before routing. This is a post-Summer scaffold, not a
    # paper-faithful latent objective.
    latent_contract_specialist_enabled: bool = False
    latent_contract_specialist_coef: float = 0.0
    latent_contract_specialist_clip: float = 1.0
    latent_contract_specialist_variant: str = "base"

    # --- V6I26: Phase-pod exclusive birth (DIAGNOSTIC / Claim B) ---
    # When set to one of open_pressure|intercept|escort|defend_lead, the trainer
    # injects that strategic scenario after every env reset. Empty = disabled.
    # Not a paper-faithful channel; no z-role reward labels.
    phase_pod_id: str = ""

    # --- V6I9: Multi-Stage Training ---
    # Controls which parameter groups are frozen at optimizer build time.
    #   ""            / "generalist"  — all parameters trainable (Stage 1: map-aware competence)
    #   "repertoire"  — freeze CNN + shared actor trunk; train only z-specific modules + critic
    #   "router"      — freeze CNN + actor trunk + z-specific modules; train only router + critic
    #
    # Note: "router" stage is designed to be used with router_freeze_actor=True as well.
    v6i9_training_stage: str = ""

    # CLI convenience: when set via --additional-steps, train_ppo() resolves
    # total_timesteps = checkpoint_global_step + additional_timesteps after load.
    # Takes precedence over total_timesteps when nonzero.
    additional_timesteps: int = 0
    # Global step at which the current run/stage started (set on resume).
    checkpoint_run_start_step: int = 0


__all__ = ["PPOConfig", "TrainMode"]
