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
from typing import Literal, Optional

import torch


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
    n_envs: int = 8
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
    run_tag: str = "ppo_latent_2v2"
    enable_metrics_csv: bool = True
    metrics_csv_path: Optional[str] = None
    episode_csv_path: Optional[str] = None
    strategy_experience_csv_path: Optional[str] = None
    # If True before training, existing non-empty metrics/episode CSVs are rotated aside so a new run
    # does not append duplicate timesteps under the same --run-tag.
    fresh_metrics_csv: bool = False
    # Set from CLI ``--preset`` only (reproducibility / run_config.json); behavior is already merged into fields below.
    cli_preset: Optional[str] = None
    # E3: optional per-step CSV (z, H(q), argmax, switch, phase). See `rl.custom_ppo.E3_STEP_TELEMETRY_FIELDS`.
    e3_step_telemetry_path: Optional[str] = None
    # SB3-compatible: ``tqdm`` (prefer ``tqdm.rich``) during rollout, ``total=remaining`` timesteps, ``update(n_envs)`` / step.
    enable_progress_bar: bool = True
    verbose_training: bool = False
    # After this many *completed* episodes, print W/L/D and win rate (0 = disabled).
    episode_log_every: int = 1000

    max_decision_steps: int = 400
    map_set: str = "train"
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
    allow_op4_in_training_pool: bool = False
    max_blue_agents: int = 2
    use_deterministic: bool = False
    # Not in *Summer Implementation Plan.docx*; when True, overrides several PPO fields below for a legacy "stable" profile. Default False so explicit config matches the spec numbers.
    use_stable_marl_ppo: bool = False
    target_kl: Optional[float] = 0.02
    actor_cnn_feature_dim: int = 128
    actor_hidden_dim: int = 256

    # Summer/ICRA latent team strategy is the default proposed algorithm.
    use_latent_strategy: bool = True
    latent_k: int = 4
    latent_z_embed_dim: int = 16
    latent_actor_conditioning: Literal["concat"] = "concat"
    latent_vf_hidden: int = 128
    latent_strategy_hidden: int = 128
    # Plan IMPLEMENTATION §6: typical λ_H ∈ [0.001, 0.01]; λ_p ∈ [0.01, 0.05] (see also §3.3 for a wider λ_p range).
    # ``maximize`` matches the plan (encourage exploratory / diverse q_phi). ``minimize`` adds +λ_H·H to the
    # minimized loss and sharpens q_phi (recommended when telemetry shows strategy_entropy≈ln K with no persistence grad).
    # ``none`` removes the H term (strategy_encoder receives no gradient from λ_H when λ_p/KL are also inactive).
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
    # parameters; the shared optimizer's step on those params is a no-op anyway
    # under Fix 5 (``latent_strategy_ppo_coef == 0`` gates main-loop q_phi loss
    # to zero), so there is no double-stepping.
    latent_episode_strategy_lr: Optional[float] = None
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
    latent_behavior_contrast_coef: float = 0.0
    latent_behavior_contrast_margin: float = 0.25
    latent_behavior_contrast_ema: float = 0.9
    latent_behavior_contrast_anneal_after_steps: int = 0
    latent_behavior_contrast_anneal_to: float = 0.0
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
    # v3i9: balanced specialist router. Keeps the marginal q_phi usage
    # distribution high-entropy across the batch while reducing conditional
    # entropy inside opponent/context buckets. No role labels or scripted
    # z meanings; buckets shape the router objective and telemetry only.
    latent_specialist_router_enabled: bool = False
    latent_marginal_balance_coef: float = 0.0
    latent_conditional_entropy_min_coef: float = 0.0
    latent_context_mi_coef: float = 0.0
    latent_specialist_warmup_steps: int = 0
    latent_specialist_ramp_steps: int = 1
    latent_specialist_min_bucket_count: int = 2
    late_entropy_floor: float = 0.0003
    commitment_type: str = "confidence_weighted_entropy"

    # v3i event refresh config parameters
    latent_event_refresh_enabled: bool = False
    latent_event_refresh_min_gap_steps: int = 20
    latent_event_refresh_max_per_episode: int = 3
    latent_event_refresh_use_q_phi: bool = True
    latent_event_refresh_force_roles: bool = False

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
    env_dense_weight: Optional[float] = None
    env_sparse_weight: Optional[float] = None
    env_reward_scale: Optional[float] = None
    env_reward_clip: Optional[float] = None
    env_stalemate_penalty: Optional[float] = None
    env_stalemate_max_steps: Optional[int] = None
    # Optional trainer-side reward shaping decay: scales (offense+pbrs+team) contribution seen by PPO.
    reward_shaping_coef_start: float = 1.0
    reward_shaping_coef_end: float = 1.0
    reward_shaping_decay_steps: int = 0
    periodic_checkpoint_steps: int = 50_000


__all__ = ["PPOConfig", "TrainMode"]
