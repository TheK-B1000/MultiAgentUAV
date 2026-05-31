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
