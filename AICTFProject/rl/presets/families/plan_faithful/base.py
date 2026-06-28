"""Plan-faithful base presets — shared setup and foundational latent variants."""
from __future__ import annotations

from rl.config.ppo_config import PPOConfig, TrainMode


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
