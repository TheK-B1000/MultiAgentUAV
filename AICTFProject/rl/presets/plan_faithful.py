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
