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
    cfg.latent_episode_strategy_coef = 0.25
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
