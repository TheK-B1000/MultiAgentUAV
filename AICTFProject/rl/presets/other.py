"""Other specialized and performance-tuning presets (wrmax, push80, A1 comparison, etc.)."""

from __future__ import annotations

from rl.train_ppo import PPOConfig, TrainMode, _resolve_2v2_checkpoint
from rl.presets.plan_faithful import apply_latent_a1_plan_faithful


def apply_latent_op3_push80_1m(cfg: PPOConfig) -> PPOConfig:
    # Tuned for faster 2v2 OP3 gains by 1M steps:
    # - sharpen latent strategy usage (lower strategy entropy, stronger persistence)
    # - increase on-policy update pressure per rollout while keeping PPO stable
    cfg.use_latent_strategy = True
    cfg.total_timesteps = 1_000_000
    cfg.mode = TrainMode.FIXED_OPPONENT.value
    cfg.fixed_opponent_tag = "OP3"
    cfg.normalize_returns = True
    cfg.clip_range_vf = 0.15
    cfg.vf_coef = 0.8
    cfg.learning_rate = 3e-4
    cfg.lr_floor_frac = 0.05
    cfg.target_kl = 0.03
    cfg.n_steps = 2048
    cfg.batch_size = 512
    cfg.n_epochs = 8
    cfg.ent_coef = 0.003
    cfg.latent_entropy_objective = "minimize"
    cfg.latent_lam_h = 0.01
    cfg.latent_lam_p = 0.04
    cfg.latent_strategy_ppo_coef = 0.2
    cfg.latent_resample_every_n = 0
    cfg.latent_resample_on_flag = False
    cfg.latent_kl_consecutive = 0.0
    cfg.latent_gae_reset_on_z_change = True
    cfg.latent_bootstrap_z_deterministic = True
    warm = _resolve_2v2_checkpoint("final_latent_fix_v4_retnorm_vf256_1m_2v2.zip")
    if warm is not None:
        cfg.load_path = warm
    cfg.run_tag = "latent_op3_push80_1m_2v2"
    return cfg


def apply_latent_train80_op3_1m(cfg: PPOConfig) -> PPOConfig:
    # Training-winrate-first profile (2v2, fixed OP3):
    # prioritize fast in-run WR improvements over broad exploration.
    cfg.use_latent_strategy = True
    cfg.total_timesteps = 1_000_000
    cfg.mode = TrainMode.FIXED_OPPONENT.value
    cfg.fixed_opponent_tag = "OP3"
    cfg.normalize_returns = True
    cfg.clip_range_vf = 0.12
    cfg.vf_coef = 0.7
    cfg.learning_rate = 2.5e-4
    cfg.lr_floor_frac = 0.05
    cfg.target_kl = 0.025
    cfg.n_steps = 2048
    cfg.batch_size = 512
    cfg.n_epochs = 10
    cfg.ent_coef = 0.001
    cfg.latent_entropy_objective = "minimize"
    cfg.latent_lam_h = 0.02
    cfg.latent_lam_p = 0.06
    cfg.latent_strategy_ppo_coef = 0.30
    cfg.latent_resample_every_n = 0
    cfg.latent_resample_on_flag = False
    cfg.latent_kl_consecutive = 0.0
    cfg.latent_gae_reset_on_z_change = True
    cfg.latent_bootstrap_z_deterministic = True
    # Auxiliary return head gives q_phi a direct supervised signal from sampled-z returns
    cfg.latent_strategy_aux_return_head = True
    cfg.latent_strategy_aux_return_coef = 0.75
    cfg.latent_strategy_tau = 1.0
    warm = _resolve_2v2_checkpoint("final_latent_fix_v4_retnorm_vf256_1m_2v2.zip")
    if warm is not None:
        cfg.load_path = warm
    cfg.run_tag = "latent_train80_op3_1m_2v2"
    return cfg


def apply_latent_op3_wrmax_1m(cfg: PPOConfig) -> PPOConfig:
    # High-WR "drift" recipe vs OP3 (~89.7% reference run): strong terminal contrast, trimmed dense PBRS,
    # reward-shaping decay, auxiliary return head, VF hidden 256. 1M default; aliases treat wrmax_2m as 1M.
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
    cfg.latent_entropy_objective = "minimize"
    cfg.latent_lam_h = 0.02
    cfg.latent_lam_p = 0.0
    cfg.latent_strategy_ppo_coef = 0.30
    cfg.latent_strategy_aux_return_head = True
    cfg.latent_strategy_aux_return_coef = 1.2
    cfg.latent_strategy_tau = 0.7
    cfg.latent_resample_every_n = 0
    cfg.latent_resample_on_flag = False
    cfg.latent_kl_consecutive = 0.0
    cfg.latent_gae_reset_on_z_change = True
    cfg.latent_bootstrap_z_deterministic = True
    cfg.latent_vf_hidden = 256
    cfg.env_win_team_reward = 1.5
    cfg.env_lose_team_punish = -1.2
    cfg.env_draw_team_penalty = -0.7
    cfg.env_action_failed_punishment = -0.02
    cfg.env_dense_weight = 0.08
    cfg.env_sparse_weight = 1.0
    cfg.env_reward_scale = 4.5
    cfg.env_stalemate_penalty = -0.08
    cfg.env_stalemate_max_steps = 120
    cfg.reward_shaping_coef_start = 1.0
    cfg.reward_shaping_coef_end = 0.3
    cfg.reward_shaping_decay_steps = 500_000
    cfg.periodic_checkpoint_steps = 50_000
    warm = _resolve_2v2_checkpoint("final_latent_fix_v4_retnorm_vf256_1m_2v2.zip")
    if warm is not None:
        cfg.load_path = warm
    cfg.run_tag = "latent_op3_wrmax_1m_2v2"
    return cfg


def apply_latent_op3_wrmax_train_2m(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_latent_op3_wrmax_1m(cfg)
    cfg.total_timesteps = 2_000_000
    cfg.run_tag = "latent_op3_wrmax_train_2m_2v2"
    return cfg
