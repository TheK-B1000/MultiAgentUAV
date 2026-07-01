"""Hypothesis training presets with opponent randomization and custom coefficient tuning."""

from __future__ import annotations

from rl.train_ppo import PPOConfig, TrainMode
from rl.presets.plan_faithful import apply_plan_option_a, apply_plan_option_b_lamp


def apply_hypothesis_flat_opprand(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_plan_option_a(cfg)
    cfg.use_latent_strategy = False
    cfg.fixed_latent_strategy = False
    cfg.mode = TrainMode.OPPONENT_POOL.value
    cfg.opponent_randomize = True
    cfg.opponent_pool = ("OP1", "OP2", "OP3")
    cfg.run_tag = "hypothesis_flat_opprand_1m_2v2"
    return cfg


def apply_hypothesis_latent_opprand_optiona(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_plan_option_a(cfg)
    cfg.mode = TrainMode.OPPONENT_POOL.value
    cfg.opponent_randomize = True
    cfg.opponent_pool = ("OP1", "OP2", "OP3")
    cfg.run_tag = "hypothesis_latent_opprand_optiona_1m_2v2"
    return cfg


def apply_hypothesis_latent_opprand_optionb_lamp_coef05(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_plan_option_b_lamp(cfg)
    cfg.mode = TrainMode.OPPONENT_POOL.value
    cfg.opponent_randomize = True
    cfg.opponent_pool = ("OP1", "OP2", "OP3")
    cfg.latent_strategy_ppo_coef = 0.5
    cfg.run_tag = "hypothesis_latent_opprand_optionb_lamp_coef05_1m_2v2"
    return cfg


def apply_hypothesis_latent_opprand_optionb_no_lamp(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_plan_option_b_lamp(cfg)
    cfg.mode = TrainMode.OPPONENT_POOL.value
    cfg.opponent_randomize = True
    cfg.opponent_pool = ("OP1", "OP2", "OP3")
    cfg.latent_lam_p = 0.0
    cfg.latent_strategy_ppo_coef = 0.5
    cfg.run_tag = "hypothesis_latent_opprand_optionb_no_lamp_coef05_1m_2v2"
    return cfg


def apply_hypothesis_latent_opprand_optionb_coef03(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_plan_option_b_lamp(cfg)
    cfg.mode = TrainMode.OPPONENT_POOL.value
    cfg.opponent_randomize = True
    cfg.opponent_pool = ("OP1", "OP2", "OP3")
    cfg.latent_strategy_ppo_coef = 0.3
    cfg.run_tag = "hypothesis_latent_opprand_optionb_lamp_coef03_1m_2v2"
    return cfg


def apply_hypothesis_flat_opprand_op35(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_hypothesis_flat_opprand(cfg)
    cfg.opponent_pool = ("OP3", "OP5_RUSHER")
    cfg.run_tag = "hypothesis_flat_opprand_op35_1m_2v2"
    return cfg


def apply_hypothesis_latent_opprand_optionb_lamp_coef05_op35(cfg: PPOConfig) -> PPOConfig:
    cfg = apply_hypothesis_latent_opprand_optionb_lamp_coef05(cfg)
    cfg.opponent_pool = ("OP3", "OP5_RUSHER")
    cfg.run_tag = "hypothesis_latent_opprand_optionb_lamp_coef05_op35_1m_2v2"
    return cfg
