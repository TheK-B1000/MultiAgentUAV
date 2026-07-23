"""V6I24: Full-policy population diagnostic preset (lean Path C fallback).

Classification: DIAGNOSTIC (not PAPER-FAITHFUL, not SUMMER-COMPATIBLE EXTENSION).
Parent: v6i21j_hardpool_balance_calibration (hardpool surface only).
Training: ordinary independent PPO runs (no PopulationTrainer / PFSP / rotation).

Architecture note
-----------------
The V6I21J-competent clone source is a latent-concat actor (``latent_k=4``).
Turning ``use_latent_strategy`` off would change body input dims and break
warm-start. V6I24 therefore keeps the latent *scaffold* but freezes ``z=0``,
disables adapters/router/strategy losses, and trains each member as an
independent full actor-critic under distinct cell pressures.
"""
from __future__ import annotations

from rl.config.ppo_config import PPOConfig, TrainMode
from rl.presets.families.plan_faithful.v6_router_adapters import (
    apply_plan_faithful_latent_v6i21j_hardpool_balance_calibration,
)


def apply_v6i24_full_policy_population(cfg: PPOConfig) -> PPOConfig:
    """V6I24 shared training surface for independent full-policy teachers.

    Proposed Preset Review
    ----------------------
    Proposed name: v6i24_full_policy_population.
    Parent preset: v6i21j_hardpool_balance_calibration.
    Classification: DIAGNOSTIC.

    Research question: can K=4 fully independent actor-critic policies, cloned
    from the same V6I21J-competent checkpoint and trained under *fixed*
    opponent×map cell pressures (no PFSP, no rotation), produce a real
    functional repertoire?

    Scientific delta: Path C fallback — abandon shared-trunk multi-z
    specialization. Each member is an ordinary independent PPO run (frozen
    z=0 scaffold for checkpoint compatibility, adapters off, router off,
    strategy losses zeroed, generalist trainable scope, frozen return-norm
    after load). Member-specific cell distributions are applied by the
    experiment runner, not by this preset.

    Budget / gates are owned by experiments/run_v6i24_full_policy_population.py
    (probes at 5u/10u/25u). Distillation is deferred to V6I24-D after pass.
    """
    cfg = apply_plan_faithful_latent_v6i21j_hardpool_balance_calibration(cfg)

    # Keep latent_k concat scaffold for V6I9/V6I21J warm-start compatibility,
    # but freeze z=0 and remove specialization machinery.
    cfg.use_latent_strategy = True
    cfg.fixed_latent_strategy = True
    cfg.fixed_latent_strategy_id = 0
    cfg.latent_assignment_mode = "fixed"
    cfg.enable_latent_z_residual = False
    cfg.latent_z_residual_alpha = 0.0
    cfg.latent_population_birth_active_z_only = False
    cfg.latent_population_birth_per_z_action_heads = False
    cfg.v6i9_training_stage = "generalist"
    cfg.train_router_when_forced = False
    cfg.train_router_critic_when_forced = False

    # No q_phi / strategy learning signals on the Path C teachers.
    cfg.latent_strategy_ppo_coef = 0.0
    cfg.latent_episode_strategy_ppo = False
    cfg.latent_episode_strategy_coef = 0.0
    if hasattr(cfg, "latent_lam_h_start"):
        cfg.latent_lam_h_start = 0.0
    if hasattr(cfg, "latent_lam_h_end"):
        cfg.latent_lam_h_end = 0.0
    if hasattr(cfg, "latent_lam_p_start"):
        cfg.latent_lam_p_start = 0.0
    if hasattr(cfg, "latent_lam_p_end"):
        cfg.latent_lam_p_end = 0.0

    # Lean diagnostic: do NOT use PopulationTrainer / pressure rotation.
    cfg.population_training_enabled = False
    cfg.population_k = 4
    cfg.population_pressure_rotation_interval = 0
    cfg.population_round_robin_updates_per_cycle = 0

    cfg.mode = TrainMode.OPPONENT_POOL.value
    cfg.opponent_randomize = True
    cfg.freeze_return_norm_after_load = True

    cfg.experiment_id = "v6i24"
    cfg.run_tag = "v6i24_full_policy_population_OP8_OP9_OP10_OP11_OP12"
    return cfg
