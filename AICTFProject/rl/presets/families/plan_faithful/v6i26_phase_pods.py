"""V6I26: Latent Response-Oracle Summer (LRO-Summer) — Claim B breakthrough.

Classification: DIAGNOSTIC (not PAPER-FAITHFUL).
Parent: v6i23_population_birth (deep per-z capacity) with contract OFF.

Scientific delta (plain English)
--------------------------------
Stop asking four symmetric latent branches to invent different strategies under
the same PPO mixture. Treat each z as an internal response-oracle policy and
train it specifically against uncovered weaknesses of the current latent
population (PSRO / VGC-Bench / Conflux-PSRO lesson).

No human strategy labels. No attack/defense rewards. Task-return payoff and
population regret drive which branch is updated.
"""
from __future__ import annotations

from rl.config.ppo_config import PPOConfig, TrainMode
from rl.presets.families.plan_faithful.v6_router_adapters import (
    apply_plan_faithful_latent_v6i23_population_birth,
)


def apply_v6i26_latent_response_oracle(cfg: PPOConfig) -> PPOConfig:
    """V6I26 LRO-Summer shared training surface.

    ## Proposed Preset Review

    ### Identity
    - Proposed name: v6i26_latent_response_oracle
    - Parent preset: v6i23_population_birth
    - Classification: DIAGNOSTIC
    - Research question: Can response-oracle training of deep per-z branches
      create G_available > 0 without strategy-specific reward labels?

    ### Intended delta vs v6i23
    - latent_contract_specialist_enabled/coef → False/0
    - latent_lro_deep_branches → True (last-two-layer trunks per z)
    - latent_lro_active_branch_only → True
    - freeze shared encoder during birth (v6i9_training_stage=repertoire)
    - router OFF; fixed z per episode during birth rounds
    - experiment_id/run_tag → v6i26 LRO

    ### Fidelity impact
    - Actor architecture changed: YES (deep per-z trunks; still concat embedding)
    - Router objective changed: NO (router disabled in birth)
    - Reward changed: YES (contract OFF; task reward only)
    - Supervision added: NO
    """
    cfg = apply_plan_faithful_latent_v6i23_population_birth(cfg)

    # Task reward only — no shared contract glue.
    cfg.latent_contract_specialist_enabled = False
    cfg.latent_contract_specialist_coef = 0.0

    # LRO capacity: deep per-z trunks + active-branch-only updates.
    cfg.enable_latent_z_residual = True
    cfg.latent_z_residual_alpha = 0.1
    cfg.latent_population_birth_active_z_only = True
    cfg.latent_population_birth_per_z_action_heads = True
    cfg.latent_lro_deep_branches = True
    cfg.latent_lro_active_branch_only = True

    # Birth: freeze shared perception; train z branches + critic.
    cfg.v6i9_training_stage = "repertoire"
    cfg.train_router_when_forced = False
    cfg.train_router_critic_when_forced = False
    cfg.latent_strategy_ppo_coef = 0.0
    cfg.latent_episode_strategy_ppo = False
    cfg.latent_episode_strategy_coef = 0.0
    cfg.recurrent_selector_hidden_dim = 0

    # Episode-fixed z during birth (branch selected by the oracle runner).
    cfg.fixed_latent_strategy = True
    cfg.fixed_latent_strategy_id = 0  # runner overrides per round
    cfg.latent_assignment_mode = "fixed"

    cfg.mode = TrainMode.OPPONENT_POOL.value
    cfg.opponent_randomize = True
    cfg.freeze_return_norm_after_load = True
    cfg.phase_pod_id = ""  # optional; LRO mixtures are payoff-driven

    cfg.experiment_id = "v6i26"
    cfg.run_tag = "v6i26_latent_response_oracle_OP8_OP9_OP10_OP11_OP12"
    return cfg


# Back-compat alias used by earlier phase-pod scaffolding.
apply_v6i26_phase_pod_population = apply_v6i26_latent_response_oracle
