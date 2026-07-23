"""V6I26: Phase-pod exclusive full-policy population (Claim B breakthrough).

Classification: DIAGNOSTIC (not PAPER-FAITHFUL).
Parent: v6i24_full_policy_population.
Delta: contract specialist OFF; phase_pod_id set per member by the runner;
exclusive scenario injection after reset (VGC single-matchup analogue).
"""
from __future__ import annotations

from rl.config.ppo_config import PPOConfig
from rl.presets.families.plan_faithful.v6i24_population import (
    apply_v6i24_full_policy_population,
)


def apply_v6i26_phase_pod_population(cfg: PPOConfig) -> PPOConfig:
    """V6I26 shared surface for exclusive phase-pod teachers.

    ## Proposed Preset Review

    ### Identity
    - Proposed name: v6i26_phase_pod_population
    - Parent preset: v6i24_full_policy_population
    - Classification: DIAGNOSTIC
    - Research question: Can exclusive phase-pod scenario teachers create
      competent policies with stable comparative advantages (Claim B)?

    ### Intended delta
    - latent_contract_specialist_enabled: True -> False
    - latent_contract_specialist_coef: (inherited) -> 0.0
    - phase_pod_id: "" -> set by runner per member
    - experiment_id / run_tag updated

    ### Fidelity impact
    - Actor architecture changed: NO
    - Router objective changed: NO
    - Reward changed: YES (contract OFF; no z-role labels)
    - Supervision added: NO (scenario geometry only)
    """
    cfg = apply_v6i24_full_policy_population(cfg)

    # Remove the shared z=0 contract glue that confounded V6I24 soft Path C.
    cfg.latent_contract_specialist_enabled = False
    cfg.latent_contract_specialist_coef = 0.0
    cfg.latent_contract_specialist_clip = 1.0
    cfg.latent_contract_specialist_variant = "base"

    # Runner sets phase_pod_id per member; default empty keeps hooks inactive.
    cfg.phase_pod_id = ""

    # Uniform OP×map cell pool; niche comes from scenario injection, not soft mixtures.
    cfg.training_cell_distribution = ()

    cfg.experiment_id = "v6i26"
    cfg.run_tag = "v6i26_phase_pod_population_OP8_OP9_OP10_OP11_OP12"
    return cfg
