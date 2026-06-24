"""Reusable PPO/latent configs aligned with the Summer Implementation Plan and paper ablations.

**Main plan-faithful first run:** K=4, sparse strategy interval 20, persistence + entropy,
no supervised labels, no opponent-ID heads, no auxiliary prediction heads.

Legacy A1 configs below still sample ``z`` once per episode for old comparisons.

**E3 baseline (fair comparison):** :func:`paper_default_no_latent_config` is
``replace(paper_default_latent_config(), use_latent_strategy=False)`` so the
Summer default opponent/training setup is held fixed while the latent path is off.

Professor-requested baselines are exposed below: curriculum and no-latent PPO.
The no-persistence ablation is retained as an optional new-method ablation.

Use a separate run / config when ablating flag-triggered resampling (plan §12).
"""

from __future__ import annotations

from dataclasses import replace

from rl.train_ppo import PPOConfig, TrainMode


def plan_faithful_latent_persist_entropy_config() -> PPOConfig:
    """Recommended Summer-plan first run: K=4, sparse z interval 20, persistence + entropy, no labels."""
    return PPOConfig(
        use_latent_strategy=True,
        latent_k=4,
        latent_resample_every_n=20,
        latent_resample_on_flag=False,
        latent_kl_consecutive=0.0,
        latent_lam_p=0.025,
        latent_lam_h=0.003,
        latent_entropy_objective="maximize",
        latent_strategy_aux_return_head=False,
        latent_strategy_aux_return_coef=0.0,
    )


def plan_faithful_latent_no_persistence_config() -> PPOConfig:
    return replace(plan_faithful_latent_persist_entropy_config(), latent_lam_p=0.0)


def plan_faithful_latent_no_entropy_config() -> PPOConfig:
    return replace(
        plan_faithful_latent_persist_entropy_config(),
        latent_lam_h=0.0,
        latent_entropy_objective="none",
    )


def plan_faithful_latent_k1_config() -> PPOConfig:
    return replace(plan_faithful_latent_persist_entropy_config(), latent_k=1)


def plan_faithful_no_latent_config() -> PPOConfig:
    return replace(plan_faithful_latent_persist_entropy_config(), use_latent_strategy=False)


def plan_faithful_latent_option_a_config() -> PPOConfig:
    """Fix D / Plan Option A: ``z`` sampled once per episode, λ_p disabled, λ_H at the bottom of the plan range.

    This is the cleanest plan-faithful contrast against
    :func:`plan_faithful_no_latent_config` — only ``use_latent_strategy`` and the
    latent hyperparameters differ. Mirrors the
    ``plan_faithful_latent_option_a`` preset in :mod:`rl.train_ppo`.
    """
    return replace(
        plan_faithful_latent_persist_entropy_config(),
        latent_resample_every_n=0,
        latent_lam_p=0.0,
        latent_lam_h=0.001,
        latent_entropy_objective="maximize",
        latent_strategy_aux_predict_phase_coef=0.0,
        latent_strategy_aux_return_head=False,
        latent_strategy_aux_return_coef=0.0,
        latent_resample_on_flag=False,
        latent_kl_consecutive=0.0,
        fixed_latent_strategy=False,
    )


def paper_default_latent_config() -> PPOConfig:
    """Primary paper table: latent on; episode-start ``z`` only; no optional §12 resampling in the run."""
    return PPOConfig(
        use_latent_strategy=True,
        latent_resample_every_n=0,
        latent_resample_on_flag=False,
        latent_kl_consecutive=0.0,
    )


def paper_default_no_latent_config() -> PPOConfig:
    """E3 / baseline: **single** field flipped vs :func:`paper_default_latent_config` — ``use_latent_strategy=False``."""
    return replace(paper_default_latent_config(), use_latent_strategy=False)


def flat_ppo_marl_baseline_config() -> PPOConfig:
    """Backward-compatible alias for the fixed-OP3 no-latent baseline."""
    return paper_default_no_latent_config()


def curriculum_baseline_config() -> PPOConfig:
    """Professor-requested baseline: Jacob-style OP1->OP2->OP3 curriculum with latent strategy off."""
    return replace(
        paper_default_latent_config(),
        mode=TrainMode.CURRICULUM.value,
        use_latent_strategy=False,
    )


def no_latent_baseline_config() -> PPOConfig:
    """Professor-requested baseline: no-latent PPO under the Summer default fixed-OP3 setting."""
    return paper_default_no_latent_config()


def fixed_opponent_no_latent_config() -> PPOConfig:
    """Backward-compatible alias for :func:`no_latent_baseline_config`."""
    return no_latent_baseline_config()


def jacob_original_baseline_config() -> PPOConfig:
    """Legacy Jacob-style control: OP1->OP2->OP3 curriculum with latent strategy disabled."""
    return curriculum_baseline_config()


def latent_no_persistence_baseline_config(*, resample_every: int = 20) -> PPOConfig:
    """Optional new-method ablation: sparse strategy refresh without persistence penalty."""
    return replace(
        paper_default_latent_config(),
        latent_resample_every_n=max(2, int(resample_every)),
        latent_lam_p=0.0,
    )


def fixed_latent_baseline_config(*, strategy_id: int = 0) -> PPOConfig:
    """Older optional ablation: latent actor/critic receive one fixed z ID for the whole run."""
    return replace(
        paper_default_latent_config(),
        fixed_latent_strategy=True,
        fixed_latent_strategy_id=max(0, int(strategy_id)),
    )


def ablation_flag_resample_config(*, base: PPOConfig | None = None) -> PPOConfig:
    """Ablate optional plan §12: resample when global flag/territory slice changes (keep all else from ``base``)."""
    c = base if base is not None else PPOConfig()
    return replace(c, latent_resample_on_flag=True)


# ---------------------------------------------------------------------------
# V6I7: Summer-Faithful Recurrent Router
# ---------------------------------------------------------------------------

def v6i7_recurrent_router_config() -> PPOConfig:
    """V6I7-A: GRU router with BPTT and decision-only conditional entropy.

    Locked decisions (see v6i7_plan.md):
    - ``router_context_mode="current"`` disables the EMA stack; q_phi receives
      the raw 35-dimensional state (34 global features plus scheduler phase)
      concatenated with the 64-dimensional GRU hidden state.
    - ``use_recurrent_selector=True`` activates the per-step GRU and BPTT path.
    - Fixed sparse cadence (``latent_resample_every_n=32``); no event-driven
      resampling.
    - Router PPO coefficient 0.10, persistence coefficient 0.02, and
      conditional router entropy coefficient 0.005 at valid decisions only.
    - Legacy marginal-coverage entropy is disabled.
    - All supervised auxiliary heads are disabled.
    """
    return PPOConfig(
        use_latent_strategy=True,
        latent_k=4,

        # GRU router and BPTT
        router_context_mode="current",
        recurrent_selector_hidden_dim=64,
        recurrent_seq_len=32,
        recurrent_burn_in=8,
        router_chunks_per_batch=4,

        # Decision-only conditional router entropy
        router_ent_coef=0.005,
        h_mode="conditional",

        # Fixed sparse strategy cadence
        latent_resample_every_n=32,
        latent_resample_on_flag=False,
        latent_event_refresh_enabled=False,
        latent_sparse_tactical_refresh_enabled=False,

        # Router objective
        latent_strategy_ppo_coef=0.10,
        latent_lam_p=0.02,

        # Disable legacy marginal entropy / coverage objective
        latent_lam_h=0.0,
        latent_entropy_objective="none",

        # Disable supervised and auxiliary objectives
        latent_strategy_aux_return_head=False,
        latent_strategy_aux_return_coef=0.0,
        latent_strategy_aux_predict_phase_coef=0.0,
        latent_cf_separation_coef=0.0,
        latent_kl_consecutive=0.0,

        # Fixed latent disabled
        fixed_latent_strategy=False,
    )

def v6i7_b0_mlp_baseline_config() -> PPOConfig:
    """V6I7-B0: MLP router (non-recurrent) with conditional entropy — BPTT ablation baseline."""
    return replace(
        v6i7_recurrent_router_config(),
        router_context_mode="",   # revert to EMA context
        recurrent_selector_hidden_dim=0,
        recurrent_seq_len=0,
        recurrent_burn_in=0,
        h_mode="conditional",
        latent_lam_h=0.0,
        router_ent_coef=0.005,
    )


def v6i7_b1_gru_conditional_config() -> PPOConfig:
    """V6I7-B1: GRU router with conditional entropy (no marginal coverage) — entropy-mode ablation."""
    return replace(
        v6i7_recurrent_router_config(),
        h_mode="conditional",
        latent_lam_h=0.0,
    )
