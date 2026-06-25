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
    """V6I7-B0: state-only MLP router under the repaired V6I7 pipeline."""
    return replace(
        v6i7_recurrent_router_config(),

        # Keep the same raw current-state input used by B1.
        router_context_mode="current",

        # Disable recurrence only.
        recurrent_selector_hidden_dim=0,
        recurrent_seq_len=0,
        recurrent_burn_in=0,
        router_chunks_per_batch=0,

        h_mode="conditional",
        latent_lam_h=0.0,
        latent_entropy_objective="none",
        router_ent_coef=0.005,
    )


def v6i7_b1_gru_conditional_config() -> PPOConfig:
    """V6I7-B1: GRU router with conditional entropy (no marginal coverage) — entropy-mode ablation."""
    return replace(
        v6i7_recurrent_router_config(),
        h_mode="conditional",
        latent_lam_h=0.0,
    )


# ---------------------------------------------------------------------------
# V6I7: Reward Separation and Forced-Latent Repertoire Training
# ---------------------------------------------------------------------------

def v6i7_sparse_router_config() -> PPOConfig:
    """V6I7 with separate sparse router reward (team wins + flag caps only).

    The actor still receives the full shaped reward.  The router's credit
    assignment uses a sparse signal composed of terminal outcomes and flag
    events to reduce dense-reward noise in the routing objective.
    """
    return replace(
        v6i7_recurrent_router_config(),
        router_reward_enabled=True,
        router_reward_win_weight=1.0,
        router_reward_flag_cap_weight=0.5,
        router_reward_sparse_weight=0.2,
        router_reward_scale=1.0,
        router_reward_normalize=True,
    )


def v6i7_repertoire_balanced_episode_config() -> PPOConfig:
    """V6I7 with balanced-episode forced-latent allocation.

    Each episode start round-robins through all K latents so every strategy
    receives equal actor-gradient coverage across the training run.  The
    router is still trained via PPO (``train_router_when_forced=False`` keeps
    forced episodes out of the router's PPO batch, matching V6 behaviour).
    """
    return replace(
        v6i7_recurrent_router_config(),
        latent_assignment_mode="balanced_episode",
        train_router_when_forced=False,
        train_router_critic_when_forced=False,
    )


def v6i7_router_critic_warmup_config() -> PPOConfig:
    """V6I7 with sparse router reward and balanced-episode warmup.

    Intended as a two-phase recipe:
    1. Train actor coverage first (balanced_episode, no router PPO).
    2. Switch to latent_assignment_mode='router' after coverage is confirmed.

    Use ``replace(v6i7_router_critic_warmup_config(), latent_assignment_mode='router')``
    to transition to the second phase.
    """
    return replace(
        v6i7_sparse_router_config(),
        latent_assignment_mode="balanced_episode",
        train_router_when_forced=False,
        train_router_critic_when_forced=True,
    )


# ---------------------------------------------------------------------------
# V6I8: Residual Adapter Actor Conditioning
#
# V6I7 = embedding-concatenation only (frozen baseline for comparison).
# V6I8 = V6I7 + per-latent residual adapters on actor trunk.
#
# Conditioning: h_z = h + g_z * A_z(h);  logits_z = W(h_z) + B_z
# A_z is Linear(256,256) zero-initialized → A_z(h)=0 at construction,
# so V6I8 loads any V6I7 checkpoint with exact behavioral equivalence.
# Gates init to 0.01 (active but weight-zero); biases zero-init.
# ---------------------------------------------------------------------------

def v6i8_adapter_balanced_config() -> PPOConfig:
    """V6I8-balanced: residual adapters + balanced-episode forced-latent warmup.

    First V6I8 experiment.  The router is held out of PPO during forced
    episodes so the actor coverage signal is clean.  Router critic is trained
    throughout so it can cold-start immediately after switching to router mode.

    Compare directly against ``v6i7_router_critic_warmup_config()`` at the
    same step budget to isolate the effect of stronger actor conditioning.
    """
    return replace(
        v6i7_recurrent_router_config(),
        latent_assignment_mode="balanced_episode",
        train_router_when_forced=False,
        train_router_critic_when_forced=True,
        enable_latent_z_residual=True,
        latent_z_gate_init=0.01,
    )


def v6i8_adapter_sparse_config() -> PPOConfig:
    """V6I8-sparse: residual adapters + sparse router reward, free router.

    Use after adapter differentiation is confirmed with ``v6i8_adapter_balanced``.
    The router assigns latents freely; the sparse reward reduces dense-reward
    noise in the routing objective.
    """
    return replace(
        v6i7_sparse_router_config(),
        enable_latent_z_residual=True,
        latent_z_gate_init=0.01,
    )


def v6i8_adapter_balanced_hardpool_config() -> PPOConfig:
    """V6I8-balanced + hard opponent pool (OP8/OP9/OP10).

    Use when OP5/OP6/OP7 no longer produce sufficient latent separation.
    OP8 (coordinated interceptor), OP9 (fortress + counterattack), and OP10
    (active escort carrier) require distinct counter-strategies so the latent
    router has a real job.
    """
    return replace(
        v6i8_adapter_balanced_config(),
        opponent_pool=("OP8", "OP9", "OP10"),
        opponent_pool_weights=(),
    )


def v6i8_adapter_sparse_hardpool_config() -> PPOConfig:
    """V6I8-sparse + hard opponent pool (OP8/OP9/OP10).

    Sparse router reward variant against the hard opponent pool.
    """
    return replace(
        v6i8_adapter_sparse_config(),
        opponent_pool=("OP8", "OP9", "OP10"),
        opponent_pool_weights=(),
    )
