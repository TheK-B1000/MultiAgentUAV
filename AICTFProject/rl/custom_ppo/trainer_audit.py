"""Startup audit logging for :class:`CustomPPOTrainer`.

These functions print one-time contract assertions about the actor / critic
input dimensions, the latent strategy contract, and the plan-faithful audit
checks. They have **no** side effects on training — no gradient updates, no
rollout state mutation, no z-state changes. That isolation is why they live
in their own module: removing them, refactoring them, or silencing them in
tests cannot change training behavior.

All three functions take the trainer as the first argument. The idempotency
flag ``trainer._decentralized_actor_contract_logged`` lives on the trainer
instance so the public entry point (called once from ``train_ppo._build_trainer``)
and the internal entry point (called from every ``collect_rollout``) share
state and only the first call actually prints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rl.global_state import GLOBAL_STATE_FIELD_NAMES
from rl.custom_ppo.policy import SharedActorCentralizedCritic

if TYPE_CHECKING:
    from rl.custom_ppo.trainer import CustomPPOTrainer


def log_decentralized_actor_contract_once(trainer: "CustomPPOTrainer") -> None:
    """One-time training log: actor consumes CNN(grid) + scalars + optional z, not global state.

    Thin wrapper kept as a stable internal entry point so ``collect_rollout``
    has a single call regardless of how the contract logging evolves.
    """
    log_input_dim_contract(trainer)


def log_input_dim_contract(trainer: "CustomPPOTrainer") -> None:
    """Print the startup input-dimension contract (idempotent via trainer flag).

    Public entry point — also called once from ``train_ppo._build_trainer`` so
    the contract appears before the first rollout. The idempotency flag lives
    on the trainer instance so multiple callers cannot double-print.
    """
    if trainer._decentralized_actor_contract_logged:
        return
    m = trainer.model
    assert isinstance(m, SharedActorCentralizedCritic)
    dims = m.input_dim_contract()
    print(
        "[PPO] Input dims: "
        f"base_global_state_dim={dims['base_global_state_dim']} "
        f"temporal_context_dim={dims['temporal_context_dim']} "
        f"q_phi_input_dim={dims['q_phi_input_dim']} "
        f"critic_context_dim={dims['critic_context_dim']} "
        f"actor_input_dim={dims['actor_input_dim']}"
    )
    if m.uses_latent_strategy:
        print(
            "[PPO] Decentralized actor contract: per-agent MLP input dim = "
            f"{m._decentralized_actor_in_dim} "
            f"(cnn {m.actor_cnn_feature_dim} + scalars {m._scalar_per_agent} + z_emb {m.z_embed_dim}); "
            f"global_state_dim={m.global_state_dim} is for q_phi/critic only."
        )
        print(
            "[PPO] Critic z contract: "
            f"context_dim={dims['critic_context_dim']} "
            f"joint_action_onehot_dim={dims['critic_joint_action_dim']} "
            f"z_onehot_dim={dims['critic_z_dim']} "
            f"critic_extra_dim={dims['critic_extra_dim']} "
            "z_present=True"
        )
    else:
        print(
            "[PPO] Decentralized actor contract: per-agent MLP input dim = "
            f"{m._decentralized_actor_in_dim} "
            f"(cnn {m.actor_cnn_feature_dim} + scalars {m._scalar_per_agent}, no z); "
            f"global_state_dim={m.global_state_dim} not used in policy."
        )
    log_plan_faithful_audit(trainer)
    trainer._decentralized_actor_contract_logged = True


def log_plan_faithful_audit(trainer: "CustomPPOTrainer") -> None:
    """Plan-faithful audit: assert forbidden modules absent; print audit lines.

    No-op when latent strategy is disabled (no-latent baselines have no z
    contract to audit). Raises ``AssertionError`` if any of the SUMMER plan's
    forbidden modules (opponent ID heads, supervised routers, Gumbel-Softmax
    z, VAE losses, hard-coded strategy labels) are attached to the model, or
    if OP4 leaks into the training opponent pool without the explicit
    ``allow_op4_in_training_pool`` override.
    """
    if not trainer.model.uses_latent_strategy:
        return
    for forbidden_attr in (
        "opponent_id_head",
        "opponent_classifier",
        "gumbel_softmax_z",
        "vae_z_head",
        "strategy_label_head",
        "supervised_router",
    ):
        if getattr(trainer.model, forbidden_attr, None) is not None:
            raise AssertionError(
                f"plan-faithful audit: forbidden module '{forbidden_attr}' is attached to the model."
            )
    cfg = trainer.cfg
    optional = {
        "aux_return_head": bool(getattr(cfg, "latent_strategy_aux_return_head", False)),
        "kl_consecutive": float(getattr(cfg, "latent_kl_consecutive", 0.0) or 0.0) > 0.0,
        "resample_on_flag": bool(getattr(cfg, "latent_resample_on_flag", False)),
        "fixed_latent_strategy": bool(getattr(cfg, "fixed_latent_strategy", False)),
    }
    print(
        "[PPO] Summer-plan audit (latent on): no supervised router labels, no opponent-ID heads, "
        "no Gumbel-Softmax, no VAE losses, no handcrafted strategy labels."
    )
    extras_on = [name for name, on in optional.items() if on]
    if extras_on:
        print(
            "[PPO] Summer-plan audit: optional add-ons ENABLED "
            f"{extras_on} (not plan-faithful first-run; treat as intentional ablation)."
        )
    else:
        print(
            "[PPO] Summer-plan audit: optional add-ons "
            "(aux_return_head, kl_consecutive, resample_on_flag, fixed_latent_strategy) all OFF "
            "— plan-faithful first-run."
        )

    m = trainer.model

    if m.uses_latent_strategy:
        print(
            "[PPO] Audit actor_input: "
            f"cnn({getattr(m, 'actor_cnn_feature_dim', '?')}) "
            f"+ per_agent_vec({getattr(m, '_scalar_per_agent', '?')}) "
            f"+ z_emb({getattr(m, 'z_embed_dim', '?')}) "
            f"= {getattr(m, '_decentralized_actor_in_dim', '?')} dim. "
            "(no phase_id, no opponent_id, no global_state in actor pathway)"
        )

    phase_coef = float(getattr(cfg, "latent_strategy_aux_predict_phase_coef", 0.0) or 0.0)
    if m.uses_latent_strategy:
        forbidden_in_global_state = [
            name for name in GLOBAL_STATE_FIELD_NAMES if "phase" in str(name).lower()
        ]
        if forbidden_in_global_state:
            raise AssertionError(
                "plan-faithful audit: q_phi/critic global_state contains phase-tagged fields "
                f"{forbidden_in_global_state}; this would leak the phase label into the input."
            )
        if phase_coef > 0.0:
            print(
                "[PPO] Audit aux_phase_head_input: expected_z_emb"
                f"({getattr(m, 'z_embed_dim', '?')}) "
                "= softmax(z_logits) @ strategy_embedding. "
                f"phase_id used only as cross-entropy TARGET (coef={phase_coef:g}). "
                f"q_phi input ({getattr(m, 'global_state_dim', '?')}-d {GLOBAL_STATE_FIELD_NAMES!r}) "
                "contains no phase fields. Phase gradient does NOT flow into actor."
            )
        else:
            print("[PPO] Audit aux_phase_head: OFF (latent_strategy_aux_predict_phase_coef=0).")

    pool = tuple(str(t).strip().upper() for t in (getattr(cfg, "opponent_pool", ()) or ()))
    allow_op4 = bool(getattr(cfg, "allow_op4_in_training_pool", False))
    op4_in_pool = "OP4" in pool
    if op4_in_pool and not allow_op4:
        raise AssertionError(
            "plan-faithful audit: OP4 present in training opponent_pool but "
            "allow_op4_in_training_pool=False; OP4 must be eval-only."
        )
    op4_status = (
        "ALLOWED (allow_op4_in_training_pool=True)" if op4_in_pool and allow_op4
        else "EXCLUDED (eval-only)" if not op4_in_pool
        else "INCONSISTENT"
    )
    print(
        "[PPO] Audit opponent_pool (training): "
        f"{list(pool) if pool else '(default fixed-opponent mode)'} ; "
        f"OP4 status: {op4_status}"
    )


__all__ = [
    "log_decentralized_actor_contract_once",
    "log_input_dim_contract",
    "log_plan_faithful_audit",
]
