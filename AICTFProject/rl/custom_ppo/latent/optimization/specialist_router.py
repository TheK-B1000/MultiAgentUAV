"""Rollout-level q_phi specialization training on tactical context keys."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from rl.custom_ppo.latent.context_buckets import specialist_context_keys_for_mode
from rl.custom_ppo.latent.preferences import (
    router_specialist_coef_scale,
    router_specialist_loss,
)

if TYPE_CHECKING:
    from rl.custom_ppo.latent.state import LatentStrategyState


class SpecialistRouterManager:
    def __init__(self, host: "LatentStrategyState") -> None:
        self.host = host

    def apply_rollout_specialist_router(self, buffer: Any) -> dict[str, float]:
        """Train q_phi specialization on tactical states observed in rollout."""
        trainer = self.host.trainer
        stats = {
            "latent_specialist_loss": 0.0,
            "latent_specialist_marginal_entropy": 0.0,
            "latent_specialist_conditional_entropy": 0.0,
            "latent_specialist_context_bucket_entropy": 0.0,
            "latent_specialist_conditional_term": 0.0,
            "latent_specialist_conditional_coef": 0.0,
            "latent_specialist_mi": 0.0,
            "latent_specialist_context_mi": 0.0,
            "latent_specialist_active_buckets": 0.0,
            "latent_specialist_coef_scale": 0.0,
            "latent_specialist_rollout_samples": 0.0,
        }
        if (
            not bool(getattr(trainer, "latent_specialist_router_enabled", False))
            or not bool(getattr(trainer, "latent_specialist_use_rollout_states", False))
            or bool(getattr(trainer, "fixed_latent_strategy", False))
            or int(getattr(buffer, "pos", 0)) <= 0
            or "global_state" not in buffer.fields
            or "opponent_id" not in buffer.fields
        ):
            return stats

        length = int(buffer.pos)
        states = buffer.fields["global_state"][:length].reshape(
            -1, buffer.fields["global_state"].shape[-1]
        )
        opponent_ids = buffer.fields["opponent_id"][:length].reshape(-1).long()
        total = int(states.shape[0])
        max_samples = max(
            1,
            int(getattr(trainer, "latent_specialist_rollout_max_samples", 8192) or 8192),
        )
        if total > max_samples:
            sample_idx = (
                torch.linspace(0, total - 1, steps=max_samples, device=states.device)
                .round()
                .long()
                .unique()
            )
            states = states.index_select(0, sample_idx)
            opponent_ids = opponent_ids.index_select(0, sample_idx)

        context_keys = specialist_context_keys_for_mode(
            mode=str(
                getattr(trainer, "latent_specialist_context_key_mode", "opponent_bucket")
                or "opponent_bucket"
            ),
            states=states,
            opponent_ids=opponent_ids,
            bucket_ids=None,
        )
        if context_keys is None:
            return stats

        warmup_steps = int(getattr(trainer, "latent_specialist_warmup_steps", 0) or 0)
        global_step = int(getattr(trainer, "global_step", 0) or 0)
        coef_scale = router_specialist_coef_scale(
            global_step=global_step,
            warmup_steps=warmup_steps,
            ramp_steps=int(getattr(trainer, "latent_specialist_ramp_steps", 1) or 0),
        )
        conditional_start = (
            float(getattr(trainer, "latent_conditional_entropy_min_coef_start", 0.0) or 0.0)
            if global_step >= warmup_steps
            else 0.0
        )

        logits = trainer.model.strategy_logits(states)
        loss, tensor_stats = router_specialist_loss(
            logits,
            context_keys=context_keys,
            latent_k=int(trainer.latent_k),
            marginal_balance_coef=float(
                getattr(trainer, "latent_marginal_balance_coef", 0.0) or 0.0
            ),
            conditional_entropy_min_coef=float(
                getattr(trainer, "latent_conditional_entropy_min_coef", 0.0) or 0.0
            ),
            conditional_entropy_min_coef_start=conditional_start,
            conditional_entropy_scope=str(
                getattr(trainer, "latent_specialist_conditional_entropy_scope", "state")
                or "state"
            ),
            context_mi_coef=float(getattr(trainer, "latent_context_mi_coef", 0.0) or 0.0),
            coef_scale=coef_scale,
            min_bucket_count=int(getattr(trainer, "latent_specialist_min_bucket_count", 2) or 2),
        )
        if loss.requires_grad and (
            coef_scale > 0.0
            or float(getattr(trainer, "latent_conditional_entropy_min_coef_start", 0.0) or 0.0)
            > 0.0
        ):
            optimizer = getattr(trainer, "latent_router_optimizer", None) or trainer.optimizer
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            strategy_module = getattr(trainer.model, "strategy_encoder", None)
            if strategy_module is not None:
                torch.nn.utils.clip_grad_norm_(
                    strategy_module.parameters(),
                    float(trainer.cfg.max_grad_norm),
                )
            optimizer.step()

        for key, value in tensor_stats.items():
            stats[key] = float(value.detach().cpu().item())
        stats["latent_specialist_rollout_samples"] = float(states.shape[0])
        return stats
