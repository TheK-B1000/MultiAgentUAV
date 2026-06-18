"""Unified router PPO with frozen rollout advantages."""

from __future__ import annotations

from typing import Any

import torch
from torch.distributions import Categorical

from rl.ppo_core import ppo_policy_loss
from rl.custom_ppo.latent.optimization.router_registry import LatentOptimizerRegistry
from rl.custom_ppo.latent.optimization.router_stepper import RouterOptimizerStepper
from rl.custom_ppo.latent.types import RouterPPOBatch, RouterPPOConfig, RouterStepResult


def grad_norm_l2(params: list[torch.nn.Parameter]) -> float:
    sq = 0.0
    any_grad = False
    for p in params:
        if p.grad is None:
            continue
        any_grad = True
        sq += float(p.grad.detach().pow(2).sum().item())
    return float(sq**0.5) if any_grad else 0.0


class RouterPPOEngine:
    def __init__(
        self,
        *,
        trainer: Any,
        registry: LatentOptimizerRegistry | None,
        fallback_optimizer: Any | None = None,
    ) -> None:
        self.trainer = trainer
        self.registry = registry
        self.fallback_optimizer = fallback_optimizer
        self.stepper = RouterOptimizerStepper(registry) if registry is not None else None

    def run(
        self,
        batch: RouterPPOBatch,
        *,
        config: RouterPPOConfig,
        value_fn,
        param_groups: list[list[torch.nn.Parameter]],
        grad_split_groups: dict[str, list[torch.nn.Parameter]] | None = None,
        collect_router_shape: bool = False,
    ) -> tuple[dict[str, float], list[RouterStepResult]]:
        trainer = self.trainer
        registry = self.registry
        fallback = self.fallback_optimizer
        if registry is None and fallback is None:
            return {}, []
        device = trainer.device
        states = batch.states
        z = batch.executed_z
        old_log_prob = batch.old_behavior_log_prob
        fixed_adv = batch.fixed_advantages
        returns = batch.returns
        hidden = batch.selector_hidden
        coef = float(config.coef)
        if coef <= 0.0:
            return {}, []

        stats: dict[str, float] = {}
        steps: list[RouterStepResult] = []
        clip_eps = max(1e-6, float(config.clip_epsilon))
        value_coef = max(0.0, float(config.value_coef))
        q_phi_params = [p for group in param_groups for p in group]

        for epoch in range(max(1, int(config.epochs))):
            logits = trainer.model.strategy_logits(states, selector_hidden=hidden)
            dist = Categorical(logits=logits)
            new_log_prob = dist.log_prob(z)
            v_z = value_fn(states, z, hidden)
            pg_loss, ppo_stats = ppo_policy_loss(new_log_prob, old_log_prob, fixed_adv, clip_eps)
            v_loss = 0.5 * (returns - v_z).pow(2).mean() if v_z is not None else torch.zeros((), device=device)
            loss = coef * pg_loss + value_coef * v_loss
            if not torch.isfinite(loss).all():
                continue
            q_phi_shape = None
            if collect_router_shape:
                probs = torch.softmax(logits, dim=-1)
                q_phi_shape = (
                    float(dist.entropy().mean().detach().cpu().item()),
                    float(probs.max(dim=-1).values.mean().detach().cpu().item()),
                )
            if self.stepper is not None:
                result = self.stepper.step(
                    loss,
                    epoch=epoch,
                    batch_name="router_ppo",
                    grad_split_groups=grad_split_groups,
                    q_phi_shape=q_phi_shape,
                )
            else:
                assert fallback is not None
                fallback.zero_grad(set_to_none=True)
                loss.backward()
                splits = None
                if grad_split_groups:
                    splits = {
                        name: float(grad_norm_l2(params))
                        for name, params in grad_split_groups.items()
                    }
                result = RouterStepResult(
                    stepped=True,
                    grad_norm=float(grad_norm_l2(q_phi_params)),
                    finite=True,
                    optimizer_steps=epoch + 1,
                    grad_splits=splits,
                    q_phi_entropy=float(q_phi_shape[0]) if q_phi_shape else 0.0,
                    q_phi_mean_max_prob=float(q_phi_shape[1]) if q_phi_shape else 0.0,
                )
                fallback.step()
            steps.append(result)
            stats["policy_loss"] = float(pg_loss.detach().cpu().item())
            stats["value_loss"] = float(v_loss.detach().cpu().item())
            stats["approx_kl"] = float(ppo_stats.get("approx_kl", 0.0))
            stats["clipfrac"] = float(ppo_stats.get("clipfrac", 0.0))
            stats["advantage_mean"] = float(fixed_adv.mean().detach().cpu().item())
            stats["advantage_std"] = float(fixed_adv.std(unbiased=False).detach().cpu().item())
            stats["grad_norm"] = float(grad_norm_l2(q_phi_params))
        return stats, steps
