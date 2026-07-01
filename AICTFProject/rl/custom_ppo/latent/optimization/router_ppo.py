"""Unified router PPO with frozen rollout advantages."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.distributions import Categorical

from rl.ppo_core import ppo_policy_loss
from rl.custom_ppo.latent.optimization.ppo_stats import EpisodeStatsAccumulator
from rl.custom_ppo.latent.optimization.router_registry import LatentOptimizerRegistry
from rl.custom_ppo.latent.optimization.router_stepper import RouterOptimizerStepper
from rl.custom_ppo.latent.types import EpisodeAuxiliaryLossBundle, RouterPPOBatch, RouterPPOConfig, RouterStepResult


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
        auxiliary_loss_fn: Callable[[torch.Tensor, int], tuple[EpisodeAuxiliaryLossBundle, dict[str, Any]]]
        | None = None,
        stats_accumulator: EpisodeStatsAccumulator | None = None,
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
        if coef <= 0.0 and auxiliary_loss_fn is None:
            return {}, []

        stats: dict[str, float] = {}
        steps: list[RouterStepResult] = []
        accum = stats_accumulator or EpisodeStatsAccumulator()
        clip_eps = max(1e-6, float(config.clip_epsilon))
        value_coef = max(0.0, float(config.value_coef))
        max_grad_norm = max(0.0, float(config.max_grad_norm))
        q_phi_params: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for group in param_groups:
            for p in group:
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                q_phi_params.append(p)
        target_kl = config.target_kl
        target_kl_mult = max(1.0, float(config.target_kl_multiplier))

        for epoch in range(max(1, int(config.epochs))):
            logits = trainer.model.strategy_logits(states, selector_hidden=hidden)
            dist = Categorical(logits=logits)
            new_log_prob = dist.log_prob(z)
            v_z = value_fn(states, z, hidden)
            pg_loss, ppo_stats = ppo_policy_loss(new_log_prob, old_log_prob, fixed_adv, clip_eps)
            v_loss = 0.5 * (returns - v_z).pow(2).mean() if v_z is not None else torch.zeros((), device=device)
            aux_bundle = None
            aux_stats: dict[str, Any] = {}
            aux_scaled = torch.zeros((), dtype=torch.float32, device=device)
            if auxiliary_loss_fn is not None:
                aux_bundle, aux_stats = auxiliary_loss_fn(logits, epoch)
                aux_scaled = aux_bundle.total_scaled()
                stats["latent_preference_loss"] = float(aux_bundle.preference.raw.detach().cpu().item())
                stats["latent_awrd_loss"] = float(aux_bundle.awrd.raw.detach().cpu().item())
                stats["latent_v3i3_event_pref_loss"] = float(
                    aux_bundle.refresh_preference.raw.detach().cpu().item()
                )
                stats["latent_specialist_loss"] = float(aux_bundle.specialist.raw.detach().cpu().item())
                stats["latent_usage_balance_loss"] = float(aux_bundle.usage_balance.scaled.detach().cpu().item())
                stats["latent_episode_entropy"] = float(aux_bundle.entropy.raw.detach().cpu().item())
            if coef > 0.0:
                loss = coef * pg_loss + value_coef * v_loss + aux_scaled
            else:
                loss = aux_scaled
            if not torch.isfinite(loss).all():
                accum.record_epoch(
                    pg_loss=float(pg_loss.detach().cpu().item()),
                    value_loss=float(v_loss.detach().cpu().item()),
                    approx_kl=float(ppo_stats.get("approx_kl", 0.0)),
                    clip_fraction=float(ppo_stats.get("clipfrac", 0.0)),
                    grad_norm=0.0,
                    aux_loss=float(aux_scaled.detach().cpu().item()),
                    ratio_mean=float(ppo_stats["ratio"].mean().detach().cpu().item()),
                    stepped=False,
                )
                continue
            q_phi_shape = None
            if collect_router_shape:
                probs = torch.softmax(logits, dim=-1)
                q_phi_shape = (
                    float(dist.entropy().mean().detach().cpu().item()),
                    float(probs.max(dim=-1).values.mean().detach().cpu().item()),
                )
            stepped = False
            grad_norm = 0.0
            if self.stepper is not None:
                result = self.stepper.step(
                    loss,
                    epoch=epoch,
                    batch_name=str(config.objective_name),
                    grad_split_groups=grad_split_groups,
                    q_phi_shape=q_phi_shape,
                    max_grad_norm=max_grad_norm,
                )
                stepped = bool(result.stepped)
                grad_norm = float(result.grad_norm)
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
                grad_norm = float(grad_norm_l2(q_phi_params))
                if splits and "encoder" in splits and "value_head" in splits:
                    grad_norm = float(
                        (float(splits["encoder"]) ** 2 + float(splits["value_head"]) ** 2) ** 0.5
                    )
                if grad_norm > 0.0 and torch.isfinite(loss).all():
                    torch.nn.utils.clip_grad_norm_(q_phi_params, max_grad_norm)
                    fallback.step()
                    stepped = True
                result = RouterStepResult(
                    stepped=stepped,
                    grad_norm=grad_norm,
                    finite=stepped,
                    optimizer_steps=accum.optimizer_steps + int(stepped),
                    grad_splits=splits,
                    q_phi_entropy=float(q_phi_shape[0]) if q_phi_shape else 0.0,
                    q_phi_mean_max_prob=float(q_phi_shape[1]) if q_phi_shape else 0.0,
                )
            steps.append(result)
            ratio_mean = float(ppo_stats["ratio"].mean().detach().cpu().item())
            approx_kl = float(ppo_stats.get("approx_kl", 0.0))
            accum.record_epoch(
                pg_loss=float(pg_loss.detach().cpu().item()),
                value_loss=float(v_loss.detach().cpu().item()),
                approx_kl=approx_kl,
                clip_fraction=float(ppo_stats.get("clipfrac", 0.0)),
                grad_norm=grad_norm,
                aux_loss=float(aux_scaled.detach().cpu().item()),
                ratio_mean=ratio_mean,
                stepped=stepped,
            )
            stats["policy_loss"] = float(pg_loss.detach().cpu().item())
            stats["value_loss"] = float(v_loss.detach().cpu().item())
            stats["approx_kl"] = approx_kl
            stats["clipfrac"] = float(ppo_stats.get("clipfrac", 0.0))
            stats["advantage_mean"] = float(fixed_adv.mean().detach().cpu().item())
            stats["advantage_std"] = float(fixed_adv.std(unbiased=False).detach().cpu().item())
            stats["grad_norm"] = grad_norm
            if aux_bundle is not None:
                stats["aux_loss"] = float(aux_scaled.detach().cpu().item())
            if aux_stats:
                stats.update(
                    {
                        k: float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v)
                        for k, v in aux_stats.items()
                    }
                )
            if (
                target_kl is not None
                and approx_kl > target_kl_mult * float(target_kl)
            ):
                accum.early_stop = 1
                accum.early_stop_kl = approx_kl
                accum.stop_reason = "target_kl"
                break

        stats.update(accum.finalize_base())
        return stats, steps

    def apply(
        self,
        batch: RouterPPOBatch,
        *,
        config: RouterPPOConfig,
        fixed_advantages: torch.Tensor | None = None,
        value_target: torch.Tensor | None = None,
        auxiliary_loss_fn: Callable[[torch.Tensor, int], tuple[EpisodeAuxiliaryLossBundle, dict[str, Any]]]
        | None = None,
        stats_accumulator: EpisodeStatsAccumulator | None = None,
    ) -> tuple[dict[str, float], list[RouterStepResult]]:
        if fixed_advantages is not None:
            batch = RouterPPOBatch(
                states=batch.states,
                executed_z=batch.executed_z,
                old_behavior_log_prob=batch.old_behavior_log_prob,
                fixed_advantages=fixed_advantages,
                returns=value_target if value_target is not None else batch.returns,
                selector_hidden=batch.selector_hidden,
            )
        elif value_target is not None:
            batch = RouterPPOBatch(
                states=batch.states,
                executed_z=batch.executed_z,
                old_behavior_log_prob=batch.old_behavior_log_prob,
                fixed_advantages=batch.fixed_advantages,
                returns=value_target,
                selector_hidden=batch.selector_hidden,
            )

        def value_fn(st, z_t, hidden):
            if self.trainer.model.episode_strategy_value_head is None:
                return torch.zeros_like(batch.returns)
            return self.trainer.model.episode_strategy_value(st, z_t, selector_hidden=hidden)

        encoder_params = [
            p
            for p in self.trainer.model.strategy_encoder.parameters()
            if p.requires_grad
        ]
        value_head_params = []
        if self.trainer.model.episode_strategy_value_head is not None:
            value_head_params = [
                p
                for p in self.trainer.model.episode_strategy_value_head.parameters()
                if p.requires_grad
            ]
        return self.run(
            batch,
            config=config,
            value_fn=value_fn,
            param_groups=[encoder_params, value_head_params],
            auxiliary_loss_fn=auxiliary_loss_fn,
            stats_accumulator=stats_accumulator,
        )
