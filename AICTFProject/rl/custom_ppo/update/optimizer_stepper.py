"""Optimizer stepping strategies with optimizer-owned clipping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from rl.custom_ppo.update.helpers import assert_finite_gradients, assert_finite_loss
from rl.custom_ppo.update.phase_policy import PhaseTrainingPolicy
from rl.custom_ppo.update.update_context import PPOUpdateContext


def clip_optimizer_grad_norm(optimizer: torch.optim.Optimizer, max_norm: float) -> float:
    """Clip gradients for parameters owned by this optimizer only."""
    params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
    if not params:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(params, float(max_norm)))


def clip_named_param_groups(
    *,
    actor_params: list[torch.nn.Parameter],
    critic_params: list[torch.nn.Parameter],
    max_norm: float,
) -> tuple[float, float]:
    """Independently clip actor and critic grads; return pre-clip norms."""
    actor_gn = 0.0
    critic_gn = 0.0
    actor_with_grad = [p for p in actor_params if p.grad is not None]
    critic_with_grad = [p for p in critic_params if p.grad is not None]
    if actor_with_grad:
        actor_gn = float(torch.nn.utils.clip_grad_norm_(actor_with_grad, float(max_norm)))
    if critic_with_grad:
        critic_gn = float(torch.nn.utils.clip_grad_norm_(critic_with_grad, float(max_norm)))
    return actor_gn, critic_gn


def _params_from_named_groups(
    optimizer: torch.optim.Optimizer,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]] | None:
    """If optimizer has named z_actor/critic groups, return those param lists."""
    by_name: dict[str, list[torch.nn.Parameter]] = {}
    for group in optimizer.param_groups:
        name = str(group.get("name") or "")
        if name:
            by_name[name] = list(group["params"])
    if "z_actor" in by_name and "critic" in by_name:
        return by_name["z_actor"], by_name["critic"]
    return None


@dataclass(frozen=True)
class OptimizerStepResult:
    actor_grad_norm: float
    critic_grad_norm: float
    router_grad_norm: float
    global_grad_norm: float
    strategy_grad_norm: float


class OptimizerStepper(Protocol):
    def step(
        self,
        *,
        total_loss: torch.Tensor,
        ppo_actor_loss: torch.Tensor,
        value_loss: torch.Tensor,
        policy_loss: torch.Tensor,
        entropy_loss: torch.Tensor,
        latent_loss: torch.Tensor,
        ent_coef: float,
        vf_coef: float,
        context: PPOUpdateContext,
        phase_policy: PhaseTrainingPolicy,
        model: torch.nn.Module,
        latent_state: Any,
        epoch_idx: int,
        mb_idx: int,
        max_grad_norm: float,
    ) -> OptimizerStepResult: ...


class SharedOptimizerStepper:
    def __init__(self, optimizer: torch.optim.Optimizer, runtime: Any | None = None) -> None:
        self.optimizer = optimizer
        self.runtime = runtime

    def step(
        self,
        *,
        total_loss: torch.Tensor,
        ppo_actor_loss: torch.Tensor,
        value_loss: torch.Tensor,
        policy_loss: torch.Tensor,
        entropy_loss: torch.Tensor,
        latent_loss: torch.Tensor,
        ent_coef: float,
        vf_coef: float,
        context: PPOUpdateContext,
        phase_policy: PhaseTrainingPolicy,
        model: torch.nn.Module,
        latent_state: Any,
        epoch_idx: int,
        mb_idx: int,
        max_grad_norm: float,
    ) -> OptimizerStepResult:
        del ppo_actor_loss, value_loss, policy_loss, entropy_loss, latent_loss
        del ent_coef, vf_coef, context, phase_policy
        self.optimizer.zero_grad(set_to_none=True)
        assert_finite_loss(total_loss, epoch_idx=epoch_idx, mb_idx=mb_idx)
        total_loss.backward()
        assert_finite_gradients(model, epoch_idx=epoch_idx, mb_idx=mb_idx)
        if self.runtime is not None:
            from rl.custom_ppo.diagnostics.competence import record_repertoire_grad_audit

            record_repertoire_grad_audit(self.runtime, model)
        strategy_grad_norm = float(latent_state.strategy_encoder_grad_norm())

        cfg = getattr(self.runtime, "cfg", None) if self.runtime is not None else None
        separate_clip = bool(getattr(cfg, "latent_lro_separate_actor_critic_clip", False))
        named = _params_from_named_groups(self.optimizer)
        actor_grad_norm = 0.0
        critic_grad_norm = 0.0
        if separate_clip and named is not None:
            actor_grad_norm, critic_grad_norm = clip_named_param_groups(
                actor_params=named[0],
                critic_params=named[1],
                max_norm=float(max_grad_norm),
            )
            grad_norm = max(actor_grad_norm, critic_grad_norm)
        else:
            grad_norm = clip_optimizer_grad_norm(self.optimizer, max_grad_norm)
        self.optimizer.step()
        return OptimizerStepResult(
            actor_grad_norm=float(actor_grad_norm),
            critic_grad_norm=float(critic_grad_norm),
            router_grad_norm=0.0,
            global_grad_norm=float(grad_norm),
            strategy_grad_norm=strategy_grad_norm,
        )


class ThreeOptimizerStepper:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def step(
        self,
        *,
        total_loss: torch.Tensor,
        ppo_actor_loss: torch.Tensor,
        value_loss: torch.Tensor,
        policy_loss: torch.Tensor,
        entropy_loss: torch.Tensor,
        latent_loss: torch.Tensor,
        ent_coef: float,
        vf_coef: float,
        context: PPOUpdateContext,
        phase_policy: PhaseTrainingPolicy,
        model: torch.nn.Module,
        latent_state: Any,
        epoch_idx: int,
        mb_idx: int,
        max_grad_norm: float,
    ) -> OptimizerStepResult:
        del policy_loss, entropy_loss, ent_coef
        runtime = self.runtime
        runtime.actor_optimizer.zero_grad(set_to_none=True)
        runtime.critic_optimizer.zero_grad(set_to_none=True)
        runtime.router_optimizer.zero_grad(set_to_none=True)
        assembled = vf_coef * value_loss
        if phase_policy.actor_step_enabled:
            assembled = assembled + ppo_actor_loss
        cfg = getattr(runtime, "cfg", None)
        current_plus_delta_router_loss = (
            str(getattr(cfg, "router_context_mode", "") or "") == "current_plus_delta"
        )
        include_latent_loss = (
            isinstance(latent_loss, torch.Tensor)
            and latent_loss.requires_grad
            and (phase_policy.router_step_enabled or not current_plus_delta_router_loss)
        )
        if include_latent_loss:
            assembled = assembled + latent_loss
        assert_finite_loss(assembled, epoch_idx=epoch_idx, mb_idx=mb_idx)
        assembled.backward()
        assert_finite_gradients(model, epoch_idx=epoch_idx, mb_idx=mb_idx)
        from rl.custom_ppo.diagnostics.competence import record_repertoire_grad_audit

        record_repertoire_grad_audit(runtime, model)
        strategy_grad_norm = float(latent_state.strategy_encoder_grad_norm())
        from rl.custom_ppo.v6i1_phase_runtime import step_v6i1_optimizers

        grads = step_v6i1_optimizers(
            runtime,
            phase=context.phase,
            actor_step=phase_policy.actor_step_enabled,
            critic_step=phase_policy.critic_step_enabled,
            router_step=phase_policy.router_step_enabled,
            max_grad_norm=float(max_grad_norm),
        )
        global_norm = max(
            float(grads.get("actor_grad_norm", 0.0)),
            float(grads.get("critic_grad_norm", 0.0)),
            float(grads.get("router_grad_norm", 0.0)),
        )
        return OptimizerStepResult(
            actor_grad_norm=float(grads.get("actor_grad_norm", 0.0)),
            critic_grad_norm=float(grads.get("critic_grad_norm", 0.0)),
            router_grad_norm=float(grads.get("router_grad_norm", 0.0)),
            global_grad_norm=global_norm,
            strategy_grad_norm=strategy_grad_norm,
        )


def build_optimizer_stepper(runtime: Any, optimizer: torch.optim.Optimizer) -> OptimizerStepper:
    if bool(getattr(runtime, "v6i1_three_optimizer_mode", False)):
        return ThreeOptimizerStepper(runtime)
    return SharedOptimizerStepper(optimizer, runtime=runtime)
