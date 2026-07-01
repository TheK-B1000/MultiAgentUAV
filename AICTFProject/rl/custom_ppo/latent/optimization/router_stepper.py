"""Single router optimizer step with finite checks and step counting."""

from __future__ import annotations

import torch

from rl.custom_ppo.latent.optimization.router_registry import LatentOptimizerRegistry
from rl.custom_ppo.latent.types import RouterStepResult


class RouterOptimizerStepper:
    def __init__(self, registry: LatentOptimizerRegistry) -> None:
        self.registry = registry
        self.optimizer_steps = 0

    def step(
        self,
        loss: torch.Tensor,
        *,
        epoch: int,
        batch_name: str,
        grad_split_groups: dict[str, list[torch.nn.Parameter]] | None = None,
        q_phi_shape: tuple[float, float] | None = None,
        max_grad_norm: float = 0.5,
    ) -> RouterStepResult:
        del epoch, batch_name
        from rl.custom_ppo.latent.optimization.router_ppo import grad_norm_l2

        if not torch.isfinite(loss).all():
            return RouterStepResult(
                stepped=False, grad_norm=0.0, finite=False, optimizer_steps=self.optimizer_steps
            )
        self.registry.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = 0.0
        for p in self.registry.router_parameters:
            if p.grad is not None and torch.isfinite(p.grad).all():
                grad_norm += float(p.grad.detach().pow(2).sum().item())
            elif p.grad is not None:
                return RouterStepResult(
                    stepped=False,
                    grad_norm=0.0,
                    finite=False,
                    optimizer_steps=self.optimizer_steps,
                )
        grad_norm = float(grad_norm**0.5)
        splits: dict[str, float] | None = None
        if grad_split_groups:
            splits = {name: float(grad_norm_l2(params)) for name, params in grad_split_groups.items()}
        torch.nn.utils.clip_grad_norm_(self.registry.router_parameters, max_grad_norm)
        self.registry.step()
        self.optimizer_steps += 1
        entropy, max_prob = (0.0, 0.0) if q_phi_shape is None else q_phi_shape
        return RouterStepResult(
            stepped=True,
            grad_norm=grad_norm,
            finite=True,
            optimizer_steps=self.optimizer_steps,
            grad_splits=splits,
            q_phi_entropy=float(entropy),
            q_phi_mean_max_prob=float(max_prob),
        )
