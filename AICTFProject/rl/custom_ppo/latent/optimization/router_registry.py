"""Authoritative router optimizer registry."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from rl.custom_ppo.curriculum_gates import is_staged_v6i1_curriculum


@dataclass(frozen=True)
class LatentOptimizerRegistry:
    router: torch.optim.Optimizer
    router_parameters: tuple[nn.Parameter, ...]

    @classmethod
    def from_trainer(cls, trainer) -> LatentOptimizerRegistry | None:
        opt = getattr(trainer, "router_optimizer", None) or getattr(
            trainer, "latent_router_optimizer", None
        )
        if opt is None:
            if is_staged_v6i1_curriculum(trainer.cfg):
                raise RuntimeError(
                    "V6 staged curriculum requires a dedicated router optimizer; none is registered."
                )
            return None
        params = tuple(
            p for group in opt.param_groups for p in group["params"] if p.requires_grad
        )
        if not params:
            raise RuntimeError("Router optimizer has no trainable parameters.")
        return cls(router=opt, router_parameters=params)

    def require_router_optimizer(self, *, staged_v6: bool) -> torch.optim.Optimizer:
        if staged_v6 and self.router is None:
            raise RuntimeError(
                "V6 staged curriculum requires a dedicated router optimizer; none is registered."
            )
        return self.router

    def step(self) -> None:
        self.router.step()

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        self.router.zero_grad(set_to_none=set_to_none)
