"""Explicit parameter and optimizer ownership registries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class ParameterRegistry:
    actor: tuple[torch.nn.Parameter, ...]
    critic: tuple[torch.nn.Parameter, ...]
    router: tuple[torch.nn.Parameter, ...]

    @classmethod
    def from_model(cls, model: torch.nn.Module) -> ParameterRegistry:
        actor: list[torch.nn.Parameter] = []
        critic: list[torch.nn.Parameter] = []
        router: list[torch.nn.Parameter] = []
        for name, p in model.named_parameters():
            is_actor = "actor_cnn" in name or "latent_actor" in name
            is_critic = "critic" in name and "episode_strategy_value_head" not in name
            is_router = (
                "strategy_encoder" in name
                or "strategy_aux_return_head" in name
                or "phase_predictor" in name
                or "selector_gru" in name
                or "episode_strategy_value_head" in name
            )
            if is_actor:
                actor.append(p)
            elif is_critic:
                critic.append(p)
            elif is_router:
                router.append(p)
        reg = cls(
            actor=tuple(actor),
            critic=tuple(critic),
            router=tuple(router),
        )
        reg.validate(model)
        return reg

    def validate(self, model: torch.nn.Module, *, strict: bool = False) -> None:
        actor_set = set(self.actor)
        critic_set = set(self.critic)
        router_set = set(self.router)
        if actor_set & critic_set:
            raise ValueError("actor and critic parameter sets overlap")
        if actor_set & router_set:
            raise ValueError("actor and router parameter sets overlap")
        if critic_set & router_set:
            raise ValueError("critic and router parameter sets overlap")
        owned = actor_set | critic_set | router_set
        trainable = {p for p in model.parameters() if p.requires_grad}
        unowned = trainable - owned
        if unowned:
            raise ValueError(
                f"{len(unowned)} trainable parameters have no owner in ParameterRegistry"
            )


@dataclass(frozen=True)
class OptimizerRegistry:
    actor: torch.optim.Optimizer | None
    critic: torch.optim.Optimizer | None
    router: torch.optim.Optimizer | None
    shared: torch.optim.Optimizer | None

    @classmethod
    def from_runtime(cls, runtime: object, shared: torch.optim.Optimizer) -> OptimizerRegistry:
        return cls(
            actor=getattr(runtime, "actor_optimizer", None),
            critic=getattr(runtime, "critic_optimizer", None),
            router=getattr(runtime, "router_optimizer", None),
            shared=shared,
        )

    def active_optimizers(self) -> Iterable[torch.optim.Optimizer]:
        for opt in (self.actor, self.critic, self.router, self.shared):
            if opt is not None:
                yield opt
