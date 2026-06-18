"""Explicit parameter and optimizer ownership registries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


def classify_parameter_name(name: str) -> str | None:
    """Return ``actor``, ``critic``, ``router``, or ``None`` for unclassified params."""
    if "actor_cnn" in name or "latent_actor" in name:
        return "actor"
    if "critic" in name and "episode_strategy_value_head" not in name:
        return "critic"
    if (
        "strategy_encoder" in name
        or "strategy_aux_return_head" in name
        or "phase_predictor" in name
        or "selector_gru" in name
        or "episode_strategy_value_head" in name
    ):
        return "router"
    return None


@dataclass(frozen=True)
class ParameterRegistry:
    actor: tuple[torch.nn.Parameter, ...]
    critic: tuple[torch.nn.Parameter, ...]
    router: tuple[torch.nn.Parameter, ...]
    actor_names: tuple[str, ...]
    critic_names: tuple[str, ...]
    router_names: tuple[str, ...]

    @classmethod
    def from_model(cls, model: torch.nn.Module) -> ParameterRegistry:
        actor: list[torch.nn.Parameter] = []
        critic: list[torch.nn.Parameter] = []
        router: list[torch.nn.Parameter] = []
        actor_names: list[str] = []
        critic_names: list[str] = []
        router_names: list[str] = []
        for name, param in model.named_parameters():
            group = classify_parameter_name(name)
            if group == "actor":
                actor.append(param)
                actor_names.append(name)
            elif group == "critic":
                critic.append(param)
                critic_names.append(name)
            elif group == "router":
                router.append(param)
                router_names.append(name)
        reg = cls(
            actor=tuple(actor),
            critic=tuple(critic),
            router=tuple(router),
            actor_names=tuple(actor_names),
            critic_names=tuple(critic_names),
            router_names=tuple(router_names),
        )
        return reg

    def validate(self, model: torch.nn.Module) -> None:
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
        unclassified_trainable: list[str] = []
        for name, param in model.named_parameters():
            if param.requires_grad and classify_parameter_name(name) is None:
                unclassified_trainable.append(name)
        if unclassified_trainable:
            preview = ", ".join(unclassified_trainable[:8])
            suffix = "..." if len(unclassified_trainable) > 8 else ""
            raise ValueError(
                f"{len(unclassified_trainable)} trainable parameters are not in any owner group: "
                f"{preview}{suffix}"
            )
        trainable = {p for p in model.parameters() if p.requires_grad}
        unowned = trainable - owned
        if unowned:
            raise ValueError(
                f"{len(unowned)} trainable parameters have no owner in ParameterRegistry"
            )

    def apply_requires_grad_for_phase(self, model: torch.nn.Module, phase: str) -> None:
        policy = {
            "A": {"actor": True, "critic": True, "router": False},
            "B": {"actor": False, "critic": True, "router": True},
            "C": {"actor": True, "critic": True, "router": True},
        }.get(phase)
        if policy is None:
            raise ValueError(f"Unknown training phase {phase!r}")
        for name, param in model.named_parameters():
            group = classify_parameter_name(name)
            if group is None:
                param.requires_grad = False
                continue
            param.requires_grad = bool(policy[group])
        self.validate(model)


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

    def validate_against(self, registry: ParameterRegistry) -> None:
        """Every optimizer param must belong to exactly one declared owner group."""
        owner_map: dict[int, str] = {}
        for group_name, params in (
            ("actor", registry.actor),
            ("critic", registry.critic),
            ("router", registry.router),
        ):
            for param in params:
                owner_map[id(param)] = group_name
        for label, opt in (
            ("actor", self.actor),
            ("critic", self.critic),
            ("router", self.router),
        ):
            if opt is None:
                continue
            for group in opt.param_groups:
                for param in group["params"]:
                    owner = owner_map.get(id(param))
                    if owner is None:
                        raise ValueError(f"{label} optimizer references unowned parameter")
                    if owner != label:
                        raise ValueError(
                            f"{label} optimizer owns {owner} parameter; expected {label}"
                        )


def validate_model_parameter_ownership(model: torch.nn.Module) -> ParameterRegistry:
    """Build registry and assert complete, non-overlapping ownership."""
    return ParameterRegistry.from_model(model)
