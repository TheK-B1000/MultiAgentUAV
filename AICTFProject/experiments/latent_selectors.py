"""Explicit latent-selector specs for router diagnostic / trace-audit runs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from rl.evaluation.router_ablation import configure_condition
from rl.evaluation.types import EvalCondition


class LatentSelector(Protocol):
    """Configure one episode's latent-selection mode on an inference policy."""

    name: str
    selection_mode: str

    def apply(self, model: Any, *, shuffled_mapping: dict[Any, Any] | None = None) -> None:
        ...

    def expected_rule(self) -> str:
        ...


@dataclass(frozen=True)
class LearnedRouterSelector:
    """q_phi switching at router opportunities."""

    strategy_interval: int
    name: str = "learned_router"
    selection_mode: str = "qphi_switching"

    @property
    def condition(self) -> EvalCondition:
        return EvalCondition(
            name="learned_qphi_switching",
            selection_rule="qphi",
            strategy_interval=int(self.strategy_interval),
            allow_switching=True,
            description="Learned q_phi at every router opportunity.",
        )

    def apply(self, model: Any, *, shuffled_mapping: dict[Any, Any] | None = None) -> None:
        if hasattr(model, "clear_eval_suite_state"):
            model.clear_eval_suite_state()
        configure_condition(model, self.condition)

    def expected_rule(self) -> str:
        return "qphi"


@dataclass(frozen=True)
class FixedLatentSelector:
    """Clamp all decisions to one z."""

    latent_id: int
    name: str = "fixed_z"
    selection_mode: str = "fixed_latent"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", f"fixed_z{int(self.latent_id)}")

    @property
    def condition(self) -> EvalCondition:
        z = int(self.latent_id)
        return EvalCondition(
            name=f"fixed_z{z}",
            selection_rule=f"fixed_z{z}",
            strategy_interval=0,
            allow_switching=False,
            fixed_latent_id=z,
            description=f"Clamp all decisions to z={z}.",
        )

    def apply(self, model: Any, *, shuffled_mapping: dict[Any, Any] | None = None) -> None:
        if hasattr(model, "clear_eval_suite_state"):
            model.clear_eval_suite_state()
        configure_condition(model, self.condition)

    def expected_rule(self) -> str:
        return f"fixed_z{int(self.latent_id)}"


@dataclass(frozen=True)
class UniformSelector:
    """Uniform z at router opportunities (isolated selector RNG)."""

    strategy_interval: int
    name: str = "uniform_z"
    selection_mode: str = "uniform_random"

    @property
    def condition(self) -> EvalCondition:
        return EvalCondition(
            name="uniform_random_at_router_opportunities",
            selection_rule="uniform",
            strategy_interval=int(self.strategy_interval),
            allow_switching=True,
            description="Uniform z at router opportunities.",
        )

    def apply(self, model: Any, *, shuffled_mapping: dict[Any, Any] | None = None) -> None:
        if hasattr(model, "clear_eval_suite_state"):
            model.clear_eval_suite_state()
        configure_condition(model, self.condition)

    def expected_rule(self) -> str:
        return "uniform"


@dataclass(frozen=True)
class ShuffledAssignmentSelector:
    """Shuffled q_phi outputs with injected mapping from learned traces."""

    strategy_interval: int
    name: str = "shuffled_router"
    selection_mode: str = "shuffled_qphi"

    @property
    def condition(self) -> EvalCondition:
        return EvalCondition(
            name="shuffled_qphi_outputs",
            selection_rule="shuffled_qphi",
            strategy_interval=int(self.strategy_interval),
            allow_switching=True,
            description="Shuffled q_phi outputs preserving marginal occupancy.",
        )

    def apply(self, model: Any, *, shuffled_mapping: dict[Any, Any] | None = None) -> None:
        if hasattr(model, "clear_eval_suite_state"):
            model.clear_eval_suite_state()
        configure_condition(model, self.condition)
        if shuffled_mapping is None:
            raise ValueError("ShuffledAssignmentSelector requires shuffled_mapping")
        if not hasattr(model, "inject_shuffled_mapping"):
            raise AttributeError("Policy does not support inject_shuffled_mapping")
        model.inject_shuffled_mapping(shuffled_mapping)

    def expected_rule(self) -> str:
        return "shuffled_qphi"


__all__ = [
    "FixedLatentSelector",
    "LatentSelector",
    "LearnedRouterSelector",
    "ShuffledAssignmentSelector",
    "UniformSelector",
]
