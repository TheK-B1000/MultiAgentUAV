"""Single source of truth for staged curriculum phase and trainability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_VALID_PHASES = frozenset({"A", "B", "C", "__legacy__"})


def resolve_training_phase(runtime: Any, *, global_step: int | None = None) -> str:
    """Resolve the training phase once per update."""
    curriculum = getattr(runtime, "v6i1_curriculum", None)
    if curriculum is None:
        return "__legacy__"
    step = int(global_step if global_step is not None else getattr(runtime, "global_step", 0))
    phase = str(curriculum.resolve_phase(step))
    if phase not in _VALID_PHASES - {"__legacy__"}:
        raise ValueError(
            f"Unknown training phase {phase!r}; expected one of {sorted(_VALID_PHASES - {'__legacy__'})}"
        )
    schedule_phase = phase
    if curriculum is not None:
        from rl.custom_ppo.v6i1_phase_runtime import v6i1_schedule_context

        schedule_phase, _, _, _ = v6i1_schedule_context(runtime)
        if str(schedule_phase) != str(phase):
            raise RuntimeError(
                f"Phase mismatch: curriculum={phase!r} schedule={schedule_phase!r}"
            )
    return phase


@dataclass(frozen=True)
class PhaseTrainingPolicy:
    phase: str
    actor_trainable: bool
    critic_trainable: bool
    router_trainable: bool
    counterfactual_active: bool
    actor_step_enabled: bool
    critic_step_enabled: bool
    router_step_enabled: bool

    @classmethod
    def from_phase(cls, phase: str) -> PhaseTrainingPolicy:
        if phase == "A":
            return cls(
                phase=phase,
                actor_trainable=True,
                critic_trainable=True,
                router_trainable=False,
                counterfactual_active=True,
                actor_step_enabled=True,
                critic_step_enabled=True,
                router_step_enabled=False,
            )
        if phase == "B":
            return cls(
                phase=phase,
                actor_trainable=False,
                critic_trainable=True,
                router_trainable=True,
                counterfactual_active=False,
                actor_step_enabled=False,
                critic_step_enabled=True,
                router_step_enabled=True,
            )
        if phase == "C":
            return cls(
                phase=phase,
                actor_trainable=True,
                critic_trainable=True,
                router_trainable=True,
                counterfactual_active=True,
                actor_step_enabled=True,
                critic_step_enabled=True,
                router_step_enabled=True,
            )
        return cls(
            phase=phase,
            actor_trainable=True,
            critic_trainable=True,
            router_trainable=True,
            counterfactual_active=True,
            actor_step_enabled=True,
            critic_step_enabled=True,
            router_step_enabled=True,
        )


def apply_phase_requires_grad(model: Any, phase: str) -> None:
    """Freeze/unfreeze parameter groups for curriculum phase."""
    from rl.custom_ppo.ppo_updater import set_model_requires_grad_for_phase

    if phase in {"A", "B", "C"}:
        set_model_requires_grad_for_phase(model, phase)
