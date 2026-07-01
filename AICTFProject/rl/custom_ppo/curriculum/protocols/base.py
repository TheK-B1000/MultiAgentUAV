"""Gate protocol interface for staged V6 curriculum families."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.curriculum.context import GateContext
from rl.custom_ppo.curriculum.types import GateResult


@runtime_checkable
class GateProtocol(Protocol):
    """Versioned gate-family orchestration contract."""

    @property
    def version(self) -> str:
        ...

    def required_families(self) -> tuple[str, ...]:
        ...

    def evaluate_online(self, context: GateContext) -> dict[str, GateResult]:
        ...

    def evaluate_boundary(self, context: GateContext) -> dict[str, GateResult]:
        ...

    def build_ranking(
        self,
        *,
        gate_results: dict[str, GateResult],
        online_report: dict[str, Any],
        matched_report: dict[str, Any],
        probe_report: dict[str, Any],
        global_step: int,
        cfg: PPOConfig,
    ) -> dict[str, Any]:
        ...


__all__ = ["GateProtocol"]
