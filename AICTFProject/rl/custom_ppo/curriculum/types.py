"""Curriculum gate types, pair indexing, and promotion helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rl.custom_ppo.gate_protocol import GATE_FAMILY_NAMES_V6I1

# Fixed pair index for forced_z_pair_jsd_{i}: (0,1)=0 … (2,3)=5
LATENT_PAIR_INDEX: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
)
PAIR_ORDER = LATENT_PAIR_INDEX

# Default v6i1 family list; protocol-specific lists come from gate_protocol.gate_family_names.
GATE_FAMILY_NAMES: tuple[str, ...] = GATE_FAMILY_NAMES_V6I1


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"
    ERROR = "ERROR"

    @classmethod
    def validate(cls, value: str) -> str:
        normalized = str(value)
        if normalized not in {member.value for member in cls}:
            raise ValueError(
                f"gate status must be one of {[m.value for m in cls]!r}, got {normalized!r}"
            )
        return normalized


class CurriculumPhase(str, Enum):
    A = "A"
    B = "B"
    C = "C"


class GateMode(str, Enum):
    ENFORCE = "enforce"
    OBSERVE_ONLY = "observe_only"

    @classmethod
    def normalize(cls, value: str) -> str:
        normalized = str(value).lower()
        if normalized not in {member.value for member in cls}:
            raise ValueError(
                f"gate mode must be one of {[m.value for m in cls]!r}, got {normalized!r}"
            )
        return normalized


@dataclass(frozen=True)
class GateResult:
    """Immutable gate-family outcome with validated status."""

    status: str
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", GateStatus.validate(self.status))
        if not isinstance(self.details, dict):
            object.__setattr__(self, "details", dict(self.details))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status}
        if self.reason:
            out["reason"] = self.reason
        if self.details:
            out.update(self.details)
        return out

    @property
    def passed(self) -> bool:
        return self.status == GateStatus.PASS.value

    @property
    def measured(self) -> bool:
        return self.status in (GateStatus.PASS.value, GateStatus.FAIL.value)


class GateFamilyResult(GateResult):
    """Backward-compatible mutable alias for legacy curriculum_gates imports."""


def gate_family_result_from_bool(
    passed: bool,
    *,
    details: dict[str, Any] | None = None,
) -> GateFamilyResult:
    return GateFamilyResult(
        status=GateStatus.PASS.value if passed else GateStatus.FAIL.value,
        details=dict(details or {}),
    )


def count_gate_families_passed(
    gate_results: dict[str, GateFamilyResult],
    *,
    families: tuple[str, ...] | None = None,
) -> int:
    names = families if families is not None else GATE_FAMILY_NAMES
    return sum(
        1
        for name in names
        if gate_results.get(name, GateFamilyResult(GateStatus.NOT_RUN.value)).passed
    )


def count_gate_families_measured(
    gate_results: dict[str, GateFamilyResult],
    *,
    families: tuple[str, ...] | None = None,
) -> int:
    names = families if families is not None else GATE_FAMILY_NAMES
    return sum(
        1
        for name in names
        if gate_results.get(name, GateFamilyResult(GateStatus.NOT_RUN.value)).measured
    )


def all_required_families_passed(
    gate_results: dict[str, GateFamilyResult],
    *,
    families: tuple[str, ...] | None = None,
) -> bool:
    """True when every required family has status PASS (independent of gate mode)."""
    names = families if families is not None else GATE_FAMILY_NAMES
    for name in names:
        result = gate_results.get(name, GateFamilyResult(GateStatus.NOT_RUN.value))
        if result.status != GateStatus.PASS.value:
            return False
    return True


def overall_gate_passed_for_promotion(
    gate_results: dict[str, GateFamilyResult],
    *,
    mode: str,
    families: tuple[str, ...] | None = None,
) -> bool:
    """In enforce mode every family must be PASS; NOT_RUN and ERROR block promotion."""
    if GateMode.normalize(mode) != GateMode.ENFORCE.value:
        return False
    return all_required_families_passed(gate_results, families=families)


@dataclass
class GateAttempt:
    """Structured record of a single Phase A gate evaluation."""

    global_step: int
    phase: str
    checkpoint: str
    gate_protocol_version: str
    required_families: tuple[str, ...]
    mode: str
    boundary_enabled: bool
    probe_enabled: bool
    gate_results: dict[str, GateFamilyResult]
    online_report: dict[str, Any]
    matched_report: dict[str, Any]
    probe_report: dict[str, Any]
    ranking_components: dict[str, Any]
    gate_passed: bool
    promotion_allowed: bool
    overall_gate_passed: bool
    promoted_to_phase_b: bool = False
    nominal_transition_to_phase_b: bool = False
    phase_a_gate_passed: bool = False
    phase_a_end_step: int | None = None


# Legacy string constants for callers that import GATE_STATUS_* from curriculum_gates.
GATE_STATUS_PASS = GateStatus.PASS.value
GATE_STATUS_FAIL = GateStatus.FAIL.value
GATE_STATUS_NOT_RUN = GateStatus.NOT_RUN.value
GATE_STATUS_ERROR = GateStatus.ERROR.value


__all__ = [
    "GATE_FAMILY_NAMES",
    "GATE_STATUS_ERROR",
    "GATE_STATUS_FAIL",
    "GATE_STATUS_NOT_RUN",
    "GATE_STATUS_PASS",
    "LATENT_PAIR_INDEX",
    "PAIR_ORDER",
    "CurriculumPhase",
    "GateAttempt",
    "GateFamilyResult",
    "GateMode",
    "GateResult",
    "GateStatus",
    "all_required_families_passed",
    "count_gate_families_measured",
    "count_gate_families_passed",
    "gate_family_result_from_bool",
    "overall_gate_passed_for_promotion",
]
