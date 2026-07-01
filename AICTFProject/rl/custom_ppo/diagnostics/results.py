"""Typed diagnostic result states.

Every ``DiagnosticResult`` preserves the observed metric, threshold, sample
count, reason, and metric version so that a ``WARN`` or ``FAIL`` status always
has enough context to be interpreted without looking at secondary tables.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Callable, Generic, Optional, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class DiagnosticStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"


@dataclasses.dataclass(frozen=True)
class DiagnosticError:
    message: str
    exc_type: str = ""

    @classmethod
    def from_exception(cls, exc: Exception) -> "DiagnosticError":
        return cls(message=str(exc), exc_type=type(exc).__name__)


@dataclasses.dataclass(frozen=True)
class DiagnosticResult(Generic[T]):
    """Typed outcome of a single diagnostic check.

    Attributes
    ----------
    status:
        Categorical verdict: PASS / FAIL / WARN / INCONCLUSIVE / ERROR.
    value:
        The observed metric value (None on ERROR before metric could be computed).
    sample_count:
        Number of data points underlying the metric. Zero means the metric is
        unavailable, not that the result is "zero".
    reason:
        Human-readable explanation of the verdict, including the observed value
        and the threshold used for the decision.
    error:
        Populated only when ``status == ERROR``; carries exception details.
    """

    status: DiagnosticStatus
    value: Optional[T]
    sample_count: int = 0
    reason: Optional[str] = None
    error: Optional[DiagnosticError] = None

    @property
    def is_pass(self) -> bool:
        return self.status == DiagnosticStatus.PASS

    @property
    def is_fail(self) -> bool:
        return self.status == DiagnosticStatus.FAIL

    @property
    def is_available(self) -> bool:
        """True when there were enough samples to compute the metric."""
        return self.sample_count > 0 and self.status != DiagnosticStatus.ERROR

    def map(self, fn: Callable[[T], U]) -> "DiagnosticResult[U]":
        """Apply ``fn`` to ``value`` while preserving status, count, and reason."""
        new_value = fn(self.value) if self.value is not None else None
        return DiagnosticResult(
            status=self.status,
            value=new_value,
            sample_count=self.sample_count,
            reason=self.reason,
            error=self.error,
        )

    @classmethod
    def unavailable(cls, reason: str = "insufficient samples") -> "DiagnosticResult[None]":
        return cls(status=DiagnosticStatus.INCONCLUSIVE, value=None, sample_count=0, reason=reason)

    @classmethod
    def from_error(cls, exc: Exception) -> "DiagnosticResult[None]":
        return cls(
            status=DiagnosticStatus.ERROR,
            value=None,
            sample_count=0,
            error=DiagnosticError.from_exception(exc),
            reason=str(exc),
        )


__all__ = ["DiagnosticStatus", "DiagnosticResult", "DiagnosticError"]
