"""Evaluation-specific exceptions.

These errors let orchestration code distinguish configuration, checkpoint,
preflight, execution, and artifact failures without changing evaluator
verdict semantics.
"""
from __future__ import annotations


class EvaluationError(RuntimeError):
    """Base class for evaluation pipeline failures."""


class EvaluationConfigError(EvaluationError, ValueError):
    """Invalid or inconsistent evaluation configuration."""


class EvaluationCheckpointError(EvaluationError):
    """Checkpoint metadata, shape, or loading failure."""


class EvaluationPreflightError(EvaluationError, TypeError):
    """A required preflight contract failed before evaluation episodes ran."""


class EvaluationArtifactError(EvaluationError):
    """Artifact writing or artifact validation failed."""


class EvaluationManifestError(EvaluationError):
    """Manifest lifecycle or atomic-write invariant failed."""


__all__ = [
    "EvaluationArtifactError",
    "EvaluationCheckpointError",
    "EvaluationConfigError",
    "EvaluationError",
    "EvaluationManifestError",
    "EvaluationPreflightError",
]
