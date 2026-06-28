"""Typed error hierarchy for the training orchestration layer.

All training-specific errors inherit from :class:`TrainingConfigError` (for
config problems) or are standalone subclasses of stdlib exceptions. Import
these from here rather than raising bare ``ValueError``/``RuntimeError`` so
callers can handle them precisely.
"""

from __future__ import annotations


class TrainingConfigError(ValueError):
    """Raised when a resolved ``PPOConfig`` violates an invariant required for training."""


class EvaluationOnlyPresetError(TrainingConfigError):
    """Raised when an evaluation-only preset is used to start a PPO training run.

    The preset's ``evaluation_only_preset`` flag is ``True`` and must not enter
    the PPO training path; use the designated evaluation runner instead.
    """


class PresetsConflictError(TrainingConfigError):
    """Raised when a CLI override is incompatible with a preset's contract.

    Example: ``v4i1`` / ``v4i3`` / ``v4i4post`` require opponent_pool == {OP5, OP6, OP7}
    exactly; passing ``--opponent-pool`` with a different set violates that.
    """


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when a ``--load`` / ``--resume`` checkpoint path does not exist at runtime."""


class TrainingAbortedError(RuntimeError):
    """Raised when training is aborted in a way that is not a clean ``KeyboardInterrupt``.

    ``KeyboardInterrupt`` is re-raised directly; this class is for programmatic
    cancellation or watchdog-initiated stops.
    """
