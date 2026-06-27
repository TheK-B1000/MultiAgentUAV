"""Typed models for the preset system.

Provides immutable identity types, status classification, and typed errors.
The resolved configuration remains a ``PPOConfig`` instance — these types
describe WHAT a preset is, not the full resolved field set.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional, Tuple

if TYPE_CHECKING:
    from rl.train_ppo import PPOConfig


class PresetStatus(str, Enum):
    ACTIVE = "ACTIVE"
    COMPATIBILITY_ALIAS = "COMPATIBILITY_ALIAS"
    HISTORICAL_REPRODUCTION = "HISTORICAL_REPRODUCTION"
    EXPERIMENTAL = "EXPERIMENTAL"
    DEPRECATED = "DEPRECATED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class PresetIdentity:
    name: str
    family: str
    version: int
    description: str
    aliases: Tuple[str, ...]
    predecessor: Optional[str]
    status: PresetStatus


@dataclass(frozen=True)
class PresetDefinition:
    identity: PresetIdentity
    apply_fn: Callable  # (PPOConfig) -> PPOConfig — mutates and returns cfg


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------

class PresetError(Exception):
    """Base for all preset-system errors."""


class PresetNotFoundError(PresetError):
    """Raised when a preset name or alias is not registered."""


class DuplicatePresetError(PresetError):
    """Raised when two definitions share the same canonical name."""


class DuplicatePresetAliasError(PresetError):
    """Raised when an alias would shadow another canonical name or alias."""


class PresetValidationError(PresetError):
    """Raised when a resolved config violates a cross-field invariant.

    Attributes
    ----------
    preset_name: str
        Name used to resolve the preset (may be an alias).
    field_path: str
        Dot-separated path to the offending field.
    observed: object
        The actual value found.
    constraint: str
        Human-readable description of what was required.
    """

    def __init__(
        self,
        message: str,
        *,
        preset_name: str = "",
        field_path: str = "",
        observed: object = None,
        constraint: str = "",
    ) -> None:
        super().__init__(message)
        self.preset_name = preset_name
        self.field_path = field_path
        self.observed = observed
        self.constraint = constraint

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.preset_name:
            parts.append(f"preset={self.preset_name!r}")
        if self.field_path:
            parts.append(f"field={self.field_path!r}")
        if self.observed is not None:
            parts.append(f"observed={self.observed!r}")
        if self.constraint:
            parts.append(f"constraint={self.constraint!r}")
        return " | ".join(parts)


class PresetSerializationError(PresetError):
    """Raised when a preset config cannot be serialized to canonical JSON."""


class PresetCompatibilityError(PresetError):
    """Raised when a legacy caller requests a preset in an incompatible way."""


__all__ = [
    "PresetStatus",
    "PresetIdentity",
    "PresetDefinition",
    "PresetError",
    "PresetNotFoundError",
    "DuplicatePresetError",
    "DuplicatePresetAliasError",
    "PresetValidationError",
    "PresetSerializationError",
    "PresetCompatibilityError",
]
