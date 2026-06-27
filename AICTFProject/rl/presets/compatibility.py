"""Backward-compatibility shims for the preset system.

Callers that import directly from ``rl.presets`` or ``rl.presets.plan_faithful``
continue to work unchanged.  This module provides re-exports for common typed
symbols so that new code can import from a single, stable location.

Nothing here changes behavior — it is a thin re-export layer only.
"""
from __future__ import annotations

# Typed errors
from rl.presets.models import (
    DuplicatePresetAliasError,
    DuplicatePresetError,
    PresetCompatibilityError,
    PresetDefinition,
    PresetError,
    PresetIdentity,
    PresetNotFoundError,
    PresetSerializationError,
    PresetStatus,
    PresetValidationError,
)

# Registry
from rl.presets.registry import PresetRegistry, build_registry_from_dict, get_registry

# Serialization utilities
from rl.presets.serialization import (
    SCHEMA_VERSION,
    canonical_config_dict,
    preset_hash,
    resolved_preset_artifact,
    to_canonical_json_bytes,
)

# Validation
from rl.presets.validation import assert_preset_valid, validate_preset

__all__ = [
    # errors
    "PresetError",
    "PresetNotFoundError",
    "DuplicatePresetError",
    "DuplicatePresetAliasError",
    "PresetValidationError",
    "PresetSerializationError",
    "PresetCompatibilityError",
    # identity
    "PresetStatus",
    "PresetIdentity",
    "PresetDefinition",
    # registry
    "PresetRegistry",
    "build_registry_from_dict",
    "get_registry",
    # serialization
    "SCHEMA_VERSION",
    "canonical_config_dict",
    "to_canonical_json_bytes",
    "preset_hash",
    "resolved_preset_artifact",
    # validation
    "validate_preset",
    "assert_preset_valid",
]
