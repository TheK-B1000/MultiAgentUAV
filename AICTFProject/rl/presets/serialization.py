"""Canonical JSON serialization and deterministic hashing for presets.

The canonical representation is used to:
* Generate golden snapshots for regression testing.
* Compute a deterministic ``preset_hash`` recorded in run manifests.
* Write ``resolved_preset.json`` artifacts for every training run.

Hash exclusions
---------------
Fields that are machine-specific or purely cosmetic are excluded from the hash
(but are still written to the artifact for traceability):

* ``device`` — "cuda" or "cpu" varies by machine; never affects the model graph.
* ``run_tag`` — a label only; changing it does not alter training behavior.
* ``cli_preset`` — records which name the CLI was called with; not a config field.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from typing import Any

from rl.presets.models import PresetSerializationError

# Fields excluded from the preset hash (machine-specific or cosmetic).
_HASH_EXCLUDED_FIELDS: frozenset[str] = frozenset(
    {
        "device",
        "run_tag",
        "cli_preset",
    }
)

_SCHEMA_VERSION = 1


def _normalize_value(v: Any) -> Any:
    """Recursively normalize a value for canonical JSON representation."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            raise PresetSerializationError(
                f"Non-finite float in preset config: {v!r}"
            )
        return v
    if isinstance(v, tuple):
        return [_normalize_value(x) for x in v]
    if isinstance(v, list):
        return [_normalize_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _normalize_value(val) for k, val in sorted(v.items())}
    return v


def canonical_config_dict(cfg: Any) -> dict[str, Any]:
    """Return a JSON-safe, canonically ordered dict from a ``PPOConfig`` instance.

    * All keys are sorted.
    * Tuples are converted to lists (JSON has no tuple type).
    * ``None`` values are preserved as JSON ``null``.
    * Non-finite floats raise ``PresetSerializationError``.
    * Machine-specific fields are included (for traceability) but flagged.
    """
    try:
        raw = asdict(cfg)
    except Exception as exc:
        raise PresetSerializationError(
            f"Failed to convert config to dict: {exc}"
        ) from exc

    return {k: _normalize_value(v) for k, v in sorted(raw.items())}


def to_canonical_json_bytes(cfg_dict: dict[str, Any]) -> bytes:
    """Serialize a canonical config dict to stable UTF-8 JSON bytes."""
    return json.dumps(cfg_dict, sort_keys=True, indent=None, separators=(",", ":")).encode("utf-8")


def preset_hash(cfg: Any, *, exclude_fields: frozenset[str] = _HASH_EXCLUDED_FIELDS) -> str:
    """Return the SHA-256 hex digest of the preset's canonical config.

    Fields in ``exclude_fields`` are removed before hashing so the hash
    reflects only behavior-affecting configuration.
    """
    full_dict = canonical_config_dict(cfg)
    hash_dict = {k: v for k, v in full_dict.items() if k not in exclude_fields}
    payload = to_canonical_json_bytes(hash_dict)
    return hashlib.sha256(payload).hexdigest()


def resolved_preset_artifact(
    *,
    canonical_name: str,
    requested_name: str,
    cfg: Any,
    schema_version: int = _SCHEMA_VERSION,
    git_commit: str = "",
    validation_passed: bool = True,
    validation_errors: list[str] | None = None,
) -> dict[str, Any]:
    """Build the ``resolved_preset.json`` artifact dict for a training run.

    This dict is written once at run start and treated as immutable for the run.
    """
    cfg_dict = canonical_config_dict(cfg)
    p_hash = preset_hash(cfg)
    aliases_used = [] if requested_name == canonical_name else [requested_name]

    return {
        "_schema_version": schema_version,
        "canonical_name": canonical_name,
        "requested_name": requested_name,
        "aliases_used": aliases_used,
        "preset_hash": p_hash,
        "git_commit": git_commit,
        "validation_passed": validation_passed,
        "validation_errors": validation_errors or [],
        "resolved_config": cfg_dict,
    }


__all__ = [
    "SCHEMA_VERSION",
    "canonical_config_dict",
    "to_canonical_json_bytes",
    "preset_hash",
    "resolved_preset_artifact",
]

SCHEMA_VERSION = _SCHEMA_VERSION
