"""Version constants and canonical telemetry mode names."""

from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover
    from enum import Enum

    class StrEnum(str, Enum):
        pass


TRAINING_METRICS_SCHEMA_VERSION = 1
PERFORMANCE_METRICS_SCHEMA_VERSION = 1
LATENT_METRICS_VERSION = "v6_legacy_equivalent"
TRAINING_EVENTS_SCHEMA_VERSION = 1


class TrainingTelemetryMode(StrEnum):
    OFF = "off"
    BASIC = "basic"
    FULL = "full"


def coerce_telemetry_mode(value: object) -> TrainingTelemetryMode:
    if isinstance(value, TrainingTelemetryMode):
        return value
    text = str(value or TrainingTelemetryMode.FULL.value).strip().lower()
    try:
        return TrainingTelemetryMode(text)
    except ValueError:
        return TrainingTelemetryMode.FULL


__all__ = [
    "LATENT_METRICS_VERSION",
    "PERFORMANCE_METRICS_SCHEMA_VERSION",
    "TRAINING_EVENTS_SCHEMA_VERSION",
    "TRAINING_METRICS_SCHEMA_VERSION",
    "TrainingTelemetryMode",
    "coerce_telemetry_mode",
]
