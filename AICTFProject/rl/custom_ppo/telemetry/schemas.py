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
    BENCHMARK = "benchmark"


def coerce_telemetry_mode(value: object) -> TrainingTelemetryMode:
    from rl.custom_ppo.telemetry.errors import TelemetryConfigurationError
    if isinstance(value, TrainingTelemetryMode):
        return value
    if value is None:
        return TrainingTelemetryMode.OFF
    text = str(value).strip().lower()
    try:
        return TrainingTelemetryMode(text)
    except ValueError as exc:
        raise TelemetryConfigurationError(
            f"Unknown training telemetry mode: {value!r}"
        ) from exc


__all__ = [
    "LATENT_METRICS_VERSION",
    "PERFORMANCE_METRICS_SCHEMA_VERSION",
    "TRAINING_EVENTS_SCHEMA_VERSION",
    "TRAINING_METRICS_SCHEMA_VERSION",
    "TrainingTelemetryMode",
    "coerce_telemetry_mode",
]
