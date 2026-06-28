"""Version constants and canonical telemetry mode names."""

from __future__ import annotations

TRAINING_METRICS_SCHEMA_VERSION = 1
PERFORMANCE_METRICS_SCHEMA_VERSION = 1
LATENT_METRICS_VERSION = "v6_legacy_equivalent"
TRAINING_EVENTS_SCHEMA_VERSION = 1

# Re-export from the standalone module to avoid a circular import through
# rl.config.ppo_config → rl.custom_ppo.telemetry.schemas → rl.custom_ppo.__init__.
from rl.telemetry_mode import TrainingTelemetryMode  # noqa: E402


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
