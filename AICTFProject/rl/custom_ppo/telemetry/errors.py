"""Typed telemetry errors."""

from __future__ import annotations


class TelemetryError(RuntimeError):
    pass


class TelemetryValidationError(TelemetryError):
    pass


class TelemetryConfigurationError(TelemetryError):
    pass


class TelemetrySerializationError(TelemetryError):
    pass


class TelemetryWriterError(TelemetryError):
    pass


class TelemetrySchemaError(TelemetryError):
    pass


class PerformanceMeasurementError(TelemetryError):
    pass


class GPUMonitorUnavailable(PerformanceMeasurementError):
    pass


class TrainingInterruptedSignal(Exception):
    pass


__all__ = [
    "GPUMonitorUnavailable",
    "PerformanceMeasurementError",
    "TelemetryError",
    "TelemetryConfigurationError",
    "TelemetrySchemaError",
    "TelemetrySerializationError",
    "TelemetryValidationError",
    "TelemetryWriterError",
    "TrainingInterruptedSignal",
]
