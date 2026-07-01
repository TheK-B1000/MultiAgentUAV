"""Standalone telemetry-mode enum with no rl.custom_ppo dependencies.

Defined here so that rl.config.ppo_config can import it without triggering
the rl.custom_ppo package initializer, which would cause a circular import
(ppo_config → rl.custom_ppo → trainer → communication → gates → ppo_config).
"""
from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        pass


class TrainingTelemetryMode(StrEnum):
    OFF = "off"
    BASIC = "basic"
    FULL = "full"
    BENCHMARK = "benchmark"


__all__ = ["TrainingTelemetryMode"]
