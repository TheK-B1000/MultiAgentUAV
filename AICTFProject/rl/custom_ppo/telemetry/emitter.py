"""Telemetry event publication interfaces."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Protocol, Optional

from rl.custom_ppo.telemetry.events import TelemetryEvent, TelemetryEnvelope
from rl.custom_ppo.telemetry.errors import TelemetryValidationError, TelemetryWriterError
from rl.custom_ppo.telemetry.validation import validate_envelope

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SinkFailure:
    sink_name: str
    exception_type: str
    message: str
    sequence: Optional[int]
    timestamp_seconds: float


class TelemetrySink(Protocol):
    def emit(self, envelope: TelemetryEnvelope) -> None:
        ...


class NullTelemetrySink:
    def emit(self, envelope: TelemetryEnvelope) -> None:
        return None


class CompositeTelemetrySink:
    def __init__(self, sinks: Iterable[TelemetrySink]) -> None:
        self.sinks = tuple(sinks)

    def emit(self, envelope: TelemetryEnvelope) -> None:
        for sink in self.sinks:
            sink.emit(envelope)


class BufferedTelemetrySink:
    def __init__(self, maxlen: int = 1024) -> None:
        self.envelopes: Deque[TelemetryEnvelope] = deque(maxlen=max(1, int(maxlen)))

    def emit(self, envelope: TelemetryEnvelope) -> None:
        self.envelopes.append(envelope)

    def drain(self) -> list[TelemetryEnvelope]:
        envelopes = list(self.envelopes)
        self.envelopes.clear()
        return envelopes


class SafeTelemetrySink:
    def __init__(self, sink: TelemetrySink) -> None:
        self.sink = sink
        self._degraded = False
        self._failures: list[SinkFailure] = []

    def emit(self, envelope: TelemetryEnvelope) -> None:
        # Validate envelope first. If validation fails, let it raise (integrity failure).
        validate_envelope(envelope)

        if self._degraded:
            return

        try:
            self.sink.emit(envelope)
        except (OSError, PermissionError, TelemetryWriterError) as exc:
            self._degraded = True
            failure = SinkFailure(
                sink_name=type(self.sink).__name__,
                exception_type=type(exc).__name__,
                message=str(exc),
                sequence=envelope.sequence,
                timestamp_seconds=float(time.time()),
            )
            self._failures.append(failure)
            logger.warning(
                f"Telemetry sink {failure.sink_name} failed with {failure.exception_type}: {failure.message}. "
                "Transitioning sink to DEGRADED state. Telemetry is now disabled for this sink."
            )


__all__ = [
    "BufferedTelemetrySink",
    "CompositeTelemetrySink",
    "NullTelemetrySink",
    "SafeTelemetrySink",
    "TelemetrySink",
    "SinkFailure",
]
