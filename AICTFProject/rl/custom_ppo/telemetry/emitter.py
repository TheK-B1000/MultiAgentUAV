"""Telemetry event publication interfaces."""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Protocol

from rl.custom_ppo.telemetry.events import TelemetryEvent
from rl.custom_ppo.telemetry.errors import TelemetryWriterError


class TelemetrySink(Protocol):
    def emit(self, event: TelemetryEvent) -> None:
        ...


class NullTelemetrySink:
    def emit(self, event: TelemetryEvent) -> None:
        return None


class CompositeTelemetrySink:
    def __init__(self, sinks: Iterable[TelemetrySink]) -> None:
        self.sinks = tuple(sinks)

    def emit(self, event: TelemetryEvent) -> None:
        for sink in self.sinks:
            sink.emit(event)


class BufferedTelemetrySink:
    def __init__(self, maxlen: int = 1024) -> None:
        self.events: Deque[TelemetryEvent] = deque(maxlen=max(1, int(maxlen)))

    def emit(self, event: TelemetryEvent) -> None:
        self.events.append(event)

    def drain(self) -> list[TelemetryEvent]:
        events = list(self.events)
        self.events.clear()
        return events


class SafeTelemetrySink:
    def __init__(self, sink: TelemetrySink) -> None:
        self.sink = sink
        self.failures: list[Exception] = []

    def emit(self, event: TelemetryEvent) -> None:
        try:
            self.sink.emit(event)
        except Exception as exc:
            self.failures.append(TelemetryWriterError(str(exc)))


__all__ = [
    "BufferedTelemetrySink",
    "CompositeTelemetrySink",
    "NullTelemetrySink",
    "SafeTelemetrySink",
    "TelemetrySink",
]
