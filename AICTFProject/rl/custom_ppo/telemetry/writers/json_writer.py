"""Structured JSONL event writer."""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any

from rl.custom_ppo.telemetry.events import TelemetryEvent
from rl.custom_ppo.telemetry.schemas import TRAINING_EVENTS_SCHEMA_VERSION
from rl.custom_ppo.telemetry.validation import validate_event


def event_to_record(event: TelemetryEvent) -> dict[str, Any]:
    validate_event(event)
    return {
        "schema_version": TRAINING_EVENTS_SCHEMA_VERSION,
        "event_type": type(event).__name__,
        "payload": dataclasses.asdict(event),
    }


class JSONLineEventWriter:
    def __init__(self, path: str) -> None:
        self.path = str(path)
        self._file: Any = None

    def emit(self, event: TelemetryEvent) -> None:
        if self._file is None:
            directory = os.path.dirname(os.path.abspath(self.path)) or "."
            os.makedirs(directory, exist_ok=True)
            self._file = open(self.path, "a", encoding="utf-8")
        self._file.write(json.dumps(event_to_record(event), sort_keys=True, separators=(",", ":")) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is None:
            return
        self._file.flush()
        self._file.close()
        self._file = None


__all__ = ["JSONLineEventWriter", "event_to_record"]
