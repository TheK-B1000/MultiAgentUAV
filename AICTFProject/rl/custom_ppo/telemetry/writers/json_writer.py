"""Structured JSONL event writer."""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any

from rl.custom_ppo.telemetry.events import TelemetryEnvelope
from rl.custom_ppo.telemetry.errors import TelemetryWriterError


def _assert_safe_payload(val: Any) -> None:
    import torch
    import numpy as np
    if isinstance(val, (torch.Tensor, np.ndarray)):
        raise ValueError(f"Telemetry payload contains raw tensor/array: {type(val)}")
    if isinstance(val, dict):
        for k, v in val.items():
            if k in ("observation", "observations", "state_dict", "model_state_dict"):
                raise ValueError(f"Telemetry payload contains forbidden key: {k}")
            _assert_safe_payload(v)
    elif isinstance(val, (list, tuple)):
        for item in val:
            _assert_safe_payload(item)


def _check_depth(val: Any, depth: int = 0) -> None:
    if depth > 10:
        raise ValueError("Telemetry payload serialization depth limit exceeded")
    if isinstance(val, dict):
        for v in val.values():
            _check_depth(v, depth + 1)
    elif isinstance(val, (list, tuple)):
        for item in val:
            _check_depth(item, depth + 1)


def envelope_to_record(envelope: TelemetryEnvelope) -> dict[str, Any]:
    _check_depth(envelope.payload)
    payload_dict = dataclasses.asdict(envelope.payload)
    _assert_safe_payload(payload_dict)
    
    return {
        "schema_version": int(envelope.schema_version),
        "event_type": str(envelope.event_type),
        "run_id": str(envelope.run_id),
        "sequence": int(envelope.sequence),
        "timestamp_seconds": float(envelope.timestamp_seconds),
        "payload": payload_dict,
    }


class JSONLineEventWriter:
    def __init__(self, path: str) -> None:
        self.path = str(path)
        self._file: Any = None

    def emit(self, envelope: TelemetryEnvelope) -> None:
        if self._file is None:
            directory = os.path.dirname(os.path.abspath(self.path)) or "."
            os.makedirs(directory, exist_ok=True)
            self._file = open(self.path, "a", encoding="utf-8")
        try:
            record = envelope_to_record(envelope)
            serialized = json.dumps(record, sort_keys=True, separators=(",", ":"))
            
            # String level pattern checks
            if "observation" in serialized or "state_dict" in serialized:
                raise ValueError("Serialized event contains forbidden pattern ('observation' or 'state_dict')")
                
            self._file.write(serialized + "\n")
            self._file.flush()
        except Exception as exc:
            raise TelemetryWriterError(str(exc)) from exc

    def close(self) -> None:
        if self._file is None:
            return
        try:
            self._file.flush()
            self._file.close()
            print(f"[DEBUG] JSONLineEventWriter: closed file at {self.path}")
        except Exception as e:
            print(f"[DEBUG] JSONLineEventWriter: failed to close file at {self.path}: {e}")
            pass
        self._file = None


__all__ = ["JSONLineEventWriter", "envelope_to_record"]
