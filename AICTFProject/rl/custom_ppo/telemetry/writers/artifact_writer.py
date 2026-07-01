"""Versioned JSON artifact writer."""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Mapping


class ArtifactWriter:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = str(output_dir)

    def write_json(self, filename: str, payload: Mapping[str, Any] | object) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)
        if dataclasses.is_dataclass(payload):
            data: Any = dataclasses.asdict(payload)
        else:
            data = dict(payload) if isinstance(payload, Mapping) else payload
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, sort_keys=True, indent=2)
            f.write("\n")
        return path


__all__ = ["ArtifactWriter"]
