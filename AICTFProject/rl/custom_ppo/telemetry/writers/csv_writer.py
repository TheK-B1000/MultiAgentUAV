"""Stable CSV helpers for optional performance artifacts."""

from __future__ import annotations

import csv
import os
from typing import Any, Iterable, Mapping


class StableCSVWriter:
    def __init__(self, path: str, fieldnames: Iterable[str]) -> None:
        self.path = str(path)
        self.fieldnames = list(fieldnames)

    def write_row(self, row: Mapping[str, Any]) -> None:
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        needs_header = not (os.path.isfile(self.path) and os.path.getsize(self.path) > 0)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            if needs_header:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in self.fieldnames})


__all__ = ["StableCSVWriter"]
