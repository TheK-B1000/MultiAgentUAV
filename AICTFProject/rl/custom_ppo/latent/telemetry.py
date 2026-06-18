"""Event counters and invariants for forced-episode telemetry."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LatentRolloutTelemetry:
    completed_episode_count: int = 0
    forced_episode_count: int = 0
    forced_count_by_z: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.int64))
    missing_episode_record_count: int = 0

    def validate_forced_invariants(self) -> None:
        if int(self.forced_count_by_z.sum()) != int(self.forced_episode_count):
            raise AssertionError("sum(forced_count_by_z) != forced_episode_count")
        if int(self.forced_episode_count) > int(self.completed_episode_count):
            raise AssertionError("forced_episode_count > completed_episode_count")
