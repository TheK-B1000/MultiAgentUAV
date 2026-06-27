"""CUDA-safe coarse timing helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PhaseTiming:
    phase: str
    duration_seconds: float
    timing_method: str
    cuda_synchronized: bool = False


class PhaseTimer:
    def __init__(
        self,
        phase: str,
        *,
        timing_method: str = "wall_clock",
        cuda_synchronized: bool = False,
    ) -> None:
        self.phase = phase
        self.timing_method = timing_method
        self.cuda_synchronized = bool(cuda_synchronized)
        self._start: Optional[float] = None
        self.result: Optional[PhaseTiming] = None

    def __enter__(self) -> "PhaseTimer":
        if self.cuda_synchronized:
            self._synchronize_cuda()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.cuda_synchronized:
            self._synchronize_cuda()
        end = time.perf_counter()
        start = self._start if self._start is not None else end
        self.result = PhaseTiming(
            phase=self.phase,
            duration_seconds=max(0.0, end - start),
            timing_method=self.timing_method,
            cuda_synchronized=self.cuda_synchronized,
        )

    @staticmethod
    def _synchronize_cuda() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            return


__all__ = ["PhaseTimer", "PhaseTiming"]
