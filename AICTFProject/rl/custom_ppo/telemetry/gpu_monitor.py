"""Optional low-frequency GPU monitor."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from rl.custom_ppo.telemetry.errors import GPUMonitorUnavailable


@dataclass(frozen=True)
class GPUSample:
    timestamp_seconds: float
    utilization_percent: Optional[float]
    memory_device_used_bytes: Optional[int]


class NullGPUMonitor:
    status = "unavailable"

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def samples(self) -> list[GPUSample]:
        return []


class NVMLGPUMonitor:
    def __init__(self, interval_seconds: float = 1.0, device_index: int = 0) -> None:
        if interval_seconds <= 0:
            raise GPUMonitorUnavailable("GPU monitor interval must be positive")
        self.interval_seconds = float(interval_seconds)
        self.device_index = int(device_index)
        self.status = "created"
        self._samples: list[GPUSample] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        try:
            import pynvml  # type: ignore

            self._nvml = pynvml
            self._nvml.nvmlInit()
            self._handle = self._nvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception as exc:
            raise GPUMonitorUnavailable(str(exc)) from exc

    def start(self) -> None:
        if self._thread is not None:
            return
        self.status = "running"
        self._thread = threading.Thread(target=self._run, name="telemetry-gpu-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2.0))
        self.status = "stopped"
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass

    def samples(self) -> list[GPUSample]:
        return list(self._samples)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                self._samples.append(
                    GPUSample(
                        timestamp_seconds=time.time(),
                        utilization_percent=float(util.gpu),
                        memory_device_used_bytes=int(mem.used),
                    )
                )
            except Exception:
                self.status = "failed"
                return
            self._stop.wait(self.interval_seconds)


def build_gpu_monitor(enabled: bool, interval_seconds: float = 1.0) -> NullGPUMonitor | NVMLGPUMonitor:
    if not enabled:
        return NullGPUMonitor()
    try:
        return NVMLGPUMonitor(interval_seconds=interval_seconds)
    except GPUMonitorUnavailable:
        return NullGPUMonitor()


__all__ = ["GPUSample", "NVMLGPUMonitor", "NullGPUMonitor", "build_gpu_monitor"]
