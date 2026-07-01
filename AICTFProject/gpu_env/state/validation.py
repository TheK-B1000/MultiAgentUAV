"""Index normalisation, phase tensor caching, and snapshot control-mask utilities.

These helpers are shared across the public API (``set_phase``, ``set_next_opponent``,
etc.) and internal logic.  They are stateless pure-mixin methods that depend only
on ``self.B``, ``self.device``, ``self._phase``, and the phase/opponent cache fields
initialised by ``_CoreStateMixin.__init__``.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from .._paths import _resolve_snapshot_path


class _ValidationMixin:
    """Provides env-index normalisation, phase tensor caching, and red-control-mask."""

    def _normalize_env_indices(
        self, env_indices: Optional[Sequence[int]] = None
    ) -> torch.Tensor:
        if env_indices is None:
            return torch.arange(self.B, device=self.device, dtype=torch.int64)
        if isinstance(env_indices, torch.Tensor):
            idx = env_indices.to(device=self.device, dtype=torch.int64).reshape(-1)
        else:
            idx = torch.as_tensor(
                list(env_indices), device=self.device, dtype=torch.int64
            ).reshape(-1)
        if idx.numel() == 0:
            return idx
        return torch.clamp(idx, 0, max(0, self.B - 1))

    def _phase_tensor_equals(self, phases: Sequence[str]) -> torch.Tensor:
        key: Tuple[str, ...] = tuple(sorted(str(p).upper() for p in phases))
        cached = self._phase_tensor_cache.get(key)
        if cached is None:
            phase_set = set(key)
            cached = torch.as_tensor(
                [str(p).upper() in phase_set for p in self._phase],
                device=self.device,
                dtype=torch.bool,
            )
            self._phase_tensor_cache[key] = cached
        return cached

    def _get_red_control_mask(self) -> torch.Tensor:
        if self._red_control_mask_dirty or self._red_control_mask is None:
            self._red_control_mask = torch.as_tensor(
                [
                    str(self._opponent_kind[i]).upper() == "SNAPSHOT"
                    and _resolve_snapshot_path(self._opponent_key[i]) is not None
                    for i in range(self.B)
                ],
                device=self.device,
                dtype=torch.bool,
            )
            self._red_control_mask_dirty = False
        return self._red_control_mask
