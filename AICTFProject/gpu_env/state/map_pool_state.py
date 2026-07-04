"""Per-episode map pool sampling for vectorized training."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from .._maps import MAP_A_OPEN, normalize_map_layout


class _MapPoolStateMixin:
    """Per-env map layout when ``cfg.map_pool`` is non-empty."""

    _map_pool: Tuple[str, ...]
    _map_layout_per_env: List[str]
    _map_id_per_env: List[int]

    def _init_map_pool_state(self) -> None:
        pool_raw = tuple(getattr(self.cfg, "map_pool", ()) or ())
        if pool_raw:
            self._map_pool = tuple(normalize_map_layout(m) for m in pool_raw)
        else:
            self._map_pool = ()
        self._map_layout_per_env = [str(self.map_layout).lower()] * int(self.B)
        self._map_id_per_env = [-1] * int(self.B)
        if self._map_pool:
            for env_i in range(int(self.B)):
                self._assign_map_layout_for_env(env_i, self._draw_map_pool_index())

    def _draw_map_pool_index(self) -> int:
        if not self._map_pool:
            return -1
        pick = int(torch.randint(len(self._map_pool), (1,), generator=self._rng, device=self.device).item())
        return pick

    def _assign_map_layout_for_env(self, env_i: int, pool_idx: int) -> None:
        if not self._map_pool:
            self._map_layout_per_env[env_i] = str(self.map_layout).lower()
            self._map_id_per_env[env_i] = -1
            return
        idx = int(pool_idx) % len(self._map_pool)
        self._map_layout_per_env[env_i] = self._map_pool[idx]
        self._map_id_per_env[env_i] = idx

    def _map_layout_for_env(self, env_i: int) -> str:
        return str(self._map_layout_per_env[int(env_i)])

    def _map_id_for_env(self, env_i: int) -> int:
        return int(self._map_id_per_env[int(env_i)])

    def _resample_map_pool(self, env_mask: torch.Tensor) -> None:
        if not self._map_pool:
            return
        idx = torch.where(env_mask)[0]
        for env_i in idx.detach().cpu().tolist():
            self._assign_map_layout_for_env(int(env_i), self._draw_map_pool_index())

    def _map_layout_mask(self, layouts: Sequence[str]) -> torch.Tensor:
        wanted = {normalize_map_layout(x) for x in layouts}
        values = [self._map_layout_for_env(i) in wanted for i in range(self.B)]
        return torch.as_tensor(values, dtype=torch.bool, device=self.device)


__all__ = ["_MapPoolStateMixin"]
