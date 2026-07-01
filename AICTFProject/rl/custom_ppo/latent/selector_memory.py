"""Recurrent selector hidden state — sole owner of GRU memory tensors."""

from __future__ import annotations

import torch


class SelectorMemory:
    def __init__(self, *, n_envs: int, hidden_dim: int, device: torch.device) -> None:
        self.hidden_dim = int(hidden_dim)
        self._current: torch.Tensor | None = (
            torch.zeros((n_envs, hidden_dim), dtype=torch.float32, device=device)
            if hidden_dim > 0
            else None
        )
        self._episode_snapshot: torch.Tensor | None = None
        self._macro_snapshot: torch.Tensor | None = None
        self._arc_snapshot: torch.Tensor | None = None

    @property
    def enabled(self) -> bool:
        return self._current is not None

    def current(self) -> torch.Tensor | None:
        return self._current

    def reset_rows(self, mask: torch.Tensor) -> None:
        if self._current is not None and bool(mask.any().item()):
            self._current[mask] = 0.0

    def snapshot_rows(self, mask: torch.Tensor) -> torch.Tensor | None:
        if self._current is None or not bool(mask.any().item()):
            return None
        return self._current[mask].detach().clone()

    def update(self, next_hidden: torch.Tensor) -> None:
        self._current = next_hidden.detach()

    def zero_all(self) -> None:
        if self._current is not None:
            self._current.zero_()
