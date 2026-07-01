"""Snapshot policy loading with mtime-keyed cache.

``_load_snapshot_policy`` resolves a snapshot key to an absolute path via
``_resolve_snapshot_path``, checks the file mtime for cache staleness, and
lazily imports ``rl.custom_ppo.load_custom_ppo_policy`` so that the state
module can be imported without an RL dependency in non-training contexts.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from .._paths import _resolve_snapshot_path
from .._specs import _make_obs_action_spaces


class _SnapshotsMixin:
    """Owns snapshot policy loading with mtime-keyed in-process cache."""

    def _load_snapshot_policy(self, snapshot_key: str) -> Optional[Any]:
        resolved = _resolve_snapshot_path(snapshot_key)
        if resolved is None:
            return None
        try:
            mtime = float(os.path.getmtime(resolved))
        except OSError:
            return None
        cached = self._snapshot_policy_cache.get(resolved)
        if cached is not None:
            cached_mtime, cached_model = cached
            if abs(cached_mtime - mtime) < 1e-9:
                return cached_model
        try:
            from rl.custom_ppo import load_custom_ppo_policy

            obs_space, action_space = _make_obs_action_spaces(
                self.Nr,
                self.cfg.n_macros,
                self.cfg.n_targets,
                num_cnn_channels=int(self.cfg.num_cnn_channels),
            )
            model = load_custom_ppo_policy(
                resolved, obs_space, action_space, device=self.device
            )
        except Exception:
            model = None
        self._snapshot_policy_cache[resolved] = (mtime, model)
        return model
