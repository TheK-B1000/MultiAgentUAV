"""Bootstrap-row selection helpers for rollout collection."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def global_state_rows_from_step_infos(
    infos: List[Dict[str, Any]],
    next_global_state: np.ndarray,
) -> np.ndarray:
    rows = []
    for env_i, info in enumerate(infos):
        if bool(info.get("truncated", False)):
            terminal_obs = info.get("terminal_observation") or {}
            rows.append(
                np.asarray(
                    terminal_obs.get("global_state", next_global_state[env_i]),
                    dtype=np.float32,
                )
            )
        else:
            rows.append(np.asarray(next_global_state[env_i], dtype=np.float32))
    return np.stack(rows, axis=0)


__all__ = ["global_state_rows_from_step_infos"]
