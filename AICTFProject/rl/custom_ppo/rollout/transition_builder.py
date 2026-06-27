"""Transition-row helpers for rollout collection."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch


def flag_territory_features_changed(
    pre: torch.Tensor,
    post: torch.Tensor,
    *,
    eps: float = 1e-4,
) -> torch.Tensor:
    d0 = (pre[:, 0:2] - post[:, 0:2]).abs() > float(eps)
    ch_float = d0.any(dim=-1)
    ch_cap = (pre[:, 2:4] - post[:, 2:4]).abs() > 0.5
    ch_capt = ch_cap.any(dim=-1)
    return ch_float | ch_capt


def obs_rows_from_next_step(
    next_obs: Dict[str, np.ndarray],
    infos: List[Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    rows: Dict[str, list[np.ndarray]] = {
        key: [] for key in ("grid", "vec", "agent_mask", "mask")
    }
    for env_i, info in enumerate(infos):
        use_terminal = bool(info.get("truncated", False)) and isinstance(
            info.get("terminal_observation"), dict
        )
        terminal_obs = info.get("terminal_observation") if use_terminal else {}
        for key in rows:
            source = (
                terminal_obs.get(key, next_obs[key][env_i])
                if isinstance(terminal_obs, dict)
                else next_obs[key][env_i]
            )
            rows[key].append(np.asarray(source, dtype=np.float32))
    return {key: np.stack(values, axis=0) for key, values in rows.items()}


__all__ = ["flag_territory_features_changed", "obs_rows_from_next_step"]
