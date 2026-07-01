"""Observation-space helpers for V6I3 local communication."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium import spaces

from rl.custom_ppo.communication.config import extra_cnn_channels, resolve_comm_config


def extend_observation_space_if_needed(observation_space: Any, cfg: Any) -> Any:
    """Return an observation space whose grid channel count includes message CNN channels."""
    extra = int(extra_cnn_channels(cfg))
    if extra <= 0:
        return observation_space
    grid_space = observation_space.spaces["grid"]
    n, c, h, w = tuple(int(v) for v in grid_space.shape)
    extended = spaces.Dict(
        {
            **dict(observation_space.spaces),
            "grid": spaces.Box(
                low=float(grid_space.low.min()),
                high=float(grid_space.high.max()),
                shape=(n, c + extra, h, w),
                dtype=grid_space.dtype,
            ),
        }
    )
    return extended


def base_env_grid_channels(observation_space: Any, cfg: Any) -> int:
    """CNN channels from the environment before message injection."""
    grid_space = observation_space.spaces["grid"]
    c = int(grid_space.shape[1])
    extra = int(extra_cnn_channels(cfg))
    if extra <= 0:
        return c
    return max(1, c - extra)


def inject_message_grid_channels(
    obs: dict[str, np.ndarray],
    *,
    message_channels: torch.Tensor | None,
    cfg: Any,
    expected_grid_channels: int | None = None,
) -> dict[str, np.ndarray]:
    """Concatenate transport message channels onto the env grid observation."""
    comm = resolve_comm_config(cfg)
    if not comm.enabled:
        return obs
    grid = np.asarray(obs["grid"], dtype=np.float32)
    extra = int(comm.message_grid_channels)
    current_channels = int(grid.shape[2])
    if expected_grid_channels is None:
        expected_channels = current_channels + extra
    else:
        expected_channels = int(expected_grid_channels)
    base_channels = expected_channels - extra
    if base_channels <= 0:
        raise ValueError(f"expected_grid_channels={expected_channels} is too small for extra={extra}")
    if current_channels == base_channels:
        base_grid = grid
    elif current_channels == expected_channels:
        base_grid = grid[:, :, :base_channels, :, :]
    else:
        raise ValueError(
            f"grid channel count {current_channels} is incompatible with "
            f"expected={expected_channels} base={base_channels} extra={extra}"
        )
    if message_channels is None:
        bsz, n_agents = int(base_grid.shape[0]), int(base_grid.shape[1])
        h, w = int(base_grid.shape[-2]), int(base_grid.shape[-1])
        zeros = np.zeros((bsz, n_agents, extra, h, w), dtype=np.float32)
        return {**obs, "grid": np.concatenate([base_grid, zeros], axis=2)}
    msg = message_channels.detach().cpu().numpy().astype(np.float32)
    if msg.shape[:2] != base_grid.shape[:2]:
        raise ValueError(
            f"message_channels batch/agents {msg.shape[:2]} != grid {base_grid.shape[:2]}"
        )
    return {**obs, "grid": np.concatenate([base_grid, msg], axis=2)}


__all__ = [
    "base_env_grid_channels",
    "extend_observation_space_if_needed",
    "inject_message_grid_channels",
]
