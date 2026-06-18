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
) -> dict[str, np.ndarray]:
    """Concatenate transport message channels onto the env grid observation."""
    comm = resolve_comm_config(cfg)
    if not comm.enabled:
        return obs
    grid = np.asarray(obs["grid"], dtype=np.float32)
    if message_channels is None:
        bsz, n_agents = int(grid.shape[0]), int(grid.shape[1])
        extra = int(comm.message_grid_channels)
        h, w = int(grid.shape[-2]), int(grid.shape[-1])
        zeros = np.zeros((bsz, n_agents, extra, h, w), dtype=np.float32)
        return {**obs, "grid": np.concatenate([grid, zeros], axis=2)}
    msg = message_channels.detach().cpu().numpy().astype(np.float32)
    if msg.shape[:2] != grid.shape[:2]:
        raise ValueError(
            f"message_channels batch/agents {msg.shape[:2]} != grid {grid.shape[:2]}"
        )
    return {**obs, "grid": np.concatenate([grid, msg], axis=2)}


__all__ = [
    "base_env_grid_channels",
    "extend_observation_space_if_needed",
    "inject_message_grid_channels",
]
