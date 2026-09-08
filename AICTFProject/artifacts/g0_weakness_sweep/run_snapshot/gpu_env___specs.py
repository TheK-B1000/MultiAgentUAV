from __future__ import annotations

from typing import List

import numpy as np
from gymnasium import spaces

from ._constants import CNN_COLS, CNN_ROWS, NUM_CNN_CHANNELS, VEC_OBS_DIM


class VecEnv:
    """Minimal vector-env base used by the local PPO trainer."""

    def __init__(self, num_envs: int, observation_space: spaces.Space, action_space: spaces.Space) -> None:
        self.num_envs = int(num_envs)
        self.observation_space = observation_space
        self.action_space = action_space

    def _get_indices(self, indices=None) -> List[int]:
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, (int, np.integer)):
            return [int(indices)]
        return [int(i) for i in indices]

    def step(self, actions: np.ndarray):
        self.step_async(actions)
        return self.step_wait()


def _make_obs_action_spaces(n_agents: int, n_macros: int, n_targets: int, *, num_cnn_channels: int = NUM_CNN_CHANNELS):
    obs_space = spaces.Dict(
        {
            "grid": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(n_agents, int(num_cnn_channels), CNN_ROWS, CNN_COLS),
                dtype=np.float32,
            ),
            "vec": spaces.Box(low=-1.0, high=1.0, shape=(n_agents, VEC_OBS_DIM), dtype=np.float32),
            "agent_mask": spaces.Box(low=0.0, high=1.0, shape=(n_agents,), dtype=np.float32),
            "mask": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(n_agents * (n_macros + n_targets),),
                dtype=np.float32,
            ),
        }
    )
    action_space = spaces.MultiDiscrete([n_macros, n_targets] * n_agents)
    return obs_space, action_space


__all__ = ["VecEnv", "_make_obs_action_spaces"]
