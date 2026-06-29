"""Gymnasium and VecEnv wrappers for BatchedCTFCore."""
from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from rl.global_state import build_global_state_batch

from ._config import GPUFieldConfig
from ._constants import CNN_COLS, CNN_ROWS, VEC_OBS_DIM
from ._core_class import BatchedCTFCore
from ._episode_payload import _build_episode_result_payload
from ._specs import VecEnv


class GPUCTFVecEnv(VecEnv):
    """Local vector-env wrapper around BatchedCTFCore."""

    def __init__(self, cfg: GPUFieldConfig):
        self.core = BatchedCTFCore(cfg)
        self.cfg = cfg
        self._n_macros = int(cfg.n_macros)
        self._n_targets = int(cfg.n_targets)
        self._n_blue = int(cfg.max_blue_agents)
        self._grid_channels = int(cfg.num_cnn_channels)
        obs_space = spaces.Dict(
            {
                "grid": spaces.Box(low=0.0, high=1.0, shape=(self._n_blue, self._grid_channels, CNN_ROWS, CNN_COLS), dtype=np.float32),
                "vec": spaces.Box(low=-1.0, high=1.0, shape=(self._n_blue, VEC_OBS_DIM), dtype=np.float32),
                "agent_mask": spaces.Box(low=0.0, high=1.0, shape=(self._n_blue,), dtype=np.float32),
                "mask": spaces.Box(low=0.0, high=1.0, shape=(self._n_blue * (self._n_macros + self._n_targets),), dtype=np.float32),
            }
        )
        action_space = spaces.MultiDiscrete([self._n_macros, self._n_targets] * self._n_blue)
        super().__init__(cfg.n_envs, obs_space, action_space)
        self._pending_actions: Optional[np.ndarray] = None
        # Optional (trainer-owned): called after terminal payloads are built but before ``reset_indices``,
        # so per-env scripted opponent changes apply to the upcoming episode (correct OP4 guard layout, etc.).
        self._before_reset_indices_hook: Optional[Any] = None

    def reset(self) -> Dict[str, np.ndarray]:
        self.core.reset_all()
        return self.core.get_obs()

    def get_obs(self) -> Dict[str, np.ndarray]:
        """Current blue-team observation dict (same layout as ``reset()``)."""
        return self.core.get_obs()

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        if seed is not None:
            self.core.reseed(int(seed))
        return [seed for _ in range(self.num_envs)]

    def state(self) -> np.ndarray:
        return self.core.get_global_state()

    def render(self, mode: str = "human"):
        if mode == "rgb_array":
            return self.core.render_rgb_array(env_index=0)
        if mode == "human":
            return self.core.render_rgb_array(env_index=0)
        raise ValueError(f"Unsupported render mode {mode!r}")

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = np.asarray(actions, dtype=np.int64)

    def step_wait(self):
        assert self._pending_actions is not None, "step_async() must be called before step_wait()"
        flat = np.asarray(self._pending_actions, dtype=np.int64).reshape(-1)
        n_exp = int(self.core.B * self.core.Nb * 2)
        if int(flat.size) != n_exp:
            raise ValueError(
                f"GPUCTFVecEnv: expected {n_exp} action values (B={self.core.B}, Nb={self.core.Nb}), "
                f"got {int(flat.size)} from predict() shape={getattr(self._pending_actions, 'shape', None)}"
            )
        actions = torch.as_tensor(flat, dtype=torch.int64, device=self.core.device)
        obs, rew, term, trunc, infos = self.core.step(actions)
        done = np.logical_or(term, trunc)
        if done.any():
            reset_mask = torch.from_numpy(done).to(self.core.device)
            # MARL / CTDE: fixed-size global features at terminal timestep (before reset) for value bootstrap.
            gs_terminal = build_global_state_batch(self.core).detach().cpu().numpy().astype(np.float32)
            for i in np.where(done)[0]:
                infos[i] = dict(infos[i])
                tobs = {k: v[i].copy() for k, v in obs.items()}
                tobs["global_state"] = gs_terminal[i].copy()
                infos[i]["terminal_observation"] = tobs
                # So training callbacks (parse_episode_result) get a single episode_result dict.
                infos[i]["episode_result"] = _build_episode_result_payload(infos[i])
            hook = getattr(self, "_before_reset_indices_hook", None)
            if callable(hook):
                hook(done, infos)
            self.core.reset_indices(reset_mask)
            obs = self.core.get_obs()
        self._pending_actions = None
        return obs, rew, done, infos

    def close(self) -> None:
        self._pending_actions = None

    def get_attr(self, attr_name: str, indices=None):
        idx = self._get_indices(indices)
        if attr_name == "render_mode":
            return [None for _ in idx]
        val = getattr(self.core, attr_name)
        return [val for _ in idx]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        idx = self._get_indices(indices)
        if len(idx) != self.num_envs:
            return
        setattr(self.core, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        idx = self._get_indices(indices)
        method = getattr(self.core, method_name)
        method_kwargs = dict(method_kwargs)
        try:
            sig = inspect.signature(method)
        except Exception:
            sig = None
        if sig is not None and "env_indices" in sig.parameters:
            method_kwargs.setdefault("env_indices", idx)
        out = method(*method_args, **method_kwargs)
        return [out for _ in idx]

    def env_is_wrapped(self, wrapper_class, indices=None):
        idx = self._get_indices(indices)
        return [False for _ in idx]

class GPUCTFSingleEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(self, cfg: Optional[GPUFieldConfig] = None):
        cfg = cfg or GPUFieldConfig(n_envs=1)
        cfg.n_envs = 1
        self.vec = GPUCTFVecEnv(cfg)
        self.action_space = self.vec.action_space
        self.observation_space = self.vec.observation_space

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.vec.core.reseed(int(seed))
        obs = self.vec.reset()
        return {k: v[0] for k, v in obs.items()}, {}

    def step(self, action):
        self.vec.step_async(np.asarray(action, dtype=np.int64)[None, ...])
        obs, rew, done, infos = self.vec.step_wait()
        terminated = bool(infos[0].get("terminated", bool(done[0])))
        truncated = bool(infos[0].get("truncated", False))
        return {k: v[0] for k, v in obs.items()}, float(rew[0]), terminated, truncated, infos[0]

    def state(self) -> np.ndarray:
        return self.vec.state()[0]

    def render(self, mode: str = "human"):
        return self.vec.render(mode=mode)

    def close(self):
        self.vec.close()

__all__ = ["GPUCTFVecEnv", "GPUCTFSingleEnv"]
