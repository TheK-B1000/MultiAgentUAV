"""
VecEnv wrapper: sample discrete team strategy z (Option A: once per episode; Option B: every N steps).

Adds keys required by latent MARL: ``z_idx``, ``z_onehot``, ``z_logits``, ``global_state``,
and sparse-resampling metadata for persistence regularization.
Call ``attach_strategy_encoder(encoder)`` after the policy is built so sampling uses q_phi(z|s).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from rl.global_state import GLOBAL_STATE_DIM, build_global_state_batch


class LatentStrategyVecEnvWrapper(VecEnvWrapper):
    def __init__(
        self,
        venv: VecEnv,
        *,
        latent_k: int = 4,
        resample_every_n: int = 0,
    ):
        self.latent_k = int(latent_k)
        self.resample_every_n = int(resample_every_n)
        if self.resample_every_n == 1:
            raise ValueError("Latent strategy resampling every timestep is not paper-aligned; use 0 or >= 2.")
        self._encoder: Optional[torch.nn.Module] = None
        self._z_deterministic = False

        self._z_idx = np.zeros((venv.num_envs,), dtype=np.int64)
        self._z_prev_idx = np.zeros((venv.num_envs,), dtype=np.int64)
        self._z_logits = np.zeros((venv.num_envs, self.latent_k), dtype=np.float32)
        self._z_resampled = np.zeros((venv.num_envs,), dtype=np.float32)
        self._z_switch = np.zeros((venv.num_envs,), dtype=np.float32)
        self._steps_since_reset = np.zeros((venv.num_envs,), dtype=np.int64)
        self._episode_switch_count = np.zeros((venv.num_envs,), dtype=np.int64)
        self._episode_resample_count = np.zeros((venv.num_envs,), dtype=np.int64)

        assert isinstance(venv.observation_space, spaces.Dict)
        d = dict(venv.observation_space.spaces)
        d["z_idx"] = spaces.Box(low=0.0, high=float(self.latent_k - 1), shape=(1,), dtype=np.float32)
        d["z_prev_idx"] = spaces.Box(low=0.0, high=float(self.latent_k - 1), shape=(1,), dtype=np.float32)
        d["z_onehot"] = spaces.Box(low=0.0, high=1.0, shape=(self.latent_k,), dtype=np.float32)
        d["z_logits"] = spaces.Box(low=-50.0, high=50.0, shape=(self.latent_k,), dtype=np.float32)
        d["z_resampled"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        d["z_switch"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        d["global_state"] = spaces.Box(low=-np.inf, high=np.inf, shape=(GLOBAL_STATE_DIM,), dtype=np.float32)
        super().__init__(venv, observation_space=spaces.Dict(d))

    @property
    def core(self):
        return self.venv.core

    def attach_strategy_encoder(self, encoder: torch.nn.Module) -> None:
        """Use the policy's ``StrategyEncoder`` (same weights as training)."""
        self._encoder = encoder

    def set_z_deterministic(self, flag: bool) -> None:
        """If True, use argmax z (evaluation); if False, sample (training)."""
        self._z_deterministic = bool(flag)

    def _core(self):
        return getattr(self.venv, "core", None)

    def _sample_z_indices(self, idx: np.ndarray, *, mark_resample: bool) -> None:
        core = self._core()
        if core is None:
            raise RuntimeError("LatentStrategyVecEnvWrapper: inner env has no .core (expected GPUCTFVecEnv).")
        if len(idx) == 0:
            return
        prev_z = self._z_idx[idx].copy()
        gs = build_global_state_batch(core)
        with torch.no_grad():
            if self._encoder is None:
                logits = torch.zeros((len(idx), self.latent_k), device=gs.device, dtype=torch.float32)
                z = torch.randint(0, self.latent_k, (len(idx),), device=gs.device)
            else:
                logits = self._encoder(gs[idx])
                if self._z_deterministic:
                    z = logits.argmax(dim=-1)
                else:
                    z = torch.distributions.Categorical(logits=logits).sample()
            logits_np = logits.detach().cpu().numpy().astype(np.float32)
            z_np = z.detach().cpu().numpy().astype(np.int64)
        self._z_idx[idx] = z_np
        self._z_prev_idx[idx] = prev_z if mark_resample else z_np
        self._z_logits[idx] = logits_np
        if mark_resample:
            switched = (z_np != prev_z).astype(np.float32)
            self._z_resampled[idx] = 1.0
            self._z_switch[idx] = switched
            self._episode_resample_count[idx] += 1
            self._episode_switch_count[idx] += switched.astype(np.int64)
        else:
            self._z_resampled[idx] = 0.0
            self._z_switch[idx] = 0.0

    def _onehot(self) -> np.ndarray:
        B = self.num_envs
        oh = np.zeros((B, self.latent_k), dtype=np.float32)
        oh[np.arange(B), self._z_idx] = 1.0
        return oh

    def _global_state_np(self) -> np.ndarray:
        core = self._core()
        return build_global_state_batch(core).detach().cpu().numpy().astype(np.float32)

    def _augment_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = dict(obs)
        B = obs["grid"].shape[0]
        out["z_idx"] = self._z_idx.astype(np.float32).reshape(B, 1)
        out["z_prev_idx"] = self._z_prev_idx.astype(np.float32).reshape(B, 1)
        out["z_onehot"] = self._onehot()
        out["z_logits"] = self._z_logits.astype(np.float32)
        out["z_resampled"] = self._z_resampled.astype(np.float32).reshape(B, 1)
        out["z_switch"] = self._z_switch.astype(np.float32).reshape(B, 1)
        out["global_state"] = self._global_state_np()
        return out

    def _augment_terminal_row(
        self,
        obs1: dict[str, np.ndarray],
        z_idx: int,
        z_prev_idx: int,
        z_logits: np.ndarray,
        *,
        gs_row: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        out = {k: v.copy() for k, v in obs1.items()}
        out["z_idx"] = np.array([[float(z_idx)]], dtype=np.float32)
        out["z_prev_idx"] = np.array([[float(z_prev_idx)]], dtype=np.float32)
        oh = np.zeros((1, self.latent_k), dtype=np.float32)
        oh[0, int(z_idx)] = 1.0
        out["z_onehot"] = oh
        out["z_logits"] = z_logits.reshape(1, -1).astype(np.float32)
        out["z_resampled"] = np.zeros((1, 1), dtype=np.float32)
        out["z_switch"] = np.zeros((1, 1), dtype=np.float32)
        if gs_row is not None:
            out["global_state"] = gs_row.astype(np.float32).reshape(1, -1)
        else:
            out["global_state"] = np.zeros((1, GLOBAL_STATE_DIM), dtype=np.float32)
        return out

    def reset(self) -> dict[str, np.ndarray]:
        obs = self.venv.reset()
        self._steps_since_reset[:] = 0
        self._episode_switch_count[:] = 0
        self._episode_resample_count[:] = 0
        self._sample_z_indices(np.arange(self.num_envs, dtype=np.int64), mark_resample=False)
        return self._augment_obs(obs)

    def step_wait(self) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]:
        obs, rewards, dones, infos = self.venv.step_wait()
        old_z = self._z_idx.copy()
        old_prev_z = self._z_prev_idx.copy()
        old_logits = self._z_logits.copy()

        d = dones.astype(bool)
        for i in np.where(d)[0]:
            if isinstance(infos[i], dict):
                infos[i]["strategy_switch_count"] = int(self._episode_switch_count[i])
                infos[i]["strategy_resample_count"] = int(self._episode_resample_count[i])
                to = infos[i].get("terminal_observation")
                if isinstance(to, dict):
                    gs_t = to.get("global_state")
                    row = {}
                    for k, v in to.items():
                        if k == "global_state":
                            continue
                        row[k] = np.expand_dims(np.asarray(v, dtype=np.float32), axis=0)
                    if isinstance(gs_t, np.ndarray) and gs_t.ndim == 1:
                        gs_t = gs_t.reshape(1, -1)
                    else:
                        gs_t = None
                    infos[i]["terminal_observation"] = self._augment_terminal_row(
                        row,
                        int(old_z[i]),
                        int(old_prev_z[i]),
                        old_logits[i],
                        gs_row=gs_t,
                    )

        self._z_prev_idx[:] = self._z_idx
        self._z_resampled[:] = 0.0
        self._z_switch[:] = 0.0
        self._steps_since_reset += 1
        if d.any():
            self._steps_since_reset[d] = 0
            self._episode_switch_count[d] = 0
            self._episode_resample_count[d] = 0
            self._sample_z_indices(np.where(d)[0].astype(np.int64), mark_resample=False)
        if self.resample_every_n > 0:
            mask = (~d) & ((self._steps_since_reset % self.resample_every_n) == 0)
            if np.any(mask):
                self._sample_z_indices(np.where(mask)[0].astype(np.int64), mark_resample=True)

        obs = self._augment_obs(obs)
        return obs, rewards, dones, infos
